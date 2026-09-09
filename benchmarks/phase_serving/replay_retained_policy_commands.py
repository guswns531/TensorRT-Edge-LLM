#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Replay trusted local HTTP command manifests, varying only policy and output path."""

import argparse
import hashlib
import json
import os
import pathlib
import subprocess
import urllib.parse


def materialize_inputs(commands, output, remaps):
    """Validate local media and preserve explicit path repairs with asset hashes."""
    documents, assets = {}, {}

    def visit(value):
        if isinstance(value, dict):
            return {key: visit(item) for key, item in value.items()}
        if isinstance(value, list):
            return [visit(item) for item in value]
        if not isinstance(value, str) or not value.startswith("file://"):
            return value
        path = urllib.parse.unquote(urllib.parse.urlsplit(value).path)
        for old, new in remaps:
            if path == old or path.startswith(old + "/"):
                path = new + path[len(old):]
                break
        asset = pathlib.Path(path)
        if not asset.is_file():
            raise FileNotFoundError(
                f"Trace/calibration media is missing: {asset}")
        if path not in assets:
            assets[path] = hashlib.sha256(asset.read_bytes()).hexdigest()
        return asset.as_uri()

    for record in commands:
        command = record["command"]
        for option in ("--trace", "--generic-warmup-trace"):
            if option not in command:
                continue
            source = pathlib.Path(command[command.index(option) + 1])
            if source not in documents:
                raw = source.read_bytes()
                identity = hashlib.sha256(str(source).encode() +
                                          raw).hexdigest()[:16]
                documents[source] = (output / "inputs" /
                                     f"{source.stem}-{identity}.json",
                                     visit(json.loads(raw)))
    return documents, assets


def enable_eos_termination(command):
    """Remove both client and backend ignore-EOS overrides in a copied command."""
    result = list(command)
    if "--ignore-eos" in result:
        result.remove("--ignore-eos")
    matches = [
        i for i, value in enumerate(result)
        if value.startswith("TRT_EDGELLM_IGNORE_EOS=")
    ]
    for index in reversed(matches):
        if index == 0 or result[index - 1] != "-e":
            raise ValueError(
                "Expected a Docker -e before the ignore-EOS assignment")
        del result[index - 1:index + 1]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commands", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--cases", nargs="+", required=True)
    parser.add_argument("--legacy-pair-eligibility", action="store_true")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--build-cache", type=pathlib.Path, required=True)
    parser.add_argument("--asset-remap",
                        action="append",
                        default=[],
                        metavar="OLD=NEW")
    parser.add_argument("--policies",
                        nargs="+",
                        choices=("exact", "scalar", "scalar-transition"),
                        default=["exact", "scalar", "scalar-transition"])
    parser.add_argument("--vision-engine-dir")
    parser.add_argument(
        "--respect-eos",
        action="store_true",
        help=
        "Separate correctness contract; not comparable to fixed-output throughput"
    )
    parser.add_argument(
        "--phase-telemetry",
        "--dispatch-telemetry",
        dest="dispatch_telemetry",
        action="store_true",
        help=
        "Diagnostic-only per-dispatch output; changes instrumentation overhead"
    )
    parser.add_argument("--telemetry-level",
                        choices=("dispatch", "audit"),
                        default="dispatch")
    args = parser.parse_args()
    if "CMAKE_BUILD_TYPE:STRING=Release" not in args.build_cache.read_text(
    ).splitlines():
        raise ValueError("Performance comparison requires a Release build")
    flags = args.build_cache.parent / "cpp/CMakeFiles/edgellmCore.dir/flags.make"
    for language in ("CXX", "CUDA"):
        effective = next(line for line in flags.read_text().splitlines()
                         if line.startswith(f"{language}_FLAGS ="))
        if not {"-O3", "-DNDEBUG"}.issubset(effective.split()):
            raise ValueError(
                f"Effective core {language} flags are not Release-optimized")
    records = json.loads(args.commands.read_text())
    selected = {record["case"]: record for record in records}
    if any(case not in selected for case in args.cases):
        raise ValueError("Requested case missing from retained manifest")
    remaps = [tuple(value.split("=", 1)) for value in args.asset_remap]
    if any(len(pair) != 2 or not all(pair) for pair in remaps):
        raise ValueError("Asset remaps must be nonempty OLD=NEW pairs")
    documents, assets = materialize_inputs(
        [selected[case] for case in args.cases], args.output, remaps)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "inputs").mkdir()
    for destination, document in documents.values():
        destination.write_text(json.dumps(document, indent=2) + "\n")
    (args.output / "input-contract.json").write_text(
        json.dumps(
            {
                "asset_remaps": remaps,
                "asset_sha256": assets,
                "inputs": {
                    str(source): str(destination)
                    for source, (destination, _) in documents.items()
                }
            },
            indent=2) + "\n")
    planned = []
    for policy in args.policies:
        for case in args.cases:
            command = list(selected[case]["command"])
            if args.respect_eos:
                command = enable_eos_termination(command)
            for option in ("--trace", "--generic-warmup-trace"):
                if option in command:
                    index = command.index(option) + 1
                    command[index] = str(documents[pathlib.Path(
                        command[index])][0])
            client_index = command.index("--client-script") + 1
            client_implementation = command[client_index]
            command[client_index] = str(
                pathlib.Path(__file__).resolve().with_name(
                    "guarded_trace_client.py"))
            destination = args.output / policy / "generic" / case / "worker-4"
            if args.dispatch_telemetry:
                mounts = [
                    value for value in command if value.endswith(":/workspace")
                ]
                if len(mounts) != 1:
                    raise ValueError(
                        "Diagnostic output requires one writable /workspace mount"
                    )
                root = pathlib.Path(mounts[0].rsplit(":", 1)[0])
                relative = destination.resolve().relative_to(root.resolve())
                telemetry = pathlib.PurePosixPath(
                    "/workspace") / relative / "run-{run}" / "dispatch.jsonl"
                image = command.index("nvcr.io/nvidia/tensorrt:26.06-py3")
                command[image:image] = [
                    "-e", "TRT_EDGELLM_EMIT_PHASE_METRICS=1", "-e",
                    "TRT_EDGELLM_PHASE_TELEMETRY_LEVEL=" +
                    args.telemetry_level, "-e",
                    "TRT_EDGELLM_PHASE_TELEMETRY_PATH=" + str(telemetry)
                ]
            command[command.index("--output-dir") + 1] = str(destination)
            command[command.index("--repeats") + 1] = str(args.repeats)
            if args.legacy_pair_eligibility:
                image = command.index("nvcr.io/nvidia/tensorrt:26.06-py3")
                command[image:image] = [
                    "-e", "TRT_EDGELLM_LEGACY_PAIR_ELIGIBILITY=1"
                ]
            matches = [
                i for i, value in enumerate(command)
                if value.startswith("TRT_EDGELLM_PHASE_POLICY=")
            ]
            if len(matches) != 1:
                raise ValueError(
                    "Expected exactly one phase policy assignment")
            command[matches[0]] = "TRT_EDGELLM_PHASE_POLICY=" + policy
            if args.vision_engine_dir:
                vision = [
                    i for i, value in enumerate(command)
                    if value.startswith("TRT_EDGELLM_VISION_ENGINE_DIR=")
                ]
                if len(vision) != 1:
                    raise ValueError(
                        "Expected one vision engine assignment for the controlled comparison"
                    )
                command[vision[
                    0]] = "TRT_EDGELLM_VISION_ENGINE_DIR=" + args.vision_engine_dir
            planned.append({
                "policy": policy,
                "case": case,
                "command": command,
                "environment": {
                    "PHASE_TRACE_CLIENT_IMPL": client_implementation
                }
            })
    (args.output /
     "commands.json").write_text(json.dumps(planned, indent=2) + "\n")
    for index, record in enumerate(planned):
        print(
            f"[{index + 1}/{len(planned)}] {record['policy']} {record['case']}",
            flush=True)
        with (args.output / f"run-{index + 1:02d}.log").open("w") as log:
            subprocess.run(record["command"],
                           stdout=log,
                           stderr=subprocess.STDOUT,
                           check=True,
                           env={
                               **os.environ,
                               **record["environment"]
                           })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
