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


def file_sha256(path):
    """Return the content identity of one retained campaign input."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def remove_explicit_slo_contract(command):
    """Remove composition-root SLOs while preserving all scheduling mechanisms."""
    result = list(command)
    prefixes = (
        "TRT_EDGELLM_VISION_TTFT_TARGET_MS=",
        "TRT_EDGELLM_VISION_DECODE_TPOT_TARGET_MS=",
        "TRT_EDGELLM_GLOBAL_DECODE_TPOT_TARGET_US=",
    )
    matches = [
        i for i, value in enumerate(result) if any(
            value.startswith(prefix) for prefix in prefixes)
    ]
    for index in reversed(matches):
        if index == 0 or result[index - 1] != "-e":
            raise ValueError("Expected a Docker -e before the SLO assignment")
        del result[index - 1:index + 1]
    return result


def remap_runtime_build(command, build_cache):
    """Use the validated Release build without changing the retained workload."""
    mounts = [value for value in command if value.endswith(":/workspace")]
    if len(mounts) != 1:
        raise ValueError("Expected one writable /workspace mount")
    host_root = pathlib.Path(mounts[0].rsplit(":", 1)[0]).resolve()
    build_root = build_cache.parent.resolve()
    try:
        relative = build_root.relative_to(host_root)
    except ValueError as error:
        raise ValueError(
            "Build cache must be below the /workspace mount") from error
    executable = [
        value for value in command
        if value.endswith("/examples/llm/llm_phase_context_smoke")
    ]
    if len(executable) != 1:
        raise ValueError("Expected one phase runtime executable")
    old_root = str(pathlib.PurePosixPath(executable[0]).parents[2])
    new_root = str(pathlib.PurePosixPath("/workspace") / relative)
    return [value.replace(old_root, new_root) for value in command]


def inject_runtime_environment(command, assignments):
    """Inject validated runtime-only ablation settings into a Docker command."""
    result = list(command)
    image = result.index("nvcr.io/nvidia/tensorrt:26.06-py3")
    for assignment in assignments:
        name, separator, value = assignment.partition("=")
        if not separator or not name.startswith("TRT_EDGELLM_") or not value:
            raise ValueError(
                "Runtime environment must use TRT_EDGELLM_NAME=VALUE")
        matches = [
            index for index, item in enumerate(result)
            if item.startswith(name + "=")
        ]
        if len(matches) > 1:
            raise ValueError(f"Duplicate runtime environment setting: {name}")
        if matches:
            result[matches[0]] = assignment
            continue
        result[image:image] = ["-e", assignment]
        image += 2
    return result


def remove_runtime_environment(command, names):
    """Remove explicit runtime settings from a copied retained command."""
    result = list(command)
    for name in names:
        if not name.startswith("TRT_EDGELLM_") or "=" in name:
            raise ValueError(
                "Dropped runtime environment must be a TRT_EDGELLM_NAME")
        matches = [
            index for index, item in enumerate(result)
            if item.startswith(name + "=")
        ]
        if len(matches) > 1:
            raise ValueError(f"Duplicate runtime environment setting: {name}")
        if not matches:
            continue
        index = matches[0]
        if index == 0 or result[index - 1] != "-e":
            raise ValueError(f"Expected a Docker -e before {name}")
        del result[index - 1:index + 1]
    return result


def enable_automatic_calibration(command, max_cycles):
    """Use complete generic-program cycles as convergence observations."""
    result = list(command)
    required = ("--warmup-requests", "--generic-warmup-trace",
                "--phase-calibration-round-requests",
                "--phase-calibration-min-requests")
    if any(option not in result for option in required):
        raise ValueError(
            "Automatic calibration requires a generic trace and calibration options"
        )
    if max_cycles < 1:
        raise ValueError("Automatic calibration needs a positive cycle cap")
    trace = pathlib.Path(result[result.index("--generic-warmup-trace") + 1])
    requests = json.loads(trace.read_bytes()).get("requests", [])
    if not requests:
        raise ValueError("Generic calibration trace contains no requests")
    cycle_requests = len(requests)
    result[result.index("--warmup-requests") + 1] = str(cycle_requests *
                                                         max_cycles)
    result[result.index("--phase-calibration-round-requests") + 1] = str(
        cycle_requests)
    result[result.index("--phase-calibration-min-requests") + 1] = "0"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commands", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--cases", nargs="+", required=True)
    parser.add_argument("--legacy-pair-eligibility", action="store_true")
    parser.add_argument("--preserve-expired-decode-candidate",
                        action="store_true")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--build-cache", type=pathlib.Path, required=True)
    parser.add_argument("--asset-remap",
                        action="append",
                        default=[],
                        metavar="OLD=NEW")
    parser.add_argument("--policies",
                        nargs="+",
                        choices=("exact", "scalar", "scalar-transition",
                                 "service-scaled-transition"),
                        default=["exact", "scalar", "scalar-transition"])
    parser.add_argument("--vision-engine-dir")
    parser.add_argument("--text-engine-dir")
    parser.add_argument("--service-normalized-authority", action="store_true")
    parser.add_argument(
        "--runtime-env",
        action="append",
        default=[],
        metavar="TRT_EDGELLM_NAME=VALUE",
        help="Runtime-only diagnostic or ablation setting; may be repeated")
    parser.add_argument(
        "--drop-runtime-env",
        action="append",
        default=[],
        metavar="TRT_EDGELLM_NAME",
        help="Remove a retained runtime setting; may be repeated")
    parser.add_argument(
        "--respect-eos",
        action="store_true",
        help=
        "Separate correctness contract; not comparable to fixed-output throughput"
    )
    parser.add_argument(
        "--no-explicit-slo",
        action="store_true",
        help="Remove composition-root TTFT/TPOT targets from the replay")
    parser.add_argument(
        "--automatic-calibration",
        action="store_true",
        help=
        "Check runtime convergence after each complete generic calibration cycle"
    )
    parser.add_argument(
        "--calibration-max-cycles",
        type=int,
        default=3,
        help="Safety cap for automatic generic calibration (default: 3)")
    parser.add_argument(
        "--calibration-stable-rounds",
        type=int,
        default=1,
        help=
        "Consecutive converged coverage cycles required before serving (default: 1)"
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
                        choices=("dispatch", "audit", "counterfactual",
                                 "full"),
                        default="dispatch")
    args = parser.parse_args()
    if args.calibration_stable_rounds <= 0:
        raise ValueError("Calibration stable rounds must be positive")
    if args.automatic_calibration and args.calibration_stable_rounds > args.calibration_max_cycles:
        raise ValueError(
            "Calibration stable rounds cannot exceed the cycle safety cap")
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
            command = remap_runtime_build(list(selected[case]["command"]),
                                          args.build_cache)
            if args.respect_eos:
                command = enable_eos_termination(command)
            if args.no_explicit_slo:
                command = remove_explicit_slo_contract(command)
            if args.automatic_calibration:
                command = enable_automatic_calibration(
                    command, args.calibration_max_cycles)
            command = remove_runtime_environment(command,
                                                 args.drop_runtime_env)
            command = inject_runtime_environment(command, args.runtime_env)
            for option in ("--trace", "--generic-warmup-trace"):
                if option in command:
                    index = command.index(option) + 1
                    command[index] = str(documents[pathlib.Path(
                        command[index])][0])
            client_index = command.index("--client-script") + 1
            client_implementation = selected[case].get("environment", {}).get(
                "PHASE_TRACE_CLIENT_IMPL", command[client_index])
            if not pathlib.Path(client_implementation).is_file():
                raise FileNotFoundError(
                    "Retained HTTP client implementation is missing: "
                    f"{client_implementation}")
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
            if args.preserve_expired_decode_candidate:
                image = command.index("nvcr.io/nvidia/tensorrt:26.06-py3")
                command[image:image] = [
                    "-e", "TRT_EDGELLM_PRESERVE_EXPIRED_DECODE_CANDIDATE=1"
                ]
            if args.service_normalized_authority:
                image = command.index("nvcr.io/nvidia/tensorrt:26.06-py3")
                command[image:image] = [
                    "-e", "TRT_EDGELLM_SERVICE_NORMALIZED_AUTHORITY=1"
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
            if args.text_engine_dir:
                executable = [
                    i for i, value in enumerate(command)
                    if value.endswith("/examples/llm/llm_phase_context_smoke")
                ]
                if len(executable) != 1 or executable[0] + 1 >= len(command):
                    raise ValueError(
                        "Expected one text engine argument after the phase runtime executable"
                    )
                command[executable[0] + 1] = args.text_engine_dir
            planned.append({
                "policy": policy,
                "case": case,
                "command": command,
                "environment": {
                    "PHASE_TRACE_CLIENT_IMPL": client_implementation,
                    **({
                        "PHASE_CALIBRATION_STABLE_ROUNDS":
                        str(args.calibration_stable_rounds)
                    } if args.automatic_calibration else {})
                }
            })
    (args.output /
     "commands.json").write_text(json.dumps(planned, indent=2) + "\n")
    repository = pathlib.Path(__file__).resolve().parents[2]
    executable = args.build_cache.parent / "examples/llm/llm_phase_context_smoke"
    source_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip()
    source_dirty = bool(
        subprocess.check_output(["git", "status", "--porcelain"],
                                cwd=repository,
                                text=True).strip())
    (args.output / "campaign-manifest.json").write_text(
        json.dumps(
            {
                "source_commit": source_commit,
                "source_dirty": source_dirty,
                "base_commands": str(args.commands.resolve()),
                "base_commands_sha256": file_sha256(args.commands),
                "build_cache": str(args.build_cache.resolve()),
                "build_cache_sha256": file_sha256(args.build_cache),
                "runtime_executable": str(executable.resolve()),
                "runtime_executable_sha256": file_sha256(executable),
                "policies": args.policies,
                "cases": args.cases,
                "repeats": args.repeats,
                "no_explicit_slo": args.no_explicit_slo,
                "automatic_calibration": args.automatic_calibration,
                "calibration_max_cycles": args.calibration_max_cycles,
                "calibration_stable_rounds": args.calibration_stable_rounds,
                "runtime_environment_added": args.runtime_env,
                "runtime_environment_removed": args.drop_runtime_env,
                "dispatch_telemetry": args.dispatch_telemetry,
                "telemetry_level": args.telemetry_level,
                "input_contract": "input-contract.json",
                "commands": "commands.json",
            },
            indent=2) + "\n")
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
