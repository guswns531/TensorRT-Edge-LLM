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
"""Finalize a completed HTTP matrix, then run isolated page-table comparisons on an idle GPU."""

import argparse
import json
import pathlib
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=pathlib.Path, required=True)
    parser.add_argument("--suite", type=pathlib.Path, required=True)
    args = parser.parse_args()
    suite = args.suite.resolve()
    commands = json.loads((suite / "full12/commands.json").read_text())
    expected = []
    for record in commands:
        command = record["command"]
        expected.append(
            (pathlib.Path(command[command.index("--output-dir") + 1]) /
             "aggregate.json", int(command[command.index("--repeats") + 1])))
    deadline = time.monotonic() + 14400
    previous = -1
    while time.monotonic() < deadline:
        completed = 0
        for path, repeats in expected:
            try:
                data = json.loads(path.read_text())
            except (FileNotFoundError, json.JSONDecodeError):
                continue
            completed += data.get("repeats") == repeats
        if completed != previous:
            print(f"Complete cells: {completed}/{len(expected)}", flush=True)
            previous = completed
        if completed == len(expected):
            break
        time.sleep(10)
    else:
        raise TimeoutError(
            "HTTP matrix incomplete; no final success report generated")
    for path, repeats in expected:
        for repeat in range(1, repeats + 1):
            validation = path.parent / f"run-{repeat:03d}" / "client/warmup-validation.json"
            batches = json.loads(validation.read_text())["batches"]
            if not batches or any(
                    batch["failed"] or batch["responses"] != batch["expected"]
                    for batch in batches):
                raise RuntimeError(
                    f"Warmup response validation failed: {validation}")
    script = pathlib.Path(__file__).with_name("report_legacy_parity.py")
    subprocess.run([
        sys.executable,
        str(script), "--results-root",
        str(args.repository / ".local/results"), "--current",
        str(suite / "full12"), "--output",
        str(suite / "report")
    ],
                   check=True)
    distributions = pathlib.Path(__file__).with_name(
        "report_request_distributions.py")
    subprocess.run([
        sys.executable,
        str(distributions), "--summary",
        str(suite / "report/summary.json"), "--results-root",
        str(args.repository / ".local/results"), "--output",
        str(suite / "report"), "--allow-missing-historical"
    ],
                   check=True)
    for _ in range(60):
        result = subprocess.run([
            "nvidia-smi", "--query-gpu=memory.used",
            "--format=csv,noheader,nounits"
        ],
                                capture_output=True,
                                text=True,
                                check=True)
        if all(int(line) < 50 for line in result.stdout.splitlines()):
            break
        time.sleep(1)
    else:
        raise RuntimeError("GPU is not idle; isolated comparison not executed")
    jobs = []
    for repeat in range(5):
        variants = ("upstream-v0101", "v0101-forward-port")
        if repeat % 2:
            variants = variants[::-1]
        for variant in variants:
            binary = suite / f"kv-table-{variant}"
            relative = binary.relative_to(args.repository / ".local")
            command = [
                "docker", "run", "--rm", "--gpus", "all", "-v",
                f"{args.repository / '.local'}:/local", "-e",
                "TRT_PACKAGE_DIR=/opt/tensorrt", "-e",
                "LD_LIBRARY_PATH=/opt/tensorrt/lib:/usr/local/cuda/lib64",
                "nvcr.io/nvidia/tensorrt:26.06-py3", f"/local/{relative}"
            ]
            jobs.append({
                "repeat": repeat + 1,
                "variant": variant,
                "command": command
            })
            with (suite /
                  f"kv-table-{variant}-r{repeat + 1}.log").open("w") as stream:
                subprocess.run(command,
                               stdout=stream,
                               stderr=subprocess.STDOUT,
                               check=True)
    (suite /
     "kv-table-commands.json").write_text(json.dumps(jobs, indent=2) + "\n")
    (suite / "completion.json").write_text(
        json.dumps(
            {
                "http_cells": len(expected),
                "http_runs": sum(repeats for _, repeats in expected),
                "warmup_validated_runs": sum(repeats
                                             for _, repeats in expected),
                "kv_comparisons": len(jobs)
            },
            indent=2) + "\n")
    print("HTTP suite and isolated comparisons complete", flush=True)


if __name__ == "__main__":
    main()
