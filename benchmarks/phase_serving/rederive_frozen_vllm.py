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
"""Recover a frozen baseline from retained runs without overwriting its historical summary."""

import argparse
import hashlib
import json
import math
import pathlib
import statistics
import sys

METRICS = ("generated_token_s_median", "achieved_req_s_median",
           "ttft_mean_of_run_means_ms", "ttft_p95_median_ms",
           "tpot_mean_of_run_means_ms", "tpot_p95_median_ms",
           "e2e_mean_of_run_means_ms", "e2e_p95_median_ms")


def identity(path):
    """Return a content identity for a retained input."""
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()
    }


def reconstruct(summary_path, raw_roots, canonical_commands):
    """Require complete recorded success coverage and byte-identical current traces."""
    historical = json.loads(summary_path.read_text())
    commands = json.loads(canonical_commands.read_text())
    traces = {}
    for record in commands:
        command = record["command"]
        path = pathlib.Path(command[command.index("--trace") + 1])
        value = identity(path)
        previous = traces.setdefault(record["case"], value)
        if previous["sha256"] != value["sha256"]:
            raise ValueError("Canonical workload has conflicting trace hashes")
    grouped = {row["workload"]: [] for row in historical["rows"]}
    seen = set()
    for root in raw_roots:
        for path in sorted(root.rglob("client/aggregate.json")):
            if path.resolve() in seen:
                raise ValueError("Repeated raw baseline input: " + str(path))
            seen.add(path.resolve())
            workload = path.relative_to(root).parts[0]
            data = json.loads(path.read_text())
            if data.get("repeats") != 1:
                raise ValueError(
                    "Baseline recovery requires one aggregate per run")
            if data.get("trace_sha256") != traces[workload]["sha256"]:
                raise ValueError("Frozen baseline workload hash differs: " +
                                 str(path))
            origin = identity(path)
            for name, provenance_path in (("commands", path.parent.parent /
                                           "commands.json"),
                                          ("version",
                                           path.parent / "version.json")):
                if provenance_path.exists():
                    origin[name] = identity(provenance_path)
            origin["contract"] = {
                key: data.get(key)
                for key in ("model", "max_in_flight", "ignore_eos",
                            "warmup_requests", "warmup_max_tokens",
                            "captured_token_ids_per_run")
            }
            grouped[workload].append((data, origin))
    rows = []
    for old in historical["rows"]:
        workload = old["workload"]
        samples = grouped[workload]
        if len(samples) != old["success_runs"]:
            raise ValueError("Retained success count differs: " + workload)
        row = {
            key: old[key]
            for key in ("workload", "attempted_runs", "success_runs",
                        "failed_runs")
        }
        row.update(trace_sha256=traces[workload]["sha256"],
                   canonical_trace=traces[workload],
                   raw_origins=[origin for _, origin in samples])
        for metric in METRICS:
            values = [float(data[metric]) for data, _ in samples]
            if not values or not all(
                    math.isfinite(value) and value > 0 for value in values):
                raise ValueError("Invalid raw baseline metric: " + metric)
            reducer = statistics.mean if "mean_of_run_means" in metric else statistics.median
            row["vllm_" + metric] = reducer(values)
        row["vllm_gpu_memory_peak_mib_max"] = max(
            data.get("gpu_memory_peak_mib_max", 0) for data, _ in samples)
        rows.append(row)
    return {
        "state":
        "validation",
        "operation":
        "retained_raw_reaggregation_no_new_gpu_measurement",
        "aggregation":
        "arithmetic_mean_of_run_means_otherwise_median",
        "correction":
        "Historical mean_of_run_means fields were medians of per-run means; raw values are unchanged.",
        "historical_summary":
        identity(summary_path),
        "canonical_commands":
        identity(canonical_commands),
        "raw_roots": [str(root.resolve()) for root in raw_roots],
        "contract_notes": [
            "Every retained trace hash matches the current canonical trace bytes.",
            "Recorded successful and failed counts remain distinct; no failed run is imputed.",
            "Warmup count is retained per origin, not homogenized across workload classes.",
            "Trace identity does not establish equal runtime configuration or identical generated output."
        ],
        "rows":
        rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-summary",
                        type=pathlib.Path,
                        required=True)
    parser.add_argument("--raw-root",
                        type=pathlib.Path,
                        action="append",
                        required=True)
    parser.add_argument("--canonical-commands",
                        type=pathlib.Path,
                        required=True)
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        help="New artifact only; omit to print JSON without writing")
    args = parser.parse_args()
    report = reconstruct(args.historical_summary, args.raw_root,
                         args.canonical_commands)
    report["command"] = [sys.executable] + sys.argv
    report["analyzer"] = identity(pathlib.Path(__file__))
    content = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            stream.write(content)
    else:
        print(content, end="")


if __name__ == "__main__":
    main()
