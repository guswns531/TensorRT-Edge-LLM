#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Compare policy variants from matching phase HTTP workload matrices."""

import argparse
import csv
import json
import pathlib

METRICS = (
    "achieved_req_s_median",
    "generated_token_s_median",
    "ttft_mean_of_run_means_ms",
    "ttft_p95_median_ms",
    "tpot_mean_of_run_means_ms",
    "tpot_p95_median_ms",
    "e2e_mean_of_run_means_ms",
    "e2e_p95_median_ms",
    "gpu_memory_peak_mib_median",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        action="append",
        required=True,
        help=
        "Variant assignment NAME=RESULT_ROOT; the first variant is the baseline.",
    )
    parser.add_argument("--output-json", type=pathlib.Path, required=True)
    parser.add_argument("--output-csv", type=pathlib.Path, required=True)
    parser.add_argument(
        "--allow-token-trace-mismatch",
        action="store_true",
        help=
        "Record token-trace fidelity per row instead of rejecting mismatched variants.",
    )
    return parser.parse_args()


def parse_variant(value: str) -> tuple[str, pathlib.Path]:
    if "=" not in value:
        raise ValueError(f"invalid variant assignment: {value}")
    name, path = value.split("=", 1)
    if not name or not path:
        raise ValueError(f"invalid variant assignment: {value}")
    return name, pathlib.Path(path)


def load_variant(name: str, root: pathlib.Path) -> dict[str, dict]:
    results = {}
    for path in sorted(root.glob("generic/*/worker-4/aggregate.json")):
        workload = path.parents[1].name
        with path.open(encoding="utf-8") as stream:
            results[workload] = json.load(stream)
    if not results:
        raise RuntimeError(
            f"no HTTP workload results found for {name} under {root}")
    return results


def relative_percent(value: float, baseline: float,
                     lower_is_better: bool) -> float:
    if baseline == 0.0:
        return 0.0
    direction = -1.0 if lower_is_better else 1.0
    return direction * (value / baseline - 1.0) * 100.0


def main() -> int:
    args = parse_args()
    variants = [parse_variant(value) for value in args.variant]
    loaded = {name: load_variant(name, root) for name, root in variants}
    baseline_name = variants[0][0]
    workloads = sorted(loaded[baseline_name])
    for name, results in loaded.items():
        if sorted(results) != workloads:
            raise RuntimeError(
                f"workload set for {name} differs from {baseline_name}")

    rows = []
    for workload in workloads:
        baseline = loaded[baseline_name][workload]
        baseline_hash = baseline["token_trace_sha256_per_run"]
        for name, _ in variants:
            result = loaded[name][workload]
            token_trace_matches_baseline = result[
                "token_trace_sha256_per_run"] == baseline_hash
            if not token_trace_matches_baseline and not args.allow_token_trace_mismatch:
                raise RuntimeError(
                    f"token trace differs for {workload}: {baseline_name} vs {name}"
                )
            row = {
                "workload": workload,
                "variant": name,
                "requests": result["requests_per_run"],
                "token_trace_sha256": result["token_trace_sha256_per_run"][0],
                "token_trace_matches_baseline": token_trace_matches_baseline,
            }
            for metric in METRICS:
                value = float(result[metric])
                row[metric] = value
                if name != baseline_name:
                    row[f"{metric}_relative_percent"] = relative_percent(
                        value, float(baseline[metric]),
                        metric.endswith("_ms") or "memory" in metric)
            rows.append(row)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as stream:
        json.dump(
            {
                "baseline": baseline_name,
                "variants": [name for name, _ in variants],
                "rows": rows
            },
            stream,
            indent=2)
        stream.write("\n")
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with args.output_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
