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
"""Summarize a vLLM option sweep without requiring third-party packages."""

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

METRICS = [
    "generated_token_s_median", "achieved_req_s_median", "ttft_median_of_run_medians_ms",
    "ttft_p95_median_ms", "tpot_median_of_run_medians_ms", "tpot_p95_median_ms",
    "e2e_median_of_run_medians_ms", "e2e_p95_median_ms"
]


def geometric_mean(values: list[float]) -> float:
    return math.exp(sum(math.log(value) for value in values) / len(values))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--baseline", default="budget-8192")
    parser.add_argument("--output-prefix", type=Path)
    args = parser.parse_args()

    rows = []
    failures = []
    by_configuration: dict[str, list[dict[str, Any]]] = {}
    for config_dir in sorted(path for path in args.input_root.iterdir() if path.is_dir()):
        status_path = config_dir / "status.json"
        status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.is_file() else {}
        if status.get("state") == "failed":
            failures.append({
                "configuration": config_dir.name,
                "error": status.get("error", "unknown failure"),
            })
        ready_memory = status.get("gpu_memory_ready") or {}
        for aggregate_path in sorted(config_dir.glob("*/aggregate.json")):
            aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
            row = {
                "configuration": config_dir.name,
                "workload": aggregate_path.parent.name,
                "gpu_memory_ready_mib": ready_memory.get("used_mib", ""),
                **{metric: aggregate[metric] for metric in METRICS},
            }
            rows.append(row)
            by_configuration.setdefault(config_dir.name, []).append(row)

    baseline_by_workload = {
        row["workload"]: row for row in by_configuration.get(args.baseline, [])
    }
    ranking = []
    for configuration, config_rows in by_configuration.items():
        matched = [row for row in config_rows if row["workload"] in baseline_by_workload]
        request_ratios = [
            float(row["achieved_req_s_median"]) /
            float(baseline_by_workload[row["workload"]]["achieved_req_s_median"])
            for row in matched
        ]
        token_ratios = [
            float(row["generated_token_s_median"]) /
            float(baseline_by_workload[row["workload"]]["generated_token_s_median"])
            for row in matched
        ]
        e2e_ratios = [
            float(baseline_by_workload[row["workload"]]["e2e_p95_median_ms"]) /
            float(row["e2e_p95_median_ms"]) for row in matched
        ]
        if not matched:
            continue
        ranking.append({
            "configuration": configuration,
            "workloads": len(matched),
            "request_rate_geomean_change_pct": (geometric_mean(request_ratios) - 1.0) * 100.0,
            "token_rate_geomean_change_pct": (geometric_mean(token_ratios) - 1.0) * 100.0,
            "e2e_p95_geomean_improvement_pct": (geometric_mean(e2e_ratios) - 1.0) * 100.0,
        })
    ranking.sort(key=lambda row: float(row["request_rate_geomean_change_pct"]), reverse=True)

    output_prefix = args.output_prefix or args.input_root / "option-sweep"
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    write_csv(output_prefix.with_name(output_prefix.name + "-summary.csv"), rows)
    write_csv(output_prefix.with_name(output_prefix.name + "-ranking.csv"), ranking)
    write_csv(output_prefix.with_name(output_prefix.name + "-failures.csv"), failures)
    print(json.dumps({"ranking": ranking, "failures": failures}, indent=2))


if __name__ == "__main__":
    main()
