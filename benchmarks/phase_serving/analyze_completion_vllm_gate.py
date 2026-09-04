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
"""Compare the promoted completion policy with a fresh vLLM HTTP suite."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any

try:
    from benchmarks.phase_serving.analyze_slo_goodput import \
        summarize as summarize_slo
except ModuleNotFoundError:
    from analyze_slo_goodput import summarize as summarize_slo

LATENCY_FIELDS = (
    "ttft_mean_ms",
    "ttft_p95_ms",
    "tpot_mean_ms",
    "tpot_p95_ms",
    "e2e_mean_ms",
    "e2e_p95_ms",
)


def _relative_percent(value: float, reference: float) -> float | None:
    if reference == 0.0:
        return 0.0 if value == 0.0 else None
    return 100.0 * (value / reference - 1.0)


def _load_vllm_case(case_root: Path, summary: dict[str, Any], ttft_ms: float,
                    tpot_ms: float, e2e_ms: float) -> dict[str, Any]:
    runs = summary.get("runs", [])
    if not runs:
        raise ValueError(f"{case_root}: vLLM case has no completed run")
    request_csvs = sorted(case_root.glob("run-*/client/run-*/requests.csv"))
    if len(request_csvs) != len(runs):
        raise ValueError(
            f"{case_root}: vLLM request CSV count does not match runs")
    slo_runs = [
        summarize_slo(path, ttft_ms, tpot_ms, e2e_ms) for path in request_csvs
    ]
    return {
        "throughput_req_s":
        statistics.median(float(run["achieved_req_s_median"]) for run in runs),
        "token_s":
        float(summary["generated_token_s_median"]),
        "ttft_mean_ms":
        float(summary["ttft_mean_ms"]),
        "ttft_p95_ms":
        float(summary["ttft_p95_ms"]),
        "tpot_mean_ms":
        float(summary["tpot_mean_ms"]),
        "tpot_p95_ms":
        float(summary["tpot_p95_ms"]),
        "e2e_mean_ms":
        float(summary["e2e_mean_ms"]),
        "e2e_p95_ms":
        float(summary["e2e_p95_ms"]),
        "joint_slo_pass_rate":
        statistics.median(float(run["pass_rate"]) for run in slo_runs),
        "joint_slo_goodput_req_s":
        statistics.median(
            float(run["request_goodput_per_s"]) for run in slo_runs),
        "gpu_memory_peak_mib":
        float(summary["gpu_memory_peak_mib"]),
        "repeats":
        len(runs),
        "request_csvs": [str(path) for path in request_csvs],
    }


def build_comparison(policy_matrix: Path, policy: str, vllm_root: Path,
                     ttft_ms: float, tpot_ms: float,
                     e2e_ms: float) -> dict[str, Any]:
    """Build workload-level policy versus fresh-vLLM comparisons."""
    matrix = json.loads(policy_matrix.read_text(encoding="utf-8"))
    policy_rows = {
        str(row["workload"]): row
        for row in matrix.get("rows", []) if row.get("policy") == policy
    }
    vllm_suite = json.loads(
        (vllm_root / "summary.json").read_text(encoding="utf-8"))
    vllm_cases = vllm_suite.get("cases", {})
    if set(policy_rows) != set(vllm_cases):
        raise ValueError("policy and vLLM workload sets do not match")

    rows = []
    for workload in sorted(policy_rows):
        current = policy_rows[workload]
        vllm = _load_vllm_case(vllm_root / workload, vllm_cases[workload],
                               ttft_ms, tpot_ms, e2e_ms)
        row = {
            "workload": workload,
            "current_repeats": int(current["slo_repeats"]),
            "vllm_repeats": int(vllm["repeats"]),
        }
        for field in ("throughput_req_s", "token_s", "joint_slo_goodput_req_s",
                      *LATENCY_FIELDS):
            current_value = float(current[field])
            vllm_value = float(vllm[field])
            row[f"current_{field}"] = current_value
            row[f"vllm_{field}"] = vllm_value
            row[f"current_vs_vllm_{field}_percent"] = _relative_percent(
                current_value, vllm_value)
        row["current_joint_slo_pass_rate"] = float(
            current["joint_slo_pass_rate"])
        row["vllm_joint_slo_pass_rate"] = float(vllm["joint_slo_pass_rate"])
        current_aggregate = json.loads(
            Path(current["aggregate"]).read_text(encoding="utf-8"))
        row["current_gpu_memory_peak_mib"] = float(
            current_aggregate["gpu_memory_peak_mib_median"])
        row["vllm_gpu_memory_peak_mib"] = float(vllm["gpu_memory_peak_mib"])
        rows.append(row)

    lower_is_better = LATENCY_FIELDS
    return {
        "schema_version": 1,
        "artifact": "completion_vllm_gate",
        "policy": policy,
        "policy_matrix": str(policy_matrix),
        "vllm_root": str(vllm_root),
        "slo": {
            "ttft_ms": ttft_ms,
            "tpot_ms": tpot_ms,
            "e2e_ms": e2e_ms,
        },
        "summary": {
            "workloads":
            len(rows),
            "geometric_mean_token_throughput_vs_vllm_percent":
            100.0 * (math.prod(row["current_token_s"] / row["vllm_token_s"]
                               for row in rows)**(1.0 / len(rows)) - 1.0),
            "throughput_wins":
            sum(row["current_throughput_req_s"] > row["vllm_throughput_req_s"]
                for row in rows),
            "token_throughput_wins":
            sum(row["current_token_s"] > row["vllm_token_s"] for row in rows),
            "joint_slo_goodput_wins":
            sum(row["current_joint_slo_goodput_req_s"] >
                row["vllm_joint_slo_goodput_req_s"] for row in rows),
            "gpu_memory_peak_wins":
            sum(row["current_gpu_memory_peak_mib"] <=
                row["vllm_gpu_memory_peak_mib"] for row in rows),
            **{
                f"{field}_wins":
                sum(row[f"current_{field}"] < row[f"vllm_{field}"] for row in rows)
                for field in lower_is_better
            },
        },
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-matrix", required=True, type=Path)
    parser.add_argument("--policy", default="scalar")
    parser.add_argument("--vllm-root", required=True, type=Path)
    parser.add_argument("--ttft-ms", type=float, default=500.0)
    parser.add_argument("--tpot-ms", type=float, default=50.0)
    parser.add_argument("--e2e-ms", type=float, default=2500.0)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = build_comparison(args.policy_matrix, args.policy,
                                  args.vllm_root, args.ttft_ms, args.tpot_ms,
                                  args.e2e_ms)
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", newline="", encoding="utf-8") as target:
            writer = csv.DictWriter(target, fieldnames=list(result["rows"][0]))
            writer.writeheader()
            writer.writerows(result["rows"])
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        parser.error(str(error))
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
