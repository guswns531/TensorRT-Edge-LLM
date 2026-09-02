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
"""Summarize same-runtime completion-policy ablations across HTTP traces."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any

from analyze_slo_goodput import summarize as summarize_slo


def parse_policy(spec: str) -> tuple[str, Path]:
    """Parse NAME=ROOT policy input."""
    name, separator, raw_root = spec.partition("=")
    if not separator or not name or not raw_root:
        raise ValueError("--policy must use NAME=ROOT")
    return name, Path(raw_root)


def load_policy(name: str, root: Path, ttft_ms: float, tpot_ms: float,
                e2e_ms: float) -> list[dict[str, Any]]:
    """Load every completed workload under one policy root."""
    rows = []
    for aggregate_path in sorted(root.glob("*/worker-4/aggregate.json")):
        workload = aggregate_path.parents[1].name
        aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
        request_csv = aggregate_path.parent / "run-001/client/run-001/requests.csv"
        slo = summarize_slo(request_csv, ttft_ms, tpot_ms, e2e_ms)
        rows.append({
            "policy":
            name,
            "workload":
            workload,
            "throughput_req_s":
            float(aggregate["achieved_req_s_median"]),
            "token_s":
            float(aggregate["generated_token_s_median"]),
            "ttft_mean_ms":
            float(aggregate["ttft_mean_of_run_means_ms"]),
            "ttft_p95_ms":
            float(aggregate["ttft_p95_median_ms"]),
            "tpot_mean_ms":
            float(aggregate["tpot_mean_of_run_means_ms"]),
            "tpot_p95_ms":
            float(aggregate["tpot_p95_median_ms"]),
            "e2e_mean_ms":
            float(aggregate["e2e_mean_of_run_means_ms"]),
            "e2e_p95_ms":
            float(aggregate["e2e_p95_median_ms"]),
            "joint_slo_pass_rate":
            float(slo["pass_rate"]),
            "joint_slo_goodput_req_s":
            float(slo["request_goodput_per_s"]),
            "token_trace_sha256":
            list(aggregate.get("token_trace_sha256_per_run", [])),
            "aggregate":
            str(aggregate_path),
            "requests_csv":
            str(request_csv),
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", action="append", required=True)
    parser.add_argument("--reference-policy", default="current")
    parser.add_argument("--ttft-ms", type=float, default=500.0)
    parser.add_argument("--tpot-ms", type=float, default=50.0)
    parser.add_argument("--e2e-ms", type=float, default=2500.0)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    try:
        for spec in args.policy:
            name, root = parse_policy(spec)
            rows.extend(
                load_policy(name, root, args.ttft_ms, args.tpot_ms,
                            args.e2e_ms))
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        parser.error(str(error))

    reference = {
        row["workload"]: row
        for row in rows if row["policy"] == args.reference_policy
    }
    if not reference:
        parser.error("reference policy has no completed workload")
    for row in rows:
        baseline = reference.get(row["workload"])
        if baseline is None:
            row["throughput_vs_reference_percent"] = None
            row["token_identity_with_reference"] = None
            continue
        row["throughput_vs_reference_percent"] = 100.0 * (
            row["throughput_req_s"] / baseline["throughput_req_s"] - 1.0)
        row["token_identity_with_reference"] = (set(
            row["token_trace_sha256"]) == set(baseline["token_trace_sha256"]))

    summaries = {}
    policies = sorted({row["policy"] for row in rows})
    for policy in policies:
        members = [row for row in rows if row["policy"] == policy]
        changes = [
            row["throughput_vs_reference_percent"] for row in members
            if row["throughput_vs_reference_percent"] is not None
        ]
        summaries[policy] = {
            "completed_workloads":
            len(members),
            "macro_throughput_req_s":
            statistics.fmean(row["throughput_req_s"] for row in members),
            "macro_joint_slo_goodput_req_s":
            statistics.fmean(row["joint_slo_goodput_req_s"]
                             for row in members),
            "macro_throughput_vs_reference_percent":
            statistics.fmean(changes) if changes else None,
            "throughput_wins_vs_reference":
            sum(change > 0.0 for change in changes),
            "exact_token_identity_workloads":
            sum(row["token_identity_with_reference"] is True
                for row in members),
        }

    artifact = {
        "schema_version": 1,
        "reference_policy": args.reference_policy,
        "slo": {
            "ttft_ms": args.ttft_ms,
            "tpot_ms": args.tpot_ms,
            "e2e_ms": args.e2e_ms,
        },
        "policy_summary": summaries,
        "rows": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    csv_rows = [{
        key: value
        for key, value in row.items()
        if key not in {"token_trace_sha256", "aggregate", "requests_csv"}
    } for row in rows]
    with args.output_csv.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(json.dumps(summaries, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
