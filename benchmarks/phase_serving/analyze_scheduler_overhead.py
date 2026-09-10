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
"""Summarize production scheduler decision cost from retained gateway logs."""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import re
import statistics

DECISION_COST = re.compile(
    r"Phase global scheduler decision cost: samples=(\d+) mean=([0-9.]+) us "
    r"p95=([0-9.]+) us max=([0-9.]+) us")


def load(path: pathlib.Path, root: pathlib.Path) -> dict[str, object]:
    """Load the single final decision-cost summary emitted by one server run."""
    matches = DECISION_COST.findall(path.read_text(errors="replace"))
    if len(matches) != 1:
        raise ValueError(f"Expected one decision-cost summary in {path}")
    relative = path.relative_to(root)
    parts = relative.parts
    if "generic" not in parts:
        raise ValueError(f"Cannot infer policy/workload from {relative}")
    generic = parts.index("generic")
    if generic == 0 or generic + 1 >= len(parts):
        raise ValueError(f"Malformed result path: {relative}")
    samples, mean_us, p95_us, maximum_us = matches[0]
    return {
        "policy": parts[generic - 1],
        "workload": parts[generic + 1],
        "path": str(path),
        "samples": int(samples),
        "mean_us": float(mean_us),
        "p95_us": float(p95_us),
        "max_us": float(maximum_us),
    }


def aggregate(runs: list[dict[str, object]]) -> list[dict[str, object]]:
    """Aggregate repeated server runs for each policy/workload pair."""
    groups: dict[tuple[str, str], list[dict[str, object]]] = {}
    for run in runs:
        key = (str(run["policy"]), str(run["workload"]))
        groups.setdefault(key, []).append(run)
    rows = []
    for (policy, workload), values in sorted(groups.items()):
        total_samples = sum(int(value["samples"]) for value in values)
        weighted_mean = (sum(
            int(value["samples"]) * float(value["mean_us"])
            for value in values) /
                         total_samples if total_samples else statistics.median(
                             float(value["mean_us"]) for value in values))
        rows.append({
            "policy":
            policy,
            "workload":
            workload,
            "runs":
            len(values),
            "samples":
            total_samples,
            "weighted_mean_us":
            weighted_mean,
            "run_p95_us_median":
            statistics.median(float(value["p95_us"]) for value in values),
            "maximum_us":
            max(float(value["max_us"]) for value in values),
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=pathlib.Path, required=True)
    parser.add_argument("--output-json", type=pathlib.Path, required=True)
    parser.add_argument("--output-csv", type=pathlib.Path, required=True)
    args = parser.parse_args()
    runs = [
        load(path, args.root) for path in sorted(
            args.root.glob("*/generic/*/worker-*/run-*/gateway.log"))
    ]
    if not runs:
        raise RuntimeError(f"No gateway logs found under {args.root}")
    summary = aggregate(runs)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(
        {
            "schema_version": 1,
            "runs": runs,
            "summary": summary,
        }, indent=2) + "\n",
                                encoding="utf-8")
    with args.output_csv.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print(json.dumps({"runs": len(runs), "groups": len(summary)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
