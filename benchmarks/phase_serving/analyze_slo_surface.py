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
"""Compute a post-hoc joint-SLO surface from request-level serving results."""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import statistics

import analyze_slo_goodput


def _parse_run(spec: str) -> tuple[str, pathlib.Path]:
    label, separator, raw_path = spec.partition("=")
    if not separator or not label or not raw_path:
        raise ValueError("--run must use LABEL=PATH")
    return label, pathlib.Path(raw_path)


def surface(runs: list[tuple[str, pathlib.Path]], ttft_values: list[float],
            tpot_values: list[float],
            e2e_values: list[float]) -> list[dict[str, object]]:
    """Evaluate every run at every supplied post-hoc SLO point."""
    rows: list[dict[str, object]] = []
    for label, path in runs:
        for ttft_ms in ttft_values:
            for tpot_ms in tpot_values:
                for e2e_ms in e2e_values:
                    result = analyze_slo_goodput.summarize(
                        path, ttft_ms, tpot_ms, e2e_ms)
                    rows.append({
                        "label": label,
                        "requests_csv": str(path),
                        "ttft_ms": ttft_ms,
                        "tpot_ms": tpot_ms,
                        "e2e_ms": e2e_ms,
                        **result,
                    })
    return rows


def aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Aggregate repeated runs without collapsing distinct SLO points."""
    groups: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in rows:
        key = (row["label"], row["ttft_ms"], row["tpot_ms"], row["e2e_ms"])
        groups.setdefault(key, []).append(row)
    metrics = (
        "requests",
        "passed_requests",
        "pass_rate",
        "duration_s",
        "request_goodput_per_s",
        "token_goodput_per_s",
        "request_throughput_per_s",
        "token_throughput_per_s",
    )
    result = []
    for key, values in sorted(groups.items()):
        row = {
            "label": key[0],
            "ttft_ms": key[1],
            "tpot_ms": key[2],
            "e2e_ms": key[3],
            "repeats": len(values),
        }
        for metric in metrics:
            row[metric] = statistics.median(
                float(value[metric]) for value in values)
        result.append(row)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",
                        action="append",
                        required=True,
                        help="LABEL=path/to/requests.csv; may be repeated")
    parser.add_argument("--ttft-ms", type=float, nargs="+", required=True)
    parser.add_argument("--tpot-ms", type=float, nargs="+", required=True)
    parser.add_argument("--e2e-ms", type=float, nargs="+", default=[0.0])
    parser.add_argument("--output-json", type=pathlib.Path, required=True)
    parser.add_argument("--output-csv", type=pathlib.Path, required=True)
    args = parser.parse_args()

    run_points = surface([_parse_run(spec) for spec in args.run], args.ttft_ms,
                         args.tpot_ms, args.e2e_ms)
    rows = aggregate(run_points)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(
        {
            "schema_version": 1,
            "run_points": run_points,
            "surface": rows,
        },
        indent=2) + "\n",
                                encoding="utf-8")
    fields = [
        "label", "ttft_ms", "tpot_ms", "e2e_ms", "repeats", "requests",
        "passed_requests", "pass_rate", "duration_s", "request_goodput_per_s",
        "token_goodput_per_s", "request_throughput_per_s",
        "token_throughput_per_s"
    ]
    with args.output_csv.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output,
                                fieldnames=fields,
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(
        json.dumps(
            {
                "runs": len(args.run),
                "points": len(run_points),
                "surface_points": len(rows)
            },
            sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
