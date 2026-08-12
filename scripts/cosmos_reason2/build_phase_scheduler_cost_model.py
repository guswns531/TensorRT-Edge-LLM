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

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Aggregate CUDA-event kernel rows into a model-neutral scheduler cost model."""

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position -
                                                                 lower)


def upper_bucket(value: int, buckets: list[int]) -> int:
    for bucket in buckets:
        if value <= bucket:
            return bucket
    return buckets[-1]


def read_rows(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as stream:
            rows.extend(csv.DictReader(stream))
    if not rows:
        raise RuntimeError("no kernel-group rows found")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        nargs="+",
        required=True,
        help="kernel-groups.csv files or directories searched recursively")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--context-buckets",
                        type=int,
                        nargs="+",
                        default=[128, 512, 1024, 1536, 2048])
    parser.add_argument("--min-samples", type=int, default=3)
    args = parser.parse_args()

    buckets = sorted(set(args.context_buckets))
    if not buckets or buckets[0] <= 0 or args.min_samples <= 0:
        parser.error("context buckets and min-samples must be positive")
    paths: list[Path] = []
    for source in args.input:
        if source.is_dir():
            paths.extend(sorted(source.rglob("kernel-groups.csv")))
        elif source.is_file():
            paths.append(source)
    if not paths:
        parser.error("no kernel-groups.csv inputs found")

    grouped: dict[tuple[str, int, int], list[float]] = {}
    for row in read_rows(paths):
        group = row["group"]
        if group not in {"prefill_engine", "decode_engine"}:
            continue
        if group == "decode_engine":
            batch = int(row["decode_batch"])
            total_context = int(row["decode_context_tokens"])
            if batch <= 0:
                continue
            context = upper_bucket(math.ceil(total_context / batch), buckets)
            phase = "decode"
        else:
            batch = int(row["prefill_batch"])
            total_tokens = int(row["prefill_tokens"])
            if batch <= 0:
                continue
            context = upper_bucket(math.ceil(total_tokens / batch), buckets)
            phase = "prefill"
        grouped.setdefault((phase, batch, context),
                           []).append(float(row["gpu_ms"]))

    normalized: list[dict[str, object]] = []
    for (phase, batch, context), values in sorted(grouped.items()):
        if len(values) < args.min_samples:
            continue
        normalized.append({
            "phase": phase,
            "batch_size": batch,
            "max_context_length": context,
            "samples": len(values),
            "median_gpu_ms": statistics.median(values),
            "p95_gpu_ms": percentile(values, 0.95),
            "max_gpu_ms": max(values),
        })
    if not any(point["phase"] == "decode" for point in normalized):
        raise RuntimeError("no decode cost point met the sample threshold")

    root = {
        "schema_version":
        1,
        "source_files": [str(path) for path in paths],
        "decode": [{
            key: value
            for key, value in point.items() if key != "phase"
        } for point in normalized if point["phase"] == "decode"],
        "prefill": [{
            key: value
            for key, value in point.items() if key != "phase"
        } for point in normalized if point["phase"] == "prefill"],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(root, indent=2) + "\n",
                                encoding="utf-8")
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(normalized[0]))
        writer.writeheader()
        writer.writerows(normalized)


if __name__ == "__main__":
    main()
