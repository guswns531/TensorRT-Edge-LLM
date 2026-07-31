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
"""Pool three benchmark repeats and emit legacy/indexed median and p95."""

import argparse
import csv
import math
import statistics
from pathlib import Path


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    rank = (len(ordered) - 1) * fraction
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (rank - lower)


def load_samples(root: Path,
                 mode: str,
                 prefix: str = "bench") -> dict[str, list[float]]:
    samples: dict[str, list[float]] = {}
    pattern = f"{prefix}-{mode}/run-*/bs-*/*_samples.csv"
    for path in sorted(root.glob(pattern)):
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        if not rows:
            raise RuntimeError(f"empty sample file: {path}")
        first = rows[0]
        phase = first["mode"]
        batch = int(first["batch_size"])
        length_name = "input_len" if phase == "prefill" else "past_kv_len"
        length = int(first[length_name])
        scenario = f"{phase}_bs{batch}_{length_name}{length}"
        samples.setdefault(scenario, []).extend(
            float(row["latency_ms"]) for row in rows)
    return samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("work_dir", type=Path)
    parser.add_argument("output_csv", type=Path)
    parser.add_argument("--expected-samples", type=int, default=300)
    parser.add_argument("--limit", type=float, default=0.03)
    parser.add_argument(
        "--use-confirmation",
        action="append",
        default=[],
        help=
        "Replace this scenario with samples from bench-confirm-{legacy,indexed}.",
    )
    args = parser.parse_args()

    by_mode = {
        mode: load_samples(args.work_dir, mode)
        for mode in ("legacy", "indexed")
    }
    for mode in ("legacy", "indexed"):
        confirmation = load_samples(args.work_dir, mode, "bench-confirm")
        for scenario in args.use_confirmation:
            if scenario not in confirmation:
                raise RuntimeError(
                    f"missing {mode} confirmation samples for {scenario}")
            by_mode[mode][scenario] = confirmation[scenario]
    if set(by_mode["legacy"]) != set(by_mode["indexed"]):
        raise RuntimeError("legacy/indexed scenario sets differ")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as stream:
        fieldnames = [
            "scenario", "stat", "legacy_ms", "indexed_ms", "regression", "pass"
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for scenario in sorted(by_mode["legacy"]):
            for mode in ("legacy", "indexed"):
                count = len(by_mode[mode][scenario])
                if count != args.expected_samples:
                    raise RuntimeError(
                        f"{mode} {scenario}: expected {args.expected_samples} samples, got {count}"
                    )
            legacy_values = by_mode["legacy"][scenario]
            indexed_values = by_mode["indexed"][scenario]
            stats = (
                ("median", statistics.median(legacy_values),
                 statistics.median(indexed_values)),
                ("p95", percentile(legacy_values,
                                   0.95), percentile(indexed_values, 0.95)),
            )
            for stat, legacy_ms, indexed_ms in stats:
                regression = indexed_ms / legacy_ms - 1.0
                writer.writerow({
                    "scenario": scenario,
                    "stat": stat,
                    "legacy_ms": f"{legacy_ms:.6f}",
                    "indexed_ms": f"{indexed_ms:.6f}",
                    "regression": f"{regression:.6f}",
                    "pass": str(regression <= args.limit).lower(),
                })


if __name__ == "__main__":
    main()
