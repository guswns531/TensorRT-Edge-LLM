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
"""Report direction-specific held-out completion calibration."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


def finite(row: dict[str, str], key: str) -> float | None:
    """Return a finite CSV scalar."""
    try:
        value = float(row[key])
    except (KeyError, ValueError):
        return None
    return value if math.isfinite(value) else None


def percentile(values: list[float], fraction: float) -> float | None:
    """Return a nearest-rank percentile."""
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(
        len(ordered) - 1,
        math.ceil(fraction * len(ordered)) - 1)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--beta", type=float, default=1.96)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    args = parser.parse_args()
    rows = list(csv.DictReader(args.input.open(encoding="utf-8")))
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("completion_prediction_ready") == "True":
            grouped[row["direction"]].append(row)

    directions = {}
    for direction, samples in sorted(grouped.items()):
        split = max(0, int(len(samples) * (1.0 - args.test_fraction)))
        held = samples[split:]
        errors = []
        covered = 0
        false_safe = 0
        evaluated = 0
        for row in held:
            for prefix in ("incumbent", "newcomer"):
                error = finite(row, f"{prefix}_prediction_absolute_error_us")
                if error is None:
                    continue
                errors.append(error)
                covered += row.get(f"{prefix}_interval_covered") == "True"
                evaluated += 1
            # A selected action whose measured horizon exceeds its serial
            # equivalent is a false-safe observation for active authority.
            compression = finite(row, "serial_equivalent_compression")
            if compression is not None and compression < 0.0:
                false_safe += 1
        directions[direction] = {
            "samples": len(samples),
            "held_out_samples": len(held),
            "evaluated_completion_components": evaluated,
            "mae_us": statistics.fmean(errors) if errors else None,
            "median_absolute_error_us":
            statistics.median(errors) if errors else None,
            "p95_absolute_error_us": percentile(errors, 0.95),
            "empirical_95_coverage":
            covered / evaluated if evaluated else None,
            "false_safe": false_safe,
        }
    artifact = {
        "schema_version": 1,
        "beta": args.beta,
        "chronological_test_fraction": args.test_fraction,
        "directions": directions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True) +
                           "\n",
                           encoding="utf-8")
    print(json.dumps(artifact, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
