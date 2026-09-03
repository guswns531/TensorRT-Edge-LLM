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
"""Build cold-to-steady contextual-controller adaptation curves."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

FAMILIES = ("pd", "ep", "ed")
DIRECTIONS = (
    "prefill_to_decode",
    "decode_to_prefill",
    "encoder_to_prefill",
    "prefill_to_encoder",
    "encoder_to_decode",
    "decode_to_encoder",
)
DEFAULT_BINS = (16, 32, 64, 128, 256)


def _records(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    marker = "PHASE_METRIC\t"
    records = []
    for line in path.read_text(encoding="utf-8",
                               errors="replace").splitlines():
        offset = line.find(marker)
        if offset < 0:
            continue
        try:
            records.append(json.loads(line[offset + len(marker):]))
        except json.JSONDecodeError:
            continue
    if not records:
        return [], {}
    latest_epoch = max(
        int(record.get("measurement_epoch", 0)) for record in records)
    selected = [
        record for record in records
        if int(record.get("measurement_epoch", 0)) == latest_epoch
    ]
    baseline = {}
    for record in records:
        if int(record.get("measurement_epoch", 0)) < latest_epoch:
            baseline = record
    return selected, baseline


def _delta(current: dict[str, Any], previous: dict[str, Any],
           key: str) -> float:
    return float(current.get(key, 0.0)) - float(previous.get(key, 0.0))


def _request_frontier(record: dict[str, Any]) -> int:
    identifiers = [
        int(value) for key in ("prefill_request_ids", "decode_request_ids")
        for value in record.get(key, []) if int(value) < 1_000_000
    ]
    return max(identifiers, default=-1)


def analyze_log(path: Path,
                bins: tuple[int, ...] = DEFAULT_BINS,
                minimum_observations: int = 4,
                maximum_rmse: float = 0.20) -> dict[str, Any]:
    records, epoch_baseline = _records(path)
    if not records:
        raise ValueError(f"no PHASE_METRIC records in {path}")
    boundaries = (*bins, math.inf)
    rows = []
    previous = epoch_baseline
    begin = 1
    measurement_start_us = float(records[0].get("host_dispatch_start_us", 0.0))
    for end in boundaries:
        upper = len(records) if math.isinf(end) else min(
            len(records), int(end))
        if upper < begin:
            continue
        selected = records[begin - 1:upper]
        if not selected:
            continue
        final = selected[-1]
        row: dict[str, Any] = {
            "decision_begin":
            begin,
            "decision_end":
            upper,
            "request_frontier":
            max((_request_frontier(record) for record in selected),
                default=-1),
            "elapsed_end_ms":
            max(
                0.0,
                float(final.get("host_dispatch_start_us",
                                measurement_start_us)) - measurement_start_us)
            / 1000.0,
            "policy_warmup_mode":
            final.get("policy_warmup_mode", "unknown"),
            "families": {},
            "directions": {},
        }
        for family in FAMILIES:
            observations_key = f"contextual_{family}_calibration_observations"
            squared_key = f"contextual_{family}_squared_error_sum"
            absolute_key = f"contextual_{family}_absolute_error_sum"
            false_safe_key = f"contextual_{family}_false_safe"
            predicted_safe_key = f"contextual_{family}_predicted_safe"
            observations = max(
                0, round(_delta(final, previous, observations_key)))
            squared_error = max(0.0, _delta(final, previous, squared_key))
            absolute_error = max(0.0, _delta(final, previous, absolute_key))
            predicted_safe = max(
                0, round(_delta(final, previous, predicted_safe_key)))
            false_safe = max(0, round(_delta(final, previous, false_safe_key)))
            row["families"][family] = {
                "observations":
                observations,
                "cumulative_observations":
                int(final.get(observations_key, 0)),
                "mae":
                absolute_error / observations if observations else 0.0,
                "rmse":
                math.sqrt(squared_error /
                          observations) if observations else 0.0,
                "last_mean":
                float(final.get(f"contextual_{family}_last_mean", 0.0)),
                "last_uncertainty":
                float(final.get(f"contextual_{family}_last_uncertainty", 0.0)),
                "last_lcb":
                float(final.get(f"contextual_{family}_last_lcb", 0.0)),
                "predicted_safe":
                predicted_safe,
                "false_safe":
                false_safe,
                "false_safe_rate":
                false_safe / predicted_safe if predicted_safe else 0.0,
            }
        for direction in DIRECTIONS:
            key = f"contextual_{direction}_observations"
            row["directions"][direction] = {
                "observations": max(0, round(_delta(final, previous, key))),
                "cumulative_observations": int(final.get(key, 0)),
            }
        rows.append(row)
        previous = final
        begin = upper + 1
        if upper == len(records):
            break

    stability = {}
    for family in FAMILIES:
        stable_bins = 0
        first = None
        first_time_ms = None
        first_request_frontier = None
        last_violation = None
        qualified = []
        for row in rows:
            values = row["families"][family]
            measured_observations = (values["cumulative_observations"] - int(
                epoch_baseline.get(
                    f"contextual_{family}_calibration_observations", 0)))
            enough = measured_observations >= minimum_observations
            accurate = values["observations"] > 0 and values[
                "rmse"] <= maximum_rmse
            safe = values["false_safe"] == 0
            qualifies = enough and accurate and safe
            qualified.append(qualifies)
            stable_bins = stable_bins + 1 if qualifies else 0
            if not qualifies:
                last_violation = row["decision_end"]
            if stable_bins >= 2 and first is None:
                first = row["decision_end"]
                first_time_ms = row["elapsed_end_ms"]
                first_request_frontier = row["request_frontier"]
        final_stable = len(qualified) >= 2 and all(qualified[-2:])
        stability[family] = {
            "stable": final_stable,
            "decision_to_stability": first,
            "time_to_stability_ms": first_time_ms,
            "request_frontier_to_stability": first_request_frontier,
            "final_stable": final_stable,
            "last_unstable_bin_end": last_violation,
            "stability_lost_after_first": first is not None
            and not final_stable,
            "minimum_observations": minimum_observations,
            "maximum_rmse": maximum_rmse,
            "required_consecutive_bins": 2,
        }
    return {
        "schema_version": 2,
        "source": str(path),
        "measurement_epoch": int(records[-1].get("measurement_epoch", 0)),
        "policy_warmup_mode": records[-1].get("policy_warmup_mode", "unknown"),
        "decisions": len(records),
        "epoch_baseline": {
            family:
            int(
                epoch_baseline.get(
                    f"contextual_{family}_calibration_observations", 0))
            for family in FAMILIES
        },
        "bins": rows,
        "stability": stability,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--bins", default="16,32,64,128,256")
    parser.add_argument("--minimum-observations", type=int, default=4)
    parser.add_argument("--maximum-rmse", type=float, default=0.20)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    bins = tuple(int(value) for value in args.bins.split(",") if value)
    if (not bins or sorted(bins) != list(bins)
            or args.minimum_observations <= 0 or args.maximum_rmse < 0.0):
        parser.error("bins must increase and thresholds must be non-negative")
    try:
        result = analyze_log(args.log, bins, args.minimum_observations,
                             args.maximum_rmse)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    serialized = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
