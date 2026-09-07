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
"""Summarize fixed-budget phase-policy calibration and serving performance."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from pathlib import Path
from typing import Any


METRICS = (
    "generated_token_s_median",
    "achieved_req_s_median",
    "ttft_mean_of_run_means_ms",
    "ttft_p95_median_ms",
    "tpot_mean_of_run_means_ms",
    "tpot_p95_median_ms",
    "e2e_mean_of_run_means_ms",
    "e2e_p95_median_ms",
)
DIRECTIONS = ("prefill_to_decode", "decode_to_prefill",
              "encoder_to_prefill", "prefill_to_encoder",
              "encoder_to_decode", "decode_to_encoder")
LOG_TIMESTAMP = re.compile(
    r"^\[(\d{2}):(\d{2}):(\d{2})\.(\d{3})\]")


def _directions(calibration: dict[str, Any]) -> list[dict[str, Any]]:
    result = []
    for family in calibration.get("contextual_policy_calibration", {}).values():
        result.extend(family.get("directions", []))
    return result


def _load_calibration(case_dir: Path) -> dict[str, Any]:
    records = []
    for path in sorted(case_dir.glob("run-*/client/calibration.json")):
        history = json.loads(path.read_text(encoding="utf-8"))
        if history:
            records.append(history[-1])
    directions = [direction for record in records
                  for direction in _directions(record)]
    observations = [float(item.get("observations", 0))
                    for item in directions]
    required = [item for item in directions if item.get("required", False)]
    ready = [item for item in required if item.get("ready", False)]
    summary = {
        "calibration_runs": len(records),
        "overall_converged_fraction": (
            sum(bool(record.get("calibration_converged", False))
                for record in records) / len(records) if records else 0.0),
        "contextual_converged_fraction": (
            sum(bool(record.get("contextual_policy_calibration_converged",
                                False)) for record in records) /
            len(records) if records else 0.0),
        "direction_observations_median": (
            statistics.median(observations) if observations else 0.0),
        "required_directions": len(required),
        "ready_required_directions": len(ready),
        "authority_ready_fraction": (len(ready) / len(required)
                                     if required else 0.0),
    }
    for name in DIRECTIONS:
        matching = [item for item in directions if item.get("direction") == name]
        observations = [float(item.get("observations", 0))
                        for item in matching]
        predictions = [float(item.get("predictions", 0))
                       for item in matching]
        summary[f"{name}_observations_median"] = (
            statistics.median(observations) if observations else 0.0)
        summary[f"{name}_ready_fraction"] = (
            sum(bool(item.get("ready", False)) for item in matching) /
            len(matching) if matching else 0.0)
        summary[f"{name}_predictions_median"] = (
            statistics.median(predictions) if predictions else 0.0)
    return summary


def _load_run_stability(case_dir: Path) -> dict[str, Any]:
    path = case_dir / "runs.csv"
    if not path.is_file():
        return {"run_count": 0, "token_s_min": 0.0, "token_s_max": 0.0,
                "token_s_cv_pct": 0.0}
    with path.open(newline="", encoding="utf-8") as stream:
        values = [float(row["generated_token_s_median"])
                  for row in csv.DictReader(stream)]
    mean = statistics.fmean(values) if values else 0.0
    return {
        "run_count": len(values),
        "token_s_min": min(values) if values else 0.0,
        "token_s_max": max(values) if values else 0.0,
        "token_s_cv_pct": (statistics.pstdev(values) / mean * 100.0
                           if len(values) > 1 and mean > 0.0 else 0.0),
    }


def _load_process_wall(case_dir: Path) -> dict[str, Any]:
    durations = []
    for path in sorted(case_dir.glob("run-*/gateway.log")):
        timestamps = []
        with path.open(encoding="utf-8", errors="replace") as source:
            for line in source:
                match = LOG_TIMESTAMP.match(line)
                if match is None:
                    continue
                hour, minute, second, millisecond = map(int, match.groups())
                timestamps.append(
                    (((hour * 60 + minute) * 60 + second) * 1000 +
                     millisecond) / 1000.0)
        if timestamps:
            duration = timestamps[-1] - timestamps[0]
            if duration < 0.0:
                duration += 24.0 * 60.0 * 60.0
            durations.append(duration)
    return {
        "process_wall_s_median": (
            statistics.median(durations) if durations else 0.0),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for budget_dir in sorted(args.input_dir.glob("warmup-*"),
                             key=lambda path: int(path.name.split("-", 1)[1])):
        budget = int(budget_dir.name.split("-", 1)[1])
        for aggregate in sorted(budget_dir.glob("*/*/aggregate.json")):
            case_dir = aggregate.parent
            payload = json.loads(aggregate.read_text(encoding="utf-8"))
            row = {
                "warmup_requests": budget,
                "case": aggregate.parents[1].name,
                "variant": aggregate.parent.name,
                **{metric: float(payload[metric]) for metric in METRICS},
                "token_trace_deterministic": bool(
                    payload.get("token_trace_deterministic", False)),
                **_load_run_stability(case_dir),
                **_load_process_wall(case_dir),
                **_load_calibration(case_dir),
            }
            rows.append(row)
    if not rows:
        raise RuntimeError(f"no warmup results found under {args.input_dir}")
    zero_by_case = {
        (row["case"], row["variant"]): row for row in rows
        if row["warmup_requests"] == 0
    }
    for row in rows:
        zero = zero_by_case.get((row["case"], row["variant"]))
        for metric in METRICS:
            baseline = float(zero[metric]) if zero is not None else 0.0
            row[f"{metric}_vs_zero_pct"] = (
                (float(row[metric]) / baseline - 1.0) * 100.0
                if baseline > 0.0 else 0.0)
        row["warmup_incremental_wall_s"] = (
            float(row["process_wall_s_median"]) -
            float(zero["process_wall_s_median"]) if zero is not None else 0.0)
    artifact = {"schema_version": 1, "rows": rows}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(artifact, indent=2) + "\n",
                                encoding="utf-8")
    with args.output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"rows": len(rows), "budgets": sorted({
        row["warmup_requests"] for row in rows
    }), "cases": sorted({row["case"] for row in rows})}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
