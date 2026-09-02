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
"""Build an actual-GPU-offset completion-vector artifact for M2 injection runs."""

from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

RECORD_PREFIX = "PHASE_SCHEDULER_EVENT\t"
DIRECTIONS = {
    "prefill_to_decode",
    "decode_to_prefill",
    "encoder_to_decode",
    "decode_to_encoder",
    "encoder_to_prefill",
    "prefill_to_encoder",
}
TARGET_BUCKETS = (0.0, 0.25, 0.5, 0.75, 0.9)


def actual_direction(incumbent_phase: str, newcomer_phase: str) -> str:
    """Return the direction implied by measured common-epoch GPU start order."""
    direction = f"{incumbent_phase}_to_{newcomer_phase}"
    if direction not in DIRECTIONS:
        raise ValueError(f"unsupported actual phase direction {direction}")
    return direction


def percentile(values: list[float], fraction: float) -> float:
    """Return a linearly interpolated percentile for a non-empty sample."""
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position -
                                                                 lower)


def distribution(values: list[float]) -> dict[str, float | int]:
    """Summarize one measured scalar and expose a robust uncertainty span."""
    if not values:
        return {"count": 0}
    median = statistics.median(values)
    p95 = percentile(values, 0.95)
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": median,
        "p95": p95,
        "uncertainty_p95_minus_median": p95 - median,
        "minimum": min(values),
        "maximum": max(values),
    }


def load_events(paths: list[Path]) -> list[dict[str, Any]]:
    """Load unified phase events and retain their source location."""
    events: list[dict[str, Any]] = []
    for path in paths:
        with path.open(encoding="utf-8", errors="replace") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.startswith(RECORD_PREFIX):
                    continue
                event = json.loads(line.removeprefix(RECORD_PREFIX))
                event["_source_log"] = str(path)
                event["_source"] = f"{path}:{line_number}"
                events.append(event)
    return events


def nearest_bucket(fraction: float) -> float:
    """Classify a requested injection by its measured CUDA start offset."""
    return min(TARGET_BUCKETS, key=lambda bucket: abs(bucket - fraction))


def build_samples(
        events: list[dict[str, Any]], bucket_tolerance: float,
        confidence_beta: float) -> tuple[list[dict[str, Any]], list[str]]:
    """Join newcomer dispatches with incumbent/newcomer completion intervals."""
    errors: list[str] = []
    completions: dict[tuple[str, str, str, int], dict[str, Any]] = {}
    decisions: dict[tuple[str, str, int], dict[str, Any]] = {}
    for event in events:
        if event.get("event_kind") == "decision":
            decisions[(str(event.get("_source_log", "")),
                       str(event.get("run_id")), int(event.get("plan_id",
                                                               0)))] = event
        if event.get("event_kind") != "completion":
            continue
        key = (str(event.get("_source_log", "")), str(event.get("run_id")),
               str(event.get("phase")), int(event.get("execution_id", 0)))
        completions[key] = event

    samples: list[dict[str, Any]] = []
    selected_injections: set[tuple[str, str, str]] = set()
    for dispatch in events:
        direction = str(dispatch.get("action_direction"))
        if dispatch.get(
                "event_kind") != "dispatch" or direction not in DIRECTIONS:
            continue
        if "injection_target_fraction" not in dispatch:
            continue
        run_id = str(dispatch.get("run_id"))
        source_log = str(dispatch.get("_source_log", ""))
        incumbent_phase = str(dispatch.get("incumbent_phase"))
        newcomer_phase = str(dispatch.get("phase"))
        incumbent_id = int(dispatch.get("incumbent_execution_id", 0))
        newcomer_id = int(dispatch.get("execution_id", 0))
        incumbent = completions.get(
            (source_log, run_id, incumbent_phase, incumbent_id))
        newcomer = completions.get(
            (source_log, run_id, newcomer_phase, newcomer_id))
        if incumbent is None or newcomer is None:
            errors.append(
                f"{dispatch['_source']}: directional dispatch has no paired completion vector"
            )
            continue
        required = {"gpu_start_us", "gpu_end_us", "gpu_duration_us"}
        if not required.issubset(incumbent) or not required.issubset(newcomer):
            errors.append(
                f"{dispatch['_source']}: paired completion vector has no common-epoch GPU interval"
            )
            continue
        if not dispatch.get("action_fidelity", False) or not incumbent.get("action_fidelity", False) \
                or not newcomer.get("action_fidelity", False):
            errors.append(
                f"{dispatch['_source']}: paired completion vector violates action fidelity"
            )
            continue

        incumbent_start = float(incumbent["gpu_start_us"])
        incumbent_end = float(incumbent["gpu_end_us"])
        newcomer_start = float(newcomer["gpu_start_us"])
        newcomer_end = float(newcomer["gpu_end_us"])
        incumbent_reference = float(
            dispatch["injection_incumbent_reference_us"])
        newcomer_reference = float(dispatch["injection_newcomer_reference_us"])
        target = float(dispatch["injection_target_fraction"])
        requested_direction = str(
            dispatch.get("injection_requested_direction", direction))
        if requested_direction not in DIRECTIONS:
            errors.append(
                f"{dispatch['_source']}: invalid requested injection direction {requested_direction}"
            )
            continue
        injection_identity = (source_log, run_id, requested_direction)
        if injection_identity in selected_injections:
            continue
        selected_injections.add(injection_identity)
        if newcomer_start < incumbent_start:
            incumbent, newcomer = newcomer, incumbent
            incumbent_phase, newcomer_phase = newcomer_phase, incumbent_phase
            incumbent_id, newcomer_id = newcomer_id, incumbent_id
            incumbent_start, newcomer_start = newcomer_start, incumbent_start
            incumbent_end, newcomer_end = newcomer_end, incumbent_end
            incumbent_reference, newcomer_reference = newcomer_reference, incumbent_reference
        direction = actual_direction(incumbent_phase, newcomer_phase)
        actual_offset = newcomer_start - incumbent_start
        actual_fraction = actual_offset / incumbent_reference
        bucket = nearest_bucket(actual_fraction)
        incumbent_duration = incumbent_end - incumbent_start
        newcomer_duration = newcomer_end - newcomer_start
        overlap = max(
            0.0,
            min(incumbent_end, newcomer_end) -
            max(incumbent_start, newcomer_start))
        makespan = max(incumbent_end, newcomer_end) - incumbent_start
        incumbent_completion = max(0.0, incumbent_end - newcomer_start)
        newcomer_completion = max(0.0, newcomer_end - newcomer_start)
        action_makespan = max(incumbent_completion, newcomer_completion)
        remaining_incumbent_reference = max(
            0.0, incumbent_reference - actual_offset)
        serial_equivalent = remaining_incumbent_reference + newcomer_reference
        compression = (serial_equivalent - action_makespan) / max(
            serial_equivalent, 1.0)
        decision = decisions.get(
            (source_log, run_id, int(dispatch.get("plan_id", 0))), {})
        selected = next((candidate
                         for candidate in decision.get("candidates", [])
                         if int(candidate.get("action_id", 0)) == int(
                             dispatch.get("action_id", 0))), {})
        incumbent_prediction = float(
            selected.get("contextual_incumbent_mean_us", 0.0))
        newcomer_prediction = float(
            selected.get("contextual_newcomer_mean_us", 0.0))
        incumbent_uncertainty = float(
            selected.get("contextual_incumbent_uncertainty_us", 0.0))
        newcomer_uncertainty = float(
            selected.get("contextual_newcomer_uncertainty_us", 0.0))
        prediction_ready = bool(
            selected.get("contextual_completion_ready", False))
        inflight_overlap = overlap > 0.0
        realization = "overlap" if inflight_overlap else "serial_realization"
        samples.append({
            "run_id":
            run_id,
            "source_log":
            source_log,
            "direction":
            direction,
            "requested_direction":
            requested_direction,
            "target_fraction":
            target,
            "actual_offset_us":
            actual_offset,
            "actual_fraction":
            actual_fraction,
            "actual_bucket":
            bucket,
            "actual_start_skew_bucket":
            f"offset_{round(bucket * 100)}"
            if inflight_overlap else "serial_realization",
            "actual_bucket_error":
            abs(actual_fraction - bucket),
            "requested_target_error":
            abs(actual_fraction - target),
            "inflight_overlap":
            inflight_overlap,
            "realization":
            realization,
            "dispatch_mode":
            str(dispatch.get("dispatch_mode", "unknown")),
            "accepted_bucket":
            inflight_overlap
            and abs(actual_fraction - bucket) <= bucket_tolerance,
            "requested_delay_us":
            int(dispatch.get("requested_injection_delay_us", 0)),
            "incumbent_phase":
            incumbent_phase,
            "incumbent_milestone":
            "tpot" if incumbent_phase == "decode" else "ttft",
            "incumbent_execution_id":
            incumbent_id,
            "incumbent_reference_us":
            incumbent_reference,
            "incumbent_duration_us":
            incumbent_duration,
            "incumbent_slowdown_us":
            incumbent_duration - incumbent_reference,
            "newcomer_phase":
            newcomer_phase,
            "newcomer_milestone":
            "tpot" if newcomer_phase == "decode" else "ttft",
            "newcomer_execution_id":
            newcomer_id,
            "newcomer_reference_us":
            newcomer_reference,
            "newcomer_duration_us":
            newcomer_duration,
            "newcomer_slowdown_us":
            newcomer_duration - newcomer_reference,
            "overlap_us":
            overlap,
            "makespan_us":
            makespan,
            "incumbent_completion_from_injection_us":
            incumbent_completion,
            "newcomer_completion_from_injection_us":
            newcomer_completion,
            "action_makespan_from_injection_us":
            action_makespan,
            "serial_equivalent_from_injection_us":
            serial_equivalent,
            "serial_equivalent_compression":
            compression,
            "completion_prediction_ready":
            prediction_ready,
            "predicted_incumbent_completion_us":
            incumbent_prediction,
            "predicted_newcomer_completion_us":
            newcomer_prediction,
            "incumbent_prediction_absolute_error_us":
            abs(incumbent_prediction -
                incumbent_completion) if prediction_ready else math.nan,
            "newcomer_prediction_absolute_error_us":
            abs(newcomer_prediction -
                newcomer_completion) if prediction_ready else math.nan,
            "incumbent_interval_covered":
            prediction_ready
            and abs(incumbent_prediction - incumbent_completion)
            <= confidence_beta * incumbent_uncertainty,
            "newcomer_interval_covered":
            prediction_ready and abs(newcomer_prediction - newcomer_completion)
            <= confidence_beta * newcomer_uncertainty,
            "completion_visible_span_us":
            abs(
                int(newcomer["completion_visible_host_ns"]) -
                int(incumbent["completion_visible_host_ns"])) / 1000.0,
            "action_fidelity":
            True,
        })
    return samples, errors


def summarize(samples: list[dict[str, Any]],
              material_effect_ratio: float) -> dict[str, Any]:
    """Aggregate accepted actual-offset samples and evaluate Gate A by direction."""
    grouped: dict[tuple[str, float],
                  list[dict[str, Any]]] = collections.defaultdict(list)
    for sample in samples:
        if sample["accepted_bucket"]:
            grouped[(sample["direction"],
                     sample["actual_bucket"])].append(sample)

    cells: list[dict[str, Any]] = []
    for (direction, bucket), group in sorted(grouped.items()):
        cell: dict[str, Any] = {
            "direction": direction,
            "actual_bucket": bucket,
            "samples": len(group)
        }
        for field in ("actual_fraction", "incumbent_duration_us",
                      "incumbent_slowdown_us", "newcomer_duration_us",
                      "newcomer_slowdown_us", "overlap_us", "makespan_us",
                      "incumbent_completion_from_injection_us",
                      "newcomer_completion_from_injection_us",
                      "action_makespan_from_injection_us",
                      "serial_equivalent_compression",
                      "completion_visible_span_us"):
            cell[field] = distribution(
                [float(sample[field]) for sample in group])
        ready = [
            sample for sample in group if sample["completion_prediction_ready"]
        ]
        cell["prediction"] = {
            "ready_samples":
            len(ready),
            "incumbent_absolute_error_us":
            distribution([
                float(sample["incumbent_prediction_absolute_error_us"])
                for sample in ready
            ]),
            "newcomer_absolute_error_us":
            distribution([
                float(sample["newcomer_prediction_absolute_error_us"])
                for sample in ready
            ]),
            "incumbent_interval_coverage":
            sum(
                bool(sample["incumbent_interval_covered"])
                for sample in ready) / len(ready) if ready else None,
            "newcomer_interval_coverage":
            sum(bool(sample["newcomer_interval_covered"])
                for sample in ready) / len(ready) if ready else None,
        }
        cells.append(cell)

    direction_gates: list[dict[str, Any]] = []
    for direction in sorted(DIRECTIONS):
        direction_cells = [
            cell for cell in cells if cell["direction"] == direction
        ]
        incumbent_reference = [
            float(sample["incumbent_reference_us"]) for sample in samples
            if sample["direction"] == direction
        ]
        newcomer_reference = [
            float(sample["newcomer_reference_us"]) for sample in samples
            if sample["direction"] == direction
        ]
        ranges: dict[str, float] = {}
        for field in ("incumbent_duration_us", "newcomer_duration_us",
                      "makespan_us"):
            medians = [
                float(cell[field]["median"]) for cell in direction_cells
                if cell[field]["count"]
            ]
            ranges[field] = max(medians) - min(medians) if len(
                medians) >= 2 else 0.0
        reference = max(
            statistics.median(incumbent_reference)
            if incumbent_reference else 0.0,
            statistics.median(newcomer_reference)
            if newcomer_reference else 0.0, 1.0)
        maximum_ratio = max(ranges.values(), default=0.0) / reference
        direction_gates.append({
            "direction":
            direction,
            "accepted_buckets":
            len(direction_cells),
            "completion_range_us":
            ranges,
            "maximum_material_effect_ratio":
            maximum_ratio,
            "material_inflight_effect":
            len(direction_cells) >= 2
            and maximum_ratio >= material_effect_ratio,
        })

    return {
        "schema_version":
        1,
        "target_buckets":
        list(TARGET_BUCKETS),
        "samples":
        len(samples),
        "accepted_samples":
        sum(bool(sample["accepted_bucket"]) for sample in samples),
        "cells":
        cells,
        "gate_a": {
            "material_effect_ratio":
            material_effect_ratio,
            "directions":
            direction_gates,
            "passed_directions":
            sum(
                bool(gate["material_inflight_effect"])
                for gate in direction_gates),
        },
    }


def write_csv(path: Path, samples: list[dict[str, Any]]) -> None:
    """Write one flat record per measured completion vector."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(
        samples[0]) if samples else ["run_id", "direction", "target_fraction"]
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(samples)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", type=Path, nargs="+")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--bucket-tolerance", type=float, default=0.13)
    parser.add_argument("--material-effect-ratio", type=float, default=0.03)
    parser.add_argument("--confidence-beta", type=float, default=1.96)
    args = parser.parse_args()
    if (args.bucket_tolerance < 0.0 or args.material_effect_ratio < 0.0
            or args.confidence_beta < 0.0):
        parser.error("tolerances must be non-negative")
    try:
        events = load_events(args.logs)
        samples, errors = build_samples(events, args.bucket_tolerance,
                                        args.confidence_beta)
        if not samples:
            raise ValueError(
                "no directional-injection completion vectors found")
        artifact = summarize(samples, args.material_effect_ratio)
        artifact["sources"] = [str(path) for path in args.logs]
        artifact["validation_errors"] = errors
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(artifact, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        write_csv(args.output_csv, samples)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(json.dumps(artifact["gate_a"], indent=2, sort_keys=True))
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
