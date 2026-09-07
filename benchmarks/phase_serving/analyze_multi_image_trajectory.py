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
"""Correlate multi-image serving metrics with E/P/D execution trajectories."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

TIMELINE_MARKER = "PHASE_TIMELINE\t"


def _run_number(aggregate: Path) -> int:
    return int(aggregate.parents[1].name.removeprefix("run-"))


def _event_path(aggregate: Path) -> Path:
    variant = aggregate.parents[2]
    return variant / "activity" / f"run-{_run_number(aggregate):03d}-events.jsonl"


def _interval_path(aggregate: Path) -> Path:
    variant = aggregate.parents[2]
    return variant / "activity" / f"run-{_run_number(aggregate):03d}-intervals.csv"


def _read_scheduler_events(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    events = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.startswith("PHASE_SCHEDULER_EVENT\t"):
                continue
            events.append(json.loads(line.split("\t", 1)[1]))
    return events


def _read_intervals(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _read_measured_timelines(
        path: Path, request_count: int) -> dict[int, list[dict[str, Any]]]:
    """Return the last complete lifecycle for each measured request ID."""
    records: dict[int, list[dict[str, Any]]] = {}
    if not path.is_file():
        return records
    lifecycles: dict[int, list[list[dict[str, Any]]]] = {}
    current: dict[int, list[dict[str, Any]]] = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.startswith(TIMELINE_MARKER):
                continue
            event = json.loads(line.split("\t", 1)[1])
            request_id = int(event["request_index"])
            if request_id >= request_count:
                continue
            current.setdefault(request_id, []).append(event)
            if event.get("stage") == "completion":
                lifecycles.setdefault(request_id,
                                      []).append(current.pop(request_id))
    for request_id, request_lifecycles in lifecycles.items():
        records[request_id] = request_lifecycles[-1]
    return records


def _cohorts(lifecycles: dict[int, list[dict[str, Any]]],
             stage: str) -> list[dict[str, Any]]:
    events = [
        event for lifecycle in lifecycles.values() for event in lifecycle
        if event.get("stage") == stage
    ]
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for event in events:
        dispatch_index = int(event.get("dispatch_index", 0))
        # Encoder batches have no dispatch sequence but deliberately share one
        # timestamp across all request records in the batch.
        timestamp_key = round(float(event["timestamp_us"])) \
            if dispatch_index == 0 else 0
        grouped.setdefault((dispatch_index, timestamp_key), []).append(event)
    cohorts = []
    for (dispatch_index, _), members in grouped.items():
        members.sort(key=lambda event: int(event["request_index"]))
        cohorts.append({
            "timestamp_us":
            min(float(event["timestamp_us"]) for event in members),
            "dispatch_index":
            dispatch_index,
            "batch_size":
            max(int(event.get("batch_size", 0)) for event in members),
            "request_ids": [int(event["request_index"]) for event in members],
        })
    return sorted(cohorts,
                  key=lambda cohort:
                  (cohort["timestamp_us"], cohort["dispatch_index"]))


def _first_stage(lifecycle: list[dict[str, Any]],
                 stage: str) -> dict[str, Any] | None:
    return next((event for event in lifecycle if event.get("stage") == stage),
                None)


def _analyze_transition_lineage(
        lifecycles: dict[int, list[dict[str, Any]]]) -> dict[str, Any]:
    if not lifecycles:
        return {}
    encoder_cohorts = _cohorts(lifecycles, "encoder_start")
    prefill_cohorts = _cohorts(lifecycles, "prefill_start")
    decode_cohorts = _cohorts(lifecycles, "decode_start")
    first_tokens = {
        request_id: _first_stage(lifecycle, "first_token")
        for request_id, lifecycle in lifecycles.items()
    }
    first_tokens = {
        request_id: event
        for request_id, event in first_tokens.items() if event is not None
    }
    if not first_tokens or not decode_cohorts:
        return {
            "encoder_cohorts": encoder_cohorts,
            "prefill_cohorts": prefill_cohorts,
            "decode_cohorts": decode_cohorts,
        }
    first_ready_us = min(
        float(event["timestamp_us"]) for event in first_tokens.values())
    first_decode = decode_cohorts[0]
    first_decode_us = float(first_decode["timestamp_us"])
    ready_before_first_decode = sorted(
        request_id for request_id, event in first_tokens.items()
        if float(event["timestamp_us"]) <= first_decode_us)
    singleton_prefix = 0
    for cohort in decode_cohorts:
        if int(cohort["batch_size"]) != 1:
            break
        singleton_prefix += 1
    full_cohort = next((cohort for cohort in decode_cohorts
                        if int(cohort["batch_size"]) >= len(lifecycles)), None)
    return {
        "encoder_cohorts":
        encoder_cohorts,
        "prefill_cohorts":
        prefill_cohorts,
        "decode_cohorts":
        decode_cohorts,
        "first_decode_ready_request_id":
        min(first_tokens,
            key=lambda request_id: float(first_tokens[request_id][
                "timestamp_us"])),
        "first_decode_ready_to_start_ms":
        max(0.0, (first_decode_us - first_ready_us) / 1000.0),
        "ready_request_ids_before_first_decode":
        ready_before_first_decode,
        "ready_rows_before_first_decode":
        len(ready_before_first_decode),
        "first_decode_batch_size":
        int(first_decode["batch_size"]),
        "first_decode_request_ids":
        first_decode["request_ids"],
        "decode_singleton_prefix_dispatches":
        singleton_prefix,
        "first_ready_to_full_decode_cohort_ms":
        ((float(full_cohort["timestamp_us"]) - first_ready_us) /
         1000.0) if full_cohort is not None else None,
    }


def analyze_run(aggregate: Path, coherent_decode_max: int,
                fragmented_decode_min: int) -> dict[str, Any]:
    metrics = json.loads(aggregate.read_text(encoding="utf-8"))
    intervals = _read_intervals(_interval_path(aggregate))
    events = _read_scheduler_events(_event_path(aggregate))
    execution_names = {
        "encoder": {"encoder_engine"},
        "prefill": {"prefill_dispatch"},
        "decode": {"decode_dispatch"},
        "copy": {"copy", "copy_dispatch"},
    }
    phase_intervals = {
        phase: [
            row for row in intervals
            if row.get("kind") == phase and row.get("name") in names
        ]
        for phase, names in execution_names.items()
    }
    phase_duration_ms = {
        phase: sum(float(row["duration_ms"]) for row in rows)
        for phase, rows in phase_intervals.items()
    }
    decisions = [
        event for event in events if event.get("event_kind") == "decision"
    ]
    request_count = int(
        sum(
            float(value.get("requests", 0.0))
            for value in metrics.get("by_request_class", {}).values()))
    lineage = _analyze_transition_lineage(
        _read_measured_timelines(_event_path(aggregate), request_count))
    # Calibration and measured traces allocate request IDs from separate
    # zero-based epochs. When telemetry spans both, the last calibration ID
    # (>= measured request count) is an unambiguous boundary. Keep only the
    # measured epoch so warmup decisions cannot masquerade as policy effects.
    if request_count > 0:
        last_calibration = -1
        id_fields = ("request_ids", "ready_encoder_request_ids",
                     "ready_prefill_request_ids", "ready_decode_request_ids")
        for index, event in enumerate(decisions):
            ids = [
                int(request_id) for field in id_fields
                for request_id in event.get(field, [])
            ]
            if ids and max(ids) >= request_count:
                last_calibration = index
        if last_calibration >= 0:
            decisions = decisions[last_calibration + 1:]
    action_counts = Counter(
        str(event.get("action_kind", "unknown")) for event in decisions)
    fallback_disagreements = sum(
        int(event.get("active_h1_selected_action_id", 0)) > 0
        and int(event.get("non_contextual_selected_action_id", 0)) > 0
        and event.get("active_h1_selected_action_id") != event.get(
            "non_contextual_selected_action_id") for event in decisions)
    fallback_pairs: Counter[str] = Counter()
    for event in decisions:
        learned_id = event.get("active_h1_selected_action_id")
        fallback_id = event.get("non_contextual_selected_action_id")
        if not learned_id or not fallback_id or learned_id == fallback_id:
            continue
        action_by_id = {
            candidate.get("action_id"): candidate.get("action_kind", "unknown")
            for candidate in event.get("candidates", [])
        }
        fallback_pairs[f"{action_by_id.get(learned_id, 'unknown')}->"
                       f"{action_by_id.get(fallback_id, 'unknown')}"] += 1
    scalar_known = 0
    direction_observations: Counter[str] = Counter()
    for event in decisions:
        selected = event.get("selected_action_id")
        for candidate in event.get("candidates", []):
            if candidate.get("action_id") != selected:
                continue
            if candidate.get("scalar_decision_cost_known", False):
                scalar_known += 1
            direction = str(candidate.get("contextual_direction", ""))
            observations = int(
                candidate.get("contextual_direction_observations", 0))
            if direction:
                direction_observations[direction] = max(
                    direction_observations[direction], observations)
            break
    decode_dispatches = len(phase_intervals["decode"])
    if request_count != 5:
        trajectory_family = "scaled"
    elif decode_dispatches <= coherent_decode_max:
        trajectory_family = "coherent"
    elif decode_dispatches >= fragmented_decode_min:
        trajectory_family = "fragmented"
    else:
        trajectory_family = "transitional"
    starts = [float(row["start_ms"]) for row in intervals]
    return {
        "root":
        str(aggregate.parents[2]),
        "run":
        _run_number(aggregate),
        "request_count":
        request_count,
        "trajectory_family":
        trajectory_family,
        "generated_token_s":
        metrics.get("generated_token_s_median"),
        "ttft_mean_ms":
        metrics.get("ttft_mean_of_run_means_ms"),
        "ttft_p95_ms":
        metrics.get("ttft_p95_median_ms"),
        "tpot_mean_ms":
        metrics.get("tpot_mean_of_run_means_ms"),
        "tpot_p95_ms":
        metrics.get("tpot_p95_median_ms"),
        "e2e_mean_ms":
        metrics.get("e2e_mean_of_run_means_ms"),
        "e2e_p95_ms":
        metrics.get("e2e_p95_median_ms"),
        "startup_to_first_gpu_ms":
        min(starts) if starts else None,
        "encoder_dispatches":
        len(phase_intervals["encoder"]),
        "prefill_dispatches":
        len(phase_intervals["prefill"]),
        "decode_dispatches":
        decode_dispatches,
        "copy_dispatches":
        len(phase_intervals["copy"]),
        "encoder_gpu_ms":
        phase_duration_ms["encoder"],
        "prefill_gpu_ms":
        phase_duration_ms["prefill"],
        "decode_gpu_ms":
        phase_duration_ms["decode"],
        "copy_gpu_ms":
        phase_duration_ms["copy"],
        "scheduler_decisions":
        len(decisions),
        "selected_scalar_known":
        scalar_known,
        "contextual_fallback_disagreements":
        fallback_disagreements,
        "successor_guard_evaluations":
        sum(
            bool(event.get("contextual_successor_guard_evaluated", False))
            for event in decisions),
        "successor_guard_overrides":
        sum(
            bool(event.get("contextual_successor_guard_applied", False))
            for event in decisions),
        "action_counts":
        dict(sorted(action_counts.items())),
        "learned_to_fallback_pairs":
        dict(sorted(fallback_pairs.items())),
        "direction_observations":
        dict(sorted(direction_observations.items())),
        "transition_lineage":
        lineage,
        "event_path":
        str(_event_path(aggregate)),
    }


def summarize(runs: list[dict[str, Any]]) -> dict[str, Any]:

    def median_present(values: list[float | int]) -> float | None:
        return statistics.median(values) if values else None

    families: dict[str, dict[str, Any]] = {}
    for family in sorted({str(run["trajectory_family"]) for run in runs}):
        members = [run for run in runs if run["trajectory_family"] == family]
        families[family] = {
            "runs":
            len(members),
            "median_generated_token_s":
            statistics.median(
                float(run["generated_token_s"]) for run in members),
            "median_decode_dispatches":
            statistics.median(
                int(run["decode_dispatches"]) for run in members),
            "median_decode_gpu_ms":
            statistics.median(float(run["decode_gpu_ms"]) for run in members),
            "median_ttft_mean_ms":
            statistics.median(float(run["ttft_mean_ms"]) for run in members),
            "median_e2e_mean_ms":
            statistics.median(float(run["e2e_mean_ms"]) for run in members),
            "median_first_decode_ready_to_start_ms":
            median_present([
                float(run["transition_lineage"]
                      ["first_decode_ready_to_start_ms"]) for run in members
                if run["transition_lineage"].get(
                    "first_decode_ready_to_start_ms") is not None
            ]),
            "median_ready_rows_before_first_decode":
            median_present([
                int(run["transition_lineage"]
                    ["ready_rows_before_first_decode"]) for run in members
                if run["transition_lineage"].get(
                    "ready_rows_before_first_decode") is not None
            ]),
            "median_decode_singleton_prefix_dispatches":
            median_present([
                int(run["transition_lineage"]
                    ["decode_singleton_prefix_dispatches"]) for run in members
                if run["transition_lineage"].get(
                    "decode_singleton_prefix_dispatches") is not None
            ]),
        }
    return {"runs": runs, "trajectory_families": families}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--coherent-decode-max", type=int, default=40)
    parser.add_argument("--fragmented-decode-min", type=int, default=50)
    args = parser.parse_args()
    if args.coherent_decode_max >= args.fragmented_decode_min:
        parser.error("coherent-decode-max must be below fragmented-decode-min")
    aggregate_paths = sorted({
        aggregate
        for root in args.roots
        for aggregate in root.glob("**/run-*/client/aggregate.json")
    })
    if not aggregate_paths:
        parser.error("no run aggregate files found")
    runs = [
        analyze_run(path, args.coherent_decode_max, args.fragmented_decode_min)
        for path in aggregate_paths
    ]
    result = summarize(runs)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2) + "\n",
                                    encoding="utf-8")
    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        scalar_keys = [
            key for key, value in runs[0].items()
            if not isinstance(value, (dict, list))
        ]
        with args.output_csv.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=scalar_keys)
            writer.writeheader()
            writer.writerows({key: run[key]
                              for key in scalar_keys} for run in runs)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
