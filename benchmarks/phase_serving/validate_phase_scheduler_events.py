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
"""Validate unified decision/dispatch/completion records from a phase-server log."""

from __future__ import annotations

import argparse
import collections
import csv
import gzip
import json
import sys
from pathlib import Path
from typing import Any

RECORD_PREFIX = "PHASE_SCHEDULER_EVENT\t"
EVENT_KINDS = {"decision", "dispatch", "completion"}
PHASES = {"encoder", "prefill", "decode", "copy"}
ACTION_DIRECTIONS = {
    "encoder_to_prefill",
    "prefill_to_encoder",
    "encoder_to_decode",
    "decode_to_encoder",
    "prefill_to_decode",
    "decode_to_prefill",
}
START_SKEW_BUCKETS = {-1, 0, 25, 50, 75, 90, 100}
DISPATCH_MODES = {"none", "single", "co_launch", "residual_augmentation"}
FIDELITY_REASONS = {
    "none", "missing_decision", "action_id_mismatch", "outstanding_mismatch"
}
PHASE_IDS = {"none": 0, "encoder": 1, "prefill": 2, "decode": 3, "copy": 4}
STATUS_IDS = {"submitted": 0, "running": 1, "completion_ready": 2}


def _snapshot_signature(event: dict[str, Any]) -> int:
    """Recompute the cross-run FNV-1a decision identity emitted by C++."""
    result = 14695981039346656037

    def add(value: int) -> None:
        nonlocal result
        result ^= value & ((1 << 64) - 1)
        result = (result * 1099511628211) & ((1 << 64) - 1)

    def add_ids(field: str) -> None:
        values = event.get(field, [])
        add(len(values))
        for value in values:
            add(int(value))

    def add_counts(field: str) -> None:
        values = event.get(field, [])
        add(len(values))
        for value in values:
            add(int(value))

    ready = event.get("ready", {})
    add(int(event.get("outstanding_before_mask", 0)))
    for field in ("encoder_rows", "prefill_rows", "prefill_tokens",
                  "decode_rows", "decode_context_tokens"):
        add(int(ready.get(field, 0)))
    add_ids("ready_encoder_request_ids")
    add_ids("ready_prefill_request_ids")
    add_counts("ready_prefill_token_counts")
    add_ids("ready_decode_request_ids")
    add_counts("ready_decode_context_lengths")
    add(int(event.get("page_pool_allocated_bundles", 0)))
    add(int(event.get("page_reservation_guaranteed_bundles", 0)))
    add(int(event.get("vision_payload_bytes", 0)))
    inflight = sorted(event.get("inflight", []),
                      key=lambda work: PHASE_IDS[str(work["phase"])])
    add(len(inflight))
    for work in inflight:
        add(PHASE_IDS[str(work["phase"])])
        add(STATUS_IDS[str(work["status"])])
        cohort = work.get("cohort", {})
        for field in ("encoder_rows", "prefill_rows", "prefill_tokens",
                      "decode_rows", "decode_context_tokens"):
            add(int(cohort.get(field, 0)))
        values = work.get("request_ids", [])
        add(len(values))
        for value in values:
            add(int(value))
    return result


def _load_events(paths: list[Path]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for path in paths:
        opener = gzip.open if path.suffix == ".gz" else Path.open
        kwargs = {"mode": "rt"} if path.suffix == ".gz" else {}
        with opener(path, encoding="utf-8", errors="replace",
                    **kwargs) as source:
            for line_number, line in enumerate(source, start=1):
                if not line.startswith(RECORD_PREFIX):
                    continue
                event = json.loads(line.removeprefix(RECORD_PREFIX))
                if not isinstance(event, dict):
                    raise ValueError(
                        f"{path}:{line_number}: event must be a JSON object")
                event["_source"] = f"{path}:{line_number}"
                event["_source_path"] = str(path)
                events.append(event)
    return events


def _require(event: dict[str, Any], fields: set[str],
             errors: list[str]) -> None:
    missing = fields - event.keys()
    if missing:
        errors.append(f"{event['_source']}: missing fields {sorted(missing)}")


def _validate_request_ids(owner: dict[str, Any], source: str,
                          errors: list[str]) -> None:
    request_ids = owner.get("request_ids")
    if not isinstance(request_ids, list):
        errors.append(f"{source}: request_ids must be an array")
        return
    if any(not isinstance(request_id, int) or request_id < 0
           for request_id in request_ids):
        errors.append(
            f"{source}: request_ids must contain non-negative integers")
    if len(request_ids) != len(set(request_ids)):
        errors.append(f"{source}: request_ids must be unique")


def _validate_phase_cohort_size(event: dict[str, Any],
                                errors: list[str]) -> None:
    phase = event.get("phase")
    field_by_phase = {
        "encoder": "encoder_rows",
        "prefill": "prefill_rows",
        "decode": "decode_rows",
    }
    field = field_by_phase.get(phase)
    cohort = event.get("cohort")
    request_ids = event.get("request_ids")
    if field is None or not isinstance(cohort, dict) or not isinstance(
            request_ids, list):
        return
    if int(cohort.get(field, -1)) != len(request_ids):
        errors.append(
            f"{event['_source']}: {phase} cohort size does not match request_ids"
        )


def _validate(
    events: list[dict[str, Any]],
    require_fidelity: bool,
    require_gpu_intervals: bool,
    require_residual_contract: bool = False,
    allow_compact_decisions: bool = False,
    activity_intervals: list[Path] | None = None
) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    decisions: dict[tuple[str, str, int], dict[str, Any]] = {}
    dispatches: dict[tuple[str, str, int, str], dict[str, Any]] = {}
    completions: dict[tuple[str, str, int, str], dict[str, Any]] = {}
    event_ids: dict[tuple[str, str], list[int]] = collections.defaultdict(list)
    kind_counts: collections.Counter[str] = collections.Counter()
    phase_counts: collections.Counter[str] = collections.Counter()
    direction_counts: collections.Counter[str] = collections.Counter()
    dispatch_mode_counts: collections.Counter[str] = collections.Counter()
    fidelity_reason_counts: collections.Counter[str] = collections.Counter()
    fidelity_failures = 0
    gpu_intervals = 0

    common = {
        "schema_version", "event_kind", "event_id", "run_id",
        "host_monotonic_ns"
    }
    for event in events:
        _require(event, common, errors)
        if errors and not common.issubset(event):
            continue
        if event["schema_version"] != 1:
            errors.append(
                f"{event['_source']}: unsupported schema_version {event['schema_version']}"
            )
        kind = event["event_kind"]
        if kind not in EVENT_KINDS:
            errors.append(f"{event['_source']}: unknown event_kind {kind!r}")
            continue
        source_path = str(event.get("_source_path", ""))
        run_id = str(event["run_id"])
        event_ids[(source_path, run_id)].append(int(event["event_id"]))
        kind_counts[kind] += 1
        incremental_action_id = event.get("incremental_action_id")
        if incremental_action_id is not None:
            _require(
                event,
                {"requested_action_direction", "requested_start_skew_percent"},
                errors)
            if not isinstance(incremental_action_id,
                              int) or incremental_action_id <= 0:
                errors.append(
                    f"{event['_source']}: incremental_action_id must be a positive integer"
                )
            if event.get("requested_action_direction"
                         ) not in ACTION_DIRECTIONS | {
                             "idle_launch", "none"
                         }:
                errors.append(
                    f"{event['_source']}: invalid requested_action_direction")
            if event.get(
                    "requested_start_skew_percent") not in START_SKEW_BUCKETS:
                errors.append(
                    f"{event['_source']}: invalid requested_start_skew_percent"
                )

        for mask_name in ("outstanding_before_mask",
                          "planned_outstanding_mask",
                          "observed_outstanding_mask"):
            if mask_name not in event:
                continue
            mask = event[mask_name]
            if not isinstance(mask, int) or mask < 0 or mask & ~0x7:
                errors.append(
                    f"{event['_source']}: {mask_name} is not a valid E/P/D mask"
                )
            elif mask.bit_count() > 2:
                errors.append(
                    f"{event['_source']}: {mask_name} illegally contains E+P+D"
                )

        if kind == "decision":
            _require(
                event, {
                    "decision_id", "snapshot_id", "plan_id", "candidates",
                    "request_ids", "selected_action_id"
                }, errors)
            _validate_request_ids(event, event["_source"], errors)
            for index, inflight in enumerate(event.get("inflight", [])):
                _validate_request_ids(
                    inflight, f"{event['_source']}: inflight[{index}]", errors)
            for index, candidate in enumerate(event.get("candidates", [])):
                _validate_request_ids(
                    candidate, f"{event['_source']}: candidates[{index}]",
                    errors)
            if "snapshot_signature" in event:
                for field in ("ready_encoder_request_ids",
                              "ready_prefill_request_ids",
                              "ready_decode_request_ids"):
                    values = event.get(field)
                    if not isinstance(values, list) or any(
                            not isinstance(request_id, int) or request_id < 0
                            for request_id in values):
                        errors.append(
                            f"{event['_source']}: {field} must contain non-negative integers"
                        )
                    elif len(values) != len(set(values)):
                        errors.append(
                            f"{event['_source']}: {field} must be unique")
                for ids_field, counts_field in (
                    ("ready_prefill_request_ids",
                     "ready_prefill_token_counts"),
                    ("ready_decode_request_ids",
                     "ready_decode_context_lengths")):
                    values = event.get(counts_field)
                    if not isinstance(values, list) or any(
                            not isinstance(count, int) or count < 0
                            for count in values):
                        errors.append(
                            f"{event['_source']}: {counts_field} must contain non-negative integers"
                        )
                    elif len(values) != len(event.get(ids_field, [])):
                        errors.append(
                            f"{event['_source']}: {counts_field} must be parallel to {ids_field}"
                        )
                if int(event["snapshot_signature"]) != _snapshot_signature(
                        event):
                    errors.append(
                        f"{event['_source']}: snapshot_signature does not match canonical state"
                    )
            if "plan_id" not in event:
                continue
            key = (source_path, run_id, int(event["plan_id"]))
            if key in decisions:
                errors.append(
                    f"{event['_source']}: duplicate decision for plan {key}")
            decisions[key] = event
            candidates_by_id = {
                candidate.get("action_id"): candidate
                for candidate in event.get("candidates", [])
            }
            selected = candidates_by_id.get(event.get("selected_action_id"))
            if selected is None:
                if not allow_compact_decisions or event.get("candidates"):
                    errors.append(
                        f"{event['_source']}: selected action is absent from candidate frontier"
                    )
                elif event.get("selected_action_id") != event.get("action_id"):
                    errors.append(
                        f"{event['_source']}: compact decision selected action does not match its action identity"
                    )
            elif selected.get("request_ids") != event.get("request_ids"):
                errors.append(
                    f"{event['_source']}: selected candidate request lineage does not match decision"
                )
            continue

        _require(
            event, {
                "execution_id", "plan_id", "action_id", "phase", "request_ids",
                "action_fidelity"
            }, errors)
        _validate_request_ids(event, event["_source"], errors)
        if not {"execution_id", "plan_id", "phase"}.issubset(event):
            continue
        phase = str(event["phase"])
        if phase not in PHASES:
            errors.append(f"{event['_source']}: unknown phase {phase!r}")
        phase_counts[phase] += 1
        _validate_phase_cohort_size(event, errors)
        execution_key = (source_path, run_id, int(event["execution_id"]),
                         phase)
        if not event.get("action_fidelity", False):
            fidelity_failures += 1
        if require_residual_contract:
            _require(
                event, {
                    "dispatch_mode", "incumbent_dispatch_age_us",
                    "observed_start_skew_percent", "action_fidelity_reason"
                }, errors)
            dispatch_mode = str(event.get("dispatch_mode"))
            fidelity_reason = str(event.get("action_fidelity_reason"))
            if dispatch_mode not in DISPATCH_MODES:
                errors.append(
                    f"{event['_source']}: invalid dispatch_mode {dispatch_mode!r}"
                )
            if fidelity_reason not in FIDELITY_REASONS:
                errors.append(
                    f"{event['_source']}: invalid action_fidelity_reason {fidelity_reason!r}"
                )
            if bool(event.get("action_fidelity")) != (
                    fidelity_reason == "none"):
                errors.append(
                    f"{event['_source']}: action_fidelity and its rejection reason disagree"
                )
            if event.get(
                    "observed_start_skew_percent") not in START_SKEW_BUCKETS:
                errors.append(
                    f"{event['_source']}: invalid observed_start_skew_percent")
            dispatch_mode_counts[dispatch_mode] += 1
            fidelity_reason_counts[fidelity_reason] += 1

        if kind == "dispatch":
            _require(
                event, {
                    "action_direction", "outstanding_before_mask",
                    "planned_outstanding_mask", "enqueue_host_ns", "cohort"
                }, errors)
            direction_counts[str(event.get("action_direction"))] += 1
            if require_residual_contract and int(
                    event.get("planned_outstanding_mask", -1)) != int(
                        event.get("observed_outstanding_mask", -2)):
                errors.append(
                    f"{event['_source']}: planned and observed outstanding sets differ"
                )
            if execution_key in dispatches:
                errors.append(
                    f"{event['_source']}: duplicate dispatch for execution {execution_key}"
                )
            dispatches[execution_key] = event
        else:
            _require(
                event, {
                    "observed_outstanding_mask", "gpu_duration_us",
                    "completion_visible_host_ns", "completion_status"
                }, errors)
            if execution_key in completions:
                errors.append(
                    f"{event['_source']}: duplicate completion for execution {execution_key}"
                )
            completions[execution_key] = event
            has_gpu_interval = "gpu_start_us" in event and "gpu_end_us" in event
            gpu_intervals += int(has_gpu_interval)
            if has_gpu_interval:
                start_us = float(event["gpu_start_us"])
                end_us = float(event["gpu_end_us"])
                duration_us = float(event["gpu_duration_us"])
                if start_us < 0.0 or end_us < start_us:
                    errors.append(
                        f"{event['_source']}: invalid common-epoch GPU interval"
                    )
                if abs((end_us - start_us) - duration_us) > 1e-3:
                    errors.append(
                        f"{event['_source']}: GPU duration does not match its interval"
                    )
            if require_gpu_intervals and not has_gpu_interval:
                errors.append(
                    f"{event['_source']}: completion has no common-epoch GPU interval"
                )

    for (source_path, run_id), ids in event_ids.items():
        if ids != sorted(ids) or len(ids) != len(set(ids)):
            errors.append(
                f"run {(source_path, run_id)!r}: event_id sequence is not strictly increasing"
            )

    for event in [*dispatches.values(), *completions.values()]:
        plan_key = (str(event.get("_source_path", "")), str(event["run_id"]),
                    int(event["plan_id"]))
        if plan_key not in decisions:
            errors.append(
                f"{event['_source']}: no decision record for plan {plan_key}")

    missing_completion = sorted(dispatches.keys() - completions.keys())
    missing_dispatch = sorted(completions.keys() - dispatches.keys())
    for execution_key in dispatches.keys() & completions.keys():
        dispatch = dispatches[execution_key]
        completion = completions[execution_key]
        if dispatch.get("request_ids") != completion.get("request_ids"):
            errors.append(
                f"{completion['_source']}: request lineage changed for {execution_key}"
            )
        if dispatch.get("plan_id") != completion.get(
                "plan_id") or dispatch.get("action_id") != completion.get(
                    "action_id"):
            errors.append(
                f"{completion['_source']}: plan/action identity changed for {execution_key}"
            )
        if dispatch.get("incremental_action_id") != completion.get(
                "incremental_action_id"):
            errors.append(
                f"{completion['_source']}: incremental action identity changed for {execution_key}"
            )
        if dispatch.get("requested_action_direction") != completion.get("requested_action_direction") \
                or dispatch.get("requested_start_skew_percent") != completion.get("requested_start_skew_percent"):
            errors.append(
                f"{completion['_source']}: incremental direction/skew changed for {execution_key}"
            )
        if int(dispatch["enqueue_host_ns"]) > int(
                completion["completion_visible_host_ns"]):
            errors.append(
                f"{completion['_source']}: completion precedes enqueue for {execution_key}"
            )
        if "incumbent_execution_id" in dispatch:
            incumbent_key = (str(dispatch.get("_source_path",
                                              "")), str(dispatch["run_id"]),
                             int(dispatch["incumbent_execution_id"]),
                             str(dispatch.get("incumbent_phase")))
            if incumbent_key not in dispatches or incumbent_key not in completions:
                errors.append(
                    f"{dispatch['_source']}: incumbent execution {incumbent_key} is incomplete"
                )
    if missing_completion:
        errors.append(
            f"{len(missing_completion)} executions have dispatch but no completion"
        )
    if missing_dispatch:
        errors.append(
            f"{len(missing_dispatch)} executions have completion but no dispatch"
        )
    if require_fidelity and fidelity_failures:
        errors.append(
            f"{fidelity_failures} dispatch/completion events violate action fidelity"
        )

    activity_dispatches = None
    if activity_intervals:
        activity_dispatches = 0
        dispatch_names = {
            "encoder_engine", "prefill_dispatch", "decode_dispatch",
            "prefill_residual_dispatch", "decode_residual_dispatch"
        }
        for path in activity_intervals:
            with path.open(newline="", encoding="utf-8") as source:
                activity_dispatches += sum(
                    row.get("name") in dispatch_names
                    for row in csv.DictReader(source))
        if activity_dispatches != len(dispatches):
            errors.append(
                f"runtime dispatch count {len(dispatches)} differs from activity dispatch count {activity_dispatches}"
            )

    summary = {
        "events": len(events),
        "event_kinds": dict(sorted(kind_counts.items())),
        "phases": dict(sorted(phase_counts.items())),
        "directions": dict(sorted(direction_counts.items())),
        "dispatch_modes": dict(sorted(dispatch_mode_counts.items())),
        "fidelity_reasons": dict(sorted(fidelity_reason_counts.items())),
        "decisions": len(decisions),
        "executions": len(dispatches),
        "action_fidelity_failures": fidelity_failures,
        "gpu_intervals": gpu_intervals,
        "activity_dispatches": activity_dispatches,
        "dispatch_without_completion": len(missing_completion),
        "completion_without_dispatch": len(missing_dispatch),
    }
    return errors, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", type=Path, nargs="+")
    parser.add_argument("--require-action-fidelity", action="store_true")
    parser.add_argument("--require-gpu-intervals", action="store_true")
    parser.add_argument("--require-residual-contract", action="store_true")
    parser.add_argument(
        "--allow-compact-decisions",
        action="store_true",
        help=
        "accept research telemetry that omits the detailed candidate frontier")
    parser.add_argument("--activity-intervals", type=Path, nargs="*")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    try:
        events = _load_events(args.logs)
        if not events:
            raise ValueError(f"no {RECORD_PREFIX.strip()} records found")
        errors, summary = _validate(events, args.require_action_fidelity,
                                    args.require_gpu_intervals,
                                    args.require_residual_contract,
                                    args.allow_compact_decisions,
                                    args.activity_intervals)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    rendered = json.dumps(summary, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
