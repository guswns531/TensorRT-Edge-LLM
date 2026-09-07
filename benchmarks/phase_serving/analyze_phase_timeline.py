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
"""Attribute request latency to host-side E/P/D critical-path intervals."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

TIMELINE_MARKER = "PHASE_TIMELINE\t"
METRIC_MARKER = "PHASE_METRIC\t"
DEFAULT_MAX_REQUEST_ID = 999_999
COMPONENT_FIELDS = (
    "encoder_queue_ms",
    "encoder_active_ms",
    "encoder_to_prefill_ready_ms",
    "vision_ready_queue_ms",
    "prefill_admission_ms",
    "prefill_initial_queue_ms",
    "prefill_active_ms",
    "prefill_chunk_gap_ms",
    "prefill_sampling_ms",
)
SUMMARY_FIELDS = COMPONENT_FIELDS + (
    "prefill_sampling_submit_to_ready_ms",
    "prefill_sampling_ready_to_collect_ms",
    "prefill_sampling_collect_to_commit_ms",
    "first_token_to_decode_ms",
    "first_decode_active_ms",
    "decode_total_active_ms",
    "decode_inter_dispatch_gap_ms",
    "decode_sampling_submit_to_ready_ms",
    "decode_sampling_ready_to_collect_ms",
    "decode_sampling_collect_to_commit_ms",
    "token_commit_to_decode_ready_ms",
    "decode_ready_queue_ms",
    "decode_tail_ms",
    "token_commit_to_slot_release_ms",
    "backend_ttft_ms",
    "backend_e2e_ms",
    "prefill_total_span_ms",
    "critical_path_residual_ms",
    "e2e_residual_ms",
    "client_ttft_ms",
    "client_tpot_ms",
    "client_e2e_ms",
    "client_dispatch_delay_ms",
    "scheduled_ttft_ms",
    "scheduled_e2e_ms",
    "frontend_to_backend_ttft_ms",
    "backend_to_client_completion_ms",
)
DISPATCH_FIELDS = (
    "prefill_dispatches",
    "decode_dispatches",
    "mean_prefill_batch_size",
    "mean_decode_batch_size",
)


def parse_timeline_logs(
        paths: Iterable[Path],
        max_request_id: int = DEFAULT_MAX_REQUEST_ID) -> list[dict[str, Any]]:
    """Parse timeline records from gateway logs.

    Args:
        paths: Gateway logs containing ``PHASE_TIMELINE`` records.
        max_request_id: Largest production request ID. Shape-warmup IDs start at
            one million and are excluded by default.

    Returns:
        Parsed records with a source label added to each record.
    """
    records = []
    for path in paths:
        with path.open(encoding="utf-8", errors="replace") as source:
            for line_number, line in enumerate(source, start=1):
                marker_offset = line.find(TIMELINE_MARKER)
                if marker_offset < 0:
                    continue
                payload = line[marker_offset + len(TIMELINE_MARKER):].strip()
                try:
                    event = json.loads(payload)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"{path}:{line_number}: invalid timeline JSON"
                    ) from error
                request_id = int(event["request_index"])
                if request_id > max_request_id:
                    continue
                event["request_index"] = request_id
                event["source"] = str(path)
                records.append(event)
    return records


def parse_dispatch_metrics(paths: Iterable[Path]) -> list[dict[str, Any]]:
    """Parse dispatch-level host/GPU realization metrics from gateway logs."""
    records = []
    for path in paths:
        with path.open(encoding="utf-8", errors="replace") as source:
            for line_number, line in enumerate(source, start=1):
                marker_offset = line.find(METRIC_MARKER)
                if marker_offset < 0:
                    continue
                payload = line[marker_offset + len(METRIC_MARKER):].strip()
                try:
                    metric = json.loads(payload)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"{path}:{line_number}: invalid dispatch metric JSON"
                    ) from error
                metric["source"] = str(path)
                records.append(metric)
    return records


def _first(events: dict[str, list[dict[str, Any]]],
           stage: str) -> dict[str, Any] | None:
    values = events.get(stage, [])
    return values[0] if values else None


def _last(events: dict[str, list[dict[str, Any]]],
          stage: str) -> dict[str, Any] | None:
    values = events.get(stage, [])
    return values[-1] if values else None


def _elapsed_ms(start: dict[str, Any] | None,
                end: dict[str, Any] | None) -> float | None:
    if start is None or end is None:
        return None
    elapsed = (float(end["timestamp_us"]) -
               float(start["timestamp_us"])) / 1000.0
    return max(0.0, elapsed)


def _pair_phase_events(
        events: dict[str, list[dict[str, Any]]],
        phase: str) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    starts = events.get(f"{phase}_start", [])
    dones = events.get(f"{phase}_done", [])
    dones_by_dispatch = {
        int(event.get("dispatch_index", 0)): event
        for event in dones
    }
    pairs = []
    used_dispatches = set()
    for start in starts:
        dispatch_index = int(start.get("dispatch_index", 0))
        done = dones_by_dispatch.get(dispatch_index)
        if done is None or dispatch_index in used_dispatches:
            continue
        if float(done["timestamp_us"]) < float(start["timestamp_us"]):
            continue
        pairs.append((start, done))
        used_dispatches.add(dispatch_index)
    return sorted(pairs, key=lambda pair: float(pair[0]["timestamp_us"]))


def _sum_pair_durations_ms(
        pairs: list[tuple[dict[str, Any], dict[str, Any]]]) -> float | None:
    if not pairs:
        return None
    return sum(_elapsed_ms(start, done) or 0.0 for start, done in pairs)


def _pair_correlated_events(
        events: dict[str, list[dict[str, Any]]], start_stage: str,
        end_stage: str) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    ends = {
        int(event.get("dispatch_index", 0)): event
        for event in events.get(end_stage, [])
    }
    pairs = []
    for start in events.get(start_stage, []):
        end = ends.get(int(start.get("dispatch_index", 0)))
        if end is not None and float(end["timestamp_us"]) >= float(
                start["timestamp_us"]):
            pairs.append((start, end))
    return pairs


def _pair_ordered_events(
        events: dict[str, list[dict[str, Any]]], start_stage: str,
        end_stage: str) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    starts = sorted(events.get(start_stage, []),
                    key=lambda event: float(event["timestamp_us"]))
    ends = sorted(events.get(end_stage, []),
                  key=lambda event: float(event["timestamp_us"]))
    pairs = []
    end_index = 0
    for start in starts:
        while end_index < len(ends) and float(
                ends[end_index]["timestamp_us"]) < float(
                    start["timestamp_us"]):
            end_index += 1
        if end_index == len(ends):
            break
        pairs.append((start, ends[end_index]))
        end_index += 1
    return pairs


def _phase_span_ms(
        pairs: list[tuple[dict[str, Any], dict[str, Any]]]) -> float | None:
    if not pairs:
        return None
    return _elapsed_ms(pairs[0][0], pairs[-1][1])


def _split_lifecycles(
        request_records: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    lifecycles = []
    current = []
    for event in request_records:
        current.append(event)
        if event["stage"] == "completion":
            lifecycles.append(current)
            current = []
    if current:
        lifecycles.append(current)
    return lifecycles


def attribute_requests(
        records: Iterable[dict[str, Any]],
        request_count: int | None = None) -> list[dict[str, Any]]:
    """Build one non-overlapping critical-path attribution per request.

    HTTP warmup and measured traffic may reuse the same request IDs. Timeline
    completion events delimit these lifecycles, and only the latest lifecycle is
    attributed. Measured traces use dense IDs starting at zero, so
    ``request_count`` removes warmup-only IDs beyond the production trace.
    """
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if request_count is not None and int(
                record["request_index"]) >= request_count:
            continue
        grouped[(str(record["source"]),
                 int(record["request_index"]))].append(record)

    attributions = []
    for (source, request_id), request_records in sorted(grouped.items()):
        request_records.sort(key=lambda event: (float(event["timestamp_us"]),
                                                str(event["stage"])))
        lifecycles = _split_lifecycles(request_records)
        request_records = lifecycles[-1]
        events: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for event in request_records:
            events[str(event["stage"])].append(event)

        vision_queued = _first(events, "vision_queued")
        server_submit = _first(events, "server_submit")
        arrival = vision_queued if vision_queued is not None else server_submit
        encoder_start = _first(events, "encoder_start")
        encoder_done = _last(events, "encoder_done")
        prefill_ready = _last(events, "prefill_ready")
        server_admit = _first(events, "server_admit")
        first_token = _first(events, "first_token")
        completion = _last(events, "completion")
        slot_released = _last(events, "slot_released")
        final_token_committed = _last(events,
                                      "decode_token_committed") or _last(
                                          events, "prefill_token_committed")
        prefill_pairs = _pair_phase_events(events, "prefill")
        decode_pairs = _pair_phase_events(events, "decode")
        first_prefill_start = prefill_pairs[0][0] if prefill_pairs else None
        final_prefill_done = prefill_pairs[-1][1] if prefill_pairs else None
        first_decode_pair = decode_pairs[0] if decode_pairs else (None, None)
        first_decode_start, first_decode_done = first_decode_pair
        final_decode_done = decode_pairs[-1][1] if decode_pairs else None

        prefill_active_ms = _sum_pair_durations_ms(prefill_pairs)
        prefill_span_ms = _phase_span_ms(prefill_pairs)
        prefill_chunk_gap_ms = None
        if prefill_active_ms is not None and prefill_span_ms is not None:
            prefill_chunk_gap_ms = max(0.0,
                                       prefill_span_ms - prefill_active_ms)
        decode_active_ms = _sum_pair_durations_ms(decode_pairs)
        decode_span_ms = _phase_span_ms(decode_pairs)
        decode_gap_ms = None
        if decode_active_ms is not None and decode_span_ms is not None:
            decode_gap_ms = max(0.0, decode_span_ms - decode_active_ms)

        prefill_sampling_submit_ready = _sum_pair_durations_ms(
            _pair_correlated_events(events, "prefill_sampling_submit",
                                    "prefill_sampling_ready"))
        prefill_sampling_ready_collect = _sum_pair_durations_ms(
            _pair_correlated_events(events, "prefill_sampling_ready",
                                    "prefill_sampling_collected"))
        prefill_sampling_collect_commit = _sum_pair_durations_ms(
            _pair_correlated_events(events, "prefill_sampling_collected",
                                    "prefill_token_committed"))
        decode_sampling_submit_ready = _sum_pair_durations_ms(
            _pair_correlated_events(events, "decode_sampling_submit",
                                    "decode_sampling_ready"))
        decode_sampling_ready_collect = _sum_pair_durations_ms(
            _pair_correlated_events(events, "decode_sampling_ready",
                                    "decode_sampling_collected"))
        decode_sampling_collect_commit = _sum_pair_durations_ms(
            _pair_correlated_events(events, "decode_sampling_collected",
                                    "decode_token_committed"))
        token_commit_ready_pairs = _pair_correlated_events(
            events, "prefill_token_committed", "decode_ready")
        token_commit_ready_pairs.extend(
            _pair_correlated_events(events, "decode_token_committed",
                                    "decode_ready"))
        token_commit_to_decode_ready = _sum_pair_durations_ms(
            token_commit_ready_pairs)
        decode_ready_queue = _sum_pair_durations_ms(
            _pair_ordered_events(events, "decode_ready", "decode_start"))

        is_vision = vision_queued is not None
        common_stages_complete = server_submit is not None and server_admit is not None \
            and first_prefill_start is not None and final_prefill_done is not None \
            and first_token is not None and completion is not None
        encoder_stages_complete = not is_vision or (
            encoder_start is not None and encoder_done is not None
            and prefill_ready is not None)
        attribution = {
            "source":
            source,
            "request_index":
            request_id,
            "observed_lifecycles":
            len(lifecycles),
            "request_class":
            "vision" if is_vision else "text",
            "backend_arrival_timestamp_us":
            float(arrival["timestamp_us"]) if arrival is not None else None,
            "backend_completion_timestamp_us":
            float(completion["timestamp_us"])
            if completion is not None else None,
            "complete":
            arrival is not None and common_stages_complete
            and encoder_stages_complete,
            "prefill_dispatches":
            len(prefill_pairs),
            "decode_dispatches":
            len(decode_pairs),
            "mean_prefill_batch_size":
            sum(int(start.get("batch_size", 0))
                for start, _ in prefill_pairs) /
            len(prefill_pairs) if prefill_pairs else None,
            "mean_decode_batch_size":
            sum(int(start.get("batch_size", 0)) for start, _ in decode_pairs) /
            len(decode_pairs) if decode_pairs else None,
            "first_prefill_batch_size":
            int(first_prefill_start.get("batch_size", 0))
            if first_prefill_start is not None else None,
            "first_decode_batch_size":
            int(first_decode_start.get("batch_size", 0))
            if first_decode_start is not None else None,
            "encoder_queue_ms":
            _elapsed_ms(vision_queued, encoder_start) if is_vision else None,
            "encoder_active_ms":
            _elapsed_ms(encoder_start, encoder_done) if is_vision else None,
            "encoder_to_prefill_ready_ms":
            _elapsed_ms(encoder_done, prefill_ready) if is_vision else None,
            "vision_ready_queue_ms":
            _elapsed_ms(prefill_ready, server_submit) if is_vision else None,
            "prefill_admission_ms":
            _elapsed_ms(server_submit, server_admit),
            "prefill_initial_queue_ms":
            _elapsed_ms(server_admit, first_prefill_start),
            "prefill_active_ms":
            prefill_active_ms,
            "prefill_chunk_gap_ms":
            prefill_chunk_gap_ms,
            "prefill_total_span_ms":
            prefill_span_ms,
            "prefill_sampling_ms":
            _elapsed_ms(final_prefill_done, first_token),
            "prefill_sampling_submit_to_ready_ms":
            prefill_sampling_submit_ready,
            "prefill_sampling_ready_to_collect_ms":
            prefill_sampling_ready_collect,
            "prefill_sampling_collect_to_commit_ms":
            prefill_sampling_collect_commit,
            "first_token_to_decode_ms":
            _elapsed_ms(first_token, first_decode_start),
            "first_decode_active_ms":
            _elapsed_ms(first_decode_start, first_decode_done),
            "decode_total_active_ms":
            decode_active_ms,
            "decode_inter_dispatch_gap_ms":
            decode_gap_ms,
            "decode_sampling_submit_to_ready_ms":
            decode_sampling_submit_ready,
            "decode_sampling_ready_to_collect_ms":
            decode_sampling_ready_collect,
            "decode_sampling_collect_to_commit_ms":
            decode_sampling_collect_commit,
            "token_commit_to_decode_ready_ms":
            token_commit_to_decode_ready,
            "decode_ready_queue_ms":
            decode_ready_queue,
            "decode_tail_ms":
            _elapsed_ms(
                final_decode_done
                if final_decode_done is not None else first_token, completion),
            "token_commit_to_slot_release_ms":
            _elapsed_ms(final_token_committed, slot_released),
            "backend_ttft_ms":
            _elapsed_ms(arrival, first_token),
            "backend_e2e_ms":
            _elapsed_ms(arrival, completion),
        }
        applicable_fields = COMPONENT_FIELDS if is_vision else COMPONENT_FIELDS[
            4:]
        components = [attribution[field] for field in applicable_fields]
        if attribution["complete"] and all(value is not None
                                           for value in components):
            attribution["critical_path_residual_ms"] = attribution[
                "backend_ttft_ms"] - sum(components)
        else:
            attribution["critical_path_residual_ms"] = None
        decode_components = [
            attribution["first_token_to_decode_ms"],
            attribution["decode_total_active_ms"],
            attribution["decode_inter_dispatch_gap_ms"],
            attribution["decode_tail_ms"],
        ]
        if attribution["complete"] and all(value is not None
                                           for value in decode_components):
            attribution["e2e_residual_ms"] = attribution["backend_e2e_ms"] \
                - attribution["backend_ttft_ms"] - sum(decode_components)
        elif attribution["complete"] and not decode_pairs:
            attribution["e2e_residual_ms"] = 0.0
        else:
            attribution["e2e_residual_ms"] = None
        attributions.append(attribution)
    return attributions


def join_client_metrics(
        rows: list[dict[str, Any]],
        client_requests_by_source: dict[str, Path]) -> list[dict[str, Any]]:
    """Join real HTTP request metrics to backend timeline attribution."""
    client_rows: dict[tuple[str, int], dict[str, str]] = {}
    for source, path in client_requests_by_source.items():
        with path.open(newline="", encoding="utf-8") as client_file:
            for client_row in csv.DictReader(client_file):
                key = (source, int(client_row["request_id"]))
                client_rows[key] = client_row

    for row in rows:
        key = (str(row["source"]), int(row["request_index"]))
        client_row = client_rows.get(key)
        if client_row is None:
            continue
        request_class = str(client_row["request_class"])
        if request_class != row["request_class"]:
            raise ValueError(
                f"request {row['request_index']}: timeline class {row['request_class']} "
                f"does not match client class {request_class}")
        row["client_ttft_ms"] = float(client_row["ttft_ms"])
        row["client_tpot_ms"] = float(client_row["tpot_ms"])
        row["client_e2e_ms"] = float(client_row["e2e_ms"])
        if client_row.get("client_dispatch_delay_us"):
            row["client_dispatch_delay_ms"] = float(
                client_row["client_dispatch_delay_us"]) / 1000.0
        if client_row.get("scheduled_arrival_us") and client_row.get(
                "first_token_us"):
            row["scheduled_ttft_ms"] = (
                float(client_row["first_token_us"]) -
                float(client_row["scheduled_arrival_us"])) / 1000.0
        if client_row.get("scheduled_arrival_us") and client_row.get(
                "completed_us"):
            row["scheduled_e2e_ms"] = (
                float(client_row["completed_us"]) -
                float(client_row["scheduled_arrival_us"])) / 1000.0
        row["frontend_to_backend_ttft_ms"] = row["client_ttft_ms"] - row[
            "backend_ttft_ms"]
        row["backend_to_client_completion_ms"] = row["client_e2e_ms"] - row[
            "backend_e2e_ms"]
    return rows


def percentile(values: Iterable[float], quantile: float) -> float | None:
    """Return a linearly interpolated percentile compatible with NumPy's default."""
    sorted_values = sorted(float(value) for value in values)
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = quantile * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return sorted_values[lower] * (1.0 -
                                   fraction) + sorted_values[upper] * fraction


def _field_summary(rows: list[dict[str, Any]],
                   field: str) -> dict[str, float | int | None]:
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    return {
        "count": len(values),
        "mean": sum(values) / len(values) if values else None,
        "median": percentile(values, 0.5),
        "p95": percentile(values, 0.95),
        "max": max(values) if values else None,
    }


def _group_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    complete_rows = [row for row in rows if row["complete"]]
    tail_field = "client_ttft_ms" if any(row.get("client_ttft_ms") is not None for row in complete_rows) \
        else "backend_ttft_ms"
    ttft_values = [
        float(row[tail_field]) for row in complete_rows
        if row.get(tail_field) is not None
    ]
    tail_threshold = percentile(ttft_values, 0.95)
    tail_rows = [
        row for row in complete_rows
        if tail_threshold is not None and row.get(tail_field) is not None
        and row[tail_field] >= tail_threshold
    ]
    means = {
        field: _field_summary(complete_rows, field)["mean"]
        for field in COMPONENT_FIELDS
    }
    mean_ttft = _field_summary(complete_rows, "backend_ttft_ms")["mean"]
    component_shares = {
        field:
        value / mean_ttft
        if value is not None and mean_ttft not in (None, 0.0) else None
        for field, value in means.items()
    }
    return {
        "requests":
        len(rows),
        "complete_requests":
        len(complete_rows),
        "incomplete_request_ids":
        [int(row["request_index"]) for row in rows if not row["complete"]],
        "metrics_ms": {
            field: _field_summary(complete_rows, field)
            for field in SUMMARY_FIELDS
        },
        "dispatch": {
            field: _field_summary(complete_rows, field)
            for field in DISPATCH_FIELDS
        },
        "mean_ttft_component_share":
        component_shares,
        "tail": {
            "selection_metric": tail_field,
            "threshold_ms": tail_threshold,
            "requests": len(tail_rows),
            "request_ids": [int(row["request_index"]) for row in tail_rows],
            "metrics_ms": {
                field: _field_summary(tail_rows, field)
                for field in SUMMARY_FIELDS
            },
            "dispatch": {
                field: _field_summary(tail_rows, field)
                for field in DISPATCH_FIELDS
            },
        },
    }


def summarize_attributions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize attribution for all, text, and vision request groups."""
    return {
        "schema_version": 2,
        "groups": {
            "all":
            _group_summary(rows),
            "text":
            _group_summary(
                [row for row in rows if row["request_class"] == "text"]),
            "vision":
            _group_summary(
                [row for row in rows if row["request_class"] == "vision"]),
        },
    }


def select_dispatch_metrics(
        metrics: list[dict[str, Any]],
        rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep dispatches inside each source's measured request lifecycle window."""
    windows = {}
    request_ids = defaultdict(set)
    for row in rows:
        source = str(row["source"])
        request_ids[source].add(int(row["request_index"]))
        start = row.get("backend_arrival_timestamp_us")
        end = row.get("backend_completion_timestamp_us")
        if start is None or end is None:
            continue
        if source not in windows:
            windows[source] = [float(start), float(end)]
        else:
            windows[source][0] = min(windows[source][0], float(start))
            windows[source][1] = max(windows[source][1], float(end))

    selected = []
    for metric in metrics:
        source = str(metric["source"])
        if source not in windows:
            continue
        timestamp = float(metric.get("host_dispatch_start_us", -1.0))
        if not windows[source][0] <= timestamp <= windows[source][1]:
            continue
        members = {
            int(request_id)
            for field in ("prefill_request_ids", "decode_request_ids")
            for request_id in metric.get(field, [])
        }
        if members & request_ids[source]:
            selected.append(metric)
    return selected


def summarize_dispatch_pipeline(
        metrics: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize selector, host submission, GPU service, and realization residual."""
    kind_names = {1: "prefill", 2: "decode", 3: "overlap"}
    rows = []
    for metric in metrics:
        dispatch_start = float(metric.get("host_dispatch_start_us", 0.0))
        submission_end = float(
            metric.get("host_submission_end_us", dispatch_start))
        completion = float(metric.get("host_completion_us", submission_end))
        gpu_ms = float(metric.get("makespan_gpu_ms", 0.0))
        submission_ms = max(0.0, submission_end - dispatch_start) / 1000.0
        wall_ms = max(0.0, completion - dispatch_start) / 1000.0
        rows.append({
            "kind":
            kind_names.get(int(metric.get("kind", 0)), "none"),
            "scheduler_decision_ms":
            float(metric.get("host_scheduler_decision_us", 0.0)) / 1000.0,
            "host_submission_ms":
            submission_ms,
            "gpu_makespan_ms":
            gpu_ms,
            "dispatch_wall_ms":
            wall_ms,
            # Host submission and GPU execution overlap after the first stream
            # work is enqueued. Wall minus GPU makespan is therefore the safe
            # combined queue-start/completion-observation residual; subtracting
            # submission as well would double-count that overlap.
            "realization_residual_ms":
            max(0.0, wall_ms - gpu_ms),
        })

    def summarize(group: list[dict[str, Any]]) -> dict[str, Any]:
        fields = (
            "scheduler_decision_ms",
            "host_submission_ms",
            "gpu_makespan_ms",
            "dispatch_wall_ms",
            "realization_residual_ms",
        )
        return {
            "dispatches": len(group),
            "metrics_ms": {
                field: _field_summary(group, field)
                for field in fields
            },
        }

    return {
        "all": summarize(rows),
        **{
            kind: summarize([row for row in rows if row["kind"] == kind])
            for kind in kind_names.values()
        },
    }


def write_attributions(
        rows: list[dict[str, Any]],
        output_dir: Path,
        dispatch_metrics: list[dict[str, Any]] | None = None) -> None:
    """Write request CSV and aggregate JSON artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else [
        "source", "request_index", "request_class", "complete"
    ]
    with (output_dir / "request-attribution.csv").open(
            "w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize_attributions(rows)
    summary["dispatch_pipeline"] = summarize_dispatch_pipeline(dispatch_metrics
                                                               or [])
    with (output_dir / "phase-attribution.json").open(
            "w", encoding="utf-8") as destination:
        json.dump(summary, destination, indent=2, sort_keys=True)
        destination.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", type=Path, nargs="+", help="Gateway log files")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-request-id",
                        type=int,
                        default=DEFAULT_MAX_REQUEST_ID)
    parser.add_argument("--request-count", type=int)
    parser.add_argument(
        "--client-requests",
        type=Path,
        nargs="*",
        help="Client requests.csv files in the same order as logs")
    args = parser.parse_args()

    records = parse_timeline_logs(args.logs, args.max_request_id)
    dispatch_metrics = parse_dispatch_metrics(args.logs)
    rows = attribute_requests(records, args.request_count)
    if args.client_requests:
        if len(args.client_requests) != len(args.logs):
            parser.error(
                "--client-requests must provide one requests.csv for every gateway log"
            )
        rows = join_client_metrics(
            rows, {
                str(log): client
                for log, client in zip(args.logs, args.client_requests)
            })
    selected_dispatch_metrics = select_dispatch_metrics(dispatch_metrics, rows)
    write_attributions(rows, args.output_dir, selected_dispatch_metrics)
    complete = sum(bool(row["complete"]) for row in rows)
    print(
        f"Attributed {complete}/{len(rows)} production requests from {len(records)} timeline events"
    )
    return 0 if rows and complete == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
