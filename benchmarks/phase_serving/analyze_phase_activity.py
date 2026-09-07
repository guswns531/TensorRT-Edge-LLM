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
"""Compare planned phase actions with measured E/P/D/Copy CUDA activity."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

METRIC_MARKER = "PHASE_METRIC\t"
TIMELINE_MARKER = "PHASE_TIMELINE\t"
PHASE_BITS = {"encoder": 0x1, "prefill": 0x2, "decode": 0x4, "copy": 0x8}


def _is_dispatch_activity(interval: dict[str, Any]) -> bool:
    return interval["kind"] in {"prefill", "decode"} and str(
        interval["name"]).endswith("_dispatch")


def _parse_marked_json(path: Path, marker: str) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8", errors="replace") as source:
        for line_number, line in enumerate(source, start=1):
            offset = line.find(marker)
            if offset < 0:
                continue
            try:
                records.append(json.loads(line[offset + len(marker):].strip()))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{path}:{line_number}: invalid JSON after {marker.strip()}"
                ) from error
    return records


def parse_phase_metrics(path: Path) -> list[dict[str, Any]]:
    """Read dispatch-level scheduler metrics from a gateway log."""
    return _parse_marked_json(path, METRIC_MARKER)


def measured_lifecycles(
        path: Path,
        request_count: int | None = None) -> dict[int, list[dict[str, Any]]]:
    """Return the last completed lifecycle for every measured request ID.

    HTTP warmup reuses production request IDs. A completion record delimits each
    lifecycle, so selecting the last completed lifecycle removes warmup without
    requiring a recorder reset in the serving process.
    """
    records = _parse_marked_json(path, TIMELINE_MARKER)
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        request_id = int(record["request_index"])
        if request_count is not None and request_id >= request_count:
            continue
        grouped[request_id].append(record)

    result = {}
    for request_id, request_records in grouped.items():
        request_records.sort(key=lambda record: float(record["timestamp_us"]))
        lifecycles = []
        current = []
        for record in request_records:
            current.append(record)
            if record.get("stage") == "completion":
                lifecycles.append(current)
                current = []
        if lifecycles:
            result[request_id] = lifecycles[-1]
    return result


def read_activity_intervals(path: Path) -> list[dict[str, Any]]:
    """Read the runtime's epoch-relative CUDA-event interval CSV."""
    intervals = []
    with path.open(newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            intervals.append({
                "interval_id": int(row["interval_id"]),
                "correlation_id": int(row["correlation_id"]),
                "kind": row["kind"],
                "name": row["name"],
                "start_ms": float(row["start_ms"]),
                "end_ms": float(row["end_ms"]),
            })
    return intervals


def select_measured_intervals(
        intervals: Iterable[dict[str, Any]],
        lifecycles: dict[int, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Select activity belonging to measured lifecycles rather than warmup.

    P/D engine intervals use dispatch index correlation IDs, while sampling
    intervals do not. E/C intervals use the first request ID in an encoder
    batch. Combine kind-aware correlation with a measured time window. The
    window starts after the final non-measured P/D engine and its immediate
    sampling tail, which excludes warmup even when request IDs are reused.
    """
    dispatch_ids_by_kind = {
        kind: {
            int(record["dispatch_index"])
            for records in lifecycles.values()
            for record in records
            if str(record.get("stage", "")).startswith(kind)
            and int(record.get("dispatch_index", 0)) > 0
        }
        for kind in ("prefill", "decode")
    }
    vision_request_ids = {
        request_id
        for request_id, records in lifecycles.items() if any(
            record.get("stage") == "vision_queued" for record in records)
    }
    intervals = list(intervals)
    selected_dispatches = [
        interval for interval in intervals
        if interval["kind"] in {"prefill", "decode"}
        and _is_dispatch_activity(interval) and interval["correlation_id"] in
        dispatch_ids_by_kind[interval["kind"]]
    ]
    if not selected_dispatches:
        return []
    first_measured_pd_ms = min(interval["start_ms"]
                               for interval in selected_dispatches)
    last_measured_pd_ms = max(interval["end_ms"]
                              for interval in selected_dispatches)
    warmup_pd_ends = [
        interval["end_ms"] for interval in intervals
        if interval["kind"] in {"prefill", "decode"}
        and _is_dispatch_activity(interval)
        and interval["correlation_id"] not in dispatch_ids_by_kind[
            interval["kind"]] and interval["end_ms"] <= first_measured_pd_ms
    ]
    warmup_pd_end_ms = max(warmup_pd_ends, default=first_measured_pd_ms)
    warmup_auxiliary_ends = [
        interval["end_ms"] for interval in intervals
        if not _is_dispatch_activity(interval) and interval["start_ms"] >=
        warmup_pd_end_ms and interval["start_ms"] <= warmup_pd_end_ms + 1.0
    ]
    measured_window_start_ms = max(warmup_auxiliary_ends,
                                   default=warmup_pd_end_ms)
    # Sampling follows its engine by a small amount on the same phase stream.
    # Keep a bounded tail so the last dispatch's sampling interval is retained.
    measured_window_end_ms = last_measured_pd_ms + 10.0
    selected_sampling = [
        interval for interval in intervals
        if interval["kind"] in {"prefill", "decode"}
        and not _is_dispatch_activity(interval)
        and interval["start_ms"] >= measured_window_start_ms
        and interval["start_ms"] <= measured_window_end_ms
    ]
    selected_ec = [
        interval for interval in intervals
        if interval["kind"] in {"encoder", "copy"}
        and interval["correlation_id"] in vision_request_ids
        and interval["start_ms"] >= measured_window_start_ms
        and interval["start_ms"] <= measured_window_end_ms
    ]
    selected = selected_dispatches + selected_sampling + selected_ec
    return sorted(selected,
                  key=lambda interval:
                  (interval["start_ms"], interval["end_ms"]))


def activity_segments(
        intervals: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sweep selected intervals into a non-overlapping 4-bit activity timeline."""
    intervals = list(intervals)
    if not intervals:
        return []
    boundaries = []
    first_start = min(interval["start_ms"] for interval in intervals)
    last_end = max(interval["end_ms"] for interval in intervals)
    for interval in intervals:
        bit = PHASE_BITS[interval["kind"]]
        boundaries.append((interval["start_ms"], bit, 1))
        boundaries.append((interval["end_ms"], bit, -1))
    boundaries.sort(key=lambda boundary: boundary[0])

    counts = Counter()
    cursor = first_start
    index = 0
    segments = []
    while index < len(boundaries):
        timestamp = boundaries[index][0]
        if timestamp > cursor:
            mask = sum(bit for bit, count in counts.items() if count > 0)
            segments.append({
                "start_ms": cursor - first_start,
                "end_ms": timestamp - first_start,
                "duration_ms": timestamp - cursor,
                "mask": mask,
                "binary_mask": f"{mask:04b}",
            })
            cursor = timestamp
        while index < len(boundaries) and boundaries[index][0] == timestamp:
            _, bit, delta = boundaries[index]
            counts[bit] += delta
            if counts[bit] < 0:
                raise ValueError("activity interval closes an inactive phase")
            index += 1
    if cursor < last_end:
        mask = sum(bit for bit, count in counts.items() if count > 0)
        segments.append({
            "start_ms": cursor - first_start,
            "end_ms": last_end - first_start,
            "duration_ms": last_end - cursor,
            "mask": mask,
            "binary_mask": f"{mask:04b}",
        })
    return segments


def _interval_overlap_ms(left: dict[str, Any], right: dict[str, Any]) -> float:
    return max(
        0.0,
        min(left["end_ms"], right["end_ms"]) -
        max(left["start_ms"], right["start_ms"]))


def _nearest_percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(
        0, min(len(ordered) - 1,
               int(fraction * len(ordered) + 0.999999) - 1))
    return ordered[index]


def summarize_activity(intervals: list[dict[str, Any]],
                       segments: list[dict[str, Any]],
                       metrics: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize stream occupancy and planned P+D action fidelity."""
    mask_ms = {f"{mask:04b}": 0.0 for mask in range(16)}
    for segment in segments:
        mask_ms[segment["binary_mask"]] += segment["duration_ms"]
    window_ms = sum(mask_ms.values())
    phase_ms = {
        phase:
        sum(duration for mask, duration in mask_ms.items()
            if int(mask, 2) & bit)
        for phase, bit in PHASE_BITS.items()
    }
    selected_dispatch_ids = {
        interval["correlation_id"]
        for interval in intervals if _is_dispatch_activity(interval)
    }
    selected_metrics = [
        metric for metric in metrics
        if int(metric.get("dispatch_index", -1)) in selected_dispatch_ids
    ]
    planned_actions = Counter(
        str(metric.get("global_action", "unknown"))
        for metric in selected_metrics)

    by_dispatch: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    for interval in intervals:
        if _is_dispatch_activity(interval):
            by_dispatch[interval["correlation_id"]][
                interval["kind"]] = interval
    planned_pd_ids = {
        int(metric["dispatch_index"])
        for metric in selected_metrics
        if metric.get("global_action") == "prefill_decode"
    }
    actual_pd = {}
    for dispatch_id, phases in by_dispatch.items():
        if "prefill" in phases and "decode" in phases:
            overlap_ms = _interval_overlap_ms(phases["prefill"],
                                              phases["decode"])
            if overlap_ms > 0.0:
                actual_pd[dispatch_id] = overlap_ms

    planned_masks = Counter(
        int(metric.get("vision_global_planned_outstanding", 0))
        for metric in selected_metrics)
    observed_masks = Counter(
        int(metric.get("vision_global_observed_outstanding", 0))
        for metric in selected_metrics)
    decode_dispatches = sorted(
        (interval for interval in intervals
         if interval["kind"] == "decode" and _is_dispatch_activity(interval)),
        key=lambda interval: interval["start_ms"])
    decode_gaps_ms = [
        max(0.0, right["start_ms"] - left["end_ms"])
        for left, right in zip(decode_dispatches, decode_dispatches[1:])
    ]
    max_non_decode_streak = 0
    non_decode_streak = 0
    for metric in sorted(selected_metrics,
                         key=lambda item: int(item["dispatch_index"])):
        action = str(metric.get("global_action", "unknown"))
        if action in {"decode", "encoder_decode", "prefill_decode"}:
            non_decode_streak = 0
        else:
            non_decode_streak += 1
            max_non_decode_streak = max(max_non_decode_streak,
                                        non_decode_streak)
    result = {
        "schema_version":
        1,
        "intervals":
        len(intervals),
        "segments":
        len(segments),
        "active_span_ms":
        window_ms,
        "phase_ms":
        phase_ms,
        "phase_ratio": {
            phase: duration / window_ms if window_ms else 0.0
            for phase, duration in phase_ms.items()
        },
        "mask_ms":
        mask_ms,
        "all_idle_ms":
        mask_ms["0000"],
        "all_idle_ratio":
        mask_ms["0000"] / window_ms if window_ms else 0.0,
        "epd_idle_ms":
        mask_ms["0000"] + mask_ms["1000"],
        "epd_idle_ratio":
        (mask_ms["0000"] + mask_ms["1000"]) / window_ms if window_ms else 0.0,
        "any_overlap_ms":
        sum(duration for mask, duration in mask_ms.items()
            if int(mask, 2).bit_count() >= 2),
        "epd_overlap_ms":
        sum(duration for mask, duration in mask_ms.items()
            if (int(mask, 2) & 0x7).bit_count() >= 2),
        "planned": {
            "dispatches": len(selected_metrics),
            "actions": dict(sorted(planned_actions.items())),
            "outstanding_masks": {
                f"{mask:04b}": count
                for mask, count in sorted(planned_masks.items())
            },
            "observed_outstanding_masks": {
                f"{mask:04b}": count
                for mask, count in sorted(observed_masks.items())
            },
        },
        "prefill_decode_fidelity": {
            "planned": len(planned_pd_ids),
            "actual_same_dispatch": len(planned_pd_ids & actual_pd.keys()),
            "missed": len(planned_pd_ids - actual_pd.keys()),
            "unplanned_same_dispatch": len(actual_pd.keys() - planned_pd_ids),
            "same_dispatch_overlap_ms": sum(actual_pd.values()),
            "dispatch_overlap_ms": {
                str(key): value
                for key, value in sorted(actual_pd.items())
            },
        },
        "decode_continuity": {
            "dispatches":
            len(decode_dispatches),
            "gap_samples":
            len(decode_gaps_ms),
            "gap_mean_ms": (sum(decode_gaps_ms) /
                            len(decode_gaps_ms) if decode_gaps_ms else 0.0),
            "gap_p95_ms":
            _nearest_percentile(decode_gaps_ms, 0.95),
            "gap_max_ms":
            max(decode_gaps_ms, default=0.0),
            "gaps_over_25_ms":
            sum(gap > 25.0 for gap in decode_gaps_ms),
            "max_consecutive_non_decode_actions":
            max_non_decode_streak,
        },
    }
    return result


def analyze(
    gateway_log: Path,
    intervals_csv: Path,
    request_count: int | None = None
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Analyze one measured workload run."""
    lifecycles = measured_lifecycles(gateway_log, request_count)
    intervals = select_measured_intervals(
        read_activity_intervals(intervals_csv), lifecycles)
    segments = activity_segments(intervals)
    summary = summarize_activity(intervals, segments,
                                 parse_phase_metrics(gateway_log))
    summary["requests"] = len(lifecycles)
    return intervals, segments, summary


def write_results(output_dir: Path, intervals: list[dict[str, Any]],
                  segments: list[dict[str, Any]], summary: dict[str,
                                                                Any]) -> None:
    """Write selected intervals, segments, and aggregate JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    interval_fields = [
        "interval_id", "correlation_id", "kind", "name", "start_ms", "end_ms"
    ]
    with (output_dir / "measured-intervals.csv").open(
            "w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=interval_fields)
        writer.writeheader()
        writer.writerows(intervals)
    segment_fields = [
        "start_ms", "end_ms", "duration_ms", "mask", "binary_mask"
    ]
    with (output_dir / "measured-segments.csv").open(
            "w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=segment_fields)
        writer.writeheader()
        writer.writerows(segments)
    with (output_dir / "activity-summary.json").open(
            "w", encoding="utf-8") as destination:
        json.dump(summary, destination, indent=2, sort_keys=True)
        destination.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gateway-log", type=Path, required=True)
    parser.add_argument("--intervals", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--request-count", type=int)
    args = parser.parse_args()

    intervals, segments, summary = analyze(args.gateway_log, args.intervals,
                                           args.request_count)
    write_results(args.output_dir, intervals, segments, summary)
    print(json.dumps(summary, sort_keys=True))
    return 0 if summary["requests"] > 0 and summary["intervals"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
