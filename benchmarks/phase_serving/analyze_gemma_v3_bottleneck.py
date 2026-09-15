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
"""Summarize measurement-reset V3 activity without clipping early encoder work."""

import argparse
import collections
import csv
import json
import pathlib
import statistics

import analyze_phase_activity


def distribution(values):
    """Return descriptive values; p95 uses linear interpolation."""
    if not values:
        return {"samples": 0}
    ordered = sorted(values)
    position = 0.95 * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return {
        "samples":
        len(values),
        "mean":
        statistics.mean(values),
        "p95":
        ordered[lower] + (position - lower) *
        (ordered[upper] - ordered[lower]),
        "max":
        max(values),
        "min":
        min(values),
    }


def stage_durations(lifecycles, start, end, first_start=False):
    """Pair request stages, retaining the first P chunk start when requested."""
    durations = []
    for events in lifecycles.values():
        pending = None
        for event in events:
            if event["stage"] == start and (pending is None
                                            or not first_start):
                pending = float(event["timestamp_us"])
            elif event["stage"] == end and pending is not None:
                durations.append(
                    (float(event["timestamp_us"]) - pending) / 1000.0)
                pending = None
    return distribution(durations)


def analyze_cell(cell):
    """Analyze a recorder reset at PHASE_EPOCH, retaining every recorded span."""
    run = cell / "run-001"
    serving = json.loads((run / "client/run-001/summary.json").read_text())
    log = run / "gateway.log"
    if 'PHASE_EPOCH\t{"epoch":1,"kind":"measurement"}' not in log.read_text():
        raise ValueError(f"{cell}: missing measurement reset contract")
    intervals = analyze_phase_activity.read_activity_intervals(
        run / "activity-intervals.csv")
    segments = analyze_phase_activity.activity_segments(intervals)
    metrics = [
        metric for metric in analyze_phase_activity.parse_phase_metrics(log)
        if metric.get("measurement_epoch") == 1
    ]
    lifecycles = analyze_phase_activity.measured_lifecycles(
        log, serving["requests"])
    activity = analyze_phase_activity.summarize_activity(
        intervals, segments, metrics)
    activity["requests"] = len(lifecycles)
    analyze_phase_activity.write_results(run / "analysis", intervals, segments,
                                         activity)
    window = activity["active_span_ms"]
    activity["mask_percent"] = {
        mask: value / window * 100
        for mask, value in activity["mask_ms"].items()
    }
    dispatches = {}
    for phase in ("prefill", "decode"):
        rows = [m for m in metrics if m.get(f"{phase}_batch", 0) > 0]
        dispatches[phase] = {
            "dispatches":
            len(rows),
            "mean_batch":
            statistics.mean(m[f"{phase}_batch"] for m in rows) if rows else 0,
            "batch_histogram":
            dict(collections.Counter(m[f"{phase}_batch"] for m in rows)),
            "gpu_ms":
            distribution([m[f"{phase}_gpu_ms"] for m in rows]),
            "total_useful_tokens":
            sum(m.get(f"{phase}_tokens", 0) for m in rows),
        }
    final_metric = metrics[-1] if metrics else {}
    prefill_refills = [
        m for m in metrics if m.get("prefill_cohort_refill_rows", 0) > 0
    ]
    names = collections.defaultdict(list)
    for interval in intervals:
        names[interval["name"]].append(interval["end_ms"] -
                                       interval["start_ms"])
    gaps = [
        segment["duration_ms"] for segment in segments if segment["mask"] == 0
    ]
    encoder_batches = []
    decisions = []
    first_request_ns = min(
        float(e["timestamp_us"]) * 1000 for events in lifecycles.values()
        for e in events)
    last_request_ns = max(
        float(e["timestamp_us"]) * 1000 for events in lifecycles.values()
        for e in events)
    epoch = 0
    for line in log.read_text().splitlines():
        if "PHASE_EPOCH\t" in line:
            epoch = json.loads(line.split("PHASE_EPOCH\t", 1)[1])["epoch"]
        if epoch == 1 and "PHASE_ENCODER_METRIC\t" in line:
            encoder_batches.append(
                json.loads(line.split("PHASE_ENCODER_METRIC\t", 1)[1]))
        if "PHASE_SCHEDULER_EVENT\t" in line:
            event = json.loads(line.split("PHASE_SCHEDULER_EVENT\t", 1)[1])
            if (event["event_kind"] == "decision" and first_request_ns <=
                    event["host_monotonic_ns"] <= last_request_ns):
                decisions.append(event)
    small_prefill = [
        e for e in decisions if e["selected_cohort"]["prefill_rows"] == 1
        and len(e["ready_prefill_request_ids"]) > 1
    ]
    single_row_frontier = [
        e for e in small_prefill if not any(
            c["legal"] and ((c["action_kind"] in {"prefill", "prefill_decode"}
                             and c["primary_batch_size"] > 1) or
                            (c["action_kind"] == "encoder_prefill"
                             and c["secondary_batch_size"] > 1))
            for c in e["candidates"])
    ]
    dispatches["encoder"] = {
        "dispatches":
        len(encoder_batches),
        "batch_histogram":
        dict(collections.Counter(m["batch_size"] for m in encoder_batches)),
        "mean_batch":
        statistics.mean(m["batch_size"]
                        for m in encoder_batches) if encoder_batches else 0,
        "execution_gpu_ms":
        distribution([m["execution_gpu_ms"] for m in encoder_batches]),
    }
    with (run / "client/run-001/requests.csv").open(newline="") as source:
        requests = {
            int(row["request_id"]): row
            for row in csv.DictReader(source)
        }
    lease_edges = []
    for request_id, events in lifecycles.items():
        row = requests[request_id]
        pages = (int(row["prompt_tokens"]) + int(row["max_output_tokens"]) +
                 127) // 128
        for event in events:
            if event["stage"] in {"server_admit", "slot_released"} and int(
                    event["kv_slot_id"]) >= 0:
                delta = 1 if event["stage"] == "server_admit" else -1
                lease_edges.append(
                    (float(event["timestamp_us"]), delta, delta * pages))
    active_owners = active_pages = peak_owners = peak_pages = 0
    for _, owner_delta, page_delta in sorted(lease_edges):
        active_owners += owner_delta
        active_pages += page_delta
        peak_owners = max(peak_owners, active_owners)
        peak_pages = max(peak_pages, active_pages)
    pairs = [("server_submit", "server_admit"),
             ("server_admit", "prefill_start"),
             ("server_admit", "first_token"),
             ("vision_queued", "encoder_start"),
             ("encoder_start", "encoder_done"),
             ("encoder_done", "prefill_ready"),
             ("prefill_ready", "prefill_start"),
             ("prefill_start", "first_token"),
             ("decode_done", "decode_sampling_collected"),
             ("decode_sampling_collected", "decode_token_committed"),
             ("decode_token_committed", "decode_start")]
    return {
        "serving":
        serving,
        "activity":
        activity,
        "idle_spans_ms":
        distribution(gaps),
        "interval_names_ms": {
            name: {
                **distribution(values), "total": sum(values)
            }
            for name, values in names.items()
        },
        "dispatches":
        dispatches,
        "decision_frontier": {
            "decisions":
            len(decisions),
            "action_histogram":
            dict(collections.Counter(e["action_kind"] for e in decisions)),
            "selected_p1_with_multiple_ready_rows":
            len(small_prefill),
            "selected_p1_without_larger_p_frontier":
            len(single_row_frontier),
        },
        "execution_density": {
            "action_boundaries":
            len(metrics),
            "boundaries_per_generated_token":
            len(metrics) / max(1, int(serving.get("generated_tokens", 0))),
            "prefill_refill_dispatches":
            len(prefill_refills),
            "prefill_refill_rows":
            sum(m.get("prefill_cohort_refill_rows", 0)
                for m in prefill_refills),
            "prefill_refill_batch":
            distribution([m.get("prefill_batch", 0)
                          for m in prefill_refills]),
        },
        "cuda_graphs": {
            "prefill_entries":
            final_metric.get("prefill_graph_entries", 0),
            "prefill_hits":
            final_metric.get("prefill_graph_hits", 0),
            "prefill_misses":
            final_metric.get("prefill_graph_misses", 0),
            "prefill_captures":
            final_metric.get("prefill_graph_captures", 0),
            "prefill_evictions":
            final_metric.get("prefill_graph_evictions", 0),
            "prefill_hit_rate":
            final_metric.get("prefill_graph_hits", 0) / max(
                1,
                final_metric.get("prefill_graph_hits", 0) +
                final_metric.get("prefill_graph_misses", 0)),
            "decode_entries":
            final_metric.get("decode_graph_entries", 0),
            "decode_hits":
            final_metric.get("decode_graph_hits", 0),
            "decode_misses":
            final_metric.get("decode_graph_misses", 0),
            "decode_captures":
            final_metric.get("decode_graph_captures", 0),
            "decode_evictions":
            final_metric.get("decode_graph_evictions", 0),
            "decode_hit_rate":
            final_metric.get("decode_graph_hits", 0) / max(
                1,
                final_metric.get("decode_graph_hits", 0) +
                final_metric.get("decode_graph_misses", 0)),
        },
        "full_reservation_reconstruction": {
            "peak_owners":
            peak_owners,
            "peak_pages_from_response_prompt_and_output_limit":
            peak_pages,
            "ending_owners":
            active_owners,
            "ending_pages":
            active_pages,
            "sampled_free_pages":
            distribution([
                m["vision_available_kv_pages"] for m in metrics
                if "vision_available_kv_pages" in m
            ]),
        },
        "host_submission_ms":
        distribution([
            (m["host_submission_end_us"] - m["host_dispatch_start_us"]) /
            1000.0 for m in metrics
        ]),
        "host_scheduler_ms":
        distribution(
            [m["host_scheduler_decision_us"] / 1000.0 for m in metrics]),
        "request_stage_ms": {
            f"{start}->{end}":
            stage_durations(lifecycles,
                            start,
                            end,
                            first_start=start == "prefill_start"
                            and end == "first_token")
            for start, end in pairs
        },
    }


def main():
    """Write a structured diagnostic report for every completed cell."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=pathlib.Path, required=True)
    args = parser.parse_args()
    report = {
        cell.name: analyze_cell(cell)
        for cell in sorted(args.result_root.iterdir())
        if cell.is_dir() and (cell / "aggregate.json").is_file()
    }
    (args.result_root / "bottleneck-analysis.json"
     ).write_text(json.dumps(report, indent=2) + "\n")
    for workload, result in report.items():
        activity = result["activity"]
        print(
            workload, "tok/s", round(result["serving"]["generated_token_s"],
                                     2), "idle%",
            round(activity["all_idle_ratio"] * 100, 2), "EPD overlap%",
            round(
                activity["epd_overlap_ms"] / activity["active_span_ms"] * 100,
                2), "duty%", {
                    k: round(v * 100, 2)
                    for k, v in activity["phase_ratio"].items()
                })


if __name__ == "__main__":
    main()
