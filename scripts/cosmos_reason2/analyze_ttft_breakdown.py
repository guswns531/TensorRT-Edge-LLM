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
"""Join request, prefill timeline, and CUDA-event data into a TTFT breakdown."""

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

PREFILL_GROUPS = (
    "prefill_prepare",
    "prefill_engine",
    "prefill_cache_commit",
    "prefill_sample",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"CSV is empty: {path}")
    return rows


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    rank = (len(ordered) - 1) * fraction
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (rank - lower)


def build_breakdown(request_csv: Path, timeline_csv: Path,
                    kernel_csv: Path) -> list[dict[str, float | int]]:
    requests = {int(row["request_id"]): row for row in read_csv(request_csv)}
    timeline: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(timeline_csv):
        timeline[int(row["request_id"])].append(row)

    kernel_ms: dict[int, dict[str,
                              float]] = defaultdict(lambda: defaultdict(float))
    for row in read_csv(kernel_csv):
        group = row["group"]
        if group in PREFILL_GROUPS:
            kernel_ms[int(row["dispatch_index"])][group] += float(
                row["gpu_ms"])

    result: list[dict[str, float | int]] = []
    for request_id, request in requests.items():
        chunks = sorted(timeline.get(request_id, []),
                        key=lambda row: int(row["token_offset"]))
        if not chunks:
            raise ValueError(f"request {request_id} has no prefill timeline")
        if int(chunks[0]["token_offset"]) != 0 or int(
                chunks[-1]["final_chunk"]) != 1:
            raise ValueError(
                f"request {request_id} has an incomplete timeline")

        scheduled = int(request["scheduled_arrival_us"])
        submitted = int(request["submitted_us"])
        admitted = int(request["admitted_us"])
        first_token = int(request["first_token_us"])
        cursor = admitted
        scheduler_wait_us = 0
        host_pack_us = 0
        phase_service_us = 0
        group_us = {group: 0.0 for group in PREFILL_GROUPS}
        for chunk in chunks:
            selected = int(chunk["dispatch_selected_us"])
            packed = int(chunk["pack_completed_us"])
            completed = int(chunk["phase_completed_us"])
            if selected < cursor or packed < selected or completed < packed:
                raise ValueError(
                    f"request {request_id} has non-monotonic phase timestamps")
            scheduler_wait_us += selected - cursor
            host_pack_us += packed - selected
            phase_service_us += completed - packed
            cursor = completed
            dispatch = int(chunk["kernel_dispatch_index"])
            if dispatch not in kernel_ms:
                raise ValueError(
                    f"request {request_id} dispatch {dispatch} has no CUDA-event sample"
                )
            for group in PREFILL_GROUPS:
                group_us[group] += kernel_ms[dispatch].get(group, 0.0) * 1000.0

        delivery_us = first_token - cursor
        if delivery_us < 0:
            raise ValueError(
                f"request {request_id} first-token timestamp precedes prefill completion"
            )
        gpu_total_us = sum(group_us.values())
        stream_residual_us = phase_service_us - gpu_total_us
        ttft_us = first_token - scheduled
        partition_us = (submitted - scheduled + admitted - submitted +
                        scheduler_wait_us + host_pack_us + phase_service_us +
                        delivery_us)
        if partition_us != ttft_us:
            raise ValueError(
                f"request {request_id} TTFT partition does not close")

        result.append({
            "request_id":
            request_id,
            "prompt_tokens":
            int(request["prompt_tokens"]),
            "prefill_chunks":
            len(chunks),
            "ttft_us":
            ttft_us,
            "arrival_queue_us":
            submitted - scheduled,
            "admission_us":
            admitted - submitted,
            "scheduler_wait_us":
            scheduler_wait_us,
            "host_pack_us":
            host_pack_us,
            "phase_service_wall_us":
            phase_service_us,
            "prefill_prepare_gpu_us":
            group_us["prefill_prepare"],
            "prefill_engine_gpu_us":
            group_us["prefill_engine"],
            "prefill_cache_commit_gpu_us":
            group_us["prefill_cache_commit"],
            "prefill_sample_gpu_us":
            group_us["prefill_sample"],
            "prefill_gpu_total_us":
            gpu_total_us,
            "stream_or_completion_residual_us":
            stream_residual_us,
            "first_token_delivery_us":
            delivery_us,
        })
    return sorted(result, key=lambda row: int(row["request_id"]))


def summarize(rows: list[dict[str, float | int]]) -> dict[str, object]:
    components = [key for key in rows[0] if key.endswith("_us")]
    metrics = {}
    for component in components:
        values = [float(row[component]) for row in rows]
        metrics[component] = {
            "median_ms": statistics.median(values) / 1000.0,
            "p95_ms": percentile(values, 0.95) / 1000.0,
            "mean_ms": statistics.mean(values) / 1000.0,
        }
    tail_count = max(1, math.ceil(len(rows) * 0.05))
    tail = sorted(rows, key=lambda row: float(row["ttft_us"]),
                  reverse=True)[:tail_count]
    return {
        "requests": len(rows),
        "p95_tail_request_count": tail_count,
        "p95_tail_request_ids": [int(row["request_id"]) for row in tail],
        "metrics": metrics,
        "p95_tail_component_mean_ms": {
            component:
            statistics.mean(float(row[component]) for row in tail) / 1000.0
            for component in components
        },
    }


def write_csv(path: Path, rows: list[dict[str, float | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--timeline", type=Path, required=True)
    parser.add_argument("--kernel-groups", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    args = parser.parse_args()

    rows = build_breakdown(args.requests, args.timeline, args.kernel_groups)
    write_csv(args.output_csv, rows)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summarize(rows), indent=2) + "\n",
                                 encoding="utf-8")
    print(f"wrote {args.output_csv} and {args.summary_json}")


if __name__ == "__main__":
    main()
