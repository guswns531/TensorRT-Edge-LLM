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

import importlib.util
import json
from pathlib import Path

import pytest

ANALYZER_PATH = Path(__file__).parents[
    2] / "benchmarks" / "phase_serving" / "analyze_phase_activity.py"
SPEC = importlib.util.spec_from_file_location("phase_activity_analysis",
                                              ANALYZER_PATH)
ANALYZER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(ANALYZER)


def _interval(kind, correlation_id, start_ms, end_ms, interval_id=1):
    return {
        "interval_id": interval_id,
        "correlation_id": correlation_id,
        "kind": kind,
        "name": f"{kind}_dispatch",
        "start_ms": start_ms,
        "end_ms": end_ms,
    }


def test_activity_segments_preserve_internal_idle_and_four_way_overlap():
    intervals = [
        _interval("encoder", 0, 1.0, 5.0),
        _interval("prefill", 10, 2.0, 4.0),
        _interval("decode", 10, 2.5, 3.5),
        _interval("copy", 0, 3.0, 6.0),
        _interval("decode", 11, 7.0, 8.0),
    ]

    segments = ANALYZER.activity_segments(intervals)
    summary = ANALYZER.summarize_activity(
        intervals,
        segments,
        [{
            "dispatch_index": 10,
            "global_action": "prefill_decode"
        }, {
            "dispatch_index": 11,
            "global_action": "decode"
        }],
    )

    assert summary["active_span_ms"] == pytest.approx(7.0)
    assert summary["mask_ms"]["1111"] == pytest.approx(0.5)
    assert summary["mask_ms"]["0000"] == pytest.approx(1.0)
    assert summary["prefill_decode_fidelity"]["planned"] == 1
    assert summary["prefill_decode_fidelity"]["actual_same_dispatch"] == 1


def test_measured_lifecycle_and_interval_selection_exclude_warmup(tmp_path):
    gateway_log = tmp_path / "gateway.log"
    events = [
        {
            "request_index": 0,
            "stage": "vision_queued",
            "timestamp_us": 0
        },
        {
            "request_index": 0,
            "stage": "prefill_start",
            "timestamp_us": 1,
            "dispatch_index": 5
        },
        {
            "request_index": 0,
            "stage": "prefill_done",
            "timestamp_us": 2,
            "dispatch_index": 5
        },
        {
            "request_index": 0,
            "stage": "completion",
            "timestamp_us": 3
        },
        {
            "request_index": 0,
            "stage": "vision_queued",
            "timestamp_us": 10
        },
        {
            "request_index": 0,
            "stage": "prefill_start",
            "timestamp_us": 11,
            "dispatch_index": 20
        },
        {
            "request_index": 0,
            "stage": "prefill_done",
            "timestamp_us": 12,
            "dispatch_index": 20
        },
        {
            "request_index": 0,
            "stage": "decode_start",
            "timestamp_us": 13,
            "dispatch_index": 21
        },
        {
            "request_index": 0,
            "stage": "decode_done",
            "timestamp_us": 14,
            "dispatch_index": 21
        },
        {
            "request_index": 0,
            "stage": "completion",
            "timestamp_us": 15
        },
    ]
    gateway_log.write_text("".join(f"PHASE_TIMELINE\t{json.dumps(event)}\n"
                                   for event in events),
                           encoding="utf-8")
    intervals = [
        _interval("encoder", 0, 1.0, 2.0, 1),
        _interval("prefill", 5, 2.0, 3.0, 2),
        {
            **_interval("decode", 20, 3.01, 3.02, 8),
            "name": "decode_sampling",
        },
        # E/C use the representative request index and a measured time window.
        _interval("encoder", 0, 10.0, 11.0, 3),
        _interval("copy", 0, 11.0, 11.5, 4),
        _interval("encoder", 999, 11.5, 11.75, 9),
        _interval("prefill", 20, 12.0, 13.0, 5),
        _interval("decode", 21, 13.0, 14.0, 6),
        {
            **_interval("decode", 999, 14.1, 14.2, 7),
            "name": "decode_sampling",
        },
    ]

    lifecycles = ANALYZER.measured_lifecycles(gateway_log, request_count=1)
    selected = ANALYZER.select_measured_intervals(intervals, lifecycles)

    assert {interval["interval_id"]
            for interval in selected} == {3, 4, 5, 6, 7}


def test_missed_planned_overlap_is_reported():
    intervals = [
        _interval("prefill", 30, 0.0, 2.0),
        _interval("decode", 30, 2.1, 3.0),
    ]
    metrics = [{"dispatch_index": 30, "global_action": "prefill_decode"}]

    summary = ANALYZER.summarize_activity(
        intervals, ANALYZER.activity_segments(intervals), metrics)

    assert summary["prefill_decode_fidelity"]["planned"] == 1
    assert summary["prefill_decode_fidelity"]["actual_same_dispatch"] == 0
    assert summary["prefill_decode_fidelity"]["missed"] == 1


def test_residual_dispatch_is_part_of_action_fidelity():
    prefill = _interval("prefill", 40, 0.0, 2.0)
    prefill["name"] = "prefill_residual_dispatch"
    intervals = [prefill, _interval("decode", 40, 1.0, 3.0)]
    metrics = [{"dispatch_index": 40, "global_action": "prefill_decode"}]

    summary = ANALYZER.summarize_activity(
        intervals, ANALYZER.activity_segments(intervals), metrics)

    assert summary["prefill_decode_fidelity"]["planned"] == 1
    assert summary["prefill_decode_fidelity"]["actual_same_dispatch"] == 1
    assert summary["prefill_decode_fidelity"][
        "same_dispatch_overlap_ms"] == 1.0
