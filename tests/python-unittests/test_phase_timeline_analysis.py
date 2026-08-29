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

ANALYZER_PATH = Path(__file__).parents[2] / "benchmarks" / "phase_serving" / "analyze_phase_timeline.py"
SPEC = importlib.util.spec_from_file_location("phase_timeline_analysis", ANALYZER_PATH)
ANALYZER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(ANALYZER)


def _event(request_id, stage, timestamp_us, dispatch_index=0, batch_size=0, source="gateway.log"):
    return {
        "source": source,
        "request_index": request_id,
        "stage": stage,
        "timestamp_us": timestamp_us,
        "dispatch_index": dispatch_index,
        "batch_size": batch_size,
        "kv_slot_id": request_id,
    }


def _text_events(request_id=1):
    return [
        _event(request_id, "server_submit", 0),
        _event(request_id, "server_admit", 100),
        _event(request_id, "prefill_start", 200, 10, 2),
        _event(request_id, "prefill_done", 1200, 10, 2),
        _event(request_id, "first_token", 1300),
        _event(request_id, "decode_start", 1500, 11, 4),
        _event(request_id, "decode_done", 2000, 11, 4),
        _event(request_id, "decode_start", 2200, 12, 4),
        _event(request_id, "decode_done", 2700, 12, 4),
        _event(request_id, "completion", 3000),
    ]


def test_text_request_attribution_partitions_ttft():
    row = ANALYZER.attribute_requests(_text_events())[0]

    assert row["request_class"] == "text"
    assert row["complete"]
    assert row["prefill_admission_ms"] == pytest.approx(0.1)
    assert row["prefill_initial_queue_ms"] == pytest.approx(0.1)
    assert row["prefill_active_ms"] == pytest.approx(1.0)
    assert row["prefill_sampling_ms"] == pytest.approx(0.1)
    assert row["first_token_to_decode_ms"] == pytest.approx(0.2)
    assert row["first_decode_active_ms"] == pytest.approx(0.5)
    assert row["decode_inter_dispatch_gap_ms"] == pytest.approx(0.2)
    assert row["decode_tail_ms"] == pytest.approx(0.3)
    assert row["backend_ttft_ms"] == pytest.approx(1.3)
    assert row["backend_e2e_ms"] == pytest.approx(3.0)
    assert row["decode_total_active_ms"] == pytest.approx(1.0)
    assert row["mean_prefill_batch_size"] == pytest.approx(2.0)
    assert row["mean_decode_batch_size"] == pytest.approx(4.0)
    assert row["critical_path_residual_ms"] == pytest.approx(0.0)
    assert row["e2e_residual_ms"] == pytest.approx(0.0)


def test_vision_request_attribution_includes_encoder_and_handoff():
    records = [
        _event(2, "vision_queued", 0),
        _event(2, "encoder_start", 1000, batch_size=4),
        _event(2, "encoder_done", 4000, batch_size=4),
        _event(2, "prefill_ready", 4200),
        _event(2, "server_submit", 5000),
        _event(2, "server_admit", 5100),
        _event(2, "prefill_release", 5200),
        _event(2, "prefill_start", 6000, 20, 2),
        _event(2, "prefill_done", 8000, 20, 2),
        _event(2, "first_token", 8200),
        _event(2, "decode_start", 8500, 21, 8),
        _event(2, "decode_done", 9000, 21, 8),
        _event(2, "completion", 10000),
    ]

    row = ANALYZER.attribute_requests(records)[0]

    assert row["request_class"] == "vision"
    assert row["encoder_queue_ms"] == pytest.approx(1.0)
    assert row["encoder_active_ms"] == pytest.approx(3.0)
    assert row["encoder_to_prefill_ready_ms"] == pytest.approx(0.2)
    assert row["vision_ready_queue_ms"] == pytest.approx(0.8)
    assert row["critical_path_residual_ms"] == pytest.approx(0.0)


def test_chunked_prefill_pairs_dispatches_and_attributes_gap():
    records = _text_events(3)
    records = [event for event in records if event["stage"] not in ("prefill_start", "prefill_done")]
    records.extend([
        _event(3, "prefill_start", 200, 30, 2),
        _event(3, "prefill_done", 700, 30, 2),
        _event(3, "prefill_start", 900, 31, 2),
        _event(3, "prefill_done", 1200, 31, 2),
    ])

    row = ANALYZER.attribute_requests(records)[0]

    assert row["prefill_dispatches"] == 2
    assert row["prefill_active_ms"] == pytest.approx(0.8)
    assert row["prefill_chunk_gap_ms"] == pytest.approx(0.2)
    assert row["prefill_total_span_ms"] == pytest.approx(1.0)
    assert row["critical_path_residual_ms"] == pytest.approx(0.0)


def test_parser_filters_warmup_and_summary_reports_incomplete(tmp_path):
    log_path = tmp_path / "gateway.log"
    events = _text_events(4) + [_event(1_000_000, "server_submit", 0)]
    lines = [f"noise PHASE_TIMELINE\t{json.dumps(event)}\n" for event in events]
    lines.append(f"PHASE_TIMELINE\t{json.dumps(_event(5, 'server_submit', 0))}\n")
    log_path.write_text("".join(lines), encoding="utf-8")

    records = ANALYZER.parse_timeline_logs([log_path])
    rows = ANALYZER.attribute_requests(records)
    summary = ANALYZER.summarize_attributions(rows)

    assert {row["request_index"] for row in rows} == {4, 5}
    assert summary["groups"]["all"]["requests"] == 2
    assert summary["groups"]["all"]["complete_requests"] == 1
    assert summary["groups"]["all"]["incomplete_request_ids"] == [5]
    assert summary["groups"]["text"]["metrics_ms"]["backend_ttft_ms"]["mean"] == pytest.approx(1.3)


def test_latest_reused_request_lifecycle_replaces_http_warmup():
    warmup = _text_events(0)
    measured = []
    for event in _text_events(0):
        shifted = dict(event)
        shifted["timestamp_us"] += 10_000
        measured.append(shifted)
    warmup_only = _text_events(1)

    rows = ANALYZER.attribute_requests(warmup + measured + warmup_only, request_count=1)

    assert len(rows) == 1
    assert rows[0]["observed_lifecycles"] == 2
    assert rows[0]["backend_ttft_ms"] == pytest.approx(1.3)
    assert rows[0]["critical_path_residual_ms"] == pytest.approx(0.0)


def test_client_metrics_add_frontend_and_completion_intervals(tmp_path):
    rows = ANALYZER.attribute_requests(_text_events(0), request_count=1)
    client_path = tmp_path / "requests.csv"
    client_path.write_text(
        "request_id,request_class,ttft_ms,tpot_ms,e2e_ms\n"
        "0,text,1.8,0.4,3.7\n",
        encoding="utf-8",
    )

    ANALYZER.join_client_metrics(rows, {"gateway.log": client_path})
    summary = ANALYZER.summarize_attributions(rows)

    assert rows[0]["frontend_to_backend_ttft_ms"] == pytest.approx(0.5)
    assert rows[0]["backend_to_client_completion_ms"] == pytest.approx(0.7)
    assert summary["groups"]["all"]["tail"]["selection_metric"] == "client_ttft_ms"
