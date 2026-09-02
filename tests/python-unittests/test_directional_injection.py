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
"""Unit tests for the M2 directional-injection completion-vector analyzer."""

from benchmarks.phase_serving.analyze_directional_injection import (
    build_samples, summarize)
from benchmarks.phase_serving.validate_phase_scheduler_events import (
    _snapshot_signature, _validate)


def completion(run_id: str, phase: str, execution_id: int, start_us: float,
               end_us: float) -> dict:
    """Return one minimal common-epoch completion record."""
    return {
        "event_kind": "completion",
        "run_id": run_id,
        "phase": phase,
        "execution_id": execution_id,
        "gpu_start_us": start_us,
        "gpu_end_us": end_us,
        "gpu_duration_us": end_us - start_us,
        "completion_visible_host_ns": round(end_us * 1000.0),
        "action_fidelity": True,
        "_source": f"{run_id}-{phase}",
    }


def dispatch(run_id: str, target: float, incumbent_id: int,
             newcomer_id: int) -> dict:
    """Return one minimal newcomer dispatch with M2 injection metadata."""
    return {
        "event_kind": "dispatch",
        "run_id": run_id,
        "phase": "decode",
        "execution_id": newcomer_id,
        "action_direction": "prefill_to_decode",
        "incumbent_phase": "prefill",
        "incumbent_execution_id": incumbent_id,
        "injection_target_fraction": target,
        "injection_requested_direction": "prefill_to_decode",
        "injection_incumbent_reference_us": 100.0,
        "injection_newcomer_reference_us": 40.0,
        "requested_injection_delay_us": round(target * 100.0),
        "action_fidelity": True,
        "_source": f"{run_id}-dispatch",
    }


def test_uses_actual_gpu_offset_and_builds_completion_vector() -> None:
    events = [
        dispatch("zero", 0.0, 1, 2),
        completion("zero", "prefill", 1, 10.0, 110.0),
        completion("zero", "decode", 2, 10.0, 54.0),
        dispatch("half", 0.5, 3, 4),
        completion("half", "prefill", 3, 200.0, 320.0),
        completion("half", "decode", 4, 250.0, 290.0),
    ]
    samples, errors = build_samples(events, bucket_tolerance=0.12)

    assert not errors
    assert len(samples) == 2
    assert samples[0]["actual_fraction"] == 0.0
    assert samples[1]["actual_fraction"] == 0.5
    assert samples[1]["incumbent_slowdown_us"] == 20.0
    assert samples[1]["overlap_us"] == 40.0
    assert samples[1]["realization"] == "overlap"
    assert samples[1]["actual_start_skew_bucket"] == "offset_50"

    artifact = summarize(samples, material_effect_ratio=0.03)
    gate = next(item for item in artifact["gate_a"]["directions"]
                if item["direction"] == "prefill_to_decode")
    assert gate["accepted_buckets"] == 2
    assert gate["material_inflight_effect"]


def test_measured_gpu_order_overrides_requested_direction() -> None:
    newcomer = dispatch("reverse", 0.25, 1, 2)
    events = [
        newcomer,
        completion("reverse", "prefill", 1, 30.0, 130.0),
        completion("reverse", "decode", 2, 10.0, 50.0),
    ]
    samples, errors = build_samples(events, bucket_tolerance=0.13)

    assert not errors
    assert len(samples) == 1
    assert samples[0]["requested_direction"] == "prefill_to_decode"
    assert samples[0]["direction"] == "decode_to_prefill"
    assert samples[0]["incumbent_reference_us"] == 40.0
    assert samples[0]["newcomer_reference_us"] == 100.0


def test_rejects_a_bucket_without_measured_gpu_overlap() -> None:
    events = [
        dispatch("serial", 0.5, 1, 2),
        completion("serial", "prefill", 1, 0.0, 40.0),
        completion("serial", "decode", 2, 50.0, 90.0),
    ]
    samples, errors = build_samples(events, bucket_tolerance=0.13)

    assert not errors
    assert len(samples) == 1
    assert not samples[0]["inflight_overlap"]
    assert not samples[0]["accepted_bucket"]
    assert samples[0]["realization"] == "serial_realization"
    assert samples[0]["actual_start_skew_bucket"] == "serial_realization"


def test_reports_causal_offset_floor_and_graph_submission_timing() -> None:
    newcomer = dispatch("late", 0.25, 1, 2)
    newcomer.update({
        "plan_id": 7,
        "enqueue_host_ns": 1_500_000,
        "prepare_start_host_ns": 1_510_000,
        "prepare_end_host_ns": 1_610_000,
        "execute_start_host_ns": 1_620_000,
        "execute_end_host_ns": 1_650_000,
        "graph_replay": True,
    })
    decision = {
        "event_kind": "decision",
        "run_id": "late",
        "plan_id": 7,
        "host_monotonic_ns": 1_400_000,
        "inflight": [{
            "execution_id": 1,
            "dispatch_age_us": 55.0
        }],
    }
    events = [
        decision,
        newcomer,
        completion("late", "prefill", 1, 10.0, 110.0),
        completion("late", "decode", 2, 65.0, 105.0),
    ]

    samples, errors = build_samples(events, bucket_tolerance=0.12)

    assert not errors
    assert len(samples) == 1
    assert not samples[0]["target_causally_reachable"]
    assert samples[0]["target_late_by_us"] == 30.0
    assert samples[0]["decision_to_enqueue_us"] == 100.0
    assert samples[0]["prepare_host_us"] == 100.0
    assert samples[0]["execute_submit_host_us"] == 30.0
    assert samples[0]["graph_replay"]


def incremental_events() -> list[dict]:
    """Return one complete M3 decision/dispatch/completion chain."""
    common = {
        "schema_version": 1,
        "run_id": "m3",
        "incremental_action_id": 99,
        "requested_action_direction": "idle_launch",
        "requested_start_skew_percent": 0,
        "_source_path": "m3.log",
    }
    decision = {
        **common,
        "event_kind": "decision",
        "event_id": 1,
        "host_monotonic_ns": 10,
        "decision_id": 1,
        "snapshot_id": 1,
        "plan_id": 1,
        "action_id": 10,
        "candidates": [{
            "action_id": 10,
            "request_ids": [1]
        }],
        "request_ids": [1],
        "selected_action_id": 10,
        "outstanding_before_mask": 0,
        "planned_outstanding_mask": 2,
        "_source": "m3.log:1",
    }
    dispatch_event = {
        **common,
        "event_kind": "dispatch",
        "event_id": 2,
        "host_monotonic_ns": 20,
        "decision_id": 1,
        "snapshot_id": 1,
        "execution_id": 1,
        "plan_id": 1,
        "action_id": 10,
        "phase": "prefill",
        "request_ids": [1],
        "action_fidelity": True,
        "action_direction": "idle_launch",
        "outstanding_before_mask": 0,
        "planned_outstanding_mask": 2,
        "enqueue_host_ns": 20,
        "cohort": {
            "prefill_rows": 1
        },
        "_source": "m3.log:2",
    }
    completion_event = {
        **common,
        "event_kind": "completion",
        "event_id": 3,
        "host_monotonic_ns": 40,
        "decision_id": 1,
        "snapshot_id": 1,
        "execution_id": 1,
        "plan_id": 1,
        "action_id": 10,
        "phase": "prefill",
        "request_ids": [1],
        "action_fidelity": True,
        "observed_outstanding_mask": 0,
        "gpu_start_us": 1.0,
        "gpu_end_us": 11.0,
        "gpu_duration_us": 10.0,
        "completion_visible_host_ns": 40,
        "completion_status": "success",
        "cohort": {
            "prefill_rows": 1
        },
        "_source": "m3.log:3",
    }
    return [decision, dispatch_event, completion_event]


def test_validates_incremental_identity_across_event_chain() -> None:
    events = incremental_events()
    errors, summary = _validate(events,
                                require_fidelity=True,
                                require_gpu_intervals=True)

    assert not errors
    assert summary["executions"] == 1


def test_accepts_matching_compact_decision_when_enabled() -> None:
    events = incremental_events()
    events[0]["candidates"] = []

    errors, summary = _validate(events,
                                require_fidelity=True,
                                require_gpu_intervals=True,
                                allow_compact_decisions=True)

    assert not errors
    assert summary["decisions"] == 1


def test_rejects_mismatched_compact_decision_identity() -> None:
    events = incremental_events()
    events[0]["candidates"] = []
    events[0]["selected_action_id"] = 11

    errors, _ = _validate(events,
                          require_fidelity=True,
                          require_gpu_intervals=True,
                          allow_compact_decisions=True)

    assert any("compact decision selected action" in error for error in errors)


def test_rejects_incremental_identity_change_and_triple_mask() -> None:
    events = incremental_events()
    events[-1]["incremental_action_id"] = 100
    events[0]["planned_outstanding_mask"] = 7
    errors, _ = _validate(events,
                          require_fidelity=True,
                          require_gpu_intervals=True)

    assert any("incremental action identity changed" in error
               for error in errors)
    assert any("illegally contains E+P+D" in error for error in errors)


def test_validates_cross_run_snapshot_signature() -> None:
    events = incremental_events()
    decision = events[0]
    decision.update({
        "ready": {
            "encoder_rows": 0,
            "prefill_rows": 1,
            "prefill_tokens": 128,
            "decode_rows": 0,
            "decode_context_tokens": 0,
        },
        "inflight": [],
        "ready_encoder_request_ids": [],
        "ready_prefill_request_ids": [1],
        "ready_prefill_token_counts": [128],
        "ready_decode_request_ids": [],
        "ready_decode_context_lengths": [],
        "page_pool_allocated_bundles": 1,
        "page_reservation_guaranteed_bundles": 1,
        "vision_payload_bytes": 0,
    })
    decision["snapshot_signature"] = _snapshot_signature(decision)
    errors, _ = _validate(events,
                          require_fidelity=True,
                          require_gpu_intervals=True)
    assert not errors

    decision["ready_prefill_request_ids"] = [2]
    errors, _ = _validate(events,
                          require_fidelity=True,
                          require_gpu_intervals=True)
    assert any("snapshot_signature does not match" in error
               for error in errors)

    decision["ready_prefill_request_ids"] = [1]
    decision["ready_prefill_token_counts"] = [64]
    errors, _ = _validate(events,
                          require_fidelity=True,
                          require_gpu_intervals=True)
    assert any("snapshot_signature does not match" in error
               for error in errors)
