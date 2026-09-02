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

from benchmarks.phase_serving.build_oracle_h1_snapshot_coverage import \
    build_coverage
from benchmarks.phase_serving.validate_phase_scheduler_events import \
    _snapshot_signature


def policy_events(policy: str, phase: str, action_id: int,
                  incremental_action_id: int, request_id: int) -> list[dict]:
    path = f"{policy}.log"
    decision = {
        "_policy":
        policy,
        "_source_path":
        path,
        "_source":
        f"{path}:1",
        "run_id":
        "run",
        "event_kind":
        "decision",
        "plan_id":
        1,
        "selected_action_id":
        action_id,
        "incremental_action_id":
        incremental_action_id,
        "action_kind":
        phase,
        "requested_action_direction":
        "idle_launch",
        "requested_start_skew_percent":
        0,
        "outstanding_before_mask":
        0,
        "ready": {
            "encoder_rows": 0,
            "prefill_rows": 1,
            "prefill_tokens": 128,
            "decode_rows": 1,
            "decode_context_tokens": 1024,
        },
        "ready_encoder_request_ids": [],
        "ready_prefill_request_ids": [1],
        "ready_prefill_token_counts": [128],
        "ready_decode_request_ids": [2],
        "ready_decode_context_lengths": [1024],
        "page_pool_allocated_bundles":
        2,
        "page_reservation_guaranteed_bundles":
        1,
        "vision_payload_bytes":
        0,
        "inflight": [],
        "request_ids": [request_id],
        "candidates": [{
            "action_id": action_id,
            "action_kind": phase,
            "legal": True,
            "predicted_completion_us": [5.0],
            "uncertainty_us": [1.0],
            "contextual_completion_valid": False,
        }],
    }
    decision["snapshot_signature"] = _snapshot_signature(decision)
    dispatch = {
        "_policy": policy,
        "_source_path": path,
        "_source": f"{path}:2",
        "run_id": "run",
        "event_kind": "dispatch",
        "plan_id": 1,
        "execution_id": 10,
        "phase": phase,
        "action_fidelity": True,
    }
    completion = {
        "_policy": policy,
        "_source_path": path,
        "_source": f"{path}:3",
        "run_id": "run",
        "event_kind": "completion",
        "plan_id": 1,
        "execution_id": 10,
        "phase": phase,
        "gpu_start_us": 100.0,
        "gpu_end_us": 105.0 if phase == "decode" else 110.0,
        "action_fidelity": True,
    }
    return [decision, dispatch, completion]


def test_joins_two_measured_actions_only_at_the_same_exact_snapshot() -> None:
    result = build_coverage({
        "myopic":
        policy_events("myopic", "prefill", 10, 100, 1),
        "h2":
        policy_events("h2", "decode", 20, 200, 2),
    })
    assert result["summary"]["snapshot_signatures"] == 1
    assert result["summary"]["exact_repeated_snapshots"] == 1
    assert result["summary"]["exact_multi_action_snapshots"] == 1
    assert result["summary"]["gate_b_candidate_coverage"]
    assert len(result["episodes"][0]["actions"]) == 2
    assert len(result["episodes"][0]["prediction_frontiers"]) == 2
    assert result["episodes"][0]["actions"][0]["candidate_action_id_stable"]
    assert result["episodes"][0]["actions"][0]["component_completion_us"]


def test_rejects_a_corrupted_exact_signature() -> None:
    events = policy_events("myopic", "prefill", 10, 100, 1)
    events[0]["snapshot_signature"] += 1
    result = build_coverage({"myopic": events})
    assert result["summary"]["snapshot_signatures"] == 0
    assert any("invalid snapshot signature" in error
               for error in result["errors"])
