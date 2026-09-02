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

from benchmarks.phase_serving.analyze_oracle_h1_snapshot_coverage import \
    analyze


def action(action_id: int, kind: str, phase: str,
           boundaries: list[float]) -> dict:
    return {
        "incremental_action_id": action_id,
        "action_kind": kind,
        "completed_phase": [phase] * len(boundaries),
        "projected_boundary_us": boundaries,
        "whole_action_completion_us": [value + 1.0 for value in boundaries],
    }


def test_ranks_only_same_phase_exact_alternatives_and_marks_sample_quality(
) -> None:
    result = analyze({
        "summary": {
            "exact_multi_action_snapshots": 1
        },
        "episodes": [{
            "signature_quality":
            "exact_v1",
            "snapshot_signature":
            7,
            "actions": [
                action(10, "decode", "decode", [5.0, 7.0]),
                action(20, "prefill_decode", "decode", [12.0]),
            ],
            "policy_selections": {
                "myopic": [10, 10],
                "probe": [20]
            },
        }],
    })
    assert result["summary"]["comparable_h1_snapshots"] == 1
    assert result["summary"]["fully_repeated_h1_snapshots"] == 0
    assert result["summary"]["pilot_regret_evaluable"]
    assert not result["summary"]["promotion_quality_coverage"]
    assert result["comparisons"]["myopic"]["agreements"] == 2
    assert result["comparisons"]["myopic"]["selection_observations"] == 2
    assert result["comparisons"]["probe"][
        "empirical_boundary_regret_us"] == 6.0


def test_retains_history_sensitive_choices_at_the_same_execution_snapshot(
) -> None:
    result = analyze({
        "summary": {
            "exact_multi_action_snapshots": 1
        },
        "episodes": [{
            "signature_quality":
            "exact_v1",
            "snapshot_signature":
            8,
            "actions": [
                action(10, "encoder", "encoder", [5.0]),
                action(20, "encoder_prefill", "encoder", [7.0]),
            ],
            "policy_selections": {
                "online": [10, 20]
            },
        }],
    })
    comparison = result["comparisons"]["online"]
    assert comparison["selection_observations"] == 2
    assert comparison["agreements"] == 1
    assert comparison["history_sensitive_snapshots"] == 1
    assert result["episodes"][0]["history_sensitive_policy_selections"] == [
        "online"
    ]


def test_skips_actions_with_different_first_completion_phases() -> None:
    result = analyze({
        "summary": {
            "exact_multi_action_snapshots": 1
        },
        "episodes": [{
            "signature_quality":
            "exact_v1",
            "snapshot_signature":
            9,
            "actions": [
                action(10, "encoder", "encoder", [5.0]),
                action(20, "prefill", "prefill", [4.0]),
            ],
            "policy_selections": {},
        }],
    })
    assert result["summary"]["comparable_h1_snapshots"] == 0
    assert result["summary"]["skipped_h1_snapshots"] == 1


def test_ranks_pair_against_serial_at_their_unique_common_phase() -> None:
    serial = action(10, "prefill", "prefill", [10.0])
    serial["candidate_action_id"] = 100
    serial["component_completion_us"] = {"prefill": [10.0]}
    # Even if P happens to finish first in both actions, P+D advances a wider
    # work frontier. Compare the common P milestone instead of allowing a
    # predicted D completion to stand in for P progress.
    overlap = action(20, "prefill_decode", "prefill", [12.0])
    overlap["candidate_action_id"] = 200
    overlap["component_completion_us"] = {
        "decode": [7.0],
        "prefill": [12.0],
    }
    result = analyze({
        "summary": {
            "exact_multi_action_snapshots": 1
        },
        "episodes": [{
            "signature_quality":
            "exact_v1",
            "snapshot_signature":
            13,
            "actions": [serial, overlap],
            "policy_selections": {
                "serial": [10],
                "probe": [20],
            },
            "prediction_frontiers": [{
                "policy":
                "m6_shadow",
                "candidates": [{
                    "action_id": 100,
                    "legal": True,
                    "predicted_completion_us": [9.0],
                    "uncertainty_us": [0.5],
                    "contextual_completion_valid": False,
                }, {
                    "action_id": 200,
                    "legal": True,
                    "contextual_completion_valid": True,
                    "contextual_completion_ready": True,
                    "contextual_direction": "prefill_to_decode",
                    "contextual_incumbent_mean_us": 11.0,
                    "contextual_incumbent_uncertainty_us": 0.5,
                    "contextual_newcomer_mean_us": 6.0,
                    "contextual_newcomer_uncertainty_us": 0.5,
                }],
            }],
        }],
    })

    assert result["summary"]["comparable_h1_snapshots"] == 1
    assert result["summary"]["common_phase_h1_snapshots"] == 1
    assert result["episodes"][0]["evaluation_mode"] == \
        "common_phase_completion"
    assert result["episodes"][0]["completed_phase"] == "prefill"
    assert result["comparisons"]["serial"]["agreements"] == 1
    assert result["comparisons"]["probe"]["agreements"] == 0
    assert result["contextual_ranking"]["evaluable_frontiers"] == 1
    assert result["contextual_ranking"]["top1_agreements"] == 1
    assert result["contextual_ranking_records"][0]["target_phase"] == "prefill"


def test_skips_snapshot_when_one_action_has_an_unstable_h1_phase() -> None:
    unstable = action(10, "encoder_decode", "encoder", [5.0, 7.0])
    unstable["completed_phase"] = ["encoder", "decode"]
    result = analyze({
        "summary": {
            "exact_multi_action_snapshots": 1
        },
        "episodes": [{
            "signature_quality":
            "exact_v1",
            "snapshot_signature":
            10,
            "actions": [unstable,
                        action(20, "decode", "decode", [6.0])],
            "policy_selections": {
                "probe": [10]
            },
        }],
    })
    assert result["summary"]["comparable_h1_snapshots"] == 0
    assert result["summary"]["unstable_action_h1_snapshots"] == 1
    assert result["skipped"][0]["unstable_actions"] == [{
        "incremental_action_id":
        10,
        "action_kind":
        "encoder_decode",
        "completed_phase_counts": {
            "decode": 1,
            "encoder": 1,
        },
    }]


def test_contextual_frontier_reports_top1_and_normalized_h1_regret() -> None:
    fast = action(10, "decode", "decode", [5.0, 7.0])
    fast["candidate_action_id"] = 100
    slow = action(20, "prefill_decode", "decode", [11.0, 13.0])
    slow["candidate_action_id"] = 200
    result = analyze({
        "summary": {
            "exact_multi_action_snapshots": 1
        },
        "episodes": [{
            "signature_quality":
            "exact_v1",
            "snapshot_signature":
            11,
            "actions": [fast, slow],
            "policy_selections": {},
            "prediction_frontiers": [{
                "policy":
                "m6_shadow",
                "candidates": [{
                    "action_id": 100,
                    "legal": True,
                    "predicted_completion_us": [8.0],
                    "uncertainty_us": [1.0],
                    "contextual_completion_valid": False,
                }, {
                    "action_id": 200,
                    "legal": True,
                    "contextual_completion_valid": True,
                    "contextual_completion_ready": True,
                    "contextual_incumbent_mean_us": 4.0,
                    "contextual_incumbent_uncertainty_us": 0.5,
                    "contextual_newcomer_mean_us": 15.0,
                    "contextual_newcomer_uncertainty_us": 1.0,
                }],
            }],
        }],
    })

    ranking = result["contextual_ranking"]
    assert ranking["frontiers"] == 1
    assert ranking["evaluable_frontiers"] == 1
    assert ranking["top1_agreements"] == 0
    assert ranking["mean_normalized_h1_regret"] == 1.0
    assert result["summary"]["contextual_ranking_regret_evaluable"]
    assert result["contextual_ranking_records"][0][
        "predicted_action_kind"] == "prefill_decode"


def test_contextual_frontier_requires_ready_predictions_for_pair_actions(
) -> None:
    serial = action(10, "decode", "decode", [5.0])
    serial["candidate_action_id"] = 100
    overlap = action(20, "prefill_decode", "decode", [7.0])
    overlap["candidate_action_id"] = 200
    result = analyze({
        "summary": {
            "exact_multi_action_snapshots": 1
        },
        "episodes": [{
            "signature_quality":
            "exact_v1",
            "snapshot_signature":
            12,
            "actions": [serial, overlap],
            "policy_selections": {},
            "prediction_frontiers": [{
                "policy":
                "cold",
                "candidates": [{
                    "action_id": 100,
                    "legal": True,
                    "predicted_completion_us": [5.0],
                    "contextual_completion_valid": False,
                }, {
                    "action_id": 200,
                    "legal": True,
                    "contextual_completion_valid": True,
                    "contextual_completion_ready": False,
                }],
            }],
        }],
    })

    assert result["contextual_ranking"]["frontiers"] == 1
    assert result["contextual_ranking"]["evaluable_frontiers"] == 0
    assert not result["summary"]["contextual_ranking_regret_evaluable"]
