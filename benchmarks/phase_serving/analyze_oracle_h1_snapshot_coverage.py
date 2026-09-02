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
"""Compare empirical H1 boundaries at exact cross-policy snapshots."""

from __future__ import annotations

import argparse
import collections
import json
import statistics
import sys
from pathlib import Path
from typing import Any


def _action_summary(action: dict[str, Any]) -> dict[str, Any]:
    boundaries = [
        float(value) for value in action.get("projected_boundary_us", [])
    ]
    completions = [
        float(value) for value in action.get("whole_action_completion_us", [])
    ]
    phase_counts = collections.Counter(
        str(value) for value in action.get("completed_phase", []))
    if not boundaries or len(boundaries) != len(completions):
        raise ValueError("action has incomplete measured boundary samples")
    component_completion_us = {
        str(phase): [float(value) for value in values]
        for phase, values in action.get("component_completion_us", {}).items()
        if values
    }
    return {
        "incremental_action_id":
        int(action["incremental_action_id"]),
        "candidate_action_id":
        int(action.get("candidate_action_id",
                       action["incremental_action_id"])),
        "candidate_action_id_stable":
        bool(action.get("candidate_action_id_stable", True)),
        "action_kind":
        str(action["action_kind"]),
        "completed_phase":
        next(iter(phase_counts)) if len(phase_counts) == 1 else None,
        "completed_phase_counts":
        dict(sorted(phase_counts.items())),
        "stable_h1_phase":
        len(phase_counts) == 1,
        "samples":
        len(boundaries),
        "median_boundary_us":
        statistics.median(boundaries),
        "min_boundary_us":
        min(boundaries),
        "max_boundary_us":
        max(boundaries),
        "median_whole_action_completion_us":
        statistics.median(completions),
        "component_completion_median_us": {
            phase: statistics.median(values)
            for phase, values in component_completion_us.items()
        },
    }


def _scalar_prediction(candidate: dict[str, Any]) -> float | None:
    values = candidate.get("predicted_completion_us", [])
    uncertainties = candidate.get("uncertainty_us", [])
    if not values:
        return None
    mean = max(0.0, float(values[0]))
    uncertainty = max(0.0, float(uncertainties[0])) if uncertainties else 0.0
    return mean + uncertainty


def _contextual_prediction(candidate: dict[str, Any],
                           target_phase: str | None = None) -> float | None:
    if not bool(candidate.get("contextual_completion_valid", False)):
        return _scalar_prediction(candidate)
    if not bool(candidate.get("contextual_completion_ready", False)):
        return None
    incumbent = max(0.0, float(candidate.get("contextual_incumbent_mean_us", 0.0))) \
        + max(0.0, float(candidate.get("contextual_incumbent_uncertainty_us", 0.0)))
    newcomer = max(0.0, float(candidate.get("contextual_newcomer_mean_us", 0.0))) \
        + max(0.0, float(candidate.get("contextual_newcomer_uncertainty_us", 0.0)))
    if target_phase is None:
        return min(incumbent, newcomer)
    direction = str(candidate.get("contextual_direction", ""))
    components = direction.split("_to_", maxsplit=1)
    if len(components) != 2:
        return None
    if target_phase == components[0]:
        return incumbent
    if target_phase == components[1]:
        return newcomer
    return None


def _empty_ranking() -> dict[str, int | float]:
    return {
        "frontiers": 0,
        "evaluable_frontiers": 0,
        "top1_agreements": 0,
        "normalized_h1_regret_sum": 0.0,
        "normalized_h1_regret_max": 0.0,
    }


def _finalize_ranking(values: dict[str, int | float]) -> None:
    evaluable = int(values["evaluable_frontiers"])
    values["top1_agreement_ratio"] = (int(values["top1_agreements"]) /
                                      evaluable if evaluable else 0.0)
    values["mean_normalized_h1_regret"] = (
        float(values["normalized_h1_regret_sum"]) /
        evaluable if evaluable else 0.0)


def analyze(payload: dict[str, Any]) -> dict[str, Any]:
    """Build a conservative empirical boundary oracle from exact snapshots."""
    records: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    comparisons: dict[str, dict[str, Any]] = {}
    contextual_ranking = _empty_ranking()
    contextual_by_policy: dict[str, dict[str, int | float]] = {}
    contextual_ranking_records: list[dict[str, Any]] = []
    common_phase_snapshots = 0
    for episode in payload.get("episodes", []):
        if episode.get("signature_quality") != "exact_v1" or len(
                episode.get("actions", [])) < 2:
            continue
        actions = [_action_summary(action) for action in episode["actions"]]
        unstable_actions = [
            action for action in actions if not action["stable_h1_phase"]
            or not action["candidate_action_id_stable"]
        ]
        if unstable_actions:
            skipped.append({
                "snapshot_signature":
                int(episode["snapshot_signature"]),
                "reason":
                "one or more actions complete different H1 phases across samples",
                "unstable_actions": [{
                    "incremental_action_id":
                    action["incremental_action_id"],
                    "action_kind":
                    action["action_kind"],
                    "completed_phase_counts":
                    action["completed_phase_counts"],
                } for action in unstable_actions],
            })
            continue
        completed_phases = {action["completed_phase"] for action in actions}
        evaluation_mode = "first_completion"
        target_phase: str | None = None
        component_phase_sets = [
            set(action["component_completion_median_us"]) for action in actions
        ]
        different_work_frontiers = all(component_phase_sets) and any(
            phases != component_phase_sets[0]
            for phases in component_phase_sets[1:])
        if len(completed_phases) == 1 and not different_work_frontiers:
            target_phase = next(iter(completed_phases))
            for action in actions:
                action["objective_median_boundary_us"] = action[
                    "median_boundary_us"]
        else:
            common_phases = set.intersection(*component_phase_sets) \
                if all(component_phase_sets) else set()
            if len(common_phases) != 1:
                skipped.append({
                    "snapshot_signature":
                    int(episode["snapshot_signature"]),
                    "reason":
                    "alternatives have neither the same first completion phase "
                    "nor one unambiguous common phase milestone",
                    "common_phases":
                    sorted(common_phases),
                })
                continue
            target_phase = next(iter(common_phases))
            evaluation_mode = "common_phase_completion"
            common_phase_snapshots += 1
            for action in actions:
                action["objective_median_boundary_us"] = action[
                    "component_completion_median_us"][target_phase]
        oracle = min(actions,
                     key=lambda action:
                     (action["objective_median_boundary_us"], action[
                         "incremental_action_id"]))
        action_by_id = {
            action["incremental_action_id"]: action
            for action in actions
        }
        action_by_candidate_id = {
            action["candidate_action_id"]: action
            for action in actions
        }
        selections: dict[str, list[int]] = {}
        history_sensitive_policies: list[str] = []
        for policy, selected_ids in episode.get("policy_selections",
                                                {}).items():
            observed = [int(value) for value in selected_ids]
            if any(selected_id not in action_by_id
                   for selected_id in observed):
                raise ValueError(
                    f"policy {policy} selected an unknown measured action")
            selections[str(policy)] = observed
            if len(set(observed)) > 1:
                history_sensitive_policies.append(str(policy))
            comparison = comparisons.setdefault(
                str(policy), {
                    "exact_snapshots": 0,
                    "selection_observations": 0,
                    "agreements": 0,
                    "empirical_boundary_regret_us": 0.0,
                    "history_sensitive_snapshots": 0,
                })
            comparison["exact_snapshots"] += 1
            comparison["selection_observations"] += len(observed)
            comparison["history_sensitive_snapshots"] += int(
                len(set(observed)) > 1)
            for selected_id in observed:
                selected = action_by_id[selected_id]
                comparison["agreements"] += int(
                    selected_id == oracle["incremental_action_id"])
                comparison["empirical_boundary_regret_us"] += max(
                    0.0, selected["objective_median_boundary_us"] -
                    oracle["objective_median_boundary_us"])
        for frontier in episode.get("prediction_frontiers", []):
            policy = str(frontier.get("policy", "unknown"))
            policy_ranking = contextual_by_policy.setdefault(
                policy, _empty_ranking())
            contextual_ranking["frontiers"] += 1
            policy_ranking["frontiers"] += 1
            candidates = {
                int(candidate["action_id"]): candidate
                for candidate in frontier.get("candidates", [])
                if bool(candidate.get("legal", False))
            }
            if not action_by_candidate_id.keys() <= candidates.keys():
                continue
            predictions = {
                action_id:
                _contextual_prediction(
                    candidates[action_id], target_phase
                    if evaluation_mode == "common_phase_completion" else None)
                for action_id in action_by_candidate_id
            }
            if any(value is None for value in predictions.values()):
                continue
            predicted_action_id = min(
                predictions,
                key=lambda action_id:
                (float(predictions[action_id]), action_id))
            predicted = action_by_candidate_id[predicted_action_id]
            regret = max(
                0.0, predicted["objective_median_boundary_us"] -
                oracle["objective_median_boundary_us"]) / max(
                    oracle["objective_median_boundary_us"],
                    sys.float_info.epsilon)
            agreement = predicted_action_id == oracle["candidate_action_id"]
            contextual_ranking_records.append({
                "snapshot_signature":
                int(episode["snapshot_signature"]),
                "policy":
                policy,
                "evaluation_mode":
                evaluation_mode,
                "target_phase":
                target_phase,
                "oracle_candidate_action_id":
                oracle["candidate_action_id"],
                "oracle_action_kind":
                oracle["action_kind"],
                "predicted_candidate_action_id":
                predicted_action_id,
                "predicted_action_kind":
                predicted["action_kind"],
                "top1_agreement":
                agreement,
                "normalized_h1_regret":
                regret,
                "predictions_us": {
                    str(action_id): float(value)
                    for action_id, value in sorted(predictions.items())
                },
                "measured_objective_us": {
                    str(action_id):
                    action_by_candidate_id[action_id]
                    ["objective_median_boundary_us"]
                    for action_id in sorted(action_by_candidate_id)
                },
            })
            for target in (contextual_ranking, policy_ranking):
                target["evaluable_frontiers"] += 1
                target["top1_agreements"] += int(agreement)
                target["normalized_h1_regret_sum"] += regret
                target["normalized_h1_regret_max"] = max(
                    float(target["normalized_h1_regret_max"]), regret)
        records.append({
            "snapshot_signature":
            int(episode["snapshot_signature"]),
            "completed_phase":
            target_phase,
            "evaluation_mode":
            evaluation_mode,
            "oracle_incremental_action_id":
            oracle["incremental_action_id"],
            "oracle_action_kind":
            oracle["action_kind"],
            "oracle_median_boundary_us":
            oracle["objective_median_boundary_us"],
            "fully_repeated_alternatives":
            all(action["samples"] >= 2 for action in actions),
            "actions":
            actions,
            "policy_selections":
            selections,
            "history_sensitive_policy_selections":
            sorted(history_sensitive_policies),
        })

    for comparison in comparisons.values():
        observations = int(comparison["selection_observations"])
        comparison["agreement_ratio"] = comparison[
            "agreements"] / observations if observations else 0.0
        comparison["mean_empirical_boundary_regret_us"] = (
            comparison["empirical_boundary_regret_us"] /
            observations if observations else 0.0)
    _finalize_ranking(contextual_ranking)
    for values in contextual_by_policy.values():
        _finalize_ranking(values)
    fully_repeated = sum(
        bool(record["fully_repeated_alternatives"]) for record in records)
    summary = {
        "exact_multi_action_snapshots":
        int(payload.get("summary", {}).get("exact_multi_action_snapshots", 0)),
        "comparable_h1_snapshots":
        len(records),
        "skipped_h1_snapshots":
        len(skipped),
        "unstable_action_h1_snapshots":
        sum("unstable_actions" in item for item in skipped),
        "fully_repeated_h1_snapshots":
        fully_repeated,
        "common_phase_h1_snapshots":
        common_phase_snapshots,
        "pilot_regret_evaluable":
        bool(records),
        "promotion_quality_coverage":
        bool(records) and fully_repeated == len(records),
        "contextual_ranking_regret_evaluable":
        int(contextual_ranking["evaluable_frontiers"]) > 0,
    }
    return {
        "schema_version": 2,
        "artifact": "oracle_h1_empirical_snapshot_analysis",
        "objective": "minimum median same-phase H1 completion boundary",
        "summary": summary,
        "comparisons": comparisons,
        "contextual_ranking": contextual_ranking,
        "contextual_ranking_by_policy": contextual_by_policy,
        "contextual_ranking_records": contextual_ranking_records,
        "episodes": records,
        "skipped": skipped,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = analyze(json.loads(args.input.read_text(encoding="utf-8")))
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    serialized = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "summary": result["summary"],
                "comparisons": result["comparisons"]
            },
            indent=2,
            sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
