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
"""Rank measured H1 replay candidates and compare policy-only selections."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


def candidate_value(candidate: dict[str, Any]) -> dict[str, Any]:
    """Return the M5 lexicographic value for one replayed candidate."""
    projection = candidate.get("projection", {})
    action_id = int(candidate.get("action_id", 0))
    budget = int(candidate.get("memory_budget_bytes", 0))
    peak = int(candidate.get("hard_peak_managed_bytes", 0))
    feasible = all(
        bool(candidate.get(field, True))
        for field in ("legal", "dependency_safe", "context_safe", "shape_safe",
                      "ownership_safe"))
    feasible = feasible and bool(projection.get("valid", False))
    feasible = feasible and int(projection.get("action_id", 0)) == action_id
    feasible = feasible and (budget == 0 or peak <= budget)

    boundary = max(0.0, float(projection.get("boundary_us", 0.0)))
    robust_boundary = max(
        boundary, float(projection.get("robust_boundary_us", boundary)))
    violation = 0.0
    progress = 0.0
    for milestone in candidate.get("milestones", []):
        slack = float(milestone.get("slack_us", math.inf))
        completion = max(
            0.0, float(milestone.get("predicted_completion_us", boundary)))
        uncertainty = max(
            0.0,
            float(milestone.get("uncertainty_us", robust_boundary - boundary)))
        if math.isfinite(slack):
            violation = max(violation,
                            completion + uncertainty - max(0.0, slack))
        if bool(milestone.get("completed_at_boundary", False)):
            units = max(0.0, float(milestone.get("progress_units", 0.0)))
            progress += units / max(1.0, slack) if math.isfinite(
                slack) else units
    efficiency = max(0.0, float(candidate.get("reference_work_us", 0.0))) \
        / max(robust_boundary, sys.float_info.epsilon)
    return {
        "hard_feasible":
        feasible,
        "robust_violation_us":
        max(0.0, violation),
        "urgency_normalized_progress":
        progress,
        "service_efficiency":
        efficiency,
        "released_ownership_bytes":
        int(candidate.get("released_ownership_bytes", 0)),
        "action_id":
        action_id,
        "source_index":
        int(candidate.get("source_index", 0)),
    }


def rank_key(
        value: dict[str, Any]) -> tuple[float, float, float, int, int, int]:
    """Map the C++ lexicographic objective to Python's ascending order."""
    return (float(value["robust_violation_us"]),
            -float(value["urgency_normalized_progress"]),
            -float(value["service_efficiency"]),
            -int(value["released_ownership_bytes"]), int(value["action_id"]),
            int(value["source_index"]))


def analyze(payload: dict[str, Any]) -> dict[str, Any]:
    """Select the measured H1 oracle and compare named policies."""
    episodes = payload.get("episodes", [])
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("replay requires at least one episode")
    records: list[dict[str, Any]] = []
    policy_names: set[str] = set()
    action_fidelity = True
    workload_names: set[str] = set()
    saturation_points: set[float] = set()
    for episode in episodes:
        candidates = episode.get("candidates", [])
        if not isinstance(candidates, list) or not candidates:
            raise ValueError(
                f"episode {episode.get('episode_id')} has no candidates")
        labels = [str(candidate.get("label", "")) for candidate in candidates]
        if any(not label
               for label in labels) or len(set(labels)) != len(labels):
            raise ValueError(
                f"episode {episode.get('episode_id')} has invalid labels")
        values = [candidate_value(candidate) for candidate in candidates]
        feasible = [
            index for index, value in enumerate(values)
            if value["hard_feasible"]
        ]
        if not feasible:
            raise ValueError(
                f"episode {episode.get('episode_id')} has no feasible candidate"
            )
        oracle_index = min(feasible, key=lambda index: rank_key(values[index]))
        oracle_label = labels[oracle_index]
        selections = episode.get("selections", {})
        if not isinstance(selections, dict):
            raise ValueError(
                f"episode {episode.get('episode_id')} has invalid selections")
        policy_names.update(str(name) for name in selections)
        for name, label in selections.items():
            if str(label) not in labels:
                raise ValueError(
                    f"episode {episode.get('episode_id')} policy {name} selected unknown {label}"
                )
        action_fidelity = action_fidelity and bool(
            episode.get("action_fidelity", False))
        workload = str(episode.get("workload", ""))
        if workload:
            workload_names.add(workload)
        if "offered_req_s" in episode:
            saturation_points.add(float(episode["offered_req_s"]))
        records.append({
            "episode_id":
            episode.get("episode_id"),
            "workload":
            workload,
            "oracle":
            oracle_label,
            "oracle_value":
            values[oracle_index],
            "selections":
            selections,
            "candidates": [{
                "label": label,
                "value": value
            } for label, value in zip(labels, values)],
        })

    comparisons: dict[str, Any] = {}
    for policy in sorted(policy_names):
        compared = 0
        agreements = 0
        changes = 0
        violation_regret = 0.0
        for record in records:
            selected_label = record["selections"].get(policy)
            if selected_label is None:
                continue
            compared += 1
            agreements += selected_label == record["oracle"]
            changes += selected_label != record["oracle"]
            selected = next(candidate["value"]
                            for candidate in record["candidates"]
                            if candidate["label"] == selected_label)
            violation_regret += max(
                0.0,
                float(selected["robust_violation_us"]) -
                float(record["oracle_value"]["robust_violation_us"]))
        comparisons[policy] = {
            "episodes": compared,
            "agreements": agreements,
            "oracle_changes": changes,
            "agreement_ratio": agreements / compared if compared else 0.0,
            "robust_violation_regret_us": violation_regret,
        }

    required_loads = {39.0, 48.8, 97.5}
    coverage = {
        "episodes":
        len(records),
        "workloads":
        len(workload_names),
        "saturation_points":
        sorted(saturation_points),
        "action_fidelity_100_percent":
        action_fidelity,
        "has_12_workloads":
        len(workload_names) >= 12,
        "has_required_saturation_points":
        required_loads.issubset(saturation_points),
    }
    coverage["gate_b_evaluable"] = bool(
        coverage["action_fidelity_100_percent"]
        and coverage["has_12_workloads"]
        and coverage["has_required_saturation_points"])
    return {
        "schema_version":
        1,
        "selector":
        "measured_oracle_h1",
        "objective_order": [
            "hard_feasibility", "robust_slo", "progress", "efficiency",
            "ownership_release", "stable_identity"
        ],
        "episodes":
        records,
        "comparisons":
        comparisons,
        "coverage":
        coverage,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = analyze(json.loads(args.input.read_text(encoding="utf-8")))
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    serialized = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "comparisons": result["comparisons"],
                "coverage": result["coverage"],
            },
            indent=2,
            sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
