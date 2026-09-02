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
"""Measure natural co-launch/residual density and completion-policy replay regret."""

from __future__ import annotations

import argparse
import collections
import json
import math
import statistics
from pathlib import Path
from typing import Any

RECORD_PREFIX = "PHASE_SCHEDULER_EVENT\t"
PAIR_ACTIONS = {"encoder_prefill", "encoder_decode", "prefill_decode"}


def load_events(paths: list[Path]) -> list[dict[str, Any]]:
    """Load unified events and retain a run identity isolated by source log."""
    events: list[dict[str, Any]] = []
    for path in paths:
        with path.open(encoding="utf-8", errors="replace") as source:
            for line in source:
                if not line.startswith(RECORD_PREFIX):
                    continue
                event = json.loads(line.removeprefix(RECORD_PREFIX))
                event["_source_log"] = str(path)
                events.append(event)
    return events


def percentile(values: list[float], fraction: float) -> float | None:
    """Return a linearly interpolated percentile."""
    if not values:
        return None
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position -
                                                                 lower)


def distribution(values: list[float]) -> dict[str, float | int | None]:
    """Summarize a replay or timing distribution."""
    return {
        "count": len(values),
        "mean": statistics.fmean(values) if values else None,
        "median": statistics.median(values) if values else None,
        "p95": percentile(values, 0.95),
    }


def action_mode(candidate: dict[str, Any]) -> str:
    """Return the legal pair-action execution mode represented by a candidate."""
    if candidate.get("action_kind") not in PAIR_ACTIONS:
        return "single"
    return "residual_augmentation" if candidate.get("residual_augmentation",
                                                    False) else "co_launch"


def analyze(events: list[dict[str, Any]],
            confidence_beta: float) -> dict[str, Any]:
    """Build opportunity, selection, coverage, disagreement, and replay metrics."""
    decisions = [
        event for event in events if event.get("event_kind") == "decision"
    ]
    dispatches = [
        event for event in events if event.get("event_kind") == "dispatch"
    ]
    completions = [
        event for event in events if event.get("event_kind") == "completion"
    ]
    completions_by_plan: dict[tuple[str, str, int], list[dict[str, Any]]] = \
        collections.defaultdict(list)
    for completion in completions:
        if "gpu_end_us" not in completion:
            continue
        completions_by_plan[(completion["_source_log"],
                             str(completion.get("run_id")),
                             int(completion.get("plan_id",
                                                0)))].append(completion)
    observed_action_makespans: dict[tuple[int, int], list[float]] = \
        collections.defaultdict(list)
    for decision in decisions:
        plan_id = int(decision.get("plan_id", 0))
        action_id = int(decision.get("selected_action_id", 0))
        action_completions = completions_by_plan.get(
            (decision["_source_log"], str(decision.get("run_id")), plan_id),
            [])
        if action_completions:
            start = min(
                float(completion.get("gpu_start_us", 0.0))
                for completion in action_completions)
            end = max(
                float(completion["gpu_end_us"])
                for completion in action_completions)
            observed_action_makespans[(int(
                decision.get("snapshot_signature",
                             0)), action_id)].append(end - start)

    opportunity_counts: collections.Counter[str] = collections.Counter()
    selection_counts: collections.Counter[str] = collections.Counter()
    direction_skew: collections.Counter[tuple[str, str, int]] = \
        collections.Counter()
    disagreements = 0
    comparable = 0
    regrets: list[float] = []
    false_safe = 0
    ready_predictions = 0
    covered_predictions = 0
    fidelity_failures = 0

    for dispatch in dispatches:
        fidelity_failures += not bool(dispatch.get("action_fidelity", False))

    for decision in decisions:
        candidates = [
            candidate for candidate in decision.get("candidates", [])
            if candidate.get("legal", False)
        ]
        pair_candidates = [
            candidate for candidate in candidates
            if candidate.get("action_kind") in PAIR_ACTIONS
        ]
        for mode in {action_mode(candidate) for candidate in pair_candidates}:
            opportunity_counts[mode] += 1
        selected_id = int(decision.get("selected_action_id", 0))
        selected = next((candidate for candidate in candidates
                         if int(candidate.get("action_id", 0)) == selected_id),
                        None)
        if selected is not None:
            selection_counts[action_mode(selected)] += 1

        ready = [
            candidate for candidate in candidates
            if candidate.get("contextual_completion_ready", False)
        ]
        if not ready:
            continue
        ready_predictions += len(ready)
        covered_predictions += sum(
            bool(candidate.get("contextual_uncertainty_calibrated", False))
            for candidate in ready)
        completion_choice = min(
            ready,
            key=lambda candidate: (
                float(candidate.get("max_slo_violation_us", 0.0)),
                max(float(candidate.get("contextual_incumbent_mean_us", 0.0)),
                    float(candidate.get("contextual_newcomer_mean_us", 0.0))),
                int(candidate.get("action_id", 0)),
            ))
        completion_id = int(completion_choice.get("action_id", 0))
        comparable += 1
        disagreements += completion_id != selected_id
        signature = int(decision.get("snapshot_signature", 0))
        observed = {
            int(candidate.get("action_id", 0)): statistics.median(samples)
            for candidate in candidates
            if (samples := observed_action_makespans.get((
                signature, int(candidate.get("action_id", 0)))))
        }
        if completion_id in observed and observed:
            oracle = min(observed.values())
            regrets.append(observed[completion_id] - oracle)
            predicted_safe = max(
                float(
                    completion_choice.get("contextual_incumbent_mean_us", 0.0))
                + confidence_beta * float(
                    completion_choice.get(
                        "contextual_incumbent_uncertainty_us", 0.0)),
                float(completion_choice.get("contextual_newcomer_mean_us",
                                            0.0)) +
                confidence_beta * float(
                    completion_choice.get("contextual_newcomer_uncertainty_us",
                                          0.0)),
            ) <= float(
                completion_choice.get("contextual_minimum_slack_us",
                                      -math.inf))
            false_safe += predicted_safe and observed[completion_id] > float(
                completion_choice.get("contextual_minimum_slack_us",
                                      -math.inf))

    for completion in completions:
        mode = str(completion.get("dispatch_mode", "unknown"))
        direction_skew[(mode, str(completion.get("action_direction", "none")),
                        int(completion.get("observed_start_skew_percent",
                                           -1)))] += 1
        fidelity_failures += not bool(completion.get("action_fidelity", False))

    return {
        "schema_version":
        1,
        "events":
        len(events),
        "decisions":
        len(decisions),
        "dispatches":
        len(dispatches),
        "completions":
        len(completions),
        "opportunities":
        dict(sorted(opportunity_counts.items())),
        "selections":
        dict(sorted(selection_counts.items())),
        "direction_skew_observations":
        [{
            "dispatch_mode": mode,
            "direction": direction,
            "observed_start_skew_percent": skew,
            "count": count,
        }
         for (mode, direction, skew), count in sorted(direction_skew.items())],
        "completion_predictions": {
            "ready":
            ready_predictions,
            "calibrated":
            covered_predictions,
            "calibrated_fraction":
            covered_predictions /
            ready_predictions if ready_predictions else None,
        },
        "selector_disagreement": {
            "comparable_decisions": comparable,
            "count": disagreements,
            "fraction": disagreements / comparable if comparable else None,
        },
        "replay_regret_us":
        distribution(regrets),
        "conformal_false_safe":
        false_safe,
        "action_fidelity_failures":
        fidelity_failures,
    }


def source_case(path: str, root: Path) -> str:
    """Derive a stable workload/load case from one matrix log path."""
    relative = Path(path).resolve().relative_to(root.resolve())
    parts = relative.parts
    if parts and parts[0].startswith("repeat-"):
        if len(parts) < 2:
            raise ValueError(f"repeat path has no case component: {path}")
        return parts[1]
    if not parts:
        raise ValueError(f"source path equals grouping root: {path}")
    return parts[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--confidence-beta", type=float, default=1.96)
    parser.add_argument("--source-root", type=Path)
    args = parser.parse_args()
    if args.confidence_beta < 0.0:
        parser.error("confidence beta must be non-negative")
    events = load_events(args.logs)
    artifact = analyze(events, args.confidence_beta)
    if args.source_root is not None:
        grouped: dict[str, list[dict[str,
                                     Any]]] = collections.defaultdict(list)
        try:
            for event in events:
                grouped[source_case(event["_source_log"],
                                    args.source_root)].append(event)
        except ValueError as error:
            parser.error(str(error))
        artifact["by_case"] = {
            case: analyze(case_events, args.confidence_beta)
            for case, case_events in sorted(grouped.items())
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True) +
                           "\n",
                           encoding="utf-8")
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
