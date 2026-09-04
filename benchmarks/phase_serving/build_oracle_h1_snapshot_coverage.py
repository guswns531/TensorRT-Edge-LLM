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
"""Join measured actions from identical cross-run phase snapshots."""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path
from typing import Any

try:
    from benchmarks.phase_serving.validate_phase_scheduler_events import \
        _snapshot_signature
except ModuleNotFoundError:
    from validate_phase_scheduler_events import _snapshot_signature

RECORD_PREFIX = "PHASE_SCHEDULER_EVENT\t"


def load_events(policy: str, paths: list[Path]) -> list[dict[str, Any]]:
    """Load scheduler events and attach policy/source lineage."""
    result: list[dict[str, Any]] = []
    for path in paths:
        with path.open(encoding="utf-8", errors="replace") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.startswith(RECORD_PREFIX):
                    continue
                event = json.loads(line.removeprefix(RECORD_PREFIX))
                event["_policy"] = policy
                event["_source_path"] = str(path)
                event["_source"] = f"{path}:{line_number}"
                result.append(event)
    return result


def _run_key(event: dict[str, Any]) -> tuple[str, str]:
    return str(event["_source_path"]), str(event.get("run_id"))


def _candidate_work_signature(event: dict[str, Any]) -> tuple[Any, ...] | None:
    """Identify hidden ready work represented only by the candidate frontier.

    The coordinator ready snapshot does not own every mechanism queue.  In
    particular, a prepared vision batch can already have moved out of the
    coordinator's pending queue while still appearing as an encoder
    candidate.  Requiring the candidate request frontier prevents E1 and E4
    states with the same visible ready counters from becoming a false
    cross-policy counterfactual pair.

    Older/minimal records without candidate request IDs retain the legacy
    signature so existing diagnostic artifacts remain readable.
    """
    candidates = event.get("candidates", [])
    if not candidates or any("request_ids" not in candidate
                             for candidate in candidates):
        return None
    return tuple(
        sorted((str(candidate.get("action_kind", "")),
                bool(candidate.get("legal", False)),
                tuple(
                    int(request_id)
                    for request_id in candidate.get("request_ids", [])))
               for candidate in candidates))


def _snapshot_key(
        event: dict[str, Any]) -> tuple[str, int, tuple[Any, ...]
                                        | None]:
    if "snapshot_signature" in event:
        return ("exact_v1", int(event["snapshot_signature"]),
                _candidate_work_signature(event))
    # Old M1--M4 logs can be audited but never promoted as exact coverage.
    return "legacy_aggregate", _snapshot_signature(event), None


def build_event_index(
    events: list[dict[str, Any]]
) -> tuple[dict[tuple[tuple[str, str], int, str], dict[str, Any]], dict[tuple[
        tuple[str, str], int], list[dict[str, Any]]]]:
    """Index one policy's events for linear-time decision materialization."""
    completions: dict[tuple[tuple[str, str], int, str], dict[str, Any]] = {}
    dispatches: dict[tuple[tuple[str, str], int],
                     list[dict[str, Any]]] = collections.defaultdict(list)
    for event in events:
        if event.get(
                "event_kind"
        ) == "completion" and "gpu_start_us" in event and "gpu_end_us" in event:
            completions[(_run_key(event), int(event["execution_id"]),
                         str(event["phase"]))] = event
        elif event.get("event_kind") == "dispatch":
            dispatches[(_run_key(event), int(event.get("plan_id",
                                                       0)))].append(event)
    return completions, dispatches


def measured_episode(
        index: tuple[dict[tuple[tuple[str, str], int, str], dict[str, Any]],
                     dict[tuple[tuple[str, str], int], list[dict[str, Any]]]],
        decision: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    """Build the selected action's completion vector at its incremental boundary."""
    run_key = _run_key(decision)
    completions, dispatches = index
    plan_dispatches = dispatches.get((run_key, int(decision["plan_id"])), [])
    if not plan_dispatches:
        return None, "selected action has no measured dispatch"
    newcomer_start = min(
        float(dispatch.get("gpu_start_us", float("inf")))
        for dispatch in plan_dispatches)
    if newcomer_start == float("inf"):
        # Dispatch records do not carry the CUDA interval. Use their matching
        # completion start, which is in the shared CUDA-event epoch.
        starts = []
        for dispatch in plan_dispatches:
            completion = completions.get(
                (run_key, int(dispatch["execution_id"]),
                 str(dispatch["phase"])))
            if completion is not None:
                starts.append(float(completion["gpu_start_us"]))
        if not starts:
            return None, "selected action has no common-epoch completion"
        newcomer_start = min(starts)

    component_keys: list[tuple[int, str, bool]] = []
    for work in decision.get("inflight", []):
        component_keys.append(
            (int(work["execution_id"]), str(work["phase"]), True))
    for dispatch in plan_dispatches:
        component_keys.append(
            (int(dispatch["execution_id"]), str(dispatch["phase"]), False))
    components: list[dict[str, Any]] = []
    seen: set[tuple[int, str]] = set()
    for execution_id, phase, incumbent in component_keys:
        key = (execution_id, phase)
        if key in seen:
            continue
        seen.add(key)
        completion = completions.get((run_key, *key))
        if completion is None:
            return None, f"missing completion for {phase}/{execution_id}"
        completion_us = float(completion["gpu_end_us"]) - newcomer_start
        if incumbent and completion_us <= 0.0:
            return None, "incumbent completed before incremental boundary"
        if completion_us < 0.0:
            return None, "newcomer completion precedes incremental boundary"
        components.append({
            "phase": phase,
            "execution_id": execution_id,
            "completion_us": completion_us,
            "uncertainty_us": 0.0,
            "incumbent": incumbent,
        })
    components.sort(key=lambda item: (float(item[
        "completion_us"]), str(item["phase"]), int(item["execution_id"])))
    if not components:
        return None, "empty completion vector"
    fidelity = all(
        bool(dispatch.get("action_fidelity", False))
        for dispatch in plan_dispatches)
    fidelity = fidelity and all(
        bool(completions[(run_key, int(item["execution_id"]),
                          str(item["phase"]))].get("action_fidelity", False))
        for item in components)
    prediction_frontier = []
    for candidate in decision.get("candidates", []):
        prediction_frontier.append({
            "action_id":
            int(candidate["action_id"]),
            "action_kind":
            str(candidate.get("action_kind", "")),
            "legal":
            bool(candidate.get("legal", False)),
            "predicted_completion_us":
            candidate.get("predicted_completion_us", []),
            "uncertainty_us":
            candidate.get("uncertainty_us", []),
            "contextual_completion_valid":
            bool(candidate.get("contextual_completion_valid", False)),
            "contextual_direction":
            str(candidate.get("contextual_direction", "")),
            "contextual_completion_ready":
            bool(candidate.get("contextual_completion_ready", False)),
            "contextual_incumbent_mean_us":
            float(candidate.get("contextual_incumbent_mean_us", 0.0)),
            "contextual_incumbent_uncertainty_us":
            float(candidate.get("contextual_incumbent_uncertainty_us", 0.0)),
            "contextual_newcomer_mean_us":
            float(candidate.get("contextual_newcomer_mean_us", 0.0)),
            "contextual_newcomer_uncertainty_us":
            float(candidate.get("contextual_newcomer_uncertainty_us", 0.0)),
            "contextual_pair_observations":
            int(candidate.get("contextual_pair_observations", 0)),
            "contextual_direction_observations":
            int(candidate.get("contextual_direction_observations", 0)),
            "contextual_direction_weight":
            float(candidate.get("contextual_direction_weight", 0.0)),
        })
    return {
        "action_id":
        int(decision["selected_action_id"]),
        "incremental_action_id":
        int(decision.get("incremental_action_id", 0)),
        "action_kind":
        str(decision.get("action_kind")),
        "requested_direction":
        str(decision.get("requested_action_direction")),
        "requested_start_skew_percent":
        int(decision.get("requested_start_skew_percent", -1)),
        "request_ids":
        decision.get("request_ids", []),
        "boundary_gpu_us":
        newcomer_start,
        "projected_boundary_us":
        float(components[0]["completion_us"]),
        "whole_action_completion_us":
        max(float(item["completion_us"]) for item in components),
        "completed_phase":
        str(components[0]["phase"]),
        "components":
        components,
        "action_fidelity":
        fidelity,
        "policy":
        str(decision["_policy"]),
        "source":
        str(decision["_source"]),
        "prediction_frontier":
        prediction_frontier,
    }, None


def build_coverage(
        policy_events: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Merge selected measured actions by deterministic snapshot signature."""
    episodes: dict[tuple[str, int, tuple[Any, ...] | None], dict[str,
                                                                 Any]] = {}
    errors: list[str] = []
    decisions = 0
    for policy, events in policy_events.items():
        event_index = build_event_index(events)
        for decision in events:
            if decision.get("event_kind") != "decision":
                continue
            decisions += 1
            quality, signature, candidate_work_signature = _snapshot_key(
                decision)
            if quality == "exact_v1" and int(
                    decision["snapshot_signature"]) != _snapshot_signature(
                        decision):
                errors.append(
                    f"{decision['_source']}: invalid snapshot signature")
                continue
            measured, error = measured_episode(event_index, decision)
            if error is not None:
                errors.append(f"{decision['_source']}: {error}")
                continue
            assert measured is not None
            key = quality, signature, candidate_work_signature
            episode = episodes.setdefault(
                key, {
                    "signature_quality": quality,
                    "snapshot_signature": signature,
                    "candidate_work_signature": candidate_work_signature,
                    "observations": [],
                    "policy_selections": collections.defaultdict(list),
                    "prediction_frontiers": [],
                })
            episode["observations"].append(measured)
            episode["policy_selections"][policy].append(
                measured["incremental_action_id"])
            episode["prediction_frontiers"].append({
                "policy":
                policy,
                "source":
                measured["source"],
                "selected_action_id":
                measured["action_id"],
                "selected_incremental_action_id":
                measured["incremental_action_id"],
                "candidates":
                measured["prediction_frontier"],
            })

    serialized: list[dict[str, Any]] = []
    exact_multi_action = 0
    exact_repeated = 0
    fidelity_failures = 0
    for episode in episodes.values():
        actions: dict[int, list[dict[str,
                                     Any]]] = collections.defaultdict(list)
        for observation in episode["observations"]:
            actions[int(
                observation["incremental_action_id"])].append(observation)
            fidelity_failures += int(not observation["action_fidelity"])
        episode["actions"] = [{
            "incremental_action_id":
            action_id,
            "candidate_action_id":
            samples[0]["action_id"],
            "candidate_action_id_stable":
            len({sample["action_id"]
                 for sample in samples}) == 1,
            "samples":
            len(samples),
            "action_kind":
            samples[0]["action_kind"],
            "requested_direction":
            samples[0]["requested_direction"],
            "requested_start_skew_percent":
            samples[0]["requested_start_skew_percent"],
            "projected_boundary_us":
            [sample["projected_boundary_us"] for sample in samples],
            "whole_action_completion_us":
            [sample["whole_action_completion_us"] for sample in samples],
            "completed_phase":
            [sample["completed_phase"] for sample in samples],
            "component_completion_us": {
                phase: [
                    float(component["completion_us"]) for sample in samples
                    for component in sample["components"]
                    if str(component["phase"]) == phase
                ]
                for phase in sorted({
                    str(component["phase"])
                    for sample in samples
                    for component in sample["components"]
                })
            },
            "sources": [sample["source"] for sample in samples],
        } for action_id, samples in sorted(actions.items())]
        episode["policy_selections"] = dict(episode["policy_selections"])
        del episode["observations"]
        if episode["signature_quality"] == "exact_v1":
            exact_repeated += len(episode["actions"]) >= 1 and sum(
                action["samples"] for action in episode["actions"]) >= 2
            exact_multi_action += len(episode["actions"]) >= 2
        serialized.append(episode)
    serialized.sort(key=lambda item:
                    (item["signature_quality"], item["snapshot_signature"]))
    return {
        "schema_version": 2,
        "artifact": "oracle_h1_snapshot_coverage",
        "policies": sorted(policy_events),
        "summary": {
            "decisions":
            decisions,
            "snapshot_signatures":
            len(serialized),
            "exact_repeated_snapshots":
            exact_repeated,
            "exact_multi_action_snapshots":
            exact_multi_action,
            "action_fidelity_failures":
            fidelity_failures,
            "gate_b_candidate_coverage":
            exact_multi_action > 0 and fidelity_failures == 0,
        },
        "episodes": serialized,
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy",
        nargs="+",
        action="append",
        metavar=("NAME", "LOG"),
        required=True,
        help="policy name followed by one or more scheduler logs")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        policy_events: dict[str, list[dict[str, Any]]] = {}
        for values in args.policy:
            name, *logs = values
            if not logs:
                raise ValueError(f"policy {name} has no logs")
            if name in policy_events:
                raise ValueError(f"duplicate policy {name}")
            policy_events[name] = load_events(name,
                                              [Path(log) for log in logs])
        result = build_coverage(policy_events)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) +
                               "\n",
                               encoding="utf-8")
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
