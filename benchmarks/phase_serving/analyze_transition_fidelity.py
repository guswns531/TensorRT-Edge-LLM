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
"""Audit physical-model safety and frozen transition disagreement."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

PAIR_ACTIONS = {"encoder_prefill", "encoder_decode", "prefill_decode"}
MODELS = ("scalar", "effect", "completion")
TRANSITION_STATE_FIELDS = (
    "first_completed_phase_mask",
    "min_first_encoder_ready_rows",
    "max_first_encoder_ready_rows",
    "min_first_prefill_ready_rows",
    "max_first_prefill_ready_rows",
    "min_first_decode_ready_rows",
    "max_first_decode_ready_rows",
    "min_encoder_ready_rows",
    "max_encoder_ready_rows",
    "min_prefill_ready_rows",
    "max_prefill_ready_rows",
    "min_decode_ready_rows",
    "max_decode_ready_rows",
    "min_reclaim_bytes",
    "max_reclaim_bytes",
)


def _events(path: Path) -> list[dict[str, Any]]:
    marker = "PHASE_SCHEDULER_EVENT\t"
    result = []
    for line in path.read_text(encoding="utf-8",
                               errors="replace").splitlines():
        offset = line.find(marker)
        if offset < 0:
            continue
        try:
            result.append(json.loads(line[offset + len(marker):]))
        except json.JSONDecodeError:
            continue
    return result


def _selected_candidate(decision: dict[str, Any]) -> dict[str, Any] | None:
    selected = int(decision.get("selected_action_id", 0))
    return next((candidate for candidate in decision.get("candidates", [])
                 if int(candidate.get("action_id", 0)) == selected), None)


def _is_valid_transition(candidate: dict[str, Any], model: str) -> bool:
    transition = candidate.get(f"{model}_transition", {})
    return bool(transition.get("evaluated")) and bool(transition.get("valid"))


def _transition_state(candidate: dict[str, Any],
                      model: str) -> tuple[int, ...]:
    transition = candidate[f"{model}_transition"]
    return tuple(
        int(transition.get(field, 0)) for field in TRANSITION_STATE_FIELDS)


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(
        len(ordered) - 1, max(0,
                              math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def _new_physics_stats() -> dict[str, Any]:
    return {
        model: {
            "ready": 0,
            "sign_correct": 0,
            "false_safe": 0,
            "prediction_errors": [],
        }
        for model in MODELS
    }


def _finalize_physics(stats: dict[str, Any], samples: int,
                      minimum_samples: int, minimum_sign_agreement: float,
                      maximum_false_safe: int) -> dict[str, Any]:
    result: dict[str, Any] = {"samples": samples}
    for model, values in stats.items():
        ready = int(values["ready"])
        errors = values["prediction_errors"]
        sign_agreement = values["sign_correct"] / ready if ready else 0.0
        false_safe = int(values["false_safe"])
        result[model] = {
            "ready":
            ready,
            "coverage":
            ready / samples if samples else 0.0,
            "sign_agreement":
            sign_agreement,
            "false_safe":
            false_safe,
            "makespan_mae_us":
            _mean([abs(value) for value in errors]),
            "makespan_error_p95_us":
            _percentile([abs(value) for value in errors], 0.95),
            "promotion_gate": {
                "minimum_samples":
                minimum_samples,
                "minimum_sign_agreement":
                minimum_sign_agreement,
                "maximum_false_safe":
                maximum_false_safe,
                "passed":
                ready >= minimum_samples
                and sign_agreement >= minimum_sign_agreement
                and false_safe <= maximum_false_safe,
            },
        }
    return result


def _analyze_paths(paths: list[Path], minimum_samples: int,
                   minimum_sign_agreement: float,
                   maximum_false_safe: int) -> dict[str, Any]:
    decisions_total = 0
    frozen_valid = 0
    candidates_total = 0
    transition_valid = defaultdict(int)
    comparisons = {
        "scalar_effect": {
            "common": 0,
            "state_disagreements": 0,
            "horizon_abs_delta_us": [],
        },
        "scalar_completion": {
            "common": 0,
            "state_disagreements": 0,
            "horizon_abs_delta_us": [],
        },
        "effect_completion": {
            "common": 0,
            "state_disagreements": 0,
            "horizon_abs_delta_us": [],
        },
    }
    physics = _new_physics_stats()
    physics_by_direction: dict[str,
                               dict[str,
                                    Any]] = defaultdict(_new_physics_stats)
    physics_samples = 0
    samples_by_direction = defaultdict(int)
    rejected = defaultdict(int)

    for path in paths:
        events = _events(path)
        decisions = {}
        for event in events:
            if event.get("event_kind") != "decision":
                continue
            decisions_total += 1
            decision_id = int(event.get("decision_id", 0))
            decisions[decision_id] = event
            frozen_valid += int(
                bool(event.get("frozen_transition_snapshot_valid", False)))
            for candidate in event.get("candidates", []):
                candidates_total += 1
                for model in MODELS:
                    transition_valid[model] += int(
                        _is_valid_transition(candidate, model))
                for left, right, key in (("scalar", "effect", "scalar_effect"),
                                         ("scalar", "completion",
                                          "scalar_completion"),
                                         ("effect", "completion",
                                          "effect_completion")):
                    if not (_is_valid_transition(candidate, left)
                            and _is_valid_transition(candidate, right)):
                        continue
                    comparison = comparisons[key]
                    comparison["common"] += 1
                    comparison["state_disagreements"] += int(
                        _transition_state(candidate, left) !=
                        _transition_state(candidate, right))
                    left_horizon = float(candidate[f"{left}_transition"].get(
                        "worst_case_robust_horizon_us", 0.0))
                    right_horizon = float(candidate[f"{right}_transition"].get(
                        "worst_case_robust_horizon_us", 0.0))
                    comparison["horizon_abs_delta_us"].append(
                        abs(left_horizon - right_horizon))

        for completion in events:
            if (completion.get("event_kind") != "completion"
                    or completion.get("action_kind") not in PAIR_ACTIONS):
                continue
            if not completion.get("action_fidelity", False):
                rejected["action_fidelity"] += 1
                continue
            decision = decisions.get(int(completion.get("decision_id", 0)))
            candidate = _selected_candidate(decision) if decision else None
            if candidate is None:
                rejected["candidate"] += 1
                continue
            epoch_value = completion.get("gpu_start_us")
            incumbent_value = completion.get("incumbent_gpu_completion_us")
            newcomer_value = completion.get("newcomer_gpu_completion_us")
            if any(value is None or not math.isfinite(float(value))
                   for value in (epoch_value, incumbent_value,
                                 newcomer_value)):
                rejected["common_epoch"] += 1
                continue
            epoch = float(epoch_value)
            actual_makespan = max(float(incumbent_value),
                                  float(newcomer_value)) - epoch
            incumbent_reference = float(
                candidate.get("contextual_incumbent_reference_us", 0.0))
            newcomer_reference = float(
                candidate.get("contextual_newcomer_reference_us", 0.0))
            serial = incumbent_reference + newcomer_reference
            if actual_makespan < 0.0 or serial <= 0.0:
                rejected["reference"] += 1
                continue
            actual_positive = actual_makespan < serial
            direction = str(candidate.get("contextual_direction", "unknown"))
            physics_samples += 1
            samples_by_direction[direction] += 1
            scalar_makespan = float(
                candidate.get("scalar_decision_makespan_us", 0.0))
            scalar_ready = bool(candidate.get("scalar_decision_cost_known"))
            effect_compression = float(
                candidate.get("contextual_effect_compression_mean", 0.0))
            effect_uncertainty = float(
                candidate.get("contextual_effect_compression_uncertainty",
                              0.0))
            effect_ready = bool(candidate.get("contextual_effect_ready"))
            completion_makespan = max(
                float(candidate.get("contextual_incumbent_mean_us", 0.0)),
                float(candidate.get("contextual_newcomer_mean_us", 0.0)))
            completion_uncertainty = max(
                float(candidate.get("contextual_incumbent_uncertainty_us",
                                    0.0)),
                float(candidate.get("contextual_newcomer_uncertainty_us",
                                    0.0)))
            completion_ready = bool(
                candidate.get("contextual_completion_ready"))
            predictions = {
                "scalar": (scalar_ready, scalar_makespan, scalar_makespan
                           < serial),
                "effect": (effect_ready, serial * (1.0 - effect_compression),
                           effect_compression - effect_uncertainty > 0.0),
                "completion":
                (completion_ready, completion_makespan,
                 completion_makespan + completion_uncertainty < serial),
            }
            for model, (ready, predicted_makespan,
                        predicted_safe) in predictions.items():
                if not ready:
                    continue
                for target in (physics[model],
                               physics_by_direction[direction][model]):
                    target["ready"] += 1
                    target["sign_correct"] += int(
                        predicted_safe == actual_positive)
                    target["false_safe"] += int(predicted_safe
                                                and not actual_positive)
                    target["prediction_errors"].append(predicted_makespan -
                                                       actual_makespan)

    rendered_comparisons = {}
    for name, values in comparisons.items():
        common = int(values["common"])
        deltas = values["horizon_abs_delta_us"]
        rendered_comparisons[name] = {
            "common":
            common,
            "state_disagreements":
            int(values["state_disagreements"]),
            "state_disagreement_fraction":
            values["state_disagreements"] / common if common else 0.0,
            "horizon_abs_delta_mean_us":
            _mean(deltas),
            "horizon_abs_delta_p95_us":
            _percentile(deltas, 0.95),
        }
    return {
        "logs":
        len(paths),
        "decisions":
        decisions_total,
        "frozen_snapshot_valid":
        frozen_valid,
        "frozen_snapshot_coverage":
        frozen_valid / decisions_total if decisions_total else 0.0,
        "candidates":
        candidates_total,
        "transition_valid": {
            model: {
                "count":
                transition_valid[model],
                "coverage":
                transition_valid[model] /
                candidates_total if candidates_total else 0.0,
            }
            for model in MODELS
        },
        "transition_disagreement":
        rendered_comparisons,
        "selected_physics":
        _finalize_physics(physics, physics_samples, minimum_samples,
                          minimum_sign_agreement, maximum_false_safe),
        "selected_physics_by_direction": {
            direction:
            _finalize_physics(stats, samples_by_direction[direction],
                              minimum_samples, minimum_sign_agreement,
                              maximum_false_safe)
            for direction, stats in sorted(physics_by_direction.items())
        },
        "rejected":
        dict(sorted(rejected.items())),
    }


def analyze(root: Path,
            minimum_samples: int = 100,
            minimum_sign_agreement: float = 0.8,
            maximum_false_safe: int = 0) -> dict[str, Any]:
    paths = sorted(root.rglob("*-events.jsonl"))
    if not paths:
        # The HTTP harness preserves backend stdout even when an explicit
        # telemetry side channel was not configured. Use it as a lossless
        # fallback for smoke and production request runs.
        paths = sorted(root.rglob("gateway.log"))
    if not paths:
        raise ValueError(f"no phase event logs under {root}")
    grouped: dict[str, list[Path]] = defaultdict(list)
    for path in paths:
        relative = path.relative_to(root)
        # Warm-up matrices use mode/workload/worker/activity/events while
        # focused replay gates usually place events directly under the case.
        # Group by the workload rather than collapsing an entire matrix into
        # its initialization mode (for example, ``generic``).
        if len(relative.parts) >= 5 and relative.parts[-2] == "activity":
            workload = relative.parts[-4]
        else:
            workload = relative.parts[0] if len(
                relative.parts) > 1 else path.stem
        grouped[workload].append(path)
    result = _analyze_paths(paths, minimum_samples, minimum_sign_agreement,
                            maximum_false_safe)
    result.update({
        "schema_version": 1,
        "mode": "frozen_transition_fidelity",
        "by_workload": {
            workload:
            _analyze_paths(workload_paths, minimum_samples,
                           minimum_sign_agreement, maximum_false_safe)
            for workload, workload_paths in sorted(grouped.items())
        },
    })
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--minimum-samples", type=int, default=100)
    parser.add_argument("--minimum-sign-agreement", type=float, default=0.8)
    parser.add_argument("--maximum-false-safe", type=int, default=0)
    args = parser.parse_args()
    result = analyze(args.root, args.minimum_samples,
                     args.minimum_sign_agreement, args.maximum_false_safe)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
