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
"""Compare Scalar, Effect-Vector, and Completion predictions on common epochs."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

PAIR_ACTIONS = {"encoder_prefill", "encoder_decode", "prefill_decode"}


def _events(path: Path) -> list[dict[str, Any]]:
    result = []
    marker = "PHASE_SCHEDULER_EVENT\t"
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


def _candidate(decision: dict[str, Any]) -> dict[str, Any] | None:
    selected = int(decision.get("selected_action_id", 0))
    for candidate in decision.get("candidates", []):
        if int(candidate.get("action_id", 0)) == selected:
            return candidate
    return None


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _rmse(values: list[float]) -> float:
    return math.sqrt(statistics.fmean(value * value
                                      for value in values)) if values else 0.0


def _analyze_paths(paths: list[Path]) -> dict[str, Any]:
    errors: dict[str, list[float]] = defaultdict(list)
    ready_samples = 0
    all_samples = 0
    order_correct = 0
    order_evaluable = 0
    order_ambiguous = 0
    by_direction: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "samples": 0,
            "ready": 0,
            "order_confident": 0,
            "order_ambiguous": 0,
            "order_correct": 0,
            "actual_newcomer_finishes_later": 0,
            "predicted_newcomer_finishes_later": 0,
        })
    rejected = defaultdict(int)

    for path in paths:
        events = _events(path)
        decisions = {
            int(event.get("decision_id", 0)): event
            for event in events if event.get("event_kind") == "decision"
        }
        for completion in events:
            if completion.get("event_kind") != "completion" or completion.get(
                    "action_kind") not in PAIR_ACTIONS:
                continue
            if not completion.get("action_fidelity", False):
                rejected["fidelity"] += 1
                continue
            decision = decisions.get(int(completion.get("decision_id", 0)))
            candidate = _candidate(decision) if decision else None
            if candidate is None or not candidate.get(
                    "contextual_effect_valid", False):
                rejected["missing_candidate"] += 1
                continue
            values = (
                completion.get("gpu_start_us"),
                completion.get("incumbent_gpu_completion_us"),
                completion.get("newcomer_gpu_completion_us"),
            )
            if any(value is None or not math.isfinite(float(value))
                   for value in values):
                rejected["missing_common_epoch"] += 1
                continue
            epoch = float(completion["gpu_start_us"])
            incumbent_us = float(
                completion["incumbent_gpu_completion_us"]) - epoch
            newcomer_us = float(
                completion["newcomer_gpu_completion_us"]) - epoch
            incumbent_ref = float(
                candidate.get("contextual_incumbent_reference_us", 0.0))
            newcomer_ref = float(
                candidate.get("contextual_newcomer_reference_us", 0.0))
            if incumbent_us < 0.0 or newcomer_us < 0.0 or incumbent_ref <= 0.0 or newcomer_ref <= 0.0:
                rejected["invalid_reference"] += 1
                continue
            serial = incumbent_ref + newcomer_ref
            actual_makespan = max(incumbent_us, newcomer_us)
            actual_compression = (serial - actual_makespan) / serial
            actual_incumbent_stretch = (incumbent_us -
                                        incumbent_ref) / incumbent_ref
            actual_order = (newcomer_us - incumbent_us) / serial
            direction = str(candidate.get("contextual_direction", "unknown"))
            sample = by_direction[direction]
            sample["samples"] += 1
            sample["actual_newcomer_finishes_later"] += int(actual_order > 0.0)
            all_samples += 1

            scalar_makespan = float(
                candidate.get("scalar_decision_makespan_us", 0.0))
            if not candidate.get("scalar_decision_cost_known",
                                 False) or scalar_makespan <= 0.0:
                scalar_makespan = float(
                    candidate.get("predicted_completion_us", [0.0])[0])
            errors["scalar_makespan_us"].append(scalar_makespan -
                                                actual_makespan)
            completion_makespan = max(
                float(candidate.get("contextual_incumbent_mean_us", 0.0)),
                float(candidate.get("contextual_newcomer_mean_us", 0.0)))
            errors["completion_makespan_us"].append(completion_makespan -
                                                    actual_makespan)

            if not candidate.get("contextual_effect_ready", False):
                continue
            ready_samples += 1
            sample["ready"] += 1
            compression = float(
                candidate.get("contextual_effect_compression_mean", 0.0))
            stretch = float(
                candidate.get("contextual_effect_incumbent_stretch_mean", 0.0))
            order = float(
                candidate.get("contextual_effect_order_margin_mean", 0.0))
            order_uncertainty = float(
                candidate.get("contextual_effect_order_margin_uncertainty",
                              0.0))
            errors["effect_compression"].append(compression -
                                                actual_compression)
            errors["effect_incumbent_stretch"].append(stretch -
                                                      actual_incumbent_stretch)
            errors["effect_order_margin"].append(order - actual_order)
            sample["predicted_newcomer_finishes_later"] += int(order > 0.0)
            if order - order_uncertainty <= 0.0 <= order + order_uncertainty:
                order_ambiguous += 1
                sample["order_ambiguous"] += 1
            elif actual_order != 0.0:
                order_evaluable += 1
                sample["order_confident"] += 1
                correct = (order > 0.0) == (actual_order > 0.0)
                order_correct += int(correct)
                sample["order_correct"] += int(correct)

    models = {}
    for name, values in errors.items():
        models[name] = {
            "samples": len(values),
            "mae": _mean([abs(value) for value in values]),
            "rmse": _rmse(values),
            "bias": _mean(values),
        }
    directions = dict(sorted(by_direction.items()))
    for sample in directions.values():
        confident = sample["order_confident"]
        sample["order_accuracy"] = sample[
            "order_correct"] / confident if confident else 0.0
    return {
        "logs": len(paths),
        "common_epoch_samples": all_samples,
        "effect_ready_samples": ready_samples,
        "effect_ready_fraction":
        ready_samples / all_samples if all_samples else 0.0,
        "order": {
            "confident_samples": order_evaluable,
            "correct": order_correct,
            "accuracy":
            order_correct / order_evaluable if order_evaluable else 0.0,
            "ambiguous": order_ambiguous,
        },
        "errors": models,
        "by_direction": directions,
        "rejected": dict(sorted(rejected.items())),
    }


def analyze(root: Path) -> dict[str, Any]:
    paths = sorted(root.rglob("*-events.jsonl"))
    if not paths:
        raise ValueError(f"no phase event logs under {root}")
    result = _analyze_paths(paths)
    grouped: dict[str, list[Path]] = defaultdict(list)
    for path in paths:
        relative = path.relative_to(root)
        workload = relative.parts[0] if len(relative.parts) > 1 else path.stem
        grouped[workload].append(path)
    result.update({
        "schema_version": 2,
        "mode": "effect_vector_common_epoch_shadow",
        "by_workload": {
            workload: _analyze_paths(workload_paths)
            for workload, workload_paths in sorted(grouped.items())
        },
    })
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = analyze(args.root)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
