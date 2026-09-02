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
"""Summarize M6 directional contextual-model shadow calibration."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

FAMILIES = ("pd", "ep", "ed")
DIRECTIONS = (
    "prefill_to_decode",
    "decode_to_prefill",
    "encoder_to_prefill",
    "prefill_to_encoder",
    "encoder_to_decode",
    "decode_to_encoder",
)
COMPLETION_PAIRS = ("prefill_decode", "encoder_prefill", "encoder_decode")


def _empty_completion_totals() -> dict[str, int | float]:
    return {
        "predictions": 0,
        "observations": 0,
        "ready_calibration_observations": 0,
        "incumbent_interval_covered": 0,
        "newcomer_interval_covered": 0,
        "ready_incumbent_interval_covered": 0,
        "ready_newcomer_interval_covered": 0,
        "conformal_calibration_observations": 0,
        "conformal_incumbent_interval_covered": 0,
        "conformal_newcomer_interval_covered": 0,
        "conformal_predicted_safe": 0,
        "conformal_false_safe": 0,
        "predicted_safe": 0,
        "false_safe": 0,
        "incumbent_absolute_error_us": 0.0,
        "incumbent_squared_error_us": 0.0,
        "newcomer_absolute_error_us": 0.0,
        "newcomer_squared_error_us": 0.0,
        "ready_incumbent_absolute_error_us": 0.0,
        "ready_incumbent_squared_error_us": 0.0,
        "ready_newcomer_absolute_error_us": 0.0,
        "ready_newcomer_squared_error_us": 0.0,
    }


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def _metric_records(path: Path) -> list[dict[str, Any]]:
    records = []
    for line in path.read_text(encoding="utf-8",
                               errors="replace").splitlines():
        marker = "PHASE_METRIC\t"
        offset = line.find(marker)
        if offset < 0:
            continue
        try:
            records.append(json.loads(line[offset + len(marker):]))
        except json.JSONDecodeError:
            continue
    return records


def analyze(root: Path,
            ranking_analysis: dict[str, Any] | None = None) -> dict[str, Any]:
    logs = sorted(root.rglob("gateway.log"))
    if not logs:
        raise ValueError(f"no gateway.log files under {root}")
    decision_us: list[float] = []
    fidelity_failures = 0
    totals = {
        family: {
            "observations": 0,
            "ready_observations": 0,
            "covered": 0,
            "ready_covered": 0,
            "predicted_safe": 0,
            "false_safe": 0,
            "absolute_error_sum": 0.0,
            "squared_error_sum": 0.0,
            "ready_absolute_error_sum": 0.0,
            "ready_squared_error_sum": 0.0,
        }
        for family in FAMILIES
    }
    direction_observations = {direction: 0 for direction in DIRECTIONS}
    completion_totals = {
        direction: _empty_completion_totals()
        for direction in DIRECTIONS
    }
    completion_pair_totals = {
        pair: _empty_completion_totals()
        for pair in COMPLETION_PAIRS
    }
    conformal_runs = {pair: [] for pair in COMPLETION_PAIRS}
    populated_logs = 0
    for log in logs:
        records = _metric_records(log)
        if not records:
            continue
        populated_logs += 1
        decision_us.extend(
            max(0.0, float(record.get("host_scheduler_decision_us", 0.0)))
            for record in records)
        fidelity_failures += max(
            int(record.get("global_action_fidelity_violations", 0))
            for record in records)
        final = records[-1]
        for family in FAMILIES:
            target = totals[family]
            target["observations"] += int(
                final.get(f"contextual_{family}_calibration_observations", 0))
            target["ready_observations"] += int(
                final.get(
                    f"contextual_{family}_ready_calibration_observations", 0))
            target["covered"] += int(
                final.get(f"contextual_{family}_interval_covered", 0))
            target["ready_covered"] += int(
                final.get(f"contextual_{family}_ready_interval_covered", 0))
            target["predicted_safe"] += int(
                final.get(f"contextual_{family}_predicted_safe", 0))
            target["false_safe"] += int(
                final.get(f"contextual_{family}_false_safe", 0))
            for name in ("absolute_error_sum", "squared_error_sum",
                         "ready_absolute_error_sum",
                         "ready_squared_error_sum"):
                target[name] += float(
                    final.get(f"contextual_{family}_{name}", 0.0))
        for direction in DIRECTIONS:
            direction_observations[direction] += int(
                final.get(f"contextual_{direction}_observations", 0))
            source = final.get("contextual_completion", {}).get(direction, {})
            target = completion_totals[direction]
            for name in target:
                target[name] += source.get(name, 0)
        for pair in COMPLETION_PAIRS:
            source = final.get("contextual_completion_pair", {}).get(pair, {})
            target = completion_pair_totals[pair]
            for name in target:
                target[name] += source.get(name, 0)
            conformal = final.get("contextual_completion_conformal",
                                  {}).get(pair, {})
            if conformal:
                conformal_runs[pair].append({
                    "scale":
                    float(conformal.get("scale", 1.0)),
                    "observations":
                    int(conformal.get("observations", 0)),
                    "ready":
                    bool(conformal.get("ready", False)),
                })

    family_results = {}
    for family, values in totals.items():
        observations = int(values["observations"])
        ready = int(values["ready_observations"])
        predicted_safe = int(values["predicted_safe"])
        family_results[family] = {
            **values,
            "mae":
            values["absolute_error_sum"] /
            observations if observations else 0.0,
            "rmse":
            math.sqrt(values["squared_error_sum"] /
                      observations) if observations else 0.0,
            "ready_mae":
            values["ready_absolute_error_sum"] / ready if ready else 0.0,
            "ready_rmse":
            math.sqrt(values["ready_squared_error_sum"] /
                      ready) if ready else 0.0,
            "interval_coverage":
            values["covered"] / observations if observations else 0.0,
            "ready_interval_coverage":
            values["ready_covered"] / ready if ready else 0.0,
            "false_safe_rate":
            values["false_safe"] / predicted_safe if predicted_safe else 0.0,
        }

    def completion_results(
            source: dict[str, dict[str, int | float]]) -> dict[str, Any]:
        results = {}
        for name, values in source.items():
            observations = int(values["observations"])
            ready = int(values["ready_calibration_observations"])
            predicted_safe = int(values["predicted_safe"])
            conformal = int(values["conformal_calibration_observations"])
            conformal_predicted_safe = int(values["conformal_predicted_safe"])
            results[name] = {
                **values,
                "incumbent_mae_us":
                values["incumbent_absolute_error_us"] /
                observations if observations else 0.0,
                "incumbent_rmse_us":
                math.sqrt(values["incumbent_squared_error_us"] /
                          observations) if observations else 0.0,
                "newcomer_mae_us":
                values["newcomer_absolute_error_us"] /
                observations if observations else 0.0,
                "newcomer_rmse_us":
                math.sqrt(values["newcomer_squared_error_us"] /
                          observations) if observations else 0.0,
                "incumbent_interval_coverage":
                values["incumbent_interval_covered"] /
                observations if observations else 0.0,
                "newcomer_interval_coverage":
                values["newcomer_interval_covered"] /
                observations if observations else 0.0,
                "ready_incumbent_mae_us":
                values["ready_incumbent_absolute_error_us"] /
                ready if ready else 0.0,
                "ready_incumbent_rmse_us":
                math.sqrt(values["ready_incumbent_squared_error_us"] /
                          ready) if ready else 0.0,
                "ready_newcomer_mae_us":
                values["ready_newcomer_absolute_error_us"] /
                ready if ready else 0.0,
                "ready_newcomer_rmse_us":
                math.sqrt(values["ready_newcomer_squared_error_us"] /
                          ready) if ready else 0.0,
                "ready_incumbent_interval_coverage":
                values["ready_incumbent_interval_covered"] /
                ready if ready else 0.0,
                "ready_newcomer_interval_coverage":
                values["ready_newcomer_interval_covered"] /
                ready if ready else 0.0,
                "conformal_incumbent_interval_coverage":
                values["conformal_incumbent_interval_covered"] /
                conformal if conformal else 0.0,
                "conformal_newcomer_interval_coverage":
                values["conformal_newcomer_interval_covered"] /
                conformal if conformal else 0.0,
                "conformal_false_safe_rate":
                values["conformal_false_safe"] /
                conformal_predicted_safe if conformal_predicted_safe else 0.0,
                "false_safe_rate":
                values["false_safe"] /
                predicted_safe if predicted_safe else 0.0,
            }
        return results

    completion_direction_results = completion_results(completion_totals)
    completion_pair_results = completion_results(completion_pair_totals)
    conformal_results = {}
    for pair, runs in conformal_runs.items():
        scales = [float(run["scale"]) for run in runs if run["ready"]]
        conformal_results[pair] = {
            "runs": runs,
            "ready_logs": len(scales),
            "median_scale": statistics.median(scales) if scales else 1.0,
            "max_scale": max(scales, default=1.0),
        }
    completion_ready = all(values["ready_calibration_observations"] > 0
                           for values in completion_direction_results.values())
    conformal_evaluable = (all(
        values["conformal_calibration_observations"] > 0
        for values in completion_direction_results.values())
                           and all(values["ready_logs"] > 0
                                   for values in conformal_results.values()))
    ranking = (ranking_analysis or {}).get("contextual_ranking", {})
    ranking_regret_evaluable = bool(
        (ranking_analysis
         or {}).get("summary", {}).get("contextual_ranking_regret_evaluable",
                                       False))
    return {
        "schema_version":
        3,
        "mode":
        "m6_contextual_shadow",
        "logs":
        len(logs),
        "logs_with_metrics":
        populated_logs,
        "scheduler_decision_us": {
            "samples": len(decision_us),
            "mean": statistics.fmean(decision_us) if decision_us else 0.0,
            "p50": _percentile(decision_us, 0.5),
            "p95": _percentile(decision_us, 0.95),
            "p99": _percentile(decision_us, 0.99),
            "max": max(decision_us, default=0.0),
        },
        "action_fidelity_failures":
        fidelity_failures,
        "direction_observations":
        direction_observations,
        "families":
        family_results,
        "completion_directions":
        completion_direction_results,
        "completion_pairs":
        completion_pair_results,
        "completion_conformal":
        conformal_results,
        "completion_calibration_evaluable":
        completion_ready,
        "conformal_calibration_evaluable":
        conformal_evaluable,
        "ranking_regret_evaluable":
        ranking_regret_evaluable,
        "ranking":
        ranking,
        "gate_c_evaluable":
        fidelity_failures == 0 and completion_ready
        and ranking_regret_evaluable,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--ranking-analysis", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        ranking_analysis = json.loads(
            args.ranking_analysis.read_text(encoding="utf-8")) \
            if args.ranking_analysis is not None else None
        result = analyze(args.root, ranking_analysis)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    serialized = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
