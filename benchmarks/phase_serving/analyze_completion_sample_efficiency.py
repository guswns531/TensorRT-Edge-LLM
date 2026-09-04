#!/usr/bin/env python3
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
"""Summarize completion-model authority as calibration budget grows."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_snapshots(paths: list[Path]) -> list[dict[str, Any]]:
    """Load all calibration status snapshots with source lineage."""
    snapshots = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"{path}: calibration payload must be a list")
        by_budget = {}
        for snapshot in payload:
            if not isinstance(snapshot, dict):
                raise ValueError(
                    f"{path}: calibration snapshot must be an object")
            budget = int(snapshot.get("completed_warmup_requests", 0))
            if budget > 0:
                by_budget[budget] = snapshot
        for budget, snapshot in by_budget.items():
            snapshots.append({
                "budget": budget,
                "snapshot": snapshot,
                "source": str(path)
            })
    return snapshots


def _median(samples: list[dict[str, Any]], field: str) -> float | None:
    values = [float(sample[field]) for sample in samples if field in sample]
    return statistics.median(values) if values else None


def summarize(paths: list[Path]) -> dict[str, Any]:
    """Aggregate direction-level sample, error, and authority progression."""
    grouped: dict[tuple[int, str, str], list[dict[str,
                                                  Any]]] = defaultdict(list)
    sources_by_budget: dict[int, set[str]] = defaultdict(set)
    for item in load_snapshots(paths):
        budget = int(item["budget"])
        snapshot = item["snapshot"]
        sources_by_budget[budget].add(str(item["source"]))
        families = snapshot.get("contextual_policy_calibration", {})
        for family, family_state in families.items():
            for direction in family_state.get("directions", []):
                grouped[(budget, str(family),
                         str(direction["direction"]))].append(direction)

    points = []
    for (budget, family, direction), samples in sorted(grouped.items()):
        model_error = [
            float(
                sample.get(
                    "completion_authority_incumbent_completion_absolute_error_us",
                    0.0)) +
            float(
                sample.get(
                    "completion_authority_newcomer_completion_absolute_error_us",
                    0.0)) for sample in samples
        ]
        reference_error = [
            float(
                sample.get(
                    "completion_authority_incumbent_reference_absolute_error_us",
                    0.0)) +
            float(
                sample.get(
                    "completion_authority_newcomer_reference_absolute_error_us",
                    0.0)) for sample in samples
        ]
        relative_error = [
            model / reference
            for model, reference in zip(model_error, reference_error)
            if reference > 0.0
        ]
        points.append({
            "budget":
            budget,
            "family":
            family,
            "direction":
            direction,
            "runs":
            len(samples),
            "posterior_observations_median":
            _median(samples, "completion_posterior_observations"),
            "authority_window_observations_median":
            _median(samples, "completion_authority_window_observations"),
            "authority_ready_fraction":
            sum(
                bool(sample.get("completion_authority_evidence_ready", False))
                for sample in samples) / len(samples),
            "authority_validated_fraction":
            sum(
                bool(sample.get("completion_authority_validated", False))
                for sample in samples) / len(samples),
            "incumbent_blend_median":
            _median(samples, "completion_authority_incumbent_blend_weight"),
            "newcomer_blend_median":
            _median(samples, "completion_authority_newcomer_blend_weight"),
            "model_to_reference_absolute_error_median":
            statistics.median(relative_error) if relative_error else None,
            "false_safe_total":
            sum(
                int(sample.get("completion_authority_false_safe", 0))
                for sample in samples),
        })
    return {
        "schema_version": 1,
        "artifact": "completion_sample_efficiency",
        "inputs": [str(path) for path in paths],
        "runs_by_budget": {
            str(budget): len(sources)
            for budget, sources in sorted(sources_by_budget.items())
        },
        "points": points,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = summarize(args.inputs)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) +
                               "\n",
                               encoding="utf-8")
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    print(
        json.dumps(
            {
                "runs_by_budget": result["runs_by_budget"],
                "points": len(result["points"]),
            },
            indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
