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
"""Replay measured directional pairs to their earliest H1 completion boundary."""

from __future__ import annotations

import argparse
import collections
import json
import math
import statistics
from pathlib import Path
from typing import Any

RECORD_PREFIX = "PHASE_SCHEDULER_EVENT\t"
PHASE_MASKS = {"encoder": 1, "prefill": 2, "decode": 4}


def load_events(paths: list[Path]) -> list[dict[str, Any]]:
    """Load unified scheduler events without retaining unrelated log text."""
    events: list[dict[str, Any]] = []
    for path in paths:
        with path.open(encoding="utf-8", errors="replace") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.startswith(RECORD_PREFIX):
                    continue
                event = json.loads(line.removeprefix(RECORD_PREFIX))
                event["_source_path"] = str(path)
                event["_source"] = f"{path}:{line_number}"
                events.append(event)
    return events


def percentile(values: list[float], fraction: float) -> float:
    """Return a linearly interpolated percentile."""
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position -
                                                                 lower)


def distribution(values: list[float]) -> dict[str, float | int]:
    """Summarize one non-empty measured distribution."""
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p95": percentile(values, 0.95),
        "minimum": min(values),
        "maximum": max(values),
    }


def build_replay(
        events: list[dict[str, Any]]) -> tuple[dict[str, Any], list[str]]:
    """Join directional dispatches with both measured completion components."""
    errors: list[str] = []
    completions: dict[tuple[str, str, str, int], dict[str, Any]] = {}
    for event in events:
        if event.get("event_kind") != "completion" or event.get(
                "phase") not in PHASE_MASKS:
            continue
        key = (str(event["_source_path"]), str(event.get("run_id")),
               str(event["phase"]), int(event.get("execution_id", 0)))
        completions[key] = event

    vectors: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int, int]] = set()
    earliest_counts: collections.Counter[str] = collections.Counter()
    for dispatch in events:
        incumbent_phase = str(dispatch.get("incumbent_phase"))
        newcomer_phase = str(dispatch.get("phase"))
        if dispatch.get("event_kind") != "dispatch" or incumbent_phase not in PHASE_MASKS \
                or newcomer_phase not in PHASE_MASKS:
            continue
        source_path = str(dispatch["_source_path"])
        run_id = str(dispatch.get("run_id"))
        incumbent_id = int(dispatch.get("incumbent_execution_id", 0))
        newcomer_id = int(dispatch.get("execution_id", 0))
        identity = (source_path, run_id, incumbent_id, newcomer_id)
        if identity in seen:
            continue
        seen.add(identity)
        incumbent = completions.get(
            (source_path, run_id, incumbent_phase, incumbent_id))
        newcomer = completions.get(
            (source_path, run_id, newcomer_phase, newcomer_id))
        if incumbent is None or newcomer is None:
            errors.append(
                f"{dispatch['_source']}: missing incumbent/newcomer completion"
            )
            continue
        required = {"gpu_start_us", "gpu_end_us", "gpu_duration_us"}
        if not required.issubset(incumbent) or not required.issubset(newcomer):
            errors.append(
                f"{dispatch['_source']}: completion vector lacks a common GPU epoch"
            )
            continue
        if not dispatch.get("action_fidelity", False) or not incumbent.get("action_fidelity", False) \
                or not newcomer.get("action_fidelity", False):
            errors.append(
                f"{dispatch['_source']}: completion vector violates action fidelity"
            )
            continue

        components = [
            {
                "phase": incumbent_phase,
                "execution_id": incumbent_id,
                "gpu_start_us": float(incumbent["gpu_start_us"]),
                "gpu_end_us": float(incumbent["gpu_end_us"]),
                "incumbent": True,
            },
            {
                "phase": newcomer_phase,
                "execution_id": newcomer_id,
                "gpu_start_us": float(newcomer["gpu_start_us"]),
                "gpu_end_us": float(newcomer["gpu_end_us"]),
                "incumbent": False,
            },
        ]
        # H1 starts when the newcomer is actually admitted into the existing
        # outstanding set. Charging the incumbent from its original CUDA
        # start double-counts work already completed before this decision.
        origin = float(newcomer["gpu_start_us"])
        if float(incumbent["gpu_end_us"]) <= origin:
            errors.append(
                f"{dispatch['_source']}: incumbent completed before the incremental boundary"
            )
            continue
        for component in components:
            component["completion_us"] = component["gpu_end_us"] - origin
            component["uncertainty_us"] = 0.0
        components.sort(
            key=lambda component: (component["completion_us"], PHASE_MASKS[
                component["phase"]], component["execution_id"]))
        earliest = components[0]
        remaining = components[1]
        earliest_counts[str(earliest["phase"])] += 1
        boundary_us = float(earliest["completion_us"])
        whole_action_us = max(
            float(component["completion_us"]) for component in components)
        vectors.append({
            "incremental_action_id":
            int(dispatch.get("incremental_action_id", 0)),
            "plan_id":
            int(dispatch.get("plan_id", 0)),
            "requested_direction":
            str(dispatch.get("requested_action_direction")),
            "planned_outstanding_mask":
            int(dispatch.get("planned_outstanding_mask", 0)),
            "incremental_boundary_gpu_us":
            origin,
            "components":
            components,
            "projected_boundary_us":
            boundary_us,
            "whole_action_completion_us":
            whole_action_us,
            "whole_action_overrun_us":
            whole_action_us - boundary_us,
            "completed_phase":
            earliest["phase"],
            "successor_outstanding_mask":
            PHASE_MASKS[str(remaining["phase"])],
            "remaining_phase":
            remaining["phase"],
            "remaining_completion_us":
            float(remaining["completion_us"]) - boundary_us,
        })

    overruns = [float(vector["whole_action_overrun_us"]) for vector in vectors]
    artifact = {
        "schema_version": 1,
        "projector": "earliest_completion_h1",
        "vectors": vectors,
        "summary": {
            "directional_pairs":
            len(vectors),
            "earliest_phase_counts":
            dict(sorted(earliest_counts.items())),
            "nonzero_whole_action_mismatches":
            sum(overrun > 1e-6 for overrun in overruns),
            "whole_action_overrun_us":
            distribution(overruns),
            "invalid_successor_masks":
            sum(
                int(vector["successor_outstanding_mask"]).bit_count() != 1
                for vector in vectors),
            "triple_phase_successors":
            sum(
                int(vector["successor_outstanding_mask"]).bit_count() > 2
                for vector in vectors),
        },
    }
    return artifact, errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    artifact, errors = build_replay(load_events(args.logs))
    if errors:
        for error in errors:
            print(error)
        return 1
    if artifact["summary"]["directional_pairs"] == 0:
        print("no directional completion vectors found")
        return 1
    payload = json.dumps(artifact, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(json.dumps(artifact["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
