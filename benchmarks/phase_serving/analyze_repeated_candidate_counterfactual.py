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
"""Find repeated alternative actions at identical phase candidate frontiers."""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import math
import statistics
from pathlib import Path
from typing import Any, TextIO

PREFIX = "PHASE_SCHEDULER_EVENT\t"


def _open(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open(encoding="utf-8", errors="replace")


def _candidate_signature(candidate: dict[str, Any]) -> tuple[Any, ...]:
    return (str(candidate.get("action_kind", "none")),
            str(candidate.get("action_direction", "none")),
            tuple(int(value) for value in candidate.get("request_ids", [])),
            str(candidate.get("residual_anchor", "none")),
            bool(candidate.get("residual_augmentation", False)))


def _frontier_signature(decision: dict[str, Any]) -> tuple[Any, ...]:
    return (int(decision.get("snapshot_signature", 0)),
            tuple(sorted(_candidate_signature(candidate)
                         for candidate in decision.get("candidates", []))))


def _selected_signature(decision: dict[str, Any]) -> tuple[Any, ...]:
    selected_id = decision.get("selected_action_id")
    selected = next((candidate for candidate in decision.get("candidates", [])
                     if candidate.get("action_id") == selected_id), None)
    if selected is not None:
        return _candidate_signature(selected)
    return (str(decision.get("action_kind", "none")), "none",
            tuple(int(value) for value in decision.get("request_ids", [])),
            "none", False)


def _load(paths: list[Path]) -> list[dict[str, Any]]:
    samples = []
    for path in paths:
        decisions: dict[tuple[str, int], dict[str, Any]] = {}
        completions: dict[tuple[str, int], list[dict[str, Any]]] = (
            collections.defaultdict(list))
        with _open(path) as source:
            for line in source:
                if not line.startswith(PREFIX):
                    continue
                event = json.loads(line.removeprefix(PREFIX))
                key = (str(event.get("run_id", "")),
                       int(event.get("plan_id", 0)))
                if event.get("event_kind") == "decision":
                    decisions[key] = event
                elif event.get("event_kind") == "completion":
                    completions[key].append(event)
        for key, decision in decisions.items():
            intervals = [(float(item["gpu_start_us"]),
                          float(item["gpu_end_us"]))
                         for item in completions.get(key, [])
                         if "gpu_start_us" in item and "gpu_end_us" in item]
            if not intervals:
                continue
            samples.append({
                "source": str(path),
                "frontier": _frontier_signature(decision),
                "action": _selected_signature(decision),
                "horizon_us": max(end for _, end in intervals) -
                              min(start for start, _ in intervals),
                "fidelity": all(bool(item.get("action_fidelity", False))
                                for item in completions[key]),
            })
    return samples


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, action="append", required=True)
    parser.add_argument("--minimum-repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.minimum_repeats < 1:
        parser.error("minimum-repeats must be positive")

    grouped: dict[tuple[Any, ...], dict[tuple[Any, ...], list[dict[str, Any]]]] = (
        collections.defaultdict(lambda: collections.defaultdict(list)))
    for sample in _load(args.events):
        grouped[sample["frontier"]][sample["action"]].append(sample)

    frontiers = []
    for frontier, actions in grouped.items():
        repeated = {
            action: values for action, values in actions.items()
            if len(values) >= args.minimum_repeats and
            all(value["fidelity"] for value in values)
        }
        if len(repeated) < 2:
            continue
        action_rows = []
        for action, values in repeated.items():
            horizons = [float(value["horizon_us"]) for value in values]
            action_rows.append({
                "action": action,
                "samples": len(values),
                "median_horizon_us": statistics.median(horizons),
                "p95_horizon_us": sorted(horizons)[
                    max(0, min(len(horizons) - 1,
                               math.ceil(0.95 * len(horizons)) - 1))],
            })
        action_rows.sort(key=lambda row: row["median_horizon_us"])
        frontiers.append({
            "snapshot_signature": frontier[0],
            "candidate_frontier": frontier[1],
            "actions": action_rows,
            "best_action": action_rows[0]["action"],
            "median_regret_us": action_rows[-1]["median_horizon_us"] -
                                action_rows[0]["median_horizon_us"],
        })
    artifact = {
        "schema_version": 1,
        "minimum_repeats": args.minimum_repeats,
        "fully_repeated_multi_action_frontiers": len(frontiers),
        "frontiers": frontiers,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps({key: value for key, value in artifact.items()
                      if key != "frontiers"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
