#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Pair deterministic phase frontiers and compare realized branch horizons."""

from __future__ import annotations

import argparse
import gzip
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, TextIO

PREFIX = "PHASE_SCHEDULER_EVENT\t"


def open_text(path: Path) -> TextIO:
    """Open plain or gzip telemetry as text."""
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open(encoding="utf-8", errors="replace")


def load_branch(paths: list[Path],
                forced_only: bool = False) -> dict[int, list[dict[str, Any]]]:
    """Index strict pre-branch states and validate each realized dispatch."""
    plans: dict[tuple[str, int], dict[str, Any]] = {}
    completions: dict[tuple[str, int], list[dict[str,
                                                 Any]]] = defaultdict(list)
    for path in paths:
        with open_text(path) as source:
            for line in source:
                if not line.startswith(PREFIX):
                    continue
                event = json.loads(line.removeprefix(PREFIX))
                key = (str(path), int(event.get("plan_id", 0)))
                if event.get("event_kind") == "decision":
                    if forced_only and not event.get("causal_replay_forced",
                                                     False):
                        continue
                    plans[key] = event
                elif event.get("event_kind") == "completion":
                    completions[key].append(event)
    snapshots: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for key, decision in plans.items():
        done = completions.get(key, [])
        intervals = [(float(item["gpu_start_us"]), float(item["gpu_end_us"]))
                     for item in done
                     if "gpu_start_us" in item and "gpu_end_us" in item]
        if not intervals:
            continue
        signature = int(
            decision.get("strict_snapshot_signature",
                         decision.get("snapshot_signature", 0)))
        dispatch_signature = int(decision.get("dispatch_signature", 0))
        completion_dispatch_signatures = [
            int(item.get("dispatch_signature", 0)) for item in done
        ]
        dispatch_identity = dispatch_signature > 0 and all(
            value == dispatch_signature
            for value in completion_dispatch_signatures)
        snapshots[signature].append({
            "action":
            decision.get("action_kind", "unknown"),
            "dispatch_signature":
            dispatch_signature,
            "request_ids":
            decision.get("request_ids", []),
            "token_work":
            decision.get("selected_cohort", {}),
            "horizon_us":
            max(end for _, end in intervals) - min(start
                                                   for start, _ in intervals),
            "fidelity":
            all(item.get("action_fidelity", False) for item in done)
            and dispatch_identity,
        })
    return snapshots


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch-a",
                        type=Path,
                        action="append",
                        required=True)
    parser.add_argument("--branch-b",
                        type=Path,
                        action="append",
                        required=True)
    parser.add_argument("--name-a", default="branch_a")
    parser.add_argument("--name-b", default="branch_b")
    parser.add_argument(
        "--forced-only",
        action="store_true",
        help="pair only decisions explicitly selected by causal replay")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    left = load_branch(args.branch_a, args.forced_only)
    right = load_branch(args.branch_b, args.forced_only)
    pairs = []
    for signature in sorted(left.keys() & right.keys()):
        for lhs, rhs in zip(left[signature], right[signature]):
            # The strict snapshot already commits the full ready frontier,
            # row order, policy state, and ownership. Selected request IDs and
            # work are expected to differ across counterfactual branches.
            pairs.append({
                "strict_snapshot_signature":
                signature,
                args.name_a:
                lhs,
                args.name_b:
                rhs,
                "horizon_delta_us":
                rhs["horizon_us"] - lhs["horizon_us"],
            })
    deltas = [pair["horizon_delta_us"] for pair in pairs]
    artifact = {
        "schema_version":
        1,
        "branch_a":
        args.name_a,
        "branch_b":
        args.name_b,
        "matched_snapshot_pairs":
        len(pairs),
        "action_disagreements":
        sum(pair[args.name_a]["action"] != pair[args.name_b]["action"]
            for pair in pairs),
        "dispatch_identity_failures":
        sum(not pair[args.name_a]["fidelity"]
            or not pair[args.name_b]["fidelity"] for pair in pairs),
        "fidelity_failures":
        sum(not pair[args.name_a]["fidelity"]
            or not pair[args.name_b]["fidelity"] for pair in pairs),
        "horizon_delta_us": {
            "median": statistics.median(deltas) if deltas else None,
            "mean": statistics.fmean(deltas) if deltas else None,
            "minimum": min(deltas) if deltas else None,
            "maximum": max(deltas) if deltas else None,
        },
        "pairs":
        pairs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True) +
                           "\n",
                           encoding="utf-8")
    print(
        json.dumps(
            {
                key: value
                for key, value in artifact.items() if key != "pairs"
            },
            indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
