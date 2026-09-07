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
"""Compare logical decision and dispatch sequences across policy branches."""

from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import re
from pathlib import Path
from typing import Any, TextIO

PREFIX = "PHASE_SCHEDULER_EVENT\t"
DECISION_COST = re.compile(
    r"Phase global scheduler decision cost: samples=(\d+) mean=([0-9.]+) us "
    r"p95=([0-9.]+) us max=([0-9.]+) us")
COHORT_FIELDS = ("encoder_rows", "prefill_rows", "prefill_tokens",
                 "decode_rows", "decode_context_tokens")


def _open(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open(encoding="utf-8", errors="replace")


def _cohort(value: Any) -> tuple[int, ...]:
    payload = value if isinstance(value, dict) else {}
    return tuple(int(payload.get(field, 0)) for field in COHORT_FIELDS)


def _candidate(candidate: dict[str, Any]) -> tuple[Any, ...]:
    return (str(candidate.get("action_kind", "none")),
            str(candidate.get("action_direction", "none")),
            int(candidate.get("primary_batch_size", 0)),
            int(candidate.get("secondary_batch_size", 0)),
            int(candidate.get("chunk_length", 0)),
            int(candidate.get("primary_context_bucket", 0)),
            int(candidate.get("secondary_context_bucket", 0)),
            str(candidate.get("execution_variant", "eager")),
            int(candidate.get("primary_work_class", 0)),
            tuple(int(value) for value in candidate.get("request_ids", [])),
            str(candidate.get("residual_anchor", "none")),
            bool(candidate.get("residual_augmentation", False)),
            bool(candidate.get("legal", True)))


def _decision(event: dict[str, Any]) -> tuple[Any, ...]:
    candidates = tuple(sorted(_candidate(item)
                              for item in event.get("candidates", [])))
    return (int(event.get("strict_snapshot_signature",
                          event.get("snapshot_signature", 0))),
            int(event.get("kv_ownership_signature", 0)),
            int(event.get("vision_lease_signature", 0)),
            str(event.get("action_kind", "none")),
            tuple(int(value) for value in event.get("request_ids", [])),
            _cohort(event.get("selected_cohort")),
            str(event.get("dispatch_mode", "none")), candidates)


def _dispatch(event: dict[str, Any]) -> tuple[Any, ...]:
    return (str(event.get("phase", "none")),
            tuple(int(value) for value in event.get("request_ids", [])),
            _cohort(event.get("cohort")),
            str(event.get("action_direction", "none")),
            str(event.get("dispatch_mode", "none")),
            int(event.get("requested_start_skew_percent", -1)),
            int(event.get("observed_start_skew_percent", -1)),
            int(event.get("planned_outstanding_mask", 0)))


def _signature(records: list[tuple[Any, ...]]) -> str:
    """Hash stable logical records while excluding all host/GPU timestamps."""
    payload = json.dumps(records, separators=(",", ":"), sort_keys=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_decision_cost(path: Path) -> dict[str, Any] | None:
    candidates = [path]
    if path.name.startswith("run-") and "events.jsonl" in path.name:
        candidates.append(path.parent.parent / "run-001" / "gateway.log")
    for candidate in candidates:
        if not candidate.is_file():
            continue
        with _open(candidate) as source:
            for line in source:
                match = DECISION_COST.search(line)
                if match:
                    return {
                        "samples": int(match.group(1)),
                        "mean_us": float(match.group(2)),
                        "p95_us": float(match.group(3)),
                        "max_us": float(match.group(4)),
                    }
    return None


def load(path: Path) -> dict[str, Any]:
    """Load stable logical records and optional host decision-cost summary."""
    decisions = []
    dispatches = []
    scalar_selection_comparisons = 0
    scalar_selection_mismatches = 0
    with _open(path) as source:
        for line in source:
            if line.startswith(PREFIX):
                event = json.loads(line.removeprefix(PREFIX))
                if event.get("event_kind") == "decision":
                    decisions.append(_decision(event))
                    scalar = event.get("scalar_h1_selected_action_id")
                    selected = event.get("selected_action_id")
                    if scalar is not None and selected is not None:
                        scalar_selection_comparisons += 1
                        scalar_selection_mismatches += scalar != selected
                elif event.get("event_kind") == "dispatch":
                    dispatches.append(_dispatch(event))
    return {
        "path": str(path),
        "decisions": decisions,
        "dispatches": dispatches,
        "decision_signature": _signature(decisions),
        "dispatch_signature": _signature(dispatches),
        "decision_cost": _load_decision_cost(path),
        "scalar_selection_comparisons": scalar_selection_comparisons,
        "scalar_selection_mismatches": scalar_selection_mismatches,
    }


def _compare_sequence(left: list[tuple[Any, ...]],
                      right: list[tuple[Any, ...]]) -> dict[str, Any]:
    common_prefix = 0
    for lhs, rhs in zip(left, right):
        if lhs != rhs:
            break
        common_prefix += 1
    mismatches = [index for index, (lhs, rhs) in enumerate(zip(left, right))
                  if lhs != rhs]
    return {
        "left_count": len(left),
        "right_count": len(right),
        "count_delta": len(right) - len(left),
        "common_prefix": common_prefix,
        "aligned_mismatches": len(mismatches),
        "first_mismatch": mismatches[0] if mismatches else None,
        "exact": left == right,
        "multiset_equal": collections.Counter(left) == collections.Counter(right),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", type=Path, action="append", required=True)
    parser.add_argument("--right", type=Path, action="append", required=True)
    parser.add_argument("--left-name", default="left")
    parser.add_argument("--right-name", default="right")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.left) != len(args.right):
        parser.error("left and right must contain the same number of runs")

    runs = []
    for left_path, right_path in zip(args.left, args.right):
        left = load(left_path)
        right = load(right_path)
        runs.append({
            args.left_name: {
                "path": left["path"],
                "decision_cost": left["decision_cost"],
                "scalar_selection_comparisons": left[
                    "scalar_selection_comparisons"],
                "scalar_selection_mismatches": left[
                    "scalar_selection_mismatches"],
                "logical_decision_signature": left["decision_signature"],
                "logical_dispatch_signature": left["dispatch_signature"],
            },
            args.right_name: {
                "path": right["path"],
                "decision_cost": right["decision_cost"],
                "scalar_selection_comparisons": right[
                    "scalar_selection_comparisons"],
                "scalar_selection_mismatches": right[
                    "scalar_selection_mismatches"],
                "logical_decision_signature": right["decision_signature"],
                "logical_dispatch_signature": right["dispatch_signature"],
            },
            "decisions": _compare_sequence(left["decisions"],
                                           right["decisions"]),
            "dispatches": _compare_sequence(left["dispatches"],
                                            right["dispatches"]),
        })
    artifact = {
        "schema_version": 1,
        "left": args.left_name,
        "right": args.right_name,
        "run_pairs": len(runs),
        "exact_decision_pairs": sum(run["decisions"]["exact"] for run in runs),
        "exact_dispatch_pairs": sum(run["dispatches"]["exact"] for run in runs),
        "multiset_dispatch_pairs": sum(
            run["dispatches"]["multiset_equal"] for run in runs),
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps({key: value for key, value in artifact.items()
                      if key != "runs"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
