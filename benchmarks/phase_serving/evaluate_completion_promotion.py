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
"""Evaluate P5 completion authority and the conditional P7 H2 gate."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

METRIC_PREFIX = "PHASE_METRIC\t"


def read_json(path: Path) -> Any:
    """Read one JSON artifact."""
    return json.loads(path.read_text(encoding="utf-8"))


def open_log(path: Path):
    """Open plain or gzip-compressed benchmark logs as text."""
    if path.suffix == ".gz":
        return gzip.open(path, mode="rt", encoding="utf-8", errors="replace")
    return path.open(encoding="utf-8", errors="replace")


def combine_aggregates(paths: list[Path]) -> dict[str, Any]:
    """Combine single-run HTTP aggregates without losing trace identity."""
    payloads = [read_json(path) for path in paths]
    throughputs = [float(item["achieved_req_s_median"]) for item in payloads]
    hashes = [
        str(value) for item in payloads
        for value in item.get("token_trace_sha256_per_run", [])
    ]
    return {
        "achieved_req_s_median":
        statistics.median(throughputs),
        "token_trace_deterministic":
        all(
            bool(item.get("token_trace_deterministic", False))
            for item in payloads),
        "token_trace_sha256_per_run":
        hashes,
    }


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


def load_last_metrics(paths: list[Path]) -> list[dict[str, Any]]:
    """Load the final cumulative phase metric from each gateway log."""
    results = []
    for path in paths:
        last = None
        with open_log(path) as source:
            for line in source:
                offset = line.find(METRIC_PREFIX)
                if offset >= 0:
                    last = json.loads(line[offset + len(METRIC_PREFIX):])
        if last is None:
            raise ValueError(
                f"{path} contains no {METRIC_PREFIX.strip()} record")
        results.append(last)
    return results


def scheduler_p95(paths: list[Path]) -> float:
    """Collect the scheduler decision latency distribution from all records."""
    samples = []
    for path in paths:
        with open_log(path) as source:
            for line in source:
                offset = line.find(METRIC_PREFIX)
                if offset < 0:
                    continue
                metric = json.loads(line[offset + len(METRIC_PREFIX):])
                samples.append(
                    float(metric.get("host_scheduler_decision_us", 0.0)))
    if not samples:
        raise ValueError("no scheduler-decision latency samples")
    result = percentile(samples, 0.95)
    assert result is not None
    return result


def selected_slo_runs(payload: dict[str, Any],
                      label: str) -> list[dict[str, Any]]:
    """Select repeated load-boundary runs using their stable label prefix."""
    runs = [
        run for run in payload.get("runs", [])
        if str(run.get("label", "")).startswith(label)
    ]
    if not runs:
        raise ValueError(f"SLO artifact has no runs beginning with {label!r}")
    return sorted(runs, key=lambda run: str(run.get("label", "")))


def token_trace_identity(off: dict[str, Any], active: dict[str, Any]) -> bool:
    """Require deterministic greedy output and the same token trace."""
    off_hashes = off.get("token_trace_sha256_per_run", [])
    active_hashes = active.get("token_trace_sha256_per_run", [])
    return bool(
        off.get("token_trace_deterministic", False)
        and active.get("token_trace_deterministic", False) and off_hashes
        and active_hashes and set(off_hashes) == set(active_hashes))


def completion_coverage(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate chronological conformal coverage by ordered direction."""
    direction_counts: dict[str, dict[str, int]] = {}
    for metric in metrics:
        for direction, telemetry in metric.get("contextual_completion",
                                               {}).items():
            counts = direction_counts.setdefault(
                direction, {
                    "observations": 0,
                    "incumbent_covered": 0,
                    "newcomer_covered": 0,
                    "false_safe": 0,
                })
            counts["observations"] += int(
                telemetry.get("conformal_calibration_observations", 0))
            counts["incumbent_covered"] += int(
                telemetry.get("conformal_incumbent_interval_covered", 0))
            counts["newcomer_covered"] += int(
                telemetry.get("conformal_newcomer_interval_covered", 0))
            counts["false_safe"] += int(
                telemetry.get("conformal_false_safe", 0))
    calibrated = {}
    for direction, counts in sorted(direction_counts.items()):
        observations = counts["observations"]
        if observations:
            calibrated[direction] = {
                **counts,
                "incumbent_coverage":
                counts["incumbent_covered"] / observations,
                "newcomer_coverage": counts["newcomer_covered"] / observations,
            }
    return {
        "directions":
        calibrated,
        "false_safe":
        sum(counts["false_safe"] for counts in direction_counts.values()),
    }


def gate(name: str, passed: bool, observed: Any,
         required: Any) -> dict[str, Any]:
    """Create a stable machine-readable gate result."""
    return {
        "name": name,
        "passed": passed,
        "observed": observed,
        "required": required,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authority-off-aggregate",
                        type=Path,
                        action="append",
                        required=True)
    parser.add_argument("--authority-active-aggregate",
                        type=Path,
                        action="append",
                        required=True)
    parser.add_argument("--authority-off-slo", type=Path, required=True)
    parser.add_argument("--authority-active-slo", type=Path, required=True)
    parser.add_argument("--authority-off-log",
                        type=Path,
                        action="append",
                        required=True)
    parser.add_argument("--authority-active-log",
                        type=Path,
                        action="append",
                        required=True)
    parser.add_argument("--event-summary", type=Path, required=True)
    parser.add_argument("--opportunity-summary", type=Path, required=True)
    parser.add_argument("--load-label", default="load48.8")
    parser.add_argument("--coverage-target", type=float, default=0.95)
    parser.add_argument("--coverage-tolerance", type=float, default=0.05)
    parser.add_argument("--scheduler-regression-percent",
                        type=float,
                        default=5.0)
    parser.add_argument("--scheduler-regression-floor-us",
                        type=float,
                        default=10.0)
    parser.add_argument("--minimum-throughput", type=float, default=40.0)
    parser.add_argument("--minimum-pass-rate", type=float, default=0.99)
    parser.add_argument("--minimum-h2-comparable-decisions",
                        type=int,
                        default=20)
    parser.add_argument("--minimum-h2-disagreement-fraction",
                        type=float,
                        default=0.01)
    parser.add_argument("--minimum-h2-regret-us", type=float, default=100.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    try:
        off_aggregate = combine_aggregates(args.authority_off_aggregate)
        active_aggregate = combine_aggregates(args.authority_active_aggregate)
        off_runs = selected_slo_runs(read_json(args.authority_off_slo),
                                     args.load_label)
        active_runs = selected_slo_runs(read_json(args.authority_active_slo),
                                        args.load_label)
        event_summary = read_json(args.event_summary)
        opportunity = read_json(args.opportunity_summary)
        active_metrics = load_last_metrics(args.authority_active_log)
        coverage = completion_coverage(active_metrics)
        off_scheduler_p95 = scheduler_p95(args.authority_off_log)
        active_scheduler_p95 = scheduler_p95(args.authority_active_log)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2

    active_throughput = float(active_aggregate["achieved_req_s_median"])
    active_pass_rate = statistics.median(
        float(run["pass_rate"]) for run in active_runs)
    off_goodput = [float(run["request_goodput_per_s"]) for run in off_runs]
    active_goodput = [
        float(run["request_goodput_per_s"]) for run in active_runs
    ]
    equal_repeat_count = len(off_goodput) == len(active_goodput)
    paired = list(zip(off_goodput, active_goodput))
    paired_wins = sum(active > off for off, active in paired)
    repeatable_improvement = (equal_repeat_count and len(paired) >= 3
                              and statistics.median(active_goodput)
                              > statistics.median(off_goodput)
                              and paired_wins >= math.ceil(len(paired) / 2))
    coverage_pass = bool(coverage["directions"]) and all(
        abs(values[name] - args.coverage_target) <= args.coverage_tolerance
        for values in coverage["directions"].values()
        for name in ("incumbent_coverage", "newcomer_coverage"))
    scheduler_limit = max(
        off_scheduler_p95 * (1.0 + args.scheduler_regression_percent / 100.0),
        off_scheduler_p95 + args.scheduler_regression_floor_us)

    gates = [
        gate(
            "token_identity",
            token_trace_identity(off_aggregate, active_aggregate), {
                "off": off_aggregate.get("token_trace_sha256_per_run"),
                "active": active_aggregate.get("token_trace_sha256_per_run"),
            }, "deterministic exact equality"),
        gate("action_fidelity",
             int(event_summary.get("action_fidelity_failures", -1)) == 0,
             event_summary.get("action_fidelity_failures"), 0),
        gate("conformal_false_safe", coverage["false_safe"] == 0,
             coverage["false_safe"], 0),
        gate("held_out_coverage", coverage_pass, coverage["directions"], {
            "target": args.coverage_target,
            "tolerance": args.coverage_tolerance,
        }),
        gate("scheduler_p95_us", active_scheduler_p95 <= scheduler_limit, {
            "off": off_scheduler_p95,
            "active": active_scheduler_p95,
        }, {
            "maximum_active": scheduler_limit,
        }),
        gate("load_throughput_req_s", active_throughput
             >= args.minimum_throughput, active_throughput,
             args.minimum_throughput),
        gate("joint_slo_pass_rate", active_pass_rate >= args.minimum_pass_rate,
             active_pass_rate, args.minimum_pass_rate),
        gate(
            "repeatable_improvement", repeatable_improvement, {
                "off_goodput_median": statistics.median(off_goodput),
                "active_goodput_median": statistics.median(active_goodput),
                "paired_wins": paired_wins,
                "paired_runs": len(paired),
                "equal_repeat_count": equal_repeat_count,
            },
            "at least three equal-count paired runs, a higher median, and a majority of paired wins"
        ),
    ]
    promotion_passed = all(item["passed"] for item in gates)
    disagreement = opportunity.get("selector_disagreement", {})
    regret = opportunity.get("replay_regret_us", {})
    h2_eligible = bool(
        promotion_passed and int(disagreement.get(
            "comparable_decisions", 0)) >= args.minimum_h2_comparable_decisions
        and float(disagreement.get("fraction")
                  or 0.0) >= args.minimum_h2_disagreement_fraction
        and float(regret.get("median") or 0.0) >= args.minimum_h2_regret_us)
    result = {
        "schema_version": 1,
        "p5_completion_authority": {
            "passed": promotion_passed,
            "gates": gates,
            "coverage": coverage,
        },
        "p7_h2_successor": {
            "eligible":
            h2_eligible,
            "requires_p5":
            promotion_passed,
            "comparable_decisions":
            disagreement.get("comparable_decisions", 0),
            "disagreement_fraction":
            disagreement.get("fraction"),
            "replay_regret_median_us":
            regret.get("median"),
            "rule":
            "earliest-completion one-step successor only; no future-arrival prediction",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if promotion_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
