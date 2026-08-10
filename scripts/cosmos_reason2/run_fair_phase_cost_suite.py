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
"""Run the Cosmos fixed-shape phase sweep and aggregate CUDA-event costs.

The Cosmos text decoder currently uses the legacy fixed-slot cache path.  The
independent sweep therefore only includes pairs whose two active phase batches
fit in the physical slot capacity.  Capacity points (prefill/decode BS16) are
also measured with a shared TensorRT context; they are phase-cost points, not
overlap claims, because the two phases intentionally reuse the 16 slots.
"""

import argparse
import csv
import math
import statistics
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Scenario:
    name: str
    profile: str
    prefill_batch: int
    decode_batch: int
    context_mode: str


OVERLAP_SCENARIOS = tuple(
    Scenario(f"overlap_p{batch}_d{batch}", "independent_overlap", batch, batch,
             "independent") for batch in (1, 2, 4, 8))
PREFILL_CAPACITY_SCENARIOS = tuple(
    Scenario(f"prefill_capacity_bs{batch}", "phase_capacity_shared", batch, 1,
             "shared") for batch in (1, 2, 4, 8, 16))
DECODE_CAPACITY_SCENARIOS = tuple(
    Scenario(f"decode_capacity_bs{batch}", "phase_capacity_shared", 1, batch,
             "shared") for batch in (1, 2, 4, 8, 16))
SCENARIOS = OVERLAP_SCENARIOS + PREFILL_CAPACITY_SCENARIOS + DECODE_CAPACITY_SCENARIOS


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    rank = (len(ordered) - 1) * fraction
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (rank - lower)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise RuntimeError(f"empty CSV: {path}")
    return rows


def benchmark_command(args: argparse.Namespace, scenario: Scenario,
                      output_dir: Path) -> list[str]:
    return [
        str(args.bench),
        "--engineDir",
        str(args.engine_dir),
        "--prefillBatch",
        str(scenario.prefill_batch),
        "--decodeBatch",
        str(scenario.decode_batch),
        "--slotCount",
        str(args.slot_count),
        "--inputLen",
        str(args.input_len),
        "--prefillChunkSize",
        str(args.chunk_size),
        "--pastKVLen",
        str(args.past_kv_len),
        "--warmup",
        str(args.warmup),
        "--iterations",
        str(args.iterations),
        "--trtContextMode",
        scenario.context_mode,
        "--outputCsv",
        str(output_dir / "phase.csv"),
        "--kernelGroupCsv",
        str(output_dir / "kernel-groups.csv"),
    ]


def phase_summary(scenario: Scenario, path: Path, warmup: int,
                  iterations: int) -> list[dict[str, object]]:
    rows = read_csv(path)
    expected = iterations * 2
    if len(rows) != expected:
        raise RuntimeError(
            f"{path}: expected {expected} measured rows, got {len(rows)}")
    result = []
    for mode in ("sequential", "independent_trt_concurrent",
                 "shared_trt_serialized"):
        selected = [row for row in rows if row["mode"] == mode]
        if not selected:
            continue
        result.append({
            "scenario":
            scenario.name,
            "profile":
            scenario.profile,
            "context_mode":
            scenario.context_mode,
            "prefill_batch":
            scenario.prefill_batch,
            "decode_batch":
            scenario.decode_batch,
            "mode":
            mode,
            "samples":
            len(selected),
            "makespan_median_ms":
            statistics.median(float(row["makespan_ms"]) for row in selected),
            "makespan_p95_ms":
            percentile([float(row["makespan_ms"]) for row in selected], 0.95),
            "prefill_median_ms":
            statistics.median(float(row["prefill_ms"]) for row in selected),
            "prefill_p95_ms":
            percentile([float(row["prefill_ms"]) for row in selected], 0.95),
            "decode_median_ms":
            statistics.median(float(row["decode_ms"]) for row in selected),
            "decode_p95_ms":
            percentile([float(row["decode_ms"]) for row in selected], 0.95),
            "overlap_fraction_median":
            statistics.median(
                float(row["overlap_fraction"]) for row in selected),
        })
    return result


def kernel_cost_rows(scenario: Scenario, path: Path, warmup: int,
                     iterations: int) -> list[dict[str, object]]:
    rows = read_csv(path)
    # Each run increments scheduler_dispatch_index once and emits one prefill
    # and one decode dispatch.  The benchmark primes both modes, then performs
    # `warmup` pairs before the measured rows, so discard those event samples.
    first_measured_dispatch = 2 * (warmup + 1)
    selected = [
        row for row in rows
        if int(row["scheduler_dispatch_index"]) >= first_measured_dispatch
    ]
    expected_min = iterations * 2
    if len({int(row["scheduler_dispatch_index"])
            for row in selected}) < expected_min:
        raise RuntimeError(
            f"{path}: missing measured kernel dispatches after priming/warmup")
    grouped: dict[tuple[str, str, int, int, int, int], list[float]] = {}
    for row in selected:
        execution = "sequential" if int(
            row["scheduler_kind"]) == 0 else "scheduled"
        key = (
            execution,
            row["group"],
            int(row["prefill_batch"]),
            int(row["decode_batch"]),
            int(row["prefill_tokens"]),
            int(row["decode_context_tokens"]),
        )
        grouped.setdefault(key, []).append(float(row["gpu_ms"]))
    result = []
    for (execution, group, prefill_batch, decode_batch, prefill_tokens,
         decode_context_tokens), values in sorted(grouped.items()):
        result.append({
            "scenario": scenario.name,
            "profile": scenario.profile,
            "context_mode": scenario.context_mode,
            "execution": execution,
            "group": group,
            "prefill_batch": prefill_batch,
            "decode_batch": decode_batch,
            "prefill_tokens": prefill_tokens,
            "decode_context_tokens": decode_context_tokens,
            "samples": len(values),
            "gpu_mean_ms": statistics.mean(values),
            "gpu_median_ms": statistics.median(values),
            "gpu_p95_ms": percentile(values, 0.95),
            "gpu_max_ms": max(values),
        })
    return result


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError(f"no rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", type=Path, required=True)
    parser.add_argument("--engine-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--slot-count", type=int, default=16)
    parser.add_argument("--input-len", type=int, default=512)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--past-kv-len", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--scenario",
                        action="append",
                        choices=tuple(item.name for item in SCENARIOS))
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    selected = [
        item for item in SCENARIOS
        if args.scenario is None or item.name in args.scenario
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    phase_rows: list[dict[str, object]] = []
    cost_rows: list[dict[str, object]] = []
    for index, scenario in enumerate(selected, start=1):
        scenario_dir = args.output_dir / scenario.name
        scenario_dir.mkdir(parents=True, exist_ok=True)
        phase_csv = scenario_dir / "phase.csv"
        kernel_csv = scenario_dir / "kernel-groups.csv"
        log_path = scenario_dir / "run.log"
        if not args.skip_existing or not (phase_csv.exists()
                                          and kernel_csv.exists()):
            print(f"[{index}/{len(selected)}] running {scenario.name}",
                  flush=True)
            with log_path.open("w", encoding="utf-8") as log:
                completed = subprocess.run(
                    benchmark_command(args, scenario, scenario_dir),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                    text=True,
                )
            if completed.returncode != 0:
                tail = "\n".join(
                    log_path.read_text(encoding="utf-8",
                                       errors="replace").splitlines()[-50:])
                raise RuntimeError(
                    f"{scenario.name} failed with {completed.returncode}:\n{tail}"
                )
        phase_rows.extend(
            phase_summary(scenario, phase_csv, args.warmup, args.iterations))
        cost_rows.extend(
            kernel_cost_rows(scenario, kernel_csv, args.warmup,
                             args.iterations))
        print(f"[{index}/{len(selected)}] {scenario.name} complete",
              flush=True)

    write_rows(args.output_dir / "phase-summary.csv", phase_rows)
    write_rows(args.output_dir / "kernel-cost-table.csv", cost_rows)
    print(f"wrote {args.output_dir / 'phase-summary.csv'}", flush=True)
    print(f"wrote {args.output_dir / 'kernel-cost-table.csv'}", flush=True)


if __name__ == "__main__":
    main()
