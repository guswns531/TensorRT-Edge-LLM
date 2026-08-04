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
"""Sweep real text requests through the two-queue phase scheduler."""

import argparse
import csv
import json
import math
import random
import statistics
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Scenario:
    name: str
    arrival_rate: float
    chunk_size: int
    prefill_batch: int
    decode_batch: int
    adaptive_scheduler: bool = False
    adaptive_chunking: bool = False


SCENARIOS = (
    Scenario("whole_p2d2_10rps", 10, 1024, 2, 2),
    Scenario("fixed128_p2d2_10rps", 10, 128, 2, 2),
    Scenario("adaptive256_p2d2_10rps", 10, 256, 2, 2, True, True),
    Scenario("whole_p2d2_30rps", 30, 1024, 2, 2),
    Scenario("fixed128_p2d2_30rps", 30, 128, 2, 2),
    Scenario("fixed64_p2d2_30rps", 30, 64, 2, 2),
    Scenario("fixed256_p2d2_30rps", 30, 256, 2, 2),
    Scenario("adaptive256_p2d2_30rps", 30, 256, 2, 2, True, True),
    Scenario("adaptive256_p1d3_30rps", 30, 256, 1, 3, True, True),
    Scenario("adaptive256_p3d1_30rps", 30, 256, 3, 1, True, True),
    Scenario("adaptive256_p2d2_1000rps", 1000, 256, 2, 2, True, True),
)


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
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


def summarize_kernel_groups(scenario: Scenario, path: Path) -> list[dict[str, object]]:
    grouped: dict[str, list[float]] = {}
    for row in read_csv(path):
        grouped.setdefault(row["group"], []).append(float(row["gpu_ms"]))
    return [{
        "scenario": scenario.name,
        "group": group,
        "count": len(values),
        "total_gpu_ms": sum(values),
        "mean_gpu_ms": statistics.mean(values),
        "median_gpu_ms": statistics.median(values),
        "p95_gpu_ms": percentile(values, 0.95),
    } for group, values in sorted(grouped.items())]


def summarize(scenario: Scenario, request_csv: Path,
              dispatch_csv: Path) -> dict[str, object]:
    requests = read_csv(request_csv)
    dispatches = read_csv(dispatch_csv)
    terminal_us = max(int(row["completed_us"]) for row in requests)
    generated_tokens = sum(int(row["output_tokens"]) for row in requests)
    ttft_ms = [float(row["ttft_us"]) / 1000.0 for row in requests]
    e2e_ms = [float(row["completed_us"]) / 1000.0 for row in requests]
    tpot_ms = [float(row["tpot_us"]) / 1000.0 for row in requests]
    prefill_batches = [int(row["prefill_batch"]) for row in dispatches
                       if int(row["prefill_batch"]) > 0]
    decode_batches = [int(row["decode_batch"]) for row in dispatches
                      if int(row["decode_batch"]) > 0]
    prefill_chunks = [int(row["prefill_tokens"]) // int(row["prefill_batch"])
                      for row in dispatches if int(row["prefill_batch"]) > 0]
    overlaps = sum(int(row["prefill_batch"]) > 0 and
                   int(row["decode_batch"]) > 0 for row in dispatches)
    return {
        "scenario": scenario.name,
        "requests": len(requests),
        "arrival_rate_rps": scenario.arrival_rate,
        "chunk_limit": scenario.chunk_size,
        "prefill_batch_limit": scenario.prefill_batch,
        "decode_batch_limit": scenario.decode_batch,
        "adaptive_scheduler": scenario.adaptive_scheduler,
        "adaptive_chunking": scenario.adaptive_chunking,
        "duration_ms": terminal_us / 1000.0,
        "achieved_rps": len(requests) * 1_000_000.0 / terminal_us,
        "generated_token_s": generated_tokens * 1_000_000.0 / terminal_us,
        "ttft_median_ms": statistics.median(ttft_ms),
        "ttft_p95_ms": percentile(ttft_ms, 0.95),
        "e2e_median_ms": statistics.median(e2e_ms),
        "e2e_p95_ms": percentile(e2e_ms, 0.95),
        "tpot_median_ms": statistics.median(tpot_ms),
        "tpot_p95_ms": percentile(tpot_ms, 0.95),
        "dispatches": len(dispatches),
        "overlap_dispatches": overlaps,
        "prefill_batch_mean": statistics.mean(prefill_batches),
        "decode_batch_mean": statistics.mean(decode_batches),
        "prefill_chunk_median": statistics.median(prefill_chunks),
    }


def materialize_poisson_trace(source: Path, destination: Path,
                              arrival_rate: float, seed: int) -> None:
    with source.open(encoding="utf-8") as stream:
        trace = json.load(stream)
    generator = random.Random(seed)
    elapsed_us = 0.0
    for index, request in enumerate(trace["requests"]):
        if index > 0:
            elapsed_us += generator.expovariate(arrival_rate) * 1_000_000.0
        request["arrival_offset_us"] = round(elapsed_us)
    with destination.open("w", encoding="utf-8") as stream:
        json.dump(trace, stream, indent=4)
        stream.write("\n")


def command_for(args: argparse.Namespace, scenario: Scenario,
                scenario_dir: Path, trace_file: Path) -> list[str]:
    command = [
        str(args.bench), "--engineDir", str(args.engine_dir),
        "--prefillBatch", str(scenario.prefill_batch),
        "--decodeBatch", str(scenario.decode_batch),
        "--inputLen", "1024", "--prefillChunkSize", str(scenario.chunk_size),
        "--pastKVLen", "128", "--warmup", str(args.warmup),
        "--iterations", str(args.iterations), "--contextAdapter",
        "--trtContextMode", args.context_mode, "--inputFile", str(trace_file),
        "--traceCsv", str(scenario_dir / "requests.csv"),
        "--traceArrivalRate", str(scenario.arrival_rate),
        "--kernelGroupCsv", str(scenario_dir / "kernel-groups.csv"),
    ]
    if scenario.adaptive_scheduler:
        command.append("--adaptiveScheduler")
    if scenario.adaptive_chunking:
        command.append("--adaptiveChunking")
    return command


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", type=Path, required=True)
    parser.add_argument("--engine-dir", type=Path, required=True)
    parser.add_argument("--input-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--context-mode", choices=("shared", "independent"),
                        default="independent")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scenario", action="append",
                        choices=tuple(item.name for item in SCENARIOS))
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    selected = [item for item in SCENARIOS
                if args.scenario is None or item.name in args.scenario]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    group_summaries = []
    for index, scenario in enumerate(selected, start=1):
        scenario_dir = args.output_dir / scenario.name
        scenario_dir.mkdir(parents=True, exist_ok=True)
        request_csv = scenario_dir / "requests.csv"
        dispatch_csv = scenario_dir / "requests-dispatch.csv"
        kernel_csv = scenario_dir / "kernel-groups.csv"
        trace_file = scenario_dir / "trace.json"
        log_path = scenario_dir / "run.log"
        required = (request_csv, dispatch_csv, kernel_csv)
        if not args.skip_existing or not all(path.exists() for path in required):
            materialize_poisson_trace(args.input_file, trace_file,
                                      scenario.arrival_rate, args.seed + index)
            print(f"[{index}/{len(selected)}] running {scenario.name}", flush=True)
            with log_path.open("w", encoding="utf-8") as log:
                result = subprocess.run(command_for(args, scenario, scenario_dir,
                                                    trace_file),
                                        stdout=log, stderr=subprocess.STDOUT,
                                        check=False, text=True)
            if result.returncode != 0:
                tail = "\n".join(log_path.read_text(
                    encoding="utf-8", errors="replace").splitlines()[-30:])
                raise RuntimeError(
                    f"{scenario.name} failed with exit code {result.returncode}:\n{tail}")
        summary = summarize(scenario, request_csv, dispatch_csv)
        summaries.append(summary)
        group_summaries.extend(summarize_kernel_groups(scenario, kernel_csv))
        print(f"[{index}/{len(selected)}] {scenario.name}: "
              f"{summary['achieved_rps']:.2f} req/s, "
              f"TTFT p95={summary['ttft_p95_ms']:.2f} ms, "
              f"TPOT p95={summary['tpot_p95_ms']:.2f} ms", flush=True)

    for name, rows in (("summary.csv", summaries),
                       ("kernel-group-summary.csv", group_summaries)):
        path = args.output_dir / name
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"summary written to {path}", flush=True)


if __name__ == "__main__":
    main()
