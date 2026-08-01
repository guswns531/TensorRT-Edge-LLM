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
"""Run and summarize reproducible Gemma phase continuous-load workloads."""

import argparse
import csv
import math
import statistics
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Workload:
    name: str
    requests: int
    arrival_rate: float
    prompt_min: int
    prompt_max: int
    output_min: int
    output_max: int
    input_len: int
    chunk_size: int
    prefill_batch: int
    decode_batch: int
    overlap_prefill_tokens: int = 128


WORKLOADS = (
    Workload("steady_10rps", 16, 10, 128, 128, 8, 8, 128, 128, 2, 2),
    Workload("steady_20rps", 16, 20, 128, 128, 8, 8, 128, 128, 2, 2),
    Workload("steady_25rps", 24, 25, 128, 128, 8, 8, 128, 128, 2, 2),
    Workload("steady_30rps", 24, 30, 128, 128, 8, 8, 128, 128, 2, 2),
    Workload("steady_35rps", 24, 35, 128, 128, 8, 8, 128, 128, 2, 2),
    Workload("steady_40rps", 24, 40, 128, 128, 8, 8, 128, 128, 2, 2),
    Workload("burst_1000rps", 24, 1000, 128, 128, 4, 8, 128, 128, 2, 2),
    Workload("long_output", 16, 100, 128, 128, 24, 32, 128, 128, 2, 2),
    Workload("whole_prefill_512", 12, 100, 512, 512, 8, 8, 512, 512, 2, 2),
    Workload("chunked_prefill_512", 12, 100, 512, 512, 8, 8, 512, 128, 2, 2),
    Workload("chunked_prefill_512_overlap256", 12, 100, 512, 512, 8, 8,
             512, 128, 2, 2, 256),
    Workload("mixed_lengths", 16, 100, 128, 512, 4, 16, 512, 128, 2, 2),
    Workload("prefill_batch_limited", 12, 100, 512, 512, 8, 8, 512, 128, 1, 3),
    Workload("decode_batch_limited", 12, 100, 128, 128, 24, 32, 128, 128, 3, 1),
)


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    rank = (len(ordered) - 1) * fraction
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (rank - lower)


def mean_or_zero(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise RuntimeError(f"empty CSV: {path}")
    return rows


def summarize(workload: Workload, request_csv: Path,
              dispatch_csv: Path) -> dict[str, object]:
    requests = read_csv(request_csv)
    dispatches = read_csv(dispatch_csv)
    if len(requests) != workload.requests:
        raise RuntimeError(
            f"{workload.name}: expected {workload.requests} requests, got {len(requests)}"
        )

    terminal_us = max(int(row["terminal_us"]) for row in requests)
    generated_tokens = sum(int(row["generated_tokens"]) for row in requests)
    pending = sum(row["initial_admission"] == "pending" for row in requests)
    ttft_ms = [float(row["ttft_us"]) / 1000.0 for row in requests]
    e2e_ms = [float(row["e2e_us"]) / 1000.0 for row in requests]
    tpot_ms = [float(row["tpot_us"]) / 1000.0 for row in requests]
    prefill_batches = [
        int(row["prefill_batch"])
        for row in dispatches
        if int(row["prefill_batch"]) > 0
    ]
    decode_batches = [
        int(row["decode_batch"])
        for row in dispatches
        if int(row["decode_batch"]) > 0
    ]
    overlap_dispatches = sum(
        int(row["prefill_batch"]) > 0 and int(row["decode_batch"]) > 0
        for row in dispatches)
    prefill_wait_ms = [
        float(row["prefill_queue_wait_us"]) / 1000.0 for row in dispatches
        if int(row["prefill_batch"]) > 0
    ]
    decode_wait_ms = [
        float(row["decode_queue_wait_us"]) / 1000.0 for row in dispatches
        if int(row["decode_batch"]) > 0
    ]

    return {
        "scenario": workload.name,
        "requests": workload.requests,
        "arrival_rate_rps": workload.arrival_rate,
        "prompt_min": workload.prompt_min,
        "prompt_max": workload.prompt_max,
        "output_min": workload.output_min,
        "output_max": workload.output_max,
        "chunk_size": workload.chunk_size,
        "prefill_batch_limit": workload.prefill_batch,
        "decode_batch_limit": workload.decode_batch,
        "overlap_prefill_tokens": workload.overlap_prefill_tokens,
        "duration_ms": terminal_us / 1000.0,
        "achieved_rps": workload.requests * 1_000_000.0 / terminal_us,
        "generated_token_s": generated_tokens * 1_000_000.0 / terminal_us,
        "pending_requests": pending,
        "ttft_median_ms": statistics.median(ttft_ms),
        "ttft_p95_ms": percentile(ttft_ms, 0.95),
        "e2e_median_ms": statistics.median(e2e_ms),
        "e2e_p95_ms": percentile(e2e_ms, 0.95),
        "tpot_median_ms": statistics.median(tpot_ms),
        "tpot_p95_ms": percentile(tpot_ms, 0.95),
        "dispatches": len(dispatches),
        "overlap_dispatches": overlap_dispatches,
        "overlap_dispatch_fraction": overlap_dispatches / len(dispatches),
        "prefill_batch_mean": mean_or_zero(prefill_batches),
        "decode_batch_mean": mean_or_zero(decode_batches),
        "prefill_wait_p95_ms": percentile(prefill_wait_ms, 0.95),
        "decode_wait_p95_ms": percentile(decode_wait_ms, 0.95),
    }


def command_for(args: argparse.Namespace, workload: Workload,
                request_csv: Path, fixed_csv: Path) -> list[str]:
    return [
        str(args.bench),
        "--engineDir",
        str(args.engine_dir),
        "--prefillBatch",
        str(workload.prefill_batch),
        "--decodeBatch",
        str(workload.decode_batch),
        "--inputLen",
        str(workload.input_len),
        "--prefillChunkSize",
        str(workload.chunk_size),
        "--pastKVLen",
        "128",
        "--warmup",
        str(args.warmup),
        "--iterations",
        str(args.iterations),
        "--contextAdapter",
        "--trtContextMode",
        args.context_mode,
        "--loadRequests",
        str(workload.requests),
        "--arrivalRate",
        str(workload.arrival_rate),
        "--loadPromptMin",
        str(workload.prompt_min),
        "--loadPromptMax",
        str(workload.prompt_max),
        "--loadOutputMin",
        str(workload.output_min),
        "--loadOutputMax",
        str(workload.output_max),
        "--maxOverlapPrefillTokens",
        str(workload.overlap_prefill_tokens),
        "--loadSeed",
        str(args.seed),
        "--loadCsv",
        str(request_csv),
        "--outputCsv",
        str(fixed_csv),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", type=Path, required=True)
    parser.add_argument("--engine-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--context-mode",
                        choices=("shared", "independent"),
                        default="independent")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--scenario",
                        action="append",
                        choices=tuple(item.name for item in WORKLOADS))
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    selected = [
        workload for workload in WORKLOADS
        if args.scenario is None or workload.name in args.scenario
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for index, workload in enumerate(selected, start=1):
        scenario_dir = args.output_dir / workload.name
        scenario_dir.mkdir(parents=True, exist_ok=True)
        request_csv = scenario_dir / "requests.csv"
        dispatch_csv = scenario_dir / "requests-dispatch.csv"
        fixed_csv = scenario_dir / "fixed.csv"
        log_path = scenario_dir / "run.log"
        if not args.skip_existing or not (request_csv.exists()
                                          and dispatch_csv.exists()):
            command = command_for(args, workload, request_csv, fixed_csv)
            print(f"[{index}/{len(selected)}] running {workload.name}",
                  flush=True)
            with log_path.open("w", encoding="utf-8") as log:
                result = subprocess.run(command,
                                        stdout=log,
                                        stderr=subprocess.STDOUT,
                                        check=False,
                                        text=True)
            if result.returncode != 0:
                tail = "\n".join(
                    log_path.read_text(encoding="utf-8",
                                       errors="replace").splitlines()[-30:])
                raise RuntimeError(
                    f"{workload.name} failed with exit code {result.returncode}:\n{tail}"
                )
        summary = summarize(workload, request_csv, dispatch_csv)
        summaries.append(summary)
        print(
            f"[{index}/{len(selected)}] {workload.name}: "
            f"{summary['achieved_rps']:.2f} req/s, "
            f"TTFT p95={summary['ttft_p95_ms']:.2f} ms, "
            f"pending={summary['pending_requests']}, "
            f"overlap={summary['overlap_dispatches']}",
            flush=True)

    summary_path = args.output_dir / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    print(f"summary written to {summary_path}", flush=True)


if __name__ == "__main__":
    main()
