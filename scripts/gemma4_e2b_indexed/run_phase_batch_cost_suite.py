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
"""Build fixed-128 prefill/decode batch cost tables from CUDA events."""

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
    requests: int
    prompt_tokens: int
    output_tokens: int


PREFILL_BATCHES = (1, 2, 4, 8, 16)
DECODE_BATCHES = (1, 2, 4, 8, 16)
OVERLAP_SPLITS = ((1, 15), (2, 14), (4, 12), (6, 10), (8, 8),
                  (10, 6), (12, 4), (14, 2), (15, 1))

SCENARIOS = tuple(
    [Scenario(f"prefill_bs{batch}", "prefill", batch, 1, 48, 128, 1)
     for batch in PREFILL_BATCHES] +
    [Scenario(f"decode_bs{batch}", "decode", 1, batch, 24, 128, 24)
     for batch in DECODE_BATCHES] +
    [Scenario(f"overlap_p{prefill}_d{decode}", "overlap", prefill, decode,
              48, 512, 16)
     for prefill, decode in OVERLAP_SPLITS])


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


def command_for(args: argparse.Namespace, scenario: Scenario,
                scenario_dir: Path) -> list[str]:
    overlap_prefill_tokens = (scenario.prefill_batch * 128
                              if scenario.profile == "overlap" else 128)
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
        "128",
        "--prefillChunkSize",
        "128",
        "--pastKVLen",
        "128",
        "--warmup",
        str(args.warmup),
        "--iterations",
        str(args.iterations),
        "--contextAdapter",
        "--trtContextMode",
        "independent",
        "--loadRequests",
        str(scenario.requests),
        "--arrivalRate",
        "1000",
        "--loadPromptMin",
        str(scenario.prompt_tokens),
        "--loadPromptMax",
        str(scenario.prompt_tokens),
        "--loadOutputMin",
        str(scenario.output_tokens),
        "--loadOutputMax",
        str(scenario.output_tokens),
        "--maxOverlapPrefillTokens",
        str(overlap_prefill_tokens),
        "--loadSeed",
        str(args.seed),
        "--loadCsv",
        str(scenario_dir / "requests.csv"),
        "--outputCsv",
        str(scenario_dir / "fixed.csv"),
        "--kernelGroupCsv",
        str(scenario_dir / "kernel-groups.csv"),
    ]


def summarize_requests(scenario: Scenario, path: Path,
                       slot_count: int) -> dict[str, object]:
    rows = read_csv(path)
    ttft = [float(row["ttft_us"]) / 1000.0 for row in rows]
    tpot = [float(row["tpot_us"]) / 1000.0 for row in rows]
    e2e = [float(row["e2e_us"]) / 1000.0 for row in rows]
    terminal_us = max(int(row["terminal_us"]) for row in rows)
    return {
        "scenario": scenario.name,
        "profile": scenario.profile,
        "slot_count": slot_count,
        "prefill_batch_limit": scenario.prefill_batch,
        "decode_batch_limit": scenario.decode_batch,
        "requests": len(rows),
        "prompt_tokens": scenario.prompt_tokens,
        "output_tokens": scenario.output_tokens,
        "achieved_rps": len(rows) * 1_000_000.0 / terminal_us,
        "ttft_mean_ms": statistics.mean(ttft),
        "ttft_median_ms": statistics.median(ttft),
        "ttft_p95_ms": percentile(ttft, 0.95),
        "tpot_mean_ms": statistics.mean(tpot),
        "tpot_median_ms": statistics.median(tpot),
        "tpot_p95_ms": percentile(tpot, 0.95),
        "e2e_mean_ms": statistics.mean(e2e),
        "e2e_median_ms": statistics.median(e2e),
        "e2e_p95_ms": percentile(e2e, 0.95),
    }


def kernel_cost_rows(scenario: Scenario, path: Path,
                     slot_count: int) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[float]] = {}
    for row in read_csv(path):
        group = row["group"]
        prefill_batch = int(row["prefill_batch"])
        decode_batch = int(row["decode_batch"])
        prefill_tokens = int(row["prefill_tokens"])
        decode_context_tokens = int(row["decode_context_tokens"])
        chunk_tokens = (prefill_tokens // prefill_batch
                        if prefill_batch else 0)
        mean_decode_context = (decode_context_tokens // decode_batch
                               if decode_batch else 0)
        key = (group, prefill_batch, decode_batch, chunk_tokens,
               mean_decode_context)
        grouped.setdefault(key, []).append(float(row["gpu_ms"]))

    result = []
    for key, values in sorted(grouped.items()):
        group, prefill_batch, decode_batch, chunk_tokens, decode_context = key
        result.append({
            "scenario": scenario.name,
            "profile": scenario.profile,
            "slot_count": slot_count,
            "group": group,
            "prefill_batch": prefill_batch,
            "decode_batch": decode_batch,
            "chunk_tokens_per_request": chunk_tokens,
            "mean_decode_context_tokens": decode_context,
            "samples": len(values),
            "gpu_mean_ms": statistics.mean(values),
            "gpu_median_ms": statistics.median(values),
            "gpu_p95_ms": percentile(values, 0.95),
            "gpu_max_ms": max(values),
        })
    return result


def kernel_group_summary_rows(scenario: Scenario, path: Path,
                              slot_count: int) -> list[dict[str, object]]:
    grouped: dict[tuple[str, int, int], list[float]] = {}
    for row in read_csv(path):
        key = (row["group"], int(row["prefill_batch"]),
               int(row["decode_batch"]))
        grouped.setdefault(key, []).append(float(row["gpu_ms"]))
    result = []
    for (group, prefill_batch, decode_batch), values in sorted(grouped.items()):
        result.append({
            "scenario": scenario.name,
            "profile": scenario.profile,
            "slot_count": slot_count,
            "group": group,
            "prefill_batch": prefill_batch,
            "decode_batch": decode_batch,
            "samples": len(values),
            "gpu_mean_ms": statistics.mean(values),
            "gpu_median_ms": statistics.median(values),
            "gpu_p95_ms": percentile(values, 0.95),
            "gpu_max_ms": max(values),
        })
    return result


def dispatch_summary_rows(scenario: Scenario, path: Path,
                          slot_count: int) -> list[dict[str, object]]:
    grouped: dict[tuple[int, int, int], list[dict[str, str]]] = {}
    for row in read_csv(path):
        key = (int(row["kind"]), int(row["prefill_batch"]),
               int(row["decode_batch"]))
        grouped.setdefault(key, []).append(row)
    result = []
    for (kind, prefill_batch, decode_batch), rows in sorted(grouped.items()):
        result_row: dict[str, object] = {
            "scenario": scenario.name,
            "profile": scenario.profile,
            "slot_count": slot_count,
            "kind": kind,
            "prefill_batch": prefill_batch,
            "decode_batch": decode_batch,
            "samples": len(rows),
        }
        for column in ("prefill_queue_wait_us", "decode_queue_wait_us",
                       "prefill_gpu_ms", "decode_gpu_ms",
                       "makespan_gpu_ms", "overlap_ratio"):
            values = [float(row[column]) for row in rows]
            result_row[f"{column}_mean"] = statistics.mean(values)
            result_row[f"{column}_p95"] = percentile(values, 0.95)
        result.append(result_row)
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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--slot-count", type=int, default=16)
    parser.add_argument("--scenario", action="append",
                        choices=tuple(item.name for item in SCENARIOS))
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    selected = [scenario for scenario in SCENARIOS
                if args.scenario is None or scenario.name in args.scenario]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    request_summaries = []
    cost_rows = []
    group_summary_rows = []
    dispatch_rows = []
    for index, scenario in enumerate(selected, start=1):
        scenario_dir = args.output_dir / scenario.name
        scenario_dir.mkdir(parents=True, exist_ok=True)
        request_csv = scenario_dir / "requests.csv"
        kernel_csv = scenario_dir / "kernel-groups.csv"
        log_path = scenario_dir / "run.log"
        if not args.skip_existing or not (request_csv.exists() and
                                          kernel_csv.exists()):
            print(f"[{index}/{len(selected)}] running {scenario.name}",
                  flush=True)
            with log_path.open("w", encoding="utf-8") as log:
                result = subprocess.run(command_for(args, scenario, scenario_dir),
                                        stdout=log,
                                        stderr=subprocess.STDOUT,
                                        check=False,
                                        text=True)
            if result.returncode != 0:
                tail = "\n".join(log_path.read_text(
                    encoding="utf-8", errors="replace").splitlines()[-40:])
                raise RuntimeError(
                    f"{scenario.name} failed with {result.returncode}:\n{tail}")
        summary = summarize_requests(scenario, request_csv, args.slot_count)
        request_summaries.append(summary)
        cost_rows.extend(kernel_cost_rows(scenario, kernel_csv,
                                          args.slot_count))
        group_summary_rows.extend(kernel_group_summary_rows(
            scenario, kernel_csv, args.slot_count))
        dispatch_rows.extend(dispatch_summary_rows(
            scenario, scenario_dir / "requests-dispatch.csv",
            args.slot_count))
        print(f"[{index}/{len(selected)}] {scenario.name}: "
              f"{summary['achieved_rps']:.2f} req/s, "
              f"TTFT p95={summary['ttft_p95_ms']:.2f} ms, "
              f"TPOT p95={summary['tpot_p95_ms']:.2f} ms", flush=True)

    write_rows(args.output_dir / "request-summary.csv", request_summaries)
    write_rows(args.output_dir / "kernel-cost-table.csv", cost_rows)
    write_rows(args.output_dir / "kernel-group-summary.csv",
               group_summary_rows)
    write_rows(args.output_dir / "dispatch-summary.csv", dispatch_rows)


if __name__ == "__main__":
    main()
