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
"""Run the real-request KV/context matrix for indexed-linear and indexed-paged.

The runner deliberately invokes the existing ``llm_phase_bench`` executable so
request admission, batching, CUDA-event timing, and TensorRT context ownership
are measured by the production path.  A single materialized trace is reused for
every engine/context/batch case.  Paged-pool pressure is reported by a
conservative host model after each run: it reserves enough 128-token bundles for
the complete requested output of each admitted request.  This is an admission
policy model, not a claim that the current runtime already blocks on pages.
"""

import argparse
import csv
import json
import math
import random
import statistics
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Engine:
    name: str
    directory: Path


@dataclass(frozen=True)
class Case:
    prefill_batch: int
    decode_batch: int
    context_mode: str

    @property
    def name(self) -> str:
        return f"p{self.prefill_batch}_d{self.decode_batch}_{self.context_mode}"


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


def parse_engine(value: str) -> Engine:
    name, separator, directory = value.partition("=")
    if not separator or not name or not directory:
        raise argparse.ArgumentTypeError("engine must be NAME=ENGINE_DIR")
    return Engine(name, Path(directory))


def materialize_trace(source: Path, destination: Path, seed: int,
                      arrival_rate: float, request_count: int,
                      repeat_count: int) -> None:
    root = json.loads(source.read_text(encoding="utf-8"))
    requests = list(root.get("requests", []))
    if not requests:
        raise RuntimeError(f"trace contains no requests: {source}")
    if request_count > 0:
        requests = requests[:request_count]
    if repeat_count <= 0:
        raise ValueError("repeat_count must be positive")
    requests = [
        json.loads(json.dumps(request))
        for _ in range(repeat_count)
        for request in requests
    ]
    generator = random.Random(seed)
    elapsed_us = 0.0
    for index, request in enumerate(requests):
        if index > 0:
            elapsed_us += generator.expovariate(arrival_rate) * 1_000_000.0
        request["arrival_offset_us"] = round(elapsed_us)
    root["requests"] = requests
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(root, indent=2) + "\n", encoding="utf-8")


def scenarios(prefill_batches: list[int], decode_batches: list[int],
              context_modes: list[str]) -> list[Case]:
    return [
        Case(prefill, decode, mode)
        for mode in context_modes
        for prefill in prefill_batches
        for decode in decode_batches
    ]


def command_for(args: argparse.Namespace, engine: Engine, case: Case,
                trace: Path, case_dir: Path) -> list[str]:
    return [
        str(args.bench),
        "--engineDir",
        str(engine.directory),
        "--prefillBatch",
        str(case.prefill_batch),
        "--decodeBatch",
        str(case.decode_batch),
        "--slotCount",
        str(args.slot_count),
        "--inputLen",
        str(args.input_len),
        "--prefillChunkSize",
        str(args.chunk_size),
        "--pastKVLen",
        str(args.past_kv_len),
        "--warmup",
        "0",
        "--iterations",
        "1",
        "--contextAdapter",
        "--trtContextMode",
        case.context_mode,
        "--inputFile",
        str(trace),
        "--traceCsv",
        str(case_dir / "requests.csv"),
        "--traceArrivalRate",
        str(args.arrival_rate),
        "--kernelGroupCsv",
        str(case_dir / "kernel-groups.csv"),
    ]


def summarize_requests(path: Path, engine: Engine, case: Case,
                       return_code: int) -> dict[str, object]:
    rows = read_csv(path)
    ttft = [float(row["ttft_us"]) / 1000.0 for row in rows]
    tpot = [float(row["tpot_us"]) / 1000.0 for row in rows]
    e2e = [float(row["e2e_ms"]) for row in rows]
    terminal_us = max(int(row["completed_us"]) for row in rows)
    output_tokens = sum(int(row["output_tokens"]) for row in rows)
    pending = sum(row["admission_status"] == "0" for row in rows)
    return {
        "engine": engine.name,
        "context_mode": case.context_mode,
        "case": case.name,
        "prefill_batch": case.prefill_batch,
        "decode_batch": case.decode_batch,
        "requests": len(rows),
        "return_code": return_code,
        "duration_ms": terminal_us / 1000.0,
        "achieved_req_s": len(rows) * 1_000_000.0 / terminal_us,
        "generated_token_s": output_tokens * 1_000_000.0 / terminal_us,
        "ttft_median_ms": statistics.median(ttft),
        "ttft_p95_ms": percentile(ttft, 0.95),
        "tpot_median_ms": statistics.median(tpot),
        "tpot_p95_ms": percentile(tpot, 0.95),
        "e2e_median_ms": statistics.median(e2e),
        "e2e_p95_ms": percentile(e2e, 0.95),
        "initial_pending_requests": pending,
    }


def summarize_dispatch(path: Path, engine: Engine, case: Case) -> list[dict[str, object]]:
    rows = read_csv(path)
    grouped: dict[tuple[int, int], list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault((int(row["prefill_batch"]), int(row["decode_batch"])), []).append(row)
    result = []
    for (prefill, decode), samples in sorted(grouped.items()):
        result.append({
            "engine": engine.name,
            "context_mode": case.context_mode,
            "case": case.name,
            "observed_prefill_batch": prefill,
            "observed_decode_batch": decode,
            "dispatches": len(samples),
            "makespan_median_ms": statistics.median(float(row["makespan_gpu_ms"]) for row in samples),
            "makespan_p95_ms": percentile([float(row["makespan_gpu_ms"]) for row in samples], 0.95),
            "prefill_median_ms": statistics.median(float(row["prefill_gpu_ms"]) for row in samples),
            "prefill_p95_ms": percentile([float(row["prefill_gpu_ms"]) for row in samples], 0.95),
            "decode_median_ms": statistics.median(float(row["decode_gpu_ms"]) for row in samples),
            "decode_p95_ms": percentile([float(row["decode_gpu_ms"]) for row in samples], 0.95),
            "overlap_median": statistics.median(float(row["overlap_ratio"]) for row in samples),
        })
    return result


def summarize_kernel(path: Path, engine: Engine, case: Case) -> list[dict[str, object]]:
    rows = read_csv(path)
    grouped: dict[str, list[float]] = {}
    for row in rows:
        grouped.setdefault(row["group"], []).append(float(row["gpu_ms"]))
    return [{
        "engine": engine.name,
        "context_mode": case.context_mode,
        "case": case.name,
        "group": group,
        "samples": len(values),
        "gpu_median_ms": statistics.median(values),
        "gpu_p95_ms": percentile(values, 0.95),
        "gpu_mean_ms": statistics.mean(values),
        "gpu_max_ms": max(values),
    } for group, values in sorted(grouped.items())]


def pressure_model(path: Path, engine: Engine, case: Case, page_bundles: int,
                   tokens_per_page: int, slot_count: int) -> dict[str, object]:
    """Estimate strict whole-request page admission pressure from completed rows."""
    rows = read_csv(path)
    requests = [{
        "id": int(row["request_id"]),
        "arrival": int(row["scheduled_arrival_us"]),
        "done": int(row["completed_us"]),
        "pages": math.ceil((int(row["prompt_tokens"]) + int(row["max_output_tokens"])) / tokens_per_page),
    } for row in rows]
    pending = list(sorted(requests, key=lambda request: (request["arrival"], request["id"])))
    active: dict[int, dict[str, int]] = {}
    now = 0
    blocked = 0
    max_pending = 0
    peak_allocated = 0
    peak_pressure = 0.0
    observed_peak_pressure = max(
        (float(request.get("admission_page_pool_pressure", 0.0)) for request in rows),
        default=0.0,
    )
    observed_peak_pending = max(
        (int(request.get("admission_pending_queue_depth", 0)) for request in rows),
        default=0,
    )
    admission_events = 0

    def release(until: int) -> None:
        for request_id in [
                request_id for request_id, request in active.items()
                if request["done"] <= until
        ]:
            del active[request_id]

    def allocated() -> int:
        return sum(request["pages"] for request in active.values())

    for request in pending:
        now = max(now, request["arrival"])
        release(now)
        waited = False
        while active and (len(active) >= slot_count or allocated() + request["pages"] > page_bundles):
            next_done = min(item["done"] for item in active.values())
            now = max(now, next_done)
            release(now)
            blocked += 1
            waited = True
        if waited:
            max_pending = max(max_pending, 1)
        active[request["id"]] = request
        admission_events += 1
        current = allocated()
        peak_allocated = max(peak_allocated, current)
        peak_pressure = max(peak_pressure, current / page_bundles)

    return {
        "engine": engine.name,
        "context_mode": case.context_mode,
        "case": case.name,
        "page_bundles": page_bundles,
        "tokens_per_page": tokens_per_page,
        "slot_count": slot_count,
        "modelled_admissions": admission_events,
        "modelled_page_block_events": blocked,
        "modelled_max_pending": max_pending,
        "modelled_peak_allocated_bundles": peak_allocated,
        "modelled_peak_pressure": peak_pressure,
        "observed_peak_admission_pressure": observed_peak_pressure,
        "observed_peak_pending_queue": observed_peak_pending,
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", type=Path, required=True)
    parser.add_argument("--source-trace", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--engine", action="append", type=parse_engine, required=True,
                        help="NAME=ENGINE_DIR; repeat for indexed-linear and indexed-paged")
    parser.add_argument("--prefill-batches", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--decode-batches", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--context-modes", nargs="+", choices=("shared", "independent"),
                        default=["shared", "independent"])
    parser.add_argument("--arrival-rate", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--request-count", type=int, default=0)
    parser.add_argument("--repeat-count", type=int, default=1,
                        help="repeat the source request set to create a longer arrival trace")
    parser.add_argument("--slot-count", type=int, default=16)
    parser.add_argument("--page-bundles", type=int, default=80)
    parser.add_argument("--tokens-per-page", type=int, default=128)
    parser.add_argument("--input-len", type=int, default=1024)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--past-kv-len", type=int, default=128)
    parser.add_argument("--case", action="append", help="run only CASE names, e.g. p4_d16_independent")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    if args.slot_count <= 0 or args.page_bundles <= 0 or args.tokens_per_page <= 0:
        parser.error("slot-count, page-bundles, and tokens-per-page must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trace = args.output_dir / "materialized-trace.json"
    materialize_trace(args.source_trace, trace, args.seed, args.arrival_rate, args.request_count,
                      args.repeat_count)
    cases = scenarios(args.prefill_batches, args.decode_batches, args.context_modes)
    if args.case:
        cases = [case for case in cases if case.name in args.case]
    summaries = []
    dispatches = []
    kernels = []
    pressure = []
    statuses = []
    total = len(args.engine) * len(cases)
    index = 0
    for engine in args.engine:
        for case in cases:
            index += 1
            case_dir = args.output_dir / engine.name / case.name
            case_dir.mkdir(parents=True, exist_ok=True)
            log_path = case_dir / "run.log"
            command = command_for(args, engine, case, trace, case_dir)
            started = time.monotonic()
            with log_path.open("w", encoding="utf-8") as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False, text=True)
            elapsed = time.monotonic() - started
            status = {
                "engine": engine.name,
                "context_mode": case.context_mode,
                "case": case.name,
                "return_code": result.returncode,
                "elapsed_s": elapsed,
                "log": str(log_path),
            }
            statuses.append(status)
            print(f"[{index}/{total}] {engine.name}/{case.name}: rc={result.returncode}", flush=True)
            request_csv = case_dir / "requests.csv"
            dispatch_csv = case_dir / "requests-dispatch.csv"
            kernel_csv = case_dir / "kernel-groups.csv"
            if result.returncode == 0 and request_csv.exists() and dispatch_csv.exists() and kernel_csv.exists():
                summaries.append(summarize_requests(request_csv, engine, case, result.returncode))
                dispatches.extend(summarize_dispatch(dispatch_csv, engine, case))
                kernels.extend(summarize_kernel(kernel_csv, engine, case))
                pressure.append(pressure_model(request_csv, engine, case, args.page_bundles,
                                               args.tokens_per_page, args.slot_count))
            elif not args.continue_on_error:
                tail = "\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-40:])
                raise RuntimeError(f"{engine.name}/{case.name} failed with {result.returncode}:\n{tail}")
    write_csv(args.output_dir / "status.csv", statuses)
    write_csv(args.output_dir / "request-summary.csv", summaries)
    write_csv(args.output_dir / "dispatch-cost-table.csv", dispatches)
    write_csv(args.output_dir / "kernel-cost-table.csv", kernels)
    write_csv(args.output_dir / "page-pressure-model.csv", pressure)
    print(f"wrote matrix artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
