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
every engine/context/batch case. Paged-pool pressure includes both the runtime's
selected admission policy and a conservative host model that reserves the
complete requested output.
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
                      repeat_count: int, total_requests: int,
                      output_multiplier: float) -> None:
    root = json.loads(source.read_text(encoding="utf-8"))
    requests = list(root.get("requests", []))
    if not requests:
        raise RuntimeError(f"trace contains no requests: {source}")
    if request_count > 0:
        requests = requests[:request_count]
    if repeat_count <= 0:
        raise ValueError("repeat_count must be positive")
    requests = [
        json.loads(json.dumps(request)) for _ in range(repeat_count)
        for request in requests
    ]
    if total_requests > 0:
        requests = requests[:total_requests]
    for request in requests:
        original_length = int(
            request.get("max_generate_length",
                        root.get("max_generate_length", 1)))
        request["max_generate_length"] = max(
            1, math.ceil(original_length * output_multiplier))
    generator = random.Random(seed)
    elapsed_us = 0.0
    for index, request in enumerate(requests):
        if index > 0:
            elapsed_us += generator.expovariate(arrival_rate) * 1_000_000.0
        request["arrival_offset_us"] = round(elapsed_us)
    root["requests"] = requests
    root["max_generate_length"] = max(
        int(request["max_generate_length"]) for request in requests)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(root, indent=2) + "\n", encoding="utf-8")


def scenarios(prefill_batches: list[int], decode_batches: list[int],
              context_modes: list[str]) -> list[Case]:
    return [
        Case(prefill, decode, mode) for mode in context_modes
        for prefill in prefill_batches for decode in decode_batches
    ]


def command_for(args: argparse.Namespace, engine: Engine, case: Case,
                trace: Path, case_dir: Path) -> list[str]:
    command = [
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
        "--maxOverlapPrefillTokens",
        str(args.max_overlap_prefill_tokens),
        "--ttftTargetMs",
        str(args.ttft_target_ms),
        "--tpotTargetMs",
        str(args.tpot_target_ms),
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
        "--pageReservationMode",
        args.page_reservation_mode,
        "--pageReservationHeadroomTokens",
        str(args.page_reservation_headroom_tokens),
        "--pageReservationOvercommitBundles",
        str(args.page_reservation_overcommit_bundles),
        "--pageReservationGrowthRequests",
        str(args.page_reservation_growth_requests),
        "--fullReservationPromptThresholdTokens",
        str(args.full_reservation_prompt_threshold_tokens),
        "--minPageGrowthRequests",
        str(args.min_page_growth_requests),
        "--pageGrowthTpotTargetMs",
        str(args.growth_tpot_target_ms),
    ]
    if args.adaptive_page_growth:
        command.append("--adaptivePageGrowth")
    if args.cuda_graph:
        if case.context_mode != "independent":
            raise ValueError(
                "CUDA graph phase execution requires independent contexts")
        command.extend(
            ["--cudaGraph", "--maxCudaGraphs",
             str(args.max_cuda_graphs)])
        if args.max_prefill_cuda_graphs > 0:
            command.extend(
                ["--maxPrefillCudaGraphs",
                 str(args.max_prefill_cuda_graphs)])
        if args.max_decode_cuda_graphs > 0:
            command.extend(
                ["--maxDecodeCudaGraphs",
                 str(args.max_decode_cuda_graphs)])
        if args.max_cuda_graph_mib > 0:
            command.extend(["--maxCudaGraphMiB", str(args.max_cuda_graph_mib)])
        if args.max_prefill_cuda_graph_mib >= 0:
            command.extend([
                "--maxPrefillCudaGraphMiB",
                str(args.max_prefill_cuda_graph_mib)
            ])
        if args.max_decode_cuda_graph_mib >= 0:
            command.extend([
                "--maxDecodeCudaGraphMiB",
                str(args.max_decode_cuda_graph_mib)
            ])
        command.extend(
            ["--cudaGraphChargeMiB",
             str(args.cuda_graph_charge_mib)])
        if args.cuda_graph_reserve_mib > 0:
            command.extend(
                ["--cudaGraphReserveMiB",
                 str(args.cuda_graph_reserve_mib)])
    if args.prefill_token_budget > 0:
        command.extend(
            ["--prefillTokenBudget",
             str(args.prefill_token_budget)])
    if args.dynamic_decode_batching:
        command.append("--dynamicDecodeBatching")
    if args.dynamic_prefill_batching:
        command.extend([
            "--dynamicPrefillBatching", "--minDynamicPrefillBatchSize",
            str(args.min_dynamic_prefill_batch_size)
        ])
    if args.prefill_slo_recovery:
        command.append("--prefillSloRecovery")
    if args.scheduler_profile != "custom":
        command.extend(["--schedulerProfile", args.scheduler_profile])
    if (args.dynamic_decode_batching or args.dynamic_prefill_batching
            or args.scheduler_profile != "custom" or args.tpot_hard_guard
            or args.require_direct_overlap_cost):
        command.extend(["--schedulerCostJson", str(args.scheduler_cost_json)])
    if args.tpot_hard_guard:
        command.extend([
            "--tpotHardGuard", "--maxConsecutiveOverlapBatches",
            str(args.max_consecutive_overlap_batches),
            "--maxPredictedDecodeDebtMs",
            str(args.max_predicted_decode_debt_ms)
        ])
    if args.require_direct_overlap_cost:
        command.append("--requireDirectOverlapCost")
    if args.wavefront_prefill_batching:
        command.extend([
            "--wavefrontPrefillBatching", "--prefillCohortSize",
            str(args.prefill_cohort_size), "--prefillCohortTurns",
            str(args.prefill_cohort_turns), "--decodeSlackSafetyFactor",
            str(args.decode_slack_safety_factor)
        ])
    if args.adaptive_scheduler:
        command.append("--adaptiveScheduler")
    return command


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


def summarize_dispatch(path: Path, engine: Engine,
                       case: Case) -> list[dict[str, object]]:
    rows = read_csv(path)
    grouped: dict[tuple[int, int], list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(
            (int(row["prefill_batch"]), int(row["decode_batch"])),
            []).append(row)
    result = []
    for (prefill, decode), samples in sorted(grouped.items()):
        summary = {
            "engine":
            engine.name,
            "context_mode":
            case.context_mode,
            "case":
            case.name,
            "observed_prefill_batch":
            prefill,
            "observed_decode_batch":
            decode,
            "dispatches":
            len(samples),
            "makespan_median_ms":
            statistics.median(
                float(row["makespan_gpu_ms"]) for row in samples),
            "makespan_p95_ms":
            percentile([float(row["makespan_gpu_ms"]) for row in samples],
                       0.95),
            "prefill_median_ms":
            statistics.median(float(row["prefill_gpu_ms"]) for row in samples),
            "prefill_p95_ms":
            percentile([float(row["prefill_gpu_ms"]) for row in samples],
                       0.95),
            "decode_median_ms":
            statistics.median(float(row["decode_gpu_ms"]) for row in samples),
            "decode_p95_ms":
            percentile([float(row["decode_gpu_ms"]) for row in samples], 0.95),
            "overlap_median":
            statistics.median(float(row["overlap_ratio"]) for row in samples),
        }
        if "page_growth_request_limit" in samples[0]:
            limits = [int(row["page_growth_request_limit"]) for row in samples]
            owners = [
                int(row["page_growth_request_owners"]) for row in samples
            ]
            pressures = [
                float(row["page_growth_tpot_pressure"]) for row in samples
            ]
            summary.update({
                "page_growth_limit_median":
                statistics.median(limits),
                "page_growth_limit_max":
                max(limits),
                "page_growth_owners_median":
                statistics.median(owners),
                "page_growth_owners_max":
                max(owners),
                "page_growth_pressure_median":
                statistics.median(pressures),
                "page_growth_pressure_p95":
                percentile(pressures, 0.95),
            })
        result.append(summary)
    return result


def summarize_kernel(path: Path, engine: Engine,
                     case: Case) -> list[dict[str, object]]:
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
        "id":
        int(row["request_id"]),
        "arrival":
        int(row["scheduled_arrival_us"]),
        "done":
        int(row["completed_us"]),
        "pages":
        math.ceil((int(row["prompt_tokens"]) + int(row["max_output_tokens"])) /
                  tokens_per_page),
    } for row in rows]
    pending = list(
        sorted(requests,
               key=lambda request: (request["arrival"], request["id"])))
    active: dict[int, dict[str, int]] = {}
    now = 0
    blocked = 0
    max_pending = 0
    peak_allocated = 0
    peak_pressure = 0.0
    observed_peak_pressure = max(
        (float(
            request.get("admission_reserved_page_pressure",
                        request.get("admission_page_pool_pressure", 0.0)))
         for request in rows),
        default=0.0,
    )
    observed_peak_pending = max(
        (int(request.get("admission_pending_queue_depth", 0))
         for request in rows),
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
        while active and (len(active) >= slot_count
                          or allocated() + request["pages"] > page_bundles):
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
    parser.add_argument(
        "--engine",
        action="append",
        type=parse_engine,
        required=True,
        help="NAME=ENGINE_DIR; repeat for indexed-linear and indexed-paged")
    parser.add_argument("--prefill-batches",
                        type=int,
                        nargs="+",
                        default=[1, 2, 4, 8])
    parser.add_argument("--decode-batches",
                        type=int,
                        nargs="+",
                        default=[1, 2, 4, 8, 16])
    parser.add_argument("--context-modes",
                        nargs="+",
                        choices=("shared", "independent"),
                        default=["shared", "independent"])
    parser.add_argument("--arrival-rate", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--request-count", type=int, default=0)
    parser.add_argument(
        "--repeat-count",
        type=int,
        default=1,
        help="repeat the source request set to create a longer arrival trace")
    parser.add_argument(
        "--total-requests",
        type=int,
        default=0,
        help=
        "trim the repeated trace to this exact request count; zero disables it"
    )
    parser.add_argument(
        "--output-multiplier",
        type=float,
        default=1.0,
        help=
        "multiply each request max_generate_length in the materialized trace")
    parser.add_argument("--slot-count", type=int, default=16)
    parser.add_argument("--page-bundles", type=int, default=80)
    parser.add_argument("--tokens-per-page", type=int, default=128)
    parser.add_argument("--input-len", type=int, default=1024)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--max-overlap-prefill-tokens", type=int, default=128)
    parser.add_argument("--ttft-target-ms", type=float, default=500.0)
    parser.add_argument("--tpot-target-ms", type=float, default=50.0)
    # Real-trace warmup must start from an empty cache. Paged KV v1 rejects
    # the legacy system-prompt cache reuse path used by non-zero pastKVLen.
    parser.add_argument("--past-kv-len", type=int, default=0)
    parser.add_argument("--case",
                        action="append",
                        help="run only CASE names, e.g. p4_d16_independent")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--cuda-graph",
                        action="store_true",
                        help="enable per-context CUDA graph capture/replay")
    parser.add_argument(
        "--max-cuda-graphs",
        type=int,
        default=128,
        help="maximum graph variants cached by each phase context")
    parser.add_argument(
        "--max-prefill-cuda-graphs",
        type=int,
        default=0,
        help="prefill graph limit override; zero inherits --max-cuda-graphs")
    parser.add_argument(
        "--max-decode-cuda-graphs",
        type=int,
        default=0,
        help="decode graph limit override; zero inherits --max-cuda-graphs")
    parser.add_argument(
        "--max-cuda-graph-mib",
        type=int,
        default=0,
        help="per-context graph memory budget; zero is unlimited")
    parser.add_argument(
        "--max-prefill-cuda-graph-mib",
        type=int,
        default=-1,
        help="prefill graph MiB override; -1 inherits the common budget")
    parser.add_argument(
        "--max-decode-cuda-graph-mib",
        type=int,
        default=-1,
        help="decode graph MiB override; -1 inherits the common budget")
    parser.add_argument(
        "--cuda-graph-charge-mib",
        type=int,
        default=4,
        help="minimum conservative device-memory charge per graph")
    parser.add_argument(
        "--cuda-graph-reserve-mib",
        type=int,
        default=0,
        help="global free-memory reserve retained while capturing graphs")
    parser.add_argument(
        "--prefill-token-budget",
        type=int,
        default=0,
        help=
        "total token budget for one compatible prefill batch; zero disables it"
    )
    parser.add_argument(
        "--dynamic-decode-batching",
        action="store_true",
        help="select decode batch size using the measured scheduler cost model"
    )
    parser.add_argument(
        "--dynamic-prefill-batching",
        action="store_true",
        help="select prefill P using profiled cost and remaining decode slack")
    parser.add_argument("--min-dynamic-prefill-batch-size",
                        type=int,
                        default=1)
    parser.add_argument(
        "--prefill-slo-recovery",
        action="store_true",
        help="allow expired TTFT to temporarily override decode interference")
    parser.add_argument(
        "--wavefront-prefill-batching",
        action="store_true",
        help="advance a bounded request cohort across fixed prefill chunks")
    parser.add_argument("--prefill-cohort-size", type=int, default=8)
    parser.add_argument("--prefill-cohort-turns", type=int, default=8)
    parser.add_argument("--decode-slack-safety-factor",
                        type=float,
                        default=0.8)
    parser.add_argument(
        "--scheduler-cost-json",
        type=Path,
        help="cost model produced by build_phase_scheduler_cost_model.py")
    parser.add_argument("--scheduler-profile",
                        choices=("custom", "latency-safe", "balanced",
                                 "long-prefill", "auto"),
                        default="custom")
    parser.add_argument("--tpot-hard-guard", action="store_true")
    parser.add_argument("--require-direct-overlap-cost", action="store_true")
    parser.add_argument("--max-consecutive-overlap-batches",
                        type=int,
                        default=4)
    parser.add_argument("--max-predicted-decode-debt-ms",
                        type=float,
                        default=50.0)
    parser.add_argument(
        "--adaptive-scheduler",
        action="store_true",
        help="enable the CUDA-event/SLO/page-pressure phase selector")
    parser.add_argument("--page-reservation-mode",
                        choices=("full", "headroom", "bounded-overcommit"),
                        default="full",
                        help="runtime KV page admission reservation policy")
    parser.add_argument(
        "--page-reservation-headroom-tokens",
        type=int,
        default=128,
        help="output tokens guaranteed per request in headroom mode")
    parser.add_argument(
        "--page-reservation-overcommit-bundles",
        type=int,
        default=1,
        help="maximum page bundles discounted per request in bounded mode")
    parser.add_argument(
        "--page-reservation-growth-requests",
        type=int,
        default=8,
        help="requests allowed to grow beyond their base reservation together")
    parser.add_argument(
        "--full-reservation-prompt-threshold-tokens",
        type=int,
        default=0,
        help=
        "use full reservation at or above this prompt length; zero disables it"
    )
    parser.add_argument(
        "--adaptive-page-growth",
        action="store_true",
        help="adapt runnable growth leases using CUDA-event TPOT pressure")
    parser.add_argument("--min-page-growth-requests",
                        type=int,
                        default=1,
                        help="initial and minimum adaptive growth lease count")
    parser.add_argument(
        "--growth-tpot-target-ms",
        type=float,
        default=20.0,
        help=
        "adaptive growth controller target for queue wait plus decode GPU time"
    )
    args = parser.parse_args()

    if args.slot_count <= 0 or args.page_bundles <= 0 or args.tokens_per_page <= 0:
        parser.error(
            "slot-count, page-bundles, and tokens-per-page must be positive")
    if args.output_multiplier <= 0.0:
        parser.error("output-multiplier must be positive")
    if (args.max_overlap_prefill_tokens < 0 or args.ttft_target_ms <= 0.0
            or args.tpot_target_ms <= 0.0):
        parser.error("phase overlap and SLO settings are invalid")
    if args.total_requests < 0:
        parser.error("total-requests must be non-negative")
    if args.max_cuda_graphs <= 0:
        parser.error("max-cuda-graphs must be positive")
    if args.max_prefill_cuda_graphs < 0 or args.max_decode_cuda_graphs < 0:
        parser.error("phase CUDA graph limits must be non-negative")
    if args.max_cuda_graph_mib < 0 or args.max_prefill_cuda_graph_mib < -1 or args.max_decode_cuda_graph_mib < -1:
        parser.error(
            "CUDA graph memory budgets are outside the supported range")
    if args.cuda_graph_charge_mib <= 0:
        parser.error("cuda-graph-charge-mib must be positive")
    if args.cuda_graph_reserve_mib < 0 or args.prefill_token_budget < 0:
        parser.error(
            "CUDA graph reserve and prefill token budget must be non-negative")
    if (args.page_reservation_headroom_tokens < 0
            or args.page_reservation_overcommit_bundles < 0
            or args.page_reservation_growth_requests <= 0
            or args.full_reservation_prompt_threshold_tokens < 0):
        parser.error("page reservation policy values must be non-negative")
    if (args.min_page_growth_requests <= 0 or args.min_page_growth_requests
            > args.page_reservation_growth_requests
            or args.growth_tpot_target_ms <= 0.0):
        parser.error("adaptive page growth bounds and target are invalid")
    if ((args.dynamic_decode_batching or args.dynamic_prefill_batching
         or args.scheduler_profile != "custom" or args.tpot_hard_guard
         or args.require_direct_overlap_cost)
            and args.scheduler_cost_json is None):
        parser.error("dynamic batching requires --scheduler-cost-json")
    if (args.max_consecutive_overlap_batches <= 0
            or args.max_predicted_decode_debt_ms < 0.0):
        parser.error("TPOT hard guard bounds are invalid")
    if (args.min_dynamic_prefill_batch_size <= 0
            or args.min_dynamic_prefill_batch_size > max(args.prefill_batches)
            or args.prefill_cohort_size <= 0 or args.prefill_cohort_turns <= 0
            or not 0.0 < args.decode_slack_safety_factor <= 1.0):
        parser.error("wavefront prefill settings are invalid")
    if args.scheduler_cost_json is not None and not args.scheduler_cost_json.is_file(
    ):
        parser.error("scheduler-cost-json does not exist")
    if args.cuda_graph and "shared" in args.context_modes:
        parser.error("--cuda-graph requires independent-only context modes")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trace = args.output_dir / "materialized-trace.json"
    materialize_trace(args.source_trace, trace, args.seed, args.arrival_rate,
                      args.request_count, args.repeat_count,
                      args.total_requests, args.output_multiplier)
    cases = scenarios(args.prefill_batches, args.decode_batches,
                      args.context_modes)
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
                result = subprocess.run(command,
                                        stdout=log,
                                        stderr=subprocess.STDOUT,
                                        check=False,
                                        text=True)
            elapsed = time.monotonic() - started
            status = {
                "engine":
                engine.name,
                "context_mode":
                case.context_mode,
                "case":
                case.name,
                "return_code":
                result.returncode,
                "elapsed_s":
                elapsed,
                "materialized_requests":
                len(json.loads(trace.read_text(encoding="utf-8"))["requests"]),
                "arrival_rate":
                args.arrival_rate,
                "output_multiplier":
                args.output_multiplier,
                "cuda_graph":
                args.cuda_graph,
                "max_cuda_graphs":
                args.max_cuda_graphs if args.cuda_graph else 0,
                "max_prefill_cuda_graphs":
                args.max_prefill_cuda_graphs if args.cuda_graph else 0,
                "max_decode_cuda_graphs":
                args.max_decode_cuda_graphs if args.cuda_graph else 0,
                "max_cuda_graph_mib":
                args.max_cuda_graph_mib if args.cuda_graph else 0,
                "max_prefill_cuda_graph_mib":
                args.max_prefill_cuda_graph_mib if args.cuda_graph else -1,
                "max_decode_cuda_graph_mib":
                args.max_decode_cuda_graph_mib if args.cuda_graph else -1,
                "cuda_graph_charge_mib":
                args.cuda_graph_charge_mib if args.cuda_graph else 0,
                "cuda_graph_reserve_mib":
                args.cuda_graph_reserve_mib if args.cuda_graph else 0,
                "prefill_token_budget":
                args.prefill_token_budget,
                "max_overlap_prefill_tokens":
                args.max_overlap_prefill_tokens,
                "ttft_target_ms":
                args.ttft_target_ms,
                "tpot_target_ms":
                args.tpot_target_ms,
                "dynamic_decode_batching":
                args.dynamic_decode_batching,
                "dynamic_prefill_batching":
                args.dynamic_prefill_batching,
                "min_dynamic_prefill_batch_size":
                args.min_dynamic_prefill_batch_size,
                "prefill_slo_recovery":
                args.prefill_slo_recovery,
                "wavefront_prefill_batching":
                args.wavefront_prefill_batching,
                "prefill_cohort_size":
                args.prefill_cohort_size,
                "prefill_cohort_turns":
                args.prefill_cohort_turns,
                "decode_slack_safety_factor":
                args.decode_slack_safety_factor,
                "scheduler_cost_json":
                str(args.scheduler_cost_json or ""),
                "scheduler_profile":
                args.scheduler_profile,
                "tpot_hard_guard":
                args.tpot_hard_guard,
                "require_direct_overlap_cost":
                args.require_direct_overlap_cost,
                "max_consecutive_overlap_batches":
                args.max_consecutive_overlap_batches,
                "max_predicted_decode_debt_ms":
                args.max_predicted_decode_debt_ms,
                "adaptive_scheduler":
                args.adaptive_scheduler,
                "page_reservation_mode":
                args.page_reservation_mode,
                "page_reservation_headroom_tokens":
                args.page_reservation_headroom_tokens,
                "page_reservation_overcommit_bundles":
                args.page_reservation_overcommit_bundles,
                "page_reservation_growth_requests":
                args.page_reservation_growth_requests,
                "full_reservation_prompt_threshold_tokens":
                args.full_reservation_prompt_threshold_tokens,
                "adaptive_page_growth":
                args.adaptive_page_growth,
                "min_page_growth_requests":
                args.min_page_growth_requests,
                "growth_tpot_target_ms":
                args.growth_tpot_target_ms,
                "log":
                str(log_path),
            }
            statuses.append(status)
            print(
                f"[{index}/{total}] {engine.name}/{case.name}: rc={result.returncode}",
                flush=True)
            request_csv = case_dir / "requests.csv"
            dispatch_csv = case_dir / "requests-dispatch.csv"
            kernel_csv = case_dir / "kernel-groups.csv"
            if result.returncode == 0 and request_csv.exists(
            ) and dispatch_csv.exists() and kernel_csv.exists():
                summaries.append(
                    summarize_requests(request_csv, engine, case,
                                       result.returncode))
                dispatches.extend(
                    summarize_dispatch(dispatch_csv, engine, case))
                kernels.extend(summarize_kernel(kernel_csv, engine, case))
                pressure.append(
                    pressure_model(request_csv, engine, case,
                                   args.page_bundles, args.tokens_per_page,
                                   args.slot_count))
            elif not args.continue_on_error:
                tail = "\n".join(
                    log_path.read_text(encoding="utf-8",
                                       errors="replace").splitlines()[-40:])
                raise RuntimeError(
                    f"{engine.name}/{case.name} failed with {result.returncode}:\n{tail}"
                )
    write_csv(args.output_dir / "status.csv", statuses)
    write_csv(args.output_dir / "request-summary.csv", summaries)
    write_csv(args.output_dir / "dispatch-cost-table.csv", dispatches)
    write_csv(args.output_dir / "kernel-cost-table.csv", kernels)
    write_csv(args.output_dir / "page-pressure-model.csv", pressure)
    print(f"wrote matrix artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
