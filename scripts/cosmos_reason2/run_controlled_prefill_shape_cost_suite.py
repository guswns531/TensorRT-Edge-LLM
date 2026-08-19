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
"""Run feasibility-aware controlled P/C/D/KV CUDA-event cost sweeps."""

import argparse
import csv
import json
import math
import statistics
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Case:
    phase: str
    prefill_batch: int
    chunk: int
    decode_batch: int
    past_kv: int

    @property
    def prefill_past_kv(self) -> int:
        return 0 if self.phase == "initial" else self.past_kv

    @property
    def name(self) -> str:
        return (f"{self.phase}_p{self.prefill_batch}_c{self.chunk}_"
                f"d{self.decode_batch}_kv{self.past_kv}")


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position -
                                                                 lower)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise RuntimeError(f"empty CSV: {path}")
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError(f"no rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def required_page_bundles(case: Case, tokens_per_page: int) -> int:
    prefill_length = case.prefill_past_kv + case.chunk
    decode_length = case.past_kv + 1
    return (case.prefill_batch * math.ceil(prefill_length / tokens_per_page) +
            case.decode_batch * math.ceil(decode_length / tokens_per_page))


def command_for(args: argparse.Namespace, case: Case, case_dir: Path,
                packed_prefill: bool) -> list[str]:
    command = [
        str(args.bench), "--engineDir",
        str(args.engine_dir), "--prefillBatch",
        str(case.prefill_batch), "--decodeBatch",
        str(case.decode_batch), "--slotCount",
        str(args.slot_count), "--inputLen",
        str(case.chunk), "--prefillChunkSize",
        str(case.chunk), "--prefillPastKVLen",
        str(case.prefill_past_kv), "--pastKVLen",
        str(case.past_kv), "--warmup",
        str(args.warmup), "--iterations",
        str(args.iterations), "--contextAdapter", "--trtContextMode",
        "independent", "--outputCsv",
        str(case_dir / "samples.csv"), "--kernelGroupCsv",
        str(case_dir / "kernel-groups.csv")
    ]
    if packed_prefill:
        command.append("--packedPrefillTokenLayout")
    return command


def normalize_dispatch_rows(
        case: Case, rows: list[dict[str, str]]) -> list[dict[str, object]]:
    normalized: list[dict[str, object]] = []

    def append(prefill_batch: int, decode_batch: int, prefill_ms: float,
               decode_ms: float, makespan_ms: float, mode: str,
               iteration: int) -> None:
        normalized.append({
            "source_case":
            case.name,
            "iteration":
            iteration,
            "mode":
            mode,
            "prefill_batch":
            prefill_batch,
            "decode_batch":
            decode_batch,
            "prefill_tokens":
            prefill_batch * case.chunk,
            "prefill_padded_tokens":
            prefill_batch * case.chunk,
            "prefill_packing_efficiency":
            1.0,
            "prefill_initial_rows":
            prefill_batch if case.phase == "initial" else 0,
            "prefill_continuation_rows":
            prefill_batch if case.phase == "continuation" else 0,
            "prefill_final_rows":
            prefill_batch,
            "prefill_past_kv_max":
            case.prefill_past_kv,
            "decode_context_tokens":
            decode_batch * case.past_kv,
            "planned_decode_max_context_length":
            case.past_kv if decode_batch > 0 else 0,
            "prefill_gpu_ms":
            prefill_ms,
            "decode_gpu_ms":
            decode_ms,
            "makespan_gpu_ms":
            makespan_ms,
        })

    for row in rows:
        iteration = int(row["iteration"])
        prefill_ms = float(row["prefill_ms"])
        decode_ms = float(row["decode_ms"])
        if row["mode"] == "sequential":
            append(case.prefill_batch, 0, prefill_ms, 0.0, prefill_ms,
                   "prefill_only", iteration)
            append(0, case.decode_batch, 0.0, decode_ms, decode_ms,
                   "decode_only", iteration)
        else:
            append(case.prefill_batch, case.decode_batch, prefill_ms,
                   decode_ms, float(row["makespan_ms"]), "overlap", iteration)
    return normalized


def summarize_case(case: Case, rows: list[dict[str, str]], page_bundles: int,
                   tokens_per_page: int) -> dict[str, object]:
    sequential = [row for row in rows if row["mode"] == "sequential"]
    overlap = [row for row in rows if row["mode"] != "sequential"]
    return {
        "case":
        case.name,
        "phase":
        case.phase,
        "prefill_batch":
        case.prefill_batch,
        "chunk":
        case.chunk,
        "decode_batch":
        case.decode_batch,
        "prefill_past_kv":
        case.prefill_past_kv,
        "decode_past_kv":
        case.past_kv,
        "required_page_bundles":
        required_page_bundles(case, tokens_per_page),
        "page_bundles":
        page_bundles,
        "samples_per_mode":
        len(sequential),
        "sequential_prefill_median_ms":
        statistics.median(float(row["prefill_ms"]) for row in sequential),
        "sequential_prefill_p95_ms":
        percentile([float(row["prefill_ms"]) for row in sequential], 0.95),
        "sequential_decode_median_ms":
        statistics.median(float(row["decode_ms"]) for row in sequential),
        "sequential_decode_p95_ms":
        percentile([float(row["decode_ms"]) for row in sequential], 0.95),
        "overlap_prefill_median_ms":
        statistics.median(float(row["prefill_ms"]) for row in overlap),
        "overlap_prefill_p95_ms":
        percentile([float(row["prefill_ms"]) for row in overlap], 0.95),
        "overlap_decode_median_ms":
        statistics.median(float(row["decode_ms"]) for row in overlap),
        "overlap_decode_p95_ms":
        percentile([float(row["decode_ms"]) for row in overlap], 0.95),
        "overlap_makespan_median_ms":
        statistics.median(float(row["makespan_ms"]) for row in overlap),
        "overlap_makespan_p95_ms":
        percentile([float(row["makespan_ms"]) for row in overlap], 0.95),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", type=Path, required=True)
    parser.add_argument("--engine-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefill-batches",
                        type=int,
                        nargs="+",
                        default=[1, 2, 4, 8])
    parser.add_argument("--chunks", type=int, nargs="+", default=[64, 128])
    parser.add_argument("--decode-batches",
                        type=int,
                        nargs="+",
                        default=[8, 16, 32, 64])
    parser.add_argument("--past-kv",
                        type=int,
                        nargs="+",
                        default=[128, 512, 1024, 1536])
    parser.add_argument("--prefill-phases",
                        nargs="+",
                        choices=("initial", "continuation"),
                        default=["initial", "continuation"])
    parser.add_argument("--slot-count", type=int, default=80)
    parser.add_argument("--tokens-per-page", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    config = json.loads(
        (args.engine_dir / "config.json").read_text(encoding="utf-8"))
    builder = config["builder_config"]
    page_bundles = int(builder["kv_cache_page_bundles"])
    max_batch = int(builder["max_batch_size"])
    max_prefill = int(builder["max_prefill_batch_size"])
    max_decode = int(builder["max_decode_batch_size"])
    max_chunk = int(builder["max_prefill_chunk_tokens"])
    max_kv = int(builder["max_kv_cache_capacity"])
    packed_prefill = bool(config.get("packed_prefill", False))
    values = (args.prefill_batches + args.chunks + args.decode_batches +
              args.past_kv)
    if (not values or any(value <= 0 for value in values)
            or args.slot_count <= 0 or args.tokens_per_page <= 0
            or args.warmup < 0 or args.iterations <= 0 or args.max_cases < 0):
        parser.error("sweep dimensions and iteration counts must be positive")

    all_cases = [
        Case(phase, prefill, chunk, decode, past)
        for phase in args.prefill_phases for prefill in args.prefill_batches
        for chunk in args.chunks for decode in args.decode_batches
        for past in args.past_kv
    ]
    selected = [
        case for case in all_cases if case.prefill_batch <= max_prefill
        and case.decode_batch <= max_decode and case.prefill_batch +
        case.decode_batch <= min(args.slot_count, max_batch)
        and case.chunk <= max_chunk and case.prefill_past_kv +
        case.chunk <= max_kv and case.past_kv + 1 <= max_kv
        and required_page_bundles(case, args.tokens_per_page) <= page_bundles
    ]
    if args.max_cases > 0:
        selected = selected[:args.max_cases]
    if not selected:
        parser.error("no requested shape fits the engine and page-pool limits")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    normalized: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    statuses: list[dict[str, object]] = []
    for index, case in enumerate(selected, start=1):
        case_dir = args.output_dir / case.name
        case_dir.mkdir(parents=True, exist_ok=True)
        sample_path = case_dir / "samples.csv"
        log_path = case_dir / "run.log"
        if not args.skip_existing or not sample_path.exists():
            print(f"[{index}/{len(selected)}] {case.name}", flush=True)
            with log_path.open("w", encoding="utf-8") as log:
                result = subprocess.run(command_for(args, case, case_dir,
                                                    packed_prefill),
                                        stdout=log,
                                        stderr=subprocess.STDOUT,
                                        text=True,
                                        check=False)
            statuses.append({
                "case": case.name,
                "return_code": result.returncode,
                "log": str(log_path),
            })
            if result.returncode != 0:
                tail = "\n".join(
                    log_path.read_text(encoding="utf-8",
                                       errors="replace").splitlines()[-40:])
                if not args.continue_on_error:
                    raise RuntimeError(
                        f"{case.name} failed with {result.returncode}:\n{tail}"
                    )
                continue
        rows = read_csv(sample_path)
        normalized.extend(normalize_dispatch_rows(case, rows))
        summaries.append(
            summarize_case(case, rows, page_bundles, args.tokens_per_page))

    write_csv(args.output_dir / "controlled-dispatch.csv", normalized)
    write_csv(args.output_dir / "case-summary.csv", summaries)
    if statuses:
        write_csv(args.output_dir / "status.csv", statuses)
    manifest = {
        "schema_version": 1,
        "engine_dir": str(args.engine_dir),
        "engine_page_bundles": page_bundles,
        "tokens_per_page": args.tokens_per_page,
        "requested_cases": len(all_cases),
        "feasible_cases": len(selected),
        "completed_cases": len(summaries),
        "prefill_phases": args.prefill_phases,
        "prefill_batches": args.prefill_batches,
        "chunks": args.chunks,
        "decode_batches": args.decode_batches,
        "past_kv": args.past_kv,
        "warmup": args.warmup,
        "iterations": args.iterations,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
