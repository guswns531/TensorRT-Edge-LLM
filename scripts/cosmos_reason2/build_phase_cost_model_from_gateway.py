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
"""Build a packed phase scheduler cost model from PHASE_METRIC gateway logs."""

import argparse
import collections
import json
import math
import pathlib
import statistics
from typing import Any, Iterable

BATCH_BUCKETS = (1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64)
PREFILL_BATCH_BUCKETS = (1, 2, 4, 8)
CHUNK_BUCKETS = (32, 64, 96, 128)
CONTEXT_BUCKETS = (128, 512, 1024, 1536, 2048)


def upper_bucket(value: int, boundaries: Iterable[int]) -> int:
    for boundary in boundaries:
        if value <= boundary:
            return boundary
    return tuple(boundaries)[-1]


def percentile(values: list[float], ratio: float = 0.95) -> float:
    ordered = sorted(values)
    return ordered[math.ceil(ratio * len(ordered)) - 1]


def read_metrics(paths: list[pathlib.Path]) -> list[dict[str, Any]]:
    rows = []
    for path in paths:
        with path.open(encoding="utf-8", errors="replace") as stream:
            for line in stream:
                if line.startswith("PHASE_METRIC\t"):
                    rows.append(json.loads(line.split("\t", 1)[1]))
    return rows


def summarize(
        rows: list[dict[str, Any]],
        engine_sha256: str,
        min_samples: int,
        source_paths: list[pathlib.Path] | None = None) -> dict[str, Any]:
    decode_groups: dict[tuple[int, int], list[dict[str, Any]]]
    decode_groups = collections.defaultdict(list)
    for row in rows:
        if row["kind"] == 2 and row["decode_batch"] > 0:
            key = (upper_bucket(row["decode_batch"], BATCH_BUCKETS),
                   upper_bucket(max(1, row.get("decode_context_max", 1)),
                                CONTEXT_BUCKETS))
            decode_groups[key].append(row)
    decode = []
    decode_baseline = {}
    for (batch, context), samples in sorted(decode_groups.items()):
        if len(samples) < min_samples:
            continue
        timings = [sample["decode_gpu_ms"] for sample in samples]
        p95 = percentile(timings)
        decode_baseline[(batch, context)] = p95
        decode.append({
            "batch_size":
            batch,
            "max_context_length":
            context,
            "max_total_context_tokens":
            max(sample.get("decode_context_tokens", 0) for sample in samples),
            "samples":
            len(samples),
            "median_gpu_ms":
            statistics.median(timings),
            "p95_gpu_ms":
            p95,
        })

    prefill_groups = collections.defaultdict(list)
    overlap_groups = collections.defaultdict(list)
    for row in rows:
        if row["prefill_batch"] <= 0:
            continue
        prefill_batch = upper_bucket(row["prefill_batch"],
                                     PREFILL_BATCH_BUCKETS)
        chunk = upper_bucket(max(1, row.get("prefill_chunk_length", 1)),
                             CHUNK_BUCKETS)
        past_value = row.get("prefill_past_kv_max", 0)
        past = upper_bucket(past_value,
                            CONTEXT_BUCKETS) if past_value > 0 else 0
        decode_batch = (upper_bucket(row["decode_batch"], BATCH_BUCKETS)
                        if row["decode_batch"] > 0 else 0)
        initial = row.get("prefill_initial_rows", 0) > 0
        prefill_groups[(prefill_batch, chunk, past, decode_batch,
                        initial)].append(row)
        if row["kind"] == 3 and row["decode_batch"] > 0:
            decode_context = upper_bucket(
                max(1, row.get("decode_context_max", 1)), CONTEXT_BUCKETS)
            overlap_groups[(prefill_batch, decode_batch, chunk, past,
                            decode_context, initial)].append(row)

    prefill = []
    for key, samples in sorted(prefill_groups.items()):
        if len(samples) < min_samples:
            continue
        p_batch, chunk, past, d_batch, initial = key
        timings = [sample["prefill_gpu_ms"] for sample in samples]
        slowdown = 0.0
        if d_batch > 0:
            observed = percentile(
                [sample["decode_gpu_ms"] for sample in samples])
            candidates = [
                cost for (batch, _), cost in decode_baseline.items()
                if batch == d_batch
            ]
            baseline = min(candidates) if candidates else observed
            slowdown = max(0.0, observed - baseline)
        prefill.append({
            "batch_size": p_batch,
            "chunk_length": chunk,
            "max_past_kv_length": past,
            "max_concurrent_decode_batch_size": d_batch,
            "initial_chunk": initial,
            "samples": len(samples),
            "median_gpu_ms": statistics.median(timings),
            "p95_gpu_ms": percentile(timings),
            "decode_slowdown_p95_ms": slowdown,
        })

    overlap = []
    for key, samples in sorted(overlap_groups.items()):
        if len(samples) < min_samples:
            continue
        p_batch, d_batch, chunk, past, d_context, initial = key
        prefill_times = [sample["prefill_gpu_ms"] for sample in samples]
        decode_times = [sample["decode_gpu_ms"] for sample in samples]
        makespans = [sample["makespan_gpu_ms"] for sample in samples]
        decode_p95 = percentile(decode_times)
        baseline = decode_baseline.get((d_batch, d_context), decode_p95)
        overlap.append({
            "prefill_batch_size":
            p_batch,
            "decode_batch_size":
            d_batch,
            "chunk_length":
            chunk,
            "max_prefill_past_kv_length":
            past,
            "max_decode_context_length":
            d_context,
            "initial_chunk":
            initial,
            "samples":
            len(samples),
            "prefill_p95_gpu_ms":
            percentile(prefill_times),
            "decode_p95_gpu_ms":
            decode_p95,
            "makespan_p95_gpu_ms":
            percentile(makespans),
            "decode_slowdown_p95_ms":
            max(0.0, decode_p95 - baseline),
        })
    return {
        "schema_version": 7,
        "prefill_layout": "packed",
        "engine_sha256": engine_sha256,
        "source_files": [str(path) for path in source_paths or []],
        "decode": decode,
        "prefill": prefill,
        "overlap": overlap,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", type=pathlib.Path, nargs="+", required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--engine-sha256", required=True)
    parser.add_argument("--min-samples", type=int, default=2)
    args = parser.parse_args()
    if args.min_samples <= 0:
        parser.error("min-samples must be positive")
    result = summarize(read_metrics(args.logs), args.engine_sha256,
                       args.min_samples, args.logs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n",
                           encoding="utf-8")
    print(
        json.dumps({
            key: len(result[key])
            for key in ("decode", "prefill", "overlap")
        }))


if __name__ == "__main__":
    main()
