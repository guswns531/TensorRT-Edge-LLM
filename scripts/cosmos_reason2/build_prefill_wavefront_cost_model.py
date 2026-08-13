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
"""Build a decode-slack-aware prefill cost model from dispatch CUDA events."""

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position -
                                                                 lower)


def upper_bucket(value: int, buckets: list[int]) -> int:
    for bucket in buckets:
        if value <= bucket:
            return bucket
    return buckets[-1]


def discover(inputs: list[Path]) -> list[Path]:
    paths: list[Path] = []
    for source in inputs:
        if source.is_dir():
            paths.extend(sorted(source.rglob("requests-dispatch.csv")))
        elif source.is_file():
            paths.append(source)
    return sorted(set(paths))


def read_rows(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                row["source_file"] = str(path)
                rows.append(row)
    return rows


def decode_context_length(row: dict[str, str], decode_batch: int) -> int:
    planned_max = row.get("planned_decode_max_context_length")
    if planned_max:
        return int(planned_max)
    return math.ceil(int(row["decode_context_tokens"]) / decode_batch)


def select_decode_cost_samples(
        all_samples: list[float],
        isolated_samples: list[float]) -> tuple[list[float], str]:
    """Prefer decode-only samples so overlap interference is modeled separately."""
    if isolated_samples:
        return isolated_samples, "decode_only"
    return all_samples, "all_dispatch_fallback"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--past-kv-buckets",
                        type=int,
                        nargs="+",
                        default=[0, 128, 512, 1024, 1536, 2048])
    parser.add_argument("--decode-batch-buckets",
                        type=int,
                        nargs="+",
                        default=[0, 8, 16, 32, 64])
    parser.add_argument("--decode-context-buckets",
                        type=int,
                        nargs="+",
                        default=[128, 512, 1024, 1536, 2048])
    parser.add_argument("--chunk-length-buckets",
                        type=int,
                        nargs="+",
                        default=[32, 64, 96, 128])
    parser.add_argument("--min-samples", type=int, default=3)
    parser.add_argument("--prefill-layout",
                        choices=["dense", "packed"],
                        default="dense")
    args = parser.parse_args()

    if args.min_samples <= 0:
        parser.error("min-samples must be positive")
    paths = discover(args.input)
    if not paths:
        parser.error("no requests-dispatch.csv inputs found")
    rows = read_rows(paths)
    required = {
        "prefill_initial_rows", "prefill_continuation_rows",
        "prefill_past_kv_max", "decode_context_tokens"
    }
    if not rows or not required.issubset(rows[0]):
        raise RuntimeError(
            "dispatch inputs do not contain detailed prefill telemetry")

    decode_baseline: dict[tuple[int, int], list[float]] = {}
    decode_points: dict[tuple[int, int], list[float]] = {}
    for row in rows:
        decode_batch = int(row["decode_batch"])
        if decode_batch <= 0 or float(row["decode_gpu_ms"]) <= 0.0:
            continue
        context_bucket = upper_bucket(decode_context_length(row, decode_batch),
                                      args.decode_context_buckets)
        batch_bucket = upper_bucket(decode_batch, args.decode_batch_buckets)
        key = (batch_bucket, context_bucket)
        decode_points.setdefault(key, []).append(float(row["decode_gpu_ms"]))
        if int(row["prefill_batch"]) == 0:
            decode_baseline.setdefault(key,
                                       []).append(float(row["decode_gpu_ms"]))

    baseline_medians = {
        key: statistics.median(values)
        for key, values in decode_baseline.items()
    }

    def baseline_for(batch: int, context: int) -> float:
        exact = baseline_medians.get((batch, context))
        if exact is not None:
            return exact
        candidates = [
            (abs(candidate_batch - batch) +
             abs(candidate_context - context) / 2048.0, value)
            for (candidate_batch,
                 candidate_context), value in baseline_medians.items()
        ]
        return min(candidates)[1] if candidates else 0.0

    grouped: dict[tuple[int, int, int, int, bool], list[dict[str, float]]] = {}
    overlap_grouped: dict[tuple[int, int, int, int, int, bool],
                          list[dict[str, float]]] = {}
    for row in rows:
        prefill_batch = int(row["prefill_batch"])
        if prefill_batch <= 0 or float(row["prefill_gpu_ms"]) <= 0.0:
            continue
        initial_rows = int(row["prefill_initial_rows"])
        continuation_rows = int(row["prefill_continuation_rows"])
        if initial_rows == prefill_batch:
            initial = True
        elif continuation_rows == prefill_batch:
            initial = False
        else:
            continue
        padded_tokens = int(
            row.get("prefill_padded_tokens") or row["prefill_tokens"])
        chunk_length = upper_bucket(padded_tokens // prefill_batch,
                                    args.chunk_length_buckets)
        past_bucket = upper_bucket(int(row["prefill_past_kv_max"]),
                                   args.past_kv_buckets)
        decode_batch = int(row["decode_batch"])
        decode_bucket = upper_bucket(decode_batch, args.decode_batch_buckets)
        slowdown = 0.0
        if decode_batch > 0 and float(row["decode_gpu_ms"]) > 0.0:
            context_bucket = upper_bucket(
                decode_context_length(row, decode_batch),
                args.decode_context_buckets)
            slowdown = max(
                0.0,
                float(row["decode_gpu_ms"]) -
                baseline_for(decode_bucket, context_bucket))
        key = (prefill_batch, chunk_length, past_bucket, decode_bucket,
               initial)
        grouped.setdefault(key, []).append({
            "prefill_gpu_ms":
            float(row["prefill_gpu_ms"]),
            "decode_slowdown_ms":
            slowdown,
            "packing_efficiency":
            float(row.get("prefill_packing_efficiency") or 1.0),
        })
        if decode_batch > 0 and float(row["decode_gpu_ms"]) > 0.0:
            context_bucket = upper_bucket(
                decode_context_length(row, decode_batch),
                args.decode_context_buckets)
            overlap_key = (prefill_batch, decode_bucket, chunk_length,
                           past_bucket, context_bucket, initial)
            overlap_grouped.setdefault(overlap_key, []).append({
                "prefill_gpu_ms":
                float(row["prefill_gpu_ms"]),
                "decode_gpu_ms":
                float(row["decode_gpu_ms"]),
                "makespan_gpu_ms":
                float(row["makespan_gpu_ms"]),
                "decode_slowdown_ms":
                slowdown,
            })

    prefill = []
    csv_rows = []
    for key, samples in sorted(grouped.items()):
        if len(samples) < args.min_samples:
            continue
        batch, chunk, past, decode, initial = key
        gpu = [sample["prefill_gpu_ms"] for sample in samples]
        slowdown = [sample["decode_slowdown_ms"] for sample in samples]
        packing = [sample["packing_efficiency"] for sample in samples]
        point = {
            "batch_size": batch,
            "chunk_length": chunk,
            "max_past_kv_length": past,
            "max_concurrent_decode_batch_size": decode,
            "initial_chunk": initial,
            "samples": len(samples),
            "median_gpu_ms": statistics.median(gpu),
            "p95_gpu_ms": percentile(gpu, 0.95),
            "decode_slowdown_p95_ms": percentile(slowdown, 0.95),
            "median_packing_efficiency": statistics.median(packing),
        }
        prefill.append(point)
        csv_rows.append({"phase": "prefill", **point})

    decode = []
    for (batch, context), values in sorted(decode_points.items()):
        samples, sample_scope = select_decode_cost_samples(
            values, decode_baseline.get((batch, context), []))
        if len(samples) < args.min_samples:
            continue
        point = {
            "batch_size": batch,
            "max_context_length": context,
            "samples": len(samples),
            "sample_scope": sample_scope,
            "median_gpu_ms": statistics.median(samples),
            "p95_gpu_ms": percentile(samples, 0.95),
        }
        decode.append(point)
        csv_rows.append({"phase": "decode", **point})

    overlap = []
    for key, samples in sorted(overlap_grouped.items()):
        if len(samples) < args.min_samples:
            continue
        prefill_batch, decode_batch, chunk, past, context, initial = key
        point = {
            "prefill_batch_size":
            prefill_batch,
            "decode_batch_size":
            decode_batch,
            "chunk_length":
            chunk,
            "max_prefill_past_kv_length":
            past,
            "max_decode_context_length":
            context,
            "initial_chunk":
            initial,
            "samples":
            len(samples),
            "prefill_p95_gpu_ms":
            percentile([sample["prefill_gpu_ms"] for sample in samples], 0.95),
            "decode_p95_gpu_ms":
            percentile([sample["decode_gpu_ms"] for sample in samples], 0.95),
            "makespan_p95_gpu_ms":
            percentile([sample["makespan_gpu_ms"] for sample in samples],
                       0.95),
            "decode_slowdown_p95_ms":
            percentile([sample["decode_slowdown_ms"] for sample in samples],
                       0.95),
        }
        overlap.append(point)
        csv_rows.append({"phase": "overlap", **point})
    if not prefill or not decode or not overlap:
        raise RuntimeError(
            "insufficient detailed prefill/decode/overlap samples")

    root = {
        "schema_version": 6,
        "prefill_layout": args.prefill_layout,
        "source_files": [str(path) for path in paths],
        "decode": decode,
        "prefill": prefill,
        "overlap": overlap,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(root, indent=2) + "\n",
                                encoding="utf-8")
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as stream:
        csv_rows = [{
            "prefill_layout": args.prefill_layout,
            **row
        } for row in csv_rows]
        fieldnames = ["phase", "prefill_layout"] + sorted({
            field
            for row in csv_rows
            for field in row if field not in {"phase", "prefill_layout"}
        })
        writer = csv.DictWriter(stream,
                                fieldnames=fieldnames,
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(csv_rows)


if __name__ == "__main__":
    main()
