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
"""Summarize direct-overlap coverage misses into targeted probe shapes."""

import argparse
import csv
from collections import Counter
from pathlib import Path


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--chunk-length-buckets",
                        type=int,
                        nargs="+",
                        default=[32, 64, 96, 128])
    parser.add_argument("--past-kv-buckets",
                        type=int,
                        nargs="+",
                        default=[0, 128, 512, 1024, 1536, 2048])
    parser.add_argument("--decode-batch-buckets",
                        type=int,
                        nargs="+",
                        default=[0, 1, 2, 4, 8, 16, 32, 64])
    parser.add_argument("--decode-context-buckets",
                        type=int,
                        nargs="+",
                        default=[128, 512, 1024, 1536, 2048])
    args = parser.parse_args()

    paths = discover(args.input)
    if not paths:
        parser.error("no requests-dispatch.csv inputs found")
    required = {
        "prefill_cost_coverage_miss", "overlap_evaluated_by_cost",
        "prefill_cost_lookup_rows", "prefill_cost_lookup_chunk_length",
        "prefill_cost_lookup_max_past_kv_length", "planned_decode_batch",
        "planned_decode_max_context_length"
    }
    shapes: Counter[tuple[int, int, int, int, int, bool]] = Counter()
    dispatches = 0
    evaluated = 0
    misses = 0
    for path in paths:
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames is None or not required.issubset(
                    reader.fieldnames):
                raise RuntimeError(
                    f"dispatch telemetry lacks cost coverage fields: {path}")
            for row in reader:
                dispatches += 1
                evaluated += int(row["overlap_evaluated_by_cost"])
                if row["prefill_cost_coverage_miss"] != "1":
                    continue
                misses += 1
                raw_past = int(row["prefill_cost_lookup_max_past_kv_length"])
                shapes[(int(row["prefill_cost_lookup_rows"]),
                        upper_bucket(
                            int(row["prefill_cost_lookup_chunk_length"]),
                            args.chunk_length_buckets),
                        upper_bucket(raw_past, args.past_kv_buckets),
                        upper_bucket(int(row["planned_decode_batch"]),
                                     args.decode_batch_buckets),
                        upper_bucket(
                            int(row["planned_decode_max_context_length"]),
                            args.decode_context_buckets), raw_past == 0)] += 1

    rows = [{
        "prefill_batch_size": shape[0],
        "chunk_length": shape[1],
        "max_prefill_past_kv_length": shape[2],
        "decode_batch_size": shape[3],
        "max_decode_context_length": shape[4],
        "initial_chunk": int(shape[5]),
        "misses": count,
    } for shape, count in sorted(shapes.items(),
                                 key=lambda item: (-item[1], item[0]))]
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream,
                                fieldnames=[
                                    "prefill_batch_size", "chunk_length",
                                    "max_prefill_past_kv_length",
                                    "decode_batch_size",
                                    "max_decode_context_length",
                                    "initial_chunk", "misses"
                                ])
        writer.writeheader()
        writer.writerows(rows)
    print(f"dispatches={dispatches} evaluated={evaluated} "
          f"coverage_misses={misses} recommended_probes={len(rows)}")


if __name__ == "__main__":
    main()
