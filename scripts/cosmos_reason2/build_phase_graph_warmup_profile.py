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
"""Build a model-independent CUDA graph warmup profile from dispatch telemetry."""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


def build_profile(dispatch_csv: Path, max_prefill_shapes: int,
                  max_decode_shapes: int,
                  repetitions: int) -> dict[str, object]:
    with dispatch_csv.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("dispatch CSV is empty")

    prefill: dict[tuple[int, int, bool],
                  dict[str, int]] = defaultdict(lambda: {
                      "count": 0,
                      "past_kv_length": 0
                  })
    decode: dict[int, dict[str, int]] = defaultdict(lambda: {
        "count": 0,
        "context_length": 1
    })
    for row in rows:
        prefill_batch = int(row["prefill_batch"])
        if prefill_batch > 0:
            padded_tokens = int(row["prefill_padded_tokens"])
            chunk_length = max(1, math.ceil(padded_tokens / prefill_batch))
            initial = int(row["prefill_initial_rows"]) == prefill_batch
            key = (prefill_batch, chunk_length, initial)
            prefill[key]["count"] += 1
            prefill[key]["past_kv_length"] = max(
                prefill[key]["past_kv_length"],
                int(row["prefill_past_kv_max"]))

        decode_batch = int(row["decode_batch"])
        if decode_batch > 0:
            context_length = int(row["planned_decode_max_context_length"])
            if context_length <= 0:
                context_length = max(
                    1,
                    math.ceil(
                        int(row["decode_context_tokens"]) / decode_batch))
            decode[decode_batch]["count"] += 1
            decode[decode_batch]["context_length"] = max(
                decode[decode_batch]["context_length"], context_length)

    ranked_prefill = sorted(prefill.items(),
                            key=lambda item:
                            (-item[1]["count"], -item[0][0], -item[0][1],
                             not item[0][2]))[:max_prefill_shapes]
    ranked_decode = sorted(decode.items(),
                           key=lambda item:
                           (-item[1]["count"], -item[0]))[:max_decode_shapes]

    return {
        "version":
        1,
        "source_dispatch_csv":
        str(dispatch_csv),
        "prefill": [{
            "batch_size":
            key[0],
            "chunk_length":
            key[1],
            "past_kv_length":
            0 if key[2] else max(1, stats["past_kv_length"]),
            "initial_chunk":
            key[2],
            "observed_dispatches":
            stats["count"],
            "repetitions":
            repetitions,
        } for key, stats in ranked_prefill],
        "decode": [{
            "batch_size": batch_size,
            "context_length": stats["context_length"],
            "observed_dispatches": stats["count"],
            "repetitions": repetitions,
        } for batch_size, stats in ranked_decode],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dispatch-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-prefill-shapes", type=int, default=4)
    parser.add_argument("--max-decode-shapes", type=int, default=32)
    parser.add_argument("--repetitions", type=int, default=2)
    args = parser.parse_args()
    if args.max_prefill_shapes <= 0 or args.max_decode_shapes <= 0:
        parser.error("phase shape limits must be positive")
    if args.repetitions < 2:
        parser.error(
            "automatic CUDA graph capture requires at least two repetitions")
    if not args.dispatch_csv.is_file():
        parser.error("dispatch CSV does not exist")

    profile = build_profile(args.dispatch_csv, args.max_prefill_shapes,
                            args.max_decode_shapes, args.repetitions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(profile, indent=2) + "\n",
                           encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
