#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Profile isolated decode costs across batch and past-KV shapes."""

import argparse
import csv
import json
import math
import statistics
import subprocess
from pathlib import Path


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    rank = (len(ordered) - 1) * fraction
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (rank - lower)


def read_samples(path: Path) -> list[float]:
    with path.open(newline="", encoding="utf-8") as stream:
        values = [float(row["latency_ms"]) for row in csv.DictReader(stream)]
    if not values:
        raise RuntimeError(f"empty decode sample CSV: {path}")
    return values


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
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
    parser.add_argument("--decode-batches",
                        type=int,
                        nargs="+",
                        default=[16, 24, 32, 48, 64])
    parser.add_argument("--past-kv-lengths",
                        type=int,
                        nargs="+",
                        default=[128, 512, 1024, 1536])
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--tokens-per-page", type=int, default=128)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    config = json.loads(
        (args.engine_dir / "config.json").read_text(encoding="utf-8"))
    builder = config["builder_config"]
    max_decode = int(
        builder.get("max_decode_batch_size", builder["max_batch_size"]))
    max_kv = int(builder["max_kv_cache_capacity"])
    page_bundles = int(builder.get("kv_cache_page_bundles", 0))
    if max(args.decode_batches) > max_decode:
        parser.error("requested decode batch exceeds the engine contract")
    if max(args.past_kv_lengths) + 1 > max_kv:
        parser.error("requested past-KV length exceeds the engine contract")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    modes = (("enqueue", True), ("cuda_graph", False))
    cases = [(batch, past, mode, no_graph) for past in args.past_kv_lengths
             for batch in args.decode_batches for mode, no_graph in modes]
    for index, (batch, past, mode, no_graph) in enumerate(cases, start=1):
        required_bundles = batch * math.ceil((past + 1) / args.tokens_per_page)
        capacity_limited = page_bundles > 0 and required_bundles > page_bundles
        common = {
            "decode_batch": batch,
            "past_kv_len": past,
            "mode": mode,
            "required_page_bundles": required_bundles,
            "available_page_bundles": page_bundles,
        }
        if capacity_limited:
            print(
                f"[{index}/{len(cases)}] D{batch}/KV{past}/{mode}: "
                "capacity-limited",
                flush=True)
            rows.append({
                **common,
                "status": "capacity_limited",
                "samples": 0,
                "decode_mean_ms": "",
                "decode_median_ms": "",
                "decode_p95_ms": "",
                "decode_max_ms": "",
                "batch_tokens_per_second_median": "",
            })
            continue
        case_dir = args.output_dir / f"d{batch}_kv{past}_{mode}"
        case_dir.mkdir(parents=True, exist_ok=True)
        sample_path = case_dir / f"e2e_decode_pastkvlen{past}_samples.csv"
        log_path = case_dir / "run.log"
        command = [
            str(args.bench), "--engineDir",
            str(args.engine_dir), "--mode", "decode", "--batchSize",
            str(batch), "--pastKVLen",
            str(past), "--warmup",
            str(args.warmup), "--iterations",
            str(args.iterations), "--outputDir",
            str(case_dir)
        ]
        if no_graph:
            command.append("--noCudaGraph")
        if not args.skip_existing or not sample_path.exists():
            print(f"[{index}/{len(cases)}] D{batch}/KV{past}/{mode}",
                  flush=True)
            with log_path.open("w", encoding="utf-8") as log:
                completed = subprocess.run(command,
                                           stdout=log,
                                           stderr=subprocess.STDOUT,
                                           check=False,
                                           text=True)
            if completed.returncode != 0:
                tail = "\n".join(
                    log_path.read_text(encoding="utf-8",
                                       errors="replace").splitlines()[-50:])
                raise RuntimeError(f"D{batch}/KV{past}/{mode} failed with "
                                   f"{completed.returncode}:\n{tail}")

        values = read_samples(sample_path)
        median_ms = statistics.median(values)
        rows.append({
            **common,
            "status":
            "measured",
            "samples":
            len(values),
            "decode_mean_ms":
            statistics.mean(values),
            "decode_median_ms":
            median_ms,
            "decode_p95_ms":
            percentile(values, 0.95),
            "decode_max_ms":
            max(values),
            "batch_tokens_per_second_median":
            batch / (median_ms / 1000.0),
        })

    write_csv(args.output_dir / "decode-cost-table.csv", rows)
    print(f"wrote {args.output_dir / 'decode-cost-table.csv'}", flush=True)


if __name__ == "__main__":
    main()
