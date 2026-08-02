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
"""Summarize repeated legacy/indexed Gemma 4 VLM real-request runs."""

import argparse
import csv
import json
import math
import re
import statistics
from pathlib import Path

STAGE_NAMES = ("vision_encoder", "llm_prefill", "llm_generation")
TIMESTAMP_PATTERN = re.compile(r"^\[(\d+):(\d+):(\d+)\.(\d+)\]")


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    rank = (len(ordered) - 1) * fraction
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (rank - lower)


def timestamp_ms(line: str) -> float:
    match = TIMESTAMP_PATTERN.match(line)
    if match is None:
        raise RuntimeError(f"missing timestamp: {line.rstrip()}")
    hours, minutes, seconds, milliseconds = (int(value)
                                             for value in match.groups())
    return ((hours * 60 + minutes) * 60 + seconds) * 1000 + milliseconds


def serving_wall_ms(path: Path) -> float:
    start = None
    end = None
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if "Starting actual benchmark runs" in line:
                start = timestamp_ms(line)
            elif "Processing complete:" in line:
                end = timestamp_ms(line)
    if start is None or end is None:
        raise RuntimeError(f"missing benchmark boundary in {path}")
    if end < start:
        end += 24 * 60 * 60 * 1000
    return end - start


def response_signature(path: Path) -> tuple[tuple[str, str], ...]:
    with path.open(encoding="utf-8") as stream:
        responses = json.load(stream)["responses"]
    return tuple((response["finish_reason"], response["output_text"])
                 for response in responses)


def load_run(root: Path, mode: str, repeat: int) -> dict[str, float]:
    profile_path = root / f"{mode}-run-{repeat}-profile.json"
    output_path = root / f"{mode}-run-{repeat}-output.json"
    log_path = root / f"{mode}-run-{repeat}.log"
    with profile_path.open(encoding="utf-8") as stream:
        profile = json.load(stream)
    with output_path.open(encoding="utf-8") as stream:
        request_count = len(json.load(stream)["responses"])
    stages = {
        stage["stage_id"]: float(stage["total_gpu_time_ms"])
        for stage in profile["stages"]
    }
    missing = set(STAGE_NAMES) - set(stages)
    if missing:
        raise RuntimeError(f"{profile_path}: missing stages {sorted(missing)}")
    wall_ms = serving_wall_ms(log_path)
    return {
        "serving_wall_ms":
        wall_ms,
        "request_throughput_rps":
        request_count * 1000.0 / wall_ms,
        "vision_gpu_ms":
        stages["vision_encoder"],
        "prefill_gpu_ms":
        stages["llm_prefill"],
        "generation_gpu_ms":
        stages["llm_generation"],
        "measured_gpu_ms":
        sum(stages[name] for name in STAGE_NAMES),
        "computed_prefill_tokens":
        float(profile["prefill"]["computed_tokens"]),
        "generated_tokens":
        float(profile["generation"]["generated_tokens"]),
        "image_tokens":
        float(profile["multimodal"]["total_image_tokens"]),
        "prefill_tokens_per_second":
        float(profile["prefill"]["tokens_per_second"]),
        "generation_tokens_per_second":
        float(profile["generation"]["tokens_per_second"]),
        "generation_ms_per_token":
        float(profile["generation"]["average_time_per_token_ms"]),
        "peak_gpu_memory_mb":
        float(profile["peak_gpu_memory_mb"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("output_csv", type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--baseline", default="legacy")
    parser.add_argument("--candidate", default="indexed")
    args = parser.parse_args()

    variants = (args.baseline, args.candidate)
    runs = {
        mode: [
            load_run(args.result_dir, mode, repeat)
            for repeat in range(1, args.repeats + 1)
        ]
        for mode in variants
    }
    baseline_signature = response_signature(
        args.result_dir / f"{args.baseline}-run-1-output.json")
    outputs_identical = all(
        response_signature(args.result_dir /
                           f"{mode}-run-{repeat}-output.json") ==
        baseline_signature for mode in variants
        for repeat in range(1, args.repeats + 1))

    metrics = tuple(runs[args.baseline][0])
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as stream:
        fieldnames = ("baseline_variant", "candidate_variant", "metric",
                      "stat", "baseline", "candidate",
                      "candidate_delta_percent")
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for metric in metrics:
            baseline_values = [run[metric] for run in runs[args.baseline]]
            candidate_values = [run[metric] for run in runs[args.candidate]]
            for stat, baseline_value, candidate_value in (
                ("median", statistics.median(baseline_values),
                 statistics.median(candidate_values)),
                ("p95", percentile(baseline_values,
                                   0.95), percentile(candidate_values, 0.95)),
            ):
                writer.writerow({
                    "baseline_variant":
                    args.baseline,
                    "candidate_variant":
                    args.candidate,
                    "metric":
                    metric,
                    "stat":
                    stat,
                    "baseline":
                    f"{baseline_value:.6f}",
                    "candidate":
                    f"{candidate_value:.6f}",
                    "candidate_delta_percent":
                    f"{(candidate_value / baseline_value - 1.0) * 100.0:.3f}",
                })

    print(f"baseline={args.baseline}")
    print(f"candidate={args.candidate}")
    print(f"outputs_identical={str(outputs_identical).lower()}")
    print(f"summary_csv={args.output_csv}")


if __name__ == "__main__":
    main()
