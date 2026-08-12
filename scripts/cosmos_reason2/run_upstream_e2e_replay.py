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
"""Replay clean-upstream batch wall costs over a real arrival trace.

The public fixed-linear runtime cannot continuously admit requests.  This
runner therefore measures every homogeneous full-BS8 batch with the unmodified
``llm_inference`` binary, extracts batch completion wall times from its log,
and replays those costs over the original arrivals with an upstream-favourable
shortest-processing-time oracle.  A final partial batch is allowed only when
explicitly requested; requests are never padded.
"""

import argparse
import csv
import json
import math
import re
import statistics
import subprocess
from pathlib import Path

TIMESTAMP_RE = re.compile(r"\[(\d\d):(\d\d):(\d\d)\.(\d\d\d)\]")
PROCESSING_RE = re.compile(r"Processing \d+ batched requests")
RESPONSE_RE = re.compile(r"Response for request (\d+) batch 0")


def timestamp_ms(line: str) -> int:
    match = TIMESTAMP_RE.search(line)
    if match is None:
        raise RuntimeError(f"missing timestamp: {line}")
    hours, minutes, seconds, milliseconds = (int(value)
                                             for value in match.groups())
    return (((hours * 60) + minutes) * 60 + seconds) * 1000 + milliseconds


def elapsed_ms(start: int, end: int) -> int:
    day_ms = 24 * 60 * 60 * 1000
    return (end - start) % day_ms


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    rank = (len(ordered) - 1) * fraction
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (rank - lower)


def run_group(args: argparse.Namespace, output_length: int,
              output_dir: Path) -> tuple[list[float], dict]:
    input_path = args.input_dir / f"input-output{output_length}.json"
    group_dir = output_dir / f"output{output_length}"
    group_dir.mkdir(parents=True, exist_ok=True)
    profile_path = group_dir / "profile.json"
    command = [
        str(args.binary),
        "--engineDir",
        str(args.engine_dir),
        "--inputFile",
        str(input_path),
        "--outputFile",
        str(group_dir / "responses.json"),
        "--batchSize",
        str(args.batch_size),
        "--warmup",
        str(args.warmup),
        "--dumpProfile",
        "--profileOutputFile",
        str(profile_path),
        "--dumpOutput",
    ]
    if args.multimodal_engine_dir is not None:
        command[3:3] = [
            "--multimodalEngineDir",
            str(args.multimodal_engine_dir)
        ]
    log_path = group_dir / "run.log"
    if args.reuse_existing:
        if not log_path.exists() or not profile_path.exists():
            raise RuntimeError(
                f"output{output_length}: no existing log/profile to reuse")
        log = log_path.read_text(encoding="utf-8")
    else:
        completed = subprocess.run(command,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT,
                                   check=False,
                                   text=True)
        log = completed.stdout
        log_path.write_text(log, encoding="utf-8")
        if completed.returncode != 0:
            raise RuntimeError(
                f"output{output_length} failed with {completed.returncode}:\n"
                + "\n".join(log.splitlines()[-50:]))

    start_ms = None
    completion_ms: dict[int, int] = {}
    for line in log.splitlines():
        if start_ms is None and PROCESSING_RE.search(line):
            start_ms = timestamp_ms(line)
        response = RESPONSE_RE.search(line)
        if response is not None:
            completion_ms.setdefault(int(response.group(1)),
                                     timestamp_ms(line))
    if start_ms is None or not completion_ms:
        raise RuntimeError(
            f"output{output_length}: missing processing or response timestamps"
        )
    request_count = len(
        json.loads(input_path.read_text(encoding="utf-8"))["requests"])
    expected_batches = math.ceil(request_count / args.batch_size)
    if sorted(completion_ms) != list(range(expected_batches)):
        raise RuntimeError(
            f"output{output_length}: expected {expected_batches} batch completions, got {sorted(completion_ms)}"
        )
    relative = [
        float(elapsed_ms(start_ms, completion_ms[index]))
        for index in range(expected_batches)
    ]
    durations = [relative[0]] + [
        relative[index] - relative[index - 1]
        for index in range(1, len(relative))
    ]
    if any(duration <= 0 for duration in durations):
        raise RuntimeError(
            f"output{output_length}: non-positive batch duration: {durations}")
    return durations, json.loads(profile_path.read_text(encoding="utf-8"))


def materialize_group_inputs(trace_requests: list[dict], input_dir: Path,
                             output_lengths: list[int],
                             batch_size: int) -> None:
    input_dir.mkdir(parents=True, exist_ok=True)
    for output_length in output_lengths:
        requests = [{
            key: value
            for key, value in request.items() if key != "arrival_offset_us"
        } for request in trace_requests
                    if int(request["max_generate_length"]) == output_length]
        payload = {
            "batch_size": batch_size,
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 1,
            "max_generate_length": output_length,
            "requests": requests,
        }
        (input_dir / f"input-output{output_length}.json").write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def validate_group_inputs(
        trace_requests: list[dict], input_dir: Path, output_lengths: list[int],
        batch_size: int, allow_partial_batches: bool
) -> dict[int, list[list[tuple[int, dict]]]]:
    result = {}
    for output_length in output_lengths:
        original = [(index, request)
                    for index, request in enumerate(trace_requests)
                    if int(request["max_generate_length"]) == output_length]
        grouped = json.loads(
            (input_dir / f"input-output{output_length}.json").read_text(
                encoding="utf-8"))["requests"]
        if len(original) != len(grouped):
            raise RuntimeError(
                f"output{output_length}: grouped input count does not match source trace"
            )
        if len(original) % batch_size and not allow_partial_batches:
            raise RuntimeError(
                f"output{output_length}: input count does not form full BS{batch_size}"
            )
        for (_, request), upstream_request in zip(original, grouped):
            if request["messages"] != upstream_request["messages"]:
                raise RuntimeError(
                    f"output{output_length}: grouped upstream input does not match source trace"
                )
        result[output_length] = [
            original[index:index + batch_size]
            for index in range(0, len(original), batch_size)
        ]
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--engine-dir", type=Path, required=True)
    parser.add_argument("--multimodal-engine-dir", type=Path)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--source-trace", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--measurement-dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument("--materialize-inputs", action="store_true")
    parser.add_argument("--allow-partial-batches", action="store_true")
    args = parser.parse_args()
    if args.batch_size != 8:
        parser.error("this upstream comparison is fixed to full BS8")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trace = json.loads(args.source_trace.read_text(encoding="utf-8"))
    trace_requests = trace["requests"]
    output_lengths = sorted(
        {int(request["max_generate_length"])
         for request in trace_requests})
    if args.materialize_inputs:
        materialize_group_inputs(trace_requests, args.input_dir,
                                 output_lengths, args.batch_size)
    grouped_jobs = validate_group_inputs(trace_requests, args.input_dir,
                                         output_lengths, args.batch_size,
                                         args.allow_partial_batches)
    measurement_dir = args.measurement_dir if args.measurement_dir is not None else args.output_dir

    jobs = []
    profiles = {}
    for output_length in output_lengths:
        durations, profile = run_group(args, output_length, measurement_dir)
        profiles[output_length] = profile
        if len(durations) != len(grouped_jobs[output_length]):
            raise RuntimeError(
                f"output{output_length}: duration/job count mismatch")
        generation_steps = int(
            profile["stages"][0]["gpu_time_stats"]["count"]) / len(durations)
        for group_index, (members, duration) in enumerate(
                zip(grouped_jobs[output_length], durations)):
            jobs.append({
                "output_length":
                output_length,
                "group_index":
                group_index,
                "members":
                members,
                "release_ms":
                max(
                    float(request["arrival_offset_us"]) / 1000.0
                    for _, request in members),
                "duration_ms":
                duration,
                "prefill_gpu_ms":
                float(profile["prefill"]["average_time_per_run_ms"]),
                "generation_steps":
                generation_steps,
            })
        print(
            f"output{output_length}: {len(durations)} batches, wall={sum(durations):.1f} ms",
            flush=True)

    now_ms = 0.0
    scheduled = []
    pending = list(jobs)
    while pending:
        ready = [job for job in pending if job["release_ms"] <= now_ms]
        if not ready:
            now_ms = min(job["release_ms"] for job in pending)
            ready = [job for job in pending if job["release_ms"] <= now_ms]
        job = min(ready,
                  key=lambda item:
                  (item["duration_ms"], item["release_ms"], item[
                      "output_length"], item["group_index"]))
        pending.remove(job)
        job["start_ms"] = now_ms
        job["done_ms"] = now_ms + job["duration_ms"]
        now_ms = job["done_ms"]
        scheduled.append(job)

    request_rows = []
    for schedule_index, job in enumerate(scheduled):
        ttft_done_ms = job["start_ms"] + job["prefill_gpu_ms"]
        for request_id, request in job["members"]:
            arrival_ms = float(request["arrival_offset_us"]) / 1000.0
            estimated_tpot_ms = max(
                0.0, (job["duration_ms"] - job["prefill_gpu_ms"]) /
                job["generation_steps"])
            request_rows.append({
                "request_id": request_id,
                "schedule_index": schedule_index,
                "output_length": job["output_length"],
                "arrival_ms": arrival_ms,
                "batch_start_ms": job["start_ms"],
                "batch_done_ms": job["done_ms"],
                "estimated_ttft_ms": ttft_done_ms - arrival_ms,
                "estimated_tpot_wall_ms": estimated_tpot_ms,
                "e2e_ms": job["done_ms"] - arrival_ms,
            })
    request_rows.sort(key=lambda row: row["request_id"])
    generated_tokens = sum(
        int(profile["generation"]["generated_tokens"])
        for profile in profiles.values())
    ttft = [float(row["estimated_ttft_ms"]) for row in request_rows]
    tpot = [float(row["estimated_tpot_wall_ms"]) for row in request_rows]
    e2e = [float(row["e2e_ms"]) for row in request_rows]
    generation_steps = sum(
        int(profile["stages"][0]["gpu_time_stats"]["count"])
        for profile in profiles.values())
    generation_gpu_ms = sum(
        float(profile["stages"][0]["total_gpu_time_ms"])
        for profile in profiles.values())
    summary = {
        "requests":
        len(request_rows),
        "batches":
        len(scheduled),
        "full_bs8_batches":
        sum(len(job["members"]) == args.batch_size for job in scheduled),
        "partial_batches":
        sum(len(job["members"]) < args.batch_size for job in scheduled),
        "generated_tokens":
        generated_tokens,
        "makespan_ms":
        now_ms,
        "generated_token_s":
        generated_tokens * 1000.0 / now_ms,
        "ttft_estimated_gpu_median_ms":
        statistics.median(ttft),
        "ttft_estimated_gpu_p95_ms":
        percentile(ttft, 0.95),
        "tpot_estimated_wall_median_ms":
        statistics.median(tpot),
        "tpot_estimated_wall_p95_ms":
        percentile(tpot, 0.95),
        "e2e_wall_median_ms":
        statistics.median(e2e),
        "e2e_wall_p95_ms":
        percentile(e2e, 0.95),
        "decode_step_gpu_mean_ms":
        generation_gpu_ms / generation_steps,
        "peak_gpu_memory_mb":
        max(
            float(profile["peak_gpu_memory_mb"])
            for profile in profiles.values()),
        "scheduler":
        "online clairvoyant SPT over measured homogeneous batch wall costs",
    }
    with (args.output_dir / "requests.csv").open("w",
                                                 newline="",
                                                 encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(request_rows[0]))
        writer.writeheader()
        writer.writerows(request_rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
