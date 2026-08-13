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
"""Replay a TensorRT Edge-LLM request trace against a vLLM server.

The runner preserves each request's arrival offset, messages, and output-token
limit.  It uses streaming chat completions so TTFT is measured at the client,
and records the server-reported prompt and completion token counts.  No vLLM
Python dependency is required on the host.
"""

import argparse
import concurrent.futures
import csv
import hashlib
import http.client
import json
import math
import statistics
import threading
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


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


def wait_until(target_ns: int) -> None:
    while True:
        remaining_ns = target_ns - time.monotonic_ns()
        if remaining_ns <= 0:
            return
        time.sleep(min(remaining_ns / 1_000_000_000.0, 0.01))


def endpoint_parts(endpoint: str) -> tuple[str, int, str]:
    parsed = urllib.parse.urlparse(endpoint)
    if parsed.scheme != "http" or not parsed.hostname:
        raise ValueError("endpoint must be an http URL")
    port = parsed.port if parsed.port is not None else 80
    base_path = parsed.path.rstrip("/")
    return parsed.hostname, port, base_path


def read_endpoint(endpoint: str, path: str, timeout: float) -> str:
    url = endpoint.rstrip("/") + path
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


def stream_request(endpoint: str,
                   model: str,
                   request_id: int,
                   request: dict[str, Any],
                   epoch_ns: int,
                   start_gate: threading.Event,
                   timeout: float,
                   max_tokens_override: int = 0,
                   include_request_index: bool = False) -> dict[str, Any]:
    start_gate.wait()
    scheduled_arrival_us = int(request.get("arrival_offset_us", 0))
    wait_until(epoch_ns + scheduled_arrival_us * 1000)
    send_ns = time.monotonic_ns()
    max_tokens = max_tokens_override or int(request["max_generate_length"])
    payload = {
        "model": model,
        "messages": request["messages"],
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": max_tokens,
        "seed": 0,
        "stream": True,
        "stream_options": {
            "include_usage": True,
        },
    }
    if include_request_index:
        payload["metadata"] = {"request_index": request_id}
    host, port, base_path = endpoint_parts(endpoint)
    connection = http.client.HTTPConnection(host, port, timeout=timeout)
    first_token_ns = 0
    prompt_tokens = 0
    output_tokens = 0
    finish_reason = ""
    output_parts = []
    status = 0
    error = ""
    try:
        connection.request("POST",
                           base_path + "/v1/chat/completions",
                           body=json.dumps(payload),
                           headers={"Content-Type": "application/json"})
        response = connection.getresponse()
        status = response.status
        if status != 200:
            error = response.read().decode("utf-8", errors="replace")[:1000]
        else:
            while True:
                raw_line = response.readline()
                if not raw_line:
                    break
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                event = json.loads(data)
                usage = event.get("usage")
                if usage:
                    prompt_tokens = int(
                        usage.get("prompt_tokens") or prompt_tokens)
                    output_tokens = int(
                        usage.get("completion_tokens") or output_tokens)
                choices = event.get("choices") or []
                if not choices:
                    continue
                choice = choices[0]
                if choice.get("finish_reason") is not None:
                    finish_reason = str(choice["finish_reason"])
                delta = choice.get("delta") or {}
                token_text = delta.get("content") or delta.get(
                    "reasoning_content") or ""
                token_ids = delta.get("token_ids") or []
                if (token_text or token_ids) and first_token_ns == 0:
                    first_token_ns = time.monotonic_ns()
                if token_text:
                    output_parts.append(str(token_text))
    except (ConnectionError, OSError, TimeoutError,
            json.JSONDecodeError) as exception:
        error = f"{type(exception).__name__}: {exception}"
    finally:
        connection.close()
    done_ns = time.monotonic_ns()
    if first_token_ns == 0:
        first_token_ns = done_ns
    ttft_ms = (first_token_ns - send_ns) / 1_000_000.0
    e2e_ms = (done_ns - send_ns) / 1_000_000.0
    tpot_ms = ((done_ns - first_token_ns) / 1_000_000.0 /
               (output_tokens - 1) if output_tokens > 1 else 0.0)
    return {
        "request_id": request_id,
        "scheduled_arrival_us": scheduled_arrival_us,
        "send_us": round((send_ns - epoch_ns) / 1000),
        "first_token_us": round((first_token_ns - epoch_ns) / 1000),
        "completed_us": round((done_ns - epoch_ns) / 1000),
        "client_dispatch_delay_us": round(
            (send_ns - epoch_ns) / 1000) - scheduled_arrival_us,
        "max_output_tokens": max_tokens,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "http_status": status,
        "finish_reason": finish_reason,
        "ttft_ms": ttft_ms,
        "tpot_ms": tpot_ms,
        "e2e_ms": e2e_ms,
        "output_prefix": "".join(output_parts)[:160].replace("\n", "\\n"),
        "error": error,
    }


def execute_requests(
        endpoint: str,
        model: str,
        requests: list[dict[str, Any]],
        timeout: float,
        max_workers: int,
        max_tokens_override: int = 0,
        include_request_index: bool = False
) -> tuple[list[dict[str, Any]], float]:
    if max_workers < len(requests):
        raise ValueError(
            "max-workers must be at least the request count to preserve burst arrivals"
        )
    start_gate = threading.Event()
    epoch_ns = time.monotonic_ns() + 500_000_000
    with concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers) as executor:
        futures = [
            executor.submit(stream_request, endpoint, model, request_id,
                            request, epoch_ns, start_gate, timeout,
                            max_tokens_override, include_request_index)
            for request_id, request in enumerate(requests)
        ]
        start_gate.set()
        rows = [future.result() for future in futures]
    rows.sort(key=lambda row: int(row["request_id"]))
    terminal_us = max(float(row["completed_us"]) for row in rows)
    return rows, terminal_us / 1000.0


def summarize(rows: list[dict[str, Any]], duration_ms: float,
              run_index: int) -> dict[str, Any]:
    successful = [
        row for row in rows
        if int(row["http_status"]) == 200 and not row["error"]
    ]
    if len(successful) != len(rows):
        failures = [row for row in rows if row not in successful]
        raise RuntimeError(
            f"{len(failures)} requests failed; first failure: {failures[0]}")
    ttft = [float(row["ttft_ms"]) for row in successful]
    tpot = [
        float(row["tpot_ms"]) for row in successful
        if int(row["output_tokens"]) > 1
    ]
    e2e = [float(row["e2e_ms"]) for row in successful]
    dispatch_delay = [
        float(row["client_dispatch_delay_us"]) / 1000.0 for row in successful
    ]
    generated_tokens = sum(int(row["output_tokens"]) for row in successful)
    prompt_tokens = sum(int(row["prompt_tokens"]) for row in successful)
    requested_tokens = sum(int(row["max_output_tokens"]) for row in successful)
    return {
        "run": run_index,
        "requests": len(successful),
        "prompt_tokens": prompt_tokens,
        "requested_output_tokens": requested_tokens,
        "generated_tokens": generated_tokens,
        "duration_ms": duration_ms,
        "achieved_req_s": len(successful) * 1000.0 / duration_ms,
        "generated_token_s": generated_tokens * 1000.0 / duration_ms,
        "ttft_median_ms": statistics.median(ttft),
        "ttft_p95_ms": percentile(ttft, 0.95),
        "ttft_p99_ms": percentile(ttft, 0.99),
        "tpot_median_ms": statistics.median(tpot) if tpot else 0.0,
        "tpot_p95_ms": percentile(tpot, 0.95),
        "tpot_p99_ms": percentile(tpot, 0.99),
        "e2e_median_ms": statistics.median(e2e),
        "e2e_p95_ms": percentile(e2e, 0.95),
        "e2e_p99_ms": percentile(e2e, 0.99),
        "client_dispatch_delay_p95_ms": percentile(dispatch_delay, 0.95),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="nvidia/Cosmos-Reason2-2B")
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup-requests", type=int, default=64)
    parser.add_argument("--warmup-max-tokens", type=int, default=32)
    parser.add_argument("--max-workers", type=int, default=512)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--include-request-index", action="store_true")
    args = parser.parse_args()
    if args.repeats <= 0 or args.warmup_requests < 0:
        parser.error(
            "repeats must be positive and warmup-requests must be non-negative"
        )

    trace_bytes = args.trace.read_bytes()
    trace = json.loads(trace_bytes)
    requests = list(trace.get("requests", []))
    if not requests:
        raise RuntimeError(f"trace contains no requests: {args.trace}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "health.txt").write_text(read_endpoint(
        args.endpoint, "/health", args.timeout),
                                                encoding="utf-8")
    (args.output_dir / "version.json").write_text(read_endpoint(
        args.endpoint, "/version", args.timeout),
                                                  encoding="utf-8")

    if args.warmup_requests:
        warmup = [
            json.loads(json.dumps(requests[index % len(requests)]))
            for index in range(args.warmup_requests)
        ]
        for request in warmup:
            request["arrival_offset_us"] = 0
        print(
            f"warmup: {len(warmup)} requests x {args.warmup_max_tokens} max tokens",
            flush=True)
        execute_requests(args.endpoint, args.model, warmup, args.timeout,
                         args.max_workers, args.warmup_max_tokens)

    summaries = []
    for run_index in range(1, args.repeats + 1):
        run_dir = args.output_dir / f"run-{run_index:03d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "metrics-before.txt").write_text(read_endpoint(
            args.endpoint, "/metrics", args.timeout),
                                                    encoding="utf-8")
        rows, duration_ms = execute_requests(
            args.endpoint,
            args.model,
            requests,
            args.timeout,
            args.max_workers,
            include_request_index=args.include_request_index)
        summary = summarize(rows, duration_ms, run_index)
        write_csv(run_dir / "requests.csv", rows)
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) +
                                              "\n",
                                              encoding="utf-8")
        (run_dir / "metrics-after.txt").write_text(read_endpoint(
            args.endpoint, "/metrics", args.timeout),
                                                   encoding="utf-8")
        summaries.append(summary)
        print(json.dumps(summary), flush=True)

    write_csv(args.output_dir / "summary.csv", summaries)
    aggregate = {
        "created_at_utc":
        datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "trace":
        str(args.trace),
        "trace_sha256":
        hashlib.sha256(trace_bytes).hexdigest(),
        "endpoint":
        args.endpoint,
        "model":
        args.model,
        "repeats":
        args.repeats,
        "warmup_requests":
        args.warmup_requests,
        "warmup_max_tokens":
        args.warmup_max_tokens,
        "generated_token_s_median":
        statistics.median(
            float(row["generated_token_s"]) for row in summaries),
        "achieved_req_s_median":
        statistics.median(float(row["achieved_req_s"]) for row in summaries),
        "ttft_median_of_run_medians_ms":
        statistics.median(float(row["ttft_median_ms"]) for row in summaries),
        "ttft_p95_median_ms":
        statistics.median(float(row["ttft_p95_ms"]) for row in summaries),
        "tpot_median_of_run_medians_ms":
        statistics.median(float(row["tpot_median_ms"]) for row in summaries),
        "tpot_p95_median_ms":
        statistics.median(float(row["tpot_p95_ms"]) for row in summaries),
        "e2e_median_of_run_medians_ms":
        statistics.median(float(row["e2e_median_ms"]) for row in summaries),
        "e2e_p95_median_ms":
        statistics.median(float(row["e2e_p95_ms"]) for row in summaries),
    }
    (args.output_dir / "aggregate.json").write_text(
        json.dumps(aggregate, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
