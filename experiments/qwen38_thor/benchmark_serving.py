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
"""Run deterministic OpenAI streaming workloads against one serving backend."""

import argparse
import concurrent.futures
import http.client
import json
import math
import random
import statistics
import time
import urllib.parse
from pathlib import Path
from typing import Any

import tokenizers

PROMPT_PATTERN = (
    "Inference servers process incoming prompts, schedule prefill and decode "
    "work, reuse cached state, and form GPU batches. The scheduler balances "
    "latency, throughput, fairness, memory pressure, and queue depth. ")
PROMPT_INSTRUCTION = (
    "\n\nExplain how an inference server balances latency and throughput. "
    "Use complete technical sentences and continue until the token budget is exhausted."
)


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position -
                                                                 lower)


def text_with_tokens(tokenizer: tokenizers.Tokenizer,
                     target_tokens: int,
                     tail: str = "") -> str:
    if target_tokens <= 0:
        return ""
    tail_tokens = tokenizer.encode(tail).ids if tail else []
    body_target = max(0, target_tokens - len(tail_tokens))
    repeated = PROMPT_PATTERN * (target_tokens // 24 + 4)
    token_ids = tokenizer.encode(repeated).ids[:body_target]
    text = tokenizer.decode(token_ids) + tail
    while len(tokenizer.encode(text).ids) < target_tokens:
        text += " x"
    return text


def make_messages(content: str, turns: int) -> list[dict[str, str]]:
    if turns <= 1:
        return [{"role": "user", "content": content}]
    segment = max(1, len(content) // turns)
    messages: list[dict[str, str]] = []
    for turn in range(turns):
        start = turn * segment
        end = len(content) if turn + 1 == turns else (turn + 1) * segment
        messages.append({"role": "user", "content": content[start:end]})
        if turn + 1 < turns:
            messages.append({"role": "assistant", "content": "Acknowledged."})
    return messages


def arrival_offsets(case: dict[str, Any], count: int,
                    seed: int) -> list[float]:
    arrival = case.get("arrival", "burst")
    if arrival == "burst":
        return [0.0] * count
    if arrival == "waves":
        wave_size = int(case["wave_size"])
        interval = float(case["wave_interval_seconds"])
        return [(index // wave_size) * interval for index in range(count)]
    rate = float(case["request_rate"])
    if arrival == "constant":
        return [index / rate for index in range(count)]
    if arrival == "poisson":
        generator = random.Random(seed)
        offsets: list[float] = []
        current = 0.0
        for _ in range(count):
            offsets.append(current)
            current += generator.expovariate(rate)
        return offsets
    raise ValueError(f"unsupported arrival mode: {arrival}")


def request_profile(case: dict[str, Any], index: int) -> tuple[int, int]:
    profiles = case.get("profiles")
    profile = profiles[index % len(profiles)] if profiles else case
    return int(profile["input_tokens"]), int(profile["output_tokens"])


def stream_request(base_url: str, model: str, request_index: int,
                   messages: list[dict[str, str]], output_tokens: int,
                   timeout: float, stream: bool) -> dict[str, Any]:
    parsed = urllib.parse.urlparse(base_url)
    connection = http.client.HTTPConnection(parsed.hostname,
                                            parsed.port,
                                            timeout=timeout)
    body = json.dumps({
        "model": model,
        "messages": messages,
        "max_tokens": output_tokens,
        "temperature": 0.0,
        "ignore_eos": True,
        "stream": stream,
        "chat_template_kwargs": {
            "enable_thinking": False
        },
        "metadata": {
            "request_index": request_index
        },
    })
    if stream:
        body_payload = json.loads(body)
        body_payload["stream_options"] = {"include_usage": True}
        body = json.dumps(body_payload)
    started = time.perf_counter()
    first_token_at: float | None = None
    token_times: list[float] = []
    usage: dict[str, int] = {}
    content_parts: list[str] = []
    status = 0
    error = ""
    try:
        connection.request("POST", "/v1/chat/completions", body,
                           {"Content-Type": "application/json"})
        response = connection.getresponse()
        status = response.status
        if status != 200:
            error = response.read().decode(errors="replace")
        elif not stream:
            event = json.loads(response.read())
            usage = event.get("usage") or {}
            choices = event.get("choices") or []
            if choices:
                message = choices[0].get("message") or {}
                content = message.get("content") or ""
                content_parts.append(content)
        else:
            while True:
                line = response.readline()
                if not line:
                    break
                decoded = line.decode(errors="replace").strip()
                if not decoded.startswith("data:"):
                    continue
                payload = decoded[5:].strip()
                if payload == "[DONE]":
                    break
                event = json.loads(payload)
                if "error" in event:
                    error = str(event["error"])
                    break
                if event.get("usage"):
                    usage = event["usage"]
                choices = event.get("choices") or []
                if not choices:
                    continue
                content = (choices[0].get("delta") or {}).get("content")
                if content:
                    now = time.perf_counter()
                    first_token_at = first_token_at or now
                    token_times.append(now)
                    content_parts.append(content)
    except Exception as exception:  # noqa: BLE001
        error = repr(exception)
    finally:
        connection.close()
    completed = time.perf_counter()
    event_tokens = len(token_times)
    completion_tokens = int(usage.get("completion_tokens", event_tokens))
    prompt_tokens = int(usage.get("prompt_tokens", 0))
    inter_token_ms = [(right - left) * 1000.0
                      for left, right in zip(token_times, token_times[1:])]
    return {
        "request_index":
        request_index,
        "status":
        status,
        "error":
        error,
        "success":
        status == 200 and not error and completion_tokens > 0,
        "prompt_tokens":
        prompt_tokens,
        "completion_tokens":
        completion_tokens,
        "ttft_ms": ((first_token_at - started) *
                    1000.0 if first_token_at is not None else 0.0),
        "e2e_ms": (completed - started) * 1000.0,
        "tpot_ms":
        (statistics.mean(inter_token_ms) if inter_token_ms else 0.0),
        "itl_ms":
        inter_token_ms,
        "text":
        "".join(content_parts),
    }


def summarize(records: list[dict[str, Any]],
              duration: float) -> dict[str, Any]:
    successful = [record for record in records if record["success"]]
    ttft = [record["ttft_ms"] for record in successful]
    tpot = [
        record["tpot_ms"] for record in successful if record["tpot_ms"] > 0
    ]
    e2e = [record["e2e_ms"] for record in successful]
    itl = [value for record in successful for value in record["itl_ms"]]
    output_tokens = sum(record["completion_tokens"] for record in successful)

    def distribution(values: list[float]) -> dict[str, float]:
        return {
            "median": percentile(values, 0.50),
            "p95": percentile(values, 0.95),
            "p99": percentile(values, 0.99),
        }

    return {
        "successful_requests": len(successful),
        "failed_requests": len(records) - len(successful),
        "duration_seconds": duration,
        "output_tokens": output_tokens,
        "request_throughput": len(successful) / duration if duration else 0.0,
        "output_token_throughput":
        output_tokens / duration if duration else 0.0,
        "ttft_ms": distribution(ttft),
        "tpot_ms": distribution(tpot),
        "itl_ms": distribution(itl),
        "e2e_ms": distribution(e2e),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", default="qwen38")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--mode",
                        choices=("stream", "nonstream"),
                        default="stream")
    args = parser.parse_args()

    config = json.loads(args.config.read_text())
    case = config["cases"][args.case]
    request_count = int(case["requests"])
    overhead = int(config["chat_template_overhead_tokens"])
    tokenizer = tokenizers.Tokenizer.from_file(str(args.tokenizer))
    shared_prefix_tokens = int(case.get("shared_prefix_tokens", 0))
    shared_prefix = text_with_tokens(tokenizer, shared_prefix_tokens)
    requests: list[tuple[list[dict[str, str]], int]] = []
    for index in range(request_count):
        input_tokens, output_tokens = request_profile(case, index)
        content_tokens = max(1, input_tokens - overhead)
        suffix_tokens = max(1, content_tokens - shared_prefix_tokens)
        request_tail = PROMPT_INSTRUCTION + f" Request number: {index}."
        suffix = text_with_tokens(tokenizer, suffix_tokens, request_tail)
        content = shared_prefix + suffix
        requests.append(
            (make_messages(content, int(case.get("turns", 1))), output_tokens))

    offsets = arrival_offsets(case, request_count, args.seed)
    benchmark_started = time.perf_counter()
    futures: list[concurrent.futures.Future[dict[str, Any]]] = []
    with concurrent.futures.ThreadPoolExecutor(
            max_workers=int(case["concurrency"])) as executor:
        for index, ((messages, output_tokens),
                    offset) in enumerate(zip(requests, offsets)):
            delay = benchmark_started + offset - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            futures.append(
                executor.submit(stream_request, args.base_url, args.model,
                                index, messages, output_tokens, args.timeout,
                                args.mode == "stream"))
        records = [future.result() for future in futures]
    duration = time.perf_counter() - benchmark_started
    result = {
        "schema_version": 1,
        "case": args.case,
        "backend": args.base_url,
        "model": args.model,
        "seed": args.seed,
        "mode": args.mode,
        "config": case,
        "summary": summarize(records, duration),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
