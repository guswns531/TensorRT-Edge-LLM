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
"""Replay a timed chat trace against any OpenAI-compatible server and report serving metrics.

The same client drives the phase gateway and vLLM, so both see identical payloads, arrival
offsets, in-flight ceiling and fixed output lengths (``ignore_eos``). Metric definitions:
TTFT = first non-empty content chunk - send; TPOT = (done - first token) / (output_tokens - 1);
generated tok/s = total output tokens / (last done - first send).
"""

import argparse
import asyncio
import csv
import hashlib
import json
import pathlib
import statistics
import time

import httpx


def percentile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    rank = (len(ordered) - 1) * q / 100.0
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def load_trace(path):
    data = json.loads(pathlib.Path(path).read_text())
    return data["requests"] if isinstance(data, dict) else data


async def run_one(client, url, model, index, request, ignore_eos, origin_ns,
                  delay_s, gate):
    await asyncio.sleep(
        max(0.0, delay_s - (time.perf_counter_ns() - origin_ns) / 1e9))
    async with gate:
        body = {
            "model": model,
            "messages": request["messages"],
            "max_tokens": request["max_generate_length"],
            "temperature": 0.0,
            "stream": True,
            "stream_options": {
                "include_usage": True
            },
        }
        if ignore_eos:
            body["ignore_eos"] = True
            body["min_tokens"] = request["max_generate_length"]
        record = {
            "index": index,
            "requested_tokens": request["max_generate_length"],
            "error": ""
        }
        send = time.perf_counter_ns()
        first = None
        chunks = 0
        usage = None
        text = []
        try:
            async with client.stream("POST", url, json=body) as response:
                if response.status_code != 200:
                    record["error"] = "HTTP %d: %s" % (response.status_code, (
                        await response.aread())[:200])
                else:
                    async for line in response.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        payload = line[6:]
                        if payload == "[DONE]":
                            break
                        event = json.loads(payload)
                        if "error" in event:
                            record["error"] = str(event["error"])[:200]
                            break
                        if event.get("usage"):
                            usage = event["usage"]
                        for choice in event.get("choices", []):
                            content = (choice.get("delta")
                                       or {}).get("content")
                            if content:
                                if first is None:
                                    first = time.perf_counter_ns()
                                chunks += 1
                                text.append(content)
        except Exception as error:  # noqa: BLE001 - every failure is recorded per request
            record["error"] = repr(error)[:200]
        done = time.perf_counter_ns()
        output = int(usage["completion_tokens"]) if usage else chunks
        record.update({
            "send_us": (send - origin_ns) / 1e3,
            "done_us": (done - origin_ns) / 1e3,
            "prompt_tokens":
            int(usage["prompt_tokens"]) if usage else -1,
            "output_tokens":
            output,
            "ttft_ms": (first - send) / 1e6 if first else None,
            "e2e_ms": (done - send) / 1e6,
            "tpot_ms": ((done - first) / 1e6 /
                        (output - 1)) if first and output > 1 else None,
            "text_sha256":
            hashlib.sha256("".join(text).encode()).hexdigest(),
        })
        return record


async def replay(url,
                 model,
                 requests,
                 max_in_flight,
                 ignore_eos,
                 timeout,
                 use_arrivals=True):
    gate = asyncio.Semaphore(max_in_flight)
    limits = httpx.Limits(max_connections=max_in_flight + 8,
                          max_keepalive_connections=max_in_flight + 8)
    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout),
                                 limits=limits) as client:
        origin = time.perf_counter_ns()
        tasks = [
            run_one(
                client, url, model, index, request, ignore_eos, origin,
                request.get("arrival_offset_us", 0) /
                1e6 if use_arrivals else 0.0, gate)
            for index, request in enumerate(requests)
        ]
        return await asyncio.gather(*tasks)


def summarize(records):
    ok = [r for r in records if not r["error"]]
    summary = {
        "requests": len(records),
        "succeeded": len(ok),
        "failed": len(records) - len(ok)
    }
    if not ok:
        return summary
    span_s = (max(r["done_us"] for r in ok) - min(r["send_us"]
                                                  for r in ok)) / 1e6
    output_tokens = sum(r["output_tokens"] for r in ok)
    summary.update({
        "duration_s":
        span_s,
        "output_tokens":
        output_tokens,
        "requested_output_tokens":
        sum(r["requested_tokens"] for r in records),
        "fixed_output_complete":
        all(r["output_tokens"] == r["requested_tokens"] for r in ok),
        "generated_token_s":
        output_tokens / span_s if span_s > 0 else None,
        "request_s":
        len(ok) / span_s if span_s > 0 else None,
    })
    for name in ("ttft_ms", "tpot_ms", "e2e_ms"):
        values = [r[name] for r in ok if r[name] is not None]
        if values:
            summary[name] = {
                "mean": statistics.fmean(values),
                "p50": percentile(values, 50),
                "p95": percentile(values, 95),
                "p99": percentile(values, 99),
            }
    return summary


def control(endpoint, action, timeout):
    response = httpx.post(endpoint + "/phase/control/" + action,
                          json={},
                          timeout=timeout)
    response.raise_for_status()
    return response.json()


def wait_ready(endpoint, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            response = httpx.get(endpoint + "/health", timeout=5)
            if response.status_code == 200:
                return
            if "error" in response.text:
                raise RuntimeError("server failed: " + response.text)
        except httpx.HTTPError:
            pass
        time.sleep(2)
    raise TimeoutError("server not ready: " + endpoint)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8001")
    parser.add_argument("--model", required=True)
    parser.add_argument("--trace", required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--max-in-flight", type=int, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup-trace")
    parser.add_argument("--warmup-requests", type=int, default=0)
    parser.add_argument(
        "--phase-calibration",
        action="store_true",
        help=
        "Bracket warmup with phase calibration_begin/end (phase gateway only)")
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--ready-timeout", type=float, default=1800.0)
    parser.add_argument("--request-timeout", type=float, default=900.0)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    url = args.endpoint + "/v1/chat/completions"
    wait_ready(args.endpoint, args.ready_timeout)
    requests = load_trace(args.trace)
    result = {
        "trace":
        args.trace,
        "trace_sha256":
        hashlib.sha256(pathlib.Path(args.trace).read_bytes()).hexdigest(),
        "max_in_flight":
        args.max_in_flight,
        "runs": []
    }

    if args.warmup_requests > 0:
        warmup = load_trace(args.warmup_trace or args.trace)
        warmup = (
            warmup *
            (args.warmup_requests // len(warmup) + 1))[:args.warmup_requests]
        if args.phase_calibration:
            result["calibration_begin"] = control(args.endpoint, "begin",
                                                  args.request_timeout)
        records = asyncio.run(
            replay(url, args.model, warmup, args.max_in_flight,
                   args.ignore_eos, args.request_timeout, False))
        result["warmup"] = summarize(records)
        if args.phase_calibration:
            result["calibration_end"] = control(args.endpoint, "end",
                                                args.request_timeout)

    for repeat in range(args.repeats):
        records = asyncio.run(
            replay(url, args.model, requests, args.max_in_flight,
                   args.ignore_eos, args.request_timeout))
        summary = summarize(records)
        summary["token_text_sha256"] = hashlib.sha256("".join(
            r["text_sha256"] for r in records).encode()).hexdigest()
        result["runs"].append(summary)
        with (args.output_dir / ("requests-%03d.csv" % repeat)).open(
                "w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)
        print(json.dumps({
            "repeat": repeat,
            **{
                k: summary.get(k)
                for k in ("succeeded", "failed", "generated_token_s", "fixed_output_complete")
            }
        }),
              flush=True)

    rates = [
        run.get("generated_token_s") for run in result["runs"]
        if run.get("generated_token_s")
    ]
    result["generated_token_s_median"] = statistics.median(
        rates) if rates else None
    (args.output_dir /
     "summary.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
