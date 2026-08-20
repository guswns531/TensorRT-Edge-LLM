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
"""Repeat one real-request trace through the phase OpenAI HTTP/SSE gateway."""

import argparse
import csv
import json
import pathlib
import signal
import statistics
import subprocess
import sys
import threading
import time
import typing
import urllib.error
import urllib.request


def read_gpu_memory_mib() -> int:
    """Return used memory for the single serving GPU."""
    output = subprocess.check_output([
        "nvidia-smi", "--query-gpu=memory.used",
        "--format=csv,noheader,nounits"
    ],
                                     text=True)
    values = [
        int(line.strip()) for line in output.splitlines() if line.strip()
    ]
    if len(values) != 1:
        raise RuntimeError("phase gate requires exactly one visible GPU")
    return values[0]


class GpuMemoryMonitor:
    """Sample device memory while a real-request trace is active."""

    def __init__(self, interval_seconds: float = 0.1) -> None:
        self.interval_seconds = interval_seconds
        self.samples: list[int] = []
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self.stop_event.is_set():
            self.samples.append(read_gpu_memory_mib())
            self.stop_event.wait(self.interval_seconds)

    def start(self) -> None:
        self.samples.append(read_gpu_memory_mib())
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join()

    @property
    def peak_mib(self) -> int:
        return max(self.samples)


def wait_for_health(endpoint: str, process: subprocess.Popen[str],
                    timeout: float) -> None:
    """Wait for a gateway and its TensorRT backend to become ready."""
    deadline = time.monotonic() + timeout
    health_url = endpoint.rstrip("/") + "/health"
    while time.monotonic() < deadline:
        return_code = process.poll()
        if return_code is not None:
            raise RuntimeError(
                f"phase gateway exited before ready with code {return_code}")
        try:
            with urllib.request.urlopen(health_url, timeout=1.0) as response:
                if response.status == 200:
                    return
        except urllib.error.HTTPError as exception:
            if exception.code != 503:
                details = exception.read().decode("utf-8", errors="replace")
                raise RuntimeError(details.strip()) from exception
        except (urllib.error.URLError, TimeoutError):
            pass
        time.sleep(0.25)
    raise TimeoutError(f"phase gateway did not become ready within {timeout}s")


def stop_gateway(process: subprocess.Popen[str],
                 timeout: float = 45.0) -> None:
    """Stop the gateway and let it retire the child GPU container."""
    if process.poll() is not None:
        return
    process.send_signal(signal.SIGINT)
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10.0)


def replace_run(command: list[str], run_index: int) -> list[str]:
    """Expand the run placeholder in a backend command."""
    return [value.replace("{run}", f"{run_index:03d}") for value in command]


def write_csv(path: pathlib.Path, rows: list[dict[str, typing.Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gateway-script", type=pathlib.Path, required=True)
    parser.add_argument("--client-script", type=pathlib.Path, required=True)
    parser.add_argument("--trace", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8001")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--model", default="nvidia/Cosmos-Reason2-2B")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--ready-timeout", type=float, default=120.0)
    parser.add_argument("--request-timeout", type=float, default=600.0)
    parser.add_argument("--max-workers", type=int, default=512)
    parser.add_argument("--sequential", action="store_true")
    parser.add_argument("--request-limit", type=int, default=0)
    parser.add_argument("backend_command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    backend_command = args.backend_command
    if backend_command and backend_command[0] == "--":
        backend_command = backend_command[1:]
    if not backend_command:
        parser.error("backend command is required after --")
    if args.repeats <= 0 or args.port <= 0 or args.request_limit < 0:
        parser.error(
            "repeats and port must be positive; request-limit must be non-negative"
        )
    for path in (args.gateway_script, args.client_script, args.trace):
        if not path.is_file():
            parser.error(f"required file does not exist: {path}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, typing.Any]] = []
    for run_index in range(1, args.repeats + 1):
        run_dir = args.output_dir / f"run-{run_index:03d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        gateway_log = run_dir / "gateway.log"
        gateway_command = [
            sys.executable,
            "-u",
            str(args.gateway_script),
            "--host",
            args.host,
            "--port",
            str(args.port),
            "--model",
            args.model,
            "--timeout",
            str(args.request_timeout),
            "--",
            *replace_run(backend_command, run_index),
        ]
        with gateway_log.open("w", encoding="utf-8") as log:
            gateway = subprocess.Popen(gateway_command,
                                       stdout=log,
                                       stderr=subprocess.STDOUT,
                                       text=True)
            try:
                wait_for_health(args.endpoint, gateway, args.ready_timeout)
                ready_memory_mib = read_gpu_memory_mib()
                client_command = [
                    sys.executable,
                    str(args.client_script),
                    "--endpoint",
                    args.endpoint,
                    "--model",
                    args.model,
                    "--trace",
                    str(args.trace),
                    "--output-dir",
                    str(run_dir / "client"),
                    "--repeats",
                    "1",
                    "--warmup-requests",
                    "0",
                    "--max-workers",
                    str(args.max_workers),
                    "--timeout",
                    str(args.request_timeout),
                    "--include-request-index",
                ]
                if args.sequential:
                    client_command.append("--sequential")
                if args.request_limit > 0:
                    client_command.extend(
                        ["--request-limit",
                         str(args.request_limit)])
                memory_monitor = GpuMemoryMonitor()
                memory_monitor.start()
                try:
                    subprocess.run(client_command, check=True)
                finally:
                    memory_monitor.stop()
                after_memory_mib = read_gpu_memory_mib()
            finally:
                stop_gateway(gateway)

        aggregate_path = run_dir / "client" / "aggregate.json"
        if not aggregate_path.is_file():
            raise RuntimeError(
                f"client aggregate is missing: {aggregate_path}")
        summary = json.loads(aggregate_path.read_text(encoding="utf-8"))
        client_summary_path = run_dir / "client" / "summary.csv"
        with client_summary_path.open(encoding="utf-8") as stream:
            client_summary = next(csv.DictReader(stream))
        for key in ("requests", "prompt_tokens", "requested_output_tokens",
                    "generated_tokens", "duration_ms"):
            summary[key] = float(client_summary[key])
        summary["run"] = run_index
        summary["gateway_log"] = str(gateway_log)
        summary["gpu_memory_ready_mib"] = ready_memory_mib
        summary["gpu_memory_peak_mib"] = memory_monitor.peak_mib
        summary["gpu_memory_after_mib"] = after_memory_mib
        summaries.append(summary)

    metric_keys = [
        "generated_token_s_median",
        "achieved_req_s_median",
        "ttft_median_of_run_medians_ms",
        "ttft_p95_median_ms",
        "tpot_median_of_run_medians_ms",
        "tpot_p95_median_ms",
        "e2e_median_of_run_medians_ms",
        "e2e_p95_median_ms",
    ]
    aggregate: dict[str, typing.Any] = {
        "trace":
        str(args.trace),
        "trace_sha256":
        summaries[0]["trace_sha256"],
        "model":
        args.model,
        "endpoint":
        args.endpoint,
        "repeats":
        args.repeats,
        "requests_per_run":
        int(summaries[0]["requests"]),
        "prompt_tokens_per_run":
        int(summaries[0]["prompt_tokens"]),
        "requested_output_tokens_per_run":
        int(summaries[0]["requested_output_tokens"]),
        "generated_tokens_per_run_min":
        int(min(summary["generated_tokens"] for summary in summaries)),
        "generated_tokens_per_run_median":
        int(
            statistics.median(summary["generated_tokens"]
                              for summary in summaries)),
        "generated_tokens_per_run_max":
        int(max(summary["generated_tokens"] for summary in summaries)),
        "captured_token_ids_per_run": [
            int(summary["captured_token_ids_per_run"][0])
            for summary in summaries
        ],
        "token_trace_sha256_per_run": [
            str(summary["token_trace_sha256_per_run"][0])
            for summary in summaries
        ],
        "token_trace_deterministic":
        len({
            str(summary["token_trace_sha256_per_run"][0])
            for summary in summaries
        }) == 1,
        "gpu_memory_ready_mib_median":
        statistics.median(
            int(summary["gpu_memory_ready_mib"]) for summary in summaries),
        "gpu_memory_peak_mib_median":
        statistics.median(
            int(summary["gpu_memory_peak_mib"]) for summary in summaries),
        "gpu_memory_after_mib_median":
        statistics.median(
            int(summary["gpu_memory_after_mib"]) for summary in summaries),
    }
    for key in metric_keys:
        aggregate[key] = statistics.median(
            float(summary[key]) for summary in summaries)
    (args.output_dir / "aggregate.json").write_text(
        json.dumps(aggregate, indent=2) + "\n", encoding="utf-8")
    write_csv(args.output_dir / "runs.csv", summaries)
    print(json.dumps(aggregate), flush=True)


if __name__ == "__main__":
    main()
