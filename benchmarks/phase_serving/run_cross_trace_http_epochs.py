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
"""Run named HTTP trace epochs without restarting the phase-serving process."""

from __future__ import annotations

import argparse
import json
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def parse_epoch(spec: str) -> tuple[str, Path]:
    """Parse one stable NAME=TRACE epoch specification."""
    name, separator, raw_path = spec.partition("=")
    if not separator or not name or not raw_path:
        raise ValueError("--epoch must use NAME=TRACE")
    return name, Path(raw_path)


def replace_placeholders(command: list[str],
                         policy_warmup_mode: str) -> list[str]:
    """Resolve backend placeholders once for the persistent process."""
    return [
        value.replace("{run}", "001").replace("{policy_warmup_mode}",
                                              policy_warmup_mode)
        for value in command
    ]


def read_endpoint(endpoint: str, path: str, timeout: float) -> str:
    """Read one gateway endpoint."""
    with urllib.request.urlopen(endpoint.rstrip("/") + path,
                                timeout=timeout) as response:
        return response.read().decode("utf-8")


def wait_for_health(endpoint: str, process: subprocess.Popen[str],
                    timeout: float) -> None:
    """Wait for the persistent gateway and backend to become ready."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        return_code = process.poll()
        if return_code is not None:
            raise RuntimeError(
                f"phase gateway exited before ready with code {return_code}")
        try:
            if read_endpoint(endpoint, "/health", 1.0).strip() == "ok":
                return
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
            pass
        time.sleep(0.25)
    raise TimeoutError(f"gateway did not become ready within {timeout}s")


def reset_measurement_epoch(endpoint: str, timeout: float) -> dict[str, Any]:
    """Reset queue telemetry while preserving process-local policy evidence."""
    results = {}
    for action in ("begin", "end"):
        request = urllib.request.Request(
            endpoint.rstrip("/") + "/control/calibration",
            data=json.dumps({
                "action": action
            }).encode(),
            headers={"Content-Type": "application/json"},
            method="POST")
        with urllib.request.urlopen(request, timeout=timeout) as response:
            results[action] = json.loads(response.read().decode("utf-8"))
    return results


def stop_gateway(process: subprocess.Popen[str]) -> None:
    """Retire the gateway and its persistent GPU child."""
    if process.poll() is not None:
        return
    process.send_signal(signal.SIGINT)
    try:
        process.wait(timeout=45.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10.0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gateway-script", type=Path, required=True)
    parser.add_argument("--client-script", type=Path, required=True)
    parser.add_argument("--epoch", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8001")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--model", default="nvidia/Cosmos-Reason2-2B")
    parser.add_argument("--ready-timeout", type=float, default=180.0)
    parser.add_argument("--request-timeout", type=float, default=600.0)
    parser.add_argument("--max-workers", type=int, default=512)
    parser.add_argument("--max-in-flight", type=int, default=64)
    parser.add_argument("--warmup-requests", type=int, default=1696)
    parser.add_argument("--warmup-max-tokens", type=int, default=128)
    parser.add_argument("--warmup-trace", type=Path, required=True)
    parser.add_argument("--phase-calibration-round-requests",
                        type=int,
                        default=424)
    parser.add_argument("--phase-calibration-min-requests",
                        type=int,
                        default=424)
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("backend_command", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    try:
        epochs = [parse_epoch(spec) for spec in args.epoch]
    except ValueError as error:
        parser.error(str(error))
    backend_command = args.backend_command
    if backend_command and backend_command[0] == "--":
        backend_command = backend_command[1:]
    if not backend_command:
        parser.error("backend command is required after --")
    if (args.port <= 0 or args.max_workers <= 0 or args.max_in_flight <= 0
            or args.warmup_requests <= 0 or args.warmup_max_tokens <= 0
            or args.phase_calibration_round_requests <= 0
            or args.phase_calibration_min_requests < 0):
        parser.error("port, worker limits, and warmup sizes must be positive")
    required_paths = [
        args.gateway_script, args.client_script, args.warmup_trace
    ] + [path for _, path in epochs]
    for path in required_paths:
        if not path.is_file():
            parser.error(f"required file does not exist: {path}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    gateway_log = args.output_dir / "gateway.log"
    gateway_command = [
        sys.executable, "-u",
        str(args.gateway_script), "--host", args.host, "--port",
        str(args.port), "--model", args.model, "--timeout",
        str(args.request_timeout), "--",
        *replace_placeholders(backend_command, "generic")
    ]
    epoch_results = []
    with gateway_log.open("w", encoding="utf-8") as log:
        gateway = subprocess.Popen(gateway_command,
                                   stdout=log,
                                   stderr=subprocess.STDOUT,
                                   text=True)
        try:
            wait_for_health(args.endpoint, gateway, args.ready_timeout)
            for index, (name, trace) in enumerate(epochs):
                epoch_dir = args.output_dir / f"epoch-{index + 1:02d}-{name}"
                command = [
                    sys.executable,
                    str(args.client_script),
                    "--endpoint",
                    args.endpoint,
                    "--model",
                    args.model,
                    "--trace",
                    str(trace),
                    "--output-dir",
                    str(epoch_dir),
                    "--repeats",
                    "1",
                    "--warmup-requests",
                    str(args.warmup_requests if index == 0 else 0),
                    "--warmup-max-tokens",
                    str(args.warmup_max_tokens),
                    "--phase-calibration-round-requests",
                    str(args.phase_calibration_round_requests),
                    "--phase-calibration-min-requests",
                    str(args.phase_calibration_min_requests),
                    "--max-workers",
                    str(args.max_workers),
                    "--max-in-flight",
                    str(args.max_in_flight),
                    "--timeout",
                    str(args.request_timeout),
                    "--include-request-index",
                ]
                if index == 0:
                    command.extend([
                        "--warmup-trace",
                        str(args.warmup_trace), "--warmup-preserve-arrivals",
                        "--phase-calibration-control"
                    ])
                else:
                    boundary = reset_measurement_epoch(args.endpoint,
                                                       args.request_timeout)
                    (epoch_dir / "boundary.json").parent.mkdir(parents=True,
                                                               exist_ok=True)
                    (epoch_dir / "boundary.json").write_text(
                        json.dumps(boundary, indent=2) + "\n",
                        encoding="utf-8")
                if args.ignore_eos:
                    command.append("--ignore-eos")
                subprocess.run(command, check=True)
                aggregate_path = epoch_dir / "aggregate.json"
                aggregate = json.loads(
                    aggregate_path.read_text(encoding="utf-8"))
                epoch_results.append({
                    "index":
                    index + 1,
                    "name":
                    name,
                    "trace":
                    str(trace),
                    "aggregate":
                    str(aggregate_path),
                    "throughput_req_s":
                    aggregate["achieved_req_s_median"],
                    "throughput_token_s":
                    aggregate["generated_token_s_median"],
                    "ttft_p95_ms":
                    aggregate["ttft_p95_median_ms"],
                    "tpot_p95_ms":
                    aggregate["tpot_p95_median_ms"],
                    "e2e_p95_ms":
                    aggregate["e2e_p95_median_ms"],
                })
        finally:
            stop_gateway(gateway)

    result = {
        "schema_version": 1,
        "process_restarts": 0,
        "policy_warmup_mode": "generic_first_epoch_only",
        "warmup_requests": args.warmup_requests,
        "warmup_trace": str(args.warmup_trace),
        "epochs": epoch_results,
        "gateway_log": str(gateway_log),
    }
    (args.output_dir / "cross-trace.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
