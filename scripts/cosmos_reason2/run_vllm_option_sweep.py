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
"""Run reproducible vLLM option sweeps on the Cosmos request traces.

Every configuration gets a fresh server process. Startup and CUDA graph
capture are intentionally excluded from request latency, while the model and
compilation caches are shared across configurations.
"""

import argparse
import json
import subprocess
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

IMAGE = ("vllm/vllm-openai@sha256:"
         "c2f3b1b964e47809b722b5e75b61b1e7b39a50f70388cf2bf2418f16a9f31da2")
MODEL_NAME = "nvidia/Cosmos-Reason2-2B"
MATCHED_KV_BYTES = 3_758_096_384


def run(command: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command,
                          check=check,
                          text=True,
                          stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT)


def remove_container(container_name: str) -> None:
    run(["docker", "rm", "--force", container_name], check=False)


def wait_for_server(endpoint: str, container_name: str, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    last_error = "server did not answer"
    while time.monotonic() < deadline:
        inspection = run([
            "docker", "inspect", "--format", "{{.State.Status}} {{.State.ExitCode}}",
            container_name
        ], check=False)
        if inspection.returncode != 0 or inspection.stdout.startswith("exited"):
            raise RuntimeError(f"server exited during startup: {inspection.stdout.strip()}")
        try:
            with urllib.request.urlopen(endpoint + "/health", timeout=2.0) as response:
                if response.status == 200:
                    return
        except Exception as exception:  # noqa: BLE001 - retain the final startup error.
            last_error = f"{type(exception).__name__}: {exception}"
        time.sleep(1.0)
    raise TimeoutError(f"server startup timed out after {timeout_s}s: {last_error}")


def read_gpu_memory() -> dict[str, int]:
    result = run([
        "nvidia-smi", "--query-gpu=memory.used,memory.free", "--format=csv,noheader,nounits"
    ])
    used, free = [int(value.strip()) for value in result.stdout.strip().split(",")]
    return {"used_mib": used, "free_mib": free}


def configuration_args(name: str, tuning_token_budget: int) -> list[str]:
    if name.startswith("budget-"):
        return ["--max-num-batched-tokens", name.removeprefix("budget-")]

    arguments = ["--max-num-batched-tokens", str(tuning_token_budget)]
    variants = {
        "auto-tuned": [],
        "attention-flash-attn": ["--attention-backend", "FLASH_ATTN"],
        "attention-flashinfer": ["--attention-backend", "FLASHINFER"],
        "attention-triton": ["--attention-backend", "TRITON_ATTN"],
        "async-scheduling": ["--async-scheduling"],
        "graph-cap-80": ["--max-cudagraph-capture-size", "80"],
        "eager": ["--enforce-eager"],
        "reserve-partial-isl": ["--no-scheduler-reserve-full-isl"],
        "watermark-005": ["--watermark", "0.05"],
        "block-32": ["--block-size", "32"],
        "performance-throughput": ["--performance-mode", "throughput"],
        "performance-interactivity": ["--performance-mode", "interactivity"],
        "dual-batch-overlap": ["--enable-dbo"],
        "triton-watermark-005": [
            "--attention-backend", "TRITON_ATTN", "--watermark", "0.05"
        ],
        "triton-async": ["--attention-backend", "TRITON_ATTN", "--async-scheduling"],
        "triton-async-watermark-005": [
            "--attention-backend", "TRITON_ATTN", "--async-scheduling", "--watermark", "0.05"
        ],
        "triton-graph-cap-80": [
            "--attention-backend", "TRITON_ATTN", "--max-cudagraph-capture-size", "80"
        ],
    }
    if name not in variants:
        raise ValueError(f"unknown configuration: {name}")
    return arguments + variants[name]


def start_server(repo_dir: Path, container_name: str, port: int,
                 extra_args: list[str]) -> list[str]:
    model_dir = repo_dir / ".local/cosmos-reason2-2b/hf"
    result_root = repo_dir / ".local/vllm-cosmos-reason2-2b"
    cache_dir = result_root / "cache"
    if not (model_dir / "model.safetensors").is_file():
        raise FileNotFoundError(f"missing checkpoint: {model_dir / 'model.safetensors'}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    command = [
        "docker", "run", "--detach", "--name", container_name, "--gpus", "all", "--ipc=host",
        "--publish", f"{port}:8000", "--volume", f"{model_dir}:/model:ro", "--volume",
        f"{cache_dir}:/root/.cache/vllm", IMAGE, "/model", "--served-model-name", MODEL_NAME,
        "--language-model-only", "--dtype", "float16", "--kv-cache-dtype", "float16",
        "--max-model-len", "2048", "--max-num-seqs", "80", "--enable-chunked-prefill",
        "--no-enable-prefix-caching", "--generation-config", "vllm", "--kv-cache-memory",
        str(MATCHED_KV_BYTES), "--disable-uvicorn-access-log"
    ] + extra_args
    run(command)
    return command


def write_server_log(container_name: str, path: Path) -> None:
    result = run(["docker", "logs", container_name], check=False)
    path.write_text(result.stdout, encoding="utf-8")


def benchmark(repo_dir: Path, endpoint: str, trace: Path, output_dir: Path,
              repeats: int, warmup_requests: int) -> None:
    command = [
        "python3", str(repo_dir / "scripts/cosmos_reason2/run_vllm_trace_bench.py"),
        "--endpoint", endpoint, "--trace", str(trace), "--output-dir", str(output_dir),
        "--repeats", str(repeats), "--warmup-requests", str(warmup_requests), "--max-workers",
        "512", "--timeout", "900"
    ]
    result = subprocess.run(command, check=False, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"trace benchmark failed with exit code {result.returncode}")


def trace_paths(repo_dir: Path) -> dict[str, Path]:
    local = repo_dir / ".local/cosmos-reason2-2b"
    return {
        "short": local / "scheduler-ablation-20260812/short-static/materialized-trace.json",
        "balanced": local / "packed-d64-graph-ab-20260813/balanced-graph-r1/materialized-trace.json",
        "decode-heavy": local / "packed-d64-graph-ab-20260813/decode-heavy-graph-r1/materialized-trace.json",
        "long-prefill": local / "workload-suite-20260812/long-prefill-static-v2/materialized-trace.json",
        "bimodal-mixed": local / "workload-suite-20260812/bimodal-static-v2/materialized-trace.json",
    }


def selected_configurations(selection: list[str]) -> list[str]:
    budget = ["budget-1024", "budget-2048", "budget-4096", "budget-8192"]
    tuning = [
        "auto-tuned", "attention-flash-attn", "attention-flashinfer", "attention-triton",
        "async-scheduling", "graph-cap-80", "eager", "reserve-partial-isl", "watermark-005",
        "block-32", "performance-throughput", "performance-interactivity", "dual-batch-overlap"
    ]
    combinations = [
        "triton-watermark-005", "triton-async", "triton-async-watermark-005",
        "triton-graph-cap-80"
    ]
    expanded = []
    for item in selection:
        expanded.extend(budget + tuning + combinations if item == "all" else
                        budget if item == "budgets" else tuning if item == "tuning" else
                        combinations if item == "combinations" else [item])
    return list(dict.fromkeys(expanded))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--configs", nargs="+", default=["all"])
    parser.add_argument("--workloads", nargs="+", default=["balanced", "decode-heavy", "bimodal-mixed"])
    parser.add_argument("--tuning-token-budget", type=int, default=1024)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--warmup-requests", type=int, default=64)
    parser.add_argument("--startup-timeout", type=float, default=600.0)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    repo_dir = Path(__file__).resolve().parents[2]
    traces = trace_paths(repo_dir)
    unknown_workloads = set(args.workloads) - set(traces)
    if unknown_workloads:
        parser.error(f"unknown workloads: {sorted(unknown_workloads)}")
    configurations = selected_configurations(args.configs)
    args.output_root.mkdir(parents=True, exist_ok=True)
    container_name = "edgellm-vllm-cosmos-option-sweep"
    endpoint = f"http://127.0.0.1:{args.port}"

    for config_name in configurations:
        config_dir = args.output_root / config_name
        if args.resume and all((config_dir / workload / "aggregate.json").is_file()
                               for workload in args.workloads):
            print(f"skip complete configuration: {config_name}", flush=True)
            continue
        config_dir.mkdir(parents=True, exist_ok=True)
        status: dict[str, Any] = {
            "configuration": config_name,
            "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "state": "starting",
        }
        remove_container(container_name)
        try:
            extra_args = configuration_args(config_name, args.tuning_token_budget)
            command = start_server(repo_dir, container_name, args.port, extra_args)
            (config_dir / "server-command.json").write_text(json.dumps(command, indent=2) + "\n",
                                                             encoding="utf-8")
            wait_for_server(endpoint, container_name, args.startup_timeout)
            status["state"] = "ready"
            status["gpu_memory_ready"] = read_gpu_memory()
            (config_dir / "status.json").write_text(json.dumps(status, indent=2) + "\n",
                                                     encoding="utf-8")
            for workload in args.workloads:
                output_dir = config_dir / workload
                if args.resume and (output_dir / "aggregate.json").is_file():
                    continue
                print(f"run {config_name}: {workload}", flush=True)
                benchmark(repo_dir, endpoint, traces[workload], output_dir, args.repeats,
                          args.warmup_requests)
                status[f"gpu_memory_after_{workload}"] = read_gpu_memory()
            status["state"] = "complete"
        except Exception as exception:  # noqa: BLE001 - persist failure and continue the sweep.
            status["state"] = "failed"
            status["error"] = f"{type(exception).__name__}: {exception}"
            print(f"failed {config_name}: {status['error']}", flush=True)
        finally:
            write_server_log(container_name, config_dir / "server.log")
            (config_dir / "status.json").write_text(json.dumps(status, indent=2) + "\n",
                                                     encoding="utf-8")
            remove_container(container_name)


if __name__ == "__main__":
    main()
