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
"""Run phase serving (V3, independent E/P/D) and vLLM on the same traces without containers.

Each (system, workload) cell starts a fresh server, warms it with the shared calibration trace,
replays the workload ``--repeats`` times with ``openai_trace_client.py`` and stops the server.
A JSON config supplies paths and capacities, e.g. ``{"model": ..., "hf_dir": ..., "engine_dir": ...,
"vision_dir": ..., "build_dir": ..., "inputs": ..., "calibration_trace": ..., "in_flight": 24,
"prefill_batch": 8, "decode_batch": 24, "vision_batch": 4, "encoder_input_tokens": 1120,
"prefill_chunk": 512, "prefill_batch_tokens": 2048, "vllm_python": ..., "vllm_args": [...]}``.
"""

import argparse
import hashlib
import json
import os
import pathlib
import signal
import subprocess
import sys
import threading
import time

HERE = pathlib.Path(__file__).resolve().parent
WORKLOADS = ("balanced", "mixed", "vision-heavy", "multi-image",
             "long-prefill", "bimodal", "decode-heavy", "short", "text-heavy",
             "poisson", "wave-drain", "late-vision")


def warmup_decode_batches(maximum):
    sizes = (1, 2, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60,
             64)
    return ",".join(str(size) for size in sizes if size <= maximum)


def phase_environment(config):
    """The retained V3 serving contract of command_for(), with host paths instead of a container."""
    engine = json.loads(
        (pathlib.Path(config["engine_dir"]) / "config.json").read_text())
    environment = {
        "TRT_EDGELLM_PHASE_IPC":
        1,
        "TRT_EDGELLM_SEMANTIC_ONLY":
        1,
        "TRT_EDGELLM_IGNORE_EOS":
        1,
        "TRT_EDGELLM_MAX_STABLE_SLOTS":
        config.get("stable_slots", config["decode_batch"]),
        "TRT_EDGELLM_MAX_INFLIGHT":
        config["in_flight"],
        "TRT_EDGELLM_MAX_PREFILL_BATCH":
        config["prefill_batch"],
        "TRT_EDGELLM_MAX_DECODE_BATCH":
        config["decode_batch"],
        "TRT_EDGELLM_FIXED_PREFILL_CHUNK":
        config["prefill_chunk"],
        "TRT_EDGELLM_MAX_PREFILL_BATCH_TOKENS":
        config["prefill_batch_tokens"],
        "TRT_EDGELLM_ENABLE_DYNAMIC_DECODE":
        1,
        "TRT_EDGELLM_ENABLE_DECODE_COHORT":
        1,
        "TRT_EDGELLM_ENABLE_BATCHED_VISION_PREFILL":
        1,
        "TRT_EDGELLM_RELEASE_VISION_PREFILL_STORAGE":
        1,
        "TRT_EDGELLM_MAX_ENCODED_VISION":
        max(config["initial_capacity"], config["vision_batch"]),
        "TRT_EDGELLM_MEASUREMENT_ENCODED_ADMISSION":
        "lifetime",
        "TRT_EDGELLM_MEASUREMENT_ENCODED_CAPACITY":
        max(config["initial_capacity"], config["vision_batch"]),
        "TRT_EDGELLM_VISION_ENCODER_BATCH_SIZE":
        config["vision_batch"],
        "TRT_EDGELLM_VISION_ENCODER_MAX_INPUT_TOKENS":
        config["encoder_input_tokens"],
        "TRT_EDGELLM_VISION_ENCODER_MAX_MEDIA":
        config["vision_batch"],
        "TRT_EDGELLM_VISION_ENCODER_BATCH_WAIT_US":
        25000,
        "TRT_EDGELLM_VISION_PREFILL_BATCH_SIZE":
        config["vision_batch"],
        "TRT_EDGELLM_PREFILL_TTFT_HARD_GUARD":
        1,
        "TRT_EDGELLM_VISION_IDLE_SLABS":
        0,
        "TRT_EDGELLM_GLOBAL_SCHEDULER":
        "active",
        "TRT_EDGELLM_SYNCHRONIZE_DECODE_SAMPLING":
        1,
        "TRT_EDGELLM_GLOBAL_SAFE_PROBE_INTERVAL":
        0,
        "TRT_EDGELLM_GLOBAL_FORMATION_REALIZED_DISPATCHES":
        4,
        "TRT_EDGELLM_IPC_ASYNC_REQUEST_ADAPTER":
        1,
        "TRT_EDGELLM_IPC_REQUEST_ADAPTER_WORKERS":
        4,
        "TRT_EDGELLM_VISION_ASYNC_PREPARATION":
        1,
        "TRT_EDGELLM_VISION_PREPARATION_PD_DISPATCH":
        1,
        "TRT_EDGELLM_VISION_ENCODER_ARBITER":
        1,
        "TRT_EDGELLM_VISION_PREFIX_PREFILL":
        1,
        "TRT_EDGELLM_POLICY_WARMUP_MODE":
        "generic",
        "TRT_EDGELLM_PHASE_POLICY":
        "service-scaled-transition",
        "TRT_EDGELLM_VISION_ENGINE_DIR":
        config["vision_dir"],
        "TRT_EDGELLM_EMIT_PHASE_METRICS":
        1,
        "TRT_EDGELLM_PHASE_TELEMETRY_LEVEL":
        "dispatch",
        "TRT_EDGELLM_PHASE_WORKSPACE_MODE":
        "independent",
        "TRT_EDGELLM_ENABLE_TRANSITION_PREDICTOR":
        1,
        "TRT_EDGELLM_CAPTURE_PHASE_GRAPHS":
        1,
        "TRT_EDGELLM_MAX_PREFILL_GRAPHS":
        0,
        "TRT_EDGELLM_MAX_DECODE_GRAPHS":
        64,
        "TRT_EDGELLM_IPC_WARMUP_DECODE_BATCHES":
        warmup_decode_batches(config["decode_batch"]),
        "CUDA_CACHE_PATH":
        str(pathlib.Path(config["build_dir"]) / "cuda-cache"),
    }
    if not engine.get("packed_prefill", False):
        # Batched (multi-row) vision prefill is implemented only for the packed token carrier.
        del environment["TRT_EDGELLM_ENABLE_BATCHED_VISION_PREFILL"]
    if engine.get("use_vision_bidirectional_attention", False):
        # Bidirectional image blocks are request-local; a text prefix prefilled ahead of the image would split them.
        del environment["TRT_EDGELLM_VISION_PREFIX_PREFILL"]
    return environment


class GpuMemorySampler:

    def __init__(self, interval=0.2):
        self.interval = interval
        self.peak = 0
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self.stop_event.is_set():
            try:
                used = subprocess.run([
                    "nvidia-smi", "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits"
                ],
                                      capture_output=True,
                                      text=True,
                                      timeout=10).stdout.split()[0]
                self.peak = max(self.peak, int(used))
            except (subprocess.SubprocessError, IndexError, ValueError):
                pass
            self.stop_event.wait(self.interval)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop_event.set()
        self.thread.join()


def server_command(system, config, port, cell):
    if system == "trt":
        build = pathlib.Path(config["build_dir"])
        backend = [
            str(build / "examples/llm/llm_phase_context_smoke"),
            config["engine_dir"], config["hf_dir"]
        ]
        return [
            sys.executable,
            str(HERE / "phase_openai_gateway.py"), "--port",
            str(port), "--model", config["model"], "--telemetry-log",
            str(cell / "phase-telemetry.log"), "--"
        ] + backend
    return [
        config["vllm_python"], "-m", "vllm.entrypoints.openai.api_server",
        "--model", config["hf_dir"], "--served-model-name", config["model"],
        "--port",
        str(port), "--allowed-local-media-path",
        str(pathlib.Path(__file__).resolve().parents[2])
    ] + config["vllm_args"]


def stop(process):
    if process.poll() is None:
        process.send_signal(signal.SIGINT)
        try:
            process.wait(timeout=90)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def wait_gpu_idle(limit_mib=1024, timeout=120):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        used = subprocess.run([
            "nvidia-smi", "--query-gpu=memory.used",
            "--format=csv,noheader,nounits"
        ],
                              capture_output=True,
                              text=True).stdout.split()
        if used and int(used[0]) < limit_mib:
            return
        time.sleep(2)


def run_cell(system, config, workload, args):
    cell = args.output_dir / system / workload
    summary_path = cell / "summary.json"
    if summary_path.exists() and not args.force:
        print("skip completed", cell, flush=True)
        return json.loads(summary_path.read_text())
    cell.mkdir(parents=True, exist_ok=True)
    trace = pathlib.Path(config["inputs"]) / (workload + ".json")
    environment = dict(os.environ)
    if system == "trt":
        environment.update({
            key: str(value)
            for key, value in phase_environment(config).items()
        })
        environment.update({
            key: str(value)
            for key, value in config.get("trt_environment", {}).items()
        })
    else:
        # vLLM JIT-compiles kernels with tools (ninja) installed next to its interpreter.
        environment["PATH"] = str(pathlib.Path(
            config["vllm_python"]).parent) + os.pathsep + environment["PATH"]
    command = server_command(system, config, args.port, cell)
    (cell / "command.json").write_text(
        json.dumps(
            {
                "server": command,
                "environment": {
                    key: environment[key]
                    for key in environment
                    if key.startswith(("TRT_EDGELLM_", "VLLM_"))
                },
                "trace": str(trace),
                "trace_sha256": hashlib.sha256(trace.read_bytes()).hexdigest()
            },
            indent=2) + "\n")
    wait_gpu_idle()
    with (cell / "server.log").open("w") as log:
        server = subprocess.Popen(command,
                                  env=environment,
                                  stdout=log,
                                  stderr=subprocess.STDOUT)
        try:
            client = [
                sys.executable,
                str(HERE / "openai_trace_client.py"), "--endpoint",
                "http://127.0.0.1:%d" % args.port, "--model", config["model"],
                "--trace",
                str(trace), "--output-dir",
                str(cell), "--max-in-flight",
                str(config["in_flight"]), "--repeats",
                str(args.repeats), "--warmup-trace",
                config["calibration_trace"], "--warmup-requests",
                str(config["warmup_requests"]), "--ignore-eos"
            ]
            if system == "trt":
                client.append("--phase-calibration")
            with GpuMemorySampler() as sampler, (
                    cell / "client.log").open("w") as client_log:
                runner = subprocess.Popen(client,
                                          stdout=client_log,
                                          stderr=subprocess.STDOUT)
                while runner.poll() is None:
                    if server.poll() is not None:
                        runner.kill()
                        runner.wait()
                        raise RuntimeError(
                            "server exited with %s during %s/%s" %
                            (server.returncode, system, workload))
                    time.sleep(1)
            print((cell / "client.log").read_text().strip(), flush=True)
            if runner.returncode != 0:
                raise RuntimeError("client failed for %s/%s (see %s)" %
                                   (system, workload, cell / "client.log"))
        finally:
            stop(server)
    summary = json.loads(summary_path.read_text())
    summary["gpu_memory_peak_mib"] = sampler.peak
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=pathlib.Path, required=True)
    parser.add_argument("--systems",
                        nargs="+",
                        choices=("trt", "vllm"),
                        default=["trt", "vllm"])
    parser.add_argument("--workloads",
                        nargs="+",
                        choices=WORKLOADS,
                        default=list(WORKLOADS))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config = json.loads(args.config.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir /
     "config.json").write_text(json.dumps(config, indent=2) + "\n")
    failures = []
    for workload in args.workloads:
        for system in args.systems:
            print("=== %s / %s" % (system, workload), flush=True)
            try:
                run_cell(system, config, workload, args)
            except Exception as error:  # noqa: BLE001 - keep the campaign running, record the failure
                failures.append({
                    "system": system,
                    "workload": workload,
                    "error": str(error)
                })
                print("FAILED", system, workload, error, flush=True)
    rows = []
    for workload in args.workloads:
        row = {"workload": workload}
        for system in args.systems:
            path = args.output_dir / system / workload / "summary.json"
            if path.exists():
                summary = json.loads(path.read_text())
                runs = summary["runs"]
                row[system] = {
                    "generated_token_s_median":
                    summary.get("generated_token_s_median"),
                    "ttft_mean_ms":
                    [run.get("ttft_ms", {}).get("mean") for run in runs],
                    "tpot_mean_ms":
                    [run.get("tpot_ms", {}).get("mean") for run in runs],
                    "fixed_output_complete":
                    all(run.get("fixed_output_complete") for run in runs),
                    "failed_requests":
                    sum(run.get("failed", 0) for run in runs),
                    "gpu_memory_peak_mib":
                    summary.get("gpu_memory_peak_mib"),
                }
        rows.append(row)
    (args.output_dir / "comparison.json").write_text(
        json.dumps({
            "rows": rows,
            "failures": failures
        }, indent=2) + "\n")
    print(json.dumps(failures, indent=1))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
