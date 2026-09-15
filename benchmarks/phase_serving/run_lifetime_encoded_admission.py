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
"""Compare static windows and lifetime admission after a common calibration procedure."""

import argparse
import hashlib
import json
import os
import pathlib
import statistics
import subprocess
import sys

IMAGE = "nvcr.io/nvidia/tensorrt@sha256:7cd94ee931d2b5b85ad1c5af723d485b2625f6ce167e1e4abe577850b96ceac3"
METRICS = ("generated_token_s_median", "ttft_mean_of_run_means_ms",
           "ttft_p95_median_ms", "tpot_mean_of_run_means_ms",
           "tpot_p95_median_ms", "e2e_mean_of_run_means_ms",
           "e2e_p95_median_ms", "gpu_memory_peak_mib_median")


def digest(path):
    """Hash immutable artifacts incrementally."""
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def model_config(repo, name):
    """Resolve retained model-specific capabilities, not workload-specific policies."""
    artifacts = repo / ".local/artifacts"
    if name == "gemma":
        model = artifacts / "v0101-forward-port/gemma-4-e2b-it-awq"
        inputs = repo / ".local/results/gemma4-e2b-awq-full12-20260911/inputs"
        return {
            "model":
            "google/gemma-4-e2b-it",
            "engine":
            model / "engine-packed-p8-d24-kv2048-p96",
            "vision":
            model / "visual-e4-soft280/visual",
            "hf":
            artifacts / "models/gemma-4-e2b-it-awq/hf",
            "traces": {
                w: inputs / (w + ".json")
                for w in ("balanced", "mixed", "vision-heavy", "multi-image")
            },
            "calibration":
            repo /
            ".local/results/gemma4-packed-prefill-g4-20260912/generic-p8-d24-e4.json",
            "calibration_requests":
            49,
            "stable_slots":
            24,
            "in_flight":
            24,
            "decode_batch":
            24,
            "encoder_input_tokens":
            1120,
            "initial_capacity":
            4,
            "larger_capacity":
            12,
            "frozen_vllm":
            repo /
            ".local/results/gemma4-vllm-capacity-sweep-20260912/selected-seq24-kv480-p4096-g24-full12"
        }
    canonical = repo / ".local/results/v0101-forward-port/heuristic-elimination-20260911/v3-profile-free-canonical-full12-3x"
    commands = json.loads((canonical / "commands.json").read_text())
    traces = {}
    calibration = None
    for record in commands:
        if record["case"] in ("balanced", "mixed", "vision-heavy",
                              "multi-image"):
            command = record["command"]
            traces[record["case"]] = repo / command[command.index("--trace") +
                                                    1]
            if record["case"] == "mixed":
                calibration = repo / command[
                    command.index("--generic-warmup-trace") + 1]
    return {
        "model":
        "nvidia/Cosmos-Reason2-2B",
        "engine":
        artifacts /
        "v0101-forward-port/text/cosmos-reason2-2b/engine-p8-d64-kv256-p128-vp1024-atomic",
        "vision":
        artifacts /
        "v0101-forward-port/cosmos-reason2-2b/vision-exact-gelu/engine/visual",
        "hf":
        artifacts / "models/cosmos-reason2-2b/hf",
        "traces":
        traces,
        "calibration":
        calibration,
        "calibration_requests":
        239,
        "stable_slots":
        80,
        "in_flight":
        64,
        "decode_batch":
        64,
        "encoder_input_tokens":
        8192,
        "initial_capacity":
        16,
        "larger_capacity":
        80,
        "frozen_vllm":
        repo /
        ".local/results/v0101-forward-port/v3-service-scale-20260910/vllm-fresh-equal-summary.json"
    }


def command_for(repo, config, cell, workload, variant, byte_budget):
    """Construct a restricted, network-free GPU backend with measured-boundary activation."""
    tools = repo / ".local/results/v0101-forward-port/replay-tools"
    build = repo / ".local/builds/v0101-release"
    mode = variant if variant in ("lifetime", "ownership") else "static"
    if variant in ("chunked", "e1", "e2", "e-dynamic-shadow",
                   "e-transition-shadow", "e-dynamic-active"):
        mode = "lifetime"
    capacity = config["initial_capacity"]
    if variant == "static-large":
        capacity = config["larger_capacity"]
    elif variant == "static-slot":
        capacity = config["stable_slots"]
    environment = {
        "TRT_PACKAGE_DIR":
        "/usr/local/tensorrt",
        "EDGELLM_PLUGIN_PATH":
        "/opt/edgellm/libNvInfer_edgellm_plugin.so.1.0",
        "LD_LIBRARY_PATH":
        "/opt/edgellm:/usr/local/cuda/lib64:/usr/local/tensorrt/lib",
        "CUDA_CACHE_PATH":
        "/tmp/cuda-cache",
        "TRT_EDGELLM_PHASE_IPC":
        1,
        "TRT_EDGELLM_SEMANTIC_ONLY":
        1,
        "TRT_EDGELLM_IGNORE_EOS":
        1,
        "TRT_EDGELLM_MAX_STABLE_SLOTS":
        config["stable_slots"],
        "TRT_EDGELLM_MAX_INFLIGHT":
        config["in_flight"],
        "TRT_EDGELLM_MAX_PREFILL_BATCH":
        8,
        "TRT_EDGELLM_MAX_DECODE_BATCH":
        config["decode_batch"],
        "TRT_EDGELLM_FIXED_PREFILL_CHUNK":
        128,
        "TRT_EDGELLM_MAX_PREFILL_BATCH_TOKENS":
        1024,
        "TRT_EDGELLM_ENABLE_DYNAMIC_DECODE":
        1,
        "TRT_EDGELLM_ENABLE_DECODE_COHORT":
        1,
        "TRT_EDGELLM_ENABLE_BATCHED_VISION_PREFILL":
        1,
        "TRT_EDGELLM_RELEASE_VISION_PREFILL_STORAGE":
        1,
        "TRT_EDGELLM_MAX_ENCODED_VISION":
        config["initial_capacity"],
        "TRT_EDGELLM_MEASUREMENT_ENCODED_ADMISSION":
        mode,
        "TRT_EDGELLM_MEASUREMENT_ENCODED_CAPACITY":
        capacity,
        "TRT_EDGELLM_VISION_ENCODER_BATCH_SIZE":
        4,
        "TRT_EDGELLM_VISION_ENCODER_MAX_INPUT_TOKENS":
        config["encoder_input_tokens"],
        "TRT_EDGELLM_VISION_ENCODER_MAX_MEDIA":
        4,
        "TRT_EDGELLM_VISION_ENCODER_BATCH_WAIT_US":
        25000,
        "TRT_EDGELLM_VISION_PREFILL_BATCH_SIZE":
        4,
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
        "/opt/vision",
        "TRT_EDGELLM_PHASE_ACTIVITY_PREFIX":
        "/opt/results/run-{run}/activity",
        "TRT_EDGELLM_EMIT_PHASE_METRICS":
        1,
        "TRT_EDGELLM_PHASE_TELEMETRY_LEVEL":
        "full"
    }
    if byte_budget > 0:
        environment["TRT_EDGELLM_MAX_ENCODED_VISION_BYTES"] = byte_budget
    if variant == "chunked":
        environment["TRT_EDGELLM_MEASUREMENT_CHUNKED_VISION_PREFILL"] = 1
    if variant in ("e1", "e2"):
        environment["TRT_EDGELLM_MEASUREMENT_ENCODER_PREPARATION_ROWS"] = int(
            variant[1:])
    if variant in ("e-dynamic-shadow", "e-transition-shadow",
                   "e-dynamic-active"):
        environment["TRT_EDGELLM_MEASUREMENT_ENCODER_PREPARATION_POLICY"] = (
            "transition-shadow" if variant == "e-transition-shadow" else
            ("shadow" if variant.endswith("shadow") else "active"))
    backend = [
        "docker", "run", "--rm", "--gpus", "all", "--network", "none",
        "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
        "--read-only", "--tmpfs", "/tmp:rw,size=268435456", "-i", "--user",
        str(os.getuid()) + ":" + str(os.getgid())
    ]
    pics = repo / "examples/multimodal/pics"
    for source, target, access in ((build, "/opt/edgellm", "ro"),
                                   (config["engine"], "/opt/model", "ro"),
                                   (config["vision"], "/opt/vision",
                                    "ro"), (config["hf"], "/opt/hf",
                                            "ro"), (pics, str(pics), "ro"),
                                   (pics,
                                    "/workspace/examples/multimodal/pics",
                                    "ro"), (cell, "/opt/results", "rw")):
        backend += ["-v", str(source) + ":" + target + ":" + access]
    for name, value in environment.items():
        backend += ["-e", name + "=" + str(value)]
    backend += [
        IMAGE, "/opt/edgellm/examples/llm/llm_phase_context_smoke",
        "/opt/model", "/opt/hf"
    ]
    return [
        sys.executable,
        str(tools / "run_phase_http_trace_bench.py"), "--gateway-script",
        str(tools / "run_phase_openai_gateway.py"), "--client-script",
        str(tools / "run_vllm_trace_bench.py"), "--trace",
        str(config["traces"][workload]), "--output-dir",
        str(cell), "--model", config["model"], "--repeats", "1",
        "--max-workers",
        str(config["in_flight"]), "--max-in-flight",
        str(config["in_flight"]), "--ready-timeout", "180",
        "--request-timeout", "600", "--policy-warmup-mode", "generic",
        "--generic-warmup-trace",
        str(config["calibration"]), "--warmup-requests",
        str(config["calibration_requests"]),
        "--phase-calibration-round-requests",
        str(config["calibration_requests"]),
        "--phase-calibration-min-requests",
        str(config["calibration_requests"]), "--ignore-eos", "--"
    ] + backend


def main():
    """Run immutable paired cells and retain their exact source, engine and workload contract."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models",
                        nargs="+",
                        choices=("gemma", "cosmos"),
                        default=["gemma", "cosmos"])
    parser.add_argument("--workloads",
                        nargs="+",
                        choices=("balanced", "mixed", "vision-heavy",
                                 "multi-image"),
                        default=["mixed", "vision-heavy", "multi-image"])
    parser.add_argument("--variants",
                        nargs="+",
                        choices=("static-base", "static-large", "static-slot",
                                 "lifetime", "ownership", "chunked", "e1",
                                 "e2", "e-dynamic-shadow",
                                 "e-transition-shadow", "e-dynamic-active"),
                        default=["static-base", "static-large", "lifetime"])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--byte-budget", type=int, default=0)
    parser.add_argument(
        "--result-root",
        type=pathlib.Path,
        default=pathlib.Path(
            ".local/results/lifetime-encoded-admission-20260913"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--compress-closed-logs", action="store_true")
    args = parser.parse_args()
    if args.repeats < 1 or args.byte_budget < 0:
        parser.error(
            "Repeat count must be positive and byte budget non-negative")
    repo = pathlib.Path(
        subprocess.check_output(["git", "rev-parse", "--show-toplevel"],
                                text=True).strip())
    configs = {m: model_config(repo, m) for m in args.models}
    binary = repo / ".local/builds/v0101-release/examples/llm/llm_phase_context_smoke"
    identity = {
        "source_commit":
        subprocess.check_output(["git", "rev-parse", "HEAD"],
                                text=True).strip(),
        "tracked_diff_sha256":
        hashlib.sha256(subprocess.check_output(["git", "diff",
                                                "HEAD"])).hexdigest(),
        "binary_sha256":
        digest(binary),
        "runner_sha256":
        digest(pathlib.Path(__file__).resolve()),
        "plugin_sha256":
        digest(repo /
               ".local/builds/v0101-release/libNvInfer_edgellm_plugin.so.1.0"),
        "container":
        IMAGE,
        "models": {}
    }
    for name, config in configs.items():
        identity["models"][name] = {
            "engine": str(config["engine"]),
            "engine_sha256": digest(config["engine"] / "llm.engine"),
            "config_sha256": digest(config["engine"] / "config.json"),
            "vision_sha256": digest(config["vision"] / "visual.engine"),
            "calibration_sha256": digest(config["calibration"]),
            "traces": {
                w: digest(config["traces"][w])
                for w in args.workloads
            }
        }
    root = args.result_root.resolve()
    commands = []
    for repeat in range(1, args.repeats + 1):
        variants = args.variants if repeat % 2 else list(
            reversed(args.variants))
        for name, config in configs.items():
            for workload in args.workloads:
                for variant in variants:
                    cell = root / name / variant / ("repeat-%03d" %
                                                    repeat) / workload
                    commands.append({
                        "model":
                        name,
                        "workload":
                        workload,
                        "variant":
                        variant,
                        "repeat":
                        repeat,
                        "cell":
                        str(cell),
                        "command":
                        command_for(repo, config, cell, workload, variant,
                                    args.byte_budget)
                    })
    if args.dry_run:
        print(
            json.dumps({
                "identity": identity,
                "commands": commands
            }, indent=2))
        return
    root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "state":
        "diagnostic",
        "identity":
        identity,
        "commands":
        commands,
        "byte_budget":
        args.byte_budget,
        "repeat_count":
        args.repeats,
        "activation":
        "drained post-calibration boundary; common static initialization per model",
        "calibration": {
            m: configs[m]["calibration_requests"]
            for m in configs
        },
        "frozen_vllm": {
            m: str(configs[m]["frozen_vllm"])
            for m in configs
        },
        "summary_path":
        str(root / "summary.json"),
        "completed": [],
        "failures": [],
        "note_references": [
            "notes/301-lifetime-encoded-admission-two-model-plan-20260913.md",
            "notes/303-service-admission-cause-and-experiment-20260914.md",
            "notes/305-small-encoder-progressive-overlap-20260914.md"
        ],
        "compress_closed_logs":
        args.compress_closed_logs
    }
    path = root / "manifest.json"
    if path.exists():
        previous = json.loads(path.read_text())
        if previous["identity"] != identity or previous["commands"] != commands:
            raise ValueError(
                "Cannot resume a different source, binary, engine or command contract"
            )
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    (root / "runner-source.py").write_bytes(
        pathlib.Path(__file__).read_bytes())
    (root / "source.patch").write_bytes(
        subprocess.check_output(["git", "diff", "HEAD"]))
    env = dict(
        os.environ,
        PHASE_TRACE_CLIENT_IMPL=str(
            repo /
            ".local/results/v0101-forward-port/replay-tools/run_vllm_trace_bench.py"
        ))
    summary = {}
    for record in commands:
        if digest(binary) != identity["binary_sha256"]:
            raise ValueError("Runtime binary changed during campaign")
        cell = pathlib.Path(record["cell"])
        cell.mkdir(parents=True, exist_ok=True)
        print("Running",
              record["model"],
              record["workload"],
              record["variant"],
              record["repeat"],
              flush=True)
        if not (cell / "aggregate.json").exists():
            with (cell / "driver.log").open("w") as log:
                outcome = subprocess.run(record["command"],
                                         env=env,
                                         stdout=log,
                                         stderr=subprocess.STDOUT,
                                         check=False)
            if outcome.returncode != 0:
                manifest["failures"].append({
                    **{
                        k: record[k]
                        for k in ("model", "workload", "variant", "repeat", "cell")
                    }, "return_code": outcome.returncode
                })
                path.write_text(json.dumps(manifest, indent=2) + "\n")
                print("Failed",
                      record["model"],
                      record["workload"],
                      record["variant"],
                      flush=True)
                if args.compress_closed_logs:
                    closed_log = cell / "run-001/gateway.log"
                    if closed_log.exists():
                        subprocess.run(["gzip", "-1", "--",
                                        str(closed_log)],
                                       check=True)
                continue
        aggregate = json.loads((cell / "aggregate.json").read_text())
        if aggregate["generated_tokens_per_run_min"] != aggregate[
                "requested_output_tokens_per_run"]:
            raise ValueError("Incomplete fixed-output workload")
        key = "/".join(
            (record["model"], record["workload"], record["variant"]))
        summary.setdefault(key, {"runs": []})["runs"].append(aggregate)
        for group in summary.values():
            group["mean_run_metrics"] = {
                m: statistics.mean(r[m] for r in group["runs"])
                for m in METRICS
            }
        (root /
         "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        manifest["completed"].append({
            k: record[k]
            for k in ("model", "workload", "variant", "repeat", "cell")
        })
        path.write_text(json.dumps(manifest, indent=2) + "\n")
        if args.compress_closed_logs:
            closed_log = cell / "run-001/gateway.log"
            if closed_log.exists():
                subprocess.run(
                    ["gzip", "-1", "--", str(closed_log)], check=True)


if __name__ == "__main__":
    main()
