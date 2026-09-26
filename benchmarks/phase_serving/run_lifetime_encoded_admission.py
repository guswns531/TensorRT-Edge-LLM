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
WORKLOADS = ("balanced", "mixed", "vision-heavy", "multi-image",
             "long-prefill", "bimodal", "decode-heavy", "short", "text-heavy",
             "poisson", "wave-drain", "late-vision")
WARMUP_DECODE_BATCHES = (1, 2, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48,
                         52, 56, 60, 64)


def digest(path):
    """Hash immutable artifacts incrementally."""
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def warmup_decode_batches(maximum):
    """Preserve the documented calibration workload; graph priming is separate."""
    return ",".join(
        str(batch) for batch in WARMUP_DECODE_BATCHES if batch <= maximum)


def model_config(repo, name, overrides=None):
    """Resolve retained model-specific capabilities, not workload-specific policies."""
    artifacts = repo / ".local/artifacts"
    if name == "gemma":
        model = artifacts / "v0101-forward-port/gemma-4-e2b-it-awq"
        inputs = repo / ".local/results/gemma4-e2b-awq-full12-20260911/inputs"
        config = {
            "model":
            "google/gemma-4-e2b-it",
            "engine":
            pathlib.Path(
                os.environ.get("GEMMA_ENGINE_DIR",
                               str(model /
                                   "engine-packed-p8-d24-kv2048-p192"))),
            "vision":
            pathlib.Path(
                os.environ.get("GEMMA_VISION_DIR",
                               str(model / "visual-e4-soft280/visual"))),
            "hf":
            artifacts / "models/gemma-4-e2b-it-awq/hf",
            "traces": {
                w: inputs / (w + ".json")
                for w in ("balanced", "mixed", "vision-heavy", "multi-image",
                          "long-prefill", "bimodal", "decode-heavy", "short",
                          "text-heavy", "poisson", "wave-drain", "late-vision")
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
            int(os.environ.get("GEMMA_ENCODER_INPUT_TOKENS", "1120")),
            "initial_capacity":
            int(os.environ.get("GEMMA_INITIAL_CAPACITY", "4")),
            "vision_batch_size":
            int(os.environ.get("GEMMA_VISION_BATCH_SIZE", "4")),
            "kv_pages":
            192,
            "larger_capacity":
            12,
            "frozen_vllm":
            repo /
            ".local/results/gemma4-vllm-capacity-sweep-20260912/selected-seq24-kv480-p4096-g24-full12"
        }
        config.update(overrides or {})
        return config
    canonical = repo / ".local/results/v0101-forward-port/heuristic-elimination-20260911/v3-profile-free-canonical-full12-3x"
    commands = json.loads((canonical / "commands.json").read_text())
    traces = {}
    calibration = None
    for record in commands:
        command = record["command"]
        traces[record["case"]] = repo / command[command.index("--trace") + 1]
        if record["case"] == "mixed" and calibration is None:
            calibration = repo / command[
                command.index("--generic-warmup-trace") + 1]
    config = {
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
        "vision_batch_size":
        4,
        "kv_pages":
        256,
        "encoder_input_tokens":
        8192,
        "initial_capacity":
        16,
        "larger_capacity":
        80,
        "frozen_vllm":
        repo /
        ".local/results/review-correction-20260926/cosmos-vllm-frozen-raw-corrected.json"
    }
    config.update(overrides or {})
    return config


def command_for(repo,
                config,
                cell,
                workload,
                variant,
                byte_budget,
                options=None):
    """Construct a restricted, network-free GPU backend with measured-boundary activation."""
    tools = repo / ".local/results/v0101-forward-port/replay-tools"
    options = options or {}
    build = pathlib.Path(
        options.get(
            "build_root",
            os.environ.get("BUILD_ROOT",
                           str(repo / ".local/builds/v0101-validation"))))
    mode = variant if variant in ("lifetime", "ownership") else "static"
    if variant in ("chunked", "e1", "e2", "e-dynamic-shadow",
                   "e-transition-shadow", "e-dynamic-active", "shared_ep",
                   "tiered_ep", "independent"):
        mode = "lifetime"
    capacity = config["initial_capacity"]
    if variant == "static-large":
        capacity = config["larger_capacity"]
    elif variant == "static-slot":
        capacity = config["stable_slots"]
    if "TRT_EDGELLM_MEASUREMENT_ENCODED_CAPACITY" in os.environ:
        capacity = int(os.environ["TRT_EDGELLM_MEASUREMENT_ENCODED_CAPACITY"])
    vision_batch = int(
        os.environ.get("TRT_EDGELLM_VISION_ENCODER_BATCH_SIZE",
                       str(config.get("vision_batch_size", 4))))
    vision_prefill = int(
        os.environ.get("TRT_EDGELLM_VISION_PREFILL_BATCH_SIZE",
                       str(config.get("vision_batch_size", 4))))
    capacity = max(capacity, vision_prefill)
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
        max(config["initial_capacity"], vision_prefill),
        "TRT_EDGELLM_MEASUREMENT_ENCODED_ADMISSION":
        mode,
        "TRT_EDGELLM_MEASUREMENT_ENCODED_CAPACITY":
        capacity,
        "TRT_EDGELLM_VISION_ENCODER_BATCH_SIZE":
        vision_batch,
        "TRT_EDGELLM_VISION_ENCODER_MAX_INPUT_TOKENS":
        int(
            os.environ.get("TRT_EDGELLM_VISION_ENCODER_MAX_INPUT_TOKENS",
                           str(config["encoder_input_tokens"]))),
        "TRT_EDGELLM_VISION_ENCODER_MAX_MEDIA":
        int(
            os.environ.get("TRT_EDGELLM_VISION_ENCODER_MAX_MEDIA",
                           str(vision_batch))),
        "TRT_EDGELLM_VISION_ENCODER_BATCH_WAIT_US":
        int(os.environ.get("TRT_EDGELLM_VISION_ENCODER_BATCH_WAIT_US",
                           "25000")),
        "TRT_EDGELLM_VISION_PREFILL_BATCH_SIZE":
        vision_prefill,
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
        options.get("telemetry_level", "full")
    }
    if "TRT_EDGELLM_DECODE_BURST_GRACE_PERIOD_US" in os.environ:
        environment["TRT_EDGELLM_DECODE_BURST_GRACE_PERIOD_US"] = os.environ[
            "TRT_EDGELLM_DECODE_BURST_GRACE_PERIOD_US"]
    for key in ("TRT_EDGELLM_ENABLE_TRANSITION_PREDICTOR",
                "TRT_EDGELLM_RESIDENT_DECODE_SHADOW",
                "TRT_EDGELLM_SHARED_EP_SINGLE_STORAGE"):
        if key in os.environ:
            environment[key] = os.environ[key]
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
    if variant in ("shared_ep", "tiered_ep", "independent", "auto"):
        environment["TRT_EDGELLM_PHASE_WORKSPACE_MODE"] = variant
    if variant == "unified_action":
        environment["TRT_EDGELLM_PHASE_WORKSPACE_MODE"] = "shared_ep"
        environment["TRT_EDGELLM_ENABLE_ADAPTIVE_PREFILL_CHUNKING"] = 1
        environment[
            "TRT_EDGELLM_ADAPTIVE_PREFILL_CHUNK_CANDIDATES"] = "128,256,512"
    graphs_enabled = options.get(
        "cuda_graphs",
        os.environ.get("ENABLE_CUDA_GRAPHS") == "1"
        or "TRT_EDGELLM_CAPTURE_PHASE_GRAPHS" in os.environ)
    if graphs_enabled:
        environment["TRT_EDGELLM_CAPTURE_PHASE_GRAPHS"] = os.environ.get(
            "TRT_EDGELLM_CAPTURE_PHASE_GRAPHS", 1)
        environment["TRT_EDGELLM_MAX_PREFILL_GRAPHS"] = os.environ.get(
            "TRT_EDGELLM_MAX_PREFILL_GRAPHS", 0)
        environment["TRT_EDGELLM_MAX_DECODE_GRAPHS"] = os.environ.get(
            "TRT_EDGELLM_MAX_DECODE_GRAPHS", 64)
        if "TRT_EDGELLM_IPC_WARMUP_DECODE_BATCHES" in os.environ:
            environment["TRT_EDGELLM_IPC_WARMUP_DECODE_BATCHES"] = os.environ[
                "TRT_EDGELLM_IPC_WARMUP_DECODE_BATCHES"]
        else:
            environment[
                "TRT_EDGELLM_IPC_WARMUP_DECODE_BATCHES"] = warmup_decode_batches(
                    config["decode_batch"])
    environment.update(options.get("environment", {}))
    if not options.get("serving_overlap_probes", True):
        environment["TRT_EDGELLM_DISABLE_SERVING_OVERLAP_PROBES"] = "1"

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


def command_environment(command):
    """Return only the explicitly forwarded backend environment, never host secrets."""
    return dict(command[index + 1].split("=", 1)
                for index, value in enumerate(command) if value == "-e")


def validate_cell_contract(cell, contract, completed):
    """Reject stale cells instead of silently reusing output from another contract."""
    path = cell / "contract.json"
    if path.exists():
        if json.loads(path.read_text()) != contract:
            raise ValueError("Cell contract differs: " + str(cell))
    elif completed:
        raise ValueError(
            "Existing aggregate has no verifiable cell contract: " + str(cell))
    else:
        path.write_text(json.dumps(contract, indent=2) + "\n")


def token_repeatability(runs):
    """A singleton's aggregate flag is not evidence of repeated token identity."""
    hashes = [
        value for run in runs
        for value in run.get("token_trace_sha256_per_run", [])
    ]
    return {
        "observations":
        len(hashes),
        "status":
        ("not_tested" if len(hashes) < 2 else
         "observed_equal" if len(set(hashes)) == 1 else "observed_different"),
        "hashes":
        hashes,
    }


def campaign_completion(manifest, finished=False):
    """Count requested cells rather than mistaking a partial summary for success."""
    requested = {record["cell"] for record in manifest["commands"]}
    completed = {record["cell"]
                 for record in manifest.get("completed", [])} & requested
    failed = {record["cell"]
              for record in manifest.get("failures", [])
              } & (requested - completed)
    missing = requested - completed
    return {
        "completion_status":
        ("complete" if not missing else "partial") if finished else "running",
        "requested_cells":
        len(requested),
        "completed_cells":
        len(completed),
        "failed_cells":
        len(failed),
        "missing_cells":
        len(missing),
        "missing_cell_paths":
        sorted(missing),
        "exit_code":
        1 if finished and missing else 0,
    }


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
                                 "multi-image", "long-prefill", "bimodal",
                                 "decode-heavy", "short", "text-heavy",
                                 "poisson", "wave-drain", "late-vision"),
                        default=["mixed", "vision-heavy", "multi-image"])
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=("static-base", "static-large", "static-slot", "lifetime",
                 "ownership", "chunked", "e1", "e2", "e-dynamic-shadow",
                 "e-transition-shadow", "e-dynamic-active", "shared_ep",
                 "tiered_ep", "independent", "unified_action"),
        default=["static-base", "static-large", "lifetime"])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--full12", action="store_true")
    parser.add_argument("--build-root",
                        type=pathlib.Path,
                        default=pathlib.Path(
                            os.environ.get("BUILD_ROOT",
                                           ".local/builds/v0101-validation")))
    parser.add_argument(
        "--binary-source-commit",
        help="Build provenance for an explicitly frozen binary")
    for model_name in ("gemma", "cosmos"):
        parser.add_argument("--" + model_name + "-engine-dir",
                            type=pathlib.Path)
        parser.add_argument("--" + model_name + "-vision-dir",
                            type=pathlib.Path)
    parser.add_argument("--transition-predictors",
                        nargs="+",
                        choices=("on", "off"),
                        default=["on"])
    parser.add_argument("--cuda-graphs", choices=("on", "off"), default="on")
    parser.add_argument(
        "--telemetry-level",
        choices=("full", "dispatch"),
        default="full",
        help="Full causal diagnostics or compact dispatch metrics")
    parser.add_argument("--max-decode-graphs", type=int, default=64)
    parser.add_argument("--max-prefill-graphs", type=int, default=0)
    parser.add_argument(
        "--serving-overlap-probes",
        choices=("on", "off"),
        default="on",
        help="Unknown serving probes; calibration remains enabled")
    parser.add_argument("--shared-ep-single-storage",
                        choices=("0", "1"),
                        default="1")
    parser.add_argument("--byte-budget", type=int, default=0)
    parser.add_argument(
        "--result-root",
        type=pathlib.Path,
        default=pathlib.Path(
            ".local/results/lifetime-encoded-admission-20260913"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--compress-closed-logs", action="store_true")
    args = parser.parse_args()
    if args.repeats < 1 or args.byte_budget < 0 or min(
            args.max_decode_graphs, args.max_prefill_graphs) < 0:
        parser.error(
            "Repeat count must be positive and byte budget non-negative")
    repo = pathlib.Path(
        subprocess.check_output(["git", "rev-parse", "--show-toplevel"],
                                text=True).strip())
    if args.full12:
        args.workloads = list(WORKLOADS)
    configs = {}
    paths = {
        "gemma": {
            "engine": args.gemma_engine_dir,
            "vision": args.gemma_vision_dir
        },
        "cosmos": {
            "engine": args.cosmos_engine_dir,
            "vision": args.cosmos_vision_dir
        },
    }
    for model_name in args.models:
        overrides = {
            key: value.resolve()
            for key, value in paths[model_name].items() if value is not None
        }
        configs[model_name] = model_config(repo, model_name, overrides)
    build = args.build_root.resolve()
    binary = build / "examples/llm/llm_phase_context_smoke"
    plugin = build / "libNvInfer_edgellm_plugin.so.1.0"
    identity = {
        "source_commit":
        subprocess.check_output(["git", "rev-parse", "HEAD"],
                                text=True).strip(),
        "tracked_diff_sha256":
        hashlib.sha256(subprocess.check_output(["git", "diff",
                                                "HEAD"])).hexdigest(),
        "source_status":
        subprocess.check_output(["git", "status", "--porcelain"],
                                text=True).splitlines(),
        "untracked_source_sha256": {
            name: digest(repo / name)
            for name in subprocess.check_output(
                ["git", "ls-files", "--others", "--exclude-standard"],
                text=True).splitlines() if (repo / name).is_file()
        },
        "binary_sha256":
        digest(binary),
        "runner_sha256":
        digest(pathlib.Path(__file__).resolve()),
        "plugin_sha256":
        digest(plugin),
        "container":
        IMAGE,
        "binary_source_commit": (subprocess.check_output(
            ["git", "rev-parse", args.binary_source_commit + "^{commit}"],
            text=True).strip() if args.binary_source_commit else None),
        "models": {}
    }
    for name, config in configs.items():
        engine_config = json.loads(
            (config["engine"] / "config.json").read_text())
        builder = engine_config["builder_config"]
        if builder.get("max_kv_pool_pages") != config["kv_pages"]:
            raise ValueError(
                "Expected KV pages %d for %s; got %r" %
                (config["kv_pages"], name, builder.get("max_kv_pool_pages")))
        vision_config = json.loads(
            (config["vision"] / "config.json").read_text())
        if "tiered_ep" in args.variants and not vision_config[
                "builder_config"].get("small_profile_max_image_tokens"):
            raise ValueError(
                "tiered_ep requires an explicitly selected multi-profile vision engine: "
                + name)
        identity["models"][name] = {
            "engine": str(config["engine"]),
            "engine_sha256": digest(config["engine"] / "llm.engine"),
            "config_sha256": digest(config["engine"] / "config.json"),
            "vision_sha256": digest(config["vision"] / "visual.engine"),
            "vision_config_sha256": digest(config["vision"] / "config.json"),
            "effective_model_config":
            json.loads(json.dumps(config, default=str)),
            "engine_builder_config": builder,
            "vision_builder_config": vision_config["builder_config"],
            "engine_sidecars_sha256": {
                path.name: digest(path)
                for path in sorted(config["engine"].iterdir())
                if path.is_file() and path.name not in ("llm.engine",
                                                        "config.json")
            },
            "calibration_sha256": digest(config["calibration"]),
            "traces": {
                w: digest(config["traces"][w])
                for w in args.workloads
            }
        }
    replay_tools = repo / ".local/results/v0101-forward-port/replay-tools"
    identity["replay_tools_sha256"] = {
        name: digest(replay_tools / name)
        for name in ("run_phase_http_trace_bench.py",
                     "run_phase_openai_gateway.py", "run_vllm_trace_bench.py")
    }
    root = args.result_root.resolve()
    commands = []
    for repeat in range(1, args.repeats + 1):
        variants = args.variants if repeat % 2 else list(
            reversed(args.variants))
        for name, config in configs.items():
            for workload in args.workloads:
                for variant in variants:
                    predictors = args.transition_predictors if repeat % 2 else list(
                        reversed(args.transition_predictors))
                    for predictor in predictors:
                        variant_label = variant + "-predictor-" + predictor
                        cell = root / name / variant_label / (
                            "repeat-%03d" % repeat) / workload
                        options = {
                            "build_root":
                            str(build),
                            "cuda_graphs":
                            args.cuda_graphs == "on",
                            "telemetry_level":
                            args.telemetry_level,
                            "serving_overlap_probes":
                            args.serving_overlap_probes == "on",
                            "environment": {
                                "TRT_EDGELLM_ENABLE_TRANSITION_PREDICTOR":
                                "1" if predictor == "on" else "0",
                                "TRT_EDGELLM_SHARED_EP_SINGLE_STORAGE":
                                args.shared_ep_single_storage,
                                "TRT_EDGELLM_MAX_PREFILL_GRAPHS":
                                args.max_prefill_graphs,
                                "TRT_EDGELLM_MAX_DECODE_GRAPHS":
                                args.max_decode_graphs,
                                "TRT_EDGELLM_IPC_WARMUP_DECODE_BATCHES":
                                os.environ.get(
                                    "TRT_EDGELLM_IPC_WARMUP_DECODE_BATCHES",
                                    warmup_decode_batches(
                                        config["decode_batch"])),
                            },
                        }
                        command = command_for(repo, config, cell, workload,
                                              variant, args.byte_budget,
                                              options)
                        commands.append({
                            "model":
                            name,
                            "workload":
                            workload,
                            "variant":
                            variant_label,
                            "workspace_variant":
                            variant,
                            "transition_predictor":
                            predictor,
                            "repeat":
                            repeat,
                            "cell":
                            str(cell),
                            "command":
                            command,
                            "effective_environment":
                            command_environment(command),
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
            "notes/305-small-encoder-progressive-overlap-20260914.md",
            "notes/330-runtime-contract-revalidation-plan-20260926.md"
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
        manifest["completed"] = previous.get("completed", [])
        manifest["failures"] = previous.get("failures", [])
    manifest.update(campaign_completion(manifest))
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
        if (digest(binary) != identity["binary_sha256"]
                or digest(plugin) != identity["plugin_sha256"]):
            raise ValueError(
                "Runtime binary or plugin changed during campaign")
        cell = pathlib.Path(record["cell"])
        cell.mkdir(parents=True, exist_ok=True)
        contract = {"identity": identity, "record": record}
        validate_cell_contract(cell, contract,
                               (cell / "aggregate.json").exists())
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
                manifest.update(campaign_completion(manifest))
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
        expected_trace = identity["models"][record["model"]]["traces"][
            record["workload"]]
        if aggregate.get("trace_sha256") != expected_trace:
            raise ValueError("Aggregate workload identity differs: " +
                             str(cell))
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
            group["token_repeatability"] = token_repeatability(group["runs"])
        (root /
         "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        completion = {
            k: record[k]
            for k in ("model", "workload", "variant", "repeat", "cell")
        }
        if completion not in manifest["completed"]:
            manifest["completed"].append(completion)
        manifest.update(campaign_completion(manifest))
        path.write_text(json.dumps(manifest, indent=2) + "\n")
        if args.compress_closed_logs:
            closed_log = cell / "run-001/gateway.log"
            if closed_log.exists():
                subprocess.run(
                    ["gzip", "-1", "--", str(closed_log)], check=True)
    manifest.update(campaign_completion(manifest, finished=True))
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    if manifest["missing_cells"]:
        print("Incomplete campaign: %d/%d requested cells completed" %
              (manifest["completed_cells"], manifest["requested_cells"]),
              file=sys.stderr)
    return manifest["exit_code"]


if __name__ == "__main__":
    sys.exit(main())
