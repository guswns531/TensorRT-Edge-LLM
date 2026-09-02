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
"""Run the six-direction, five-offset M2 real-request injection matrix."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

DIRECTIONS = (
    "prefill_to_decode",
    "decode_to_prefill",
    "encoder_to_decode",
    "decode_to_encoder",
    "encoder_to_prefill",
    "prefill_to_encoder",
)
TARGETS = (0.0, 0.25, 0.5, 0.75, 0.9)
PHASE_BY_DIRECTION = {
    "prefill_to_decode": ("prefill", "decode", "pd"),
    "decode_to_prefill": ("decode", "prefill", "pd"),
    "encoder_to_decode": ("encoder", "decode", "ed"),
    "decode_to_encoder": ("decode", "encoder", "ed"),
    "encoder_to_prefill": ("encoder", "prefill", "ep"),
    "prefill_to_encoder": ("prefill", "encoder", "ep"),
}


def container_path(path: Path, workspace: Path) -> str:
    """Map one workspace-owned host path into the benchmark container."""
    try:
        relative = path.resolve().relative_to(workspace.resolve())
    except ValueError as error:
        raise ValueError(
            f"path must be inside workspace {workspace}: {path}") from error
    return str(Path("/workspace") / relative)


def build_traces(args: argparse.Namespace) -> dict[str, Path]:
    """Create one fixed real-request frontier for each phase pair."""
    trace_dir = args.output_dir / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    traces = {
        direction: trace_dir / f"{direction}.json"
        for direction in DIRECTIONS
    }
    subprocess.run([
        sys.executable,
        str(args.pd_trace_builder),
        "--output",
        str(traces["prefill_to_decode"]),
        "--decode-requests",
        "32",
        "--decode-output-tokens",
        "192",
        "--prefill-waves",
        "8",
        "--prefill-wave-size",
        "8",
        "--prefill-start-ms",
        "100",
        "--prefill-wave-interval-ms",
        "20",
    ],
                   check=True)
    traces["decode_to_prefill"].write_bytes(
        traces["prefill_to_decode"].read_bytes())
    for direction in DIRECTIONS[2:]:
        _, _, kind = PHASE_BY_DIRECTION[direction]
        encoder_incumbent = direction.startswith("encoder_to_")
        command = [
            sys.executable,
            str(args.encoder_trace_builder),
            "--output",
            str(traces[direction]),
            "--image",
            str(args.image),
            "--mode",
            kind,
            "--encoder-requests",
            "32",
            "--prefill-requests",
            "24",
            "--decode-requests",
            "32",
            "--decode-output-tokens",
            "192",
            # Interleaving puts both request classes in the first client
            # frontier. The coordinator then holds the mature phase candidate
            # only for E->phase and controls the actual launch order.
            "--phase-order",
            "interleaved" if encoder_incumbent else "phase_first",
        ]
        if kind == "ed":
            command.extend(
                ("--late-arrival-us", "0" if encoder_incumbent else "300000"))
        subprocess.run(command, check=True)
    return traces


def env_option(name: str, value: object) -> list[str]:
    """Build one docker environment option."""
    return ["-e", f"{name}={value}"]


def cleanup_case_containers(case_name: str, repeats: int) -> None:
    """Remove only containers owned by one directional-injection cell."""
    for run_index in range(1, repeats + 1):
        name = f"phase-p3-{case_name.replace('_', '-')}-run-{run_index:03d}"
        try:
            subprocess.run(["docker", "rm", "-f", name],
                           check=False,
                           stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL,
                           timeout=30.0)
        except subprocess.TimeoutExpired:
            print(f"WARNING: timed out cleaning container {name}",
                  file=sys.stderr,
                  flush=True)


def run_case(args: argparse.Namespace, traces: dict[str, Path], direction: str,
             target: float) -> list[Path]:
    """Execute one requested cell and return its side-channel event streams."""
    incumbent, newcomer, pair = PHASE_BY_DIRECTION[direction]
    references = {
        "encoder": args.encoder_us,
        "prefill": args.prefill_us,
        "decode": args.decode_us,
    }
    case_name = f"{direction}-o{round(target * 100):02d}"
    case_dir = args.output_dir / case_name
    logs = sorted(case_dir.glob("activity/run-*-events.jsonl"))
    aggregates = sorted(case_dir.glob("run-*/client/aggregate.json"))
    complete_runs = {
        f"run-{int(path.parents[1].name.split('-')[-1]):03d}"
        for path in aggregates if path.stat().st_size > 0
    }
    complete_logs = [
        path for path in logs
        if path.stem.removesuffix("-events") in complete_runs
        and path.stat().st_size > 0
    ]
    if args.skip_existing and len(complete_logs) == args.repeats:
        return complete_logs
    if args.skip_existing and (logs or aggregates):
        print(
            f"rerunning partial cell {case_name}: "
            f"{len(complete_logs)}/{args.repeats} complete repeats",
            flush=True)

    workspace = args.workspace.resolve()
    activity_prefix = container_path(case_dir / "activity" / "run-{run}",
                                     workspace)
    backend = [
        "docker",
        "run",
        "--rm",
        "--name",
        f"phase-p3-{case_name.replace('_', '-')}-run-{{run}}",
        "--gpus",
        "all",
        "--ipc",
        "host",
        "-i",
        "-v",
        f"{workspace}:/workspace",
        "-v",
        f"{workspace}:{workspace}:ro",
        "-w",
        "/workspace",
    ]
    environment = {
        "EDGELLM_PLUGIN_PATH":
        container_path(args.plugin, workspace),
        "LD_LIBRARY_PATH":
        f"/opt/tensorrt/lib:/usr/local/cuda/lib64:{container_path(args.binary.parent, workspace)}",
        "TRT_EDGELLM_PHASE_IPC":
        1,
        "TRT_EDGELLM_IGNORE_EOS":
        1,
        "TRT_EDGELLM_MAX_STABLE_SLOTS":
        args.stable_slots,
        "TRT_EDGELLM_MAX_INFLIGHT":
        args.stable_slots,
        "TRT_EDGELLM_MAX_PREFILL_BATCH":
        args.prefill_batch,
        "TRT_EDGELLM_MAX_DECODE_BATCH":
        args.decode_batch,
        "TRT_EDGELLM_FIXED_PREFILL_CHUNK":
        128,
        "TRT_EDGELLM_ENABLE_BATCHED_VISION_PREFILL":
        1,
        "TRT_EDGELLM_RELEASE_VISION_PREFILL_STORAGE":
        1,
        "TRT_EDGELLM_MAX_ENCODED_VISION":
        16,
        "TRT_EDGELLM_VISION_ENCODER_BATCH_SIZE":
        args.encoder_batch,
        "TRT_EDGELLM_VISION_ENCODER_BATCH_WAIT_US":
        25000,
        "TRT_EDGELLM_VISION_ENCODER_CREDIT_WAIT_US":
        25000,
        "TRT_EDGELLM_VISION_ENCODER_CREDIT_TARGET":
        args.encoder_batch,
        "TRT_EDGELLM_VISION_ENCODER_MAX_INPUT_TOKENS":
        8192,
        "TRT_EDGELLM_VISION_ENCODER_MAX_MEDIA":
        8,
        "TRT_EDGELLM_VISION_EXCLUSIVE_INPUT_TOKENS":
        0,
        "TRT_EDGELLM_VISION_PREFILL_BATCH_SIZE":
        min(4, args.prefill_batch),
        "TRT_EDGELLM_VISION_TTFT_TARGET_MS":
        5000,
        "TRT_EDGELLM_VISION_IDLE_SLABS":
        0,
        "TRT_EDGELLM_VISION_ENGINE_DIR":
        container_path(args.vision_engine_dir, workspace),
        "TRT_EDGELLM_GLOBAL_SCHEDULER":
        "active",
        "TRT_EDGELLM_DISABLE_GLOBAL_OVERLAP_WARMUP":
        1,
        # Completion learning remains shadow-only. Directional injection owns
        # the research action, while the hierarchical model observes the
        # resulting CUDA completion vector without changing dispatch policy.
        "TRT_EDGELLM_CONTEXTUAL_PD":
        "shadow",
        "TRT_EDGELLM_CONTEXTUAL_EP":
        "shadow",
        "TRT_EDGELLM_CONTEXTUAL_ED":
        "shadow",
        "TRT_EDGELLM_CONTEXTUAL_PD_CONFIDENCE_BETA":
        1.96,
        "TRT_EDGELLM_CONTEXTUAL_EP_CONFIDENCE_BETA":
        1.96,
        "TRT_EDGELLM_CONTEXTUAL_ED_CONFIDENCE_BETA":
        1.96,
        "TRT_EDGELLM_EMIT_PHASE_METRICS":
        1,
        "TRT_EDGELLM_PHASE_TELEMETRY_LEVEL":
        "counterfactual",
        "TRT_EDGELLM_PHASE_ACTIVITY_PREFIX":
        activity_prefix,
        "TRT_EDGELLM_PHASE_TELEMETRY_PATH":
        activity_prefix + "-events.jsonl",
        "TRT_EDGELLM_IPC_ASYNC_REQUEST_ADAPTER":
        1,
        "TRT_EDGELLM_IPC_REQUEST_ADAPTER_WORKERS":
        8,
        "TRT_EDGELLM_DIRECTIONAL_INJECTION":
        direction,
        "TRT_EDGELLM_DIRECTIONAL_INJECTION_TARGET":
        target,
        "TRT_EDGELLM_DIRECTIONAL_INJECTION_INCUMBENT_US":
        references[incumbent],
        "TRT_EDGELLM_DIRECTIONAL_INJECTION_NEWCOMER_US":
        references[newcomer],
    }
    if args.completion_conformal:
        environment["TRT_EDGELLM_COMPLETION_CONFORMAL"] = 1
        environment[
            "TRT_EDGELLM_COMPLETION_CONFORMAL_MIN_OBSERVATIONS"] = args.completion_conformal_min_observations
        environment[
            "TRT_EDGELLM_COMPLETION_CONFORMAL_WINDOW"] = args.completion_conformal_window
        environment[
            "TRT_EDGELLM_COMPLETION_CONFORMAL_TARGET"] = args.completion_conformal_target
    if pair == "pd":
        environment[
            "TRT_EDGELLM_EXPERIMENTAL_OVERLAP_PERCENT"] = args.overlap_percent
    elif pair == "ep":
        environment[
            "TRT_EDGELLM_EXPERIMENTAL_ENCODER_PREFILL_OVERLAP_PERCENT"] = args.overlap_percent
    else:
        environment[
            "TRT_EDGELLM_EXPERIMENTAL_ENCODER_DECODE_OVERLAP_PERCENT"] = args.overlap_percent
    for name, value in environment.items():
        backend.extend(env_option(name, value))
    backend.extend((
        args.docker_image,
        container_path(args.binary, workspace),
        container_path(args.engine_dir, workspace),
        container_path(args.checkpoint_dir, workspace),
    ))

    command = [
        sys.executable,
        str(args.harness),
        "--gateway-script",
        str(args.gateway_script),
        "--client-script",
        str(args.client_script),
        "--trace",
        str(traces[direction]),
        "--output-dir",
        str(case_dir),
        "--repeats",
        str(args.repeats),
        "--warmup-requests",
        "0",
        "--phase-calibration-min-requests",
        "0",
        "--max-workers",
        "8",
        "--max-in-flight",
        str(args.stable_slots),
        "--ready-timeout",
        str(args.ready_timeout),
        "--request-timeout",
        str(args.request_timeout),
        "--ignore-eos",
        "--",
        *backend,
    ]
    if args.dry_run:
        case_dir.mkdir(parents=True, exist_ok=True)
        (case_dir / "command.json").write_text(json.dumps(command, indent=2) +
                                               "\n",
                                               encoding="utf-8")
        return []
    try:
        subprocess.run(command, check=True)
    finally:
        cleanup_case_containers(case_name, args.repeats)
    logs = sorted(case_dir.glob("activity/run-*-events.jsonl"))
    if not logs:
        raise RuntimeError(
            f"no telemetry event stream produced for {case_name}")
    if len(logs) != args.repeats:
        raise RuntimeError(
            f"expected {args.repeats} event streams for {case_name}, found {len(logs)}"
        )
    return logs


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    benchmark_dir = Path(__file__).resolve().parent
    workspace_default = root.parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=workspace_default)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--plugin", type=Path, required=True)
    parser.add_argument("--engine-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--vision-engine-dir", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--encoder-us", type=float, required=True)
    parser.add_argument("--prefill-us", type=float, required=True)
    parser.add_argument("--decode-us", type=float, required=True)
    parser.add_argument("--direction",
                        choices=(*DIRECTIONS, "all"),
                        default="all")
    parser.add_argument("--target", type=float, action="append")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--ready-timeout", type=float, default=180.0)
    parser.add_argument("--request-timeout", type=float, default=180.0)
    parser.add_argument("--stable-slots", type=int, default=80)
    parser.add_argument("--encoder-batch", type=int, default=4)
    parser.add_argument("--prefill-batch", type=int, default=8)
    parser.add_argument("--decode-batch", type=int, default=32)
    parser.add_argument("--overlap-percent",
                        type=int,
                        choices=range(0, 101),
                        default=100)
    parser.add_argument("--completion-conformal", action="store_true")
    parser.add_argument("--completion-conformal-min-observations",
                        type=int,
                        default=16)
    parser.add_argument("--completion-conformal-window", type=int, default=128)
    parser.add_argument("--completion-conformal-target",
                        type=float,
                        default=0.95)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--docker-image",
                        default="nvcr.io/nvidia/tensorrt:26.06-py3")
    parser.add_argument("--harness",
                        type=Path,
                        default=workspace_default /
                        "scripts/cosmos_reason2/run_phase_http_trace_bench.py")
    parser.add_argument("--gateway-script",
                        type=Path,
                        default=workspace_default /
                        "scripts/cosmos_reason2/run_phase_openai_gateway.py")
    parser.add_argument("--client-script",
                        type=Path,
                        default=workspace_default /
                        "scripts/cosmos_reason2/run_vllm_trace_bench.py")
    parser.add_argument("--pd-trace-builder",
                        type=Path,
                        default=benchmark_dir /
                        "build_overlap_opportunity_trace.py")
    parser.add_argument("--encoder-trace-builder",
                        type=Path,
                        default=benchmark_dir /
                        "build_encoder_overlap_trace.py")
    parser.add_argument("--analyzer",
                        type=Path,
                        default=benchmark_dir /
                        "analyze_directional_injection.py")
    args = parser.parse_args()
    if min(args.encoder_us, args.prefill_us, args.decode_us) <= 0.0:
        parser.error("isolated phase references must be positive")
    targets = args.target or list(TARGETS)
    if any(target not in TARGETS for target in targets):
        parser.error(f"targets must be selected from {TARGETS}")
    if (args.repeats <= 0 or min(args.stable_slots, args.encoder_batch,
                                 args.prefill_batch, args.decode_batch) <= 0
            or min(args.ready_timeout, args.request_timeout) <= 0.0):
        parser.error("repeats and capacities must be positive")
    if (args.completion_conformal_min_observations <= 0
            or args.completion_conformal_window
            < args.completion_conformal_min_observations):
        parser.error(
            "completion conformal window must cover a positive minimum observation count"
        )
    if not 0.0 < args.completion_conformal_target < 1.0:
        parser.error(
            "completion conformal target must be between zero and one")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    traces = build_traces(args)
    directions = DIRECTIONS if args.direction == "all" else (args.direction, )
    logs: list[Path] = []
    total = len(directions) * len(targets)
    for index, (direction, target) in enumerate(((direction, target)
                                                 for direction in directions
                                                 for target in targets),
                                                start=1):
        print(f"[{index}/{total}] direction={direction} target={target:.2f}",
              flush=True)
        logs.extend(run_case(args, traces, direction, target))

    artifact = args.output_dir / "directional-injection-artifact.json"
    samples = args.output_dir / "directional-injection-samples.csv"
    if not args.dry_run:
        subprocess.run([
            sys.executable,
            str(args.analyzer),
            *(str(log) for log in logs),
            "--output-json",
            str(artifact),
            "--output-csv",
            str(samples),
        ],
                       check=True)
    manifest = {
        "schema_version": 1,
        "directions": list(directions),
        "targets": targets,
        "references_us": {
            "encoder": args.encoder_us,
            "prefill": args.prefill_us,
            "decode": args.decode_us
        },
        "logs": [str(log) for log in logs],
        "artifact": str(artifact),
        "samples": str(samples),
        "dry_run": args.dry_run,
        "completion_conformal": {
            "enabled": args.completion_conformal,
            "minimum_observations": args.completion_conformal_min_observations,
            "window": args.completion_conformal_window,
            "target": args.completion_conformal_target,
        },
    }
    (args.output_dir / "matrix-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
