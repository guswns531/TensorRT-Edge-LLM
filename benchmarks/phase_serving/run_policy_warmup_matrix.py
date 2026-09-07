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
"""Run policy-evidence and runtime-lifecycle warmup A/B matrices."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

MODES = ("graph_only", "zero_start", "generic_reset", "generic",
         "trace_derived")
POLICY_VARIANTS = {
    # V0--V2 share the exact same candidate, feasibility, SLO, ownership, and
    # dispatch mechanisms.  This single environment setting changes only the
    # decision-cost authority and bounded transition evaluation.
    "v0_exact": {
        "TRT_EDGELLM_PHASE_POLICY": "exact",
    },
    "v1_scalar": {
        "TRT_EDGELLM_PHASE_POLICY": "scalar",
    },
    "v2_scalar_transition": {
        "TRT_EDGELLM_PHASE_POLICY": "scalar-transition",
    },
}

DEPRECATED_POLICY_ENVIRONMENT = {
    "TRT_EDGELLM_CONTEXTUAL_PD",
    "TRT_EDGELLM_CONTEXTUAL_EP",
    "TRT_EDGELLM_CONTEXTUAL_ED",
    "TRT_EDGELLM_ENABLE_GLOBAL_FORMATION_AWARE",
    "TRT_EDGELLM_CONTEXTUAL_SUCCESSOR_GUARD",
    "TRT_EDGELLM_COMPLETION_CONFORMAL",
    "TRT_EDGELLM_COMPLETION_CONFORMAL_ACTIVE",
}


def _set_option(command: list[str], option: str, value: str) -> None:
    if option in command:
        command[command.index(option) + 1] = value
    else:
        separator = command.index("--")
        command[separator:separator] = [option, value]


def _drop_option(command: list[str], option: str, has_value: bool) -> None:
    while option in command:
        index = command.index(option)
        del command[index:index + (2 if has_value else 1)]


def _inject_backend_mode(command: list[str]) -> None:
    assignment = "TRT_EDGELLM_POLICY_WARMUP_MODE={policy_warmup_mode}"
    if assignment in command:
        return
    try:
        image = next(
            index for index, value in enumerate(command)
            if value.startswith("nvcr.io/") or value.startswith("docker.io/"))
    except StopIteration as error:
        raise ValueError(
            "backend command has no recognizable container image") from error
    command[image:image] = ["-e", assignment]


def _set_backend_environment(command: list[str], name: str,
                             value: str) -> None:
    index = 0
    while index + 1 < len(command):
        if command[index] == "-e" and command[index + 1].split("=",
                                                               1)[0] == name:
            del command[index:index + 2]
            continue
        index += 1
    image = next(
        index for index, token in enumerate(command)
        if token.startswith("nvcr.io/") or token.startswith("docker.io/"))
    command[image:image] = ["-e", f"{name}={value}"]


def _drop_backend_environment(command: list[str], name: str) -> None:
    index = 0
    while index + 1 < len(command):
        if command[index] == "-e" and command[index + 1].split("=", 1)[0] == name:
            del command[index:index + 2]
            continue
        index += 1


def _replace_backend_build(command: list[str],
                           backend_build_root: str) -> None:
    if not backend_build_root:
        return
    binary_suffix = "/examples/llm/llm_phase_context_smoke"
    plugin_name = "/libNvInfer_edgellm_plugin.so.1.0"
    for index, value in enumerate(command):
        if value.endswith(binary_suffix):
            command[index] = backend_build_root.rstrip("/") + binary_suffix
        elif value.startswith("EDGELLM_PLUGIN_PATH="):
            command[index] = ("EDGELLM_PLUGIN_PATH=" +
                              backend_build_root.rstrip("/") + plugin_name)
        elif value.startswith("LD_LIBRARY_PATH="):
            paths = value.split("=", 1)[1].split(":")
            paths = [
                path for path in paths if not path.endswith("/examples/llm")
            ]
            paths.append(backend_build_root.rstrip("/") + "/examples/llm")
            command[index] = "LD_LIBRARY_PATH=" + ":".join(paths)


def _replace_backend_engine(command: list[str],
                            backend_engine_dir: str) -> None:
    if not backend_engine_dir:
        return
    binary_suffix = "/examples/llm/llm_phase_context_smoke"
    try:
        binary_index = next(index for index, value in enumerate(command)
                            if value.endswith(binary_suffix))
    except StopIteration as error:
        raise ValueError(
            "backend command has no phase context binary") from error
    if binary_index + 1 >= len(command):
        raise ValueError("phase context binary has no engine argument")
    command[binary_index + 1] = backend_engine_dir


def _container_workspace_path(command: list[str], path: Path) -> str:
    """Map a host output path through the command's writable /workspace mount."""
    for index, token in enumerate(command[:-1]):
        if token != "-v":
            continue
        fields = command[index + 1].split(":")
        if len(fields) >= 2 and fields[1] == "/workspace":
            relative = path.resolve().relative_to(Path(fields[0]).resolve())
            return str(Path("/workspace") / relative)
    raise ValueError("backend command has no writable /workspace mount")


def _is_vision_trace(path: Path) -> bool:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return any(
        part.get("type") == "image_url"
        for request in payload.get("requests", [])
        for message in request.get("messages", []) for part in (message.get(
            "content", []) if isinstance(message.get("content"), list) else [])
        if isinstance(part, dict))


def prepare_command(entry: dict[str, Any],
                    mode: str,
                    output_dir: Path,
                    repeats: int,
                    generic_text: Path,
                    generic_vlm: Path,
                    backend_build_root: str = "",
                    backend_engine_dir: str = "",
                    policy_variant: str = "v2_scalar_transition",
                    backend_environment: tuple[str, ...] = (),
                    client_max_in_flight: int = 0,
                    capture_phase_telemetry: bool = False,
                    trace_override: str = "",
                    generic_calibration_max_rounds: int = 1) -> list[str]:
    if policy_variant not in POLICY_VARIANTS:
        raise ValueError(f"unknown policy variant: {policy_variant}")
    command = list(entry["command"])
    _set_option(command, "--output-dir", str(output_dir))
    _set_option(command, "--repeats", str(repeats))
    # generic_reset executes the same generic calibration requests as generic,
    # but policy_reset discards only the contextual policy posterior. Exact
    # CUDA costs, graphs, TensorRT contexts, allocators, and engine tactics stay
    # warm, isolating contextual learning from physical/runtime calibration.
    client_mode = "generic" if mode == "generic_reset" else mode
    _set_option(command, "--policy-warmup-mode", client_mode)
    if trace_override:
        _set_option(command, "--trace", trace_override)
    if client_max_in_flight > 0:
        _set_option(command, "--max-in-flight", str(client_max_in_flight))
    _inject_backend_mode(command)
    if mode == "generic_reset":
        _set_backend_environment(command, "TRT_EDGELLM_POLICY_WARMUP_MODE",
                                 "policy_reset")
    _replace_backend_build(command, backend_build_root)
    _replace_backend_engine(command, backend_engine_dir)
    for name in DEPRECATED_POLICY_ENVIRONMENT:
        _drop_backend_environment(command, name)
    for name, value in POLICY_VARIANTS[policy_variant].items():
        _set_backend_environment(command, name, value)
    if capture_phase_telemetry:
        activity_prefix = _container_workspace_path(
            command, output_dir / "activity" / "run-{run}")
        _set_backend_environment(command, "TRT_EDGELLM_EMIT_PHASE_METRICS",
                                 "1")
        _set_backend_environment(command, "TRT_EDGELLM_PHASE_TELEMETRY_LEVEL",
                                 "counterfactual")
        _set_backend_environment(command, "TRT_EDGELLM_PHASE_ACTIVITY_PREFIX",
                                 activity_prefix)
        _set_backend_environment(command, "TRT_EDGELLM_PHASE_TELEMETRY_PATH",
                                 activity_prefix + "-events.jsonl")
    # Explicit caller overrides are applied last.  In particular, callers may
    # request full request-lineage telemetry while retaining the activity-file
    # plumbing installed by capture_phase_telemetry.
    for assignment in backend_environment:
        name, value = assignment.split("=", 1)
        _set_backend_environment(command, name, value)
    _drop_option(command, "--generic-warmup-trace", True)
    if mode in ("generic_reset", "generic"):
        trace = Path(command[command.index("--trace") + 1])
        calibration = generic_vlm if _is_vision_trace(trace) else generic_text
        request_count = len(
            json.loads(calibration.read_text(encoding="utf-8"))["requests"])
        _set_option(command, "--generic-warmup-trace", str(calibration))
        _set_option(command, "--warmup-requests",
                    str(request_count * generic_calibration_max_rounds))
        # The generic trace encodes action coverage in both request ordering
        # and arrival offsets. Execute the complete trace as one calibration
        # round so the client cannot truncate a later phase-order cycle or
        # restart its arrival epoch in the middle of the trace.
        _set_option(command, "--phase-calibration-round-requests",
                    str(request_count))
        _set_option(command, "--phase-calibration-min-requests",
                    str(request_count))
    elif mode in ("graph_only", "zero_start"):
        _set_option(command, "--warmup-requests", "0")
        _set_option(command, "--phase-calibration-min-requests", "0")
    return command


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-commands", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--generic-text", type=Path, required=True)
    parser.add_argument("--generic-vlm", type=Path, required=True)
    parser.add_argument("--backend-build-root", default="")
    parser.add_argument("--backend-engine-dir", default="")
    parser.add_argument("--client-max-in-flight", type=int, default=0)
    parser.add_argument(
        "--generic-calibration-max-rounds",
        type=int,
        default=1,
        help=("repeat the workload-agnostic calibration trace until authority "
              "converges, up to this many complete rounds"))
    parser.add_argument("--trace-override", type=Path)
    parser.add_argument("--policy-variant",
                        choices=sorted(POLICY_VARIANTS),
                        default="v2_scalar_transition")
    parser.add_argument("--modes", default=",".join(MODES))
    parser.add_argument("--cases", default="")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--capture-phase-telemetry",
        action="store_true",
        help="write per-run CUDA activity and unified decision side channels")
    parser.add_argument(
        "--backend-env",
        action="append",
        default=[],
        help="override one backend NAME=VALUE assignment; may be repeated")
    args = parser.parse_args()
    modes = tuple(value for value in args.modes.split(",") if value)
    if not modes or any(mode not in MODES for mode in modes):
        parser.error(f"modes must be drawn from {','.join(MODES)}")
    if (args.repeats <= 0 or args.client_max_in_flight < 0
            or args.generic_calibration_max_rounds <= 0):
        parser.error(
            "repeats and generic-calibration-max-rounds must be positive; "
            "client-max-in-flight cannot be negative"
        )
    if any("=" not in assignment or not assignment.split("=", 1)[0]
           for assignment in args.backend_env):
        parser.error("backend-env values must use NAME=VALUE")
    selected_cases = {value for value in args.cases.split(",") if value}
    entries = json.loads(args.base_commands.read_text(encoding="utf-8"))
    if selected_cases:
        entries = [
            entry for entry in entries if entry["case"] in selected_cases
        ]
    if not entries:
        parser.error("no selected base commands")
    commands = []
    for mode in modes:
        for entry in entries:
            case = str(entry["case"])
            output = args.output_dir / mode / case / str(entry["variant"])
            command = prepare_command(
                entry, mode, output, args.repeats, args.generic_text,
                args.generic_vlm, args.backend_build_root,
                args.backend_engine_dir, args.policy_variant,
                tuple(args.backend_env), args.client_max_in_flight,
                args.capture_phase_telemetry,
                str(args.trace_override) if args.trace_override else "",
                args.generic_calibration_max_rounds)
            commands.append({
                "mode": mode,
                "policy_variant": args.policy_variant,
                "case": case,
                "variant": entry["variant"],
                "command": command,
            })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "commands.json").write_text(
        json.dumps(commands, indent=2) + "\n", encoding="utf-8")
    if not args.dry_run:
        for item in commands:
            print(json.dumps(
                {key: item[key]
                 for key in ("mode", "case", "variant")}),
                  flush=True)
            subprocess.run(item["command"], check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
