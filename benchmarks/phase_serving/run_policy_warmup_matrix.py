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
"""Run graph-only/zero/generic/trace-derived policy warmup A/B matrices."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

MODES = ("graph_only", "zero_start", "generic", "trace_derived")
POLICY_VARIANTS = {
    "full_active": {
        "TRT_EDGELLM_CONTEXTUAL_PD": "active",
        "TRT_EDGELLM_CONTEXTUAL_EP": "active",
        "TRT_EDGELLM_CONTEXTUAL_ED": "active",
    },
    "pd_only": {
        "TRT_EDGELLM_CONTEXTUAL_PD": "active",
        "TRT_EDGELLM_CONTEXTUAL_EP": "shadow",
        "TRT_EDGELLM_CONTEXTUAL_ED": "shadow",
    },
    "full_shadow": {
        "TRT_EDGELLM_CONTEXTUAL_PD": "shadow",
        "TRT_EDGELLM_CONTEXTUAL_EP": "shadow",
        "TRT_EDGELLM_CONTEXTUAL_ED": "shadow",
    },
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
                    policy_variant: str = "full_active") -> list[str]:
    if policy_variant not in POLICY_VARIANTS:
        raise ValueError(f"unknown policy variant: {policy_variant}")
    command = list(entry["command"])
    _set_option(command, "--output-dir", str(output_dir))
    _set_option(command, "--repeats", str(repeats))
    _set_option(command, "--policy-warmup-mode", mode)
    _inject_backend_mode(command)
    _replace_backend_build(command, backend_build_root)
    for name, value in POLICY_VARIANTS[policy_variant].items():
        _set_backend_environment(command, name, value)
    _drop_option(command, "--generic-warmup-trace", True)
    if mode == "generic":
        trace = Path(command[command.index("--trace") + 1])
        calibration = generic_vlm if _is_vision_trace(trace) else generic_text
        request_count = len(
            json.loads(calibration.read_text(encoding="utf-8"))["requests"])
        _set_option(command, "--generic-warmup-trace", str(calibration))
        _set_option(command, "--warmup-requests", str(request_count))
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
    parser.add_argument("--policy-variant",
                        choices=sorted(POLICY_VARIANTS),
                        default="full_active")
    parser.add_argument("--modes", default=",".join(MODES))
    parser.add_argument("--cases", default="")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    modes = tuple(value for value in args.modes.split(",") if value)
    if not modes or any(mode not in MODES for mode in modes):
        parser.error(f"modes must be drawn from {','.join(MODES)}")
    if args.repeats <= 0:
        parser.error("repeats must be positive")
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
            command = prepare_command(entry, mode, output, args.repeats,
                                      args.generic_text, args.generic_vlm,
                                      args.backend_build_root,
                                      args.policy_variant)
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
