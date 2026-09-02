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
"""Run a frozen phase-serving command manifest with exact-event instrumentation."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

POLICY_ENVIRONMENTS = {
    "current": {
        "TRT_EDGELLM_CONTEXTUAL_PD": "active",
        "TRT_EDGELLM_CONTEXTUAL_EP": "active",
        "TRT_EDGELLM_CONTEXTUAL_ED": "active",
        "TRT_EDGELLM_COMPLETION_CONFORMAL": "0",
        "TRT_EDGELLM_COMPLETION_CONFORMAL_ACTIVE": "0",
    },
    "serial": {
        "TRT_EDGELLM_CONTEXTUAL_PD": "shadow",
        "TRT_EDGELLM_CONTEXTUAL_EP": "shadow",
        "TRT_EDGELLM_CONTEXTUAL_ED": "shadow",
        "TRT_EDGELLM_EXPERIMENTAL_OVERLAP_PERCENT": "0",
        "TRT_EDGELLM_EXPERIMENTAL_ENCODER_PREFILL_OVERLAP_PERCENT": "0",
        "TRT_EDGELLM_EXPERIMENTAL_ENCODER_DECODE_OVERLAP_PERCENT": "0",
    },
    "always_overlap": {
        "TRT_EDGELLM_CONTEXTUAL_PD": "shadow",
        "TRT_EDGELLM_CONTEXTUAL_EP": "shadow",
        "TRT_EDGELLM_CONTEXTUAL_ED": "shadow",
        "TRT_EDGELLM_EXPERIMENTAL_OVERLAP_PERCENT": "100",
        "TRT_EDGELLM_EXPERIMENTAL_ENCODER_PREFILL_OVERLAP_PERCENT": "100",
        "TRT_EDGELLM_EXPERIMENTAL_ENCODER_DECODE_OVERLAP_PERCENT": "100",
    },
    "immediate_cost": {
        "TRT_EDGELLM_CONTEXTUAL_PD": "disabled",
        "TRT_EDGELLM_CONTEXTUAL_EP": "disabled",
        "TRT_EDGELLM_CONTEXTUAL_ED": "disabled",
        "TRT_EDGELLM_COMPLETION_CONFORMAL": "0",
        "TRT_EDGELLM_COMPLETION_CONFORMAL_ACTIVE": "0",
    },
    "completion_shadow": {
        "TRT_EDGELLM_CONTEXTUAL_PD": "active",
        "TRT_EDGELLM_CONTEXTUAL_EP": "active",
        "TRT_EDGELLM_CONTEXTUAL_ED": "active",
        "TRT_EDGELLM_COMPLETION_CONFORMAL": "1",
        "TRT_EDGELLM_COMPLETION_CONFORMAL_ACTIVE": "0",
    },
    "completion_active": {
        "TRT_EDGELLM_CONTEXTUAL_PD": "active",
        "TRT_EDGELLM_CONTEXTUAL_EP": "active",
        "TRT_EDGELLM_CONTEXTUAL_ED": "active",
        "TRT_EDGELLM_COMPLETION_CONFORMAL": "1",
        "TRT_EDGELLM_COMPLETION_CONFORMAL_ACTIVE": "1",
    },
    "completion_no_uncertainty": {
        "TRT_EDGELLM_CONTEXTUAL_PD": "active",
        "TRT_EDGELLM_CONTEXTUAL_EP": "active",
        "TRT_EDGELLM_CONTEXTUAL_ED": "active",
        "TRT_EDGELLM_COMPLETION_CONFORMAL": "1",
        "TRT_EDGELLM_COMPLETION_CONFORMAL_ACTIVE": "1",
        "TRT_EDGELLM_COMPLETION_AUTHORITY_USES_UNCERTAINTY": "0",
    },
    "completion_no_residual": {
        "TRT_EDGELLM_CONTEXTUAL_PD": "active",
        "TRT_EDGELLM_CONTEXTUAL_EP": "active",
        "TRT_EDGELLM_CONTEXTUAL_ED": "active",
        "TRT_EDGELLM_COMPLETION_CONFORMAL": "1",
        "TRT_EDGELLM_COMPLETION_CONFORMAL_ACTIVE": "1",
        "TRT_EDGELLM_COMPLETION_USE_RESIDUAL_FEATURES": "0",
    },
    "completion_new_work_only": {
        "TRT_EDGELLM_CONTEXTUAL_PD": "active",
        "TRT_EDGELLM_CONTEXTUAL_EP": "active",
        "TRT_EDGELLM_CONTEXTUAL_ED": "active",
        "TRT_EDGELLM_COMPLETION_CONFORMAL": "1",
        "TRT_EDGELLM_COMPLETION_CONFORMAL_ACTIVE": "1",
        "TRT_EDGELLM_COMPLETION_AUTHORITY_PREDICTS_INCUMBENT": "0",
    },
    "completion_h2": {
        "TRT_EDGELLM_CONTEXTUAL_PD": "active",
        "TRT_EDGELLM_CONTEXTUAL_EP": "active",
        "TRT_EDGELLM_CONTEXTUAL_ED": "active",
        "TRT_EDGELLM_COMPLETION_CONFORMAL": "1",
        "TRT_EDGELLM_COMPLETION_CONFORMAL_ACTIVE": "1",
        "TRT_EDGELLM_ENABLE_GLOBAL_FORMATION_AWARE": "1",
    },
    "myopic": {
        "TRT_EDGELLM_GLOBAL_SAFE_PROBE_SLACK_MULTIPLIER": "0",
    },
    "legacy": {
        "TRT_EDGELLM_GLOBAL_LEGACY_COMPATIBILITY": "1",
        "TRT_EDGELLM_GLOBAL_SAFE_PROBE_SLACK_MULTIPLIER": "0",
    },
    "h2": {
        "TRT_EDGELLM_ENABLE_GLOBAL_FORMATION_AWARE": "1",
        "TRT_EDGELLM_GLOBAL_SAFE_PROBE_SLACK_MULTIPLIER": "0",
    },
    "safe_probe": {
        "TRT_EDGELLM_DISABLE_GLOBAL_OVERLAP_WARMUP": "1",
        "TRT_EDGELLM_GLOBAL_OVERLAP_MIN_SAMPLES": "1",
        "TRT_EDGELLM_GLOBAL_SAFE_PROBE_SLACK_MULTIPLIER": "1",
        "TRT_EDGELLM_GLOBAL_SAFE_PROBE_INTERVAL": "1",
    },
    # M6 learns from the exact actions selected by the unchanged myopic
    # policy. All contextual heads are shadow-only and therefore have no
    # authority over dispatch ordering.
    "m6_shadow": {
        "TRT_EDGELLM_GLOBAL_SAFE_PROBE_SLACK_MULTIPLIER": "0",
        "TRT_EDGELLM_CONTEXTUAL_PD": "shadow",
        "TRT_EDGELLM_CONTEXTUAL_EP": "shadow",
        "TRT_EDGELLM_CONTEXTUAL_ED": "shadow",
        "TRT_EDGELLM_CONTEXTUAL_PD_CONFIDENCE_BETA": "1.96",
        "TRT_EDGELLM_CONTEXTUAL_EP_CONFIDENCE_BETA": "1.96",
        "TRT_EDGELLM_CONTEXTUAL_ED_CONFIDENCE_BETA": "1.96",
    },
    # Counterfactual-label collection is deliberately separate from the
    # production-equivalent shadow survey. It authorizes only the existing
    # bounded safe probes while every learned head remains non-authoritative.
    "m6_shadow_probe": {
        "TRT_EDGELLM_DISABLE_GLOBAL_OVERLAP_WARMUP": "1",
        "TRT_EDGELLM_GLOBAL_OVERLAP_MIN_SAMPLES": "1",
        "TRT_EDGELLM_GLOBAL_SAFE_PROBE_SLACK_MULTIPLIER": "1",
        "TRT_EDGELLM_GLOBAL_SAFE_PROBE_INTERVAL": "1",
        "TRT_EDGELLM_CONTEXTUAL_PD": "shadow",
        "TRT_EDGELLM_CONTEXTUAL_EP": "shadow",
        "TRT_EDGELLM_CONTEXTUAL_ED": "shadow",
        "TRT_EDGELLM_CONTEXTUAL_PD_CONFIDENCE_BETA": "1.96",
        "TRT_EDGELLM_CONTEXTUAL_EP_CONFIDENCE_BETA": "1.96",
        "TRT_EDGELLM_CONTEXTUAL_ED_CONFIDENCE_BETA": "1.96",
    },
}

POLICY_ENVIRONMENT_NAMES = {
    "TRT_EDGELLM_CONTEXTUAL_PD",
    "TRT_EDGELLM_CONTEXTUAL_EP",
    "TRT_EDGELLM_CONTEXTUAL_ED",
    "TRT_EDGELLM_COMPLETION_CONFORMAL",
    "TRT_EDGELLM_COMPLETION_CONFORMAL_ACTIVE",
    "TRT_EDGELLM_COMPLETION_AUTHORITY_USES_UNCERTAINTY",
    "TRT_EDGELLM_COMPLETION_USE_RESIDUAL_FEATURES",
    "TRT_EDGELLM_COMPLETION_AUTHORITY_PREDICTS_INCUMBENT",
    "TRT_EDGELLM_EXPERIMENTAL_OVERLAP_PERCENT",
    "TRT_EDGELLM_EXPERIMENTAL_ENCODER_PREFILL_OVERLAP_PERCENT",
    "TRT_EDGELLM_EXPERIMENTAL_ENCODER_DECODE_OVERLAP_PERCENT",
    "TRT_EDGELLM_ENABLE_GLOBAL_FORMATION_AWARE",
} | {
    name
    for environment in POLICY_ENVIRONMENTS.values()
    for name in environment
}


def _replace_option(command: list[str], option: str, value: str) -> None:
    index = command.index(option)
    command[index + 1] = value


def _remove_docker_environment(command: list[str], names: set[str]) -> None:
    index = 0
    while index + 1 < len(command):
        if command[index] == "-e":
            name = command[index + 1].split("=", 1)[0]
            if name in names:
                del command[index:index + 2]
                continue
        index += 1


def _set_docker_environment(command: list[str], name: str, value: str) -> None:
    _remove_docker_environment(command, {name})
    image_index = next(index for index, token in enumerate(command)
                       if token.startswith("nvcr.io/nvidia/tensorrt:"))
    command[image_index:image_index] = ["-e", f"{name}={value}"]


def materialize_command(entry: dict[str, Any],
                        policy: str,
                        output_root: Path,
                        host_workspace: Path,
                        container_workspace: Path,
                        plugin_path: str,
                        binary_path: str,
                        matrix_repeat: int = 1) -> dict[str, Any]:
    """Return one event-enabled, one-repeat command without changing its trace contract."""
    if policy not in POLICY_ENVIRONMENTS:
        raise ValueError(f"unknown policy: {policy}")
    case = str(entry["case"])
    variant = str(entry.get("variant", "default"))
    command = [str(token) for token in entry["command"]]
    output_dir = output_root / case / variant
    _replace_option(command, "--output-dir", str(output_dir))
    _replace_option(command, "--repeats", "1")
    trace_index = command.index("--trace") + 1
    trace_path = Path(command[trace_index])
    if not trace_path.is_absolute():
        command[trace_index] = str(host_workspace / trace_path)

    _remove_docker_environment(command, POLICY_ENVIRONMENT_NAMES)
    _set_docker_environment(command, "TRT_EDGELLM_EMIT_PHASE_METRICS", "1")
    relative_output = output_dir.resolve().relative_to(
        host_workspace.resolve())
    activity_prefix = container_workspace / relative_output / "activity" / "run-001"
    _set_docker_environment(command, "TRT_EDGELLM_PHASE_ACTIVITY_PREFIX",
                            str(activity_prefix))
    _set_docker_environment(command, "TRT_EDGELLM_PHASE_RUN_ID",
                            f"oracle-h1-{policy}-{case}-r{matrix_repeat:03d}")
    _set_docker_environment(command, "EDGELLM_PLUGIN_PATH", plugin_path)
    _set_docker_environment(
        command, "LD_LIBRARY_PATH",
        "/opt/tensorrt/lib:/usr/local/cuda/lib64:" +
        str(Path(binary_path).parent))
    for name, value in POLICY_ENVIRONMENTS[policy].items():
        _set_docker_environment(command, name, value)
    binary_index = next(index for index, token in enumerate(command)
                        if token.endswith("/llm_phase_context_smoke"))
    command[binary_index] = binary_path
    return {
        "case": case,
        "variant": variant,
        "policy": policy,
        "matrix_repeat": matrix_repeat,
        "output_dir": str(output_dir),
        "command": command,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--policy",
                        choices=sorted(POLICY_ENVIRONMENTS),
                        required=True)
    parser.add_argument("--host-workspace", type=Path, required=True)
    parser.add_argument("--container-workspace",
                        type=Path,
                        default=Path("/workspace"))
    parser.add_argument("--plugin-path", required=True)
    parser.add_argument("--binary-path", required=True)
    parser.add_argument("--case", action="append", dest="cases")
    parser.add_argument(
        "--deduplicate-cases",
        action="store_true",
        help="retain only the first source entry for each case/variant pair")
    parser.add_argument("--matrix-repeats", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        entries = json.loads(args.source_manifest.read_text(encoding="utf-8"))
        if not isinstance(entries, list):
            raise ValueError("source command manifest must be a list")
        if args.matrix_repeats < 1:
            raise ValueError("matrix repeats must be positive")
        requested = set(args.cases or [])
        selected = [
            entry for entry in entries
            if not requested or str(entry.get("case")) in requested
        ]
        if args.deduplicate_cases:
            unique = []
            seen = set()
            for entry in selected:
                key = (str(entry.get("case")),
                       str(entry.get("variant", "default")))
                if key in seen:
                    continue
                seen.add(key)
                unique.append(entry)
            selected = unique
        if requested - {str(entry.get("case")) for entry in selected}:
            raise ValueError("requested case is absent from source manifest")
        commands = []
        for matrix_repeat in range(1, args.matrix_repeats + 1):
            repeat_root = args.output_dir
            if args.matrix_repeats > 1:
                repeat_root /= f"repeat-{matrix_repeat:03d}"
            commands.extend(
                materialize_command(
                    entry, args.policy, repeat_root, args.host_workspace,
                    args.container_workspace, args.plugin_path,
                    args.binary_path, matrix_repeat) for entry in selected)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        command_manifest = args.output_dir / "commands.json"
        command_manifest.write_text(json.dumps(commands, indent=2) + "\n",
                                    encoding="utf-8")
        for index, item in enumerate(commands, start=1):
            print(
                f"[{index}/{len(commands)}] {item['case']} "
                f"({args.policy}, repeat {item['matrix_repeat']})",
                flush=True)
            if not args.dry_run:
                subprocess.run(item["command"], check=True)
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError,
            json.JSONDecodeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
