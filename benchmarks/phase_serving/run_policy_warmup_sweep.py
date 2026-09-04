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
"""Run fixed-budget generic calibration sweeps for one phase policy."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import run_policy_warmup_matrix


def _parse_budgets(value: str) -> tuple[int, ...]:
    budgets = tuple(int(item) for item in value.split(",") if item)
    if not budgets or any(item < 0 for item in budgets):
        raise ValueError("warmup budgets must be non-negative integers")
    if len(set(budgets)) != len(budgets):
        raise ValueError("warmup budgets must be unique")
    return budgets


def prepare_budget_command(entry: dict[str, Any], budget: int,
                           output_dir: Path, repeats: int,
                           generic_text: Path, generic_vlm: Path,
                           backend_build_root: str,
                           backend_engine_dir: str,
                           policy_variant: str,
                           backend_environment: tuple[str, ...],
                           client_max_in_flight: int) -> list[str]:
    """Materialize one exact-budget command without convergence early stop."""
    command = run_policy_warmup_matrix.prepare_command(
        entry,
        "generic",
        output_dir,
        repeats,
        generic_text,
        generic_vlm,
        backend_build_root,
        backend_engine_dir,
        policy_variant,
        backend_environment,
        client_max_in_flight,
    )
    run_policy_warmup_matrix._set_option(command, "--warmup-requests",
                                         str(budget))
    run_policy_warmup_matrix._set_option(
        command, "--phase-calibration-min-requests", str(budget))
    if budget > 0:
        calibration = generic_vlm if run_policy_warmup_matrix._is_vision_trace(
            Path(command[command.index("--trace") + 1])) else generic_text
        round_requests = len(
            json.loads(calibration.read_text(encoding="utf-8"))["requests"])
        run_policy_warmup_matrix._set_option(
            command, "--phase-calibration-round-requests",
            str(min(budget, round_requests)))
    return command


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-commands", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--generic-text", type=Path, required=True)
    parser.add_argument("--generic-vlm", type=Path, required=True)
    parser.add_argument("--warmup-budgets", default="0,424,848,1272,1696")
    parser.add_argument("--backend-build-root", default="")
    parser.add_argument("--backend-engine-dir", default="")
    parser.add_argument("--client-max-in-flight", type=int, default=0)
    parser.add_argument("--policy-variant",
                        choices=sorted(
                            run_policy_warmup_matrix.POLICY_VARIANTS),
                        default="full_active")
    parser.add_argument("--cases", default="")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--backend-env", action="append", default=[])
    args = parser.parse_args()
    try:
        budgets = _parse_budgets(args.warmup_budgets)
    except ValueError as error:
        parser.error(str(error))
    if args.repeats <= 0 or args.client_max_in_flight < 0:
        parser.error("repeats must be positive and max-in-flight non-negative")
    if any("=" not in value or not value.split("=", 1)[0]
           for value in args.backend_env):
        parser.error("backend-env values must use NAME=VALUE")

    entries = json.loads(args.base_commands.read_text(encoding="utf-8"))
    selected_cases = {item for item in args.cases.split(",") if item}
    if selected_cases:
        entries = [entry for entry in entries
                   if str(entry["case"]) in selected_cases]
    if not entries:
        parser.error("no selected base commands")

    commands = []
    for budget in budgets:
        for entry in entries:
            case = str(entry["case"])
            variant = str(entry.get("variant", "default"))
            output = args.output_dir / f"warmup-{budget}" / case / variant
            command = prepare_budget_command(
                entry, budget, output, args.repeats, args.generic_text,
                args.generic_vlm, args.backend_build_root,
                args.backend_engine_dir, args.policy_variant,
                tuple(args.backend_env), args.client_max_in_flight)
            commands.append({
                "warmup_requests": budget,
                "case": case,
                "variant": variant,
                "command": command,
            })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "commands.json").write_text(
        json.dumps(commands, indent=2) + "\n", encoding="utf-8")
    if not args.dry_run:
        for item in commands:
            print(json.dumps({key: item[key]
                              for key in ("warmup_requests", "case",
                                          "variant")}), flush=True)
            subprocess.run(item["command"], check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
