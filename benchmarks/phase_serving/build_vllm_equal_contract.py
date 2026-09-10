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
"""Build a vLLM command matrix from an immutable phase HTTP trace contract."""

from __future__ import annotations

import argparse
import copy
import json
import pathlib


def _replace_option(command: list[str], option: str, value: str) -> None:
    if command.count(option) != 1:
        raise ValueError(f"Expected one {option} option")
    command[command.index(option) + 1] = value


def build(phase_records: list[dict],
          template: dict,
          repeats: int,
          warmup_requests: int | None = None) -> list[dict]:
    """Project only measured trace/client fields onto one frozen vLLM setup."""
    result = []
    for record in phase_records:
        phase = record["command"]
        client = copy.deepcopy(template["client"])
        trace = phase[phase.index("--trace") + 1]
        max_in_flight = phase[phase.index("--max-in-flight") + 1]
        _replace_option(client, "--trace", trace)
        _replace_option(client, "--max-in-flight", max_in_flight)
        _replace_option(client, "--repeats", str(repeats))
        if warmup_requests is not None:
            _replace_option(client, "--warmup-requests", str(warmup_requests))
        phase_ignores_eos = "--ignore-eos" in phase
        client_ignores_eos = "--ignore-eos" in client
        if phase_ignores_eos != client_ignores_eos:
            if phase_ignores_eos:
                client.append("--ignore-eos")
            else:
                client.remove("--ignore-eos")
        result.append({
            "name": record["case"],
            "server": copy.deepcopy(template["server"]),
            "client": client,
            "health_url": template["health_url"],
        })
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase-commands", type=pathlib.Path, required=True)
    parser.add_argument("--vllm-template", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--client-repeats", type=int, default=1)
    parser.add_argument("--warmup-requests", type=int)
    args = parser.parse_args()
    if args.client_repeats <= 0:
        parser.error("client repeats must be positive")
    if args.warmup_requests is not None and args.warmup_requests < 0:
        parser.error("warmup requests must be non-negative")
    phase_records = json.loads(args.phase_commands.read_text())
    templates = json.loads(args.vllm_template.read_text())
    if not phase_records or not templates:
        raise ValueError("Input command manifests cannot be empty")
    records = build(phase_records, templates[0], args.client_repeats,
                    args.warmup_requests)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, indent=2) + "\n")
    print(
        json.dumps({
            "cases": len(records),
            "client_repeats": args.client_repeats
        }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
