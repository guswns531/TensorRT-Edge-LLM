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
"""Build a reproducible real-request trace with changing arrival load."""

import argparse
import copy
import json
import random
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LoadPhase:
    name: str
    request_count: int
    arrival_rate: float
    gap_ms: float


def parse_phase(value: str) -> LoadPhase:
    """Parse NAME:REQUESTS:REQUESTS_PER_SECOND:GAP_MS."""
    fields = value.split(":")
    if len(fields) != 4:
        raise argparse.ArgumentTypeError(
            "phase must be NAME:REQUESTS:REQUESTS_PER_SECOND:GAP_MS")
    name = fields[0]
    try:
        request_count = int(fields[1])
        arrival_rate = float(fields[2])
        gap_ms = float(fields[3])
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "phase contains a non-numeric value") from error
    if not name or request_count <= 0 or arrival_rate <= 0.0 or gap_ms < 0.0:
        raise argparse.ArgumentTypeError(
            "phase values are outside the supported range")
    return LoadPhase(name, request_count, arrival_rate, gap_ms)


def materialize(source: dict, phases: list[LoadPhase], seed: int) -> dict:
    """Return a trace that cycles source requests across piecewise loads."""
    source_requests = source.get("requests", [])
    if not source_requests:
        raise ValueError("source trace contains no requests")
    generator = random.Random(seed)
    requests = []
    phase_metadata = []
    cursor_us = 0
    source_index = 0
    for phase_index, phase in enumerate(phases):
        cursor_us += round(phase.gap_ms * 1000.0)
        first_request_index = len(requests)
        start_us = cursor_us
        for request_index in range(phase.request_count):
            if request_index > 0:
                cursor_us += round(
                    generator.expovariate(phase.arrival_rate) * 1_000_000.0)
            request = copy.deepcopy(source_requests[source_index %
                                                    len(source_requests)])
            request["arrival_offset_us"] = cursor_us
            requests.append(request)
            source_index += 1
        phase_metadata.append({
            "name": phase.name,
            "phase_index": phase_index,
            "first_request_index": first_request_index,
            "request_count": phase.request_count,
            "arrival_rate": phase.arrival_rate,
            "start_offset_us": start_us,
            "last_arrival_offset_us": cursor_us,
        })
    result = copy.deepcopy(source)
    result["requests"] = requests
    result["load_phases"] = phase_metadata
    result["load_seed"] = seed
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--phase",
                        action="append",
                        type=parse_phase,
                        required=True)
    args = parser.parse_args()

    source = json.loads(args.source.read_text(encoding="utf-8"))
    result = materialize(source, args.phase, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n",
                           encoding="utf-8")


if __name__ == "__main__":
    main()
