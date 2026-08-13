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
"""Choose the memory-safe Cosmos packed-prefill serving contract."""

import argparse
import json
import statistics
from pathlib import Path


def trace_stats(path: Path) -> dict[str, float | int]:
    root = json.loads(path.read_text(encoding="utf-8"))
    requests = root.get("requests", root) if isinstance(root, dict) else root
    if not isinstance(requests, list) or not requests:
        raise ValueError("trace must contain a non-empty request list")
    outputs = [
        int(request.get("max_generate_length", 0)) for request in requests
    ]
    arrivals = [
        int(request.get("arrival_offset_us", 0)) for request in requests
    ]
    span_us = max(arrivals) - min(arrivals) if len(arrivals) > 1 else 0
    offered_rps = (len(requests) - 1) * 1.0e6 / span_us if span_us > 0 else 0.0
    return {
        "request_count": len(requests),
        "output_tokens": sum(outputs),
        "output_median": statistics.median(outputs),
        "arrival_span_us": span_us,
        "offered_rps": offered_rps,
    }


def select_policy(stats: dict[str, float | int], available_mib: int,
                  graph_budget_mib: int, reserve_mib: int,
                  graph_profile_available: bool) -> tuple[str, str]:
    required_mib = graph_budget_mib + reserve_mib
    graph_safe = graph_profile_available and available_mib >= required_mib
    backlog = int(stats["request_count"]) >= 128 or float(
        stats["offered_rps"]) >= 100.0
    if graph_safe:
        return "chunk128_graph", (
            "measured 128-token graph path is fastest and the graph budget plus "
            "free-memory reserve fits")
    if backlog:
        return "chunk256_graph_off", (
            "graph headroom is insufficient and the trace has enough backlog to "
            "benefit from larger prefill chunks")
    return "chunk128_graph_off", (
        "graph headroom is insufficient and a short trace does not justify the "
        "256-token engine workspace")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--chunk128-engine", type=Path, required=True)
    parser.add_argument("--chunk256-engine", type=Path, required=True)
    parser.add_argument("--graph-profile", type=Path)
    parser.add_argument("--available-mib", type=int, required=True)
    parser.add_argument("--graph-budget-mib", type=int, default=240)
    parser.add_argument("--reserve-mib", type=int, default=256)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.available_mib, args.graph_budget_mib, args.reserve_mib) < 0:
        parser.error("memory values cannot be negative")

    stats = trace_stats(args.trace)
    profile_available = args.graph_profile is not None and args.graph_profile.is_file(
    )
    policy, reason = select_policy(stats, args.available_mib,
                                   args.graph_budget_mib, args.reserve_mib,
                                   profile_available)
    if policy == "chunk128_graph":
        engine = args.chunk128_engine
        chunk_size = 128
        cuda_graph = True
    elif policy == "chunk256_graph_off":
        engine = args.chunk256_engine
        chunk_size = 256
        cuda_graph = False
    else:
        engine = args.chunk128_engine
        chunk_size = 128
        cuda_graph = False
    result = {
        "schema_version": 1,
        "policy": policy,
        "reason": reason,
        "trace": str(args.trace),
        "trace_stats": stats,
        "memory": {
            "available_mib":
            args.available_mib,
            "graph_budget_mib":
            args.graph_budget_mib,
            "reserve_mib":
            args.reserve_mib,
            "required_graph_headroom_mib":
            args.graph_budget_mib + args.reserve_mib,
        },
        "runner": {
            "engine":
            str(engine),
            "chunk_size":
            chunk_size,
            "cuda_graph":
            cuda_graph,
            "cuda_graph_warmup_profile":
            str(args.graph_profile) if cuda_graph else None,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n",
                           encoding="utf-8")


if __name__ == "__main__":
    main()
