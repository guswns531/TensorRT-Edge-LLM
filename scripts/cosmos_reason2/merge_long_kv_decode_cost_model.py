#!/usr/bin/env python3
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
"""Merge isolated long-KV decode costs into a serving scheduler model."""

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--decode-cost-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--mode",
                        choices=["enqueue", "cuda_graph"],
                        default="cuda_graph")
    args = parser.parse_args()

    root = json.loads(args.base_model.read_text(encoding="utf-8"))
    if not isinstance(root.get("decode"), list) or not root["decode"]:
        parser.error("base model must contain decode cost points")

    # Preserve heterogeneous real-request points, but make their observed total
    # coverage explicit instead of leaving the runtime to infer it.
    merged: dict[tuple[int, int, int], dict[str, object]] = {}
    for point in root["decode"]:
        batch = int(point["batch_size"])
        context = int(point["max_context_length"])
        total = int(point.get("max_total_context_tokens", batch * context))
        point["max_total_context_tokens"] = total
        merged[(batch, context, total)] = point

    capacity: list[dict[str, object]] = []
    with args.decode_cost_csv.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["mode"] != args.mode:
                continue
            batch = int(row["decode_batch"])
            context = int(row["past_kv_len"])
            required = int(row["required_page_bundles"])
            available = int(row["available_page_bundles"])
            status = row["status"]
            capacity.append({
                "batch_size": batch,
                "max_context_length": context,
                "required_page_bundles": required,
                "available_page_bundles": available,
                "status": status,
            })
            if status != "measured":
                continue
            total = batch * context
            point = {
                "batch_size": batch,
                "max_context_length": context,
                "max_total_context_tokens": total,
                "samples": int(row["samples"]),
                "sample_scope": "isolated_uniform_decode",
                "median_gpu_ms": float(row["decode_median_ms"]),
                "p95_gpu_ms": float(row["decode_p95_ms"]),
            }
            # Replace only an identical coverage cell. A real mixed-request
            # point with a different total bound remains independently useful.
            merged[(batch, context, total)] = point

    root["schema_version"] = max(int(root.get("schema_version", 0)), 8)
    root["decode"] = sorted(
        merged.values(),
        key=lambda point:
        (int(point["batch_size"]), int(point["max_context_length"]),
         int(point["max_total_context_tokens"])))
    root["decode_capacity"] = capacity
    sources = list(root.get("source_files", []))
    for path in (args.base_model, args.decode_cost_csv):
        if str(path) not in sources:
            sources.append(str(path))
    root["source_files"] = sources
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(root, indent=2) + "\n",
                                encoding="utf-8")


if __name__ == "__main__":
    main()
