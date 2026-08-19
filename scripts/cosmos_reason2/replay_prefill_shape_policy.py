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
"""Replay bounded prefill-shape decisions against recorded queue snapshots.

This is deliberately a local-decision replay, not an E2E latency simulator.
It answers which profiled (batch, chunk) shape a cost-aware policy would have
selected at each recorded prefill dispatch. Selected policies must still pass
the real TensorRT trace benchmark because a different choice changes future
queue and KV state.
"""

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


def discover(inputs: list[Path]) -> list[Path]:
    paths: list[Path] = []
    for source in inputs:
        if source.is_dir():
            paths.extend(sorted(source.rglob("requests-dispatch.csv")))
        elif source.is_file():
            paths.append(source)
    return sorted(set(paths))


def load_rows(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                row["source_file"] = str(path)
                rows.append(row)
    return rows


def select_prefill_cost(costs: list[dict[str, object]], batch: int, chunk: int,
                        past_kv: int, decode_batch: int,
                        initial: bool) -> dict[str, object] | None:
    covered = [
        point for point in costs
        if int(point["batch_size"]) == batch and int(point["chunk_length"]) >=
        chunk and int(point["max_past_kv_length"]) >= past_kv
        and int(point["max_concurrent_decode_batch_size"]) >= decode_batch
        and bool(point["initial_chunk"]) == initial
    ]
    if not covered:
        return None
    return min(covered,
               key=lambda point:
               (int(point["chunk_length"]), int(point["max_past_kv_length"]),
                int(point["max_concurrent_decode_batch_size"])))


def select_overlap_cost(costs: list[dict[str, object]], batch: int,
                        decode_batch: int, chunk: int, past_kv: int,
                        decode_context: int,
                        initial: bool) -> dict[str, object] | None:
    covered = [
        point for point in costs if int(point["prefill_batch_size"]) == batch
        and int(point["decode_batch_size"]) >= decode_batch
        and int(point["chunk_length"]) >= chunk
        and int(point["max_prefill_past_kv_length"]) >= past_kv
        and int(point["max_decode_context_length"]) >= decode_context
        and bool(point["initial_chunk"]) == initial
    ]
    if not covered:
        return None
    return min(covered,
               key=lambda point:
               (int(point["chunk_length"]), int(point["decode_batch_size"]),
                int(point["max_prefill_past_kv_length"]),
                int(point["max_decode_context_length"])))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--cost-json", type=Path, required=True)
    parser.add_argument("--chunk-candidates",
                        type=int,
                        nargs="+",
                        default=[64, 128])
    parser.add_argument("--max-prefill-batch", type=int, default=8)
    parser.add_argument("--max-decode-batch", type=int, default=64)
    parser.add_argument("--token-budget", type=int, default=1024)
    parser.add_argument("--decode-penalty-weight", type=float, default=1.0)
    parser.add_argument("--enqueue-cost-ms", type=float, default=0.0)
    parser.add_argument("--drain-backlog-tokens", type=int, default=0)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    candidates = sorted(set(args.chunk_candidates))
    if (not candidates or candidates[0] <= 0 or args.max_prefill_batch <= 0
            or args.max_decode_batch <= 0 or args.token_budget < 0
            or not math.isfinite(args.decode_penalty_weight)
            or args.decode_penalty_weight < 0.0
            or not math.isfinite(args.enqueue_cost_ms)
            or args.enqueue_cost_ms < 0.0 or args.drain_backlog_tokens < 0):
        parser.error("shape candidates and scoring parameters are invalid")
    paths = discover(args.input)
    if not paths:
        parser.error("no requests-dispatch.csv inputs found")
    model = json.loads(args.cost_json.read_text(encoding="utf-8"))
    prefill_costs = model.get("prefill", [])
    overlap_costs = model.get("overlap", [])
    if not prefill_costs:
        parser.error("cost model contains no prefill points")

    decisions: list[dict[str, object]] = []
    eligible_dispatches = 0
    covered_dispatches = 0
    for row in load_rows(paths):
        observed_batch = int(row["prefill_batch"])
        if observed_batch <= 0:
            continue
        if int(row["prefill_final_rows"]) == observed_batch:
            # The runtime keeps a completion that already fits in one turn
            # intact unless completion splitting is explicitly enabled.
            continue
        initial_rows = int(row["prefill_initial_rows"])
        continuation_rows = int(row["prefill_continuation_rows"])
        if initial_rows == observed_batch:
            initial = True
        elif continuation_rows == observed_batch:
            initial = False
        else:
            continue
        padded_tokens = int(
            row.get("prefill_padded_tokens") or row["prefill_tokens"])
        observed_chunk = math.ceil(padded_tokens / observed_batch)
        if observed_chunk < candidates[0]:
            continue
        eligible_dispatches += 1
        available_rows = max(observed_batch,
                             int(row.get("prefill_cost_lookup_rows") or 0))
        available_rows = min(available_rows, args.max_prefill_batch)
        past_kv = int(row["prefill_past_kv_max"])
        decode_batch = int(row["decode_batch"])
        prefill_remaining_tokens = int(row["prefill_remaining_tokens"])
        drain_mode = (args.drain_backlog_tokens > 0 and
                      prefill_remaining_tokens >= args.drain_backlog_tokens)
        decode_context = int(row.get("planned_decode_max_context_length") or 0)
        if decode_context <= 0 and decode_batch > 0:
            decode_context = math.ceil(
                int(row["decode_context_tokens"]) / decode_batch)

        best: dict[str, object] | None = None
        evaluated = 0
        for chunk in candidates:
            if chunk > observed_chunk:
                continue
            budget_rows = available_rows
            if args.token_budget > 0:
                budget_rows = min(budget_rows,
                                  max(1, args.token_budget // chunk))
            for batch in range(1, budget_rows + 1):
                prefill = select_prefill_cost(prefill_costs, batch, chunk,
                                              past_kv, decode_batch, initial)
                if prefill is None:
                    continue
                gpu_ms = float(prefill["p95_gpu_ms"])
                slowdown_ms = float(prefill["decode_slowdown_p95_ms"])
                if decode_batch > 0 and overlap_costs:
                    overlap = select_overlap_cost(overlap_costs, batch,
                                                  decode_batch, chunk, past_kv,
                                                  decode_context, initial)
                    if overlap is not None:
                        gpu_ms = float(overlap["prefill_p95_gpu_ms"])
                        slowdown_ms = float(overlap["decode_slowdown_p95_ms"])
                decode_pressure = min(1.0,
                                      decode_batch / args.max_decode_batch)
                effective_ms = (gpu_ms + args.decode_penalty_weight *
                                decode_pressure * slowdown_ms +
                                args.enqueue_cost_ms)
                score = batch * chunk / effective_ms
                evaluated += 1
                candidate = {
                    "selected_prefill_batch": batch,
                    "selected_chunk": chunk,
                    "predicted_prefill_p95_ms": gpu_ms,
                    "predicted_decode_slowdown_p95_ms": slowdown_ms,
                    "score_tokens_per_effective_ms": score,
                    "productive_tokens": batch * chunk,
                }
                best_score = (0.0 if best is None else float(
                    best["score_tokens_per_effective_ms"]))
                best_tokens = (0 if best is None else int(
                    best["productive_tokens"]))
                if (best is None
                        or (drain_mode and batch * chunk > best_tokens)
                        or (drain_mode and batch * chunk == best_tokens
                            and score > best_score)
                        or (not drain_mode and score > best_score)
                        or (not drain_mode and score == best_score
                            and batch * chunk > best_tokens)):
                    best = candidate
        if best is None:
            continue
        covered_dispatches += 1
        decisions.append({
            "source_file": row["source_file"],
            "dispatch_index": row["dispatch_index"],
            "initial_chunk": initial,
            "past_kv": past_kv,
            "decode_batch": decode_batch,
            "decode_context": decode_context,
            "available_prefill_rows": available_rows,
            "prefill_remaining_tokens": prefill_remaining_tokens,
            "drain_mode": drain_mode,
            "observed_prefill_batch": observed_batch,
            "observed_chunk": observed_chunk,
            "candidates_evaluated": evaluated,
            **best,
        })

    if not decisions:
        raise RuntimeError("no replay dispatch had cost coverage")
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(decisions[0]))
        writer.writeheader()
        writer.writerows(decisions)
    shape_counts: dict[str, int] = {}
    changed = 0
    for decision in decisions:
        key = f"P{decision['selected_prefill_batch']}_C{decision['selected_chunk']}"
        shape_counts[key] = shape_counts.get(key, 0) + 1
        changed += (decision["observed_prefill_batch"],
                    decision["observed_chunk"]) != (
                        decision["selected_prefill_batch"],
                        decision["selected_chunk"])
    summary = {
        "schema_version":
        1,
        "replay_kind":
        "local_dispatch_decision",
        "source_files": [str(path) for path in paths],
        "cost_json":
        str(args.cost_json),
        "chunk_candidates":
        candidates,
        "decode_penalty_weight":
        args.decode_penalty_weight,
        "max_decode_batch":
        args.max_decode_batch,
        "enqueue_cost_ms":
        args.enqueue_cost_ms,
        "drain_backlog_tokens":
        args.drain_backlog_tokens,
        "eligible_dispatches":
        eligible_dispatches,
        "covered_dispatches":
        covered_dispatches,
        "coverage":
        covered_dispatches / eligible_dispatches,
        "changed_dispatches":
        changed,
        "changed_fraction":
        changed / covered_dispatches,
        "drain_mode_dispatches":
        sum(bool(decision["drain_mode"]) for decision in decisions),
        "selected_shapes":
        dict(sorted(shape_counts.items())),
        "median_score_tokens_per_effective_ms":
        statistics.median(
            float(decision["score_tokens_per_effective_ms"])
            for decision in decisions),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2) + "\n",
                                encoding="utf-8")


if __name__ == "__main__":
    main()
