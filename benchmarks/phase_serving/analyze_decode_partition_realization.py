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
"""Compare measured-service decode partitions with the async dispatch trajectory."""

import argparse
import collections
import gzip
import itertools
import json
import pathlib
import statistics


def load_records(path, measurement_epoch=None):
    """Read full phase metrics and token-commit timeline from a gateway log."""
    opener = gzip.open if path.suffix == ".gz" else open
    metrics = []
    commits = collections.defaultdict(list)
    with opener(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if line.startswith("PHASE_METRIC\t"):
                metric = json.loads(line.split("\t", 1)[1])
                if measurement_epoch is None or metric.get(
                        "measurement_epoch") == measurement_epoch:
                    metrics.append(metric)
            elif line.startswith("PHASE_TIMELINE\t"):
                event = json.loads(line.split("\t", 1)[1])
                if event["stage"] == "decode_token_committed":
                    commits[event["request_index"]].append(
                        event["timestamp_us"])
    return metrics, commits


def analyze(metrics, commits):
    """Evaluate one-token progress for the frontier selected by each split DP."""
    if not metrics or not any("predicted_decode_partition" in metric
                              for metric in metrics):
        raise ValueError(
            "Full decode partition diagnostic metrics are required")
    metrics = sorted(
        metrics,
        key=lambda metric:
        (metric["host_dispatch_start_us"], metric["dispatch_index"]))
    results = []
    covered_decisions = 0
    service_density_agreements = 0
    for start, metric in enumerate(metrics):
        partition = metric.get("predicted_decode_partition", [])
        frontier = metric.get("predicted_decode_frontier_ids", [])
        lengths = metric.get("predicted_decode_frontier_lengths", [])
        candidates = metric.get("predicted_decode_candidates", [])
        if partition and candidates:
            dense = next(
                (item
                 for item in candidates if item["batch"] == len(frontier)),
                None)
            known = [
                item for item in candidates
                if item["service_samples"] > 0 and item["gpu_samples"] > 0
                and item["selection_service_ms"] > 0
            ]
            if dense in known:
                covered_decisions += 1
                density_choice = min(known,
                                     key=lambda item:
                                     (item["selection_service_ms"] / item[
                                         "batch"], -item["batch"]))
                service_density_agreements += density_choice[
                    "batch"] == partition[0]
        if len(partition) < 2:
            continue
        if sum(partition) != len(frontier) or len(frontier) != len(lengths):
            raise ValueError("Invalid predicted decode frontier")
        if len(set(frontier)) != len(frontier):
            raise ValueError("Duplicate request in predicted decode frontier")
        planned = []
        offset = 0
        for batch in partition:
            planned.append(set(frontier[offset:offset + batch]))
            offset += batch
        frontier_set = set(frontier)
        seen = set()
        actual = []
        duplicates = set()
        outsiders = set()
        duplicate_dispatch_rows = 0
        actual_dispatch_rows = 0
        decode_window = itertools.islice(
            (following for following in metrics[start:]
             if following.get("decode_request_ids")),
            len(frontier) + 4)
        for following in decode_window:
            ids = following["decode_request_ids"]
            current = set(ids)
            actual_dispatch_rows += len(ids)
            duplicate_dispatch_rows += len(current & frontier_set & seen)
            duplicates.update((current & frontier_set) & seen)
            outsiders.update(current - frontier_set)
            seen.update(current & frontier_set)
            actual.append(current)
            if seen == frontier_set:
                break
        start_us = metric["host_dispatch_start_us"]
        committed = []
        for request_id in frontier:
            future = next(
                (time
                 for time in commits.get(request_id, []) if time >= start_us),
                None)
            if future is None:
                break
            committed.append(future)
        result = {
            "dispatch_index":
            metric["dispatch_index"],
            "frontier_rows":
            len(frontier),
            "partition":
            partition,
            "predicted_service_ms":
            metric.get("predicted_decode_drain_service_ms", 0.0),
            "first_batch_matches":
            bool(actual) and actual[0] == planned[0],
            "same_frontier_realized":
            len(actual) == len(planned)
            and all(got == expected for got, expected in zip(actual, planned)),
            "realized_batches": [len(group) for group in actual],
            "actual_dispatch_rows":
            actual_dispatch_rows,
            "duplicate_dispatch_rows":
            duplicate_dispatch_rows,
            "duplicate_before_drain":
            bool(duplicates),
            "outside_frontier_before_drain":
            bool(outsiders),
            "all_frontier_rows_dispatched":
            seen == frontier_set,
            "all_frontier_rows_committed":
            len(committed) == len(frontier),
        }
        if len(committed) == len(frontier):
            horizon = max(committed)
            result["observed_commit_horizon_ms"] = (horizon -
                                                    start_us) / 1000.0
            result["extra_committed_tokens"] = sum(
                start_us <= time <= horizon for request_id in frontier
                for time in commits.get(request_id, [])) - len(frontier)
        results.append(result)
    complete = [r for r in results if r["all_frontier_rows_committed"]]
    matched = [r for r in complete if r["same_frontier_realized"]]
    return {
        "opportunities":
        len(results),
        "first_batch_matches":
        sum(r["first_batch_matches"] for r in results),
        "same_frontier_realized":
        sum(r["same_frontier_realized"] for r in results),
        "duplicate_before_drain":
        sum(r["duplicate_before_drain"] for r in results),
        "outside_frontier_before_drain":
        sum(r["outside_frontier_before_drain"] for r in results),
        "complete_commit_horizons":
        len(complete),
        "complete_commit_horizon_mean_ms":
        statistics.mean(r["observed_commit_horizon_ms"]
                        for r in complete) if complete else None,
        "opportunities_with_extra_commits":
        sum(r.get("extra_committed_tokens", 0) > 0 for r in results),
        "covered_service_density_decisions":
        covered_decisions,
        "service_density_first_batch_agreements":
        service_density_agreements,
        "matched_commit_horizon_mean_ms":
        statistics.mean(r["observed_commit_horizon_ms"]
                        for r in matched) if matched else None,
        "matched_predicted_service_mean_ms":
        statistics.mean(r["predicted_service_ms"]
                        for r in matched) if matched else None,
        "examples":
        results[:20],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gateway_log", type=pathlib.Path)
    parser.add_argument("--measurement-epoch", type=int, default=1)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()
    report = analyze(*load_records(args.gateway_log, args.measurement_epoch))
    serialized = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(serialized)
    else:
        print(serialized, end="")


if __name__ == "__main__":
    main()
