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

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Decompose request-weighted decode cycles from retained HTTP phase timelines."""

import argparse
import collections
import gzip
import hashlib
import json
import pathlib
import statistics


def distribution(values):
    """Report interpolated p95 and mean, without pooling across separate runs."""
    if not values:
        return {"count": 0, "mean": None, "p95": None}
    ordered = sorted(values)
    position = (len(ordered) - 1) * 0.95
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return {
        "count":
        len(values),
        "mean":
        statistics.mean(values),
        "p95":
        ordered[lower] + (position - lower) * (ordered[upper] - ordered[lower])
    }


def analyze(log):
    """Match host timeline stages by request; distinguish them from CUDA kernel timings."""
    epoch = 0
    waiting = {}
    running = {}
    sampling = {}
    sampling_submitted = {}
    sampling_observed = {}
    sampling_collected = {}
    vision = set()
    per_request = collections.defaultdict(
        lambda: collections.defaultdict(list))
    phases = collections.defaultdict(list)
    blocks = 0
    quanta = []
    unmatched = collections.Counter()
    opener = gzip.open if log.suffix == ".gz" else open
    with opener(log, "rt") as stream:
        for line in stream:
            if "PHASE_EPOCH\t" in line:
                epoch = json.loads(line.split("PHASE_EPOCH\t", 1)[1])["epoch"]
            if epoch != 1:
                continue
            if "PHASE_METRIC\t" in line:
                row = json.loads(line.split("PHASE_METRIC\t", 1)[1])
                for phase in ("prefill", "decode"):
                    if row.get(phase + "_batch", 0) > 0:
                        phases[phase].append(
                            (row[phase + "_batch"], row[phase + "_gpu_ms"]))
                blocks = max(blocks,
                             row.get("vision_admission_service_blocks", 0))
                if "vision_admission_decode_age_quanta" in row:
                    quanta.append(row["vision_admission_decode_age_quanta"])
            if "PHASE_TIMELINE\t" not in line:
                continue
            row = json.loads(line.split("PHASE_TIMELINE\t", 1)[1])
            request, stage, time = row["request_index"], row["stage"], row[
                "timestamp_us"]
            if stage == "vision_queued":
                vision.add(request)
            if stage == "decode_ready":
                waiting[request] = time
            elif stage == "decode_start":
                if request in waiting:
                    per_request[request]["ready_wait_ms"].append(
                        (time - waiting.pop(request)) / 1000)
                else:
                    unmatched[stage] += 1
                running[request] = time
            elif stage == "decode_done":
                if request in running:
                    per_request[request]["host_execution_ms"].append(
                        (time - running.pop(request)) / 1000)
                else:
                    unmatched[stage] += 1
                sampling[request] = time
                if request in sampling_submitted:
                    per_request[request]["done_to_sampling_submit_ms"].append(
                        (sampling_submitted[request] - time) / 1000)
            elif stage == "decode_sampling_submit":
                sampling_submitted[request] = time
                if request in sampling:
                    per_request[request]["done_to_sampling_submit_ms"].append(
                        (time - sampling[request]) / 1000)
            elif stage == "decode_sampling_ready":
                if request in sampling:
                    per_request[request][
                        "done_to_sampling_observed_ms"].append(
                            (time - sampling[request]) / 1000)
                if request in sampling_submitted:
                    per_request[request]["sampling_submit_to_ready_ms"].append(
                        (time - sampling_submitted.pop(request)) / 1000)
                sampling_observed[request] = time
            elif stage == "decode_sampling_collected":
                if request in sampling_observed:
                    per_request[request]["sampling_collect_ms"].append(
                        (time - sampling_observed.pop(request)) / 1000)
                sampling_collected[request] = time
            elif stage == "decode_token_committed":
                if request in sampling:
                    per_request[request]["completion_to_commit_ms"].append(
                        (time - sampling.pop(request)) / 1000)
                else:
                    unmatched[stage] += 1
                if request in sampling_collected:
                    per_request[request]["collect_to_commit_ms"].append(
                        (time - sampling_collected.pop(request)) / 1000)
    classes = {}
    for label in ("all", "text", "vision"):
        requests = [
            r for r in per_request
            if label == "all" or ((r in vision) == (label == "vision"))
        ]
        classes[label] = {"requests": len(requests)}
        for component in ("ready_wait_ms", "host_execution_ms",
                          "completion_to_commit_ms",
                          "done_to_sampling_submit_ms",
                          "sampling_submit_to_ready_ms",
                          "done_to_sampling_observed_ms",
                          "sampling_collect_ms", "collect_to_commit_ms"):
            classes[label][component] = distribution([
                statistics.mean(per_request[r][component]) for r in requests
                if per_request[r][component]
            ])
    return {
        "log": str(log),
        "log_sha256": hashlib.sha256(log.read_bytes()).hexdigest(),
        "per_request_mean_components": classes,
        "phase_gpu": {
            p: {
                "dispatches": len(rows),
                "mean_batch": statistics.mean(b for b, _ in rows),
                "total_ms": sum(t for _, t in rows),
                "mean_ms": statistics.mean(t for _, t in rows)
            }
            for p, rows in phases.items()
        },
        "service_blocks": blocks,
        "sampled_decode_age_quanta": distribution(quanta),
        "unmatched_stages": dict(unmatched),
        "outstanding_at_end": {
            "ready": len(waiting),
            "running": len(running),
            "sampling": len(sampling),
            "sampling_submitted": len(sampling_submitted)
        }
    }


def main():
    """Retain stage decomposition alongside the campaign's immutable manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=pathlib.Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.result_root / "manifest.json").read_text())
    results = {}
    for cell in manifest["completed"]:
        path = pathlib.Path(cell["cell"]) / "run-001/gateway.log"
        if not path.exists():
            path = path.with_suffix(path.suffix + ".gz")
        key = "/".join(
            str(cell[k]) for k in ("model", "workload", "variant", "repeat"))
        results[key] = analyze(path)
    (args.result_root / "decode-service-analysis.json").write_text(
        json.dumps(
            {
                "script_sha256":
                hashlib.sha256(pathlib.Path(
                    __file__).read_bytes()).hexdigest(),
                "note":
                "notes/303-service-admission-cause-and-experiment-20260914.md",
                "cells":
                results
            },
            indent=2) + "\n")


if __name__ == "__main__":
    main()
