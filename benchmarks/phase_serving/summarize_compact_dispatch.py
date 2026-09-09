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
"""Retain TensorRT layer/tactic and input contracts without executing an engine."""
"""Compare repeated HTTP outputs under full-token and first-EOS contracts."""
"""Summarize compact phase telemetry without treating GPU sums as wall time."""

import argparse
import collections
import json
import pathlib
import statistics


def stats(values):
    """Retain negative host intervals instead of silently clipping them."""
    if not values:
        return {"count": 0}
    ordered = sorted(values)
    position = .95 * (len(ordered) - 1)
    low = int(position)
    p95 = ordered[low] + (ordered[min(low + 1,
                                      len(ordered) - 1)] -
                          ordered[low]) * (position - low)
    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "p95": p95,
        "sum": sum(values),
        "negative_count": sum(value < 0 for value in values)
    }


def analyze(path):
    """Read exactly one measurement epoch, excluding calibration records."""
    dispatches, encoders = [], []
    epochs = 0
    for line in path.read_text().splitlines():
        kind, separator, payload = line.partition("\t")
        if not separator:
            continue
        event = json.loads(payload)
        if kind == "PHASE_EPOCH" and event["kind"] == "measurement":
            epochs += 1
            epoch = event["epoch"]
            dispatches, encoders = [], []
        elif epochs and kind == "PHASE_METRIC":
            if event["measurement_epoch"] != epoch:
                raise ValueError("Dispatch epoch mismatch")
            dispatches.append(event)
        elif epochs and kind == "PHASE_ENCODER_METRIC":
            encoders.append(event)
    if epochs != 1 or not dispatches:
        raise ValueError(
            "Expected one completed measurement epoch with dispatches")
    phases = {}
    for phase in ["prefill", "decode"]:
        rows = [row for row in dispatches if row[phase + "_batch"] > 0]
        phases[phase] = {
            "dispatches":
            len(rows),
            "batch_mean":
            statistics.mean(row[phase + "_batch"]
                            for row in rows) if rows else None,
            "batch_histogram":
            dict(collections.Counter(row[phase + "_batch"] for row in rows)),
            "gpu_ms":
            stats([row[phase + "_gpu_ms"] for row in rows]),
            "host_submission_us":
            stats([
                row["host_submission_end_us"] - row["host_dispatch_start_us"]
                for row in rows
            ]),
        }
    phases["encoder"] = {
        "dispatches":
        len(encoders),
        "batch_mean":
        statistics.mean(row["batch_size"]
                        for row in encoders) if encoders else None,
        "batch_histogram":
        dict(collections.Counter(row["batch_size"] for row in encoders)),
        "gpu_ms":
        stats([row["gpu_ms"] for row in encoders]),
    }
    decode = sorted((row for row in dispatches if row["decode_batch"]),
                    key=lambda row: row["host_dispatch_start_us"])
    gaps = [
        (right["host_dispatch_start_us"] - left["host_completion_us"]) / 1000
        for left, right in zip(decode, decode[1:])
    ]
    return {
        "source":
        str(path),
        "phases":
        phases,
        "host_scheduler_decision_us":
        stats([row["host_scheduler_decision_us"] for row in dispatches]),
        "decode_host_completion_to_next_dispatch_ms":
        stats(gaps),
        "caveat":
        "Diagnostic instrumentation. GPU spans may overlap; their sum is not makespan. "
        "Host completion is observation time, not GPU completion. D gaps can contain E/P "
        "work or no ready D; they are not pure CPU overhead or proof of starvation."
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    files = sorted(args.root.glob("*/generic/*/worker-4/run-*/dispatch.jsonl"))
    if not files:
        raise ValueError("No dispatch files")
    rows = [analyze(path) for path in files]
    with args.output.open("x") as destination:
        json.dump(rows, destination, indent=2)
    print(f"Analyzed {len(rows)} measurement epochs")


if __name__ == "__main__":
    main()
