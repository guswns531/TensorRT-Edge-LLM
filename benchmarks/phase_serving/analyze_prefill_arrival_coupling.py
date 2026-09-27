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
"""Compare client ingress and prefill formation in paired HTTP phase traces."""

import argparse
import collections
import csv
import gzip
import json
import pathlib
import statistics


def percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    return ordered[lower] + (ordered[min(lower + 1,
                                         len(ordered) - 1)] -
                             ordered[lower]) * (position - lower)


def load_cell(cell, measurement_epoch):
    requests_path = cell / "client/run-001/requests.csv"
    with requests_path.open(encoding="utf-8") as stream:
        requests = {
            int(row["request_id"]): row
            for row in csv.DictReader(stream)
        }
    log = cell / "gateway.log.gz"
    if log.is_file():
        stream = gzip.open(log, "rt", encoding="utf-8")
    else:
        stream = (cell / "gateway.log").open(encoding="utf-8")
    metrics = []
    events = []
    with stream:
        for line in stream:
            if line.startswith("PHASE_METRIC\t"):
                metric = json.loads(line.split("\t", 1)[1])
                if metric.get("measurement_epoch") == measurement_epoch:
                    metrics.append(metric)
            elif line.startswith("PHASE_SCHEDULER_EVENT\t"):
                events.append(json.loads(line.split("\t", 1)[1]))
    return requests, metrics, events


def prefill_summary(requests, metrics):
    prefill = [
        metric for metric in metrics if metric.get("prefill_request_ids")
    ]
    waiting = [
        (float(row["send_us"]) - float(row["scheduled_arrival_us"])) / 1000.0
        for row in requests.values()
    ]
    arrival_ttft = [
        (float(row["first_token_us"]) - float(row["scheduled_arrival_us"])) /
        1000.0 for row in requests.values()
    ]
    arrival_e2e = [
        (float(row["completed_us"]) - float(row["scheduled_arrival_us"])) /
        1000.0 for row in requests.values()
    ]
    return {
        "prefill_dispatches":
        len(prefill),
        "prefill_row_appearances":
        sum(len(x["prefill_request_ids"]) for x in prefill),
        "prefill_tokens":
        sum(x["prefill_tokens"] for x in prefill),
        "prefill_singletons":
        sum(len(x["prefill_request_ids"]) == 1 for x in prefill),
        "prefill_batch_histogram":
        dict(
            sorted(
                collections.Counter(
                    len(x["prefill_request_ids"]) for x in prefill).items())),
        "client_wait_median_ms":
        statistics.median(waiting),
        "client_wait_p95_ms":
        percentile(waiting, .95),
        "arrival_ttft_mean_ms":
        statistics.mean(arrival_ttft),
        "arrival_ttft_p95_ms":
        percentile(arrival_ttft, .95),
        "arrival_e2e_mean_ms":
        statistics.mean(arrival_e2e),
        "arrival_e2e_p95_ms":
        percentile(arrival_e2e, .95)
    }


def matching_decision(metric, events):
    start = metric["host_dispatch_start_us"] * 1000.0
    end = metric["host_completion_us"] * 1000.0
    request_ids = metric["prefill_request_ids"]
    dispatches = [
        event for event in events
        if event.get("event_kind") == "dispatch" and event.get("phase") ==
        "prefill" and event.get("request_ids") == request_ids and start -
        1.0e6 <= event.get("enqueue_host_ns", 0) <= end + 1.0e6
    ]
    if not dispatches:
        return None
    dispatch = min(dispatches,
                   key=lambda event: abs(event["enqueue_host_ns"] - start))
    return next(
        (event for event in events if event.get("event_kind") == "decision"
         and event.get("decision_id") == dispatch.get("decision_id")), None)


def analyze_pair(left_requests, left_metrics, left_events, right_requests,
                 right_metrics, right_events):
    if left_requests.keys() != right_requests.keys():
        raise ValueError("Request IDs differ")
    for request_id in left_requests:
        lhs = left_requests[request_id]
        rhs = right_requests[request_id]
        for field in ("scheduled_arrival_us", "prompt_tokens",
                      "max_output_tokens"):
            if lhs[field] != rhs[field]:
                raise ValueError("Scheduled request contract differs")
    left_prefill = [
        metric for metric in left_metrics if metric.get("prefill_request_ids")
    ]
    right_prefill = [
        metric for metric in right_metrics if metric.get("prefill_request_ids")
    ]
    first_mismatch = None
    for index, (lhs, rhs) in enumerate(zip(left_prefill, right_prefill)):
        if (lhs["prefill_request_ids"],
                lhs["prefill_tokens"]) != (rhs["prefill_request_ids"],
                                           rhs["prefill_tokens"]):
            left_decision = matching_decision(lhs, left_events)
            right_decision = matching_decision(rhs, right_events)
            involved = sorted(
                set(lhs["prefill_request_ids"])
                | set(rhs["prefill_request_ids"]))
            first_mismatch = {
                "index":
                index,
                "left_dispatch_index":
                lhs["dispatch_index"],
                "right_dispatch_index":
                rhs["dispatch_index"],
                "left_request_ids":
                lhs["prefill_request_ids"],
                "right_request_ids":
                rhs["prefill_request_ids"],
                "left_ready_prefill_ids":
                left_decision.get("ready_prefill_request_ids")
                if left_decision else None,
                "right_ready_prefill_ids":
                right_decision.get("ready_prefill_request_ids")
                if right_decision else None,
                "left_action":
                left_decision.get("action_kind") if left_decision else None,
                "right_action":
                right_decision.get("action_kind") if right_decision else None,
                "involved_sends_us": [{
                    "request_id":
                    request_id,
                    "left":
                    float(left_requests[request_id]["send_us"]),
                    "right":
                    float(right_requests[request_id]["send_us"])
                } for request_id in involved]
            }
            break
    send_deltas = [
        abs(
            float(left_requests[index]["send_us"]) -
            float(right_requests[index]["send_us"])) / 1000.0
        for index in left_requests
    ]
    return {
        "requests": len(left_requests),
        "actual_send_abs_delta_median_ms": statistics.median(send_deltas),
        "actual_send_abs_delta_p95_ms": percentile(send_deltas, .95),
        "actual_send_abs_delta_max_ms": max(send_deltas),
        "left": prefill_summary(left_requests, left_metrics),
        "right": prefill_summary(right_requests, right_metrics),
        "first_prefill_mismatch": first_mismatch
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-cell", type=pathlib.Path, required=True)
    parser.add_argument("--right-cell", type=pathlib.Path, required=True)
    parser.add_argument("--measurement-epoch", type=int, default=1)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    left = load_cell(args.left_cell, args.measurement_epoch)
    right = load_cell(args.right_cell, args.measurement_epoch)
    result = analyze_pair(*left, *right)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n",
                           encoding="utf-8")


if __name__ == "__main__":
    main()
