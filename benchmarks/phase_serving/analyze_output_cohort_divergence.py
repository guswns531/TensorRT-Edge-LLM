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
"""Locate a greedy-token split and its preceding measured decode cohorts."""

import argparse
import array
import csv
import gzip
import heapq
import json
import pathlib


def load_cell(cell, request_id, epoch):
    requests = cell / "client/run-001/requests.csv"
    with requests.open(encoding="utf-8", newline="") as stream:
        rows = {int(row["request_id"]): row for row in csv.DictReader(stream)}
    if request_id not in rows:
        raise ValueError("Diagnostic request is absent from " + str(requests))
    row = rows[request_id]
    if row["http_status"] != "200" or row.get("error"):
        raise ValueError("Diagnostic request did not complete successfully")
    tokens = [int(token) for token in row["output_token_ids"].split()]
    if len(tokens) != int(row["output_tokens"]):
        raise ValueError("Diagnostic request token capture is incomplete")
    metrics = []
    log = cell / "gateway.log"
    if not log.is_file():
        log = cell / "gateway.log.gz"
    opener = gzip.open if log.suffix == ".gz" else open
    with opener(log, "rt", encoding="utf-8") as stream:
        for line in stream:
            if line.startswith("PHASE_METRIC\t"):
                metric = json.loads(line.split("\t", 1)[1])
                if metric.get("measurement_epoch") == epoch:
                    metrics.append(metric)
    metrics.sort(key=lambda metric: metric["host_dispatch_start_us"])
    prefill = [
        metric["prefill_request_ids"] for metric in metrics
        if metric.get("prefill_request_ids")
    ]
    decode = [
        metric["decode_request_ids"] for metric in metrics
        if request_id in metric.get("decode_request_ids", [])
    ]
    if len(decode) != len(tokens) - 1:
        raise ValueError(
            "Vanilla decode iterations do not match captured tokens")
    return tokens, prefill, decode


def logit_evidence(cell, request_id, step, selected_token, compared_token_ids):
    base = cell / "logits" / f"request-{request_id}-step-{step}"
    metadata_path = base.with_suffix(".json")
    data_path = base.with_suffix(".fp32")
    if not metadata_path.exists() and not data_path.exists():
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    logits = array.array("f")
    logits.frombytes(data_path.read_bytes())
    if (metadata["step"] != step or metadata["request_id"] != request_id
            or metadata["vocab"] != len(logits)
            or metadata["selected_token"] != selected_token):
        raise ValueError("Logit metadata and request CSV disagree")
    top = heapq.nlargest(2, range(len(logits)), key=logits.__getitem__)
    if len(top) != 2 or top[0] != metadata["selected_token"]:
        raise ValueError("Captured argmax does not match selected token")
    return {
        "selected_token": top[0],
        "runner_up_token": top[1],
        "top_two_margin": logits[top[0]] - logits[top[1]],
        "compared_token_logits": {
            str(token): logits[token]
            for token in compared_token_ids
        },
        "batch_members": metadata["batch_members"],
        "sample_row": metadata["sample_row"],
    }


def first_difference(left, right):
    return next(
        (index
         for index, pair in enumerate(zip(left, right)) if pair[0] != pair[1]),
        min(len(left), len(right)))


def analyze(left_cell, right_cell, request_id, epoch):
    left_tokens, left_prefill, left_decode = load_cell(left_cell, request_id,
                                                       epoch)
    right_tokens, right_prefill, right_decode = load_cell(
        right_cell, request_id, epoch)
    if left_tokens == right_tokens:
        return {"request_id": request_id, "tokens_equal": True}
    step = first_difference(left_tokens, right_tokens)
    if step == 0 or step >= min(len(left_tokens), len(right_tokens)):
        raise ValueError("A same-length post-prefill token split is required")
    logit_tokens = [left_tokens[step], right_tokens[step]]
    return {
        "request_id":
        request_id,
        "tokens_equal":
        False,
        "first_mismatch_token":
        step,
        "left_token":
        left_tokens[step],
        "right_token":
        right_tokens[step],
        "first_prefill_membership_mismatch":
        first_difference(left_prefill, right_prefill),
        "first_decode_cohort_mismatch_for_request":
        first_difference(left_decode, right_decode),
        "left_decode_cohort_at_mismatch":
        left_decode[step - 1],
        "right_decode_cohort_at_mismatch":
        right_decode[step - 1],
        "left_logits":
        logit_evidence(left_cell, request_id, step, left_tokens[step],
                       logit_tokens),
        "right_logits":
        logit_evidence(right_cell, request_id, step, right_tokens[step],
                       logit_tokens),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-cell", type=pathlib.Path, required=True)
    parser.add_argument("--right-cell", type=pathlib.Path, required=True)
    parser.add_argument("--request-id", type=int, required=True)
    parser.add_argument("--measurement-epoch", type=int, default=1)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    result = analyze(args.left_cell, args.right_cell, args.request_id,
                     args.measurement_epoch)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n",
                           encoding="utf-8")


if __name__ == "__main__":
    main()
