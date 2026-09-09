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

import argparse
import csv
import json
import pathlib


def through_eos(tokens, eos_ids):
    """Keep the first EOS token; retain a length-limited sequence unchanged."""
    for index, token in enumerate(tokens):
        if token in eos_ids:
            return tokens[:index + 1]
    return tokens


def read_run(path):
    """Read a successful request set without silently discarding duplicates."""
    result = {}
    with path.open() as source:
        for row in csv.DictReader(source):
            request = row["request_id"]
            if request in result:
                raise ValueError(f"Duplicate request {request}: {path}")
            if row["http_status"] != "200" or row.get("error"):
                raise ValueError(f"Failed request {request}: {path}")
            tokens = [int(value) for value in row["output_token_ids"].split()]
            if len(tokens) != int(row["output_tokens"]):
                raise ValueError(f"Incomplete token capture: {path}")
            result[request] = tokens
    if not result:
        raise ValueError(f"No requests: {path}")
    return result


def compare(reference, candidate, eos_ids):
    """Report identity without treating post-EOS divergence as a strict pass."""
    if reference.keys() != candidate.keys():
        raise ValueError("Request membership differs")
    differences = []
    for request, expected in reference.items():
        actual = candidate[request]
        if actual != expected:
            mismatch = next((i for i, pair in enumerate(zip(expected, actual))
                             if pair[0] != pair[1]),
                            min(len(expected), len(actual)))
            differences.append({
                "request_id":
                request,
                "first_mismatch_token":
                mismatch,
                "through_eos_equal":
                through_eos(expected, eos_ids) == through_eos(actual, eos_ids),
            })
    return {
        "requests":
        len(reference),
        "full_equal":
        len(reference) - len(differences),
        "through_eos_equal":
        len(reference) - sum(not row["through_eos_equal"]
                             for row in differences),
        "differences":
        differences,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=pathlib.Path, required=True)
    parser.add_argument("--candidate", type=pathlib.Path, required=True)
    parser.add_argument("--eos-id", type=int, action="append", required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    reference_files = sorted(
        args.reference.glob("run-*/client/run-001/requests.csv"))
    candidate_files = sorted(
        args.candidate.glob("run-*/client/run-001/requests.csv"))
    if not reference_files or not candidate_files:
        raise ValueError("Both directories need completed request CSVs")
    reference = read_run(reference_files[0])
    rows = []
    for label, files in [("reference", reference_files),
                         ("candidate", candidate_files)]:
        for path in files:
            rows.append({
                "side": label,
                "source": str(path),
                **compare(reference, read_run(path), set(args.eos_id))
            })
    report = {
        "reference":
        str(reference_files[0]),
        "eos_ids":
        args.eos_id,
        "runs":
        rows,
        "note":
        "Through-EOS comparison is diagnostic, not an actual early-termination performance run."
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as destination:
        json.dump(report, destination, indent=2)
    print(
        json.dumps({
            "runs":
            len(rows),
            "full_identity":
            all(row["full_equal"] == row["requests"] for row in rows),
            "through_eos_identity":
            all(row["through_eos_equal"] == row["requests"] for row in rows)
        }))


if __name__ == "__main__":
    main()
