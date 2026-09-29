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
"""Build a zero-shot MMLU serving trace and score retained serving outputs against it."""

import argparse
import csv
import json
import pathlib
import re
import sys

LETTERS = "ABCD"


def build(args):
    """Write a phase-serving trace plus its answer key from the cais/mmlu parquet split."""
    import pyarrow.parquet as parquet

    table = parquet.read_table(args.parquet).to_pylist()
    if args.limit:
        table = table[:args.limit]
    requests, answers = [], []
    skipped = 0
    for index, row in enumerate(table):
        subject = row["subject"].replace("_", " ")
        prompt = (
            f"The following is a multiple choice question about {subject}.\n\n"
            + row["question"] +
            "".join(f"\n{LETTERS[i]}. {choice}"
                    for i, choice in enumerate(row["choices"])) +
            "\n\nAnswer with only the letter of the correct option.")
        if args.max_prompt_chars and len(prompt) > args.max_prompt_chars:
            skipped += 1
            continue
        requests.append({
            "messages": [{
                "role": "user",
                "content": prompt
            }],
            "max_generate_length": args.max_tokens,
            "arrival_offset_us": 0,
            "source_request_index": index,
            "wave_index": 0,
        })
        answers.append({
            "request_id": len(answers),
            "source_index": index,
            "subject": row["subject"],
            "answer": LETTERS[int(row["answer"])]
        })
    trace = {
        "batch_size": 1,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 1,
        "max_generate_length": args.max_tokens,
        "requests": requests,
        "workload": "mmlu-zero-shot",
        "source_workload": str(args.parquet),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(trace))
    args.answers.write_text(json.dumps(answers))
    print(
        f"{len(requests)} requests -> {args.output} ({skipped} over the prompt limit skipped)"
    )


def predicted_letter(text):
    """First standalone option letter in the generated prefix; None if the output names no option."""
    match = re.search(r"\b([ABCD])\b", text or "")
    return match.group(1) if match else None


def score(args):
    """Score each retained requests.csv and report per-pair agreement of predictions."""
    csv.field_size_limit(sys.maxsize)
    answers = {
        str(row["request_id"]): row["answer"]
        for row in json.loads(args.answers.read_text())
    }
    runs = {}
    for path in args.requests:
        rows = {r["request_id"]: r for r in csv.DictReader(path.open())}
        predictions = {
            rid: predicted_letter(r["output_prefix"])
            for rid, r in rows.items()
        }
        correct = sum(
            predictions.get(rid) == answer for rid, answer in answers.items())
        unparsed = sum(predictions.get(rid) is None for rid in answers)
        runs[str(path)] = (predictions, rows)
        print(
            f"{path}: accuracy {correct}/{len(answers)} = {100.0 * correct / len(answers):.2f}% "
            f"(unparsed {unparsed})")
    names = list(runs)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            lp, lr = runs[left]
            rp, rr = runs[right]
            same_prediction = sum(
                lp.get(rid) == rp.get(rid) for rid in answers)
            same_tokens = sum(
                lr.get(rid, {}).get("output_token_ids") == rr.get(rid, {}).get(
                    "output_token_ids") for rid in answers)
            print(
                f"{left} vs {right}: same prediction {same_prediction}/{len(answers)}, "
                f"same tokens {same_tokens}/{len(answers)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build_parser = sub.add_parser("build")
    build_parser.add_argument("--parquet", type=pathlib.Path, required=True)
    build_parser.add_argument("--output", type=pathlib.Path, required=True)
    build_parser.add_argument("--answers", type=pathlib.Path, required=True)
    build_parser.add_argument("--max-tokens", type=int, default=4)
    build_parser.add_argument("--limit", type=int, default=0)
    build_parser.add_argument("--max-prompt-chars", type=int, default=3200)
    score_parser = sub.add_parser("score")
    score_parser.add_argument("--answers", type=pathlib.Path, required=True)
    score_parser.add_argument("requests", type=pathlib.Path, nargs="+")
    args = parser.parse_args()
    build(args) if args.command == "build" else score(args)


if __name__ == "__main__":
    main()
