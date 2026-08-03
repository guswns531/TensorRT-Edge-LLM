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
"""Compare same-trace phase scheduler request and dispatch CSV files."""

import argparse
import csv
import math
from pathlib import Path


def percentile(values: list[float], quantile: float) -> float:
    values = sorted(values)
    return values[math.floor(quantile * (len(values) - 1))]


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as input_file:
        return list(csv.DictReader(input_file))


def trace_key(rows: list[dict[str, str]]) -> list[tuple[str, ...]]:
    return [(row["request_id"], row["scheduled_arrival_us"],
             row["prompt_tokens"], row["max_output_tokens"]) for row in rows]


def summarize(name: str, request_csv: Path) -> list[dict[str, str]]:
    rows = load_rows(request_csv)
    terminal_us = max(int(row["terminal_us"]) for row in rows)
    generated_tokens = sum(int(row["generated_tokens"]) for row in rows)
    dispatch_csv = request_csv.with_name(f"{request_csv.stem}-dispatch.csv")
    dispatch_count = len(load_rows(dispatch_csv))
    result = []
    priorities = sorted({int(row["priority"]) for row in rows})
    for priority in [None, *priorities]:
        selected = rows if priority is None else [
            row for row in rows if int(row["priority"]) == priority
        ]
        ttft_ms = [float(row["ttft_us"]) / 1000.0 for row in selected]
        tpot_ms = [float(row["tpot_us"]) / 1000.0 for row in selected]
        result.append({
            "mode":
            name,
            "priority":
            "all" if priority is None else str(priority),
            "requests":
            str(len(selected)),
            "request_per_s":
            f"{len(rows) * 1.0e6 / terminal_us:.3f}"
            if priority is None else "",
            "token_per_s":
            f"{generated_tokens * 1.0e6 / terminal_us:.3f}"
            if priority is None else "",
            "ttft_p50_ms":
            f"{percentile(ttft_ms, 0.5):.3f}",
            "ttft_p95_ms":
            f"{percentile(ttft_ms, 0.95):.3f}",
            "tpot_p50_ms":
            f"{percentile(tpot_ms, 0.5):.3f}",
            "tpot_p95_ms":
            f"{percentile(tpot_ms, 0.95):.3f}",
            "dispatches":
            str(dispatch_count) if priority is None else "",
        })
    return result


def parse_run(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError("run must use NAME=REQUEST_CSV")
    return name, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",
                        action="append",
                        required=True,
                        type=parse_run)
    args = parser.parse_args()

    runs = [(name, load_rows(path), path) for name, path in args.run]
    expected_trace = trace_key(runs[0][1])
    for name, rows, _ in runs[1:]:
        if trace_key(rows) != expected_trace:
            raise ValueError(
                f"{name} does not use the same arrival/prompt/output trace")

    summaries = [
        row for name, _, path in runs for row in summarize(name, path)
    ]
    fieldnames = list(summaries[0])
    writer = csv.DictWriter(__import__("sys").stdout,
                            fieldnames=fieldnames,
                            lineterminator="\n")
    writer.writeheader()
    writer.writerows(summaries)


if __name__ == "__main__":
    main()
