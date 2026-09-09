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
"""Compute pooled request latency, including scheduled-arrival waiting, from retained CSVs."""

import argparse
import csv
import json
import math
import pathlib
import statistics


def latency_statistics(values):
    values = sorted(values)
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("Latency samples must be nonempty and finite")
    index = (len(values) - 1) * 0.95
    lower = int(index)
    upper = min(lower + 1, len(values) - 1)
    return statistics.mean(values), values[lower] + (
        values[upper] - values[lower]) * (index - lower)


def request_files(row, results_root):
    if "source" in row:
        parent = pathlib.Path(row["source"]).parent
        return sorted(
            parent.glob("run-*/client/run-*/requests.csv")) or sorted(
                parent.glob("run-*/requests.csv"))
    aliases = {
        "multi-image": "multi",
        "wave-drain": "wave",
        "late-vision": "late"
    }
    root = results_root / "baselines/vllm-frozen-12x3" / aliases.get(
        row["workload"], row["workload"])
    return sorted(root.glob("run-*/client/run-*/requests.csv"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=pathlib.Path, required=True)
    parser.add_argument("--results-root", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--allow-missing-historical", action="store_true")
    args = parser.parse_args()
    records, missing = [], []
    for row in json.loads(args.summary.read_text())["rows"]:
        files = request_files(row, args.results_root)
        if len(files) != row["repeats"]:
            if args.allow_missing_historical and row["historical"]:
                missing.append({
                    "workload": row["workload"],
                    "variant": row["variant"],
                    "expected_runs": row["repeats"],
                    "available_csvs": len(files)
                })
                continue
            raise ValueError(
                f"Missing request CSVs for {row['variant']}/{row['workload']}: {len(files)}"
            )
        samples = []
        for path in files:
            with path.open() as stream:
                samples.extend(csv.DictReader(stream))
        if any(
                int(sample["http_status"]) != 200 or sample["error"]
                for sample in samples):
            raise ValueError(
                f"Failed measurement request in {row['variant']}/{row['workload']}"
            )
        metrics = {
            name: [float(sample[name + "_ms"]) for sample in samples]
            for name in ("ttft", "tpot", "e2e")
        }
        for name, finish in (("arrival_ttft", "first_token_us"),
                             ("arrival_e2e", "completed_us")):
            metrics[name] = [
                (float(sample[finish]) - float(sample["scheduled_arrival_us"]))
                / 1000.0 for sample in samples
            ]
        result = {
            "workload": row["workload"],
            "variant": row["variant"],
            "runs": len(files),
            "requests": len(samples),
            "token_s": row["token_s"],
            "sources": [str(path.resolve()) for path in files]
        }
        for name, values in metrics.items():
            mean, p95 = latency_statistics(values)
            result[name + "_mean_ms"] = mean
            result[name + "_p95_ms"] = p95
        records.append(result)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "request-distributions.json"
     ).write_text(json.dumps(records, indent=2) + "\n")
    (args.output / "request-distributions-missing.json"
     ).write_text(json.dumps(missing, indent=2) + "\n")
    with (args.output / "request-distributions.csv").open(
            "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    lines = [
        "# Pooled request distributions", "",
        "Mean and p95 are computed across all measured requests from all repeats. "
        "Arrival-relative values include client admission waiting; ordinary values start at send. "
        "These p95 values differ from the median of per-run p95 values. No confidence interval is inferred.",
        "",
        f"Unavailable historical rows (not estimated from partial repeats): {json.dumps(missing)}",
        "",
        "| Workload | Variant | Requests | TTFT mean/p95 | TPOT mean/p95 | E2E mean/p95 | "
        "Arrival TTFT mean/p95 | Arrival E2E mean/p95 |",
        "|---|---|---:|---:|---:|---:|---:|---:|"
    ]
    for row in sorted(records,
                      key=lambda item: (item["workload"], item["variant"])):
        values = [
            f"{row[name + '_mean_ms']:.2f}/{row[name + '_p95_ms']:.2f}"
            for name in ("ttft", "tpot", "e2e", "arrival_ttft", "arrival_e2e")
        ]
        lines.append(
            f"| {row['workload']} | {row['variant']} | {row['requests']} | {' | '.join(values)} |"
        )
    (args.output / "request-distributions.md").write_text("\n".join(lines) +
                                                          "\n")
    print(f"Pooled request distributions: {len(records)} comparison rows")


if __name__ == "__main__":
    main()
