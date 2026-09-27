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
"""Compare paired same-binary startup calibration cells, including missing coverage."""

import argparse
import json
import math
import pathlib
import statistics

import analyze_lifetime_encoded_admission as dispatch_report
import report_workspace_revalidation as serving_report


def summarize(root, include_dispatch=False):
    """Keep startup cost separate from steady-state HTTP latency and throughput."""
    manifest = json.loads((root / "manifest.json").read_text())
    completed = {record["cell"] for record in manifest.get("completed", [])}
    rows, missing = [], []
    commands = manifest["commands"]
    references = {
        (record["model"], record["workload"], record["repeat"]): record
        for record in commands
        if record["variant"] == "independent-predictor-on"
    }
    for record in commands:
        if record["variant"] != "independent-startup-predictor-on":
            continue
        key = (record["model"], record["workload"], record["repeat"])
        baseline = references.get(key)
        if (baseline is None or record["cell"] not in completed
                or baseline["cell"] not in completed):
            missing.append(key)
            continue
        current = json.loads(
            (pathlib.Path(record["cell"]) / "aggregate.json").read_text())
        previous = json.loads(
            (pathlib.Path(baseline["cell"]) / "aggregate.json").read_text())
        for field in ("trace_sha256", "requests_per_run",
                      "requested_output_tokens_per_run"):
            if current[field] != previous[field]:
                raise ValueError("Paired workload contract differs: " + field)
        startup = json.loads((pathlib.Path(record["cell"]) /
                              "run-001/startup.json").read_text())
        changes = {}
        for metric in serving_report.METRICS:
            a, b = float(current[metric]), float(previous[metric])
            if not all(math.isfinite(value) and value > 0 for value in (a, b)):
                raise ValueError("Invalid metric: " + metric)
            changes[metric] = 100 * (a / b - 1)
        row = {
            "model": key[0],
            "workload": key[1],
            "repeat": key[2],
            "delta_percent": changes,
            "startup_elapsed_ms": startup["elapsed_ms"],
            "startup_frontier_covered": startup["frontier_covered"],
            "startup_decode_probe_requests": startup["decode_probe_requests"],
            "baseline_peak_mib": previous["gpu_memory_peak_mib_median"],
            "startup_peak_mib": current["gpu_memory_peak_mib_median"],
            "baseline": baseline["cell"],
            "startup": record["cell"],
        }
        if include_dispatch:
            for name, cell in (("baseline", baseline["cell"]),
                               ("startup", record["cell"])):
                dispatches = dispatch_report.analyze_cell(
                    pathlib.Path(cell))["dispatches"]
                row[name + "_dispatches"] = {
                    phase: dispatches[phase]
                    for phase in ("prefill", "decode")
                }
        rows.append(row)
    return {
        "rows":
        rows,
        "missing_pairs":
        missing,
        "complete":
        bool(rows) and not missing
        and manifest.get("completion_status") == "complete",
        "source_manifest":
        str(root / "manifest.json"),
        "startup_scope":
        "Post-load preparation, not total process-to-ready time; coverage is not RLS convergence"
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=pathlib.Path, required=True)
    parser.add_argument("--include-dispatch", action="store_true")
    args = parser.parse_args()
    report = summarize(args.result_root.resolve(), args.include_dispatch)
    lines = [
        "# Startup calibration paired comparison", "",
        "Throughput: positive is better. Latency: negative is better. All deltas are against the same-binary legacy calibration.",
        "", "| Model/workload/repeat | " + " | ".join(serving_report.LABELS) +
        " | Startup ms | Peak MiB old/new |",
        "|---|" + "---:|" * (len(serving_report.METRICS) + 2)
    ]
    for row in report["rows"]:
        values = [
            "%+.2f%%" % row["delta_percent"][metric]
            for metric in serving_report.METRICS
        ]
        lines.append("| %s/%s/%s | %s | %.1f | %s/%s |" %
                     (row["model"], row["workload"], row["repeat"],
                      " | ".join(values), row["startup_elapsed_ms"],
                      row["baseline_peak_mib"], row["startup_peak_mib"]))
    for model in sorted({row["model"] for row in report["rows"]}):
        values = [row for row in report["rows"] if row["model"] == model]
        ratios = [
            1 + row["delta_percent"][serving_report.METRICS[0]] / 100
            for row in values
        ]
        gain = 100 * (
            math.exp(statistics.mean(math.log(value) for value in ratios)) - 1)
        lines += [
            "",
            "%s: %d paired runs; throughput geometric-mean delta %+.2f%%." %
            (model, len(values), gain)
        ]
    lines += [
        "",
        "Complete: %s. Missing pairs: %s." %
        (report["complete"], report["missing_pairs"]), "",
        report["startup_scope"],
        "Single repeats do not establish a promotion gate."
    ]
    (args.result_root / "startup-comparison.json"
     ).write_text(json.dumps(report, indent=2) + "\n")
    (args.result_root / "startup-comparison.md").write_text("\n".join(lines) +
                                                            "\n")
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
