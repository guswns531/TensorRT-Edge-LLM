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
"""Compare no activity observer, encoder-only labels, and full activity export on one binary."""

import argparse
import json
import math
import pathlib
import statistics
import subprocess
import sys

MODES = ("off", "encoder", "full")
METRICS = ("generated_token_s_median", "ttft_mean_of_run_means_ms",
           "tpot_mean_of_run_means_ms", "e2e_mean_of_run_means_ms")


def block_order(block):
    """Rotate mode order per block so drift is not aliased onto one mode."""
    shift = block % len(MODES)
    return MODES[shift:] + MODES[:shift]


def collect(root, blocks):
    """Return {(model, workload): {mode: {metric: [values per block]}}}."""
    cells = {}
    for block in range(blocks):
        for mode in MODES:
            summary_path = root / f"block-{block + 1}" / mode / "summary.json"
            if not summary_path.is_file():
                continue
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            for key, value in summary.items():
                model, workload = key.split("/")[:2]
                runs = value.get("runs", [])
                if len(runs) != 1:
                    raise ValueError(f"Expected one run in {summary_path}")
                target = cells.setdefault(
                    (model, workload),
                    {}).setdefault(mode, {metric: []
                                          for metric in METRICS})
                for metric in METRICS:
                    target[metric].append(float(runs[0][metric]))
    return cells


def summarize(cells):
    """Median per mode, percentage change of encoder/full against off."""
    rows = []
    for (model, workload), modes in sorted(cells.items()):
        if "off" not in modes:
            continue
        row = {"model": model, "workload": workload, "blocks": {}}
        for mode, values in modes.items():
            row["blocks"][mode] = len(values[METRICS[0]])
            row[mode] = {
                metric: statistics.median(series)
                for metric, series in values.items()
            }
        for mode in ("encoder", "full"):
            if mode in modes:
                row[mode + "_vs_off_pct"] = {
                    metric:
                    (row[mode][metric] / row["off"][metric] - 1.0) * 100.0
                    for metric in METRICS
                }
        rows.append(row)
    overall = {}
    for mode in ("encoder", "full"):
        deltas = [
            row[mode + "_vs_off_pct"]["generated_token_s_median"]
            for row in rows if mode + "_vs_off_pct" in row
        ]
        if deltas:
            overall[mode + "_throughput_geomean_vs_off_pct"] = (math.exp(
                statistics.mean(math.log1p(delta / 100.0)
                                for delta in deltas)) - 1.0) * 100.0
    return {"rows": rows, "overall": overall}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=pathlib.Path, required=True)
    parser.add_argument("--build-root", type=pathlib.Path, required=True)
    parser.add_argument("--binary-source-commit", required=True)
    parser.add_argument("--models", nargs="+", default=("gemma", "cosmos"))
    parser.add_argument("--workloads",
                        nargs="+",
                        default=("balanced", "mixed", "vision-heavy",
                                 "multi-image", "long-prefill"))
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    runner = pathlib.Path(__file__).with_name(
        "run_lifetime_encoded_admission.py")
    root = args.result_root.resolve()
    if not args.summarize_only:
        root.mkdir(parents=True, exist_ok=False)
        for block in range(args.blocks):
            for mode in block_order(block):
                subprocess.run([
                    sys.executable,
                    str(runner), "--models", *args.models, "--workloads",
                    *args.workloads, "--variants", "independent",
                    "--transition-predictors", "on", "--cuda-graphs", "on",
                    "--serving-overlap-probes", "on", "--telemetry-level",
                    "dispatch", "--activity-observer", mode, "--repeats", "1",
                    "--build-root",
                    str(args.build_root), "--binary-source-commit",
                    args.binary_source_commit, "--compress-closed-logs",
                    "--result-root",
                    str(root / f"block-{block + 1}" / mode)
                ],
                               check=True)
    result = summarize(collect(root, args.blocks))
    (root / "overhead.json").write_text(json.dumps(result, indent=2) + "\n",
                                        encoding="utf-8")
    print(json.dumps(result["overall"], indent=2))


if __name__ == "__main__":
    main()
