#!/usr/bin/env python3
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
"""Run the Qwen3.8 HTTP matrix with Thor clock and power telemetry."""

import argparse
import json
import re
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

GPU_MIN = Path("/sys/class/devfreq/gpu-gpc-0/min_freq")
GPU_MAX = Path("/sys/class/devfreq/gpu-gpc-0/max_freq")
EMC_MIN = Path("/sys/class/devfreq/bwmgr/min_freq")
EMC_MAX = Path("/sys/class/devfreq/bwmgr/max_freq")
POWER_PATTERN = re.compile(r"\b(VDD_GPU|VDD_CPU_SOC_MSS|VIN) (\d+)mW")
TEMP_PATTERN = re.compile(r"\b(cpu|tj|gpu)@([0-9.]+)C")


def require_locked_thor_clocks() -> dict[str, int]:
    clocks = {
        "gpu_min_hz": int(GPU_MIN.read_text()),
        "gpu_max_hz": int(GPU_MAX.read_text()),
        "emc_min_hz": int(EMC_MIN.read_text()),
        "emc_max_hz": int(EMC_MAX.read_text()),
    }
    if clocks["gpu_min_hz"] != clocks["gpu_max_hz"]:
        raise RuntimeError(
            "Thor GPU clock is not locked; run sudo jetson_clocks")
    if clocks["emc_min_hz"] != clocks["emc_max_hz"]:
        raise RuntimeError(
            "Thor EMC clock is not locked; run sudo jetson_clocks")
    return clocks


def median_summary(result_paths: list[Path]) -> dict[str, Any]:
    results = [json.loads(path.read_text()) for path in result_paths]
    summaries = [result["summary"] for result in results]

    def median(path: tuple[str, ...]) -> float:
        values: list[float] = []
        for summary in summaries:
            value: Any = summary
            for key in path:
                value = value[key]
            values.append(float(value))
        return statistics.median(values)

    return {
        "runs":
        len(results),
        "successful_requests":
        min(summary["successful_requests"] for summary in summaries),
        "failed_requests":
        max(summary["failed_requests"] for summary in summaries),
        "output_token_throughput":
        median(("output_token_throughput", )),
        "ttft_ms":
        median(("ttft_ms", "median")),
        "tpot_ms":
        median(("tpot_ms", "median")),
        "e2e_ms":
        median(("e2e_ms", "median")),
    }


def summarize_tegrastats(path: Path) -> dict[str, Any]:
    power: dict[str, list[int]] = {}
    temperatures: dict[str, list[float]] = {}
    for line in path.read_text().splitlines():
        for name, value in POWER_PATTERN.findall(line):
            power.setdefault(name, []).append(int(value))
        for name, value in TEMP_PATTERN.findall(line):
            temperatures.setdefault(name, []).append(float(value))
    return {
        "samples": max((len(values) for values in power.values()), default=0),
        "power_mw": {
            name: {
                "mean": statistics.mean(values),
                "max": max(values),
            }
            for name, values in power.items()
        },
        "temperature_c": {
            name: {
                "mean": statistics.mean(values),
                "max": max(values),
            }
            for name, values in temperatures.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--config",
                        type=Path,
                        default=Path("experiments/qwen38_thor/workloads.json"))
    parser.add_argument("--tokenizer",
                        type=Path,
                        default=Path("data/qwen38/nvfp4/tokenizer.json"))
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument(
        "--cases",
        help="Comma-separated case names; default is every configured case")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--mode",
                        choices=("stream", "nonstream"),
                        default="stream")
    parser.add_argument("--timeout", type=float, default=1200.0)
    args = parser.parse_args()

    if args.repeats <= 0:
        parser.error("--repeats must be positive")
    config = json.loads(args.config.read_text())
    cases = args.cases.split(",") if args.cases else list(config["cases"])
    unknown = [case for case in cases if case not in config["cases"]]
    if unknown:
        parser.error(f"unknown cases: {','.join(unknown)}")

    clocks = require_locked_thor_clocks()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    telemetry_path = args.results_dir / "tegrastats.log"
    benchmark_script = Path(__file__).with_name("benchmark_serving.py")
    result_paths: dict[str, list[Path]] = {case: [] for case in cases}

    with telemetry_path.open("w") as telemetry:
        monitor = subprocess.Popen(
            ["tegrastats", "--interval", "1000"],
            stdout=telemetry,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            for case in cases:
                for repetition in range(1, args.repeats + 1):
                    output = args.results_dir / case / f"run-{repetition}.json"
                    output.parent.mkdir(parents=True, exist_ok=True)
                    subprocess.run(
                        [
                            sys.executable,
                            str(benchmark_script),
                            "--base-url",
                            args.base_url,
                            "--model",
                            "qwen38",
                            "--config",
                            str(args.config),
                            "--case",
                            case,
                            "--tokenizer",
                            str(args.tokenizer),
                            "--output",
                            str(output),
                            "--mode",
                            args.mode,
                            "--timeout",
                            str(args.timeout),
                        ],
                        check=True,
                    )
                    result_paths[case].append(output)
        finally:
            monitor.terminate()
            monitor.wait(timeout=10)

    summary = {
        "schema_version": 1,
        "backend": args.backend,
        "base_url": args.base_url,
        "mode": args.mode,
        "thor_clocks": clocks,
        "cases": {
            case: median_summary(paths)
            for case, paths in result_paths.items()
        },
        "telemetry": summarize_tegrastats(telemetry_path),
    }
    (args.results_dir /
     "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
