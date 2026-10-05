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
"""Measure throughput/latency versus decode batch and pick the capacity knee per system.

Step 2 of capacity selection (step 1, plan_phase_capacity.py, bounds the batch by memory).
For each decode batch D the load scales with D: client in-flight = D, vLLM max-num-seqs = D,
bulk text traces carry 4*D requests and vision mixes D/32 times the base mix, and the phase
calibration trace covers decode rows up to D. The recommended D per system is the smallest
whose throughput geomean is within --knee-fraction of the best D and whose worst TPOT p95
meets --tpot-p95-slo-ms.
"""

import argparse
import copy
import json
import pathlib
import statistics
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent


def build_inputs(args, decode_batch, prefill_batch, encoder_batch, directory):
    if (directory / "manifest.json").exists():
        return
    subprocess.run([
        sys.executable,
        str(HERE / "build_serving_workloads.py"), "--tokenizer",
        str(args.tokenizer), "--output-dir",
        str(directory), "--bulk-requests",
        str(4 * decode_batch), "--mix-scale",
        str(max(1, decode_batch // 32))
    ],
                   check=True,
                   stdout=subprocess.DEVNULL)
    subprocess.run([
        sys.executable,
        str(HERE / "build_generic_policy_calibration_trace.py"), "--output",
        str(directory / "calibration.json"), "--image-url", "file://" +
        str(HERE.parents[1] / "examples/multimodal/pics/giant_panda.jpeg"),
        "--max-prefill-batch",
        str(prefill_batch), "--max-decode-batch",
        str(decode_batch), "--max-encoder-batch",
        str(encoder_batch), "--prefill-tokens",
        str(args.calibration_prefill_tokens)
    ],
                   check=True,
                   stdout=subprocess.DEVNULL)


def config_for(base, decode_batch, directory):
    config = copy.deepcopy(base)
    calibration = directory / "calibration.json"
    trace = json.loads(calibration.read_text())
    config.update(inputs=str(directory),
                  calibration_trace=str(calibration),
                  warmup_requests=len(
                      trace["requests"] if isinstance(trace, dict) else trace),
                  decode_batch=decode_batch,
                  in_flight=decode_batch,
                  stable_slots=max(decode_batch,
                                   base.get("stable_slots", decode_batch)))
    vllm_args = config["vllm_args"]
    vllm_args[vllm_args.index("--max-num-seqs") + 1] = str(decode_batch)
    return config


def cell_metrics(path):
    run = json.loads(path.read_text())["runs"][0]
    return {
        "generated_token_s":
        run.get("generated_token_s"),
        "ttft_p95_ms":
        run.get("ttft_ms", {}).get("p95"),
        "tpot_p95_ms":
        run.get("tpot_ms", {}).get("p95"),
        "complete":
        bool(run.get("fixed_output_complete")) and run.get("failed", 1) == 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",
                        type=pathlib.Path,
                        required=True,
                        help="Base campaign config")
    parser.add_argument("--tokenizer", type=pathlib.Path, required=True)
    parser.add_argument("--decode-batches", type=int, nargs="+", required=True)
    parser.add_argument("--workloads",
                        nargs="+",
                        default=["balanced", "decode-heavy", "mixed"])
    parser.add_argument("--systems", nargs="+", default=["trt", "vllm"])
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--calibration-prefill-tokens", type=int, default=1024)
    parser.add_argument("--knee-fraction", type=float, default=0.95)
    parser.add_argument("--tpot-p95-slo-ms", type=float, default=50.0)
    args = parser.parse_args()

    base = json.loads(args.config.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for decode_batch in args.decode_batches:
        directory = args.output_dir / ("D%d" % decode_batch)
        inputs = directory / "inputs"
        build_inputs(args, decode_batch, base["prefill_batch"],
                     base["vision_batch"], inputs)
        config = config_for(base, decode_batch, inputs)
        config_path = directory / "config.json"
        config_path.write_text(json.dumps(config, indent=2) + "\n")
        subprocess.run([
            sys.executable,
            str(HERE / "run_serving_comparison.py"), "--config",
            str(config_path), "--reuse-server", "--repeats", "1",
            "--output-dir",
            str(directory / "runs"), "--systems", *args.systems, "--workloads",
            *args.workloads
        ])
        for system in args.systems:
            for workload in args.workloads:
                path = directory / "runs" / system / workload / "summary.json"
                if path.exists():
                    results.setdefault(system, {}).setdefault(
                        decode_batch, {})[workload] = cell_metrics(path)

    recommendation = {}
    for system, by_batch in results.items():
        scores = {}
        for decode_batch, cells in by_batch.items():
            if len(cells) == len(args.workloads) and all(
                    cell["complete"] for cell in cells.values()):
                scores[decode_batch] = (statistics.geometric_mean(
                    cell["generated_token_s"] for cell in cells.values()),
                                        max(cell["tpot_p95_ms"] or 0.0
                                            for cell in cells.values()))
        if not scores:
            continue
        best = max(score[0] for score in scores.values())
        eligible = [
            decode_batch
            for decode_batch, (rate, tpot) in sorted(scores.items())
            if rate >= args.knee_fraction *
            best and tpot <= args.tpot_p95_slo_ms
        ]
        recommendation[system] = {
            "scores": {
                str(d): {
                    "throughput_geomean": round(rate, 1),
                    "worst_tpot_p95_ms": round(tpot, 1)
                }
                for d, (rate, tpot) in sorted(scores.items())
            },
            "recommended_decode_batch": eligible[0] if eligible else None,
        }
    summary = {
        "results": {
            s: {
                str(d): c
                for d, c in b.items()
            }
            for s, b in results.items()
        },
        "recommendation": recommendation
    }
    (args.output_dir /
     "sweep.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(recommendation, indent=2))


if __name__ == "__main__":
    main()
