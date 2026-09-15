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
"""Run single-knob KV admission and persistent vision capacity diagnostics."""

import argparse
import hashlib
import json
import os
import pathlib
import shutil
import statistics
import subprocess


def digest(path):
    """Hash artifact content without loading an engine into host RAM."""
    content_hash = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            content_hash.update(chunk)
    return content_hash.hexdigest()


def run_campaign(args):
    """Interleave paired runs; each cell starts a fresh generic-calibrated server."""
    repo = pathlib.Path(
        subprocess.check_output(["git", "rev-parse", "--show-toplevel"],
                                text=True).strip())
    root = args.result_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    scripts = repo / "benchmarks/phase_serving"
    model = repo / ".local/artifacts/v0101-forward-port/gemma-4-e2b-it-awq"
    engines = {
        pages: model / f"engine-packed-p8-d24-kv2048-p{pages}"
        for pages in (96, 192)
    }
    binary = repo / ".local/builds/v0101-release/examples/llm/llm_phase_context_smoke"
    trace_root = repo / ".local/results/gemma4-e2b-awq-full12-20260911/inputs"
    stages = {
        "kv": {
            "workloads": ["balanced", "long-prefill", "bimodal"],
            "variants": [(192, 96, 4, 0), (192, 192, 4, 0)],
        },
        "vision": {
            "workloads": ["mixed", "vision-heavy", "multi-image"],
            "variants": [(96, 96, 4, 0), (96, 96, 8, 0), (96, 96, 12, 0)],
        },
        "formation": {
            "workloads": [
                "balanced", "long-prefill", "bimodal", "mixed", "vision-heavy",
                "multi-image"
            ],
            "variants": [(96, 96, 4, 0), (96, 96, 4, 1)],
        },
    }
    selected = stages if args.stage == "all" else {
        args.stage: stages[args.stage]
    }
    workloads = sorted(
        {w
         for stage in selected.values()
         for w in stage["workloads"]})
    identity = {
        "source_commit":
        subprocess.check_output(["git", "rev-parse", "HEAD"],
                                text=True).strip(),
        "tracked_diff_sha256":
        hashlib.sha256(subprocess.check_output(["git", "diff",
                                                "HEAD"])).hexdigest(),
        "binary_sha256":
        digest(binary),
        "plugin_sha256":
        digest(repo /
               ".local/builds/v0101-release/libNvInfer_edgellm_plugin.so.1.0"),
        "engines": {
            str(p): {
                "path": str(e),
                "engine_sha256": digest(e / "llm.engine"),
                "config_sha256": digest(e / "config.json")
            }
            for p, e in engines.items()
        },
        "runner_sha256":
        digest(scripts / "run_gemma_v3_activity_diagnostic.sh"),
        "orchestrator_sha256":
        digest(pathlib.Path(__file__).resolve()),
        "analyzer_sha256":
        digest(scripts / "analyze_gemma_v3_bottleneck.py"),
        "vision_engine_sha256":
        digest(model / "visual-e4-soft280/visual/visual.engine"),
        "traces": {
            w: digest(trace_root / f"{w}.json")
            for w in workloads
        },
    }
    runner_snapshot = root / f"runner-{identity['runner_sha256']}.sh"
    if not runner_snapshot.exists():
        shutil.copyfile(scripts / "run_gemma_v3_activity_diagnostic.sh",
                        runner_snapshot)
    source_patch = root / f"source-{identity['tracked_diff_sha256']}.patch"
    if not source_patch.exists():
        source_patch.write_bytes(
            subprocess.check_output(["git", "diff", "HEAD"]))
    manifest_path = root / f"manifest-{args.stage}.json"
    if manifest_path.exists():
        prior = json.loads(manifest_path.read_text())
        if prior["identity"] != identity or prior[
                "repeat_count"] != args.repeats:
            raise ValueError(
                "Cannot resume cells with a different binary, source, engine, trace, or repeat contract"
            )
    manifest = {
        "state":
        "diagnostic",
        "identity":
        identity,
        "repeat_count":
        args.repeats,
        "command": [
            "python3",
            str(pathlib.Path(__file__).resolve()), "--stage", args.stage,
            "--repeats",
            str(args.repeats), "--result-root",
            str(root)
        ],
        "contract": {
            "policy":
            "V3 service-scaled-transition",
            "prefill_batch":
            8,
            "decode_batch":
            24,
            "chunk_tokens":
            128,
            "encoder_batch":
            4,
            "max_in_flight":
            24,
            "slo":
            "none",
            "generic_http_calibration_requests":
            49,
            "generic_calibration_sha256":
            digest(
                repo /
                ".local/results/gemma4-packed-prefill-g4-20260912/generic-p8-d24-e4.json"
            ),
            "telemetry":
            "full",
            "sampling":
            "synchronized",
            "dedicated_vision_copy":
            False,
            "kv_budget_activation":
            "after calibration, with streams drained and zero stable leases",
            "vision_route":
            "disabled for KV-only stage; enabled for vision and formation stages",
            "frozen_vllm_result":
            str(repo /
                ".local/results/gemma4-vllm-capacity-sweep-20260912/selected-seq24-kv480-p4096-g24-full12"
                ),
            "encoder_capacity_activation":
            "initialization; learning trajectory may differ during generic calibration",
            "formation_activation":
            "initialization; learning trajectory may differ during generic calibration"
        },
        "stages":
        selected,
        "completed": [],
        "summary_path":
        str(root / f"summary-{args.stage}.json"),
        "note_references": [
            "notes/299-gemma4-v3-mask-and-admission-bottleneck-20260913.md",
            "notes/300-gemma4-v3-capacity-ab-20260913.md"
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    records = []
    for stage_name, stage in selected.items():
        for repeat in range(1, args.repeats + 1):
            variants = stage["variants"] if repeat % 2 else list(
                reversed(stage["variants"]))
            for physical, available, encoded, disable_wavefront in variants:
                name = f"p{physical}-budget{available}-encoded{encoded}"
                if stage_name == "formation":
                    name += f"-wavefront{1 - disable_wavefront}"
                cell = root / stage_name / name / f"repeat-{repeat:03d}"
                env = dict(os.environ,
                           RESULT_ROOT=str(cell),
                           CASES=" ".join(stage["workloads"]),
                           ENGINE_ROOT=str(engines[physical]),
                           ALLOCATABLE_KV_PAGES=str(available),
                           ENCODED_CAPACITY=str(encoded))
                env["ENABLE_VISION"] = "0" if stage_name == "kv" else "1"
                env["KV_BUDGET_AT_MEASUREMENT"] = "1"
                env["DISABLE_WAVEFRONT_PREFILL"] = str(disable_wavefront)
                print(f"Running {stage_name} {name} repeat {repeat}",
                      flush=True)
                subprocess.run(["bash", str(runner_snapshot)],
                               env=env,
                               check=True)
                subprocess.run([
                    "python3",
                    str(scripts / "analyze_gemma_v3_bottleneck.py"),
                    "--result-root",
                    str(cell)
                ],
                               check=True)
                analysis = json.loads(
                    (cell / "bottleneck-analysis.json").read_text())
                for workload, report in analysis.items():
                    records.append({
                        "stage": stage_name,
                        "variant": name,
                        "repeat": repeat,
                        "workload": workload,
                        "report": report,
                        "result_root": str(cell / workload)
                    })
                manifest["completed"].append({
                    "stage": stage_name,
                    "variant": name,
                    "repeat": repeat
                })
                manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    groups = {}
    for record in records:
        key = f"{record['stage']}/{record['variant']}/{record['workload']}"
        groups.setdefault(key, []).append(record["report"])
    summary = {}
    for key, reports in groups.items():
        numeric = {
            metric: statistics.mean(r["serving"][metric] for r in reports)
            for metric in reports[0]["serving"]
            if isinstance(reports[0]["serving"][metric], (int, float)) and all(
                isinstance(r["serving"][metric], (int, float))
                for r in reports)
        }
        summary[key] = {
            "repeats": len(reports),
            "mean_serving_metrics": numeric,
            "reports": reports
        }
    (root / f"summary-{args.stage}.json"
     ).write_text(json.dumps(summary, indent=2) + "\n")


def main():
    """Parse the bounded diagnostic campaign configuration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage",
                        choices=("kv", "vision", "formation", "all"),
                        default="all")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--result-root",
        type=pathlib.Path,
        default=pathlib.Path(".local/results/gemma4-v3-capacity-ab-20260913"))
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    run_campaign(args)


if __name__ == "__main__":
    main()
