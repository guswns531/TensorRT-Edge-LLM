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
"""Bounded, paired decode trials; never changes serving defaults."""

import argparse
import json
import math
import os
import pathlib
import statistics
import subprocess


def distribution(values):
    """Report mean/median and nearest-rank p95 of measured milliseconds."""
    if not values or any(not math.isfinite(v) or v <= 0 for v in values):
        raise ValueError("Trial timings must be finite and positive")
    ordered = sorted(values)
    return {
        "mean": statistics.mean(ordered),
        "median": statistics.median(ordered),
        "p95": ordered[math.ceil(.95 * len(ordered)) - 1],
    }


def summarize(report):
    """Reject incomplete/unpaired trials or graph fallback before comparison."""
    config = report["config"]
    samples = report["samples"]
    expected = {(r, i, v)
                for r in range(config["rounds"])
                for i in range(config["iterations"])
                for v in ("dense", "split")}
    actual = {(s["round"], s["iteration"], s["variant"]) for s in samples}
    if actual != expected or len(samples) != len(expected):
        raise ValueError("Missing or duplicate paired samples")
    if report["graph_misses"] or report["policy_applied"]:
        raise ValueError("Trial must use graphs without applying policy")
    for r in range(config["rounds"]):
        for i in range(config["iterations"]):
            pair = [
                s for s in samples if (s["round"], s["iteration"]) == (r, i)
            ]
            if {s["order"] for s in pair} != {0, 1}:
                raise ValueError("Invalid paired execution order")
    metrics = ("gpu_ms", "drain_ms", "successor_gpu_ms", "two_tick_ms")
    result = {
        "config": config,
        "compared_tokens": report["compared_tokens"],
        "mismatched_tokens": report["mismatched_tokens"],
        "per_round": [],
    }
    for variant in ("dense", "split"):
        result[variant] = {
            metric:
            distribution(
                [s[metric] for s in samples if s["variant"] == variant])
            for metric in metrics
        }
    result["split_change_pct"] = {
        metric: {
            stat:
            100 *
            (result["split"][metric][stat] / result["dense"][metric][stat] - 1)
            for stat in ("mean", "median", "p95")
        }
        for metric in metrics
    }
    for r in range(config["rounds"]):
        result["per_round"].append({
            "round": r,
            **{
                variant: {
                    metric:
                    distribution([
                        s[metric] for s in samples if s["round"] == r and s["variant"] == variant
                    ])
                    for metric in metrics
                }
                for variant in ("dense", "split")
            }
        })
    return result


def main():
    """Freeze provenance and execute only the two selected equal-work points."""
    import run_lifetime_encoded_admission as serving

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-root", type=pathlib.Path, required=True)
    parser.add_argument("--result-root", type=pathlib.Path, required=True)
    args = parser.parse_args()
    repo = pathlib.Path(
        subprocess.check_output(["git", "rev-parse", "--show-toplevel"],
                                text=True).strip())
    root = args.result_root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    build = args.build_root.resolve()
    binary = build / "examples/llm/llm_phase_context_smoke"
    plugin = build / "libNvInfer_edgellm_plugin.so.1.0"
    manifest = {
        "state":
        "diagnostic",
        "status":
        "running",
        "source_commit":
        subprocess.check_output(["git", "rev-parse", "HEAD"],
                                text=True).strip(),
        "source_status":
        subprocess.check_output(["git", "status", "--porcelain"],
                                text=True).splitlines(),
        "binary_sha256":
        serving.digest(binary),
        "plugin_sha256":
        serving.digest(plugin),
        "runner_sha256":
        serving.digest(pathlib.Path(__file__)),
        "container":
        serving.IMAGE,
        "repeat_contract":
        "5 in-process blocks, 20 warmup + 50 paired trials each",
        "note_references":
        ["notes/343-startup-decode-equal-work-trials-20260927.md"],
        "summary_path":
        str(root / "summary.json"),
        "cells": [],
    }
    (root / "source.patch").write_bytes(
        subprocess.check_output(["git", "diff", "HEAD"]))
    for relative in (
            "examples/llm/phaseDecodeEqualWorkTrial.inc",
            "benchmarks/phase_serving/run_decode_equal_work_trials.py"):
        (root / pathlib.Path(relative).name).write_bytes(
            (repo / relative).read_bytes())
    summary = {}
    for model, rows, split in (("gemma", 8, 4), ("cosmos", 64, 32)):
        config = serving.model_config(repo, model)
        cell = root / model
        cell.mkdir()
        output = cell / "raw.json"
        trial = {
            "rows": rows,
            "split": split,
            "past_tokens": 128,
            "warmup": 20,
            "iterations": 50,
            "rounds": 5,
            "output": "/workspace/" + str(output.relative_to(repo))
        }
        trial_path = cell / "trial.json"
        trial_path.write_text(json.dumps(trial, indent=2) + "\n")
        env = {
            "TRT_PACKAGE_DIR":
            "/opt/tensorrt",
            "LD_LIBRARY_PATH":
            "/opt/tensorrt/lib:/workspace/" + str(build.relative_to(repo)),
            "EDGELLM_PLUGIN_PATH":
            "/workspace/" + str(plugin.relative_to(repo)),
            "TRT_EDGELLM_DECODE_EQUAL_WORK_TRIAL":
            "/workspace/" + str(trial_path.relative_to(repo)),
            "TRT_EDGELLM_MAX_DECODE_BATCH":
            str(config["decode_batch"]),
            "TRT_EDGELLM_MAX_STABLE_SLOTS":
            str(config["stable_slots"])
        }
        command = [
            "docker", "run", "--rm", "--network", "none", "--gpus", "all",
            "--user", f"{os.getuid()}:{os.getgid()}", "-v",
            str(repo) + ":/workspace", "-w", "/workspace"
        ]
        for key, value in env.items():
            command.extend(["-e", key + "=" + value])
        command += [
            serving.IMAGE, "/workspace/" + str(binary.relative_to(repo)),
            "/workspace/" + str(config["engine"].relative_to(repo)),
            "/workspace/" + str(config["hf"].relative_to(repo))
        ]
        record = {
            "model": model,
            "command": command,
            "trial": trial,
            "engine": str(config["engine"]),
            "engine_files": {
                p.name: serving.digest(p)
                for p in sorted(config["engine"].iterdir()) if p.is_file()
            },
            "raw": str(output)
        }
        manifest["cells"].append(record)
        (root /
         "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print("Running", model, flush=True)
        with (cell / "backend.log").open("w") as log:
            outcome = subprocess.run(command,
                                     stdout=log,
                                     stderr=subprocess.STDOUT,
                                     check=False)
        record["return_code"] = outcome.returncode
        if outcome.returncode:
            manifest["status"] = "failed"
            (root / "manifest.json"
             ).write_text(json.dumps(manifest, indent=2) + "\n")
            raise RuntimeError("Trial failed; see " +
                               str(cell / "backend.log"))
        summary[model] = summarize(json.loads(output.read_text()))
        (root /
         "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    manifest["status"] = "complete"
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
