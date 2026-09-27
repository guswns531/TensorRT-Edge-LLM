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

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Controlled async-server decode transitions and batch-dependent token checks."""

import argparse
import json
import math
import os
import pathlib
import statistics
import subprocess

import run_lifetime_encoded_admission as serving


def summarize(raw):
    """Validate dispatch contracts before summarizing isolated controlled costs."""
    cfg = raw["config"]
    episodes = raw["episodes"]
    expected = {(r, v) for r in range(cfg["rounds"]) for v in range(3)}
    if {(e["round"], e["variant"])
            for e in episodes} != expected or len(episodes) != len(expected):
        raise ValueError("Missing or duplicate episodes")
    result = {}
    for round_index in range(cfg["rounds"]):
        if {e["order"]
                for e in episodes if e["round"] == round_index} != {0, 1, 2}:
            raise ValueError("Invalid episode order")
    for variant, name in enumerate(("dense", "split", "singleton")):
        group = [e for e in episodes if e["variant"] == variant]
        for episode in group:
            if len(episode["tokens"]) != cfg["rows"] or any(
                    len(t) != cfg["output_tokens"] for t in episode["tokens"]):
                raise ValueError("Incomplete token capture")
            for turn in range(cfg["output_tokens"] - 1):
                batch = 1 if variant == 2 else (
                    cfg["split"] if variant == 1 and turn < cfg["split_turns"]
                    else cfg["rows"])
                dispatches = [
                    d for d in episode["dispatches"] if d["turn"] == turn
                ]
                shape = [(o, min(batch, cfg["rows"] - o))
                         for o in range(0, cfg["rows"], batch)]
                if [(d["offset"], d["batch"]) for d in dispatches] != shape:
                    raise ValueError("Incorrect equal-work dispatch plan")
            for d in episode["dispatches"]:
                if sorted(d["actual_rows"]) != list(
                        range(d["offset"], d["offset"] + d["batch"])):
                    raise ValueError("Dispatch membership differs")
                if d["graph"] != cfg.get("graph_replay", True):
                    raise ValueError("Unexpected graph execution mode")
                for field in ("gpu_ms", "service_ms", "commit_to_prepare_ms"):
                    if not math.isfinite(d[field]) or d[field] < 0:
                        raise ValueError("Invalid timing")
            if cfg.get("inspect_turn", -1) >= 0:
                inspected = episode.get("logits", [])
                if len(inspected
                       ) != 1 or inspected[0]["turn"] != cfg["inspect_turn"]:
                    raise ValueError("Missing inspected row logits")
                top = inspected[0]["top"]
                if len(top) != 8 or any(not math.isfinite(x["logit"])
                                        for x in top):
                    raise ValueError("Invalid inspected logits")
                if any(top[i]["logit"] < top[i + 1]["logit"]
                       for i in range(7)):
                    raise ValueError("Inspected logits are not ranked")
                if top[0]["token"] != episode["tokens"][0][cfg["inspect_turn"]
                                                           + 1]:
                    raise ValueError(
                        "Inspected argmax differs from committed token")
        all_tokens = [t for e in group for t in e["tokens"]]
        result[name] = {
            "first_tick_ms":
            statistics.mean(e["first_tick_ms"] for e in group),
            "two_tick_ms":
            statistics.mean(e["two_tick_ms"] for e in group),
            "first_gpu_ms":
            statistics.mean(
                sum(d["gpu_ms"] for d in e["dispatches"] if d["turn"] == 0)
                for e in group),
            "first_service_ms":
            statistics.mean(
                sum(d["service_ms"] for d in e["dispatches"] if d["turn"] == 0)
                for e in group),
            "commit_to_prepare_mean_ms":
            statistics.mean(d["commit_to_prepare_ms"] for e in group
                            for d in e["dispatches"] if d["turn"] > 0),
            "unique_sequences":
            len({tuple(t)
                 for t in all_tokens}),
            "sample_sequences":
            len(all_tokens),
            "reference_tokens":
            all_tokens[0],
        }
        if cfg.get("inspect_turn", -1) >= 0:
            result[name]["inspected_top"] = group[0]["logits"][0]["top"]
            result[name]["inspected_top2_margin"] = statistics.mean(
                e["logits"][0]["top"][0]["logit"] -
                e["logits"][0]["top"][1]["logit"] for e in group)
    reference = result["singleton"]["reference_tokens"]
    for variant, name in enumerate(("dense", "split", "singleton")):
        tokens = [
            t for e in episodes if e["variant"] == variant for t in e["tokens"]
        ]
        result[name]["identical_to_singleton"] = sum(t == reference
                                                     for t in tokens)
        result[name]["first_divergences"] = [
            next((i for i, (a, b) in enumerate(zip(t, reference)) if a != b),
                 None) for t in tokens
        ]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-root", type=pathlib.Path, required=True)
    parser.add_argument("--result-root", type=pathlib.Path, required=True)
    parser.add_argument("--inspection-only", action="store_true")
    args = parser.parse_args()
    repo = pathlib.Path(
        subprocess.check_output(["git", "rev-parse", "--show-toplevel"],
                                text=True).strip())
    root = args.result_root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    build = args.build_root.resolve()
    binary = build / "examples/llm/llm_phase_context_smoke"
    plugin = build / "libNvInfer_edgellm_plugin.so.1.0"
    trace = repo / ".local/results/gemma4-e2b-awq-full12-20260911/inputs/balanced.json"
    request = json.loads(trace.read_text())["requests"][4]
    manifest = {
        "state":
        "diagnostic",
        "status":
        "running",
        "source_commit":
        subprocess.check_output(["git", "rev-parse", "HEAD"],
                                text=True).strip(),
        "source_status":
        subprocess.check_output(["git", "status", "--porcelain"], text=True),
        "binary_sha256":
        serving.digest(binary),
        "plugin_sha256":
        serving.digest(plugin),
        "source_trace_sha256":
        serving.digest(trace),
        "container":
        serving.IMAGE,
        "cells": [],
        "note_references": [
            "notes/346-decode-logit-and-partition-realization-20260927.md"
            if args.inspection_only else
            "notes/345-async-decode-cohort-and-token-diagnostic-20260927.md"
        ],
        "summary_path":
        str(root / "summary.json")
    }
    (root / "source.patch").write_bytes(
        subprocess.check_output(["git", "diff", "HEAD"]))
    for source in (pathlib.Path(__file__),
                   repo / "examples/llm/phaseAsyncDecodeTrial.inc"):
        (root / source.name).write_bytes(source.read_bytes())
    summary = {}
    cells = (("gemma", "graph_logits", 8, 4, 15, 14, 2, True),
             ("gemma", "eager_logits", 8, 4, 15, 14, 2,
              False)) if args.inspection_only else (("gemma", "two_tick", 8, 4,
                                                     3, 1, 5, True),
                                                    ("gemma", "quality", 8, 4,
                                                     128, 127, 2, True),
                                                    ("cosmos", "two_tick", 64,
                                                     32, 3, 1, 5, True))
    for model, kind, rows, split, outputs, turns, rounds, graph_replay in cells:
        cell = root / (model + "-" + kind)
        cell.mkdir()
        config = serving.model_config(repo, model)

        def mounted(path):
            return "/workspace/" + str(path.relative_to(repo))

        trial = {
            "rows": rows,
            "split": split,
            "output_tokens": outputs,
            "split_turns": turns,
            "rounds": rounds,
            "graph_replay": graph_replay,
            "request": request,
            "output": mounted(cell / "raw.json")
        }
        if args.inspection_only:
            trial["inspect_turn"] = 12
        (cell / "trial.json").write_text(json.dumps(trial, indent=2) + "\n")
        env = {
            "TRT_PACKAGE_DIR": "/opt/tensorrt",
            "LD_LIBRARY_PATH": "/opt/tensorrt/lib:" + mounted(build),
            "EDGELLM_PLUGIN_PATH": mounted(plugin),
            "TRT_EDGELLM_ASYNC_DECODE_TRIAL": mounted(cell / "trial.json"),
            "TRT_EDGELLM_MAX_DECODE_BATCH": str(config["decode_batch"]),
            "TRT_EDGELLM_MAX_STABLE_SLOTS": str(config["stable_slots"]),
            "TRT_EDGELLM_MAX_INFLIGHT": str(config["stable_slots"]),
            "TRT_EDGELLM_IGNORE_EOS": "1",
            "TRT_EDGELLM_SEMANTIC_ONLY": "1",
            "TRT_EDGELLM_GLOBAL_SCHEDULER": "active",
            "TRT_EDGELLM_CAPTURE_PHASE_GRAPHS": "1"
        }
        command = [
            "docker", "run", "--rm", "--network", "none", "--gpus", "all",
            "--user", f"{os.getuid()}:{os.getgid()}", "-v",
            str(repo) + ":/workspace", "-w", "/workspace"
        ]
        for key, value in env.items():
            command += ["-e", key + "=" + value]
        command += [
            serving.IMAGE,
            mounted(binary),
            mounted(config["engine"]),
            mounted(config["hf"])
        ]
        record = {
            "name": cell.name,
            "command": command,
            "config": trial,
            "engine": str(config["engine"]),
            "engine_files": {
                p.name: serving.digest(p)
                for p in config["engine"].iterdir() if p.is_file()
            }
        }
        manifest["cells"].append(record)
        (root /
         "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print("Running", cell.name, flush=True)
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
            raise RuntimeError("Trial failed: " + str(cell))
        summary[cell.name] = summarize(
            json.loads((cell / "raw.json").read_text()))
        (root /
         "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    manifest["status"] = "complete"
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
