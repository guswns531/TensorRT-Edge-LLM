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
"""Recompute all serving metrics from retained per-run aggregates, never narrative scorecards."""

import argparse
import csv
import hashlib
import json
import math
import pathlib
import statistics
import subprocess
import sys

METRICS = (
    "generated_token_s_median",
    "ttft_mean_of_run_means_ms",
    "ttft_p95_median_ms",
    "tpot_mean_of_run_means_ms",
    "tpot_p95_median_ms",
    "e2e_mean_of_run_means_ms",
    "e2e_p95_median_ms",
)
LABELS = ("tok/s", "TTFT mean", "TTFT p95", "TPOT mean", "TPOT p95",
          "E2E mean", "E2E p95")
WORKLOADS = ("balanced", "mixed", "vision-heavy", "multi-image",
             "long-prefill", "bimodal", "decode-heavy", "short", "text-heavy",
             "poisson", "wave-drain", "late-vision")
DEFAULT_COSMOS_VLLM = pathlib.Path(".local/results/review-correction-20260926/"
                                   "cosmos-vllm-frozen-raw-corrected.json")


def file_identity(path):
    """Identify a retained JSON input without importing an inference framework."""
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()
    }


def load_frozen(path):
    """Accept corrected or historical Cosmos rows and Gemma per-workload aggregates."""
    if path.is_file():
        data = json.loads(path.read_text())
        return {
            row["workload"]: {
                "metrics": {
                    metric: row["vllm_" + metric]
                    for metric in METRICS
                },
                "repeat_count": row.get("success_runs"),
                "trace_sha256": row.get("trace_sha256"),
                "raw_origins": row.get("raw_origins", []),
                "identity": file_identity(path),
            }
            for row in data["rows"]
        }
    rows = {}
    for aggregate_path in sorted(path.glob("*/aggregate.json")):
        data = json.loads(aggregate_path.read_text())
        rows[aggregate_path.parent.name] = {
            "metrics": {
                metric: data[metric]
                for metric in METRICS
            },
            "repeat_count": data.get("repeats"),
            "trace_sha256": data.get("trace_sha256"),
            "identity": file_identity(aggregate_path),
        }
    if not rows:
        raise ValueError("No frozen baseline aggregates: " + str(path))
    return rows


def repeatability(runs):
    """Describe observations, not a proof of deterministic inference."""
    hashes = [
        value for run in runs
        for value in run.get("token_trace_sha256_per_run", [])
    ]
    return {
        "samples":
        len(hashes),
        "status":
        ("not_tested" if len(hashes) < 2 else
         "observed_equal" if len(set(hashes)) == 1 else "observed_different"),
    }


def compare_runs(runs, baseline):
    """Respect each metric's mean-of-means or median contract, retaining dispersion."""
    result = {
        "run_count": len(runs),
        "repeatability": repeatability(runs),
        "metrics": {},
        "baseline_repeat_count": baseline.get("repeat_count"),
    }
    for metric in METRICS:
        values = [float(run[metric]) for run in runs]
        reference = float(baseline["metrics"][metric])
        current = (statistics.mean(values) if "mean_of_run_means" in metric
                   else statistics.median(values))
        if not all(
                math.isfinite(value) and value > 0.0
                for value in values + [reference]):
            raise ValueError("Serving metric must be finite and positive: " +
                             metric)
        result["metrics"][metric] = {
            "current": current,
            "frozen_vllm": reference,
            "delta_percent": (current / reference - 1.0) * 100.0,
            "minimum": min(values),
            "maximum": max(values),
            "stddev": statistics.stdev(values) if len(values) > 1 else None,
        }
    throughput_reference = result["metrics"]["generated_token_s_median"][
        "frozen_vllm"]
    result["throughput_repeats"] = [{
        "generated_token_s_median":
        float(run["generated_token_s_median"]),
        "delta_percent":
        (float(run["generated_token_s_median"]) / throughput_reference - 1.0) *
        100.0,
        "beats_frozen_vllm":
        float(run["generated_token_s_median"]) > throughput_reference,
    } for run in runs]
    result["peak_memory_mib"] = max(
        run.get("gpu_memory_peak_mib_median", 0) for run in runs)
    return result


def collect_campaign(root, baselines, require_completed=False):
    """Load raw cell aggregates, preserving each artifact's identity and incomplete coverage."""
    groups = {}
    manifest_path = root / "manifest.json"
    manifest = json.loads(
        manifest_path.read_text()) if manifest_path.exists() else {}

    def record_key(record):
        return (record["model"], record["variant"], int(record["repeat"]),
                record["workload"])

    requested = {record_key(record) for record in manifest.get("commands", [])}
    completed = {
        record_key(record)
        for record in manifest.get("completed", [])
    }
    failed = {record_key(record) for record in manifest.get("failures", [])}
    if not require_completed:
        failed -= completed
    expected = len(requested)
    accepted = set()
    excluded = []
    for path in sorted(root.glob("*/*/repeat-*/*/aggregate.json")):
        model, variant, repeat, workload = path.relative_to(root).parts[:4]
        cell_key = (model, variant, int(repeat.split("-", 1)[1]), workload)
        reason = ("failed_cell" if cell_key in failed else "not_requested" if
                  requested and cell_key not in requested else "not_completed"
                  if require_completed and cell_key not in completed else None)
        if reason:
            excluded.append({"path": str(path), "reason": reason})
            continue
        data = json.loads(path.read_text())
        if (data.get("requested_output_tokens_per_run", 0) <= 0
                or data.get("generated_tokens_per_run_min")
                != data["requested_output_tokens_per_run"]):
            raise ValueError("Incomplete fixed-output workload: " + str(path))
        baseline = baselines[model][workload]
        if baseline.get("trace_sha256") and baseline[
                "trace_sha256"] != data.get("trace_sha256"):
            raise ValueError("Frozen vLLM workload hash differs: " + str(path))
        key = (model, variant, workload)
        group = groups.setdefault(key, {"runs": [], "origins": []})
        group["runs"].append(data)
        group["origins"].append(dict(file_identity(path), repeat=repeat))
        accepted.add(cell_key)
    rows = []
    for (model, variant, workload), group in sorted(groups.items()):
        row = compare_runs(group["runs"], baselines[model][workload])
        row.update(model=model,
                   variant=variant,
                   workload=workload,
                   origins=group["origins"])
        for repeat, origin in zip(row["throughput_repeats"], group["origins"]):
            repeat["origin"] = origin
        row["baseline_trace_identity"] = (
            "hash_matched" if baselines[model][workload].get("trace_sha256")
            else "not_available_in_frozen_summary")
        rows.append(row)
    if not rows and not requested:
        raise ValueError("No raw cell aggregates: " + str(root))
    retained_cells = sum(len(group["runs"]) for group in groups.values())
    return {
        "root": str(root.resolve()),
        "manifest":
        file_identity(manifest_path) if manifest_path.exists() else None,
        "expected_cells": expected or None,
        "retained_cells": retained_cells,
        "requested_cells_complete":
        requested == accepted if expected else None,
        "failed_cells": len(failed & requested),
        "missing_cells": len(requested - accepted) if expected else None,
        "excluded_aggregates": excluded,
        "rows": rows,
    }


def merged_cell_contract(root, manifest, record):
    """Validate recorded identities; only the cell's output destination may differ."""
    identity = manifest["identity"]
    global_fields = ("binary_source_commit", "binary_sha256", "plugin_sha256",
                     "container", "runner_sha256", "replay_tools_sha256")
    for field in global_fields:
        if not identity.get(field):
            raise ValueError("Missing runtime identity: " + field)
    model = identity["models"][record["model"]]
    for field in ("engine_sha256", "config_sha256", "vision_sha256",
                  "vision_config_sha256", "engine_sidecars_sha256",
                  "calibration_sha256", "effective_model_config",
                  "engine_builder_config", "vision_builder_config"):
        if not model.get(field):
            raise ValueError("Missing model identity: " + field)
    trace_hash = model["traces"][record["workload"]]
    if not trace_hash:
        raise ValueError("Missing workload hash")
    cell = (root / record["model"] / record["variant"] /
            ("repeat-%03d" % int(record["repeat"])) / record["workload"])
    if pathlib.Path(record["cell"]).resolve() != cell.resolve():
        raise ValueError("Cell path does not match its source campaign: " +
                         str(cell))
    command = list(record["command"])
    output_index = command.index("--output-dir") + 1
    if command[output_index] != record["cell"]:
        raise ValueError("Command output directory differs from cell")
    command[output_index] = "<CELL_OUTPUT>"
    output_mount = record["cell"] + ":/opt/results:rw"
    command = [
        "<CELL_OUTPUT>:/opt/results:rw" if token == output_mount else token
        for token in command
    ]
    environment = {}
    for index, token in enumerate(command[:-1]):
        if token == "-e":
            name, value = command[index + 1].split("=", 1)
            if name in environment:
                raise ValueError("Duplicate command environment variable: " +
                                 name)
            environment[name] = value
    if environment != record["effective_environment"]:
        raise ValueError("Recorded environment differs from command")
    return {
        "runtime": {
            field: identity[field]
            for field in global_fields
        },
        "model": {
            key: value
            for key, value in model.items() if key != "traces"
        },
        "trace_sha256": trace_hash,
        "command": command,
        "effective_environment": environment,
        "activation": manifest.get("activation"),
        "byte_budget": manifest.get("byte_budget"),
        "calibration_requests": manifest["calibration"][record["model"]],
        "record_options": {
            key: value
            for key, value in record.items()
            if key not in ("model", "variant", "workload", "repeat", "cell",
                           "command", "effective_environment")
        },
    }


def collect_merged_campaign(roots, baselines):
    """Pool compatible single-run cells, never campaign summary medians."""
    roots = [root.resolve() for root in roots]
    if not roots or len(set(roots)) != len(roots):
        raise ValueError("Merged campaigns need distinct source roots")
    contracts = {}
    groups = {}
    sources = []
    runtime_contract = None
    model_contracts = {}
    seen_origins = set()
    for source_index, root in enumerate(roots, 1):
        manifest = json.loads((root / "manifest.json").read_text())
        records = {}
        for record in manifest["commands"]:
            key = (record["model"], record["variant"], int(record["repeat"]),
                   record["workload"])
            if key in records:
                raise ValueError("Duplicate requested source cell: " +
                                 str(key))
            contract = merged_cell_contract(root, manifest, record)
            group_key = (record["model"], record["variant"],
                         record["workload"])
            if runtime_contract is not None and contract[
                    "runtime"] != runtime_contract:
                raise ValueError("Merged runtime identity mismatch: " +
                                 str(root))
            runtime_contract = contract["runtime"]
            model_key = record["model"]
            if model_key in model_contracts and model_contracts[
                    model_key] != contract["model"]:
                raise ValueError("Merged model identity mismatch: " +
                                 model_key)
            model_contracts[model_key] = contract["model"]
            if group_key in contracts and contracts[group_key] != contract:
                raise ValueError("Merged cell contract mismatch: " +
                                 str(group_key))
            contracts[group_key] = contract
            records[key] = record
        for field in ("completed", "failures"):
            seen = set()
            for record in manifest.get(field, []):
                key = (record["model"], record["variant"],
                       int(record["repeat"]), record["workload"])
                if key in seen or key not in records:
                    raise ValueError("Invalid " + field + " source cell: " +
                                     str(key))
                seen.add(key)
        source = collect_campaign(root, baselines, require_completed=True)
        source_id = "source-%02d:%s" % (source_index, root.name)
        for row in source["rows"]:
            group_key = (row["model"], row["variant"], row["workload"])
            group = groups.setdefault(group_key, {"runs": [], "origins": []})
            for origin in row["origins"]:
                path = pathlib.Path(origin["path"]).resolve()
                if path in seen_origins:
                    raise ValueError("Duplicate raw source cell: " + str(path))
                seen_origins.add(path)
                data = json.loads(path.read_text())
                if data.get("repeats") != 1:
                    raise ValueError(
                        "Merged aggregate must contain exactly one run: " +
                        str(path))
                if data.get("trace_sha256"
                            ) != contracts[group_key]["trace_sha256"]:
                    raise ValueError(
                        "Aggregate trace differs from manifest: " + str(path))
                group["runs"].append(data)
                group["origins"].append(
                    dict(origin,
                         source_root=str(root),
                         source_repeat=origin["repeat"],
                         repeat=source_id + "/" + origin["repeat"]))
        sources.append({
            key: value
            for key, value in source.items() if key != "rows"
        })
    rows = []
    for (model, variant, workload), group in sorted(groups.items()):
        row = compare_runs(group["runs"], baselines[model][workload])
        row.update(model=model,
                   variant=variant,
                   workload=workload,
                   origins=group["origins"])
        for repeat, origin in zip(row["throughput_repeats"], group["origins"]):
            repeat["origin"] = origin
        row["baseline_trace_identity"] = (
            "hash_matched" if baselines[model][workload].get("trace_sha256")
            else "not_available_in_frozen_summary")
        rows.append(row)
    return {
        "roots": [str(root) for root in roots],
        "manifest": [source["manifest"] for source in sources],
        "sources":
        sources,
        "runtime_identity":
        runtime_contract,
        "cell_contract_sha256": [{
            "model":
            key[0],
            "variant":
            key[1],
            "workload":
            key[2],
            "sha256":
            hashlib.sha256(json.dumps(value,
                                      sort_keys=True).encode()).hexdigest()
        } for key, value in sorted(contracts.items())],
        "expected_cells":
        sum(source["expected_cells"] or 0 for source in sources),
        "retained_cells":
        sum(source["retained_cells"] for source in sources),
        "requested_cells_complete":
        all(source["requested_cells_complete"] for source in sources),
        "failed_cells":
        sum(source["failed_cells"] for source in sources),
        "missing_cells":
        sum(source["missing_cells"] or 0 for source in sources),
        "excluded_aggregates":
        [item for source in sources for item in source["excluded_aggregates"]],
        "rows":
        rows,
    }


def summary_by_model(rows):
    """Keep throughput and each latency win count independent."""
    groups = {}
    for row in rows:
        groups.setdefault(row["model"] + "/" + row["variant"], []).append(row)
    result = {}
    for key, group in groups.items():
        missing = sorted(set(WORKLOADS) - {row["workload"] for row in group})
        minimum_repeats = min(row["run_count"] for row in group)
        result[key] = {
            "workloads": len(group),
            "missing_full12_workloads": missing,
            "minimum_repeats": minimum_repeats,
            "full12_three_repeat_coverage": not missing
            and minimum_repeats >= 3,
            "metrics": {},
        }
        for metric in METRICS:
            ratios = [
                row["metrics"][metric]["current"] /
                row["metrics"][metric]["frozen_vllm"] for row in group
            ]
            higher_better = metric == "generated_token_s_median"
            result[key]["metrics"][metric] = {
                "geomean_delta_percent":
                (math.exp(statistics.mean(math.log(ratio)
                                          for ratio in ratios)) - 1) * 100,
                "wins":
                sum(ratio > 1 if higher_better else ratio < 1
                    for ratio in ratios),
            }
    return result


def markdown(report):
    """Render paired values and delta for every metric, not throughput-only victory labels."""
    lines = [
        "# Raw-result workspace revalidation", "",
        "Latency means are arithmetic means of per-run means; throughput and latency p95 are medians across runs. "
        "Latency units are ms. "
        "Each cell is Current / frozen vLLM (relative change). Lower latency is better. "
        "A one-run token hash does not test repeatability; no confidence interval is claimed.",
        ""
    ]
    for label, campaign in report["campaigns"].items():
        lines += [
            "## " + label, "",
            "Retained/requested cells: %s/%s. Full-12 coverage and repeat counts are independent of metric wins."
            % (campaign["retained_cells"], campaign["expected_cells"]), ""
        ]
        if campaign.get("missing_cells"):
            lines += [
                "Warning: %d requested cells are missing; %d cells have unresolved failures. "
                "Only successful retained cells are tabulated; this is not a complete campaign."
                % (campaign["missing_cells"], campaign["failed_cells"]), ""
            ]
        unverified = sorted({
            row["model"]
            for row in campaign["rows"]
            if row["baseline_trace_identity"] != "hash_matched"
        })
        if unverified:
            lines += [
                "Warning: frozen baseline trace hashes are unavailable for %s. These comparisons reuse the "
                "documented historical contract, but this report cannot verify byte-identical traces."
                % ", ".join(unverified), ""
            ]
        for key, coverage in summary_by_model(campaign["rows"]).items():
            lines.append(
                "- %s: %d/12 workloads; minimum %d repeats; Full-12 ×3 coverage: %s."
                %
                (key, coverage["workloads"], coverage["minimum_repeats"],
                 "yes" if coverage["full12_three_repeat_coverage"] else "no"))
        lines += [
            "",
            "| Model/variant/workload | Runs | " + " | ".join(LABELS) + " |",
            "|---|---:|" + "---:|" * len(METRICS)
        ]
        for row in campaign["rows"]:
            cells = [
                "%.2f / %.2f (%+.2f%%)" %
                (row["metrics"][metric]["current"],
                 row["metrics"][metric]["frozen_vllm"],
                 row["metrics"][metric]["delta_percent"]) for metric in METRICS
            ]
            lines.append("| " + "/".join((row["model"], row["variant"],
                                          row["workload"])) + " | " +
                         str(row["run_count"]) + " | " + " | ".join(cells) +
                         " |")
        lines += [
            "", "### Throughput repeat margins", "",
            "Each retained cell is compared strictly against the same frozen baseline aggregate; "
            "a tie is not a win. These are not paired baseline runs or confidence intervals.",
            "",
            "| Model/variant/workload | Per-repeat tok/s (change; beats frozen) | Min | Max | Frozen | Wins/runs |",
            "|---|---|---:|---:|---:|---:|"
        ]
        for row in campaign["rows"]:
            repeats = row.get("throughput_repeats", [])
            if not repeats:
                continue
            values = row["metrics"]["generated_token_s_median"]
            cells = [
                "%s: %.2f (%+.2f%%; %s)" %
                (repeat.get("origin", {}).get("repeat", str(index + 1)),
                 repeat["generated_token_s_median"], repeat["delta_percent"],
                 "yes" if repeat["beats_frozen_vllm"] else "no")
                for index, repeat in enumerate(repeats)
            ]
            lines.append("| %s | %s | %.2f | %.2f | %.2f | %d/%d |" %
                         ("/".join(
                             (row["model"], row["variant"], row["workload"])),
                          "; ".join(cells), values["minimum"],
                          values["maximum"], values["frozen_vllm"],
                          sum(repeat["beats_frozen_vllm"]
                              for repeat in repeats), len(repeats)))
        lines.append("")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign",
                        action="append",
                        default=[],
                        help="LABEL=RESULT_ROOT")
    parser.add_argument(
        "--merged-campaign",
        action="append",
        default=[],
        help="LABEL=ROOT1,ROOT2,...; strict same-runtime single-run merge")
    parser.add_argument(
        "--gemma-vllm",
        type=pathlib.Path,
        default=pathlib.Path(
            ".local/results/gemma4-vllm-capacity-sweep-20260912/"
            "selected-seq24-kv480-p4096-g24-full12"))
    parser.add_argument("--cosmos-vllm",
                        type=pathlib.Path,
                        default=DEFAULT_COSMOS_VLLM)
    parser.add_argument("--output-prefix", type=pathlib.Path, required=True)
    args = parser.parse_args()
    if not args.campaign and not args.merged_campaign:
        parser.error(
            "At least one --campaign or --merged-campaign is required")
    baselines = {
        "gemma": load_frozen(args.gemma_vllm),
        "cosmos": load_frozen(args.cosmos_vllm)
    }
    report = {
        "aggregation": "mean_of_run_means_for_latency_means_otherwise_median",
        "baselines": baselines,
        "campaigns": {}
    }
    flat_rows = []
    assignments = [(item, False) for item in args.campaign
                   ] + [(item, True) for item in args.merged_campaign]
    for assignment, merged in assignments:
        label, path = assignment.split("=", 1)
        if label in report["campaigns"]:
            raise ValueError("Duplicate campaign label: " + label)
        campaign = (collect_merged_campaign(
            [pathlib.Path(root)
             for root in path.split(",")], baselines) if merged else
                    collect_campaign(pathlib.Path(path), baselines))
        campaign["summary"] = summary_by_model(campaign["rows"])
        report["campaigns"][label] = campaign
        for row in campaign["rows"]:
            flat = {
                key: row[key]
                for key in ("model", "variant", "workload", "run_count",
                            "peak_memory_mib")
            }
            flat["campaign"] = label
            flat["token_repeatability"] = row["repeatability"]["status"]
            for metric, values in row["metrics"].items():
                flat.update({
                    metric + "_" + field: value
                    for field, value in values.items()
                })
            flat_rows.append(flat)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    args.output_prefix.with_suffix(".json").write_text(
        json.dumps(report, indent=2) + "\n")
    args.output_prefix.with_suffix(".md").write_text(markdown(report))
    with args.output_prefix.with_suffix(".csv").open("w",
                                                     newline="") as stream:
        writer = csv.DictWriter(stream,
                                fieldnames=list(flat_rows[0]) if flat_rows else
                                ["campaign", "model", "workload"])
        writer.writeheader()
        writer.writerows(flat_rows)
    manifest = {
        "state":
        "diagnostic",
        "operation":
        "read_only_raw_result_reanalysis_no_gpu_measurement",
        "command": [sys.executable] + sys.argv,
        "source_commit":
        subprocess.check_output(["git", "rev-parse", "HEAD"],
                                text=True).strip(),
        "tracked_diff_sha256":
        hashlib.sha256(subprocess.check_output(["git", "diff",
                                                "HEAD"])).hexdigest(),
        "analyzer":
        file_identity(pathlib.Path(__file__)),
        "baseline_inputs": {
            model: {
                workload: baseline["identity"]
                for workload, baseline in workloads.items()
            }
            for model, workloads in baselines.items()
        },
        "campaign_manifests": {
            label: campaign["manifest"]
            for label, campaign in report["campaigns"].items()
        },
        "summary_path":
        str(args.output_prefix.with_suffix(".json")),
        "note_references": [
            "notes/328-dual-model-24-clean-sweep-20260921.md",
            "notes/329-gemma-vision-scaling-and-concurrency-resolution-20260921.md"
        ],
        "repeat_count":
        "recorded per workload; not inferred from singleton determinism flags",
    }
    args.output_prefix.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
