#!/usr/bin/env python3
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
"""Report retained v0.10 and fresh v0.10.1 policy results without hiding missing cells."""

import argparse
import csv
import json
import math
import pathlib
import statistics

METRICS = {
    "token_s": "generated_token_s_median",
    "ttft_mean_ms": "ttft_mean_of_run_means_ms",
    "ttft_p95_ms": "ttft_p95_median_ms",
    "tpot_mean_ms": "tpot_mean_of_run_means_ms",
    "tpot_p95_ms": "tpot_p95_median_ms",
    "e2e_mean_ms": "e2e_mean_of_run_means_ms",
    "e2e_p95_ms": "e2e_p95_median_ms",
}
POLICIES = ("exact", "scalar", "scalar-transition")
EQUAL_CAP_CASES = ("balanced", "bimodal", "decode-heavy", "long-prefill")


def aggregate_row(path, workload, variant, historical):
    data = json.loads(path.read_text())
    runs = [
        json.loads(run.read_text())
        for run in sorted(path.parent.glob("run-*/client/aggregate.json"))
    ]
    spread = {}
    metrics = {key: data[value] for key, value in METRICS.items()}
    if runs:
        for metric, field in METRICS.items():
            reducer = statistics.mean if metric.endswith(
                "_mean_ms") else statistics.median
            metrics[metric] = reducer(run[field] for run in runs)
        for metric in ("token_s", "e2e_mean_ms"):
            values = [run[METRICS[metric]] for run in runs]
            spread[metric + "_run_min"] = min(values)
            spread[metric + "_run_max"] = max(values)
    return {
        "workload":
        workload,
        "variant":
        variant,
        "historical":
        historical,
        "source":
        str(path.resolve()),
        "repeats":
        data["repeats"],
        "token_hashes":
        data.get("token_trace_sha256_per_run", []),
        "gpu_peak_mib":
        data.get("gpu_memory_peak_mib_median",
                 data.get("gpu_memory_peak_mib_max")),
        "metrics_recomputed_from_runs":
        bool(runs),
        **metrics,
        **spread,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=pathlib.Path, required=True)
    parser.add_argument("--current", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    commands = json.loads((args.current / "commands.json").read_text())
    workloads = sorted({record["case"] for record in commands})
    rows, missing = [], []
    for workload in workloads:
        for index, policy in enumerate(POLICIES):
            for version, root, directory in (
                ("old", args.results_root / "current", f"v{index}-full12"),
                ("current", args.current, policy),
            ):
                path = root / directory / "generic" / workload / "worker-4" / "aggregate.json"
                if not path.exists():
                    missing.append(f"{version}/V{index}/{workload}")
                    continue
                rows.append(
                    aggregate_row(path, workload, f"{version} V{index}",
                                  version == "old"))
    if missing and not args.allow_partial:
        raise RuntimeError("Missing results: " + ", ".join(missing))
    baseline_root = args.results_root / "v0101-forward-port"
    with (baseline_root /
          "244-old-current-vllm-latency-audit.csv").open() as stream:
        for record in csv.DictReader(stream):
            workload = record["workload"]
            if record["variant"] != "vllm" or workload not in workloads:
                continue
            if workload in EQUAL_CAP_CASES:
                path = baseline_root / "closure-gates" / "vllm-equal-cap-r1" / workload / "client/aggregate.json"
                rows.append(aggregate_row(path, workload, "frozen vLLM", True))
            else:
                frozen = {
                    "workload": workload,
                    "variant": "frozen vLLM",
                    "historical": True,
                    "repeats": int(record["repeats"]),
                    **{
                        key: float(record[key])
                        for key in METRICS
                    }
                }
                aliases = {
                    "multi-image": "multi",
                    "wave-drain": "wave",
                    "late-vision": "late"
                }
                root = args.results_root / "baselines/vllm-frozen-12x3" / aliases.get(
                    workload, workload)
                sources = sorted(root.glob("run-*/client/aggregate.json"))
                if len(sources) == frozen["repeats"]:
                    runs = [
                        json.loads(source.read_text()) for source in sources
                    ]
                    recomputed = []
                    for metric, field in METRICS.items():
                        if not all(field in run for run in runs):
                            continue
                        reducer = statistics.mean if metric.endswith(
                            "_mean_ms") else statistics.median
                        frozen[metric] = reducer(run[field] for run in runs)
                        recomputed.append(metric)
                    frozen["metrics_recomputed_from_runs"] = len(
                        recomputed) == len(METRICS)
                    frozen["recomputed_metrics"] = recomputed
                    frozen["aggregate_sources"] = [
                        str(source.resolve()) for source in sources
                    ]
                    peaks = [
                        run.get("gpu_memory_peak_mib_median",
                                run.get("gpu_memory_peak_mib_max"))
                        for run in runs
                    ]
                    if all(value is not None for value in peaks):
                        frozen["gpu_peak_mib"] = statistics.median(peaks)
                rows.append(frozen)
    fidelity, policy_effects = [], []
    for workload in workloads:
        variants = {
            row["variant"]: row
            for row in rows
            if row["workload"] == workload and "token_hashes" in row
        }
        for index in range(3):
            old = variants.get(f"old V{index}")
            current = variants.get(f"current V{index}")
            if old is not None and current is not None:
                fidelity.append({
                    "workload":
                    workload,
                    "policy":
                    f"V{index}",
                    "current_repeat_identity":
                    len(set(current["token_hashes"])) == 1,
                    "cross_version_observed_hash_sets_equal":
                    set(old["token_hashes"]) == set(current["token_hashes"])
                })
        for version in ("old", "current"):
            baseline = variants.get(f"{version} V0")
            if baseline is None:
                continue
            for index in (1, 2):
                row = variants.get(f"{version} V{index}")
                if row is not None:
                    policy_effects.append({
                        "workload":
                        workload,
                        "version":
                        version,
                        "policy":
                        f"V{index}",
                        "throughput_vs_v0_percent":
                        (row["token_s"] / baseline["token_s"] - 1) * 100
                    })
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(
        json.dumps(
            {
                "missing": missing,
                "rows": rows,
                "fidelity": fidelity,
                "policy_effects": policy_effects
            },
            indent=2) + "\n")
    with (args.output / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(dict.fromkeys(key for row in rows for key in row)))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Legacy parity comparison", "", f"Missing cells: {len(missing)}", "",
        "Old and vLLM are retained measurements; current is fresh. Not a paired KV-only experiment.",
        "",
        "Latency means are recomputed as arithmetic means from individual runs when available; "
        "the historical outer harness used medians even for fields named mean. "
        "p95 and throughput are medians across runs, not pooled p95.", "",
        "| Workload | Variant | n | token/s | TTFT mean/p95 ms | TPOT mean/p95 ms | E2E mean/p95 ms |",
        "|---|---|---:|---:|---:|---:|---:|"
    ]
    for row in sorted(rows,
                      key=lambda item: (item["workload"], item["variant"])):
        pairs = [
            f"{row[f'{metric}_mean_ms']:.2f}/{row[f'{metric}_p95_ms']:.2f}"
            for metric in ("ttft", "tpot", "e2e")
        ]
        lines.append(
            f"| {row['workload']} | {row['variant']} | {row['repeats']} | "
            f"{row['token_s']:.2f} | {' | '.join(pairs)} |")
    lines.extend([
        "", "## Within-version policy effect", "",
        "Historical comparisons are not causal paired experiments.", "",
        "| Workload | Version | Policy | Throughput vs V0 |",
        "|---|---|---|---:|"
    ])
    for effect in policy_effects:
        lines.append(
            f"| {effect['workload']} | {effect['version']} | {effect['policy']} | "
            f"{effect['throughput_vs_v0_percent']:+.2f}% |")
    lines.extend([
        "", "## Observed GPU memory", "",
        "Sampled process/device peak, not allocator accounting. Missing frozen values are not inferred.",
        "", "| Workload | Variant | Peak MiB |", "|---|---|---:|"
    ])
    for row in sorted(rows,
                      key=lambda item: (item["workload"], item["variant"])):
        if row.get("gpu_peak_mib") is not None:
            lines.append(
                f"| {row['workload']} | {row['variant']} | {row['gpu_peak_mib']:.0f} |"
            )
    lines.extend([
        "", "## Output fidelity", "",
        "Within-current identity below is across observed repeats, not proof of semantic correctness.",
        "",
        "| Workload | Policy | Current repeat hashes equal | Old/current hash sets equal |",
        "|---|---|---|---|"
    ])
    for item in fidelity:
        lines.append(
            f"| {item['workload']} | {item['policy']} | {item['current_repeat_identity']} | "
            f"{item['cross_version_observed_hash_sets_equal']} |")
    lines.extend([
        "", "## Per-policy throughput summary", "",
        "Geometric mean of workload ratios; partial rows are explicitly counted. No confidence interval inferred.",
        "",
        "| Policy | Matched workloads | vs old same policy | vs frozen vLLM | Wins vs vLLM |",
        "|---|---:|---:|---:|---:|"
    ])
    for index in range(3):
        old_ratios, vllm_ratios = [], []
        for workload in workloads:
            by_variant = {
                row["variant"]: row
                for row in rows if row["workload"] == workload
            }
            current = by_variant.get(f"current V{index}")
            old = by_variant.get(f"old V{index}")
            vllm = by_variant.get("frozen vLLM")
            if current and old and vllm:
                old_ratios.append(current["token_s"] / old["token_s"])
                vllm_ratios.append(current["token_s"] / vllm["token_s"])
        if old_ratios:
            old_gain = 100 * (
                math.exp(sum(map(math.log, old_ratios)) / len(old_ratios)) - 1)
            vllm_gain = 100 * (math.exp(
                sum(map(math.log, vllm_ratios)) / len(vllm_ratios)) - 1)
            lines.append(
                f"| V{index} | {len(old_ratios)}/{len(workloads)} | {old_gain:+.2f}% | "
                f"{vllm_gain:+.2f}% | {sum(value > 1 for value in vllm_ratios)} |"
            )
    lines.extend([
        "", "## Observed repeat variation", "",
        "Run minima/maxima, not confidence intervals or pooled request percentiles.",
        "",
        "| Workload | Variant | token/s min–max | E2E run-mean min–max ms |",
        "|---|---|---:|---:|"
    ])
    for row in sorted(rows,
                      key=lambda item: (item["workload"], item["variant"])):
        if row["variant"].startswith("current") and "token_s_run_min" in row:
            lines.append(
                f"| {row['workload']} | {row['variant']} | "
                f"{row['token_s_run_min']:.2f}–{row['token_s_run_max']:.2f} | "
                f"{row['e2e_mean_ms_run_min']:.2f}–{row['e2e_mean_ms_run_max']:.2f} |"
            )
    variants = [
        f"{version} V{index}" for version in ("old", "current")
        for index in range(3)
    ] + ["frozen vLLM"]
    lines.extend([
        "", "## Throughput matrix (token/s)", "",
        "| Workload | " + " | ".join(variants) + " |",
        "|---|" + "---:|" * len(variants)
    ])
    for workload in workloads:
        matched = {
            row["variant"]: row["token_s"]
            for row in rows if row["workload"] == workload
        }
        values = [
            f"{matched[variant]:.2f}" if variant in matched else "pending"
            for variant in variants
        ]
        lines.append(f"| {workload} | {' | '.join(values)} |")
    (args.output / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"Reported {len(rows)} rows; missing {len(missing)} cells")


if __name__ == "__main__":
    main()
