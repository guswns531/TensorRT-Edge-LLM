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
"""Attribute two-model lifetime-admission trade-offs and preserve compressed raw logs."""

import argparse
import collections
import csv
import gzip
import hashlib
import json
import pathlib
import statistics
import subprocess

SERVING_METRICS = ("generated_token_s_median", "achieved_req_s_median",
                   "ttft_mean_of_run_means_ms", "ttft_p95_median_ms",
                   "tpot_mean_of_run_means_ms", "tpot_p95_median_ms",
                   "e2e_mean_of_run_means_ms", "e2e_p95_median_ms",
                   "gpu_memory_peak_mib_median")


def summarize_reports(sources, excluded_keys=()):
    """Aggregate explicit evidence sources without silently overwriting a confounded run."""
    groups = {}
    for root, reports in sources:
        for key, report in reports.items():
            if key in excluded_keys and root == sources[0][0]:
                continue
            group = "/".join(key.split("/")[:3])
            groups.setdefault(group, []).append((str(root), key, report))
    rows = {}
    for group, evidence in groups.items():
        metrics = {}
        for metric in SERVING_METRICS:
            values = [r["serving"][metric] for _, _, r in evidence]
            metrics[metric] = {
                "mean": statistics.mean(values),
                "min": min(values),
                "max": max(values)
            }
        rows[group] = {
            "success_runs":
            len(evidence),
            "metrics":
            metrics,
            "origins": [{
                "root": root,
                "key": key,
                "aggregate_sha256": r["aggregate_sha256"]
            } for root, key, r in evidence]
        }
    for group, row in rows.items():
        model, workload, _ = group.split("/")
        row["delta_percent"] = {}
        for variant in ("static-base", "static-large", "static-slot",
                        "lifetime"):
            reference = rows.get("/".join((model, workload, variant)))
            if reference is not None:
                row["delta_percent"][variant] = {
                    metric:
                    100.0 * (row["metrics"][metric]["mean"] /
                             reference["metrics"][metric]["mean"] - 1.0)
                    for metric in SERVING_METRICS
                    if reference["metrics"][metric]["mean"] > 0
                }
    return rows


def log_stream(path):
    """Open a retained raw or losslessly compressed backend log."""
    if path.exists():
        return path.open()
    return gzip.open(str(path) + ".gz", "rt")


def encoder_overlap_percent(masks):
    """Separate epoch-normalized concurrency from overlap conditional on E being active."""
    total = sum(masks.values())
    active = sum(t for mask, t in masks.items() if int(mask, 2) & 1)
    ep = sum(t for mask, t in masks.items() if int(mask, 2) & 3 == 3)
    ed = sum(t for mask, t in masks.items() if int(mask, 2) & 5 == 5)
    paired = sum(t for mask, t in masks.items()
                 if int(mask, 2) & 1 and int(mask, 2) & 6)
    return {
        "e_active_epoch_percent": 100 * active / total if total else 0,
        "ep_epoch_percent": 100 * ep / total if total else 0,
        "ed_epoch_percent": 100 * ed / total if total else 0,
        "e_overlapped_fraction_percent": 100 * paired / active if active else 0
    }


def distribution(values):
    """Return descriptive mean and interpolated p95."""
    if not values:
        return {"samples": 0}
    ordered = sorted(values)
    position = 0.95 * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return {
        "samples":
        len(values),
        "mean":
        statistics.mean(values),
        "p95":
        ordered[lower] + (position - lower) *
        (ordered[upper] - ordered[lower]),
        "max":
        max(values)
    }


def analyze_cell(cell):
    """Read only the measurement epoch; shared-slab accounting is emitted by the runtime."""
    metrics = []
    encoder_batches = []
    vision_queued = {}
    encoder_wait_ms = []
    epoch = 0
    with log_stream(cell / "run-001/gateway.log") as stream:
        for line in stream:
            if "PHASE_EPOCH\t" in line:
                epoch = json.loads(line.split("PHASE_EPOCH\t", 1)[1])["epoch"]
            if epoch == 1 and "PHASE_METRIC\t" in line:
                metrics.append(json.loads(line.split("PHASE_METRIC\t", 1)[1]))
            if epoch == 1 and "PHASE_ENCODER_METRIC\t" in line:
                encoder_batches.append(
                    json.loads(line.split("PHASE_ENCODER_METRIC\t", 1)[1]))
            if epoch == 1 and "PHASE_TIMELINE\t" in line:
                event = json.loads(line.split("PHASE_TIMELINE\t", 1)[1])
                request = event["request_index"]
                if event["stage"] == "vision_queued":
                    vision_queued[request] = event["timestamp_us"]
                elif event[
                        "stage"] == "encoder_start" and request in vision_queued:
                    encoder_wait_ms.append(
                        (event["timestamp_us"] - vision_queued.pop(request)) /
                        1000.0)
    masks = collections.defaultdict(float)
    with (cell / "run-001/activity-segments.csv").open() as stream:
        for row in csv.DictReader(stream):
            masks[row["binary_mask"]] += float(row["duration_ms"])
    total = sum(masks.values())
    if total <= 0 or not metrics:
        raise ValueError("Missing measured activity or phase metrics")
    dispatches = {}
    for phase in ("prefill", "decode"):
        selected = [m for m in metrics if m.get(phase + "_batch", 0) > 0]
        dispatches[phase] = {
            "count":
            len(selected),
            "mean_batch":
            statistics.mean(m[phase + "_batch"]
                            for m in selected) if selected else 0,
            "histogram":
            dict(collections.Counter(m[phase + "_batch"] for m in selected)),
            "useful_tokens":
            sum(m.get(phase + "_tokens", 0) for m in selected)
        }
    dispatches["encoder"] = {
        "count":
        len(encoder_batches),
        "mean_batch":
        statistics.mean(m["batch_size"]
                        for m in encoder_batches) if encoder_batches else 0,
        "histogram":
        dict(collections.Counter(m["batch_size"] for m in encoder_batches)),
        "total_gpu_ms":
        sum(m["execution_gpu_ms"] for m in encoder_batches),
        "gpu_ms":
        distribution([m["execution_gpu_ms"] for m in encoder_batches])
    }
    preparation_policy_batches = []
    preparation_transition_decode_rows = []
    preparation_transition_horizon_ms = []
    previous_evaluations = 0
    for metric in metrics:
        evaluations = metric.get(
            "vision_encoder_preparation_policy_evaluations", 0)
        if evaluations > previous_evaluations:
            new_evaluations = evaluations - previous_evaluations
            preparation_policy_batches.extend(
                [metric.get("vision_encoder_preparation_policy_batch", 0)] *
                new_evaluations)
            transition_rows = metric.get(
                "vision_encoder_preparation_transition_decode_rows", 0)
            transition_horizon = metric.get(
                "vision_encoder_preparation_transition_horizon_ms", 0)
            if transition_rows > 0 and transition_horizon > 0:
                preparation_transition_decode_rows.extend([transition_rows] *
                                                          new_evaluations)
                preparation_transition_horizon_ms.extend([transition_horizon] *
                                                         new_evaluations)
            previous_evaluations = evaluations
    aggregate = json.loads((cell / "aggregate.json").read_text())
    requests = aggregate["requests_per_run"]
    outputs = aggregate["requested_output_tokens_per_run"]
    if dispatches["decode"]["useful_tokens"] != outputs - requests:
        raise ValueError("Decode token accounting mismatch")
    with (cell / "run-001/client/run-001/requests.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != requests or any(
            int(r["http_status"]) != 200 or r["error"] for r in rows):
        raise ValueError("Incomplete or failed HTTP request")
    maxima = {
        key: max(m.get(key, 0) for m in metrics)
        for key in ("vision_admission_budget_bytes",
                    "vision_admission_retained_bytes",
                    "vision_admission_reserved_bytes",
                    "vision_admission_byte_blocks", "vision_downstream",
                    "vision_encoder_preparation_policy_evaluations",
                    "vision_encoder_preparation_policy_coverage_misses",
                    "vision_encoder_preparation_policy_selection_changes",
                    "vision_encoder_preparation_policy_applied_changes")
    }
    return {
        "analysis_schema":
        6,
        "serving":
        aggregate,
        "dispatches":
        dispatches,
        "maxima":
        maxima,
        "last_sampled_retained_bytes":
        metrics[-1].get("vision_admission_retained_bytes", 0),
        "calibration_records":
        json.loads((cell / "run-001/client/calibration.json").read_text()),
        "mask_percent": {
            format(mask, "04b"):
            masks.get(format(mask, "04b"), 0.0) / total * 100
            for mask in range(16)
        },
        "activity_span_ms":
        total,
        "encoder_overlap_percent":
        encoder_overlap_percent(masks),
        "encoder_queue_wait_ms":
        distribution(encoder_wait_ms),
        "encoder_preparation_policy": {
            "evaluations":
            previous_evaluations,
            "batch_histogram":
            dict(collections.Counter(preparation_policy_batches)),
            "coverage_misses":
            maxima["vision_encoder_preparation_policy_coverage_misses"],
            "selection_changes":
            maxima["vision_encoder_preparation_policy_selection_changes"],
            "applied_changes":
            maxima["vision_encoder_preparation_policy_applied_changes"],
            "transition_decode_rows":
            distribution(preparation_transition_decode_rows),
            "transition_horizon_ms":
            distribution(preparation_transition_horizon_ms)
        },
        "scheduler_decision_us":
        distribution([m["host_scheduler_decision_us"] for m in metrics]),
        "arrival_ttft_ms":
        distribution([
            (float(r["first_token_us"]) - float(r["scheduled_arrival_us"])) /
            1000.0 for r in rows
        ]),
        "arrival_e2e_ms":
        distribution([
            (float(r["completed_us"]) - float(r["scheduled_arrival_us"])) /
            1000.0 for r in rows
        ])
    }


def main():
    """Analyze completed cells only; compression never touches a running backend's log."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=pathlib.Path, required=True)
    parser.add_argument("--compress-completed-logs", action="store_true")
    parser.add_argument("--exclude-key", action="append", default=[])
    parser.add_argument("--supplement-result-root",
                        action="append",
                        type=pathlib.Path,
                        default=[])
    args = parser.parse_args()
    root = args.result_root.resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    previous_path = root / "analysis.json"
    previous = json.loads(
        previous_path.read_text()) if previous_path.exists() else {}
    reports = {}
    for record in manifest["completed"]:
        cell = pathlib.Path(record["cell"]).resolve()
        if root not in cell.parents:
            raise ValueError("Completed cell is outside this campaign")
        key = "/".join((record["model"], record["workload"], record["variant"],
                        str(record["repeat"])))
        aggregate_hash = hashlib.sha256(
            (cell / "aggregate.json").read_bytes()).hexdigest()
        cached = previous.get(key, {})
        if cached.get("aggregate_sha256") == aggregate_hash and cached.get(
                "analysis_schema") == 6:
            reports[key] = cached
        else:
            reports[key] = analyze_cell(cell)
            reports[key]["aggregate_sha256"] = aggregate_hash
            if "uncompressed_log_sha256" in cached:
                reports[key]["uncompressed_log_sha256"] = cached[
                    "uncompressed_log_sha256"]
        log = cell / "run-001/gateway.log"
        if args.compress_completed_logs and log.exists():
            with log.open("rb") as stream:
                value = hashlib.sha256()
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    value.update(chunk)
            reports[key]["uncompressed_log_sha256"] = value.hexdigest()
            subprocess.run(["gzip", "-1", "--", str(log)], check=True)
    (root / "analysis.json").write_text(json.dumps(reports, indent=2) + "\n")
    failures = []
    for record in manifest.get("failures", []):
        cell = pathlib.Path(record["cell"]).resolve()
        if root not in cell.parents:
            raise ValueError("Failed cell is outside this campaign")
        log = cell / "run-001/gateway.log"
        if not log.exists() and not pathlib.Path(str(log) + ".gz").exists():
            continue
        errors = collections.deque(maxlen=8)
        with log_stream(log) as stream:
            for line in stream:
                if "out of memory" in line or "what():" in line or "terminate called" in line:
                    errors.append(line.strip())
        failures.append({**record, "backend_error_lines": list(errors)})
        if args.compress_completed_logs and log.exists():
            subprocess.run(["gzip", "-1", "--", str(log)], check=True)
    (root / "failure-analysis.json"
     ).write_text(json.dumps(failures, indent=2) + "\n")
    sources = [(root, reports)]
    for supplemental in args.supplement_result_root:
        source = supplemental.resolve()
        source_manifest = json.loads((source / "manifest.json").read_text())
        if source_manifest["identity"]["binary_sha256"] != manifest[
                "identity"]["binary_sha256"]:
            raise ValueError(
                "Supplemental evidence uses a different runtime binary")
        if source_manifest["identity"]["tracked_diff_sha256"] != manifest[
                "identity"]["tracked_diff_sha256"]:
            raise ValueError(
                "Supplemental evidence uses a different source patch")
        for model, identity in source_manifest["identity"]["models"].items():
            reference = manifest["identity"]["models"][model]
            for field in ("engine_sha256", "config_sha256", "vision_sha256",
                          "calibration_sha256"):
                if identity[field] != reference[field]:
                    raise ValueError(
                        "Supplemental model/calibration identity differs")
        sources.append(
            (source, json.loads((source / "analysis.json").read_text())))
    comparison = {
        "aggregation":
        "arithmetic mean of independent run metrics; min/max retain run spread",
        "excluded_primary_keys": args.exclude_key,
        "rows": summarize_reports(sources, args.exclude_key),
        "failures": failures
    }
    (root /
     "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
    print("Analyzed", len(reports), "completed cells", flush=True)


if __name__ == "__main__":
    main()
