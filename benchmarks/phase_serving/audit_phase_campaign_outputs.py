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
"""Audit retained HTTP outputs without running inference or claiming semantic correctness."""

import argparse
import csv
import hashlib
import json
import pathlib
import sys


def identity(path):
    """Identify the exact retained input inspected by this audit."""
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()
    }


def model_eos(model, model_identity):
    """Read the pinned engine EOS contract, retaining Gemma's end-of-turn diagnostic token."""
    config_path = pathlib.Path(model_identity["engine"]) / "config.json"
    origin = identity(config_path)
    if origin["sha256"] != model_identity["config_sha256"]:
        raise ValueError("Engine config changed since campaign: " +
                         str(config_path))
    eos = json.loads(config_path.read_text())["eos_token_id"]
    tokens = set(eos if isinstance(eos, list) else [eos])
    if model == "gemma":
        tokens.update((1, 50, 106))
    return sorted(int(token) for token in tokens), origin


def audit_cell(record, model_identity, eos_tokens):
    """Inspect every request row; total output-token equality alone is insufficient."""
    cell = pathlib.Path(record["cell"])
    report = {
        key: record[key]
        for key in ("model", "variant", "workload", "repeat")
    }
    report.update(cell=str(cell),
                  issues=[],
                  requests=[],
                  first_eos_requests=[])

    def issue(category, message, request_id=None):
        report["issues"].append({
            "category": category,
            "message": message,
            "request_id": request_id
        })

    try:
        command = record["command"]
        trace_path = pathlib.Path(command[command.index("--trace") + 1])
        report["trace"] = identity(trace_path)
        expected_hash = model_identity["traces"][record["workload"]]
        if report["trace"]["sha256"] != expected_hash:
            raise ValueError("Trace changed since campaign")
        expected = json.loads(trace_path.read_text())["requests"]
        report["expected_requests"] = len(expected)
        aggregate_path = cell / "aggregate.json"
        report["aggregate"] = identity(aggregate_path)
        aggregate = json.loads(aggregate_path.read_text())
        if aggregate.get("trace_sha256") != expected_hash or aggregate.get(
                "repeats") != 1:
            raise ValueError(
                "Expected one matching trace run per completed campaign cell")
        csv_paths = sorted(cell.glob("run-*/client/run-*/requests.csv"))
        if len(csv_paths) != 1:
            raise ValueError("Expected exactly one per-request CSV, found %d" %
                             len(csv_paths))
        report["request_csv"] = identity(csv_paths[0])
        with csv_paths[0].open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        report["observed_requests"] = len(rows)
        seen = set()
        for row in rows:
            try:
                request_id = int(row["request_id"])
                if request_id in seen or not 0 <= request_id < len(expected):
                    issue("request_identity",
                          "Duplicate or unexpected request index", request_id)
                    continue
                seen.add(request_id)
                target = int(expected[request_id]["max_generate_length"])
                output_count = int(row["output_tokens"])
                token_ids = [
                    int(token) for token in row["output_token_ids"].split()
                ]
                status = int(row["http_status"])
                transport_error = row["error"]
                if status != 200 or transport_error:
                    issue("transport",
                          "HTTP %d: %s" % (status, transport_error),
                          request_id)
                if int(row["max_output_tokens"]
                       ) != target or output_count != target or target <= 0:
                    issue("output_count",
                          "Trace/request/generated token counts differ",
                          request_id)
                captured_complete = len(
                    token_ids) == output_count and output_count > 0
                if not captured_complete:
                    issue("token_capture",
                          "Captured token IDs do not cover generated output",
                          request_id)
                first_token = token_ids[0] if token_ids else None
                if first_token in eos_tokens:
                    report["first_eos_requests"].append(request_id)
                stop_index = next((index
                                   for index, token in enumerate(token_ids)
                                   if token in eos_tokens), None)
                prefix = token_ids[:stop_index +
                                   1] if stop_index is not None else token_ids
                report["requests"].append({
                    "request_id":
                    request_id,
                    "expected_tokens":
                    target,
                    "output_tokens":
                    output_count,
                    "captured_token_ids":
                    len(token_ids),
                    "capture_complete":
                    captured_complete,
                    "first_token_id":
                    first_token,
                    "token_ids_sha256":
                    hashlib.sha256(
                        json.dumps(token_ids,
                                   separators=(",",
                                               ":")).encode()).hexdigest(),
                    "first_stop_index":
                    stop_index,
                    "first_stop_token_id":
                    token_ids[stop_index] if stop_index is not None else None,
                    "through_first_stop_token_count":
                    len(prefix),
                    "through_first_stop_sha256":
                    hashlib.sha256(
                        json.dumps(prefix,
                                   separators=(",",
                                               ":")).encode()).hexdigest()
                    if captured_complete else None,
                })
            except (KeyError, TypeError, ValueError) as error:
                issue("malformed_request", str(error), row.get("request_id"))
        missing = sorted(set(range(len(expected))) - seen)
        if missing or len(rows) != len(expected):
            issue(
                "request_count", "Missing indices %s; rows=%d expected=%d" %
                (missing, len(rows), len(expected)))
        report["requests"].sort(key=lambda request: request["request_id"])
        generated = sum(request["output_tokens"]
                        for request in report["requests"])
        captured = sum(request["captured_token_ids"]
                       for request in report["requests"])
        target_total = sum(
            int(request["max_generate_length"]) for request in expected)
        if (aggregate.get("requests_per_run") != len(expected)
                or aggregate.get("requested_output_tokens_per_run")
                != target_total
                or aggregate.get("generated_tokens_per_run_min") != generated
                or aggregate.get("captured_token_ids_per_run") != [captured]):
            issue("aggregate_contract",
                  "Aggregate totals differ from trace or per-request CSV")
    except (OSError, KeyError, TypeError, ValueError) as error:
        issue("missing_or_invalid_artifact", str(error))
    report["integrity_passed"] = not report["issues"]
    report["first_eos_count"] = len(report["first_eos_requests"])
    return report


def compare_repeats(cells):
    """Compare complete token sequences per request, never empty capture hashes."""
    groups = {}
    for cell in cells:
        groups.setdefault((cell["model"], cell["variant"], cell["workload"]),
                          []).append(cell)
    comparisons = []
    for (model, variant, workload), group in sorted(groups.items()):
        group.sort(key=lambda cell: cell["repeat"])
        by_repeat = [{
            row["request_id"]: row
            for row in cell["requests"]
        } for cell in group]
        request_ids = sorted(
            {request_id
             for rows in by_repeat
             for request_id in rows})
        comparable = []
        mismatches = []
        prefix_comparable = []
        prefix_mismatches = []
        for request_id in request_ids:
            rows = [repeat.get(request_id) for repeat in by_repeat]
            if len(rows) < 2 or not all(row and row["capture_complete"]
                                        for row in rows):
                continue
            comparable.append(request_id)
            if len({row["token_ids_sha256"] for row in rows}) > 1:
                mismatches.append(request_id)
            if all(row["through_first_stop_token_count"] > 0
                   and row["through_first_stop_sha256"] for row in rows):
                prefix_comparable.append(request_id)
                if len({row["through_first_stop_sha256"] for row in rows}) > 1:
                    prefix_mismatches.append(request_id)

        def result(compared, different):
            status = ("not_tested" if len(group) < 2 else "observed_different"
                      if different else "invalid_or_incomplete"
                      if len(compared) != len(request_ids) or not all(
                          cell["integrity_passed"]
                          for cell in group) else "observed_equal")
            return {
                "status":
                status,
                "comparable_requests":
                len(compared),
                "different_request_ids":
                different,
                "exact_request_agreement": (len(compared) - len(different)) /
                len(compared) if compared else None,
            }

        comparisons.append({
            "model":
            model,
            "variant":
            variant,
            "workload":
            workload,
            "repeats": [cell["repeat"] for cell in group],
            **result(comparable, mismatches),
            "through_first_stop":
            result(prefix_comparable, prefix_mismatches),
        })
    return comparisons


def audit_campaign(root):
    """Audit completed cells while keeping failed and unfinished coverage visible."""
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    models = {}
    for model, model_identity in manifest["identity"]["models"].items():
        tokens, config = model_eos(model, model_identity)
        models[model] = {"eos_tokens": tokens, "config": config}

    def key(record):
        return tuple(record[field]
                     for field in ("model", "variant", "workload", "repeat"))

    requested = {key(record): record for record in manifest["commands"]}
    completed = {key(record) for record in manifest.get("completed", [])}
    if not completed.issubset(requested):
        raise ValueError(
            "Completed cells are not part of the requested campaign")
    cells = [
        audit_cell(requested[cell_key],
                   manifest["identity"]["models"][cell_key[0]],
                   models[cell_key[0]]["eos_tokens"])
        for cell_key in sorted(completed)
    ]
    issues = sum(len(cell["issues"]) for cell in cells)
    return {
        "operation":
        "retained_request_output_audit_no_inference",
        "manifest":
        identity(manifest_path),
        "models":
        models,
        "requested_cells":
        len(requested),
        "completed_cells":
        len(completed),
        "missing_cells":
        len(requested.keys() - completed),
        "unresolved_failed_cells":
        len({key(record)
             for record in manifest.get("failures", [])} - completed),
        "integrity_issue_count":
        issues,
        "integrity_passed":
        bool(requested) and requested.keys() == completed and issues == 0,
        "first_eos_count":
        sum(cell["first_eos_count"] for cell in cells),
        "semantics":
        "not_evaluated; first EOS is an anomaly flag under forced ignore-EOS, not proof of corruption",
        "comparison_contract":
        "raw compares all captured tokens; through_first_stop includes the first model EOS/stop token, "
        "or the full capture when absent; both require complete nonempty capture and at least two repeats",
        "cells":
        cells,
        "cross_repeat":
        compare_repeats(cells),
    }


def audit_campaigns(roots):
    """Combine output audits without asserting cross-source runtime equivalence."""
    roots = [root.resolve() for root in roots]
    if not roots or len(set(roots)) != len(roots):
        raise ValueError("Output audits need distinct campaign roots")
    sources = []
    cells = []
    seen_cells = set()
    seen_csvs = set()
    token_contracts = {}
    for source_index, root in enumerate(roots, 1):
        source = audit_campaign(root)
        source_id = "source-%02d:%s" % (source_index, root.name)
        for cell in source["cells"]:
            cell_path = str(pathlib.Path(cell["cell"]).resolve())
            csv_path = cell.get("request_csv", {}).get("path")
            if cell_path in seen_cells or (csv_path and csv_path in seen_csvs):
                raise ValueError("Duplicate source output artifact: " +
                                 cell_path)
            seen_cells.add(cell_path)
            if csv_path:
                seen_csvs.add(csv_path)
            group = (cell["model"], cell["variant"], cell["workload"])
            if cell.get("trace"):
                contract = (cell["trace"]["sha256"],
                            tuple(
                                source["models"][cell["model"]]["eos_tokens"]))
                if group in token_contracts and token_contracts[
                        group] != contract:
                    raise ValueError(
                        "Cross-source trace/stop contract mismatch: " +
                        str(group))
                token_contracts[group] = contract
            local_repeat = cell["repeat"]
            run_id = source_id + "/repeat-%03d" % local_repeat
            cells.append(
                dict(cell,
                     source_root=str(root),
                     source_repeat=local_repeat,
                     source_run_id=run_id,
                     repeat=run_id))
        sources.append(
            dict(source_id=source_id,
                 root=str(root),
                 **{
                     key: value
                     for key, value in source.items()
                     if key not in ("cells", "cross_repeat")
                 }))
    report = {
        "operation":
        "retained_multi_campaign_output_audit_no_inference",
        "sources":
        sources,
        "manifest": [source["manifest"] for source in sources],
        "integrity_passed":
        all(source["integrity_passed"] for source in sources),
        "semantics":
        sources[0]["semantics"],
        "comparison_contract":
        sources[0]["comparison_contract"],
        "cross_source_runtime_contract":
        "not_validated; this audit verifies per-source output integrity and shared trace/stop semantics, "
        "not binary, engine, calibration, command or environment equality; validate those separately",
        "cells":
        cells,
        "cross_repeat":
        compare_repeats(cells),
    }
    for field in ("requested_cells", "completed_cells", "missing_cells",
                  "unresolved_failed_cells", "integrity_issue_count",
                  "first_eos_count"):
        report[field] = sum(source[field] for source in sources)
    return report


def markdown(report):
    """Render transport/count integrity separately from EOS anomalies and repeatability."""
    lines = [
        "# Phase campaign output audit", "",
        "Completed/requested: %d/%d; unresolved failed: %d; integrity issues: %d."
        % (report["completed_cells"], report["requested_cells"],
           report["unresolved_failed_cells"], report["integrity_issue_count"]),
        "", report["semantics"], "",
        report.get("cross_source_runtime_contract", ""), "",
        "| Model/variant/workload | Repeat | Requests | Integrity issues | First-EOS request IDs |",
        "|---|---:|---:|---:|---|"
    ]
    for cell in report["cells"]:
        lines.append("| %s | %s | %s | %d | %s |" %
                     ("/".join(cell[field]
                               for field in ("model", "variant", "workload")),
                      cell["repeat"], cell.get("observed_requests", "missing"),
                      len(cell["issues"]), cell["first_eos_requests"]))
    lines += [
        "", "## Per-request exact token comparison across repeats", "",
        "Equality is observational, not a proof of semantic correctness. Single runs are not tested.",
        "", report["comparison_contract"], "",
        "| Model/variant/workload | Repeats | Raw exact | Through first stop exact | Raw different IDs | Prefix different IDs |",
        "|---|---|---|---|---|---|"
    ]

    def agreement(row):
        count = row["comparable_requests"]
        if not count:
            return row["status"]
        return "%d/%d (%s)" % (count - len(row["different_request_ids"]),
                               count, row["status"])

    for row in report["cross_repeat"]:
        lines.append("| %s | %s | %s | %s | %s | %s |" %
                     ("/".join(row[field]
                               for field in ("model", "variant",
                                             "workload")), row["repeats"],
                      agreement(row), agreement(row["through_first_stop"]),
                      row["different_request_ids"],
                      row["through_first_stop"]["different_request_ids"]))
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=pathlib.Path, required=True)
    parser.add_argument(
        "--additional-campaign",
        type=pathlib.Path,
        action="append",
        default=[],
        help=
        "Combine another retained source; runtime-contract equality must be checked separately"
    )
    parser.add_argument("--output-prefix", type=pathlib.Path, required=True)
    args = parser.parse_args()
    report = (audit_campaigns([args.campaign] + args.additional_campaign)
              if args.additional_campaign else audit_campaign(args.campaign))
    report["command"] = [sys.executable] + sys.argv
    report["analyzer"] = identity(pathlib.Path(__file__))
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    args.output_prefix.with_suffix(".json").write_text(
        json.dumps(report, indent=2) + "\n")
    args.output_prefix.with_suffix(".md").write_text(markdown(report))
    return 0 if report["integrity_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
