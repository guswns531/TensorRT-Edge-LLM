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
"""Summarize opt-in resident decode observations, not policy effects or GPU idle time."""

import argparse
import collections
import gzip
import hashlib
import json
import math
import pathlib

G_MARKER = "PHASE_RESIDENT_DECODE_SHADOW "
G_STAGES = ("unknown", "queued", "inflight", "sampling", "capacity_wait")
G_MEASURED_SOURCES = ("runtime_exact", "runtime_covering",
                      "runtime_interpolated")


def distribution(values):
    """Return interpolated percentiles across request-observations, not unique requests."""
    ordered = sorted(values)

    def percentile(fraction):
        if not ordered:
            return None
        position = fraction * (len(ordered) - 1)
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        return ordered[lower] + (position - lower) * (ordered[upper] -
                                                      ordered[lower])

    return {
        "count": len(ordered),
        "p50": percentile(0.5),
        "p95": percentile(0.95)
    }


def validate(record):
    """Reject authority-bearing or internally inconsistent records, including warmup."""
    if (record.get("schema_version") != 1 or record.get("mode") != "shadow"
            or record.get("authority_applied") is not False
            or record.get("observation_point") != "post_selection"):
        raise ValueError(
            "Expected schema-1 post-selection shadow with authority_applied=false"
        )
    requests = record["requests"]
    if record["resident_count"] != len(requests):
        raise ValueError("Resident count disagrees with request-observations")
    ids = [request["request_id"] for request in requests]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate request ID in one immutable snapshot")
    for request in requests:
        if request["stage"] not in G_STAGES:
            raise ValueError("Unknown resident stage")
        if type(request["coverage_known"]) is not bool:
            raise ValueError("coverage_known must be boolean")
        if type(request["candidate_ready"]) is not bool:
            raise ValueError("candidate_ready must be boolean")
        if type(request["reference_valid"]) is not bool:
            raise ValueError("reference_valid must be boolean")
        count = request["covered_candidates"]
        if type(count
                ) is not int or count < 0 or count > record["candidate_count"]:
            raise ValueError("Invalid candidate coverage count")
        if request["coverage_known"] and record["candidate_count"] == 0:
            raise ValueError(
                "Empty frontier cannot establish protection coverage")
        age = request["service_age_quanta"]
        if age is not None and (not isinstance(age, (int, float))
                                or isinstance(age, bool)
                                or not math.isfinite(age) or age < 0):
            raise ValueError(
                "Service age must be finite, non-negative, or null")


class Summary:
    """Streaming counts with per-stage age samples for exact offline percentiles."""

    def __init__(self):
        self.record_count = 0
        self.stages = collections.Counter()
        self.counts = collections.Counter()
        self.unique_ids = set()
        self.sampling_ids = set()
        self.ages = collections.defaultdict(list)

    def add(self, record):
        """Count repeated observations explicitly; do not infer causal protection losses."""
        self.record_count += 1
        for request in record["requests"]:
            stage = request["stage"]
            self.stages[stage] += 1
            self.unique_ids.add(request["request_id"])
            self.counts["request_observations"] += 1
            self.counts["candidate_ready_observations"] += request[
                "candidate_ready"]
            known = request["coverage_known"]
            self.counts["coverage_unknown_observations"] += not known
            reference = request["reference_us"]
            measured = (request["reference_valid"]
                        and request["reference_source"] in G_MEASURED_SOURCES
                        and isinstance(reference, (int, float))
                        and not isinstance(reference, bool)
                        and math.isfinite(reference) and reference > 0)
            self.counts["measured_reference_observations"] += measured
            age = request["service_age_quanta"]
            if age is not None:
                if not measured:
                    raise ValueError(
                        "Service age provided without a valid measured reference"
                    )
                self.ages["all"].append(age)
                self.ages[stage].append(age)
            if stage == "sampling":
                self.sampling_ids.add(request["request_id"])
                uncovered = known and request["covered_candidates"] == 0
                self.counts["sampling_uncovered_observations"] += uncovered
                self.counts[
                    "sampling_coverage_unknown_observations"] += not known
                self.counts[
                    "sampling_measured_reference_observations"] += measured
                self.counts[
                    "sampling_uncovered_measured_reference_observations"] += uncovered and measured
                if uncovered and age is not None:
                    self.ages["sampling_uncovered"].append(age)

    def result(self):
        """Keep missing reference/coverage evidence separate from an observed zero."""
        return {
            "record_count": self.record_count,
            "unique_request_id_count": len(self.unique_ids),
            "unique_sampling_request_id_count": len(self.sampling_ids),
            "request_observations_by_stage": {
                stage: self.stages[stage]
                for stage in G_STAGES
            },
            "counts": {
                name: self.counts[name]
                for name in (
                    "request_observations", "candidate_ready_observations",
                    "coverage_unknown_observations",
                    "measured_reference_observations",
                    "sampling_uncovered_observations",
                    "sampling_coverage_unknown_observations",
                    "sampling_measured_reference_observations",
                    "sampling_uncovered_measured_reference_observations")
            },
            "service_age_quanta_per_request_observation": {
                stage: distribution(self.ages[stage])
                for stage in ("all", *G_STAGES, "sampling_uncovered")
            }
        }


def analyze(log, epoch=1):
    """Use the requested measurement epoch when present; disclose unscoped logs otherwise."""
    log = pathlib.Path(log)
    scoped, unscoped = Summary(), Summary()
    markers = []
    selected = False
    selected_boundaries = 0
    total_records = 0
    opener = gzip.open if log.suffix == ".gz" else open
    with opener(log, "rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                if "PHASE_EPOCH\t" in line:
                    marker = json.loads(line.split("PHASE_EPOCH\t", 1)[1])
                    markers.append(marker)
                    selected = marker.get(
                        "kind") == "measurement" and marker["epoch"] == epoch
                    selected_boundaries += selected
                    if selected_boundaries > 1:
                        raise ValueError(
                            "Repeated target measurement epoch: analyze each run separately"
                        )
                if G_MARKER not in line:
                    continue
                record = json.loads(line.split(G_MARKER, 1)[1])
                validate(record)
                total_records += 1
                if not markers:
                    unscoped.add(record)
                elif selected:
                    scoped.add(record)
            except (ValueError, KeyError, TypeError) as error:
                raise ValueError(f"{log}:{line_number}: {error}") from error
    result = (scoped if markers else unscoped).result()
    digest = hashlib.sha256()
    with log.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    result.update({
        "log":
        str(log),
        "log_sha256":
        digest.hexdigest(),
        "status":
        "observed" if result["record_count"] else "not_observed",
        "scope": ("measurement_epoch" if selected_boundaries else
                  "measurement_epoch_missing" if markers else "unscoped_log"),
        "requested_measurement_epoch":
        epoch,
        "epoch_markers":
        markers,
        "excluded_shadow_records":
        total_records - result["record_count"],
        "interpretation":
        ("Diagnostic post-selection observations; counts are not unique requests or time-weighted occupancy. "
         "Missing candidate membership is not evidence of causal delay or a policy-authority change. "
         "Unscoped logs may include calibration; host clocks are not GPU completion timestamps."
         )
    })
    return result


def main():
    """Write one independent summary per raw log without pooling different runs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs", nargs="+", type=pathlib.Path, required=True)
    parser.add_argument("--epoch", type=int, default=1)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    result = {
        "schema_version":
        1,
        "script_sha256":
        hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
        "logs": [analyze(log, args.epoch) for log in args.logs]
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) +
                           "\n",
                           encoding="utf-8")


if __name__ == "__main__":
    main()
