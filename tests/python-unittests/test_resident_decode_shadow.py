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
"""CPU-only parsing contracts for resident decode shadow diagnostics."""

import copy
import gzip
import importlib.util
import json
import pathlib
import tempfile
import unittest

G_PATH = pathlib.Path(__file__).parents[
    2] / "benchmarks/phase_serving/analyze_resident_decode_shadow.py"
G_SPEC = importlib.util.spec_from_file_location(
    "analyze_resident_decode_shadow", G_PATH)
G_ANALYZER = importlib.util.module_from_spec(G_SPEC)
G_SPEC.loader.exec_module(G_ANALYZER)


def request(request_id, stage="sampling", age=2.0, known=True, covered=0):
    """Construct one emitted request-observation without inventing GPU clocks."""
    return {
        "request_id": request_id,
        "stage": stage,
        "candidate_ready": stage == "queued",
        "reference_valid": age is not None,
        "reference_source":
        "runtime_covering" if age is not None else "unknown",
        "reference_us": 2000.0 if age is not None else 0.0,
        "service_age_quanta": age,
        "service_epoch": 3,
        "coverage_known": known,
        "covered_candidates": covered
    }


def record(*requests):
    """Use the same authority contract as the runtime emitter."""
    return {
        "schema_version": 1,
        "mode": "shadow",
        "authority_applied": False,
        "observation_point": "post_selection",
        "resident_count": len(requests),
        "candidate_count": 2,
        "requests": list(requests)
    }


def line(value):
    """Include the real logger's arbitrary timestamp/location prefix."""
    return "[00:01:02.003] [INFO] [phaseThreeCoordinator.cpp:1] " + G_ANALYZER.G_MARKER + json.dumps(
        value) + "\n"


class ResidentDecodeShadowTest(unittest.TestCase):

    def analyze(self, text, compressed=False):
        with tempfile.TemporaryDirectory() as folder:
            path = pathlib.Path(folder) / ("gateway.log.gz"
                                           if compressed else "gateway.log")
            if compressed:
                with gzip.open(path, "wt", encoding="utf-8") as stream:
                    stream.write(text)
            else:
                path.write_text(text, encoding="utf-8")
            return G_ANALYZER.analyze(path)

    def test_separates_observation_counts_unique_ids_and_unknown_coverage(
            self):
        text = line(
            record(request(1, "queued", 1.0, covered=1),
                   request(2, age=None, known=False)))
        text += line(record(request(1, age=3.0), request(2, covered=1)))
        result = self.analyze(text)
        self.assertEqual(result["scope"], "unscoped_log")
        self.assertEqual(result["record_count"], 2)
        self.assertEqual(result["unique_request_id_count"], 2)
        self.assertEqual(result["unique_sampling_request_id_count"], 2)
        self.assertEqual(result["counts"]["request_observations"], 4)
        self.assertEqual(result["request_observations_by_stage"]["sampling"],
                         3)
        for name in ("candidate_ready_observations",
                     "coverage_unknown_observations",
                     "sampling_uncovered_observations",
                     "sampling_coverage_unknown_observations",
                     "sampling_uncovered_measured_reference_observations"):
            self.assertEqual(result["counts"][name], 1, name)
        self.assertEqual(
            result["counts"]["sampling_measured_reference_observations"], 2)
        ages = result["service_age_quanta_per_request_observation"]["all"]
        self.assertEqual(ages["count"], 3)
        self.assertEqual(ages["p50"], 2.0)
        self.assertAlmostEqual(ages["p95"], 2.9)

    def test_measurement_marker_excludes_calibration_and_later_epochs(self):
        text = line(record(request(99, age=1000.0)))
        text += 'PHASE_EPOCH\t{"epoch":1,"kind":"measurement"}\n'
        text += line(record(request(1)))
        text += 'PHASE_EPOCH\t{"epoch":2,"kind":"measurement"}\n'
        text += line(record(request(2, age=1000.0)))
        result = self.analyze(text)
        self.assertEqual(result["scope"], "measurement_epoch")
        self.assertEqual(result["record_count"], 1)
        self.assertEqual(result["excluded_shadow_records"], 2)
        self.assertEqual(
            result["service_age_quanta_per_request_observation"]["all"]["p95"],
            2.0)

    def test_compressed_and_uncompressed_data_have_identical_summary(self):
        text = 'PHASE_EPOCH\t{"epoch":1,"kind":"measurement"}\n' + line(
            record(request(1)))
        plain, compressed = self.analyze(text), self.analyze(text,
                                                             compressed=True)
        for key in ("log", "log_sha256"):
            plain.pop(key)
            compressed.pop(key)
        self.assertEqual(plain, compressed)

    def test_rejects_authority_or_nonshadow_records_even_before_measurement(
            self):
        for key, value in (("authority_applied", True),
                           ("authority_applied", 0), ("mode", "active"),
                           ("observation_point", "pre_selection")):
            invalid = record(request(1))
            invalid[key] = value
            with self.subTest(key=key,
                              value=value), self.assertRaises(ValueError):
                self.analyze(
                    line(invalid) +
                    'PHASE_EPOCH\t{"epoch":1,"kind":"measurement"}\n')

    def test_rejects_duplicate_ids_and_unmeasured_or_nonfinite_age(self):
        with self.assertRaises(ValueError):
            self.analyze(line(record(request(1), request(1))))
        for age in (-1.0, float("inf"), float("nan")):
            with self.subTest(age=age), self.assertRaises(ValueError):
                self.analyze(line(record(request(1, age=age))))
        unmeasured = request(1)
        unmeasured["reference_valid"] = False
        with self.assertRaises(ValueError):
            self.analyze(line(record(unmeasured)))

    def test_missing_measurement_and_absent_diagnostics_are_not_a_pass(self):
        text = 'PHASE_EPOCH\t{"epoch":2,"kind":"measurement"}\n' + line(
            record(request(1)))
        result = self.analyze(text)
        self.assertEqual(result["scope"], "measurement_epoch_missing")
        self.assertEqual(result["record_count"], 0)
        self.assertIsNone(
            result["service_age_quanta_per_request_observation"]["all"]["p95"])
        self.assertEqual(self.analyze("ordinary log\n")["record_count"], 0)

    def test_empty_frontier_is_unknown_not_uncovered(self):
        value = record(request(1, known=False))
        value["candidate_count"] = 0
        result = self.analyze(line(value))
        self.assertEqual(
            result["counts"]["sampling_coverage_unknown_observations"], 1)
        self.assertEqual(result["counts"]["sampling_uncovered_observations"],
                         0)
        invalid = copy.deepcopy(value)
        invalid["requests"][0]["coverage_known"] = True
        with self.assertRaises(ValueError):
            self.analyze(line(invalid))

    def test_repeated_target_epoch_is_not_silently_pooled(self):
        marker = 'PHASE_EPOCH\t{"epoch":1,"kind":"measurement"}\n'
        with self.assertRaises(ValueError):
            self.analyze(marker + line(record(request(1))) + marker)


if __name__ == "__main__":
    unittest.main()
