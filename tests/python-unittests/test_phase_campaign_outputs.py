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
"""CPU-only quality audits for retained fixed-output HTTP campaign artifacts."""

import csv
import importlib.util
import json
import pathlib
import tempfile
import unittest

G_PATH = pathlib.Path(__file__).parents[
    2] / "benchmarks/phase_serving/audit_phase_campaign_outputs.py"
G_SPEC = importlib.util.spec_from_file_location("phase_output_audit", G_PATH)
G_AUDIT = importlib.util.module_from_spec(G_SPEC)
G_SPEC.loader.exec_module(G_AUDIT)


class PhaseCampaignOutputsTest(unittest.TestCase):

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = pathlib.Path(self.directory.name)
        engine = self.root / "engine"
        engine.mkdir()
        config = engine / "config.json"
        config.write_text(json.dumps({"eos_token_id": [151643, 151645]}))
        trace = self.root / "trace.json"
        trace.write_text(
            json.dumps({"requests": [{
                "max_generate_length": 2
            }] * 2}))
        self.model = {
            "engine": str(engine),
            "config_sha256": G_AUDIT.identity(config)["sha256"],
            "traces": {
                "mixed": G_AUDIT.identity(trace)["sha256"]
            }
        }
        self.trace = trace
        self.records = []
        self.csv_paths = []
        for repeat in (1, 2):
            cell = self.root / "cosmos" / "shared_ep-predictor-on" / (
                "repeat-%03d" % repeat) / "mixed"
            csv_path = cell / "run-001/client/run-001/requests.csv"
            csv_path.parent.mkdir(parents=True)
            self.write_rows(csv_path,
                            [self.row(0, "151645 2"),
                             self.row(1, "3 4")])
            (cell / "aggregate.json").write_text(
                json.dumps({
                    "repeats": 1,
                    "trace_sha256": self.model["traces"]["mixed"],
                    "requests_per_run": 2,
                    "requested_output_tokens_per_run": 4,
                    "generated_tokens_per_run_min": 4,
                    "captured_token_ids_per_run": [4]
                }))
            self.records.append({
                "model": "cosmos",
                "variant": "shared_ep-predictor-on",
                "workload": "mixed",
                "repeat": repeat,
                "cell": str(cell),
                "command": ["client", "--trace",
                            str(trace)]
            })
            self.csv_paths.append(csv_path)
        self.manifest = {
            "identity": {
                "models": {
                    "cosmos": self.model
                }
            },
            "commands": self.records,
            "completed": self.records,
            "failures": []
        }
        self.write_manifest()

    def write_manifest(self):
        (self.root / "manifest.json").write_text(json.dumps(self.manifest))

    @staticmethod
    def row(index, tokens, count=2, status=200, error=""):
        return {
            "request_id": index,
            "max_output_tokens": 2,
            "output_tokens": count,
            "output_token_ids": tokens,
            "http_status": status,
            "error": error
        }

    @staticmethod
    def write_rows(path, rows):
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def test_eos_is_flagged_but_not_a_semantic_failure(self):
        report = G_AUDIT.audit_campaign(self.root)
        self.assertTrue(report["integrity_passed"])
        self.assertEqual(report["first_eos_count"], 2)
        self.assertEqual(report["cross_repeat"][0]["status"], "observed_equal")
        self.assertEqual(report["cross_repeat"][0]["comparable_requests"], 2)
        self.assertEqual(report["cross_repeat"][0]["exact_request_agreement"],
                         1)
        self.assertIn("not proof of corruption", G_AUDIT.markdown(report))

    def test_compensating_output_counts_cannot_hide_request_errors(self):
        self.write_rows(
            self.csv_paths[0],
            [self.row(0, "1", 1), self.row(1, "2 3 4", 3)])
        report = G_AUDIT.audit_campaign(self.root)
        self.assertFalse(report["integrity_passed"])
        issues = report["cells"][0]["issues"]
        self.assertEqual(
            sum(issue["category"] == "output_count" for issue in issues), 2)

    def test_transport_errors_are_retained(self):
        self.write_rows(self.csv_paths[0], [
            self.row(0, "1 2", status=500, error="server failed"),
            self.row(1, "3 4")
        ])
        report = G_AUDIT.audit_campaign(self.root)
        self.assertFalse(report["integrity_passed"])
        self.assertEqual(report["cells"][0]["issues"][0]["category"],
                         "transport")

    def test_empty_capture_hashes_are_not_repeatability_evidence(self):
        for path in self.csv_paths:
            self.write_rows(path, [self.row(0, ""), self.row(1, "")])
        report = G_AUDIT.audit_campaign(self.root)
        self.assertFalse(report["integrity_passed"])
        self.assertEqual(report["cross_repeat"][0]["status"],
                         "invalid_or_incomplete")
        self.assertIsNone(report["cross_repeat"][0]["exact_request_agreement"])

    def test_cross_repeat_mismatch_is_localized_to_request(self):
        self.write_rows(self.csv_paths[1],
                        [self.row(0, "151645 2"),
                         self.row(1, "3 5")])
        report = G_AUDIT.audit_campaign(self.root)
        self.assertTrue(report["integrity_passed"])
        comparison = report["cross_repeat"][0]
        self.assertEqual(comparison["status"], "observed_different")
        self.assertEqual(comparison["different_request_ids"], [1])
        self.assertEqual(comparison["exact_request_agreement"], 0.5)

    def test_partial_campaign_does_not_audit_failed_cell_as_success(self):
        self.manifest["completed"] = self.records[:1]
        self.manifest["failures"] = self.records[1:]
        self.write_manifest()
        report = G_AUDIT.audit_campaign(self.root)
        self.assertFalse(report["integrity_passed"])
        self.assertEqual(report["unresolved_failed_cells"], 1)
        self.assertEqual(len(report["cells"]), 1)
        self.assertEqual(report["cross_repeat"][0]["status"], "not_tested")

    def test_duplicate_and_missing_requests_are_detected(self):
        self.write_rows(
            self.csv_paths[0],
            [self.row(0, "1 2"), self.row(0, "1 2")])
        report = G_AUDIT.audit_campaign(self.root)
        categories = {
            issue["category"]
            for issue in report["cells"][0]["issues"]
        }
        self.assertIn("request_identity", categories)
        self.assertIn("request_count", categories)

    def test_trace_and_model_identity_are_checked(self):
        self.trace.write_text("{}")
        report = G_AUDIT.audit_campaign(self.root)
        self.assertFalse(report["integrity_passed"])
        config = pathlib.Path(self.model["engine"]) / "config.json"
        config.write_text("{}")
        with self.assertRaisesRegex(ValueError, "Engine config changed"):
            G_AUDIT.audit_campaign(self.root)

    def test_gemma_eos_diagnostic_includes_turn_boundary(self):
        tokens, _ = G_AUDIT.model_eos("gemma", self.model)
        self.assertTrue({1, 50, 106}.issubset(tokens))


if __name__ == "__main__":
    unittest.main()
