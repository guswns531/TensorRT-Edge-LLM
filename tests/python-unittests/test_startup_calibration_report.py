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
"""Contract checks for startup A/B reports."""

import json
import pathlib
import sys
import tempfile
import unittest

sys.path.insert(
    0,
    str(
        pathlib.Path(__file__).resolve().parents[2] /
        "benchmarks/phase_serving"))
import report_startup_calibration as report


class StartupReportTest(unittest.TestCase):

    def fixture(self, root, complete=True, mismatch=False):
        commands = []
        for variant in ("independent-predictor-on",
                        "independent-startup-predictor-on"):
            cell = root / variant
            (cell / "run-001").mkdir(parents=True)
            commands.append({
                "model": "model",
                "workload": "short",
                "repeat": 1,
                "variant": variant,
                "cell": str(cell)
            })
            data = {metric: 10.0 for metric in report.serving_report.METRICS}
            data.update(trace_sha256="trace",
                        requests_per_run=1,
                        requested_output_tokens_per_run=4,
                        gpu_memory_peak_mib_median=1024)
            if variant.startswith("independent-startup"):
                data[report.serving_report.METRICS[0]] = 11.0
                if mismatch:
                    data["trace_sha256"] = "other"
            (cell / "aggregate.json").write_text(json.dumps(data))
            (cell / "run-001/startup.json").write_text(
                json.dumps({
                    "elapsed_ms": 15000,
                    "frontier_covered": True,
                    "decode_probe_requests": 100
                }))
        (root / "manifest.json").write_text(
            json.dumps({
                "commands":
                commands,
                "completed":
                commands if complete else commands[:1],
                "completion_status":
                "complete" if complete else "partial"
            }))

    def test_complete_pair_preserves_metric_sign(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            self.fixture(root)
            result = report.summarize(root)
            self.assertTrue(result["complete"])
            self.assertAlmostEqual(
                result["rows"][0]["delta_percent"][
                    report.serving_report.METRICS[0]], 10)

    def test_incomplete_pair_cannot_report_success(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            self.fixture(root, complete=False)
            result = report.summarize(root)
            self.assertFalse(result["complete"])
            self.assertEqual(len(result["missing_pairs"]), 1)
            self.assertEqual(result["rows"], [])

    def test_different_request_contract_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            self.fixture(root, mismatch=True)
            with self.assertRaisesRegex(ValueError, "contract differs"):
                report.summarize(root)


if __name__ == "__main__":
    unittest.main()
