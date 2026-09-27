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
"""CPU-only contracts for request-token and decode-cohort divergence."""

import array
import importlib.util
import json
import pathlib
import tempfile
import unittest
import unittest.mock

G_PATH = pathlib.Path(__file__).parents[
    2] / "benchmarks/phase_serving/analyze_output_cohort_divergence.py"
G_SPEC = importlib.util.spec_from_file_location("output_cohort_divergence",
                                                G_PATH)
G_TOOL = importlib.util.module_from_spec(G_SPEC)
G_SPEC.loader.exec_module(G_TOOL)


class OutputCohortDivergenceTest(unittest.TestCase):

    def test_first_changed_token_uses_preceding_decode_iteration(self):
        left = ([7, 8, 506], [[1], [2]], [[1, 2], [2, 1]])
        right = ([7, 8, 1156], [[1], [2]], [[1, 2], [1, 2]])
        with unittest.mock.patch.object(G_TOOL,
                                        "load_cell",
                                        side_effect=[left, right]):
            with unittest.mock.patch.object(G_TOOL,
                                            "logit_evidence",
                                            return_value=None):
                result = G_TOOL.analyze(pathlib.Path("left"),
                                        pathlib.Path("right"), 1, 1)
        self.assertEqual(result["first_mismatch_token"], 2)
        self.assertEqual(result["first_prefill_membership_mismatch"], 2)
        self.assertEqual(result["first_decode_cohort_mismatch_for_request"], 1)
        self.assertEqual(result["left_decode_cohort_at_mismatch"], [2, 1])
        self.assertEqual(result["right_decode_cohort_at_mismatch"], [1, 2])

    def test_logit_metadata_requires_csv_token_to_match_argmax(self):
        with tempfile.TemporaryDirectory() as directory:
            cell = pathlib.Path(directory)
            folder = cell / "logits"
            folder.mkdir()
            base = folder / "request-1-step-2"
            base.with_suffix(".json").write_text(
                json.dumps({
                    "step": 2,
                    "request_id": 1,
                    "vocab": 4,
                    "selected_token": 2,
                    "batch_members": [1, 3],
                    "sample_row": 0
                }))
            base.with_suffix(".fp32").write_bytes(
                array.array("f", [0.0, 1.0, 4.0, 3.5]).tobytes())
            evidence = G_TOOL.logit_evidence(cell, 1, 2, 2, [2, 3])
            self.assertEqual(evidence["selected_token"], 2)
            self.assertAlmostEqual(evidence["top_two_margin"], 0.5)
            with self.assertRaisesRegex(ValueError,
                                        "metadata and request CSV"):
                G_TOOL.logit_evidence(cell, 1, 2, 3, [3, 2])


if __name__ == "__main__":
    unittest.main()
