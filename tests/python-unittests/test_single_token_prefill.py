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
"""CPU-only checks for the singleton-prefill diagnostic contract."""

import importlib.util
import pathlib
import sys
import unittest


def load_tool():
    path = (pathlib.Path(__file__).parents[2] /
            "benchmarks/phase_serving/check_single_token_prefill.py")
    spec = importlib.util.spec_from_file_location("single_token_prefill", path)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


G_TOOL = load_tool()


class SingleTokenPrefillTest(unittest.TestCase):
    """Fail closed for the exact dispatch and first-token symptom."""

    @staticmethod
    def row(tokens="8291 563"):
        return {
            "request_id": "0",
            "http_status": "200",
            "error": "",
            "prompt_tokens": "129",
            "output_token_ids": tokens,
            "max_output_tokens": "2",
        }

    @staticmethod
    def events():
        return [{
            "event_kind": "dispatch",
            "phase": "prefill",
            "request_ids": [0],
            "cohort": {
                "prefill_tokens": tokens
            },
        } for tokens in (128, 1)]

    def test_valid_fixture(self):
        result = G_TOOL.inspect_rows([self.row()], self.events(), 1, {106})
        self.assertTrue(result["passed"])
        self.assertTrue(result["duplicate_prompt_token_identity"])

    def test_first_eos_is_failure_despite_fixed_output_length(self):
        result = G_TOOL.inspect_rows([self.row("106 563")], self.events(), 1,
                                     {106})
        self.assertFalse(result["passed"])
        self.assertIn("starts with EOS", result["failures"][0])

    def test_missing_or_batched_tail_does_not_validate_singleton_fix(self):
        events = self.events()
        events[-1]["request_ids"] = [0, 1]
        result = G_TOOL.inspect_rows([self.row()], events, 1, {106})
        self.assertFalse(result["passed"])
        result = G_TOOL.inspect_rows([self.row()], [], 1, {106})
        self.assertFalse(result["passed"])

    def test_different_valid_generation_is_recorded_not_silently_required(
            self):
        other = self.row("8291 1408")
        other["request_id"] = "1"
        events = self.events()
        second = self.events()
        for event in second:
            event["request_ids"] = [1]
        result = G_TOOL.inspect_rows([self.row(), other], events + second, 2,
                                     {106})
        self.assertTrue(result["passed"])
        self.assertFalse(result["duplicate_prompt_token_identity"])

    def test_derives_isolated_command_without_mutating_parent(self):
        command = ["python3", "harness.py"]
        for option in ("--trace", "--output-dir", "--repeats", "--max-workers",
                       "--max-in-flight", "--policy-warmup-mode",
                       "--warmup-requests",
                       "--phase-calibration-min-requests"):
            command.extend([option, "old"])
        command.extend([
            "--",
            "docker",
            "run",
            "-v",
            "/old:/opt/edgellm:ro",
            "-v",
            "/oldout:/opt/results:rw",
            "-e",
            "TRT_EDGELLM_CAPTURE_PHASE_GRAPHS=1",
            "-e",
            "TRT_EDGELLM_IPC_WARMUP_DECODE_SHAPES=8:128",
            "image",
            "/opt/edgellm/examples/llm/llm_phase_context_smoke",
            "/opt/model",
        ])
        original = command.copy()
        result = G_TOOL.diagnostic_command({"command": command}, "/new",
                                           "/trace", "/out")
        self.assertEqual(command, original)
        self.assertIn("/new:/opt/edgellm:ro", result)
        self.assertIn("/out:/opt/results:rw", result)
        self.assertIn("TRT_EDGELLM_CAPTURE_PHASE_GRAPHS=0", result)
        self.assertNotIn("TRT_EDGELLM_CAPTURE_PHASE_GRAPHS=1", result)
        self.assertNotIn("TRT_EDGELLM_IPC_WARMUP_DECODE_SHAPES=8:128", result)
        self.assertEqual(result[result.index("--max-in-flight") + 1], "1")
        self.assertEqual(result[result.index("--policy-warmup-mode") + 1],
                         "zero_start")


if __name__ == "__main__":
    unittest.main()
