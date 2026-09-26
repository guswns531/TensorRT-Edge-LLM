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
"""CPU-only cancellation protocol tests, without inference or GPU dependencies."""

import importlib.util
import pathlib
import unittest


def load_tool():
    """Load the standalone lifecycle script without model dependencies."""
    path = (pathlib.Path(__file__).parents[2] /
            "benchmarks/phase_serving/check_phase_ipc_lifecycle.py")
    spec = importlib.util.spec_from_file_location("phase_ipc_lifecycle", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


G_TOOL = load_tool()


class DockerEnvironmentTest(unittest.TestCase):
    """Minimal startup changes only the explicitly recorded environment keys."""

    def test_replaces_duplicates_adds_missing_and_preserves_tail(self):
        command = [
            "docker",
            "run",
            "--rm",
            "-e",
            "WARMUP=generic",
            "--env",
            "WARMUP=trace_derived",
            "--env=WARMUP=generic",
            "-e",
            "CAPACITY=24",
            "image:fixed",
            "/binary",
            "--env=WARMUP=argument",
        ]
        result = G_TOOL.replace_docker_environment(
            command, {
                "WARMUP": "zero_start",
                "GRAPHS": "0"
            }, command.index("image:fixed"))
        self.assertEqual(result, [
            "docker",
            "run",
            "--rm",
            "-e",
            "CAPACITY=24",
            "-e",
            "WARMUP=zero_start",
            "-e",
            "GRAPHS=0",
            "image:fixed",
            "/binary",
            "--env=WARMUP=argument",
        ])
        self.assertIn("WARMUP=generic", command)

    def test_unsets_explicit_shape_override_without_empty_assignment(self):
        command = [
            "docker", "run", "-e", "SHAPES=1:128,2:512", "image", "/bin"
        ]
        result = G_TOOL.replace_docker_environment(command, {"SHAPES": None},
                                                   command.index("image"))
        self.assertEqual(result, ["docker", "run", "image", "/bin"])

    def test_empty_overrides_preserve_normal_startup(self):
        command = ["docker", "run", "-e", "MODE=generic", "image", "/bin"]
        self.assertEqual(
            G_TOOL.replace_docker_environment(command, {},
                                              command.index("image")), command)

    def test_minimal_startup_preserves_resource_contract(self):
        protected = {
            "TRT_EDGELLM_MAX_STABLE_SLOTS": "24",
            "TRT_EDGELLM_MAX_INFLIGHT": "24",
            "TRT_EDGELLM_MAX_PREFILL_BATCH": "8",
            "TRT_EDGELLM_MAX_DECODE_BATCH": "24",
            "TRT_EDGELLM_VISION_ENCODER_BATCH_SIZE": "4",
            "TRT_EDGELLM_FIXED_PREFILL_CHUNK": "128",
            "TRT_EDGELLM_PHASE_WORKSPACE_MODE": "shared_ep",
        }
        command = ["docker", "run"]
        for name, value in protected.items():
            command.extend(["-e", name + "=" + value])
        command.extend([
            "-e", "TRT_EDGELLM_CAPTURE_PHASE_GRAPHS=1", "-e",
            "TRT_EDGELLM_IPC_WARMUP_DECODE_SHAPES=1:128", "image", "/bin"
        ])
        result = G_TOOL.replace_docker_environment(
            command, G_TOOL.MINIMAL_STARTUP_OVERRIDES, command.index("image"))
        for name, value in protected.items():
            self.assertIn(name + "=" + value, result)
        self.assertIn("TRT_EDGELLM_DISABLE_IPC_SHAPE_WARMUP=1", result)
        self.assertIn("TRT_EDGELLM_CAPTURE_PHASE_GRAPHS=0", result)
        self.assertIn("TRT_EDGELLM_POLICY_WARMUP_MODE=zero_start", result)
        self.assertNotIn("TRT_EDGELLM_IPC_WARMUP_DECODE_SHAPES=1:128", result)

    def test_rejects_malformed_docker_environment(self):
        with self.assertRaises(ValueError):
            G_TOOL.replace_docker_environment(
                ["docker", "run", "-e", "image", "/bin"], {"A": "1"}, 3)


class CancellationAttemptsTest(unittest.TestCase):
    """Only accepted cancellation permits readmission in the lifecycle probe."""

    def test_busy_rejection_retries_at_next_token(self):
        tracker = G_TOOL.CancellationAttempts()
        self.assertTrue(tracker.at_token(0))
        self.assertFalse(tracker.acknowledge(False))
        self.assertFalse(tracker.at_token(0))
        self.assertTrue(tracker.at_token(1))
        self.assertTrue(tracker.acknowledge(True))
        self.assertEqual(tracker.report()["attempts"], 2)
        self.assertEqual(tracker.report()["busy_rejections"], 1)
        self.assertFalse(tracker.report()["acknowledgement_pending"])

    def test_does_not_enqueue_duplicate_pending_attempt(self):
        tracker = G_TOOL.CancellationAttempts()
        self.assertTrue(tracker.at_token(0))
        self.assertFalse(tracker.at_token(1))
        self.assertFalse(tracker.at_token(2))
        self.assertEqual(tracker.report()["attempts"], 1)
        self.assertFalse(tracker.acknowledge(False))
        self.assertTrue(tracker.at_token(3))

    def test_accepted_cancellation_is_terminal(self):
        tracker = G_TOOL.CancellationAttempts()
        self.assertTrue(tracker.at_token(0))
        self.assertTrue(tracker.acknowledge(True))
        self.assertFalse(tracker.at_token(1))
        self.assertEqual(tracker.report()["attempts"], 1)

    def test_rejects_unsolicited_or_malformed_acknowledgement(self):
        tracker = G_TOOL.CancellationAttempts()
        with self.assertRaises(AssertionError):
            tracker.acknowledge(True)
        self.assertTrue(tracker.at_token(0))
        with self.assertRaises(AssertionError):
            tracker.acknowledge(None)
        self.assertTrue(tracker.report()["acknowledgement_pending"])


if __name__ == "__main__":
    unittest.main()
