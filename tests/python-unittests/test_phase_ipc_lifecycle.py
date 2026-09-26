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
