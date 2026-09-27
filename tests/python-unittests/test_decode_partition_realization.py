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
"""Measured-service partition trajectory checks."""

import importlib.util
import pathlib
import unittest

G_PATH = pathlib.Path(__file__).parents[
    2] / "benchmarks/phase_serving/analyze_decode_partition_realization.py"
G_SPEC = importlib.util.spec_from_file_location("partition_realization",
                                                G_PATH)
G_TOOL = importlib.util.module_from_spec(G_SPEC)
G_SPEC.loader.exec_module(G_TOOL)


class DecodePartitionRealizationTest(unittest.TestCase):

    def metric(self, index, rows, frontier=None):
        result = {
            "dispatch_index": index,
            "host_dispatch_start_us": float(index * 1000),
            "decode_request_ids": rows,
        }
        if frontier is not None:
            result.update({
                "predicted_decode_partition": [2, 2],
                "predicted_decode_frontier_ids": frontier,
                "predicted_decode_frontier_lengths": [128] * 4,
                "predicted_decode_drain_service_ms": 2.0,
            })
        return result

    def test_same_frontier_realizes_the_partition(self):
        metrics = [
            self.metric(1, [1, 2], [1, 2, 3, 4]),
            self.metric(2, [3, 4])
        ]
        commits = {request_id: [2500.0] for request_id in range(1, 5)}
        report = G_TOOL.analyze(metrics, commits)
        self.assertEqual(report["same_frontier_realized"], 1)
        self.assertEqual(report["complete_commit_horizons"], 1)
        self.assertEqual(report["matched_commit_horizon_mean_ms"], 1.5)

    def test_requeued_rows_can_overtake_the_residual(self):
        metrics = [
            self.metric(1, [1, 2], [1, 2, 3, 4]),
            self.metric(2, [1, 2]),
            self.metric(3, [3, 4])
        ]
        commits = {request_id: [3500.0] for request_id in range(1, 5)}
        report = G_TOOL.analyze(metrics, commits)
        self.assertEqual(report["first_batch_matches"], 1)
        self.assertEqual(report["same_frontier_realized"], 0)
        self.assertEqual(report["duplicate_before_drain"], 1)

    def test_invalid_frontier_fails(self):
        metrics = [self.metric(1, [1, 2], [1, 2, 3, 3])]
        with self.assertRaises(ValueError):
            G_TOOL.analyze(metrics, {})
        with self.assertRaises(ValueError):
            G_TOOL.analyze([self.metric(1, [1, 2])], {})


if __name__ == "__main__":
    unittest.main()
