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
"""Validation of paired diagnostic result contracts."""

import copy
import importlib.util
import pathlib
import unittest

G_PATH = pathlib.Path(__file__).parents[
    2] / "benchmarks/phase_serving/run_decode_equal_work_trials.py"
G_SPEC = importlib.util.spec_from_file_location("equal_work", G_PATH)
G_TOOL = importlib.util.module_from_spec(G_SPEC)
G_SPEC.loader.exec_module(G_TOOL)


class DecodeEqualWorkTest(unittest.TestCase):

    def report(self):
        return {
            "config": {
                "rounds": 1,
                "iterations": 1
            },
            "graph_misses":
            0,
            "policy_applied":
            False,
            "compared_tokens":
            16,
            "mismatched_tokens":
            0,
            "samples": [{
                "round": 0,
                "iteration": 0,
                "variant": variant,
                "order": order,
                "gpu_ms": value,
                "drain_ms": value,
                "successor_gpu_ms": 1,
                "two_tick_ms": value + 1
            } for order, (variant, value) in enumerate((("dense", 2), ("split",
                                                                       4)))],
        }

    def test_summary_uses_equal_work_and_reports_both_ticks(self):
        result = G_TOOL.summarize(self.report())
        self.assertEqual(result["split_change_pct"]["gpu_ms"]["mean"], 100)
        self.assertAlmostEqual(
            result["split_change_pct"]["two_tick_ms"]["mean"], 200 / 3)

    def test_incomplete_or_duplicate_pairs_fail(self):
        report = self.report()
        report["samples"].pop()
        with self.assertRaises(ValueError):
            G_TOOL.summarize(report)
        report = self.report()
        report["samples"].append(copy.deepcopy(report["samples"][0]))
        with self.assertRaises(ValueError):
            G_TOOL.summarize(report)

    def test_graph_fallback_and_bad_order_fail(self):
        report = self.report()
        report["graph_misses"] = 1
        with self.assertRaises(ValueError):
            G_TOOL.summarize(report)
        report = self.report()
        report["samples"][1]["order"] = 0
        with self.assertRaises(ValueError):
            G_TOOL.summarize(report)

    def test_nonfinite_timings_fail(self):
        for values in ([], [float("nan")], [float("inf")], [0], [-1]):
            with self.assertRaises(ValueError):
                G_TOOL.distribution(values)
