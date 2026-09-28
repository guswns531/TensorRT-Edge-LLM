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
"""Activity-observer overhead campaign aggregation contracts."""

import importlib.util
import json
import pathlib
import tempfile
import unittest

G_PATH = pathlib.Path(__file__).parents[
    2] / "benchmarks/phase_serving/run_activity_observer_overhead.py"
G_SPEC = importlib.util.spec_from_file_location("observer_overhead", G_PATH)
G_TOOL = importlib.util.module_from_spec(G_SPEC)
G_SPEC.loader.exec_module(G_TOOL)


class ActivityObserverOverheadTest(unittest.TestCase):

    def test_block_order_rotates_every_mode_through_every_position(self):
        orders = [G_TOOL.block_order(block) for block in range(3)]
        for position in range(3):
            self.assertEqual({order[position]
                              for order in orders}, set(G_TOOL.MODES))

    def test_summarizes_median_change_against_no_observer(self):
        throughput = {
            "off": [100.0, 102.0, 98.0],
            "encoder": [99.0] * 3,
            "full": [97.0] * 3
        }
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            for block in range(3):
                for mode, values in throughput.items():
                    run = {metric: 10.0 for metric in G_TOOL.METRICS}
                    run["generated_token_s_median"] = values[block]
                    path = root / f"block-{block + 1}" / mode
                    path.mkdir(parents=True)
                    (path / "summary.json").write_text(
                        json.dumps({
                            "gemma/mixed/independent-predictor-on": {
                                "runs": [run]
                            }
                        }))
            result = G_TOOL.summarize(G_TOOL.collect(root, 3))
        row = result["rows"][0]
        self.assertEqual(row["blocks"], {"off": 3, "encoder": 3, "full": 3})
        self.assertAlmostEqual(
            row["encoder_vs_off_pct"]["generated_token_s_median"], -1.0)
        self.assertAlmostEqual(
            result["overall"]["full_throughput_geomean_vs_off_pct"], -3.0)


if __name__ == "__main__":
    unittest.main()
