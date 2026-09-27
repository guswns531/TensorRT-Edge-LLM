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

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Controlled async decode report validation."""

import importlib.util
import pathlib
import sys
import unittest

G_PATH = pathlib.Path(__file__).parents[
    2] / "benchmarks/phase_serving/run_async_decode_trials.py"
G_SPEC = importlib.util.spec_from_file_location("async_trial", G_PATH)
G_TOOL = importlib.util.module_from_spec(G_SPEC)
sys.path.insert(0, str(G_PATH.parent))
try:
    G_SPEC.loader.exec_module(G_TOOL)
finally:
    sys.path.pop(0)


class AsyncDecodeTrialTest(unittest.TestCase):

    def fixture(self):
        episodes = []
        for variant in range(3):
            dispatches = []
            for turn in range(2):
                batch = 1 if variant == 2 or (variant == 1
                                              and turn == 0) else 2
                dispatches.extend(
                    {
                        "turn": turn,
                        "offset": offset,
                        "batch": batch,
                        "graph": True,
                        "gpu_ms": 1.0,
                        "service_ms": 1.1,
                        "commit_to_prepare_ms": .1,
                        "actual_rows": list(range(offset, offset + batch))
                    } for offset in range(0, 2, batch))
            episodes.append({
                "round": 0,
                "order": variant,
                "variant": variant,
                "first_tick_ms": 1.2,
                "two_tick_ms": 2.4,
                "tokens": [[1, 2, 3], [1, 2, 3]],
                "dispatches": dispatches
            })
        return {
            "config": {
                "rounds": 1,
                "rows": 2,
                "split": 1,
                "split_turns": 1,
                "output_tokens": 3
            },
            "episodes": episodes
        }

    def test_equal_work_and_token_contract(self):
        result = G_TOOL.summarize(self.fixture())
        self.assertEqual(result["split"]["first_gpu_ms"], 2)
        self.assertEqual(result["dense"]["identical_to_singleton"], 2)

    def test_missing_episode_and_row_fail(self):
        raw = self.fixture()
        raw["episodes"].pop()
        with self.assertRaises(ValueError):
            G_TOOL.summarize(raw)
        raw = self.fixture()
        raw["episodes"][1]["dispatches"].pop()
        with self.assertRaises(ValueError):
            G_TOOL.summarize(raw)

    def test_eager_and_invalid_timing_fail(self):
        for field, value in (("graph", False), ("gpu_ms", float("nan")),
                             ("service_ms", -1)):
            raw = self.fixture()
            raw["episodes"][0]["dispatches"][0][field] = value
            with self.assertRaises(ValueError):
                G_TOOL.summarize(raw)

    def test_row_order_allowed_but_membership_must_match(self):
        raw = self.fixture()
        raw["episodes"][0]["dispatches"][0]["actual_rows"].reverse()
        G_TOOL.summarize(raw)
        raw["episodes"][0]["dispatches"][0]["actual_rows"] = [0, 0]
        with self.assertRaises(ValueError):
            G_TOOL.summarize(raw)

    def test_divergence_is_reported_not_quality_pass(self):
        raw = self.fixture()
        raw["episodes"][0]["tokens"][1][1] = 99
        result = G_TOOL.summarize(raw)
        self.assertEqual(result["dense"]["unique_sequences"], 2)
        self.assertEqual(result["dense"]["identical_to_singleton"], 1)
        self.assertEqual(result["dense"]["first_divergences"], [None, 1])

    def test_inspection_matches_committed_argmax_and_execution_mode(self):
        raw = self.fixture()
        raw["config"]["inspect_turn"] = 0
        raw["config"]["graph_replay"] = False
        for episode in raw["episodes"]:
            episode["logits"] = [{
                "turn":
                0,
                "actual_row":
                0,
                "top": [{
                    "token": token,
                    "logit": 10.0 - rank
                } for rank, token in enumerate(range(2, 10))]
            }]
            for dispatch in episode["dispatches"]:
                dispatch["graph"] = False
        self.assertEqual(
            G_TOOL.summarize(raw)["dense"]["inspected_top2_margin"], 1.0)
        raw["episodes"][0]["logits"][0]["top"][0]["token"] = 99
        with self.assertRaises(ValueError):
            G_TOOL.summarize(raw)

    def test_single_turn_branch_and_graph_coverage(self):
        raw = self.fixture()
        raw["config"].update({
            "rows": 3,
            "split": 2,
            "output_tokens": 4,
            "branch_turn": 1,
            "graph_batches": [1, 2]
        })
        for episode in raw["episodes"]:
            episode["tokens"] = [[1, 2, 3, 4] for _ in range(3)]
            dispatches = []
            for turn in range(3):
                batch = G_TOOL.decode_batch(raw["config"], episode["variant"],
                                            turn)
                for offset in range(0, 3, batch):
                    count = min(batch, 3 - offset)
                    prepare = (turn + 1) * 1000000 + offset * 100000
                    dispatches.append({
                        "turn":
                        turn,
                        "offset":
                        offset,
                        "batch":
                        count,
                        "graph":
                        count in (1, 2),
                        "gpu_ms":
                        1.0,
                        "service_ms":
                        1.1,
                        "commit_to_prepare_ms":
                        .1,
                        "actual_rows":
                        list(range(offset, offset + count)),
                        "prepare_ns":
                        prepare,
                        "drained_ns":
                        prepare + 100000
                    })
            episode["dispatches"] = dispatches
        result = G_TOOL.summarize(raw)
        self.assertEqual(result["dense"]["branch_tick_ms"], .1)
        self.assertEqual(result["split"]["branch_tick_ms"], .3)
        self.assertEqual(result["singleton"]["branch_gpu_ms"], 3.0)
        raw["episodes"][0]["dispatches"][1]["graph"] = True
        with self.assertRaises(ValueError):
            G_TOOL.summarize(raw)
