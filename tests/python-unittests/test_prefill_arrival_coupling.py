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
"""Paired HTTP ingress and prefill formation analysis contracts."""

import importlib.util
import pathlib
import unittest

G_PATH = pathlib.Path(__file__).parents[
    2] / "benchmarks/phase_serving/analyze_prefill_arrival_coupling.py"
G_SPEC = importlib.util.spec_from_file_location("prefill_arrival", G_PATH)
G_TOOL = importlib.util.module_from_spec(G_SPEC)
G_SPEC.loader.exec_module(G_TOOL)


class PrefillArrivalCouplingTest(unittest.TestCase):

    def requests(self, sends):
        return {
            index: {
                "scheduled_arrival_us": str(index * 1000),
                "send_us": str(send),
                "first_token_us": str(send + 2000),
                "completed_us": str(send + 5000),
                "prompt_tokens": "10",
                "max_output_tokens": "4"
            }
            for index, send in enumerate(sends)
        }

    def metric(self, index, request_ids):
        return {
            "dispatch_index": index,
            "host_dispatch_start_us": 1000.0,
            "host_completion_us": 1010.0,
            "prefill_request_ids": request_ids,
            "prefill_tokens": 10 * len(request_ids)
        }

    def events(self, request_ids, ready):
        return [{
            "event_kind": "dispatch",
            "phase": "prefill",
            "request_ids": request_ids,
            "enqueue_host_ns": 1000000,
            "decision_id": 1
        }, {
            "event_kind": "decision",
            "decision_id": 1,
            "action_kind": "prefill",
            "ready_prefill_request_ids": ready
        }]

    def test_pairs_actual_sends_with_ready_prefill_state(self):
        result = G_TOOL.analyze_pair(self.requests([0, 1000, 2000]),
                                     [self.metric(1, [0, 1])],
                                     self.events([0, 1], [0, 1]),
                                     self.requests([0, 2000, 4000]),
                                     [self.metric(2, [0])],
                                     self.events([0], [0]))
        self.assertEqual(result["actual_send_abs_delta_p95_ms"], 1.9)
        self.assertEqual(result["left"]["prefill_row_appearances"], 2)
        self.assertEqual(result["right"]["prefill_singletons"], 1)
        mismatch = result["first_prefill_mismatch"]
        self.assertEqual(mismatch["left_ready_prefill_ids"], [0, 1])
        self.assertEqual(mismatch["right_ready_prefill_ids"], [0])

    def test_rejects_different_scheduled_work(self):
        left = self.requests([0, 1000])
        right = self.requests([0, 1000])
        right[1]["prompt_tokens"] = "11"
        with self.assertRaises(ValueError):
            G_TOOL.analyze_pair(left, [], [], right, [], [])

    def test_server_submit_order_is_distinct_from_client_send_order(self):
        left = {
            0:
            dict(server_submit=1000,
                 server_admit=2000,
                 prefill_start=3000,
                 first_token=4000),
            1:
            dict(server_submit=1100,
                 server_admit=2100,
                 prefill_start=3100,
                 first_token=4100)
        }
        right = {
            0:
            dict(server_submit=1200,
                 server_admit=2200,
                 prefill_start=3200,
                 first_token=4200),
            1:
            dict(server_submit=1150,
                 server_admit=2150,
                 prefill_start=3150,
                 first_token=4150)
        }
        report = G_TOOL.analyze_server_transitions(left, right)
        for stage in ("server_submit", "server_admit", "prefill_start"):
            order = report["stage_order"][stage]
            self.assertEqual(order["first_mismatch"], 0)
            self.assertEqual(order["left_at_mismatch"], [0, 1])
            self.assertEqual(order["right_at_mismatch"], [1, 0])
            self.assertEqual(order["inversions"], 1)
        del right[0]["server_admit"]
        with self.assertRaises(ValueError):
            G_TOOL.analyze_server_transitions(left, right)


if __name__ == "__main__":
    unittest.main()
