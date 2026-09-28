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
"""Deterministic phase IPC submission gate contracts."""

import contextlib
import importlib.util
import io
import pathlib
import threading
import time
import unittest

G_PATH = pathlib.Path(__file__).parents[
    2] / "benchmarks/phase_serving/run_ordered_phase_gateway.py"
G_SPEC = importlib.util.spec_from_file_location("ordered_gateway", G_PATH)
G_TOOL = importlib.util.module_from_spec(G_SPEC)
G_SPEC.loader.exec_module(G_TOOL)


class FakeBroker:

    def __init__(self, command):
        self.command = command
        self.submitted = []

    def submit(self, request_index, request):
        self.submitted.append(request_index)
        return request

    def control(self, action, timeout):
        return {"type": "control", "action": action}


class OrderedPhaseGatewayTest(unittest.TestCase):

    def test_reorders_concurrent_handlers_and_resets_at_calibration_boundary(
            self):
        broker = G_TOOL.ordered_broker_type(FakeBroker)([], 1.0)
        errors = []

        def submit(index):
            try:
                broker.submit(index, {"index": index})
            except ValueError as error:
                errors.append(error)

        with contextlib.redirect_stdout(io.StringIO()):
            threads = [
                threading.Thread(target=submit, args=(index, ))
                for index in (2, 1)
            ]
            for thread in threads:
                thread.start()
            time.sleep(.01)
            submit(0)
            for thread in threads:
                thread.join(timeout=1.0)
            broker.control("end", 1.0)
            broker.submit(0, {})
        self.assertFalse(errors)
        self.assertEqual(broker.submitted, [0, 1, 2, 0])
        self.assertTrue(all(not thread.is_alive() for thread in threads))

    def test_missing_earlier_request_times_out(self):
        broker = G_TOOL.ordered_broker_type(FakeBroker)([], .001)
        with self.assertRaises(ValueError):
            broker.submit(1, {})

    def test_order_failure_fails_successors_until_reset(self):

        class RejectingBroker(FakeBroker):

            def submit(self, request_index, request):
                if request_index == 0:
                    raise RuntimeError("rejected")
                return super().submit(request_index, request)

        broker = G_TOOL.ordered_broker_type(RejectingBroker)([], 60.0)
        with self.assertRaises(RuntimeError):
            broker.submit(0, {})
        started = time.monotonic()
        with self.assertRaises(ValueError):
            broker.submit(1, {})
        self.assertLess(time.monotonic() - started, 1.0)
        broker.control("begin", 1.0)
        with self.assertRaises(RuntimeError):
            broker.submit(0, {})
