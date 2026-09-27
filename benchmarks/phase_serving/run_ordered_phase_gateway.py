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
"""Opt-in HTTP gateway that submits one trace to phase IPC in request order."""

import argparse
import importlib.util
import pathlib
import threading
import time


def ordered_broker_type(base_broker_type):

    class OrderedBroker(base_broker_type):

        def __init__(self, command, timeout):
            super().__init__(command)
            self.order_timeout = timeout
            self.order_condition = threading.Condition()
            self.next_request_index = 0

        def submit(self, request_index, request):
            deadline = time.monotonic() + self.order_timeout
            with self.order_condition:
                while request_index != self.next_request_index:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise ValueError(
                            f"ordered ingress waited for request {self.next_request_index}"
                        )
                    self.order_condition.wait(remaining)
                result = super().submit(request_index, request)
                self.next_request_index += 1
                self.order_condition.notify_all()
                return result

        def control(self, action, timeout):
            result = super().control(action, timeout)
            if action in ("begin", "status", "end"):
                with self.order_condition:
                    self.next_request_index = 0
                    self.order_condition.notify_all()
            return result

    return OrderedBroker


def load_gateway():
    path = pathlib.Path(__file__).parents[
        2] / ".local/results/v0101-forward-port/replay-tools/run_phase_openai_gateway.py"
    if not path.is_file():
        raise FileNotFoundError("Retained phase HTTP gateway is required: " +
                                str(path))
    spec = importlib.util.spec_from_file_location("phase_gateway_base", path)
    gateway = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gateway)
    return gateway


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--model", default="nvidia/Cosmos-Reason2-2B")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--max-stream-tokens-per-chunk", type=int, default=64)
    parser.add_argument("backend_command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.backend_command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("backend command is required after --")
    if args.timeout <= 0 or args.max_stream_tokens_per_chunk <= 0:
        parser.error("timeout and stream chunk size must be positive")
    gateway = load_gateway()
    broker = ordered_broker_type(gateway.EventBroker)(command, args.timeout)
    server = gateway.PhaseHTTPServer(
        (args.host, args.port),
        gateway.make_handler(broker, args.model, args.timeout,
                             args.max_stream_tokens_per_chunk))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        broker.close()


if __name__ == "__main__":
    main()
