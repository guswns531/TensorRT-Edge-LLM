#!/usr/bin/env python3
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
"""Expose the phase IPC benchmark through an OpenAI-compatible SSE endpoint."""

import argparse
import json
import queue
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


class PhaseHTTPServer(ThreadingHTTPServer):
    """Thread-per-request server sized for bursty online inference traces."""

    request_queue_size = 1024
    daemon_threads = True


class EventBroker:

    def __init__(self, command: list[str]) -> None:
        self.process = subprocess.Popen(command,
                                        stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT,
                                        text=True,
                                        bufsize=1)
        self.ready = threading.Event()
        self.write_lock = threading.Lock()
        self.queues: dict[int, queue.Queue[dict[str, Any]]] = {}
        self.queues_lock = threading.Lock()
        threading.Thread(target=self._read_events, daemon=True).start()

    def _read_events(self) -> None:
        assert self.process.stdout is not None
        for line in self.process.stdout:
            if not line.startswith("PHASE_EVENT\t"):
                print(line, end="", flush=True)
                continue
            event = json.loads(line.split("\t", 1)[1])
            if event["type"] == "ready":
                self.ready.set()
                continue
            request_index = int(event["request_index"])
            with self.queues_lock:
                destination = self.queues.get(request_index)
            if destination is not None:
                destination.put(event)

    def submit(self, request_index: int) -> queue.Queue[dict[str, Any]]:
        destination: queue.Queue[dict[str, Any]] = queue.Queue()
        with self.queues_lock:
            if request_index in self.queues:
                raise ValueError(f"request_index {request_index} was reused")
            self.queues[request_index] = destination
        assert self.process.stdin is not None
        with self.write_lock:
            self.process.stdin.write(
                json.dumps({"request_index": request_index}) + "\n")
            self.process.stdin.flush()
        return destination

    def release(self, request_index: int) -> None:
        with self.queues_lock:
            self.queues.pop(request_index, None)


def make_handler(broker: EventBroker, model: str,
                 timeout: float) -> type[BaseHTTPRequestHandler]:

    class Handler(BaseHTTPRequestHandler):

        protocol_version = "HTTP/1.1"

        def log_message(self, format_string: str, *args: object) -> None:
            del format_string, args

        def send_text(self, status: int, text: str) -> None:
            body = text.encode()
            self.send_response(status)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path == "/health":
                return_code = broker.process.poll()
                if broker.ready.is_set():
                    self.send_text(200, "ok\n")
                elif return_code is not None:
                    self.send_text(
                        500, f"backend exited with code {return_code}\n")
                else:
                    self.send_text(503, "loading\n")
            elif self.path == "/version":
                body = json.dumps({"backend": "TensorRT-Edge-LLM phase IPC"})
                self.send_text(200, body)
            elif self.path == "/metrics":
                self.send_text(200, "phase_gateway_ready 1\n")
            else:
                self.send_text(404, "not found\n")

        def _send_sse(self, event: dict[str, Any]) -> None:
            payload = ("data: " + json.dumps(event, separators=(",", ":")) +
                       "\n\n").encode()
            self.wfile.write(payload)
            self.wfile.flush()

        def do_POST(self) -> None:
            if self.path != "/v1/chat/completions":
                self.send_text(404, "not found\n")
                return
            content_length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(content_length))
            metadata = payload.get("metadata") or {}
            if "request_index" not in metadata or not payload.get("stream"):
                self.send_text(
                    400, "streaming metadata.request_index is required\n")
                return
            request_index = int(metadata["request_index"])
            try:
                events = broker.submit(request_index)
            except ValueError as exception:
                self.send_text(409, str(exception) + "\n")
                return

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            completion_tokens = 0
            try:
                while True:
                    event = events.get(timeout=timeout)
                    if event["type"] == "token":
                        completion_tokens += 1
                        self._send_sse({
                            "model":
                            model,
                            "choices": [{
                                "index": 0,
                                "delta": {
                                    "content": event["text"],
                                    "token_ids": [event["token_id"]],
                                },
                                "finish_reason": None,
                            }],
                        })
                        continue
                    finish_reason = ("stop" if event["finish_reason"]
                                     == "end-of-sequence" else "length")
                    self._send_sse({
                        "model":
                        model,
                        "choices": [{
                            "index": 0,
                            "delta": {},
                            "finish_reason": finish_reason,
                        }],
                        "usage": {
                            "prompt_tokens":
                            event["prompt_tokens"],
                            "completion_tokens":
                            event["output_tokens"],
                            "total_tokens":
                            event["prompt_tokens"] + event["output_tokens"],
                        },
                    })
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                    break
            finally:
                broker.release(request_index)

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--model", default="nvidia/Cosmos-Reason2-2B")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("backend_command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.backend_command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("backend command is required after --")
    broker = EventBroker(command)
    server = PhaseHTTPServer((args.host, args.port),
                             make_handler(broker, args.model, args.timeout))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        if broker.process.poll() is None:
            broker.process.terminate()
        broker.process.wait(timeout=30)


if __name__ == "__main__":
    main()
