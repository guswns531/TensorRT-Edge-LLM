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
"""OpenAI chat-completions gateway for the phase IPC backend (llm_phase_context_smoke).

The backend reads one JSON request per stdin line and writes ``PHASE_EVENT\\t{json}`` lines
(``ready``, ``token``, ``completion``, ``error``, ``control``). Other ``PHASE_*`` records are
telemetry and are appended to ``--telemetry-log``.
"""

import argparse
import itertools
import json
import os
import queue
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class EventBroker:
    """Owns the backend process and demultiplexes its events per request index."""

    def __init__(self, command, telemetry_log=None):
        self.process = subprocess.Popen(command,
                                        stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE,
                                        text=True,
                                        bufsize=1)
        self.write_lock = threading.Lock()
        self.queues_lock = threading.Lock()
        self.queues = {}
        self.ids = itertools.count()
        self.ready = threading.Event()
        self.ready_event = None
        self.dead = None
        self.telemetry = open(telemetry_log, "a") if telemetry_log else None
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()

    def _read(self):
        for line in self.process.stdout:
            line = line.rstrip("\n")
            if not line.startswith("PHASE_EVENT\t"):
                if not line.startswith("PHASE_"):
                    print(line, file=sys.stderr, flush=True)
                elif self.telemetry is not None:
                    self.telemetry.write(line + "\n")
                continue
            event = json.loads(line.split("\t", 1)[1])
            if event.get("type") == "ready":
                self.ready_event = event
                self.ready.set()
                continue
            index = event.get("request_index")
            with self.queues_lock:
                target = self.queues.get(index)
            if target is not None:
                target.put(event)
        self.dead = "backend exited with code %s" % self.process.wait()
        print(self.dead, file=sys.stderr, flush=True)
        self.ready.set()
        with self.queues_lock:
            for target in self.queues.values():
                target.put({"type": "error", "message": self.dead})
        # A dead backend cannot recover; let in-flight handlers report the error, then exit.
        time.sleep(2)
        os._exit(1)

    def _send(self, payload):
        events = queue.Queue()
        index = next(self.ids)
        payload["request_index"] = index
        with self.queues_lock:
            self.queues[index] = events
        with self.write_lock:
            if self.dead:
                raise RuntimeError(self.dead)
            self.process.stdin.write(json.dumps(payload) + "\n")
            self.process.stdin.flush()
        return index, events

    def release(self, index):
        with self.queues_lock:
            self.queues.pop(index, None)

    def submit(self, request):
        return self._send({"request": request})

    def control(self, action, timeout):
        index, events = self._send({"type": "calibration_" + action})
        try:
            while True:
                event = events.get(timeout=timeout)
                if event.get("type") in ("control", "error"):
                    return event
        finally:
            self.release(index)

    def close(self):
        if self.process.poll() is None:
            self.process.stdin.close()
            try:
                self.process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self.process.kill()
        if self.telemetry is not None:
            self.telemetry.close()


def make_handler(broker, model, timeout):

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def handle(self):
            try:
                super().handle()
            except ConnectionError:
                pass

        def _json(self, status, body):
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/health":
                if broker.dead:
                    self._json(503, {"error": broker.dead})
                elif broker.ready.is_set():
                    self._json(200, {"status": "ok"})
                else:
                    self._json(503, {"status": "starting"})
            elif self.path == "/v1/models":
                self._json(200, {
                    "object": "list",
                    "data": [{
                        "id": model,
                        "object": "model"
                    }]
                })
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
            if self.path.startswith("/phase/control/"):
                action = self.path.rsplit("/", 1)[1]
                if action not in ("begin", "end", "status"):
                    self._json(400, {"error": "unknown control action"})
                    return
                self._json(200, broker.control(action, timeout))
                return
            if self.path != "/v1/chat/completions":
                self._json(404, {"error": "not found"})
                return
            request = {"messages": body.get("messages", [])}
            max_tokens = body.get("max_completion_tokens",
                                  body.get("max_tokens"))
            if max_tokens is not None:
                request["max_tokens"] = int(max_tokens)
            index, events = broker.submit(request)
            try:
                if body.get("stream"):
                    self._stream(index, events)
                else:
                    self._complete(index, events)
            finally:
                broker.release(index)

        def _next(self, events):
            return events.get(timeout=timeout)

        def _complete(self, index, events):
            text = []
            while True:
                event = self._next(events)
                kind = event.get("type")
                if kind == "token":
                    text.append(event.get("text", ""))
                elif kind == "completion":
                    self._json(
                        200, {
                            "id":
                            "chatcmpl-%d" % index,
                            "object":
                            "chat.completion",
                            "created":
                            int(time.time()),
                            "model":
                            model,
                            "choices": [{
                                "index": 0,
                                "message": {
                                    "role": "assistant",
                                    "content": "".join(text)
                                },
                                "finish_reason": _finish(event)
                            }],
                            "usage":
                            _usage(event)
                        })
                    return
                elif kind == "error":
                    self._json(
                        500, {"error": event.get("message", "backend error")})
                    return

        def _stream(self, index, events):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            base = {
                "id": "chatcmpl-%d" % index,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model
            }

            def send(payload):
                data = ("data: " + payload + "\n\n").encode()
                self.wfile.write(b"%x\r\n%s\r\n" % (len(data), data))
                self.wfile.flush()

            while True:
                event = self._next(events)
                kind = event.get("type")
                if kind == "token":
                    send(
                        json.dumps({
                            **base, "choices": [{
                                "index": 0,
                                "delta": {
                                    "content": event.get("text", "")
                                },
                                "finish_reason": None
                            }]
                        }))
                elif kind == "completion":
                    send(
                        json.dumps({
                            **base, "choices": [{
                                "index": 0,
                                "delta": {},
                                "finish_reason": _finish(event)
                            }],
                            "usage":
                            _usage(event)
                        }))
                    break
                elif kind == "error":
                    send(
                        json.dumps({
                            "error": {
                                "message": event.get("message",
                                                     "backend error")
                            }
                        }))
                    break
            send("[DONE]")
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()

    return Handler


def _finish(event):
    return "stop" if event.get(
        "finish_reason") == "end-of-sequence" else "length"


def _usage(event):
    prompt = int(event.get("prompt_tokens", 0))
    output = int(event.get("output_tokens", 0))
    return {
        "prompt_tokens": prompt,
        "completion_tokens": output,
        "total_tokens": prompt + output
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--model", required=True)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--telemetry-log")
    parser.add_argument("backend_command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.backend_command[1:] if args.backend_command[:1] == [
        "--"
    ] else args.backend_command
    if not command:
        parser.error("backend command is required after --")
    broker = EventBroker(command, args.telemetry_log)
    server = ThreadingHTTPServer((args.host, args.port),
                                 make_handler(broker, args.model,
                                              args.timeout))
    server.daemon_threads = True
    print("gateway listening on %s:%d" % (args.host, args.port),
          file=sys.stderr,
          flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        broker.close()


if __name__ == "__main__":
    main()
