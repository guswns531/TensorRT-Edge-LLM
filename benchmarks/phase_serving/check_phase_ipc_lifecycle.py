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
"""Exercise decode cancellation and readmission with a campaign contract."""

import argparse
import copy
import hashlib
import json
import pathlib
import queue
import subprocess
import threading
import time

MINIMAL_STARTUP_OVERRIDES = {
    "TRT_EDGELLM_DISABLE_IPC_SHAPE_WARMUP": "1",
    "TRT_EDGELLM_DISABLE_GLOBAL_OVERLAP_WARMUP": "1",
    "TRT_EDGELLM_CAPTURE_PHASE_GRAPHS": "0",
    "TRT_EDGELLM_ONLINE_GRAPH_CAPTURE": "0",
    "TRT_EDGELLM_MAX_PREFILL_GRAPHS": "0",
    "TRT_EDGELLM_MAX_DECODE_GRAPHS": "0",
    "TRT_EDGELLM_POLICY_WARMUP_MODE": "zero_start",
    # Explicit shape lists take precedence over the disable-shape-warmup flag.
    "TRT_EDGELLM_IPC_WARMUP_DECODE_SHAPES": None,
}

DEBUG_BACKTRACE_FRAMES = 24


def sanitizer_prefix(force_synchronization_limit=None):
    """Build optional instrumentation-only launch bounds without runtime overrides."""
    command = [
        "compute-sanitizer", "--tool", "memcheck", "--error-exitcode", "99"
    ]
    if force_synchronization_limit is not None:
        if force_synchronization_limit <= 0:
            raise ValueError(
                "Sanitizer synchronization limit must be positive")
        command.extend([
            "--force-synchronization-limit",
            str(force_synchronization_limit)
        ])
    return command


def enable_debug_backtrace(command):
    """Follow only this launch's sanitizer child, with no core dump or host attach."""
    sanitizer_index = command.index("compute-sanitizer")
    image_index = sanitizer_index - 1
    docker_options = command[:image_index]
    network_none = "--network=none" in docker_options or any(
        docker_options[index:index + 2] == ["--network", "none"]
        for index in range(len(docker_options)))
    if not network_none:
        raise ValueError("Debug backtrace requires Docker --network none")
    command = replace_docker_environment(
        command, {
            "CUDA_ENABLE_COREDUMP_ON_EXCEPTION": "0",
            "CUDA_ENABLE_CPU_COREDUMP_ON_EXCEPTION": "0",
            "CUDA_ENABLE_USER_TRIGGERED_COREDUMP": "0",
        }, image_index)
    sanitizer_index = command.index("compute-sanitizer")
    image_index = sanitizer_index - 1
    debug_prefix = [
        "cuda-gdb-minimal", "--nx", "--quiet", "--batch",
        "--return-child-result"
    ]
    for instruction in ("set pagination off", "set confirm off",
                        "set startup-with-shell off",
                        "set follow-fork-mode child", "set detach-on-fork on",
                        "set follow-exec-mode same",
                        "set print thread-events off",
                        "set print frame-arguments none",
                        "set print elements 16", "set debuginfod enabled off",
                        "handle SIGPIPE nostop noprint pass", "run",
                        "info inferiors", "info proc exe",
                        "thread apply all -c bt " +
                        str(DEBUG_BACKTRACE_FRAMES)):
        debug_prefix.extend(["-ex", instruction])
    debug_prefix.append("--args")
    return (command[:image_index] + [
        "--cap-add=SYS_PTRACE", "--security-opt=seccomp=unconfined",
        "--ulimit=core=0:0", command[image_index]
    ] + debug_prefix + command[sanitizer_index:])


def make_request_fixture(source_requests, text_only):
    """Return independent request payloads without changing backend limits."""
    if text_only:
        prompts = (
            "Explain how a computer stores and retrieves information.",
            "Explain why seasons change throughout the year.",
        )
        return [{
            "messages": [{
                "role": "user",
                "content": prompt
            }]
        } for prompt in prompts]
    if not source_requests:
        raise ValueError("The source lifecycle trace has no requests")
    return copy.deepcopy(source_requests)


def replace_docker_environment(command, overrides, image_index):
    """Replace/add Docker environment entries; None removes a named entry."""
    if command[:2] != ["docker", "run"] or not 2 < image_index < len(command):
        raise ValueError(
            "Expected a docker run command with an image boundary")
    result = []
    index = 0
    while index < image_index:
        token = command[index]
        if token in ("-e", "--env"):
            if index + 1 >= image_index:
                raise ValueError("Docker environment argument has no value")
            assignment = command[index + 1]
            if assignment.split("=", 1)[0] not in overrides:
                result.extend(command[index:index + 2])
            index += 2
        elif token.startswith("--env="):
            if token[len("--env="):].split("=", 1)[0] not in overrides:
                result.append(token)
            index += 1
        else:
            result.append(token)
            index += 1
    for name, value in overrides.items():
        if value is not None:
            result.extend(["-e", name + "=" + value])
    return result + command[image_index:]


def collect_launched_identity(command, repository):
    """Hash the actual mounted executable/plugin independently of parent metadata."""
    mount = None
    plugin = "/opt/edgellm/libNvInfer_edgellm_plugin.so.1.0"
    for index, token in enumerate(command):
        if token in ("-v", "--volume") and index + 1 < len(command):
            parts = command[index + 1].split(":")
            if len(parts) >= 2 and parts[1] == "/opt/edgellm":
                mount = pathlib.Path(parts[0]).resolve(strict=True)
        assignment = ""
        if token in ("-e", "--env") and index + 1 < len(command):
            assignment = command[index + 1]
        elif token.startswith("--env="):
            assignment = token[len("--env="):]
        if assignment.startswith("EDGELLM_PLUGIN_PATH="):
            plugin = assignment.split("=", 1)[1]
    if mount is None:
        raise ValueError("Lifecycle command must bind mount /opt/edgellm")

    def mounted_file(container_path):
        relative = pathlib.PurePosixPath(container_path).relative_to(
            "/opt/edgellm")
        host_path = (mount / relative).resolve(strict=True)
        digest = hashlib.sha256()
        with host_path.open("rb") as data:
            for block in iter(lambda: data.read(1024 * 1024), b""):
                digest.update(block)
        return {
            "container_path": container_path,
            "host_path": str(host_path),
            "sha256": digest.hexdigest(),
            "size_bytes": host_path.stat().st_size,
        }

    source_commit = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        text=True).strip()
    source_status = subprocess.check_output(
        ["git", "-C", str(repository), "status", "--porcelain"], text=True)
    return {
        "observed_at": "immediately_before_backend_launch",
        "build_mount": str(mount),
        "binary":
        mounted_file("/opt/edgellm/examples/llm/llm_phase_context_smoke"),
        "plugin": mounted_file(plugin),
        "source_checkout": {
            "path": str(repository),
            "commit": source_commit,
            "dirty": bool(source_status),
            "status_porcelain": source_status.splitlines(),
            "proves_artifact_build_commit": False,
        },
    }


class CancellationAttempts:
    """Retry only after another token when the runtime rejects a busy cancellation."""

    def __init__(self):
        self.attempts = 0
        self.busy_rejections = 0
        self.pending = False
        self.accepted = False
        self.last_attempt_output_index = -1

    def at_token(self, output_index):
        """Return whether this new token boundary should initiate one attempt."""
        if (self.accepted or self.pending
                or output_index <= self.last_attempt_output_index):
            return False
        self.attempts += 1
        self.pending = True
        self.last_attempt_output_index = output_index
        return True

    def acknowledge(self, accepted):
        """Record acceptance of cancellation, not completion of reclamation."""
        if not self.pending or not isinstance(accepted, bool):
            raise AssertionError("Unexpected cancellation acknowledgement")
        self.pending = False
        self.accepted = accepted
        self.busy_rejections += int(not accepted)
        return accepted

    def report(self):
        """Return cancellation outcome without classifying busy as corruption."""
        return {
            "attempts": self.attempts,
            "busy_rejections": self.busy_rejections,
            "accepted": self.accepted,
            "acknowledgement_pending": self.pending,
            "last_attempt_output_index": self.last_attempt_output_index,
        }


def run(args):
    """Cancel active decode, then verify surviving and readmitted work."""
    manifest = json.loads(args.manifest.read_text())
    record = next(
        item for item in manifest["commands"]
        if item["model"] == args.model and item["workload"] == "multi-image")
    full_command = record["command"]
    command = full_command[full_command.index("--") + 1:]
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    for index, value in enumerate(command):
        if value.endswith(":/opt/results:rw"):
            command[index] = str(output) + ":/opt/results:rw"
    startup_overrides = (MINIMAL_STARTUP_OVERRIDES
                         if args.minimal_startup else {})
    if startup_overrides:
        image_index = command.index(
            "/opt/edgellm/examples/llm/llm_phase_context_smoke") - 1
        command = replace_docker_environment(command, startup_overrides,
                                             image_index)
    if args.sanitizer:
        index = command.index(
            "/opt/edgellm/examples/llm/llm_phase_context_smoke")
        command[index:index] = sanitizer_prefix(
            args.force_synchronization_limit)
        if args.debug_backtrace:
            command = enable_debug_backtrace(command)
    source_trace = pathlib.Path(full_command[full_command.index("--trace") +
                                             1])
    trace = json.loads(source_trace.read_text())
    requests = make_request_fixture(trace["requests"], args.text_only)
    events = queue.Queue()
    observed = []
    submitted = []
    launched_identity = collect_launched_identity(
        command,
        pathlib.Path(__file__).resolve().parents[2])
    process = subprocess.Popen(command,
                               stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT,
                               text=True,
                               bufsize=1)

    def collect():
        with (output / "backend.log").open("w") as log:
            for line in process.stdout:
                log.write(line)
                if line.startswith("PHASE_EVENT\t"):
                    events.put(json.loads(line.split("\t", 1)[1]))
            events.put({"type": "backend_closed"})

    thread = threading.Thread(target=collect, daemon=True)
    thread.start()
    deadline = time.monotonic() + args.timeout

    def receive():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Lifecycle deadline expired")
        event = events.get(timeout=remaining)
        observed.append(event)
        if event["type"] in ("error", "backend_closed"):
            raise RuntimeError("Backend failed: " + str(event))
        return event

    def send(payload):
        process.stdin.write(json.dumps(payload) + "\n")
        process.stdin.flush()

    def submit(request_id, source, tokens):
        request = copy.deepcopy(source)
        request["max_output_tokens"] = tokens
        payload = {"request_index": request_id, "request": request}
        submitted.append(payload)
        send(payload)

    passed = False
    error = None
    harness_initiated_terminate = False
    harness_initiated_kill = False
    cancellation = CancellationAttempts()
    completions = {}
    try:
        while receive()["type"] != "ready":
            pass
        submit(1, requests[0], 128)
        submit(2, requests[-1], 32)
        while not cancellation.accepted or set(completions) != {2, 3}:
            event = receive()
            request_id = event.get("request_index")
            if event["type"] == "token" and request_id == 1:
                if cancellation.at_token(event["output_index"]):
                    send({"type": "cancel", "request_index": 1})
            elif event["type"] == "cancelled" and request_id == 1:
                if cancellation.acknowledge(event.get("cancelled")):
                    submit(3, requests[-1], 16)
            elif event["type"] == "completion":
                if request_id == 1:
                    if cancellation.accepted:
                        raise AssertionError(
                            "Accepted cancellation completed normally")
                    raise AssertionError(
                        "No cancellation was accepted before "
                        "request completion; in-flight "
                        "rejections are legitimate busy outcomes")
                completions[request_id] = event["output_tokens"]
        if completions != {2: 32, 3: 16}:
            raise AssertionError(
                "Survivor/readmission token contract differs: " +
                str(completions))
        process.stdin.close()
        if process.wait(timeout=max(1.0, deadline - time.monotonic())) != 0:
            raise RuntimeError("Backend exit code: " + str(process.returncode))
        passed = True
    except (AssertionError, RuntimeError, TimeoutError, queue.Empty,
            subprocess.TimeoutExpired) as exc:
        error = str(exc)
    finally:
        if process.poll() is None:
            harness_initiated_terminate = True
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                harness_initiated_kill = True
                process.kill()
                process.wait()
        thread.join(timeout=10)
        report = {
            "passed": passed,
            "error": error,
            "backend_exit": {
                "returncode":
                process.returncode,
                "returncode_owner":
                ("docker_debugger_wrapper"
                 if args.debug_backtrace else "docker_backend_wrapper"),
                "harness_initiated_terminate":
                harness_initiated_terminate,
                "harness_initiated_kill":
                harness_initiated_kill,
            },
            "model": args.model,
            "sanitizer": args.sanitizer,
            "sanitizer_instrumentation": {
                "force_synchronization_limit":
                args.force_synchronization_limit,
                "changes_execution_concurrency":
                args.force_synchronization_limit is not None,
            },
            "debug_backtrace": {
                "enabled": args.debug_backtrace,
                "frame_limit_per_thread": DEBUG_BACKTRACE_FRAMES,
                "host_attach": False,
                "core_dump_requested": False,
                "target_identity_must_be_verified_from_backtrace":
                args.debug_backtrace,
                "independent_memory_safety_gate": False,
            },
            "result_state": "diagnostic",
            "request_fixture": {
                "kind": ("builtin_text_only"
                         if args.text_only else "source_multimodal_trace"),
                "source_trace":
                str(source_trace.resolve()),
                "source_trace_requests_used":
                not args.text_only,
                "submitted_requests":
                submitted,
                "vision_engine_loading":
                "source_campaign_unchanged",
                "performance_comparable":
                False,
            },
            "startup_contract": {
                "mode": ("minimal_startup"
                         if args.minimal_startup else "source_campaign"),
                "environment_overrides":
                startup_overrides,
                "null_override_means":
                "unset",
                "performance_comparable":
                False,
                "source_memory_and_batch_limits_preserved":
                True,
            },
            "command": command,
            "parent_manifest_reference": {
                "path": str(args.manifest.resolve()),
                "identity": manifest["identity"],
                "is_actual_launch_provenance": False,
            },
            "launched_identity": launched_identity,
            "events": observed,
            "cancellation": cancellation.report(),
            "survivor_readmission_output_tokens": completions,
            "coverage": {
                "cancel_target": "decode after first emitted token",
                "retry_boundary":
                "subsequent token after busy acknowledgement",
                "gpu_memory_safety":
                "memcheck" if args.sanitizer else "not checked",
                "kv_page_reuse_verified": False,
                "encoder_requested_by_fixture": not args.text_only,
                "vision_payload_lifecycle_verified": False,
                "encoder_or_sampling_pending_cancel_verified": False,
                "binary_plugin_identity_observed_before_launch": True,
                "artifact_build_commit_verified": False,
            },
        }
        (output /
         "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "passed": passed,
        "error": error,
        "output": str(output)
    }))
    return 0 if passed else 1


def main():
    """Run one explicitly selected backend; never mutate the source campaign."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=pathlib.Path, required=True)
    parser.add_argument("--model", choices=("gemma", "cosmos"), required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--sanitizer", action="store_true")
    parser.add_argument(
        "--force-synchronization-limit",
        type=int,
        help="With --sanitizer only: bound pending instrumented launches; "
        "changes concurrency and is not a performance measurement")
    parser.add_argument(
        "--debug-backtrace",
        action="store_true",
        help=
        "With --sanitizer only: batch cuda-gdb-minimal follows the launched "
        "child; enable container SYS_PTRACE/seccomp override, no host attach/core"
    )
    parser.add_argument(
        "--text-only",
        action="store_true",
        help="Diagnostic only: replace image requests with simple text; "
        "retain backend engine loading, memory/batch limits, and output lengths"
    )
    parser.add_argument(
        "--minimal-startup",
        action="store_true",
        help="Diagnostic only: disable startup policy/shape warmup and graphs; "
        "retain the source memory and batch limits (not a performance run)")
    args = parser.parse_args()
    if args.force_synchronization_limit is not None:
        if not args.sanitizer or args.force_synchronization_limit <= 0:
            parser.error("--force-synchronization-limit requires --sanitizer "
                         "and a positive integer")
    if args.debug_backtrace and not args.sanitizer:
        parser.error("--debug-backtrace requires --sanitizer")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
