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
"""Reproduce Gemma's 129-token prompt with a singleton P128/P1 continuation."""

import argparse
import csv
import gzip
import hashlib
import json
import os
import pathlib
import signal
import subprocess

from check_phase_ipc_lifecycle import (MINIMAL_STARTUP_OVERRIDES,
                                       collect_launched_identity,
                                       replace_docker_environment)


def sha256(path):
    """Hash a file without loading a binary or engine into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def replace_option(command, option, value):
    """Replace exactly one known option in the frozen harness command."""
    if command.count(option) != 1:
        raise ValueError("Expected one command option: " + option)
    command[command.index(option) + 1] = str(value)


def diagnostic_command(record, build_root, trace, output):
    """Keep the frozen serving machinery, changing only diagnostic run limits."""
    original = list(record["command"])
    boundary = original.index("--")
    front = original[:boundary]
    backend = original[boundary + 1:]
    for option, value in {
            "--trace": trace,
            "--output-dir": output,
            "--repeats": 1,
            "--max-workers": 1,
            "--max-in-flight": 1,
            "--policy-warmup-mode": "zero_start",
            "--warmup-requests": 0,
            "--phase-calibration-min-requests": 0,
    }.items():
        replace_option(front, option, value)
    mounts = {"edgellm": 0, "results": 0}
    for index, value in enumerate(backend):
        if value.endswith(":/opt/edgellm:ro"):
            backend[index] = str(build_root) + ":/opt/edgellm:ro"
            mounts["edgellm"] += 1
        elif value.endswith(":/opt/results:rw"):
            backend[index] = str(output) + ":/opt/results:rw"
            mounts["results"] += 1
    if mounts != {"edgellm": 1, "results": 1}:
        raise ValueError("Expected unique build and result mounts")
    overrides = dict(MINIMAL_STARTUP_OVERRIDES)
    overrides.update({
        "TRT_EDGELLM_MAX_INFLIGHT": "1",
        "TRT_EDGELLM_FIXED_PREFILL_CHUNK": "128",
        "TRT_EDGELLM_PHASE_TELEMETRY_LEVEL": "full",
        "TRT_EDGELLM_IPC_WARMUP_DECODE_BATCHES": None,
    })
    image_index = backend.index(
        "/opt/edgellm/examples/llm/llm_phase_context_smoke") - 1
    backend = replace_docker_environment(backend, overrides, image_index)
    return front + ["--"] + backend


def inspect_rows(rows, events, expected_requests, eos_tokens):
    """Check fixture-specific behavior, not general semantic correctness."""
    failures = []
    generations = []
    dispatches = {}
    for event in events:
        if (event.get("event_kind") == "dispatch"
                and event.get("phase") == "prefill"):
            ids = event["request_ids"]
            for request_id in ids:
                dispatches.setdefault(str(request_id), []).append(
                    event["cohort"]["prefill_tokens"] if len(ids) == 1 else -1)
    if len(rows) != expected_requests:
        failures.append("Request count does not match the fixture")
    for row in rows:
        request_id = row["request_id"]
        tokens = [int(value) for value in row["output_token_ids"].split()]
        generations.append(tokens)
        if int(row["http_status"]) != 200 or row.get("error"):
            failures.append(f"Request {request_id} failed HTTP inference")
        if int(row["prompt_tokens"]) != 129:
            failures.append(f"Request {request_id} was not a 129-token prompt")
        if dispatches.get(request_id) != [128, 1]:
            failures.append(f"Request {request_id} did not dispatch P128/P1")
        if not tokens or tokens[0] in eos_tokens:
            failures.append(
                f"Request {request_id} starts with EOS or no token")
        if len(tokens) != int(row["max_output_tokens"]):
            failures.append(
                f"Request {request_id} token capture is incomplete")
    return {
        "passed":
        not failures,
        "failures":
        failures,
        "prefill_dispatch_tokens":
        dispatches,
        "first_token_ids":
        [tokens[0] if tokens else None for tokens in generations],
        "duplicate_prompt_token_identity":
        bool(generations)
        and all(tokens == generations[0] for tokens in generations),
        "requests":
        len(rows),
    }


def inspect_output(output, expected_requests, eos_tokens):
    """Read raw HTTP tokens and scheduler events from the isolated run."""
    with (output / "run-001/client/run-001/requests.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    path = output / "run-001/gateway.log"
    opener = path.open
    if not path.exists():
        path = path.with_suffix(".log.gz")
        opener = lambda: gzip.open(path, "rt")
    events = []
    with opener() as stream:
        for line in stream:
            if "Phase CUDA graph measurement start:" in line:
                events.clear()
            if "PHASE_SCHEDULER_EVENT\t" in line:
                events.append(
                    json.loads(line.split("PHASE_SCHEDULER_EVENT\t", 1)[1]))
    return inspect_rows(rows, events, expected_requests, eos_tokens)


def run_bounded(command, log, timeout):
    """Terminate only this diagnostic's process group if its deadline expires."""
    process = subprocess.Popen(command,
                               stdout=log,
                               stderr=subprocess.STDOUT,
                               start_new_session=True)
    try:
        return process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=pathlib.Path, required=True)
    parser.add_argument("--build-root", type=pathlib.Path, required=True)
    parser.add_argument("--trace", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("timeout must be positive")
    manifest = json.loads(args.manifest.read_text())
    records = manifest.get("commands", [manifest.get("record")])
    record = next(
        record for record in records
        if record and record["model"] == "gemma" and record["workload"] ==
        "balanced" and record["variant"] == "shared_ep-predictor-on")
    build_root, trace, output = (path.resolve()
                                 for path in (args.build_root, args.trace,
                                              args.output_dir))
    command = diagnostic_command(record, build_root, trace, output)
    launched = collect_launched_identity(
        command[command.index("--") + 1:],
        pathlib.Path(__file__).resolve().parents[2])
    hf = pathlib.Path(manifest["identity"]["models"]["gemma"]
                      ["effective_model_config"]["hf"])
    eos_tokens = set()
    for name in ("config.json", "generation_config.json"):
        values = json.loads((hf / name).read_text()).get("eos_token_id", [])
        eos_tokens.update(values if isinstance(values, list) else [values])
    result = {
        "state":
        "diagnostic",
        "purpose":
        "Singleton packed-prefill continuation correctness; not performance",
        "command":
        command,
        "source_contract":
        str(args.manifest.resolve()),
        "source_contract_sha256":
        sha256(args.manifest),
        "trace_sha256":
        sha256(trace),
        "launched_identity":
        launched,
        "binary_sha256":
        launched["binary"]["sha256"],
        "plugin_sha256":
        launched["plugin"]["sha256"],
        "eos_tokens":
        sorted(eos_tokens),
        "expected_prompt_tokens":
        129,
        "expected_prefill_chunks": [128, 1],
        "graphs_enabled":
        False,
        "policy_calibration_requests":
        0,
        "correctness_scope":
        "First-token EOS fixture oracle and metadata shape, not exact cross-engine identity",
    }
    if args.dry_run:
        print(json.dumps(result, indent=2))
        return
    output.mkdir(parents=True, exist_ok=False)
    (output / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    try:
        with (output / "driver.log").open("w") as log:
            result["returncode"] = run_bounded(command, log, args.timeout)
        if result["returncode"] != 0:
            raise RuntimeError("HTTP diagnostic command failed")
        result.update(
            inspect_output(output,
                           len(json.loads(trace.read_text())["requests"]),
                           eos_tokens))
    except Exception as error:
        result.update(passed=False, error=str(error))
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result.get("passed") else 1)


if __name__ == "__main__":
    main()
