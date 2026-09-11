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
"""Validate every warmup response while retaining a trusted historical HTTP client."""

import hashlib
import importlib.util
import json
import os
import pathlib
import sys


def validate_rows(rows, expected):
    """Do not count failed or missing responses as successful calibration evidence."""
    failures = [
        row for row in rows
        if int(row.get("http_status", 0)) != 200 or row.get("error")
    ]
    if len(rows) != expected or failures:
        first = failures[0] if failures else "missing responses"
        raise RuntimeError(
            f"Request batch invalid: {len(rows)}/{expected} rows, "
            f"{len(failures)} failures; first: {first}")


def calibration_signature(result):
    """Return the decision-relevant coverage state of one calibration round."""
    contextual = []
    for family, status in sorted(
            result.get("contextual_policy_calibration", {}).items()):
        for direction in status.get("directions", []):
            if direction.get("required", False):
                contextual.append((family, direction.get("direction"),
                                   direction.get("ready", False)))
    if result.get("contextual_policy_calibration_converged", False):
        return "contextual", tuple(sorted(contextual))
    exact = sorted(
        (cost.get("action"), cost.get("primary_batch_size"),
         cost.get("secondary_batch_size"), cost.get("chunk_length"),
         cost.get("primary_context_bucket"),
         cost.get("secondary_context_bucket"), cost.get("execution_variant"),
         cost.get("residual_anchor"), cost.get("status"))
        for cost in result.get("calibration_cost_keys", [])
        if cost.get("required", False))
    return "exact", tuple(exact)


def main():
    source = pathlib.Path(os.environ["PHASE_TRACE_CLIENT_IMPL"])
    spec = importlib.util.spec_from_file_location(
        "retained_phase_trace_client", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    execute = module.execute_requests
    control = module.set_phase_calibration
    calibrating = False
    validation = []
    stable_signature = None
    stable_rounds = 0
    required_stable_rounds = int(
        os.environ.get("PHASE_CALIBRATION_STABLE_ROUNDS", "1"))
    if required_stable_rounds <= 0:
        raise ValueError("PHASE_CALIBRATION_STABLE_ROUNDS must be positive")
    output = pathlib.Path(sys.argv[sys.argv.index("--output-dir") + 1])

    def guarded_control(endpoint, action, timeout):
        nonlocal calibrating, stable_signature, stable_rounds
        if action == "begin":
            stable_signature = None
            stable_rounds = 0
        result = control(endpoint, action, timeout)
        if action == "status" and required_stable_rounds > 1:
            raw_converged = bool(result.get("calibration_converged", False))
            signature = calibration_signature(result)
            if raw_converged:
                stable_rounds = stable_rounds + 1 if signature == stable_signature else 1
            else:
                stable_rounds = 0
            stable_signature = signature
            result["raw_calibration_converged"] = raw_converged
            result["calibration_stable_rounds"] = stable_rounds
            result["calibration_required_stable_rounds"] = required_stable_rounds
            result["calibration_converged"] = (
                raw_converged
                and stable_rounds >= required_stable_rounds)
        if action in ("begin", "end"):
            calibrating = action == "begin"
        return result

    def guarded_execute(*args, **kwargs):
        rows, duration = execute(*args, **kwargs)
        requests = args[2] if len(args) > 2 else kwargs["requests"]
        if calibrating:
            validation.append({
                "expected":
                len(requests),
                "responses":
                len(rows),
                "failed":
                sum(
                    int(row.get("http_status", 0)) != 200
                    or bool(row.get("error")) for row in rows),
                "request_classes":
                sorted({module.request_class(request)
                        for request in requests})
            })
            (output / "warmup-validation.json").write_text(
                json.dumps(
                    {
                        "client_sha256":
                        hashlib.sha256(source.read_bytes()).hexdigest(),
                        "batches":
                        validation
                    },
                    indent=2) + "\n")
        validate_rows(rows, len(requests))
        return rows, duration

    module.execute_requests = guarded_execute
    module.set_phase_calibration = guarded_control
    return module.main()


if __name__ == "__main__":
    raise SystemExit(main())
