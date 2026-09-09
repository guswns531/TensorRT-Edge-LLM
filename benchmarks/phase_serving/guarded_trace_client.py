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
    output = pathlib.Path(sys.argv[sys.argv.index("--output-dir") + 1])

    def guarded_control(endpoint, action, timeout):
        nonlocal calibrating
        result = control(endpoint, action, timeout)
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
