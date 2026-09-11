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
"""Materialize a capability-scaled version of a retained 12-workload gate."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any


CASE_LIMITS = {
    "short": 48,
    "balanced": 64,
    "decode-heavy": 64,
    "long-prefill": 64,
    "bimodal": 64,
    "text-heavy": 64,
    "mixed": 64,
    "vision-heavy": 64,
    "poisson": 64,
    "wave-drain": 20,
    "multi-image": 20,
    "late-vision": 32,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _containerize_image_urls(value: Any) -> None:
    if isinstance(value, dict):
        if value.get("type") == "image_url":
            image_url = value.get("image_url")
            if isinstance(image_url, dict):
                url = image_url.get("url")
                marker = "/examples/multimodal/"
                if isinstance(url, str) and marker in url:
                    suffix = url.split(marker, 1)[1]
                    image_url["url"] = f"file:///workspace/examples/multimodal/{suffix}"
        for nested in value.values():
            _containerize_image_urls(nested)
    elif isinstance(value, list):
        for nested in value:
            _containerize_image_urls(nested)


def _repeat_requests(requests: list[dict[str, Any]], count: int,
                     interval_us: int) -> list[dict[str, Any]]:
    source_span = max(int(request.get("arrival_offset_us", 0))
                      for request in requests)
    wave_stride = max(source_span + interval_us, interval_us)
    result: list[dict[str, Any]] = []
    for index in range(count):
        source_index = index % len(requests)
        wave = index // len(requests)
        request = copy.deepcopy(requests[source_index])
        request["arrival_offset_us"] = (int(
            request.get("arrival_offset_us", 0)) + wave * wave_stride)
        request["source_request_index"] = source_index
        request["wave_index"] = wave
        semantic_id = request.get("semantic_id")
        if semantic_id is not None:
            request["semantic_id"] = f"{semantic_id}-port-{index:03d}"
        result.append(request)
    return result


def materialize_trace(source: dict[str, Any], case: str, request_limit: int,
                      repeat_interval_us: int) -> dict[str, Any]:
    """Preserve a workload role while bounding one model-port campaign."""
    requests = source.get("requests")
    if not isinstance(requests, list) or not requests:
        raise ValueError(f"{case}: source trace has no requests")
    if request_limit <= 0 or repeat_interval_us < 0:
        raise ValueError("request limit must be positive and interval non-negative")
    result = copy.deepcopy(source)
    result["workload"] = f"{case}-model-port"
    result["source_workload"] = source.get("workload", case)
    result["port_gate"] = {
        "case": case,
        "request_limit": request_limit,
        "repeat_interval_us": repeat_interval_us,
    }
    result["requests"] = _repeat_requests(requests, request_limit,
                                          repeat_interval_us)
    _containerize_image_urls(result)
    return result


def _find_trace(command: list[str]) -> Path:
    try:
        return Path(command[command.index("--trace") + 1])
    except (ValueError, IndexError) as error:
        raise ValueError("retained command has no --trace argument") from error


def build_gate(commands: list[dict[str, Any]], output_dir: Path,
               repeat_interval_us: int) -> dict[str, Any]:
    """Materialize all twelve role-preserving traces and their provenance."""
    cases = {str(entry["case"]): entry for entry in commands}
    if set(cases) != set(CASE_LIMITS):
        missing = sorted(set(CASE_LIMITS) - set(cases))
        extra = sorted(set(cases) - set(CASE_LIMITS))
        raise ValueError(f"full12 mismatch: missing={missing}, extra={extra}")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_cases: list[dict[str, Any]] = []
    for case, request_limit in CASE_LIMITS.items():
        source_path = _find_trace(cases[case]["command"])
        source = json.loads(source_path.read_text(encoding="utf-8"))
        trace = materialize_trace(source, case, request_limit,
                                  repeat_interval_us)
        output_path = output_dir / f"{case}.json"
        output_path.write_text(json.dumps(trace, indent=2) + "\n",
                               encoding="utf-8")
        arrivals = [
            int(request.get("arrival_offset_us", 0))
            for request in trace["requests"]
        ]
        manifest_cases.append({
            "case": case,
            "role_request_count": request_limit,
            "source": str(source_path),
            "source_sha256": _sha256(source_path),
            "trace": str(output_path),
            "trace_sha256": _sha256(output_path),
            "arrival_span_us": max(arrivals) - min(arrivals),
        })
    manifest = {
        "schema_version": 1,
        "kind": "model_port_full12_workload_gate",
        "cases": manifest_cases,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commands", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repeat-interval-us", type=int, default=250_000)
    args = parser.parse_args()
    if args.repeat_interval_us < 0:
        parser.error("repeat interval must be non-negative")
    try:
        commands = json.loads(args.commands.read_text(encoding="utf-8"))
        manifest = build_gate(commands, args.output_dir,
                              args.repeat_interval_us)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        parser.error(str(error))
    print(json.dumps({
        "output_dir": str(args.output_dir),
        "cases": len(manifest["cases"]),
        "requests": sum(case["role_request_count"]
                        for case in manifest["cases"]),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
