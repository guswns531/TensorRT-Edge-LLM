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
"""Validate the frozen M0 phase-serving baseline and telemetry contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as source:
        value = json.load(source)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: top-level JSON value must be an object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _validate_manifest_shape(manifest: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required = {
        "schema_version",
        "baseline_id",
        "repository",
        "platform",
        "model_contract",
        "request_contract",
        "policy_contract",
        "frozen_results",
        "telemetry_contract",
        "artifacts",
    }
    for key in sorted(required - manifest.keys()):
        errors.append(f"manifest is missing required key {key!r}")
    if manifest.get("schema_version") != 1:
        errors.append("manifest schema_version must be 1")

    artifacts = manifest.get("artifacts", [])
    if not isinstance(artifacts, list):
        errors.append("manifest artifacts must be a list")
        return errors

    artifact_ids = [
        artifact.get("id") for artifact in artifacts
        if isinstance(artifact, dict)
    ]
    duplicates = {
        artifact_id
        for artifact_id in artifact_ids if artifact_ids.count(artifact_id) > 1
    }
    for artifact_id in sorted(duplicates):
        errors.append(f"duplicate artifact id {artifact_id!r}")

    known_ids = set(artifact_ids)
    policy = manifest.get("policy_contract", {})
    for mode in ("production_default", "research_opt_in"):
        artifact_id = policy.get(mode, {}).get("commands_artifact")
        if artifact_id not in known_ids:
            errors.append(
                f"{mode} references unknown commands artifact {artifact_id!r}")
    schema_id = manifest.get("telemetry_contract", {}).get("schema_artifact")
    if schema_id not in known_ids:
        errors.append(
            f"telemetry contract references unknown schema artifact {schema_id!r}"
        )
    return errors


def _validate_artifacts(
        manifest: dict[str, Any], repo: Path, workspace: Path,
        skip_large_bytes: int | None) -> tuple[list[str], int, int]:
    errors: list[str] = []
    checked = 0
    checked_bytes = 0
    bases = {"repository": repo, "workspace": workspace}
    for artifact in manifest["artifacts"]:
        artifact_id = artifact["id"]
        base_name = artifact.get("base")
        if base_name not in bases:
            errors.append(
                f"{artifact_id}: unknown artifact base {base_name!r}")
            continue
        path = bases[base_name] / artifact["path"]
        if not path.is_file():
            errors.append(f"{artifact_id}: missing artifact {path}")
            continue

        size = path.stat().st_size
        expected_size = artifact.get("bytes")
        if expected_size is not None and size != expected_size:
            errors.append(
                f"{artifact_id}: size {size} does not match frozen size {expected_size}"
            )
            continue
        if skip_large_bytes is not None and size > skip_large_bytes:
            print(
                f"SKIP {artifact_id}: {size} bytes exceeds --skip-large-bytes")
            continue

        actual_hash = _sha256(path)
        expected_hash = artifact["sha256"]
        if actual_hash != expected_hash:
            errors.append(
                f"{artifact_id}: sha256 {actual_hash} does not match {expected_hash}"
            )
            continue
        checked += 1
        checked_bytes += size
    return errors, checked, checked_bytes


def _validate_repository(manifest: dict[str, Any], repo: Path) -> list[str]:
    expected_head = manifest["repository"]["head_commit"]
    result = subprocess.run(["git", "rev-parse", "HEAD"],
                            cwd=repo,
                            check=True,
                            capture_output=True,
                            text=True)
    actual_head = result.stdout.strip()
    if actual_head != expected_head:
        return [
            f"repository HEAD {actual_head} does not match frozen HEAD {expected_head}"
        ]
    return []


def _validate_telemetry_schema(manifest: dict[str, Any], repo: Path,
                               workspace: Path) -> list[str]:
    artifact_id = manifest["telemetry_contract"]["schema_artifact"]
    artifact = next(item for item in manifest["artifacts"]
                    if item["id"] == artifact_id)
    base = repo if artifact["base"] == "repository" else workspace
    schema = _load_json(base / artifact["path"])

    errors: list[str] = []
    if schema.get("properties", {}).get("schema_version",
                                        {}).get("const") != 1:
        errors.append("telemetry schema must freeze schema_version=1")
    kinds = set(
        schema.get("properties", {}).get("event_kind", {}).get("enum", []))
    required_kinds = set(
        manifest["telemetry_contract"]["required_event_kinds"])
    if kinds != required_kinds:
        errors.append(
            f"telemetry event kinds {sorted(kinds)} do not match {sorted(required_kinds)}"
        )
    required_common = set(schema.get("required", []))
    for key in ("schema_version", "event_kind", "event_id", "run_id",
                "host_monotonic_ns"):
        if key not in required_common:
            errors.append(f"telemetry schema is missing common field {key!r}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo",
                        type=Path,
                        default=Path(__file__).resolve().parents[2])
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).resolve().parent / "manifests" /
        "inflight_m0_baseline.json",
    )
    parser.add_argument(
        "--skip-large-bytes",
        type=int,
        help=
        "Skip hashing artifacts larger than this byte count; omitted means validate every artifact.",
    )
    args = parser.parse_args()

    repo = args.repo.resolve()
    workspace = args.workspace_root.resolve()
    manifest = _load_json(args.manifest.resolve())

    errors = _validate_manifest_shape(manifest)
    if not errors:
        artifact_errors, checked, checked_bytes = _validate_artifacts(
            manifest, repo, workspace, args.skip_large_bytes)
        errors.extend(artifact_errors)
        errors.extend(_validate_repository(manifest, repo))
        errors.extend(_validate_telemetry_schema(manifest, repo, workspace))
    else:
        checked = 0
        checked_bytes = 0

    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1

    print(f"M0 baseline validation passed: {checked} artifacts, "
          f"{checked_bytes / (1024 * 1024 * 1024):.2f} GiB hashed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
