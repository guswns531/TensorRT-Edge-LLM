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

"""Validate the source boundaries of the phase-serving runtime."""

from __future__ import annotations

import argparse
import fnmatch
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as source:
        return json.load(source)


def _git_paths(repo: Path, baseline_ref: str) -> set[str]:
    diff_command = ["git", "diff", "--name-only", f"{baseline_ref}...HEAD"]
    diff = subprocess.run(diff_command, cwd=repo, check=True, capture_output=True, text=True)
    untracked_command = ["git", "ls-files", "--others", "--exclude-standard"]
    untracked = subprocess.run(untracked_command, cwd=repo, check=True, capture_output=True, text=True)
    return {line for output in (diff.stdout, untracked.stdout) for line in output.splitlines() if line}


def _is_excluded(path: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(path, pattern) for pattern in patterns)


def _validate_runtime_text(repo: Path, contract: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    runtime = contract["production_runtime"]
    forbidden_text = runtime["forbidden_text"]
    forbidden_includes = runtime["forbidden_include_fragments"]
    source_suffixes = {".cpp", ".cu", ".cuh", ".h", ".hpp"}

    for root_name in runtime["roots"]:
        root = repo / root_name
        for path in sorted(candidate for candidate in root.rglob("*") if candidate.suffix in source_suffixes):
            text = path.read_text(encoding="utf-8")
            relative = path.relative_to(repo)
            for token in forbidden_text:
                if token in text:
                    errors.append(f"{relative}: production runtime contains forbidden text {token!r}")
            for line_number, line in enumerate(text.splitlines(), start=1):
                if not line.lstrip().startswith("#include"):
                    continue
                for fragment in forbidden_includes:
                    if fragment in line:
                        errors.append(f"{relative}:{line_number}: forbidden dependency {fragment!r}")
    return errors


def _validate_feature_ownership(repo: Path, contract: dict[str, Any]) -> list[str]:
    ownership_path = repo / contract["ownership_manifest"]
    ownership = _load_json(ownership_path)
    owned_paths = {
        path
        for feature in ownership["features"].values()
        for path in feature["paths"]
    }
    excluded = ownership["excluded_from_product_diff"]
    changed_paths = _git_paths(repo, ownership["baseline_ref"])
    product_paths = {path for path in changed_paths if not _is_excluded(path, excluded)}

    errors: list[str] = []
    for path in sorted(product_paths - owned_paths):
        errors.append(f"{path}: changed product path has no feature owner")
    for path in sorted(owned_paths - product_paths):
        errors.append(f"{path}: feature owner entry is stale or missing from the product diff")
    return errors


def _validate_composition_boundary(repo: Path, contract: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    boundary = contract["composition_boundary"]
    prefix = boundary["environment_prefix"]
    for path_name in boundary["paths"]:
        path = repo / path_name
        if not path.is_file():
            errors.append(f"{path_name}: composition-boundary source does not exist")
    if not any(prefix in (repo / path_name).read_text(encoding="utf-8")
               for path_name in boundary["paths"] if (repo / path_name).is_file()):
        errors.append(f"composition boundary does not own the {prefix!r} compatibility controls")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path(__file__).resolve().parent / "manifests" / "phase_runtime_contract.json",
    )
    args = parser.parse_args()

    repo = args.repo.resolve()
    contract = _load_json(args.contract.resolve())
    errors = []
    errors.extend(_validate_runtime_text(repo, contract))
    errors.extend(_validate_feature_ownership(repo, contract))
    errors.extend(_validate_composition_boundary(repo, contract))

    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1

    print("Phase runtime boundary validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
