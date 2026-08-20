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
"""Materialize a tied runtime variant around a byte-identical TensorRT engine."""

import argparse
import hashlib
import json
import os
import pathlib
import shutil
from typing import Any


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def link_or_copy(source: pathlib.Path, destination: pathlib.Path) -> None:
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def read_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-onnx-dir",
                        type=pathlib.Path,
                        required=True)
    parser.add_argument("--tied-onnx-dir", type=pathlib.Path, required=True)
    parser.add_argument("--baseline-engine-dir",
                        type=pathlib.Path,
                        required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    args = parser.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error(f"output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    onnx_hashes = {}
    for filename in ("model.onnx", "model.onnx.data"):
        baseline_path = args.baseline_onnx_dir / filename
        tied_path = args.tied_onnx_dir / filename
        baseline_hash = sha256(baseline_path)
        tied_hash = sha256(tied_path)
        if baseline_hash != tied_hash:
            raise RuntimeError(f"baseline/tied ONNX differs for {filename}: "
                               f"{baseline_hash} != {tied_hash}")
        onnx_hashes[filename] = baseline_hash

    baseline_config = read_json(args.baseline_engine_dir / "config.json")
    tied_config = read_json(args.tied_onnx_dir / "config.json")
    tied_weights = tied_config.get("external_weight_files") or []
    if len(tied_weights) != 1 or tied_weights[0].get("source") != "embedding":
        raise RuntimeError(
            "tied ONNX does not contain one embedding-source weight alias")
    baseline_config["external_weight_files"] = tied_weights

    engine_source = args.baseline_engine_dir / "llm.engine"
    link_or_copy(engine_source, args.output_dir / "llm.engine")
    link_or_copy(args.tied_onnx_dir / "embedding.safetensors",
                 args.output_dir / "embedding.safetensors")
    for filename in ("tokenizer.json", "tokenizer_config.json",
                     "processed_chat_template.json"):
        source = args.baseline_engine_dir / filename
        if source.is_file():
            link_or_copy(source, args.output_dir / filename)

    (args.output_dir / "config.json").write_text(
        json.dumps(baseline_config, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "baseline_onnx_dir":
        str(args.baseline_onnx_dir.resolve()),
        "tied_onnx_dir":
        str(args.tied_onnx_dir.resolve()),
        "baseline_engine_dir":
        str(args.baseline_engine_dir.resolve()),
        "onnx_sha256":
        onnx_hashes,
        "engine_sha256":
        sha256(engine_source),
        "engine_inode_shared":
        ((args.output_dir /
          "llm.engine").stat().st_ino == engine_source.stat().st_ino),
    }
    (args.output_dir / "tied_variant_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
