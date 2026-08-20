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

import json
import pathlib
import sys
import unittest.mock

import pytest

from scripts.cosmos_reason2 import materialize_tied_engine_variant


def write_safetensors(path: pathlib.Path, data_bytes: int) -> None:
    header = json.dumps({
        "weight": {
            "dtype": "F16",
            "shape": [data_bytes // 2],
            "data_offsets": [0, data_bytes],
        }
    }).encode()
    path.write_bytes(
        len(header).to_bytes(8, byteorder="little") + header +
        bytes(data_bytes))


def materialize(tmp_path: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    baseline_onnx = tmp_path / "onnx-baseline"
    tied_onnx = tmp_path / "onnx-tied"
    baseline_engine = tmp_path / "engine-baseline"
    output = tmp_path / "engine-tied"
    for directory in (baseline_onnx, tied_onnx, baseline_engine):
        directory.mkdir()
    for filename, content in (("model.onnx", b"onnx"), ("model.onnx.data",
                                                        b"weights")):
        (baseline_onnx / filename).write_bytes(content)
        (tied_onnx / filename).write_bytes(content)
    (baseline_engine / "llm.engine").write_bytes(b"engine-bytes")
    (baseline_engine / "config.json").write_text("{}", encoding="utf-8")
    tied_config = {
        "external_weight_files": [{
            "source": "embedding",
            "tensors": ["external_lm_head_weight"],
        }]
    }
    (tied_onnx / "config.json").write_text(json.dumps(tied_config),
                                           encoding="utf-8")
    write_safetensors(tied_onnx / "embedding.safetensors", 16)
    arguments = [
        "materialize_tied_engine_variant.py", "--baseline-onnx-dir",
        str(baseline_onnx), "--tied-onnx-dir",
        str(tied_onnx), "--baseline-engine-dir",
        str(baseline_engine), "--output-dir",
        str(output)
    ]
    with unittest.mock.patch.object(sys, "argv", arguments):
        materialize_tied_engine_variant.main()
    return output, baseline_engine


def test_materializes_engine_contract_and_reuses_engine_inode(tmp_path):
    output, baseline_engine = materialize(tmp_path)
    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    contract = config["tied_engine_contract"]

    assert contract["version"] == 1
    assert contract["engine_bytes"] == len(b"engine-bytes")
    assert contract["engine_sample_bytes"] == len(b"engine-bytes")
    assert contract[
        "engine_sample_crc32"] == materialize_tied_engine_variant.sample_crc32(
            output / "llm.engine", len(b"engine-bytes"))
    assert contract["embedding_bytes"] == 16
    assert (output /
            "llm.engine").stat().st_ino == (baseline_engine /
                                            "llm.engine").stat().st_ino
    assert not (output / "external_lm_head_weight.safetensors").exists()


def test_rejects_nonidentical_onnx(tmp_path):
    _, baseline_engine = materialize(tmp_path)
    baseline_onnx = tmp_path / "onnx-baseline"
    tied_onnx = tmp_path / "onnx-tied"
    tied_onnx.joinpath("model.onnx").write_bytes(b"different")
    output = tmp_path / "engine-mismatch"
    arguments = [
        "materialize_tied_engine_variant.py", "--baseline-onnx-dir",
        str(baseline_onnx), "--tied-onnx-dir",
        str(tied_onnx), "--baseline-engine-dir",
        str(baseline_engine), "--output-dir",
        str(output)
    ]

    with unittest.mock.patch.object(sys, "argv", arguments), pytest.raises(
            RuntimeError, match="baseline/tied ONNX differs"):
        materialize_tied_engine_variant.main()
