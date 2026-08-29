# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the phase cost bundle build and fleet aggregation tool."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


def _module():
    script = Path(
        __file__).resolve().parents[2] / "scripts" / "phase_cost_bundle.py"
    spec = importlib.util.spec_from_file_location("phase_cost_bundle", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fingerprint() -> dict:
    return {
        "model_hash": "model",
        "onnx_hash": "onnx",
        "config_hash": "config",
        "engine_hash": "engine",
        "external_weight_hash": "weights",
        "precision": "fp16",
        "kv_dtype": "fp16",
        "capability": {
            "max_prefill_batch_size": 8,
            "max_decode_batch_size": 64,
            "max_encoder_batch_size": 4,
            "prefill_chunk_tokens": 128,
            "max_kv_cache_capacity": 2048,
            "kv_page_tokens": 128,
            "kv_bytes_per_token": 0,
            "vision_output_bytes_per_token": 0,
            "graph_shapes": [],
        },
        "gpu": {
            "compute_capability": "8.6",
            "sm_count": 68,
            "memory_bytes": 10 * 1024**3,
            "product_name": "RTX 3080",
            "uuid": "node-a",
        },
        "software": {
            "tensorrt": "11",
            "cuda": "13.3",
            "driver": "",
            "plugin_hash": "plugin"
        },
    }


def test_build_keeps_only_action_fidelity_samples(tmp_path: Path) -> None:
    module = _module()
    metrics = tmp_path / "metrics.jsonl"
    rows = [
        {
            "global_decision_applied": True,
            "global_action": "decode",
            "global_execution_variant": "eager",
            "prefill_class": 0,
            "decode_batch": 32,
            "decode_context_max": 512,
            "global_reference_work_ms": 10.0,
            "makespan_gpu_ms": 7.0,
        },
        {
            "global_decision_applied": False,
            "global_action": "decode",
            "decode_batch": 64,
            "decode_context_max": 512,
            "global_reference_work_ms": 10.0,
            "makespan_gpu_ms": 8.0,
        },
    ]
    metrics.write_text("\n".join(json.dumps(row) for row in rows),
                       encoding="utf-8")
    records = module._records_from_metrics([metrics], 32)
    assert len(records) == 1
    assert records[0]["key"]["primary_batch_size"] == 32
    assert records[0]["key"]["primary_context_bucket"] == 1
    assert records[0]["observations"] == [{
        "reference_work_ms": 10.0,
        "makespan_ms": 7.0
    }]


def test_shape_contract_rejects_heterogeneous_fleet_data() -> None:
    module = _module()
    first = _fingerprint()
    second = _fingerprint()
    second["capability"]["max_decode_batch_size"] = 32
    assert module._shape_contract(first) != module._shape_contract(second)


def test_build_persists_independent_promotion_gates(tmp_path: Path) -> None:
    module = _module()
    metrics = tmp_path / "metrics.jsonl"
    metrics.write_text(
        json.dumps({
            "global_decision_applied": True,
            "global_action": "decode",
            "decode_batch": 4,
            "decode_context_max": 128,
            "global_reference_work_ms": 2.0,
            "makespan_gpu_ms": 1.5,
        }),
        encoding="utf-8",
    )
    fingerprint = tmp_path / "fingerprint.json"
    fingerprint.write_text(json.dumps(_fingerprint()), encoding="utf-8")
    output = tmp_path / "bundle.json"
    args = module._parser().parse_args([
        "build",
        "--metrics",
        str(metrics),
        "--fingerprint",
        str(fingerprint),
        "--promote-decode",
        "--promote-overlap",
        "--output",
        str(output),
    ])
    args.run(args)

    promotion = json.loads(output.read_text(encoding="utf-8"))["promotion"]
    assert promotion == {
        "decode_batching": True,
        "prefill_batching": False,
        "overlap_selection": True
    }


def test_promote_rebinds_controlled_records_to_exact_artifacts(
        tmp_path: Path) -> None:
    module = _module()
    source_fingerprint = _fingerprint()
    source_fingerprint["engine_hash"] = ""
    source_bundle = {
        "schema_version": 1,
        "bundle_version": "calibration-v1",
        "source": "build",
        "created_at_unix_ns": 1,
        "deployment": source_fingerprint,
        "records": [{
            "key": {
                "action": "decode"
            },
            "observations": []
        }],
    }
    input_path = tmp_path / "input.json"
    fingerprint_path = tmp_path / "fingerprint.json"
    output_path = tmp_path / "output.json"
    input_path.write_text(json.dumps(source_bundle), encoding="utf-8")
    fingerprint_path.write_text(json.dumps(_fingerprint()), encoding="utf-8")
    args = module._parser().parse_args([
        "promote",
        "--input",
        str(input_path),
        "--fingerprint",
        str(fingerprint_path),
        "--bundle-version",
        "exact-v2",
        "--promote-decode",
        "--output",
        str(output_path),
    ])
    args.run(args)

    promoted = json.loads(output_path.read_text(encoding="utf-8"))
    assert promoted["bundle_version"] == "exact-v2"
    assert promoted["deployment"]["engine_hash"] == "engine"
    assert promoted["records"] == source_bundle["records"]
    assert promoted["promotion"]["decode_batching"] is True


def test_fingerprint_hashes_engine_config_external_weights_and_plugin(
        tmp_path: Path) -> None:
    module = _module()
    engine_dir = tmp_path / "engine"
    engine_dir.mkdir()
    config = {"max_batch_size": 8, "kv_cache_dtype": "fp16", "dtype": "fp16"}
    config_path = engine_dir / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    engine_path = engine_dir / "llm.engine"
    engine_path.write_bytes(b"engine")
    external_path = engine_dir / "embedding.safetensors"
    external_path.write_bytes(b"weights")
    plugin_path = tmp_path / "plugin.so"
    plugin_path.write_bytes(b"plugin")
    args = argparse.Namespace(
        engine_dir=engine_dir,
        model_hash=None,
        onnx_hash=None,
        precision=None,
        compute_capability=None,
        sm_count=0,
        memory_bytes=0,
        gpu_name=None,
        gpu_uuid=None,
        tensorrt=None,
        cuda=None,
        driver=None,
        plugin=plugin_path,
    )

    fingerprint = module._fingerprint_from_engine(args)

    assert fingerprint["config_hash"] == hashlib.sha256(
        config_path.read_bytes()).hexdigest()
    assert fingerprint["engine_hash"] == hashlib.sha256(b"engine").hexdigest()
    assert fingerprint["external_weight_hash"] == hashlib.sha256(
        b"weights").hexdigest()
    assert fingerprint["software"]["plugin_hash"] == hashlib.sha256(
        b"plugin").hexdigest()
