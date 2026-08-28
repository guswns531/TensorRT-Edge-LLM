# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the phase cost bundle build and fleet aggregation tool."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _module():
    script = Path(__file__).resolve().parents[2] / "scripts" / "phase_cost_bundle.py"
    spec = importlib.util.spec_from_file_location("phase_cost_bundle", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fingerprint() -> dict:
    return {
        "model_hash": "model",
        "onnx_hash": "onnx",
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
        "software": {"tensorrt": "11", "cuda": "13.3", "driver": "", "plugin_hash": "plugin"},
    }


def test_build_keeps_only_action_fidelity_samples(tmp_path: Path) -> None:
    module = _module()
    metrics = tmp_path / "metrics.jsonl"
    rows = [
        {
            "global_decision_applied": True,
            "global_action": "decode",
            "global_execution_variant": "eager",
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
    metrics.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    records = module._records_from_metrics([metrics], 32)
    assert len(records) == 1
    assert records[0]["key"]["primary_batch_size"] == 32
    assert records[0]["key"]["primary_context_bucket"] == 1
    assert records[0]["observations"] == [{"reference_work_ms": 10.0, "makespan_ms": 7.0}]


def test_shape_contract_rejects_heterogeneous_fleet_data() -> None:
    module = _module()
    first = _fingerprint()
    second = _fingerprint()
    second["capability"]["max_decode_batch_size"] = 32
    assert module._shape_contract(first) != module._shape_contract(second)
