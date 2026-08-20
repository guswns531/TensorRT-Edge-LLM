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

from scripts.cosmos_reason2 import build_phase_cost_model_from_gateway


def metric(kind, prefill_batch=0, decode_batch=0):
    return {
        "kind": kind,
        "prefill_batch": prefill_batch,
        "decode_batch": decode_batch,
        "prefill_chunk_length": 128,
        "prefill_initial_rows": prefill_batch,
        "prefill_past_kv_max": 0,
        "decode_context_max": 300,
        "decode_context_tokens": 600,
        "prefill_gpu_ms": 12.0,
        "decode_gpu_ms": 8.0,
        "makespan_gpu_ms": 13.0,
    }


def test_summarizes_compatible_decode_prefill_and_overlap_points():
    rows = [metric(2, decode_batch=31), metric(2, decode_batch=31)]
    rows += [metric(3, prefill_batch=3, decode_batch=31)] * 2

    result = build_phase_cost_model_from_gateway.summarize(rows, "engine", 2)

    assert result["decode"][0]["batch_size"] == 32
    assert result["decode"][0]["max_context_length"] == 512
    assert result["prefill"][0]["batch_size"] == 4
    assert result["prefill"][0]["max_past_kv_length"] == 0
    assert result["overlap"][0]["decode_batch_size"] == 32
    assert result["overlap"][0]["chunk_length"] == 128
