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

import csv
import runpy
from pathlib import Path

SCRIPT = (Path(__file__).resolve().parents[2] / "scripts" / "cosmos_reason2" /
          "build_phase_graph_warmup_profile.py")
GLOBALS = runpy.run_path(str(SCRIPT))


def test_profile_ranks_phase_binding_shapes_by_frequency(tmp_path):
    dispatch_csv = tmp_path / "dispatch.csv"
    fields = [
        "prefill_batch", "prefill_padded_tokens", "prefill_initial_rows",
        "prefill_past_kv_max", "decode_batch", "decode_context_tokens",
        "planned_decode_max_context_length"
    ]
    rows = [
        [4, 512, 4, 0, 64, 32768, 768],
        [4, 512, 4, 0, 64, 32768, 1024],
        [2, 256, 0, 384, 32, 8192, 512],
    ]
    with dispatch_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        writer.writerows(rows)

    profile = GLOBALS["build_profile"](dispatch_csv, 2, 2, 2)

    assert profile["version"] == 1
    assert profile["prefill"][0] == {
        "batch_size": 4,
        "chunk_length": 128,
        "past_kv_length": 0,
        "initial_chunk": True,
        "observed_dispatches": 2,
        "repetitions": 2,
    }
    assert profile["prefill"][1]["past_kv_length"] == 384
    assert profile["decode"][0]["batch_size"] == 64
    assert profile["decode"][0]["context_length"] == 1024
