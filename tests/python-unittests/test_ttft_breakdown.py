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
          "analyze_ttft_breakdown.py")
GLOBALS = runpy.run_path(str(SCRIPT))


def write_rows(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_breakdown_closes_ttft_across_chunked_prefill(tmp_path):
    requests = tmp_path / "requests.csv"
    timeline = tmp_path / "timeline.csv"
    kernels = tmp_path / "kernels.csv"
    write_rows(requests, [{
        "request_id": 7,
        "scheduled_arrival_us": 0,
        "submitted_us": 10,
        "admitted_us": 20,
        "first_token_us": 200,
        "prompt_tokens": 8,
    }])
    write_rows(timeline, [{
        "request_id": 7,
        "token_offset": 0,
        "final_chunk": 0,
        "dispatch_selected_us": 30,
        "pack_completed_us": 40,
        "phase_completed_us": 90,
        "kernel_dispatch_index": 3,
    }, {
        "request_id": 7,
        "token_offset": 4,
        "final_chunk": 1,
        "dispatch_selected_us": 110,
        "pack_completed_us": 120,
        "phase_completed_us": 190,
        "kernel_dispatch_index": 5,
    }])
    kernel_rows = []
    for dispatch, engine_ms in ((3, 0.03), (5, 0.04)):
        for group, gpu_ms in (("prefill_prepare", 0.005), ("prefill_engine",
                                                           engine_ms),
                              ("prefill_cache_commit",
                               0.002), ("prefill_sample", 0.003)):
            kernel_rows.append({
                "dispatch_index": dispatch,
                "group": group,
                "gpu_ms": gpu_ms,
            })
    write_rows(kernels, kernel_rows)

    rows = GLOBALS["build_breakdown"](requests, timeline, kernels)

    assert len(rows) == 1
    row = rows[0]
    assert row["ttft_us"] == 200
    assert row["scheduler_wait_us"] == 30
    assert row["host_pack_us"] == 20
    assert row["phase_service_wall_us"] == 120
    assert row["prefill_engine_gpu_us"] == 70
    assert row["first_token_delivery_us"] == 10
