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
import json
import runpy
from pathlib import Path

SCRIPT = (Path(__file__).resolve().parents[2] / "scripts" / "cosmos_reason2" /
          "run_real_request_kv_matrix.py")
GLOBALS = runpy.run_path(str(SCRIPT))


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_pressure_model_blocks_until_a_page_bundle_is_released(tmp_path):
    request_csv = tmp_path / "requests.csv"
    write_rows(request_csv, [
        {
            "request_id": "1",
            "scheduled_arrival_us": "0",
            "completed_us": "100",
            "prompt_tokens": "128",
            "max_output_tokens": "128",
        },
        {
            "request_id": "2",
            "scheduled_arrival_us": "1",
            "completed_us": "200",
            "prompt_tokens": "128",
            "max_output_tokens": "128",
        },
    ])
    engine = GLOBALS["Engine"]("paged", tmp_path)
    case = GLOBALS["Case"](1, 1, "independent")
    result = GLOBALS["pressure_model"](request_csv, engine, case, 2, 128, 2)
    assert result["modelled_page_block_events"] == 1
    assert result["modelled_max_pending"] == 1
    assert result["modelled_peak_allocated_bundles"] == 2
    assert result["modelled_peak_pressure"] == 1.0


def test_scenarios_cover_all_requested_batch_pairs():
    cases = GLOBALS["scenarios"]([1, 2, 4, 8], [1, 2, 4, 8, 16],
                                 ["shared", "independent"])
    assert len(cases) == 40
    assert {case.name
            for case in cases} >= {"p8_d16_independent", "p1_d1_shared"}


def test_engine_uses_packed_prefill_reads_engine_contract(tmp_path):
    engine = GLOBALS["Engine"]("packed", tmp_path)
    (tmp_path / "config.json").write_text(json.dumps({"packed_prefill": True}),
                                          encoding="utf-8")

    assert GLOBALS["engine_uses_packed_prefill"](engine)


def test_engine_uses_packed_prefill_defaults_to_legacy_layout(tmp_path):
    engine = GLOBALS["Engine"]("legacy", tmp_path)
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")

    assert not GLOBALS["engine_uses_packed_prefill"](engine)
