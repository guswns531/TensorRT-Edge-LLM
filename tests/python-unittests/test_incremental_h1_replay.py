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

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).parents[
    2] / "benchmarks" / "phase_serving" / "analyze_incremental_h1_replay.py"
SPEC = importlib.util.spec_from_file_location("analyze_incremental_h1_replay",
                                              SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _event(kind: str, phase: str, execution_id: int,
           **extra: object) -> dict[str, object]:
    return {
        "_source_path": "run.log",
        "_source": "run.log:1",
        "run_id": "test",
        "event_kind": kind,
        "phase": phase,
        "execution_id": execution_id,
        "action_fidelity": True,
        **extra,
    }


def test_replays_earliest_event_instead_of_whole_pair() -> None:
    events = [
        _event("dispatch",
               "decode",
               2,
               incumbent_phase="encoder",
               incumbent_execution_id=1,
               incremental_action_id=99,
               plan_id=7,
               requested_action_direction="encoder_to_decode",
               planned_outstanding_mask=5),
        _event("completion",
               "encoder",
               1,
               gpu_start_us=0.0,
               gpu_end_us=10.0,
               gpu_duration_us=10.0),
        _event("completion",
               "decode",
               2,
               gpu_start_us=1.0,
               gpu_end_us=3.0,
               gpu_duration_us=2.0),
    ]

    artifact, errors = MODULE.build_replay(events)

    assert not errors
    assert artifact["summary"]["directional_pairs"] == 1
    vector = artifact["vectors"][0]
    assert vector["completed_phase"] == "decode"
    assert vector["projected_boundary_us"] == 2.0
    assert vector["whole_action_completion_us"] == 9.0
    assert vector["whole_action_overrun_us"] == 7.0
    assert vector["incremental_boundary_gpu_us"] == 1.0
    assert vector["successor_outstanding_mask"] == 1
    assert vector["remaining_phase"] == "encoder"


def test_cli_artifact_is_json_serializable(tmp_path: Path) -> None:
    events = [
        _event("dispatch",
               "prefill",
               4,
               incumbent_phase="decode",
               incumbent_execution_id=3,
               incremental_action_id=101,
               plan_id=8,
               requested_action_direction="decode_to_prefill",
               planned_outstanding_mask=6),
        _event("completion",
               "decode",
               3,
               gpu_start_us=2.0,
               gpu_end_us=6.0,
               gpu_duration_us=4.0),
        _event("completion",
               "prefill",
               4,
               gpu_start_us=3.0,
               gpu_end_us=9.0,
               gpu_duration_us=6.0),
    ]
    artifact, errors = MODULE.build_replay(events)
    output = tmp_path / "artifact.json"
    output.write_text(json.dumps(artifact), encoding="utf-8")

    assert not errors
    assert json.loads(output.read_text(
        encoding="utf-8"))["summary"]["invalid_successor_masks"] == 0
