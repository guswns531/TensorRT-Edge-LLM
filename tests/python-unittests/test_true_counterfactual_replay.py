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

import importlib.util
import json
from pathlib import Path


def _module():
    root = Path(__file__).resolve().parents[2]
    path = root / "benchmarks/phase_serving/analyze_true_counterfactual_replay.py"
    spec = importlib.util.spec_from_file_location("true_counterfactual_replay",
                                                  path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_branch(path: Path,
                  action: str,
                  dispatch_signature: int,
                  forced: bool = True) -> None:
    events = [{
        "event_kind": "decision",
        "plan_id": 3,
        "strict_snapshot_signature": 17,
        "dispatch_signature": dispatch_signature,
        "causal_replay_forced": forced,
        "action_kind": action,
        "request_ids": [1] if action == "encoder" else [1, 2],
        "selected_cohort": {
            "encoder_rows": 1,
            "prefill_rows": 0 if action == "encoder" else 1,
        },
    }, {
        "event_kind": "completion",
        "plan_id": 3,
        "dispatch_signature": dispatch_signature,
        "action_fidelity": True,
        "gpu_start_us": 10.0,
        "gpu_end_us": 20.0,
    }]
    path.write_text("".join("PHASE_SCHEDULER_EVENT\t" + json.dumps(event) +
                            "\n" for event in events),
                    encoding="utf-8")


def test_pairs_different_actions_from_same_strict_snapshot(tmp_path):
    module = _module()
    encoder = tmp_path / "encoder.jsonl"
    encoder_prefill = tmp_path / "encoder-prefill.jsonl"
    _write_branch(encoder, "encoder", 101)
    _write_branch(encoder_prefill, "encoder_prefill", 102)

    left = module.load_branch([encoder])
    right = module.load_branch([encoder_prefill])

    assert left.keys() == right.keys()
    assert left[17][0]["request_ids"] != right[17][0]["request_ids"]
    assert left[17][0]["fidelity"]
    assert right[17][0]["fidelity"]


def test_rejects_decision_completion_dispatch_mismatch(tmp_path):
    module = _module()
    path = tmp_path / "mismatch.jsonl"
    _write_branch(path, "encoder", 101)
    contents = path.read_text(encoding="utf-8").replace(
        '"dispatch_signature": 101, "action_fidelity"',
        '"dispatch_signature": 102, "action_fidelity"')
    path.write_text(contents, encoding="utf-8")

    result = module.load_branch([path])

    assert not result[17][0]["fidelity"]


def test_forced_only_excludes_natural_decisions(tmp_path):
    module = _module()
    path = tmp_path / "natural.jsonl"
    _write_branch(path, "encoder", 101, forced=False)

    assert module.load_branch([path])
    assert not module.load_branch([path], forced_only=True)
