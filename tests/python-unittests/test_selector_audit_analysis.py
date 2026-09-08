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
import pathlib

import pytest

SPEC = importlib.util.spec_from_file_location(
    'selector_audit',
    pathlib.Path(__file__).parents[2] /
    'benchmarks/phase_serving/analyze_selector_audit.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_actual_inputs_override_preview_evidence(tmp_path):
    source = tmp_path / 'events.jsonl'
    event = dict(
        event_kind='decision',
        decision_id=1,
        action_kind='encoder',
        ready=dict(decode_rows=4),
        candidates=[dict(action_kind='decode', max_slo_violation_us=0)],
        selector_audit=dict(inputs=[
            dict(action_id=2,
                 action_kind='encoder',
                 hard_feasible=True,
                 max_slo_violation_us=30)
        ],
                            selected_action_id=2,
                            reason='minimum_violation',
                            post_select_override=False))
    source.write_text(
        'PHASE_EPOCH\t{"kind":"measurement"}\nPHASE_SCHEDULER_EVENT\t' +
        json.dumps(event))
    counts = MODULE.analyze(source)['counts']
    assert counts['decode_candidate_absent'] == 1
    assert counts['late_choice_with_safe_decode'] == 0
    event['selector_audit']['inputs'].append(
        dict(action_id=3,
             action_kind='decode',
             hard_feasible=True,
             max_slo_violation_us=0))
    source.write_text(
        'PHASE_EPOCH\t{"kind":"measurement"}\nPHASE_SCHEDULER_EVENT\t' +
        json.dumps(event))
    assert MODULE.analyze(
        source)['counts']['late_choice_with_safe_decode'] == 1


def test_requires_epoch_and_valid_selection(tmp_path):
    source = tmp_path / 'events.jsonl'
    source.write_text('')
    with pytest.raises(ValueError, match='measurement epoch'):
        MODULE.analyze(source)
