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

import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    'phase_logits',
    Path(__file__).parents[2] /
    'benchmarks/phase_serving/compare_phase_logits.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_near_tie_and_prefix_boundary():
    meta = {
        'request_id': 2,
        'step': 1,
        'prefix': [7],
        'vocab': 2,
        'batch_members': [1, 2],
        'phase_row': 1
    }
    result = MODULE.compare_step(meta, meta, [1., 1.001], [1.001, 1.])
    assert not result['argmax_equal']
    assert result['left_margin'] == pytest.approx(.001)
    assert MODULE.compare_step(meta, {
        **meta, 'prefix': [8]
    }, [], []) == {
        'same_prefix': False
    }


def test_invalid_logits_fail():
    meta = {
        'request_id': 2,
        'step': 0,
        'prefix': [],
        'vocab': 1,
        'batch_members': [2],
        'phase_row': 0
    }
    with pytest.raises(ValueError):
        MODULE.compare_step(meta, meta, [float('nan')], [1.])
    with pytest.raises(ValueError):
        MODULE.compare_step(meta, meta, [], [])
