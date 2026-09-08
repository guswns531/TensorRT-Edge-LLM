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

import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    'xqa_parity',
    Path(__file__).parents[2] / 'benchmarks/phase_serving/run_xqa_parity.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_exact_and_tolerant_outputs():
    assert MODULE.compare_outputs([1.0, -1.0], [1.0, -1.0])['exact']
    result = MODULE.compare_outputs([1.0, -1.0], [1.01, -1.01])
    assert not result['exact']
    assert result['within_1e2']
    assert not MODULE.compare_outputs([0.0], [0.02])['within_1e2']


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf')])
def test_nonfinite_outputs_fail(value):
    result = MODULE.compare_outputs([value], [value])
    assert not result['exact']
    assert not result['within_1e2']
    assert result['max_abs_error'] is None


@pytest.mark.parametrize('reference,candidate', [([], []), ([1.0], [])])
def test_invalid_shapes_fail(reference, candidate):
    with pytest.raises(ValueError):
        MODULE.compare_outputs(reference, candidate)
