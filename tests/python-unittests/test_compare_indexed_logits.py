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

import runpy
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

_SCRIPT = (Path(__file__).resolve().parents[2] / "scripts" / "cosmos_reason2" /
           "compare_indexed_logits.py")
_GLOBALS = runpy.run_path(str(_SCRIPT))


def test_cosine_returns_exact_one_for_identical_logits():
    logits = torch.tensor([0.1, -0.3, 1.2], dtype=torch.float16)
    assert _GLOBALS["cosine"](logits, logits.clone()) == 1.0


def test_top_two_reports_indices_values_and_margin_inputs():
    logits = torch.tensor([1.0, 4.0, 3.5, -2.0])
    top1, top1_value, top2, top2_value = _GLOBALS["top_two"](logits)
    assert (top1, top2) == (1, 2)
    assert top1_value - top2_value == pytest.approx(0.5)
