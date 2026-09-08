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

import pytest
import torch

from tensorrt_edgellm.models.qwen3_vl import modeling_qwen3_vl_visual as visual


def _linear(_config, input_size, output_size, bias, module_name):
    return torch.nn.Linear(input_size, output_size, bias=bias)


@pytest.mark.parametrize('postshuffle', [False, True])
def test_patch_merger_uses_exact_gelu(monkeypatch, postshuffle):
    monkeypatch.setattr(visual, 'make_linear', _linear)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        merger = visual.Qwen3VLPatchMerger(4,
                                           8,
                                           2,
                                           None,
                                           use_postshuffle_norm=postshuffle)
    inputs = torch.linspace(-3, 3, 64).reshape(16, 4)
    normalized = merger.norm(inputs.reshape(-1, 16) if postshuffle else inputs)
    hidden = merger.linear_fc1(normalized.reshape(-1, 16))
    expected = merger.linear_fc2(torch.nn.GELU()(hidden))
    approximate = merger.linear_fc2(torch.nn.GELU(approximate='tanh')(hidden))
    actual = merger(inputs)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert not torch.equal(actual, approximate)


def test_vision_block_mlp_keeps_tanh_gelu(monkeypatch):
    monkeypatch.setattr(visual, 'make_linear', _linear)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        mlp = visual.Qwen3VLMLP(4, 8, None)
    inputs = torch.linspace(-3, 3, 16).reshape(4, 4)
    expected = mlp.linear_fc2(
        torch.nn.GELU(approximate='tanh')(mlp.linear_fc1(inputs)))
    torch.testing.assert_close(mlp(inputs), expected, atol=0, rtol=0)
