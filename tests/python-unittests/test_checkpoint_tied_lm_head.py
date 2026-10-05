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
"""tie_word_embeddings must not overwrite an lm_head.weight shipped by the checkpoint."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

safetensors_torch = pytest.importorskip("safetensors.torch")

from tensorrt_edgellm.checkpoint.loader import load_weights
from tensorrt_edgellm.models.linear import FP16Linear


class _TinyTiedModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(tie_word_embeddings=True)
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(8, 4, dtype=torch.float16)
        self.lm_head = FP16Linear(4, 8)

    def tie_weights(self):
        self.lm_head.weight = nn.Parameter(
            self.model.embed_tokens.weight.detach().clone(),
            requires_grad=False)


def _load(tmp_path, tensors):
    safetensors_torch.save_file(tensors, str(tmp_path / "model.safetensors"))
    model = _TinyTiedModel()
    load_weights(model, str(tmp_path))
    return model


def test_checkpoint_lm_head_is_kept(tmp_path):
    embed = torch.randn(8, 4, dtype=torch.float16)
    head = torch.randn(8, 4, dtype=torch.float16)
    model = _load(tmp_path, {
        "model.embed_tokens.weight": embed,
        "lm_head.weight": head
    })
    assert torch.equal(model.lm_head.weight, head)
    assert torch.equal(model.model.embed_tokens.weight, embed)


def test_missing_lm_head_is_tied(tmp_path):
    embed = torch.randn(8, 4, dtype=torch.float16)
    model = _load(tmp_path, {"model.embed_tokens.weight": embed})
    assert torch.equal(model.lm_head.weight, embed)
