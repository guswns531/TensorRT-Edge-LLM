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

import types

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
pytest.importorskip("onnx.helper")
pytest.importorskip("onnx.numpy_helper")
safetensors_torch = pytest.importorskip("safetensors.torch")

from tensorrt_edgellm import external_weights


def _make_initializer(name, array):
    return onnx.numpy_helper.from_array(array, name=name)


def _make_float_value_info(name):
    return onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT,
                                              [1, 2])


def test_resolve_externalize_all_includes_nvfp4_moe():
    assert external_weights.EXTERNAL_WEIGHT_NVFP4_MOE in (
        external_weights.resolve_externalize_weights(
            external_weights.EXTERNAL_WEIGHT_ALL))


def test_externalize_nvfp4_moe_plugin_initializers(tmp_path):
    onnx_path = tmp_path / "model.onnx"
    plugin_inputs = [
        "router_logits",
        "hidden_states",
        "fc1_qweights",
        "fc1_blocks_scale",
        "fc1_alpha",
        "fc2_qweights",
        "fc2_blocks_scale",
        "fc2_alpha",
        "input_global_scale",
        "down_input_scale",
        "e_score_correction_bias",
    ]
    expected_external_names = plugin_inputs[2:]
    initializers = [
        _make_initializer("fc1_qweights",
                          np.arange(16, dtype=np.int8).reshape(2, 2, 4)),
        _make_initializer("fc1_blocks_scale",
                          np.arange(8, dtype=np.int8).reshape(2, 2, 2)),
        _make_initializer("fc1_alpha", np.ones((2, ), dtype=np.float32)),
        _make_initializer("fc2_qweights",
                          np.arange(16, dtype=np.int8).reshape(2, 4, 2)),
        _make_initializer("fc2_blocks_scale",
                          np.arange(8, dtype=np.int8).reshape(2, 2, 2)),
        _make_initializer("fc2_alpha", np.ones((2, ), dtype=np.float32)),
        _make_initializer("input_global_scale", np.ones((2, ),
                                                        dtype=np.float32)),
        _make_initializer("down_input_scale", np.ones((2, ),
                                                      dtype=np.float32)),
        _make_initializer("e_score_correction_bias",
                          np.zeros((2, ), dtype=np.float32)),
        _make_initializer("non_plugin_weight", np.ones((1, ),
                                                       dtype=np.float32)),
    ]
    node = onnx.helper.make_node(
        "Nvfp4MoePlugin",
        plugin_inputs,
        ["moe_output"],
        domain="trt_edgellm",
    )
    graph = onnx.helper.make_graph(
        [node],
        "nvfp4_moe_external_weight_test",
        [
            _make_float_value_info("hidden_states"),
            _make_float_value_info("router_logits"),
        ],
        [_make_float_value_info("moe_output")],
        initializers,
    )
    model = onnx.helper.make_model(
        graph,
        opset_imports=[
            onnx.helper.make_opsetid("", 24),
            onnx.helper.make_opsetid("trt_edgellm", 1),
        ],
    )
    onnx.save_model(model, onnx_path)

    manifest = external_weights.externalize_model_weights(
        str(onnx_path), object(), externalize_weights=["nvfp4_moe"])

    assert manifest == [{
        "file": "external_nvfp4_moe_weights.safetensors",
        "kind": "nvfp4_moe_weights",
        "tensors": expected_external_names,
    }]
    saved_tensors = safetensors_torch.load_file(
        str(tmp_path / "external_nvfp4_moe_weights.safetensors"))
    assert set(saved_tensors) == set(expected_external_names)

    patched_model = onnx.load(onnx_path, load_external_data=False)
    graph_inputs = {
        graph_input.name
        for graph_input in patched_model.graph.input
    }
    remaining_initializers = {
        initializer.name
        for initializer in patched_model.graph.initializer
    }

    assert set(expected_external_names).issubset(graph_inputs)
    assert set(expected_external_names).isdisjoint(remaining_initializers)
    assert "non_plugin_weight" in remaining_initializers


def test_externalize_nvfp4_moe_streams_existing_external_data(tmp_path):
    onnx_path = tmp_path / "model_external.onnx"
    data_path = tmp_path / "model.onnx.data"
    plugin_inputs = [
        "router_logits",
        "hidden_states",
        "fc1_qweights",
        "fc1_blocks_scale",
        "fc1_alpha",
        "e_score_correction_bias",
    ]
    arrays = {
        "fc1_qweights": np.arange(16, dtype=np.int8).reshape(2, 2, 4),
        "fc1_blocks_scale": np.arange(8, dtype=np.int8).reshape(2, 2, 2),
        "fc1_alpha": np.ones((2, ), dtype=np.float32),
        "e_score_correction_bias": np.zeros((2, ), dtype=np.float32),
        "retained_weight": np.arange(4, dtype=np.float16),
    }
    expected_external_names = plugin_inputs[2:]
    initializers = [
        _make_initializer(name, array) for name, array in arrays.items()
    ]
    node = onnx.helper.make_node(
        "Nvfp4MoePlugin",
        plugin_inputs,
        ["moe_output"],
        domain="trt_edgellm",
    )
    graph = onnx.helper.make_graph(
        [node],
        "nvfp4_moe_external_data_test",
        [
            _make_float_value_info("hidden_states"),
            _make_float_value_info("router_logits"),
        ],
        [_make_float_value_info("moe_output")],
        initializers,
    )
    model = onnx.helper.make_model(
        graph,
        opset_imports=[
            onnx.helper.make_opsetid("", 24),
            onnx.helper.make_opsetid("trt_edgellm", 1),
        ],
    )
    onnx.save_model(
        model,
        onnx_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=data_path.name,
        size_threshold=0,
    )
    metadata_model = onnx.load(onnx_path, load_external_data=False)
    assert all(init.external_data for init in metadata_model.graph.initializer)

    manifest = external_weights.externalize_model_weights(
        str(onnx_path), object(), externalize_weights=["nvfp4_moe"])

    assert manifest == [{
        "file": "external_nvfp4_moe_weights.safetensors",
        "kind": "nvfp4_moe_weights",
        "tensors": expected_external_names,
    }]
    assert data_path.exists()
    saved_tensors = safetensors_torch.load_file(
        str(tmp_path / "external_nvfp4_moe_weights.safetensors"))
    for tensor_name in expected_external_names:
        np.testing.assert_array_equal(saved_tensors[tensor_name].numpy(),
                                      arrays[tensor_name])

    patched_model = onnx.load(onnx_path, load_external_data=False)
    graph_inputs = {
        graph_input.name
        for graph_input in patched_model.graph.input
    }
    remaining_initializers = {
        initializer.name
        for initializer in patched_model.graph.initializer
    }

    assert set(expected_external_names).issubset(graph_inputs)
    assert set(expected_external_names).isdisjoint(remaining_initializers)
    assert "retained_weight" in remaining_initializers


def test_externalize_nvfp4_moe_geforce_plugin_initializers(tmp_path):
    onnx_path = tmp_path / "model_geforce.onnx"
    plugin_inputs = [
        "router_logits",
        "hidden_states",
        "geforce_fc1_qweights",
        "geforce_fc1_blocks_scale",
        "geforce_fc1_alpha",
        "geforce_fc2_qweights",
        "geforce_fc2_blocks_scale",
        "geforce_fc2_alpha",
        "geforce_input_global_scale",
        "geforce_down_input_scale",
        "geforce_e_score_correction_bias",
    ]
    expected_external_names = plugin_inputs[2:]
    initializers = [
        _make_initializer("geforce_fc1_qweights",
                          np.arange(16, dtype=np.int8).reshape(2, 2, 4)),
        _make_initializer("geforce_fc1_blocks_scale",
                          np.arange(8, dtype=np.int8).reshape(2, 2, 2)),
        _make_initializer("geforce_fc1_alpha", np.ones((2, ),
                                                       dtype=np.float32)),
        _make_initializer("geforce_fc2_qweights",
                          np.arange(16, dtype=np.int8).reshape(2, 4, 2)),
        _make_initializer("geforce_fc2_blocks_scale",
                          np.arange(8, dtype=np.int8).reshape(2, 2, 2)),
        _make_initializer("geforce_fc2_alpha", np.ones((2, ),
                                                       dtype=np.float32)),
        _make_initializer("geforce_input_global_scale",
                          np.ones((2, ), dtype=np.float32)),
        _make_initializer("geforce_down_input_scale",
                          np.ones((2, ), dtype=np.float32)),
        _make_initializer("geforce_e_score_correction_bias",
                          np.zeros((2, ), dtype=np.float32)),
        _make_initializer("non_plugin_weight", np.ones((1, ),
                                                       dtype=np.float32)),
    ]
    node = onnx.helper.make_node(
        "NvFP4MoEPluginGeforce",
        plugin_inputs,
        ["moe_output"],
        domain="trt_edgellm",
    )
    graph = onnx.helper.make_graph(
        [node],
        "nvfp4_moe_geforce_external_weight_test",
        [
            _make_float_value_info("hidden_states"),
            _make_float_value_info("router_logits"),
        ],
        [_make_float_value_info("moe_output")],
        initializers,
    )
    model = onnx.helper.make_model(
        graph,
        opset_imports=[
            onnx.helper.make_opsetid("", 24),
            onnx.helper.make_opsetid("trt_edgellm", 1),
        ],
    )
    onnx.save_model(model, onnx_path)

    manifest = external_weights.externalize_model_weights(
        str(onnx_path), object(), externalize_weights=["nvfp4_moe"])

    assert manifest == [{
        "file": "external_nvfp4_moe_weights.safetensors",
        "kind": "nvfp4_moe_weights",
        "tensors": expected_external_names,
    }]
    saved_tensors = safetensors_torch.load_file(
        str(tmp_path / "external_nvfp4_moe_weights.safetensors"))
    assert set(saved_tensors) == set(expected_external_names)

    patched_model = onnx.load(onnx_path, load_external_data=False)
    graph_inputs = {
        graph_input.name
        for graph_input in patched_model.graph.input
    }
    remaining_initializers = {
        initializer.name
        for initializer in patched_model.graph.initializer
    }

    assert set(expected_external_names).issubset(graph_inputs)
    assert set(expected_external_names).isdisjoint(remaining_initializers)
    assert "non_plugin_weight" in remaining_initializers


def _make_tied_lm_head_model(onnx_path, tie_word_embeddings=True):
    weight_name = "lm_head.weight"
    initializers = [
        _make_initializer(weight_name,
                          np.arange(6, dtype=np.float16).reshape(3, 2))
    ]
    nodes = [
        onnx.helper.make_node("Transpose", [weight_name], ["head_weight_t"],
                              perm=[1, 0]),
        onnx.helper.make_node("MatMul", ["hidden_states", "head_weight_t"],
                              ["logits"]),
    ]
    graph = onnx.helper.make_graph(
        nodes,
        "tied_lm_head_external_weight_test",
        [
            onnx.helper.make_tensor_value_info(
                "hidden_states", onnx.TensorProto.FLOAT16, [1, 2])
        ],
        [
            onnx.helper.make_tensor_value_info(
                "logits", onnx.TensorProto.FLOAT16, [1, 3])
        ],
        initializers,
    )
    onnx.save_model(
        onnx.helper.make_model(
            graph, opset_imports=[onnx.helper.make_opsetid("", 24)]),
        onnx_path)
    return types.SimpleNamespace(
        config=types.SimpleNamespace(
            tie_word_embeddings=tie_word_embeddings,
            hidden_size=2,
            vocab_size=3,
            reduced_vocab_size=None,
            draft_vocab_size=None,
        ),
        lm_head=types.SimpleNamespace(weight=np.empty((3, 2))),
    )


def test_reuse_tied_lm_head_exposes_embedding_alias_without_sidecar(tmp_path):
    onnx_path = tmp_path / "model.onnx"
    model = _make_tied_lm_head_model(onnx_path)

    manifest = external_weights.externalize_model_weights(
        str(onnx_path),
        model,
        externalize_weights=["lm_head"],
        reuse_tied_lm_head=True)

    assert manifest == [{
        "source": "embedding",
        "kind": "tied_lm_head_weight",
        "tensors": ["lm_head.weight"],
    }]
    assert not (tmp_path / "external_lm_head_weight.safetensors").exists()
    patched_model = onnx.load(onnx_path, load_external_data=False)
    head_input = next(graph_input for graph_input in patched_model.graph.input
                      if graph_input.name == "lm_head.weight")
    assert [dim.dim_value
            for dim in head_input.type.tensor_type.shape.dim] == [2, 3]
    assert "lm_head.weight" not in {
        initializer.name
        for initializer in patched_model.graph.initializer
    }
    assert not any(node.op_type == "Transpose"
                   for node in patched_model.graph.node)
    matmul = next(node for node in patched_model.graph.node
                  if node.op_type == "MatMul")
    assert list(matmul.input)[1] == "lm_head.weight"


def test_reuse_tied_lm_head_rejects_untied_model(tmp_path):
    onnx_path = tmp_path / "model.onnx"
    model = _make_tied_lm_head_model(onnx_path, tie_word_embeddings=False)

    with pytest.raises(ValueError, match="tie_word_embeddings=True"):
        external_weights.externalize_model_weights(
            str(onnx_path),
            model,
            externalize_weights=["lm_head"],
            reuse_tied_lm_head=True)


def test_reuse_tied_lm_head_preserves_optimized_weight_layout(tmp_path):
    onnx_path = tmp_path / "model.onnx"
    weight_name = "lm_head.weight"
    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node("MatMul", ["hidden_states", weight_name],
                                  ["logits"])
        ],
        "optimized_tied_lm_head_test",
        [
            onnx.helper.make_tensor_value_info(
                "hidden_states", onnx.TensorProto.FLOAT16, [1, 2])
        ],
        [
            onnx.helper.make_tensor_value_info(
                "logits", onnx.TensorProto.FLOAT16, [1, 3])
        ],
        [
            _make_initializer(weight_name,
                              np.arange(6, dtype=np.float16).reshape(2, 3))
        ],
    )
    onnx.save_model(
        onnx.helper.make_model(
            graph, opset_imports=[onnx.helper.make_opsetid("", 24)]),
        onnx_path)
    model = types.SimpleNamespace(
        config=types.SimpleNamespace(
            tie_word_embeddings=True,
            hidden_size=2,
            vocab_size=3,
            reduced_vocab_size=None,
            draft_vocab_size=None,
        ),
        lm_head=types.SimpleNamespace(weight=np.empty((3, 2))),
    )

    external_weights.externalize_model_weights(str(onnx_path),
                                               model,
                                               externalize_weights=["lm_head"],
                                               reuse_tied_lm_head=True)

    patched_model = onnx.load(onnx_path, load_external_data=False)
    head_input = next(graph_input for graph_input in patched_model.graph.input
                      if graph_input.name == weight_name)
    assert [dim.dim_value
            for dim in head_input.type.tensor_type.shape.dim] == [2, 3]
    assert not any(node.name == "TiedLmHeadEmbeddingTranspose"
                   for node in patched_model.graph.node)
    matmul = next(node for node in patched_model.graph.node
                  if node.op_type == "MatMul")
    assert list(matmul.input)[1] == weight_name
