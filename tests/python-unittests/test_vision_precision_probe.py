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
from pathlib import Path

import onnx
import pytest

SPEC = importlib.util.spec_from_file_location(
    'vision_precision',
    Path(__file__).parents[2] /
    'benchmarks/phase_serving/export_vision_precision_probe.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_exact_mergers_preserve_mlp_activation():
    nodes = []
    for index, prefix in enumerate(
        ['blocks.0.mlp', 'deepstack_merger_list.0', 'merger']):
        hidden, activated = f'h{index}', f'g{index}'
        nodes.extend([
            onnx.helper.make_node('Gemm', ['x', prefix + '.linear_fc1.weight'],
                                  [hidden]),
            onnx.helper.make_node('Gelu', [hidden], [activated],
                                  approximate='tanh'),
            onnx.helper.make_node('Gemm',
                                  [activated, prefix + '.linear_fc2.weight'],
                                  [f'y{index}'])
        ])
    model = onnx.helper.make_model(
        onnx.helper.make_graph(nodes, 'mergers', [], []))
    assert MODULE.restore_exact_merger_gelu(model) == 2
    assert [n.attribute[0].s for n in model.graph.node
            if n.op_type == 'Gelu'] == [b'tanh', b'none', b'none']
    model.graph.node[-2].input[0] = 'wrong'
    with pytest.raises(ValueError, match='topology'):
        MODULE.restore_exact_merger_gelu(model)


def test_merger_interface_and_casts():
    inputs = [
        'x', 'merger.linear_fc1.weight', 'b1', 'merger.linear_fc2.weight', 'b2'
    ]
    tensors = [
        onnx.helper.make_tensor_value_info(x, onnx.TensorProto.FLOAT16, [2, 2])
        for x in inputs
    ]
    nodes = [
        onnx.helper.make_node('Gemm', inputs[:3], ['h']),
        onnx.helper.make_node('Gelu', ['h'], ['g']),
        onnx.helper.make_node('Gemm', ['g'] + inputs[3:], ['output'])
    ]
    model = onnx.helper.make_model(
        onnx.helper.make_graph(nodes, 'test', tensors, [
            onnx.helper.make_tensor_value_info(
                'output', onnx.TensorProto.FLOAT16, [2, 2])
        ]),
        opset_imports=[onnx.helper.make_opsetid('', 20)])
    MODULE.upcast_final_merger(model)
    onnx.checker.check_model(model)
    inferred = onnx.shape_inference.infer_shapes(model, check_type=True)
    assert inferred.graph.output[
        0].type.tensor_type.elem_type == onnx.TensorProto.FLOAT16
    types = {
        x.name: x.type.tensor_type.elem_type
        for x in inferred.graph.value_info
    }
    assert types['h'] == types['g'] == onnx.TensorProto.FLOAT
    assert sum(n.op_type == 'Cast' for n in model.graph.node) == 6
    with pytest.raises(ValueError):
        MODULE.upcast_final_merger(model)


def test_checkpoint_externalization_and_shape_guard():
    model = onnx.helper.make_model(onnx.helper.make_graph([], 'test', [], []))
    model.graph.initializer.append(
        onnx.helper.make_tensor('merger.linear_fc1.weight',
                                onnx.TensorProto.FLOAT16, [2], [1., 2.]))
    binding = {
        'engine_name': 'model.visual.merger.linear_fc1.weight',
        'shape': [3],
        'dtype': 'F16',
        'checkpoint_keys': ['original']
    }
    config = {'checkpoint_weight_bindings': [binding]}
    with pytest.raises(ValueError):
        MODULE.externalize(model, config)
    binding['shape'] = [2]
    assert MODULE.externalize(model, config) == 1
    assert not model.graph.initializer
    assert model.graph.input[0].name == 'merger.linear_fc1.weight'
    assert config['checkpoint_weight_bindings'][0]['checkpoint_keys'] == [
        'original'
    ]


def test_folded_norm_restores_runtime_cast():
    value = onnx.helper.make_tensor_value_info('folded',
                                               onnx.TensorProto.FLOAT, [2])
    value.metadata_props.add(key='pkg.onnxscript.optimizer.folded_from',
                             value="['merger.norm.weight']")
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Identity', ['folded'], ['output'])],
        'test', [], [
            onnx.helper.make_tensor_value_info('output',
                                               onnx.TensorProto.FLOAT, [2])
        ],
        initializer=[
            onnx.helper.make_tensor('folded', onnx.TensorProto.FLOAT, [2],
                                    [1., 2.])
        ],
        value_info=[value])
    model = onnx.helper.make_model(graph)
    config = {
        'checkpoint_weight_bindings': [{
            'engine_name':
            'model.visual.merger.norm.weight',
            'shape': [2],
            'dtype':
            'F16',
            'checkpoint_keys': ['model.visual.merger.norm.weight']
        }]
    }
    assert MODULE.externalize(model, config, externalize_norms=True) == 1
    assert model.graph.input[
        0].type.tensor_type.elem_type == onnx.TensorProto.FLOAT16
    assert model.graph.node[0].op_type == 'Cast'
    assert model.graph.node[0].output[0] == 'folded'
    onnx.checker.check_model(model)
