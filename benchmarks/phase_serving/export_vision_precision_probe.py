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
"""Export a paired Qwen3-VL final-merger precision probe with checkpoint inputs."""

import argparse
import ast
import copy
import hashlib
import json
import math
import pathlib
import shutil

import onnx


def merger_gelu_nodes(model):
    """Validate and locate the checkpoint merger activation chains."""
    nodes = list(model.graph.node)
    targets = []
    for index, node in enumerate(nodes):
        if node.op_type != 'Gemm' or len(node.input) < 2:
            continue
        weight = node.input[1]
        if not (weight.startswith('merger.')
                or weight.startswith('deepstack_merger_list.')):
            continue
        if not weight.endswith('.linear_fc1.weight'):
            continue
        if index + 2 >= len(nodes):
            raise ValueError('Incomplete merger')
        activation, output = nodes[index + 1:index + 3]
        if (activation.op_type != 'Gelu'
                or list(activation.input) != [node.output[0]]
                or output.op_type != 'Gemm' or len(output.input) < 2
                or output.input[0] != activation.output[0] or output.input[1]
                != weight.replace('.linear_fc1.', '.linear_fc2.')):
            raise ValueError('Unsupported merger GELU topology')
        targets.append(activation)
    if not targets:
        raise ValueError('No merger GELU nodes')
    return targets


def restore_exact_merger_gelu(model):
    """Match checkpoint merger GELU without changing the vision block MLPs."""
    targets = merger_gelu_nodes(model)
    for node in targets:
        retained = [a for a in node.attribute if a.name != 'approximate']
        del node.attribute[:]
        node.attribute.extend(retained)
        node.attribute.append(onnx.helper.make_attribute(
            'approximate', 'none'))
    return len(targets)


def decompose_merger_gelu(model):
    """Use the direct builder's FP32 erf arithmetic with FP16 merger boundaries."""
    targets = {node.output[0] for node in merger_gelu_nodes(model)}
    used = {
        name
        for node in model.graph.node
        for name in (*node.input, *node.output)
    }
    used.update(tensor.name for tensor in model.graph.initializer)
    rewritten = []
    for node in model.graph.node:
        if node.op_type != 'Gelu' or node.output[0] not in targets:
            rewritten.append(node)
            continue
        prefix = node.output[0] + '_exact_erf_'
        names = {
            key: prefix + key
            for key in ('x32', 'sqrt2', 'one', 'half', 'scaled', 'erf', 'sum',
                        'factor', 'result')
        }
        if used.intersection(names.values()):
            raise ValueError('Merger erf tensor name collision')
        used.update(names.values())
        for key, value in [('sqrt2', math.sqrt(2.0)), ('one', 1.0),
                           ('half', 0.5)]:
            model.graph.initializer.append(
                onnx.helper.make_tensor(names[key], onnx.TensorProto.FLOAT,
                                        [1, 1], [value]))
        rewritten.append(
            onnx.helper.make_node('Cast', [node.input[0]], [names['x32']],
                                  name=names['x32'],
                                  to=onnx.TensorProto.FLOAT))
        for op, inputs, output in [('Div', ['x32', 'sqrt2'], 'scaled'),
                                   ('Erf', ['scaled'], 'erf'),
                                   ('Add', ['one', 'erf'], 'sum'),
                                   ('Mul', ['sum', 'half'], 'factor'),
                                   ('Mul', ['x32', 'factor'], 'result')]:
            rewritten.append(
                onnx.helper.make_node(op, [names[key] for key in inputs],
                                      [names[output]],
                                      name=names[output]))
        rewritten.append(
            onnx.helper.make_node('Cast', [names['result']],
                                  list(node.output),
                                  name=prefix + 'fp16',
                                  to=onnx.TensorProto.FLOAT16))
    del model.graph.node[:]
    model.graph.node.extend(rewritten)
    return len(targets)


def upcast_final_merger(model):
    """Keep the public FP16 output while computing the two merger GEMMs in FP32."""
    nodes = list(model.graph.node)
    targets = [
        i for i, n in enumerate(nodes) if n.op_type == 'Gemm'
        and len(n.input) == 3 and n.input[1].startswith('merger.linear_fc')
    ]
    if len(targets) != 2:
        raise ValueError('Expected exactly two final-merger Gemms')
    first, second = targets
    if (second != first + 2 or nodes[first + 1].op_type != 'Gelu'
            or nodes[first + 1].input[0] != nodes[first].output[0]
            or nodes[second].input[0] != nodes[first + 1].output[0]):
        raise ValueError('Unsupported final-merger topology')
    if nodes[second].output[0] != 'output':
        raise ValueError('Final merger must produce the public output')
    rewritten = []
    for index, node in enumerate(nodes):
        if index in targets:
            for slot in ([0, 1, 2] if index == first else [1, 2]):
                source = node.input[slot]
                name = source + '_merger_fp32'
                rewritten.append(
                    onnx.helper.make_node('Cast', [source], [name],
                                          name=name,
                                          to=onnx.TensorProto.FLOAT))
                node.input[slot] = name
            if index == second:
                node.output[0] = 'output_merger_fp32'
        rewritten.append(node)
        if index == second:
            rewritten.append(
                onnx.helper.make_node('Cast', ['output_merger_fp32'],
                                      ['output'],
                                      name='output_merger_fp16',
                                      to=onnx.TensorProto.FLOAT16))
    del model.graph.node[:]
    model.graph.node.extend(rewritten)
    # Intermediate dtype annotations describe the original FP16 graph.
    del model.graph.value_info[:]


def externalize(model, config, externalize_norms=False):
    """Reuse validated checkpoint bindings for matching, untransformed initializers."""
    prefix = 'model.visual.'
    by_name = {
        x['engine_name'][len(prefix):]: x
        for x in config['checkpoint_weight_bindings']
        if x['engine_name'].startswith(prefix)
    }
    retained = []
    bindings = []
    casts = []
    folded = {}
    if externalize_norms:
        for value in model.graph.value_info:
            for prop in value.metadata_props:
                if prop.key == 'pkg.onnxscript.optimizer.folded_from':
                    names = ast.literal_eval(prop.value.replace("\\'", "'"))
                    if len(names) == 1 and '.norm' in names[0] and names[
                            0] in by_name:
                        folded[value.name] = names[0]
    for tensor in model.graph.initializer:
        source = folded.get(tensor.name, tensor.name)
        if source not in by_name:
            retained.append(tensor)
            continue
        binding = copy.deepcopy(by_name[source])
        expected_type = onnx.TensorProto.FLOAT if tensor.name in folded else onnx.TensorProto.FLOAT16
        if tensor.data_type != expected_type or list(
                tensor.dims) != binding['shape']:
            raise ValueError('Checkpoint shape/dtype mismatch: ' + tensor.name)
        binding['engine_name'] = source
        bindings.append(binding)
        model.graph.input.append(
            onnx.helper.make_tensor_value_info(source,
                                               onnx.TensorProto.FLOAT16,
                                               tensor.dims))
        if tensor.name in folded:
            casts.append(
                onnx.helper.make_node('Cast', [source], [tensor.name],
                                      name=tensor.name + '_checkpoint_cast',
                                      to=onnx.TensorProto.FLOAT))
    if not bindings:
        raise ValueError('No compatible checkpoint bindings')
    del model.graph.initializer[:]
    model.graph.initializer.extend(retained)
    nodes = casts + list(model.graph.node)
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    config['checkpoint_weight_bindings'] = bindings
    return len(bindings)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--onnx-dir', type=pathlib.Path, required=True)
    parser.add_argument('--checkpoint-config',
                        type=pathlib.Path,
                        required=True)
    parser.add_argument('--output-dir', type=pathlib.Path, required=True)
    parser.add_argument('--merger-fp32', action='store_true')
    parser.add_argument('--externalize-norms', action='store_true')
    parser.add_argument('--exact-merger-gelu', action='store_true')
    parser.add_argument('--merger-erf-fp32', action='store_true')
    args = parser.parse_args()
    if args.merger_erf_fp32 and args.merger_fp32:
        parser.error('--merger-erf-fp32 requires FP16 GEMM boundaries')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    path = args.onnx_dir / 'model.onnx'
    model = onnx.load(path, load_external_data=False)
    config = json.loads(args.checkpoint_config.read_text())
    count = externalize(model, config, args.externalize_norms)
    onnx.external_data_helper.load_external_data_for_model(
        model, str(args.onnx_dir))
    exact_mergers = restore_exact_merger_gelu(
        model) if args.exact_merger_gelu else 0
    if args.merger_fp32:
        upcast_final_merger(model)
    erf_mergers = decompose_merger_gelu(model) if args.merger_erf_fp32 else 0
    onnx.checker.check_model(model)
    onnx.save(model, args.output_dir / 'model.onnx')
    (args.output_dir / 'config.json').write_text(json.dumps(config, indent=2))
    shutil.copyfile(args.onnx_dir / 'preprocessor_config.json',
                    args.output_dir / 'preprocessor_config.json')
    (args.output_dir / 'precision-probe.json').write_text(
        json.dumps(
            {
                'source_onnx': str(path.resolve()),
                'source_onnx_sha256': hashlib.sha256(
                    path.read_bytes()).hexdigest(),
                'checkpoint_config': str(args.checkpoint_config.resolve()),
                'merger_fp32': args.merger_fp32,
                'externalize_norms': args.externalize_norms,
                'exact_merger_gelu_count': exact_mergers,
                'erf_fp32_merger_count': erf_mergers,
                'external_weight_count': count,
                'experimental_only': True,
            },
            indent=2))


if __name__ == '__main__':
    main()
