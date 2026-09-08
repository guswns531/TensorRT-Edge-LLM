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
"""Alternate two vision engines on shared fixed tensors and real checkpoint weights."""

import argparse
import ctypes
import hashlib
import json
import pathlib
import struct

import numpy as np
import tensorrt as trt
from cuda.bindings import runtime as cuda


def checked(result):
    if result[0] != cuda.cudaError_t.cudaSuccess:
        raise RuntimeError(str(result[0]))
    return result[1] if len(result) == 2 else result[1:]


def checkpoint_tensor(path, header, offset, binding):
    if binding['source_layout'] != 'fp16' or len(
            binding['checkpoint_keys']) != 1:
        raise ValueError(
            'Only single-key FP16 checkpoint bindings are supported')
    entry = header[binding['checkpoint_keys'][0]]
    if entry['shape'] != binding['shape']:
        raise ValueError('Checkpoint shape mismatch')
    begin, end = entry['data_offsets']
    with path.open('rb') as source:
        source.seek(offset + begin)
        raw = source.read(end - begin)
    if entry['dtype'] == 'BF16':
        value = (
            np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16).view(
                np.float32)
    elif entry['dtype'] == 'F16':
        value = np.frombuffer(raw, dtype=np.float16)
    else:
        raise ValueError('Unsupported checkpoint dtype')
    return value.astype(np.float16).reshape(entry['shape'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine',
                        type=pathlib.Path,
                        action='append',
                        required=True)
    parser.add_argument('--checkpoint', type=pathlib.Path, required=True)
    parser.add_argument('--plugin', type=pathlib.Path, required=True)
    parser.add_argument('--output', type=pathlib.Path, required=True)
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--patches-per-image', type=int, default=512)
    args = parser.parse_args()
    if len(args.engine) != 2 or args.output.exists():
        parser.error('Provide two engines and a new output path')
    if min(args.warmup, args.iterations, args.repeats) < 1:
        parser.error('Counts must be positive')
    if args.patches_per_image not in (512, 2048):
        parser.error('Supported fixed image shapes are 512 or 2048 patches')
    plugin = ctypes.CDLL(str(args.plugin), mode=ctypes.RTLD_GLOBAL)
    logger = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(logger, '')
    runtime = trt.Runtime(logger)
    engines = [
        runtime.deserialize_cuda_engine(p.read_bytes()) for p in args.engine
    ]
    if any(e is None for e in engines):
        raise RuntimeError('Engine deserialization failed')
    configs = [
        json.loads(p.with_name('config.json').read_text()) for p in args.engine
    ]
    if configs[0]['checkpoint_weight_bindings'] != configs[1][
            'checkpoint_weight_bindings']:
        raise ValueError('Checkpoint binding contracts differ')
    with args.checkpoint.open('rb') as source:
        length = struct.unpack('<Q', source.read(8))[0]
        header = json.loads(source.read(length))
    stream = checked(cuda.cudaStreamCreateWithFlags(
        cuda.cudaStreamNonBlocking))
    contexts = [e.create_execution_context() for e in engines]
    allocations = []
    events = []

    def upload(value):
        value = np.ascontiguousarray(value)
        pointer = checked(cuda.cudaMalloc(value.nbytes))
        allocations.append(pointer)
        # Setup is outside the timed execution region.
        checked(
            cuda.cudaMemcpy(pointer, value.ctypes.data, value.nbytes,
                            cuda.cudaMemcpyKind.cudaMemcpyHostToDevice))
        return pointer

    rows = []
    try:
        for binding in configs[0]['checkpoint_weight_bindings']:
            pointer = upload(
                checkpoint_tensor(args.checkpoint, header, length + 8,
                                  binding))
            for context in contexts:
                if not context.set_tensor_address(binding['engine_name'],
                                                  pointer):
                    raise RuntimeError('Weight binding failed')
        begin = checked(cuda.cudaEventCreate())
        end = checked(cuda.cudaEventCreate())
        events.extend([begin, end])
        for batch in [1, 2, 4]:
            patches = batch * args.patches_per_image
            inputs = {
                'input':
                np.random.default_rng(0).standard_normal(
                    (patches, 1536)).astype(np.float16),
                'rotary_pos_emb':
                np.zeros((patches, 32), np.float32),
                'cu_seqlens':
                np.arange(batch + 1, dtype=np.int32) * args.patches_per_image,
                'fast_pos_embed_idx':
                np.tile(np.arange(patches, dtype=np.int64) % 512, (4, 1)),
                'fast_pos_embed_weight':
                np.full((4, patches), 0.25, np.float16),
                'max_seqlen_carrier':
                np.zeros(args.patches_per_image, np.int32),
            }
            input_hash = hashlib.sha256()
            shape_allocation_start = len(allocations)
            for name, value in inputs.items():
                input_hash.update(name.encode())
                input_hash.update(value.tobytes())
                pointer = upload(value)
                for context in contexts:
                    if not context.set_input_shape(
                            name,
                            value.shape) or not context.set_tensor_address(
                                name, pointer):
                        raise RuntimeError('Input binding failed: ' + name)
            outputs = []
            for engine, context in zip(engines, contexts):
                if context.infer_shapes():
                    raise RuntimeError('Unspecified shapes')
                output = {}
                for i in range(engine.num_io_tensors):
                    name = engine.get_tensor_name(i)
                    if engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                        value = np.zeros(context.get_tensor_shape(name),
                                         dtype=trt.nptype(
                                             engine.get_tensor_dtype(name)))
                        pointer = upload(value)
                        if not context.set_tensor_address(name, pointer):
                            raise RuntimeError('Output binding failed')
                        output[name] = (pointer, value)
                outputs.append(output)
            for repeat in range(args.repeats):
                for index in ([0, 1] if repeat % 2 == 0 else [1, 0]):
                    context = contexts[index]
                    for _ in range(args.warmup):
                        if not context.execute_async_v3(int(stream)):
                            raise RuntimeError('Warmup failed')
                    checked(cuda.cudaStreamSynchronize(stream))
                    samples = []
                    for _ in range(args.iterations):
                        checked(cuda.cudaEventRecord(begin, stream))
                        if not context.execute_async_v3(int(stream)):
                            raise RuntimeError('Enqueue failed')
                        checked(cuda.cudaEventRecord(end, stream))
                        checked(cuda.cudaEventSynchronize(end))
                        samples.append(
                            checked(cuda.cudaEventElapsedTime(begin, end)))
                    digest = hashlib.sha256()
                    for pointer, value in outputs[index].values():
                        checked(
                            cuda.cudaMemcpy(
                                value.ctypes.data, pointer, value.nbytes,
                                cuda.cudaMemcpyKind.cudaMemcpyDeviceToHost))
                        if not np.isfinite(value).all():
                            raise RuntimeError('Non-finite encoder output')
                        digest.update(value.tobytes())
                    row = dict(engine=index,
                               batch=batch,
                               patches=patches,
                               repeat=repeat,
                               median_ms=float(np.median(samples)),
                               p95_ms=float(np.percentile(samples, 95)),
                               samples_ms=samples,
                               input_sha256=input_hash.hexdigest(),
                               output_sha256=digest.hexdigest())
                    rows.append(row)
                    print(json.dumps({
                        k: v
                        for k, v in row.items() if k != 'samples_ms'
                    }),
                          flush=True)
            checked(cuda.cudaStreamSynchronize(stream))
            for pointer in allocations[shape_allocation_start:]:
                checked(cuda.cudaFree(pointer))
            del allocations[shape_allocation_start:]
        result = dict(command=vars(args),
                      synthetic_inputs=True,
                      plugin=plugin._name,
                      tensorrt=trt.__version__,
                      engines=[
                          dict(path=str(p),
                               sha256=hashlib.sha256(
                                   p.read_bytes()).hexdigest(),
                               memory_bytes=e.device_memory_size_v2)
                          for p, e in zip(args.engine, engines)
                      ],
                      rows=rows)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x') as target:
            json.dump(result, target, indent=2, default=str)
    finally:
        checked(cuda.cudaStreamSynchronize(stream))
        for event in events:
            checked(cuda.cudaEventDestroy(event))
        for pointer in allocations:
            checked(cuda.cudaFree(pointer))
        checked(cuda.cudaStreamDestroy(stream))


if __name__ == '__main__':
    main()
