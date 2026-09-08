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
"""Retain TensorRT layer/tactic and input contracts without executing an engine."""

import argparse
import ctypes
import hashlib
import json
import pathlib

import tensorrt as trt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine', type=pathlib.Path, required=True)
    parser.add_argument('--plugin', type=pathlib.Path, required=True)
    parser.add_argument('--output', type=pathlib.Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    plugin = ctypes.CDLL(str(args.plugin), mode=ctypes.RTLD_GLOBAL)
    logger = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(logger, '')
    data = args.engine.read_bytes()
    with trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(data)
        if engine is None:
            raise RuntimeError('Engine deserialization failed')
        inspector = engine.create_engine_inspector()
        result = {
            'engine':
            str(args.engine),
            'sha256':
            hashlib.sha256(data).hexdigest(),
            'tensorrt_version':
            trt.__version__,
            'plugin':
            plugin._name,
            'device_memory_size':
            engine.device_memory_size_v2,
            'io': [],
            'layers':
            json.loads(
                inspector.get_engine_information(
                    trt.LayerInformationFormat.JSON)),
        }
        for index in range(engine.num_io_tensors):
            name = engine.get_tensor_name(index)
            result['io'].append({
                'name': name,
                'dtype': str(engine.get_tensor_dtype(name)),
                'shape': list(engine.get_tensor_shape(name)),
                'mode': str(engine.get_tensor_mode(name)),
            })
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x') as output:
            json.dump(result, output, indent=2)


if __name__ == '__main__':
    main()
