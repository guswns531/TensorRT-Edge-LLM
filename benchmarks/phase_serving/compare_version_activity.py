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
"""Run retained v0.10.0 and current binaries against one fingerprinted engine."""

import argparse
import hashlib
import json
import os
import pathlib
import subprocess
import zlib

import replay_retained_policy_commands as replay


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=pathlib.Path, required=True)
    parser.add_argument('--case', default='mixed')
    args = parser.parse_args()
    root = pathlib.Path('/home/sslab/TensorRT-Edge-LLM')
    source = root / '.local/v0101-forward-artifacts/cosmos-reason2-2b/engine-p8-d64-kv256-p128-vp1024-atomic'
    records = json.loads((
        root /
        '.local/results/v0101-forward-port/host-path-20260908/full12/commands.json'
    ).read_text())
    record = next(item for item in records if item['case'] == args.case)
    args.output.mkdir(parents=True, exist_ok=False)
    alias = args.output / 'engine'
    alias.mkdir()
    for path in source.iterdir():
        if path.is_file() and path.name != 'config.json':
            (alias / path.name).symlink_to(path)
    engine = source / 'llm.engine'
    size = engine.stat().st_size
    sample = 1048576
    crc = 0
    with engine.open('rb') as stream:
        for offset in sorted({0, size // 3, size * 2 // 3, size - sample}):
            stream.seek(offset)
            crc = zlib.crc32(stream.read(sample), crc)
    config = json.loads((source / 'config.json').read_text())
    with (source / 'embedding.safetensors').open('rb') as stream:
        length = int.from_bytes(stream.read(8), 'little')
        header = json.loads(stream.read(length))
    tensors = [value for key, value in header.items() if key != '__metadata__']
    if len(tensors) != 1 or tensors[0]['dtype'] != 'F16':
        raise ValueError('Expected one FP16 tied embedding')
    offsets = tensors[0]['data_offsets']
    config['tied_engine_contract'] = dict(version=1,
                                          engine_file='llm.engine',
                                          engine_bytes=size,
                                          engine_sample_bytes=sample,
                                          engine_sample_crc32=f'{crc:08x}',
                                          embedding_bytes=offsets[1] -
                                          offsets[0])
    (alias / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    documents, assets = replay.materialize_inputs(
        [record], args.output,
        [(str(root / '.local/upstream-v010'), str(root))])
    (args.output / 'inputs').mkdir()
    for destination, document in documents.values():
        destination.write_text(json.dumps(document, indent=2) + '\n')
    planned = []
    for version, build in [('v0100', 'v010-forward-build'),
                           ('v0101', 'v0101-forward-build-make')]:
        command = [
            value.replace('v0101-forward-build-make', build)
            for value in record['command']
        ]
        for option in ['--trace', '--generic-warmup-trace']:
            index = command.index(option) + 1
            command[index] = str(documents[pathlib.Path(command[index])][0])
        command[command.index('--output-dir') + 1] = str(args.output / version)
        command[command.index('--repeats') + 1] = '1'
        client = command[command.index('--client-script') + 1]
        command[command.index('--client-script') + 1] = str(
            pathlib.Path(__file__).with_name(
                'guarded_trace_client.py').resolve())
        command[-2] = str(alias)
        prefix = '/workspace/' + str(args.output.relative_to(
            root)) + '/' + version + '/run-{run}/activity'
        image_index = command.index('nvcr.io/nvidia/tensorrt:26.06-py3')
        command[image_index:image_index] = [
            '-e', 'TRT_PACKAGE_DIR=/opt/tensorrt', '-e',
            'TRT_EDGELLM_LEGACY_PAIR_ELIGIBILITY=1', '-e',
            'TRT_EDGELLM_EMIT_PHASE_METRICS=1', '-e',
            'TRT_EDGELLM_PHASE_TELEMETRY_LEVEL=full', '-e',
            'TRT_EDGELLM_PHASE_ACTIVITY_PREFIX=' + prefix, '-e',
            'TRT_EDGELLM_PHASE_TELEMETRY_PATH=' + prefix + '-events.jsonl'
        ]
        binary = root / '.local' / build / 'examples/llm/llm_phase_context_smoke'
        planned.append(
            dict(version=version,
                 command=command,
                 binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                 environment={'PHASE_TRACE_CLIENT_IMPL': client}))
    (args.output /
     'commands.json').write_text(json.dumps(planned, indent=2) + '\n')
    (args.output / 'contract.json').write_text(
        json.dumps(dict(
            case=args.case,
            assets=assets,
            comparison=
            'Same v0.10.1 engine; compatibility metadata only; not historical engine reproduction',
            config=config),
                   indent=2) + '\n')
    for item in planned:
        print(item['version'], flush=True)
        with (args.output / (item['version'] + '.log')).open('w') as log:
            result = subprocess.run(item['command'],
                                    stdout=log,
                                    stderr=subprocess.STDOUT,
                                    env={
                                        **os.environ,
                                        **item['environment']
                                    },
                                    check=False)
        print('exit', result.returncode, flush=True)
        if result.returncode:
            raise RuntimeError('Comparison stopped: ' + item['version'])


if __name__ == '__main__':
    main()
