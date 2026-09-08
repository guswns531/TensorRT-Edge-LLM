#!/usr/bin/env python3
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
"""Compare independently linked prebuilt/JIT probes on identical paged inputs."""

import argparse
import csv
import hashlib
import json
import math
import pathlib
import struct
import subprocess


def compare_outputs(reference, candidate):
    if not reference or len(reference) != len(candidate):
        raise ValueError('Outputs must have equal, nonzero lengths')
    pairs = list(zip(reference, candidate))
    finite = all(math.isfinite(a) and math.isfinite(b) for a, b in pairs)
    differences = [abs(a - b) for a, b in pairs] if finite else []
    return {
        'finite':
        finite,
        'exact':
        finite and all(a == b for a, b in pairs),
        'max_abs_error':
        max(differences) if finite else None,
        'mean_abs_error':
        sum(differences) / len(differences) if finite else None,
        'within_1e2':
        finite and all(abs(a - b) <= .01 + .01 * abs(a) for a, b in pairs),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe-dir', type=pathlib.Path, required=True)
    parser.add_argument('--output-dir', type=pathlib.Path, required=True)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('repeats must be positive')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    commands = []
    manifest = {'binary_sha256': {}, 'commands': commands}
    for mode in ('prebuilt', 'jit'):
        manifest['binary_sha256'][mode] = hashlib.sha256(
            (args.probe_dir / mode).read_bytes()).hexdigest()
    for batch in (1, 8, 32, 64):
        for context in (128, 512, 1536):
            for repeat in range(args.repeats):
                outputs = {}
                timings = {}
                modes = ('prebuilt',
                         'jit') if repeat % 2 == 0 else ('jit', 'prebuilt')
                for mode in modes:
                    prefix = args.output_dir / f'b{batch}-c{context}-r{repeat}-{mode}'
                    output = prefix.with_suffix('.fp16')
                    command = [
                        str(args.probe_dir / mode),
                        str(batch),
                        str(context),
                        str(output)
                    ]
                    commands.append(command)
                    with (args.output_dir /
                          'manifest.json').open('w') as stream:
                        json.dump(manifest, stream, indent=2)
                    result = subprocess.run(command,
                                            text=True,
                                            capture_output=True,
                                            check=False)
                    prefix.with_suffix('.log').write_text(result.stdout +
                                                          result.stderr)
                    result.check_returncode()
                    line = next(x for x in result.stdout.splitlines()
                                if x.startswith('XQA_PROBE,'))
                    fields = line.split(',')
                    timings[mode] = [float(fields[3]), float(fields[4])]
                    payload = output.read_bytes()
                    if len(payload) != batch * 16 * 128 * 2:
                        raise RuntimeError('Unexpected output size')
                    outputs[mode] = [
                        x[0] for x in struct.iter_unpack('<e', payload)
                    ]
                row = {
                    'batch':
                    batch,
                    'context':
                    context,
                    'repeat':
                    repeat,
                    **compare_outputs(outputs['prebuilt'], outputs['jit']), 'prebuilt_median_ms':
                    timings['prebuilt'][0],
                    'prebuilt_p95_ms':
                    timings['prebuilt'][1],
                    'jit_median_ms':
                    timings['jit'][0],
                    'jit_p95_ms':
                    timings['jit'][1]
                }
                rows.append(row)
                with (args.output_dir / 'summary.csv').open(
                        'w', newline='') as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
                print(json.dumps(row), flush=True)
                if not row['within_1e2']:
                    raise RuntimeError('XQA parity tolerance failed')


if __name__ == '__main__':
    main()
