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
"""Compare bounded diagnostic logits only while generated prefixes still agree."""

import argparse
import json
import math
import pathlib
import struct


def compare_step(left_meta, right_meta, left, right):
    if left_meta['request_id'] != right_meta['request_id'] or left_meta[
            'step'] != right_meta['step']:
        raise ValueError('Request and step must match')
    if left_meta['prefix'] != right_meta['prefix']:
        return {'same_prefix': False}
    if not left or len(left) != len(right) or len(
            left) != left_meta['vocab'] or len(right) != right_meta['vocab']:
        raise ValueError('Vocabulary dimensions do not match')
    if not all(math.isfinite(x) for x in left + right):
        raise ValueError('Nonfinite logits')
    top_left = sorted(range(len(left)), key=lambda i: left[i],
                      reverse=True)[:2]
    top_right = sorted(range(len(right)), key=lambda i: right[i],
                       reverse=True)[:2]
    errors = [abs(x - y) for x, y in zip(left, right)]
    return {
        'same_prefix': True,
        'same_members':
        left_meta['batch_members'] == right_meta['batch_members'],
        'same_phase_row': left_meta['phase_row'] == right_meta['phase_row'],
        'argmax_equal': top_left[0] == top_right[0],
        'left_top2': top_left,
        'right_top2': top_right,
        'left_margin': left[top_left[0]] - left[top_left[-1]],
        'right_margin': right[top_right[0]] - right[top_right[-1]],
        'max_abs_error': max(errors),
        'mean_abs_error': sum(errors) / len(errors),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('left', type=pathlib.Path)
    parser.add_argument('right', type=pathlib.Path)
    parser.add_argument('--output', type=pathlib.Path, required=True)
    args = parser.parse_args()
    results = []
    files = sorted(args.left.glob('request-*-step-*.json'),
                   key=lambda p: int(p.stem.rsplit('-', 1)[1]))
    if not files:
        parser.error('No diagnostic snapshots')
    for path in files:
        other = args.right / path.name
        if not other.exists():
            raise ValueError('Missing paired snapshot: ' + str(other))
        metadata = [json.loads(p.read_text()) for p in (path, other)]
        logits = [[
            x[0]
            for x in struct.iter_unpack('<f',
                                        p.with_suffix('.fp32').read_bytes())
        ] for p in (path, other)]
        results.append({
            'step': metadata[0]['step'],
            **compare_step(*metadata, *logits)
        })
    args.output.write_text(json.dumps(results, indent=2))
    print(
        json.dumps({
            'steps':
            len(results),
            'same_prefix':
            sum(x['same_prefix'] for x in results),
            'first_argmax_divergence':
            next((x['step']
                  for x in results if x.get('argmax_equal') is False), None)
        }))


if __name__ == '__main__':
    main()
