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
"""Summarize measured E/P/D completion and dispatch gaps without mixing clock domains."""

import argparse
import collections
import json
import pathlib
import statistics


def distribution(values):
    if not values:
        return {'count': 0}
    ordered = sorted(values)
    position = 0.95 * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return dict(count=len(values),
                mean=statistics.mean(values),
                median=statistics.median(values),
                p95=ordered[low] + (ordered[high] - ordered[low]) *
                (position - low),
                maximum=max(values),
                negative_count=sum(v < 0 for v in values))


def covered_duration(start, end, intervals):
    clipped = sorted((max(start, a), min(end, b)) for a, b in intervals
                     if a < end and b > start)
    total = 0.0
    cursor = start
    for a, b in clipped:
        total += max(0.0, b - max(a, cursor))
        cursor = max(cursor, b)
    return total


def analyze(path):
    events = []
    epochs = 0
    for line in path.read_text().splitlines():
        kind, separator, value = line.partition('\t')
        if not separator:
            continue
        if kind == 'PHASE_EPOCH' and json.loads(
                value)['kind'] == 'measurement':
            events = []
            epochs += 1
        elif kind == 'PHASE_SCHEDULER_EVENT':
            events.append(json.loads(value))
    if epochs != 1:
        raise ValueError('Expected one explicit measurement epoch')
    dispatches = {
        (e['phase'], e['execution_id']): e
        for e in events if e['event_kind'] == 'dispatch'
    }
    complete = [e for e in events if e['event_kind'] == 'completion']
    if len(complete) != len(dispatches):
        raise ValueError('Incomplete dispatch/completion trace')
    phases = {}
    for phase in ['encoder', 'prefill', 'decode']:
        rows = sorted((e for e in complete if e['phase'] == phase),
                      key=lambda e: e['gpu_start_us'])
        if not rows:
            raise ValueError('Missing phase: ' + phase)
        phases[phase] = rows
    summary = {}
    for phase, rows in phases.items():
        summary[phase] = dict(
            dispatches=len(rows),
            batch_histogram=dict(
                collections.Counter(r['cohort'][phase + '_rows']
                                    for r in rows)),
            gpu_ms=distribution([r['gpu_duration_us'] / 1000 for r in rows]),
            gpu_total_ms=sum(r['gpu_duration_us'] / 1000 for r in rows))
    decode = phases['decode']
    host_gaps, gpu_gaps, e_covered, p_covered = [], [], [], []
    e_intervals = [(e['gpu_start_us'], e['gpu_end_us'])
                   for e in phases['encoder']]
    p_intervals = [(e['gpu_start_us'], e['gpu_end_us'])
                   for e in phases['prefill']]
    for previous, current in zip(decode, decode[1:]):
        dispatch = dispatches['decode', current['execution_id']]
        host_gaps.append((dispatch['enqueue_host_ns'] -
                          previous['completion_visible_host_ns']) / 1e6)
        start, end = previous['gpu_end_us'], current['gpu_start_us']
        gpu_gaps.append((end - start) / 1000)
        e_covered.append(covered_duration(start, end, e_intervals) / 1000)
        p_covered.append(covered_duration(start, end, p_intervals) / 1000)
    downstream = {}
    for phase in ['prefill', 'decode']:
        delays = []
        for encoder in phases['encoder']:
            for request in encoder['request_ids']:
                candidates = [
                    r for r in phases[phase] if request in r['request_ids']
                    and r['gpu_start_us'] >= encoder['gpu_end_us']
                ]
                if candidates:
                    delays.append(
                        (candidates[0]['gpu_start_us'] - encoder['gpu_end_us'])
                        / 1000)
        downstream[phase + '_gpu_delay_ms'] = distribution(delays)
    return dict(
        source=str(path),
        phases=summary,
        decode_snapshot_notice_to_next_enqueue_ms=distribution(host_gaps),
        decode_gpu_end_to_next_start_ms=distribution(gpu_gaps),
        encoder_covered_decode_gap_ms=distribution(e_covered),
        prefill_covered_decode_gap_ms=distribution(p_covered),
        encoder_to_first_downstream=downstream,
        caveat=
        ('Global D gaps include periods without ready decode work; E/P coverage may overlap. '
         'Host completion is a coordinator snapshot notice, not first CUDA completion visibility; '
         'negative host gaps are retained and must not be used as CPU turnaround latency.'
         ))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--events',
                        type=pathlib.Path,
                        action='append',
                        required=True)
    parser.add_argument('--output', type=pathlib.Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = [analyze(path) for path in args.events]
    with args.output.open('x') as output:
        json.dump(result, output, indent=2)


if __name__ == '__main__':
    main()
