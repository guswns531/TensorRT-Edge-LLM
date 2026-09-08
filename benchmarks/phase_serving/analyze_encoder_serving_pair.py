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
        actions_with_ready_decode=dict(
            collections.Counter(e['action_kind'] for e in events
                                if e['event_kind'] == 'decision'
                                and e['ready']['decode_rows'] > 0)),
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


def analyze_ready_path(path):
    """Join producer tickets to the next request-local decode start in one host clock."""
    timeline = []
    epochs = 0
    for line in path.read_text().splitlines():
        kind, separator, value = line.partition('\t')
        if not separator:
            continue
        if kind == 'PHASE_EPOCH' and json.loads(
                value)['kind'] == 'measurement':
            timeline = []
            epochs += 1
        elif kind == 'PHASE_TIMELINE':
            timeline.append(json.loads(value))
    if epochs != 1 or not timeline:
        raise ValueError(
            'Expected one measurement epoch with request timeline')
    tickets = collections.defaultdict(dict)
    starts = collections.defaultdict(list)
    for event in timeline:
        request = event['request_index']
        stage = event['stage']
        if stage == 'decode_start':
            starts[request].append(event)
        if 'sampling_' in stage or stage.endswith(
                'token_committed') or stage == 'decode_ready':
            key = (request, event['dispatch_index'])
            if stage in tickets[key]:
                raise ValueError('Duplicate ticket stage')
            tickets[key][stage] = event['timestamp_us']
    for events in starts.values():
        events.sort(key=lambda event: event['timestamp_us'])
    rows = []
    for (request,
         ticket), stages in sorted(tickets.items(),
                                   key=lambda item: min(item[1].values())):
        prefix = 'prefill' if 'prefill_sampling_submit' in stages else 'decode'
        names = [
            prefix + '_sampling_submit', prefix + '_sampling_ready',
            prefix + '_sampling_collected', prefix + '_token_committed'
        ]
        if any(name not in stages for name in names):
            raise ValueError('Incomplete sampling ticket')
        values = [stages[name] for name in names]
        if values != sorted(values):
            raise ValueError('Nonmonotonic producer timestamps')
        row = dict(request=request,
                   ticket=ticket,
                   producer=prefix,
                   submit_to_handling_ms=(values[1] - values[0]) / 1000,
                   handling_to_collect_ms=(values[2] - values[1]) / 1000,
                   collect_to_commit_ms=(values[3] - values[2]) / 1000)
        if 'decode_ready' in stages:
            ready = stages['decode_ready']
            if ready < values[-1]:
                raise ValueError('Ready precedes token commit')
            candidates = starts[request]
            if not candidates or candidates[0]['timestamp_us'] < ready:
                raise ValueError('Missing or unmatched decode start')
            start = candidates.pop(0)
            row.update(
                commit_to_ready_ms=(ready - values[-1]) / 1000,
                ready_to_decode_start_ms=(start['timestamp_us'] - ready) /
                1000,
                ready_host_us=ready,
                decode_start_host_us=start['timestamp_us'],
                next_decode_dispatch=start['dispatch_index'],
                next_decode_batch=start['batch_size'])
        rows.append(row)
    if any(starts.values()):
        raise ValueError('Decode starts without producer readiness')
    phase_starts = {}
    phase_intervals = collections.defaultdict(list)
    for event in sorted(timeline, key=lambda value: value['timestamp_us']):
        phase, _, suffix = event['stage'].partition('_')
        if phase not in ('encoder', 'prefill',
                         'decode') or suffix not in ('start', 'done'):
            continue
        key = (phase, event['request_index'], event['dispatch_index'])
        if suffix == 'start':
            phase_starts[key] = event['timestamp_us']
        elif key in phase_starts:
            phase_intervals[phase].append(
                (phase_starts.pop(key), event['timestamp_us']))
    for row in rows:
        if 'ready_host_us' not in row:
            continue
        start, end = row['ready_host_us'], row['decode_start_host_us']
        for phase, intervals in phase_intervals.items():
            row[phase + '_host_span_while_ready_ms'] = covered_duration(
                start, end, intervals) / 1000
        all_intervals = [
            interval for intervals in phase_intervals.values()
            for interval in intervals
        ]
        row['uncovered_host_span_while_ready_ms'] = (
            end - start - covered_duration(start, end, all_intervals)) / 1000
    fields = [
        'submit_to_handling_ms', 'handling_to_collect_ms',
        'collect_to_commit_ms', 'commit_to_ready_ms',
        'ready_to_decode_start_ms', 'encoder_host_span_while_ready_ms',
        'prefill_host_span_while_ready_ms', 'decode_host_span_while_ready_ms',
        'uncovered_host_span_while_ready_ms'
    ]
    summaries = {}
    for producer in ['all', 'prefill', 'decode']:
        selected = [
            row for row in rows
            if producer == 'all' or row['producer'] == producer
        ]
        summaries[producer] = {
            field:
            distribution([row[field] for row in selected if field in row])
            for field in fields
        }
    return dict(
        rows=rows,
        summaries=summaries,
        caveat=
        ('Sampling-ready denotes CPU ticket handling after readiness detection, not GPU completion. '
         'Phase host-span coverage is not GPU utilization and phase spans may overlap.'
         ))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--events',
                        type=pathlib.Path,
                        action='append',
                        required=True)
    parser.add_argument('--output', type=pathlib.Path, required=True)
    parser.add_argument('--request-timeline', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = [analyze(path) for path in args.events]
    if args.request_timeline:
        for path, item in zip(args.events, result):
            item['ready_path'] = analyze_ready_path(path)
    with args.output.open('x') as output:
        json.dump(result, output, indent=2)


if __name__ == '__main__':
    main()
