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
"""Decompose retained same-engine runs without treating previews as selector inputs."""

import argparse
import collections
import csv
import json
import pathlib
import statistics


def analyze(folder):
    """Return phase intervals, inter-decode gap coverage and measured decisions."""
    intervals = list(
        csv.DictReader((folder / 'analysis/measured-intervals.csv').open()))
    segments = list(
        csv.DictReader((folder / 'analysis/measured-segments.csv').open()))
    for row in intervals + segments:
        for key in ['start_ms', 'end_ms']:
            row[key] = float(row[key])
    start = min(row['start_ms'] for row in intervals)
    # Segment CSV timestamps are already relative to the first measured interval.
    for row in intervals:
        row['start_ms'] -= start
        row['end_ms'] -= start
    decodes = [row for row in intervals if row['name'] == 'decode_dispatch']
    gaps = []
    for previous, following in zip(decodes, decodes[1:]):
        left, right = previous['end_ms'], following['start_ms']
        coverage = collections.Counter()
        for segment in segments:
            duration = max(
                0,
                min(right, segment['end_ms']) - max(left, segment['start_ms']))
            coverage[segment['binary_mask']] += duration
        if abs(sum(coverage.values()) - (right - left)) > 1e-5:
            raise ValueError('Gap coverage and interval clocks disagree')
        gaps.append(
            dict(start_ms=left,
                 end_ms=right,
                 duration_ms=right - left,
                 previous=previous['correlation_id'],
                 following=following['correlation_id'],
                 mask_ms=dict(coverage)))
    measured = False
    metrics, decisions = [], []
    for line in (folder / 'run-001/activity-events.jsonl').open():
        tag, _, raw = line.partition('\t')
        if tag == 'PHASE_EPOCH':
            measured = True
            continue
        if not measured:
            continue
        event = json.loads(raw)
        if tag == 'PHASE_METRIC':
            metrics.append(event)
        elif tag == 'PHASE_SCHEDULER_EVENT' and event[
                'event_kind'] == 'decision':
            decisions.append(event)
    preview_counts = collections.Counter()
    reasons = collections.Counter()
    for event in decisions:
        for candidate in event['candidates']:
            if candidate['action_kind'] != 'prefill_decode':
                continue
            preview_counts['present'] += 1
            preview_counts['scalar_authority'] += int(
                candidate['contextual_scalar_authority_applied'])
            preview_counts['cost_known'] += int(
                candidate['scalar_decision_cost_known'])
            preview_counts['selected_' + event['action_kind']] += 1
            audit = event.get('selector_audit')
            if audit:
                reasons[audit['reason']] += 1
    phases = {}
    for phase, field in [('prefill', 'prefill_batch'),
                         ('decode', 'decode_batch')]:
        batches = [row for row in metrics if row[field] > 0]
        phases[phase] = dict(count=len(batches),
                             rows=sum(row[field] for row in batches),
                             mean_batch=statistics.mean(row[field]
                                                        for row in batches),
                             mean_gpu_ms=statistics.mean(row[phase + '_gpu_ms']
                                                         for row in batches))
    counter_fields = [
        'contextual_pd_predictions', 'contextual_pd_ready',
        'contextual_pd_observations', 'contextual_pd_decision_disagreements',
        'contextual_pd_calibration_observations',
        'contextual_pd_positive_selections',
        'contextual_pd_negative_selections', 'vision_direct_output_batches',
        'vision_direct_output_bytes'
    ]
    counters = {
        key:
        dict(first=metrics[0][key],
             last=metrics[-1][key],
             delta=metrics[-1][key] - metrics[0][key])
        for key in counter_fields
    }
    total_gap_masks = collections.Counter()
    for gap in gaps:
        total_gap_masks.update(gap['mask_ms'])
    requests = list(
        csv.DictReader(
            (folder / 'run-001/client/run-001/requests.csv').open()))
    return dict(preview_counts=dict(preview_counts),
                actual_audit_reasons=dict(reasons),
                phases=phases,
                counters=counters,
                total_gap_mask_ms=dict(total_gap_masks),
                longest_gaps=sorted(gaps,
                                    key=lambda row: row['duration_ms'],
                                    reverse=True)[:5],
                first_decode_ms=decodes[0]['start_ms'],
                last_decode_ms=decodes[-1]['end_ms'],
                requests=requests)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=pathlib.Path, required=True)
    args = parser.parse_args()
    result = {
        version: analyze(args.root / version)
        for version in ['v0100', 'v0101']
    }
    before = {
        row['request_id']: row
        for row in result['v0100'].pop('requests')
    }
    after = {row['request_id']: row for row in result['v0101'].pop('requests')}
    if before.keys() != after.keys():
        raise ValueError('Request identities differ')
    paired = {}
    for request_class in sorted(
        {row['request_class']
         for row in before.values()}):
        keys = [
            key for key in before
            if before[key]['request_class'] == request_class
        ]
        paired[request_class] = {}
        for metric in ['ttft_ms', 'tpot_ms', 'e2e_ms']:
            changes = [
                float(after[key][metric]) - float(before[key][metric])
                for key in keys
            ]
            paired[request_class][metric] = dict(
                mean_delta=statistics.mean(changes),
                improved=sum(value < 0 for value in changes),
                requests=len(keys))
    result['paired_requests'] = paired
    (args.root / 'detail.json').write_text(json.dumps(result, indent=2) + '\n')
    for value in result.values():
        if 'longest_gaps' in value:
            value['longest_gaps'] = value['longest_gaps'][:2]
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
