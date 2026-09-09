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
"""Compare additional-lateness ranking on actual all-late selector frontiers."""

import argparse
import collections
import json
import math
import pathlib


def compare(audit):
    """Return one model-only counterfactual, never a measured alternative."""
    eligible = [
        row for row in audit['inputs'] if row['hard_feasible']
        and row['frontier_eligible'] and not row['dominated']
    ]
    if not eligible:
        raise ValueError('All-late selection has no eligible frontier')
    for row in eligible:
        for key in ('additional_violation_us', 'service_compression',
                    'selection_horizon_us'):
            if not math.isfinite(row[key]) or row[key] < 0:
                raise ValueError('Invalid shadow input: ' + key)
    selected = [
        row for row in eligible
        if row['action_id'] == audit['selected_action_id']
    ]
    if len(selected) != 1:
        raise ValueError(
            'Selected action missing or duplicated in eligible frontier')
    actual = selected[0]
    shadow = min(eligible,
                 key=lambda row: (row['additional_violation_us'], -row[
                     'service_compression'], row['action_id']))
    return dict(
        changed=actual['action_id'] != shadow['action_id'],
        actual=actual,
        shadow=shadow,
        predicted_added_delay_saved_us=actual['additional_violation_us'] -
        shadow['additional_violation_us'],
        compression_delta=shadow['service_compression'] -
        actual['service_compression'])


def analyze(path):
    events, epochs = [], 0
    for line in path.read_text().splitlines():
        prefix, separator, value = line.partition('\t')
        if not separator:
            continue
        event = json.loads(value)
        if prefix == 'PHASE_EPOCH' and event['kind'] == 'measurement':
            events.clear()
            epochs += 1
        elif prefix == 'PHASE_SCHEDULER_EVENT' and event[
                'event_kind'] == 'decision':
            events.append(event)
    if epochs != 1:
        raise ValueError('Expected one measurement epoch')
    rows, counts = [], collections.Counter()
    for event in events:
        for scope, key in [('final', 'selector_audit'),
                           ('pd', 'pd_selector_audit')]:
            audit = event.get(key)
            if audit is None or audit[
                    'reason'] != 'all_late_efficiency_recovery':
                continue
            result = compare(audit)
            counts[scope + '_all_late'] += 1
            counts[scope + '_changed'] += int(result['changed'])
            rows.append(
                dict(decision_id=event['decision_id'], scope=scope, **result))
    return dict(
        source=str(path),
        counts=dict(counts),
        rows=rows,
        caveat=
        'Predicted alternative on the existing pruned frontier; no alternative execution, '
        'future cohort simulation, or causal performance gain is measured.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('events', nargs='+', type=pathlib.Path)
    parser.add_argument('--output', required=True, type=pathlib.Path)
    args = parser.parse_args()
    result = [analyze(path) for path in args.events]
    with args.output.open('x') as output:
        json.dump(result, output, indent=2)


if __name__ == '__main__':
    main()
