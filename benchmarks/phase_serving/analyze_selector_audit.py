#!/usr/bin/env python3
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
"""Attribute D-ready decisions using actual selector inputs, never preview unions."""

import argparse
import collections
import json
import pathlib


def analyze(path):
    """Return decision counts and non-D choices in one measurement epoch."""
    decisions = []
    epochs = 0
    for line in path.read_text().splitlines():
        prefix, separator, value = line.partition('\t')
        if not separator:
            continue
        if prefix == 'PHASE_EPOCH' and json.loads(
                value)['kind'] == 'measurement':
            decisions.clear()
            epochs += 1
        elif prefix == 'PHASE_SCHEDULER_EVENT':
            event = json.loads(value)
            if event['event_kind'] == 'decision':
                decisions.append(event)
    if epochs != 1:
        raise ValueError('Expected one explicit measurement epoch')
    counts = collections.Counter()
    rows = []
    for event in decisions:
        counts['decisions'] += 1
        audit = event.get('selector_audit')
        if audit is None:
            counts['unavailable'] += 1
        else:
            counts['audited'] += 1
        if event['ready']['decode_rows'] == 0:
            continue
        counts['decode_ready'] += 1
        if event['action_kind'] in ('decode', 'prefill_decode',
                                    'encoder_decode'):
            counts['decode_served'] += 1
            continue
        counts['decode_ready_non_decode_action'] += 1
        if audit is None:
            counts['non_decode_audit_unavailable'] += 1
            continue
        inputs = audit['inputs']
        selected = [
            row for row in inputs
            if row['action_id'] == audit['selected_action_id']
        ]
        if len(selected) != 1:
            raise ValueError(
                'Selected action must occur exactly once in actual inputs')
        chosen = selected[0]
        decode = [row for row in inputs if row['action_kind'] == 'decode']
        safe_decode = any(
            row['hard_feasible'] and row['max_slo_violation_us'] == 0
            for row in decode)
        classification = (
            'absent' if not decode else 'hard_infeasible' if not any(
                row['hard_feasible']
                for row in decode) else 'safe' if safe_decode else 'slo_late')
        counts['decode_candidate_' + classification] += 1
        counts['reason_' + audit['reason']] += 1
        pd_audit = event.get('pd_selector_audit')
        if pd_audit is not None:
            counts['pd_reason_' + pd_audit['reason']] += 1
            pd_decode = [
                row for row in pd_audit['inputs']
                if row['action_kind'] == 'decode'
            ]
            counts['pd_safe_decode'] += int(
                any(row['hard_feasible'] and row['max_slo_violation_us'] == 0
                    for row in pd_decode))
            counts['pd_decode_absent'] += int(not pd_decode)
        counts['post_select_overrides'] += int(audit['post_select_override'])
        counts['late_choice_with_safe_decode'] += int(
            safe_decode and chosen['max_slo_violation_us'] > 0
            and not audit['post_select_override'])
        rows.append(
            dict(decision_id=event['decision_id'],
                 action_kind=event['action_kind'],
                 decode_ready=event['ready']['decode_rows'],
                 decode_candidate=classification,
                 selector=audit,
                 pd_selector=pd_audit))
    return dict(source=str(path), counts=dict(counts), non_decode_rows=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('events', type=pathlib.Path, nargs='+')
    parser.add_argument('--output', type=pathlib.Path, required=True)
    args = parser.parse_args()
    args.output.write_text(
        json.dumps([analyze(path) for path in args.events], indent=2) + '\n')


if __name__ == '__main__':
    main()
