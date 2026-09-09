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
"""Check metadata prerequisites for a same-state P versus P+D branch experiment."""

import argparse
import collections
import hashlib
import json
import pathlib


def metadata_errors(event):
    """Missing capture is not an empty ownership state or a replay checkpoint."""
    errors = []
    for key in ('kv_ownership_signature', 'vision_lease_signature',
                'strict_snapshot_signature', 'scalar_policy_state_signature'):
        if not event.get(key):
            errors.append('missing_' + key)
    for phase, lengths in [('prefill', 'token_counts'),
                           ('decode', 'context_lengths')]:
        ids = event.get('ready_' + phase + '_request_ids', [])
        values = event.get('ready_' + phase + '_' + lengths, [])
        if len(ids) != event['ready'][phase +
                                      '_rows'] or len(values) != len(ids):
            errors.append('incomplete_' + phase + '_rows')
    if not event.get('candidates'):
        errors.append('missing_candidate_snapshots')
    return errors


def inspect(path, prefill_batch, decode_batch):
    events, epochs = [], 0
    for line in path.read_text().splitlines():
        tag, separator, value = line.partition('\t')
        if not separator:
            continue
        event = json.loads(value)
        if tag == 'PHASE_EPOCH' and event['kind'] == 'measurement':
            events.clear()
            epochs += 1
        elif tag == 'PHASE_SCHEDULER_EVENT' and event[
                'event_kind'] == 'decision':
            events.append(event)
    if epochs != 1:
        raise ValueError('Expected exactly one measurement epoch')
    matches, counts = [], collections.Counter()
    for event in events:
        counts['decisions'] += 1
        for scope, name in [('final', 'selector_audit'),
                            ('pd', 'pd_selector_audit')]:
            audit = event.get(name)
            if audit is None:
                continue
            eligible = [
                row for row in audit['inputs'] if row['hard_feasible']
                and row['frontier_eligible'] and not row['dominated']
            ]
            singles = [
                row for row in eligible if row['action_kind'] == 'prefill'
                and row.get('primary_batch') == prefill_batch
            ]
            pairs = [
                row for row in eligible
                if row['action_kind'] == 'prefill_decode'
                and row.get('primary_batch') == prefill_batch
                and row.get('secondary_batch', 0) > 0 and (
                    decode_batch is None
                    or row.get('secondary_batch') == decode_batch)
            ]
            if not singles or not pairs:
                continue
            counts[scope + '_shape_opportunities'] += 1
            errors = metadata_errors(event)
            snapshots = {
                row['action_id']: row
                for row in event.get('candidates', [])
            }
            for single in singles:
                for pair in pairs:
                    row_errors = list(errors)
                    p = snapshots.get(single['action_id'])
                    pd = snapshots.get(pair['action_id'])
                    if p is None or pd is None:
                        row_errors.append('selected_frontier_snapshot_missing')
                    elif (len(p['request_ids']) != prefill_batch
                          or len(pd['request_ids'])
                          != prefill_batch + pair['secondary_batch']
                          or not set(p['request_ids']).issubset(
                              pd['request_ids'])):
                        row_errors.append('candidate_membership_mismatch')
                    fingerprint = None
                    if not row_errors:
                        payload = {
                            key: event[key]
                            for key in ('strict_snapshot_signature',
                                        'kv_ownership_signature',
                                        'vision_lease_signature',
                                        'scalar_policy_state_signature')
                        }
                        payload.update(scope=scope, prefill=p, pair=pd)
                        fingerprint = hashlib.sha256(
                            json.dumps(payload,
                                       sort_keys=True,
                                       separators=(',',
                                                   ':')).encode()).hexdigest()
                    matches.append(
                        dict(decision_id=event['decision_id'],
                             scope=scope,
                             actual_action=event['action_kind'],
                             errors=row_errors,
                             metadata_fingerprint=fingerprint,
                             single=single,
                             pair=pair,
                             runtime_replay_ready=False))
    return dict(
        source=str(path),
        counts=dict(counts),
        matches=matches,
        caveat=
        'Matching metadata is not a GPU checkpoint: KV/vision tensor contents, '
        'full posterior and phase-local bindings are not restored by this tool.'
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('events', nargs='+', type=pathlib.Path)
    parser.add_argument('--prefill-batch', type=int, default=2)
    parser.add_argument('--decode-batch', type=int, default=48)
    parser.add_argument(
        '--any-decode-batch',
        action='store_true',
        help=
        'Broader diagnostic only; not equivalent to the original D48 target')
    parser.add_argument('--output', required=True, type=pathlib.Path)
    args = parser.parse_args()
    if args.prefill_batch <= 0 or args.decode_batch <= 0:
        parser.error('Batch sizes must be positive')
    rows = [
        inspect(path, args.prefill_batch,
                None if args.any_decode_batch else args.decode_batch)
        for path in args.events
    ]
    groups = collections.defaultdict(set)
    for row in rows:
        for match in row['matches']:
            if match['metadata_fingerprint']:
                groups[match['metadata_fingerprint']].add(row['source'])
    repeated = {
        key: sorted(value)
        for key, value in groups.items() if len(value) > 1
    }
    with args.output.open('x') as output:
        json.dump(dict(runs=rows,
                       repeated_metadata_across_files=repeated,
                       physical_branch_comparisons=0),
                  output,
                  indent=2)


if __name__ == '__main__':
    main()
