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
"""Characterize fixed queue scales against immutable phase-service costs."""

import argparse
import collections
import json
import math
import pathlib
import statistics


FIXED_SCALE_US = {
    'prefill': 5000.0,
    'decode': 2000.0,
}


def _percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def _summary(values):
    if not values:
        return dict(count=0, mean=None, p50=None, p95=None, maximum=None)
    return dict(count=len(values),
                mean=statistics.fmean(values),
                p50=statistics.median(values),
                p95=_percentile(values, 0.95),
                maximum=max(values))


def _protected_services(event):
    unique = {}
    for kind in FIXED_SCALE_US:
        state = event.get(f'{kind}_service', {})
        if state.get('reference_valid'):
            service = {
                'kind': kind,
                'request_id': state.get('request_id'),
                'reference_us': state.get('reference_us'),
                'reference_source': state.get('reference_source'),
                'elapsed_service_us': state.get('ready_wait_us'),
                'has_explicit_slo': state.get('has_explicit_slo'),
                'absolute_slack_us': state.get('absolute_slack_us'),
                'service_epoch': state.get('service_epoch'),
                'service_age_quanta': state.get('service_age_quanta'),
            }
            unique[(kind, service['request_id'], service['service_epoch'])] = service
    for candidate in event.get('mechanism_candidates', []):
        services = candidate.get('protected_services', [])
        for service in services:
            key = (service.get('kind'), service.get('request_id'),
                   service.get('service_epoch'))
            unique.setdefault(key, service)
    return unique.values()


def analyze(path):
    """Summarize one explicit measurement epoch from a scheduler event log."""
    decisions = []
    measurement_epochs = 0
    for line in path.read_text().splitlines():
        prefix, separator, value = line.partition('\t')
        if not separator:
            continue
        if prefix == 'PHASE_EPOCH' and json.loads(value)['kind'] == 'measurement':
            decisions.clear()
            measurement_epochs += 1
        elif prefix == 'PHASE_SCHEDULER_EVENT':
            event = json.loads(value)
            if event.get('event_kind') == 'decision':
                decisions.append(event)
    if measurement_epochs != 1:
        raise ValueError(f'Expected one measurement epoch in {path}')

    counts = collections.Counter()
    source_counts = collections.Counter()
    service_ages = collections.defaultdict(list)
    fixed_to_service = collections.defaultdict(list)
    non_decode_durations_ms = []
    p_only_streaks = []
    current_p_streak = 0
    non_decode_start_ns = None
    previous_ns = None
    for event in decisions:
        counts['decisions'] += 1
        host_ns = event.get('host_monotonic_ns')
        if previous_ns is not None and host_ns is not None and host_ns < previous_ns:
            raise ValueError('Decision host timestamps are not monotonic')
        previous_ns = host_ns
        ready_decode = event.get('ready', {}).get('decode_rows', 0) > 0
        action = event.get('action_kind')
        serves_decode = action in ('decode', 'prefill_decode', 'encoder_decode')
        if ready_decode and not serves_decode:
            counts['decode_ready_non_decode'] += 1
            if non_decode_start_ns is None:
                non_decode_start_ns = host_ns
            if action == 'prefill':
                current_p_streak += 1
            elif current_p_streak:
                p_only_streaks.append(current_p_streak)
                current_p_streak = 0
        else:
            if non_decode_start_ns is not None and host_ns is not None:
                non_decode_durations_ms.append((host_ns - non_decode_start_ns) / 1.0e6)
                non_decode_start_ns = None
            if current_p_streak:
                p_only_streaks.append(current_p_streak)
                current_p_streak = 0

        guard = event.get('decode_guard_audit')
        if guard is not None:
            counts['decode_guard_audited'] += 1
            counts['decode_candidate_suppressed'] += int(guard['candidate_suppressed'])
            counts['decode_candidate_restored'] += int(guard['candidate_restored'])

        for service in _protected_services(event):
            kind = service.get('kind')
            reference = service.get('reference_us')
            elapsed = service.get('elapsed_service_us')
            source = service.get('reference_source', 'unknown')
            if kind not in FIXED_SCALE_US or reference is None or elapsed is None or reference <= 0:
                continue
            source_counts[f'{kind}:{source}'] += 1
            fixed_to_service[kind].append(FIXED_SCALE_US[kind] / reference)
            service_ages[kind].append(elapsed / reference)
            explicit = bool(service.get('has_explicit_slo', False))
            counts[f'{kind}_explicit_slo'] += int(explicit)
            counts[f'{kind}_no_explicit_slo'] += int(not explicit)
            counts[f'{kind}_service_age_over_one'] += int(elapsed / reference >= 1.0)
            legacy_slack = service.get('slack_us')
            if kind == 'prefill' and not explicit and legacy_slack is not None and legacy_slack <= 0:
                counts['prefill_pseudo_expired'] += 1

    if current_p_streak:
        p_only_streaks.append(current_p_streak)
    if non_decode_start_ns is not None and previous_ns is not None:
        non_decode_durations_ms.append((previous_ns - non_decode_start_ns) / 1.0e6)

    return dict(source=str(path),
                counts=dict(sorted(counts.items())),
                reference_sources=dict(sorted(source_counts.items())),
                service_age_quanta={
                    kind: _summary(values)
                    for kind, values in sorted(service_ages.items())
                },
                fixed_scale_to_service_ratio={
                    kind: _summary(values)
                    for kind, values in sorted(fixed_to_service.items())
                },
                p_only_streak=_summary(p_only_streaks),
                decode_ready_non_decode_duration_ms=_summary(
                    non_decode_durations_ms))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('events', type=pathlib.Path, nargs='+')
    parser.add_argument('--output', type=pathlib.Path, required=True)
    args = parser.parse_args()
    result = [analyze(path) for path in args.events]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
