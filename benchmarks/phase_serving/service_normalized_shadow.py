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
"""Offline service-normalized diagnostics; no production selection authority."""

import argparse
import collections
import hashlib
import json
import math
import pathlib


def number(value, name, positive=False):
    """Reject missing/nonfinite measurements instead of inventing fallback cost."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError('Missing numeric field: ' + name)
    if not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError('Invalid numeric field: ' + name)
    return float(value)


def keyed(rows, key):
    """Require unique identities across the fixed protected request frontier."""
    result = {}
    for row in rows:
        identity = row[key]
        if identity in result:
            raise ValueError('Duplicate ' + key)
        result[identity] = row
    return result


def evaluate(snapshot):
    """Score first-service milestones over one common work target, not reward.

    References belong to the snapshot, never to individual actions. Candidates
    must project service for every protected request, including unselected rows.
    This diagnostic does not model recurrent service after the first milestone.
    """
    if snapshot.get('schema_version') != 1:
        raise ValueError('Unsupported schema')
    if snapshot.get('clock_domain') != 'host_monotonic':
        raise ValueError('Use a single explicit host service-clock domain')
    if not snapshot.get('target_work_signature'):
        raise ValueError('Missing common work target')
    requests = keyed(snapshot['requests'], 'request_id')
    if not requests:
        raise ValueError('Empty protected frontier')
    for request in requests.values():
        if request['phase'] not in ('encoder', 'prefill', 'decode'):
            raise ValueError('Unknown phase')
        expected = 'next_token' if request[
            'phase'] == 'decode' else 'first_token'
        if request['milestone'] != expected:
            raise ValueError(
                'Service milestone must survive E to P transition')
        if request['reference_source'] != 'isolated_execution':
            raise ValueError('Reference must exclude queueing and overlap')
        if not request.get('reference_cohort_signature'):
            raise ValueError('Missing canonical reference cohort identity')
        number(request['elapsed_us'], 'elapsed_us')
        number(request['reference_us'], 'reference_us', positive=True)
    candidates = keyed(snapshot['candidates'], 'action_id')
    summaries = []
    for identity, action in candidates.items():
        if not isinstance(action['hard_feasible'], bool):
            raise ValueError('Feasibility must be explicit')
        if not action['hard_feasible']:
            continue
        if action['target_work_signature'] != snapshot[
                'target_work_signature']:
            raise ValueError('Different work targets are not comparable')
        if 'reference_us' in action:
            raise ValueError('Action-dependent normalization is forbidden')
        horizon = number(action['horizon_us'], 'horizon_us', positive=True)
        projections = keyed(action['services'], 'request_id')
        if projections.keys() != requests.keys():
            raise ValueError('Missing or extra protected request projection')
        values = []
        for request_id, request in requests.items():
            projection = projections[request_id]
            if projection['milestone'] != request['milestone']:
                raise ValueError('Mismatched service milestone')
            if 'reference_us' in projection:
                raise ValueError('Action-dependent normalization is forbidden')
            delay = number(projection['completion_us'], 'completion_us')
            uncertainty = number(projection['uncertainty_us'],
                                 'uncertainty_us')
            if delay + uncertainty > horizon:
                raise ValueError('Service outside evaluated horizon')
            now = request['elapsed_us'] / request['reference_us']
            increment = (delay + uncertainty) / request['reference_us']
            values.append(
                dict(request_id=request_id,
                     phase=request['phase'],
                     current_age=now,
                     additional_age=increment,
                     age_at_service=now + increment))
        summary = dict(
            action_id=identity,
            horizon_us=horizon,
            max_age_at_service=max(row['age_at_service'] for row in values),
            mean_age_at_service=sum(row['age_at_service']
                                    for row in values) / len(values),
            max_additional_age=max(row['additional_age'] for row in values),
            requests=values)
        if not all(
                math.isfinite(summary[key])
                for key in ('max_age_at_service', 'mean_age_at_service',
                            'max_additional_age')):
            raise ValueError('Normalized score overflow; do not clamp urgency')
        summaries.append(summary)
    if not summaries:
        raise ValueError('No feasible complete candidate')
    # Pareto diagnostics avoid silently promoting a max/sum weighting policy.
    dimensions = ('max_age_at_service', 'mean_age_at_service', 'horizon_us')
    frontier = []
    for right in summaries:
        dominated = any(
            all(left[key] <= right[key]
                for key in dimensions) and any(left[key] < right[key]
                                               for key in dimensions)
            for left in summaries)
        if not dominated:
            frontier.append(right['action_id'])
    return dict(
        candidates=summaries,
        pareto_action_ids=sorted(frontier),
        diagnostic_lexicographic_action_id=min(
            summaries,
            key=lambda row:
            (row['max_age_at_service'], row['mean_age_at_service'], row[
                'horizon_us'], row['action_id']))['action_id'],
        production_authority=False,
        limitation=
        'First-service milestones only; not recurrent-token or full trajectory utility.'
    )


def service_clocks(event):
    """Read host commit clocks without treating queue readiness as service."""
    if 'service_clocks' not in event:
        return None
    now = event['host_monotonic_ns']
    clocks = keyed(event['service_clocks'], 'request_id')
    rows = []
    for phase in ('encoder', 'prefill', 'decode'):
        for identity in event.get('ready_' + phase + '_request_ids', []):
            clock = clocks.get(identity)
            if clock is None:
                raise ValueError('Ready request missing service clock')
            submitted = clock['submitted_host_ns']
            committed = clock['last_token_committed_host_ns']
            if not 0 < submitted <= now or committed < 0 or committed > now:
                raise ValueError('Invalid host service-clock ordering')
            if committed and committed < submitted:
                raise ValueError('Token predates request submission')
            if phase == 'decode' and committed == 0:
                raise ValueError('Ready decode lacks token commit clock')
            origin = committed if phase == 'decode' else submitted
            rows.append(
                dict(request_id=identity,
                     phase=phase,
                     elapsed_us=(now - origin) / 1000.0))
    return rows


def snapshot_from_event(event):
    """Build a request-local shadow snapshot without inventing outcomes.

    The selected mechanism candidate defines the protected request frontier.
    Alternatives missing any of those request-local projections are omitted;
    unsupported residual paths therefore remain missing rather than inheriting
    an aggregate oldest-request prediction.
    """
    clocks = service_clocks(event)
    if clocks is None:
        raise ValueError('Missing runtime service clocks')
    ready_clock_rows = keyed(clocks, 'request_id')
    raw_clocks = keyed(event['service_clocks'], 'request_id')
    candidates = event.get('mechanism_candidates')
    if not candidates:
        raise ValueError('Missing mechanism-only frontier')
    by_id = keyed(candidates, 'action_id')
    selected = by_id.get(event.get('selected_action_id'))
    if selected is None:
        raise ValueError('Selected action missing from mechanism frontier')

    valid_sources = {
        'runtime_exact', 'runtime_interpolated', 'runtime_covering',
        'static_profile', 'derived_isolated'
    }

    def services(action):
        result = {}
        for row in action.get('protected_services', []):
            identity = row.get('request_id', 0)
            source = row.get('reference_source', 'unknown')
            reference = row.get('reference_us', 0)
            if not identity or source not in valid_sources or not isinstance(
                    reference, (int, float)) or reference <= 0:
                continue
            if identity in result:
                raise ValueError('Duplicate request-local service projection')
            result[identity] = row
        return result

    selected_services = services(selected)
    if not selected_services:
        raise ValueError('Selected action lacks canonical request references')
    request_rows = []
    for identity, projection in sorted(selected_services.items()):
        raw_clock = raw_clocks.get(identity)
        if raw_clock is None:
            raise ValueError('Protected request missing runtime service clock')
        phase = projection['kind']
        if phase not in ('encoder', 'prefill', 'decode'):
            raise ValueError('Unsupported protected phase')
        if identity in ready_clock_rows:
            elapsed_us = ready_clock_rows[identity]['elapsed_us']
        else:
            now = event['host_monotonic_ns']
            submitted = raw_clock['submitted_host_ns']
            committed = raw_clock['last_token_committed_host_ns']
            if not 0 < submitted <= now or committed < 0 or committed > now:
                raise ValueError('Invalid protected service-clock ordering')
            if phase == 'decode' and committed == 0:
                raise ValueError('Protected decode lacks token commit clock')
            origin = committed if phase == 'decode' else submitted
            elapsed_us = (now - origin) / 1000.0
        request_rows.append(
            dict(
                request_id=identity,
                phase=phase,
                milestone='next_token' if phase == 'decode' else 'first_token',
                elapsed_us=elapsed_us,
                reference_us=projection['reference_us'],
                reference_source='isolated_execution',
                reference_provenance=projection['reference_source'],
                reference_cohort_signature='{}:{:.9g}:{}'.format(
                    phase, projection['reference_us'],
                    projection['reference_source'])))
    signature_payload = [(row['request_id'], row['milestone'],
                          row['reference_cohort_signature'])
                         for row in request_rows]
    target = hashlib.sha256(
        json.dumps(signature_payload,
                   separators=(',', ':')).encode()).hexdigest()[:24]

    action_rows = []
    for action in candidates:
        projected = services(action)
        if projected.keys() != selected_services.keys():
            continue
        consistent = all(
            math.isclose(projected[identity]['reference_us'],
                         selected_services[identity]['reference_us'],
                         rel_tol=1e-9,
                         abs_tol=1e-6)
            and projected[identity]['reference_source'] ==
            selected_services[identity]['reference_source']
            for identity in projected)
        if not consistent:
            raise ValueError('Action-dependent canonical reference')
        service_rows = []
        horizon = number(action['predicted_completion_us'],
                         'predicted_completion_us')
        for identity, projection in sorted(projected.items()):
            completion = number(projection['predicted_completion_us'],
                                'predicted_completion_us')
            uncertainty = number(projection['uncertainty_us'],
                                 'uncertainty_us')
            horizon = max(horizon, completion + uncertainty)
            phase = projection['kind']
            service_rows.append(
                dict(request_id=identity,
                     milestone='next_token'
                     if phase == 'decode' else 'first_token',
                     completion_us=completion,
                     uncertainty_us=uncertainty))
        action_rows.append(
            dict(action_id=action['action_id'],
                 hard_feasible=action['legal'],
                 target_work_signature=target,
                 horizon_us=max(horizon, 1e-9),
                 services=service_rows))
    if not action_rows:
        raise ValueError('No complete mechanism candidate')
    return dict(schema_version=1,
                clock_domain='host_monotonic',
                target_work_signature=target,
                requests=request_rows,
                candidates=action_rows)


def audit_log(path):
    """Check retained runtime coverage without reconstructing missing clocks."""
    counts = collections.Counter()
    rows = []
    epochs = 0
    active = False
    for line in path.open():
        marker, separator, raw = line.partition('\t')
        if not separator:
            continue
        event = json.loads(raw)
        if marker == 'PHASE_EPOCH' and event['kind'] == 'measurement':
            epochs += 1
            active = True
            counts.clear()
            rows.clear()
        elif active and marker == 'PHASE_SCHEDULER_EVENT' and event[
                'event_kind'] == 'decision':
            counts['decisions'] += 1
            guard = event.get('decode_guard_audit') or {}
            counts['explicit_guard_audits'] += bool(
                event.get('decode_guard_audit'))
            counts['explicit_d_suppressed'] += bool(
                guard.get('candidate_suppressed'))
            counts['both_expired_d_suppressed'] += bool(
                guard.get('candidate_suppressed')
                and guard.get('decode_expired'))
            clocks = service_clocks(event)
            if clocks is not None:
                counts['clock_snapshots'] += 1
                counts['validated_ready_clock_rows'] += len(clocks)
                counts['validated_ready_decode_clocks'] += sum(
                    row['phase'] == 'decode' for row in clocks)
            formation = event.get('scalar_formation') or {}
            counts['v2_formation_evaluated'] += bool(
                formation.get('evaluated'))
            counts['v2_formation_valid'] += bool(formation.get('valid'))
            snapshot = event.get('service_normalized_snapshot')
            if snapshot is None:
                try:
                    snapshot = snapshot_from_event(event)
                    counts['reconstructed_service_snapshot'] += 1
                except ValueError as error:
                    counts['missing_service_snapshot'] += 1
                    counts['reconstruction_' +
                           str(error).replace(' ', '_')] += 1
                    continue
            result = evaluate(snapshot)
            rows.append(dict(decision_id=event['decision_id'], result=result))
            counts['scored'] += 1
    if epochs != 1:
        raise ValueError('Expected one measurement epoch')
    counts.setdefault('scored', 0)
    if not counts['explicit_guard_audits']:
        counts['explicit_d_suppressed'] = None
        counts['both_expired_d_suppressed'] = None
    return dict(
        source=str(path),
        counts=dict(counts),
        rows=rows,
        limitation=
        'Mechanism frontier is reconstructed when request-local references exist; no alternative branch measurement or policy change.'
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--snapshot', type=pathlib.Path)
    source.add_argument('--events', nargs='+', type=pathlib.Path)
    parser.add_argument('--output', type=pathlib.Path, required=True)
    args = parser.parse_args()
    result = evaluate(json.loads(
        args.snapshot.read_text())) if args.snapshot else [
            audit_log(path) for path in args.events
        ]
    with args.output.open('x') as destination:
        json.dump(result, destination, indent=2)


if __name__ == '__main__':
    main()
