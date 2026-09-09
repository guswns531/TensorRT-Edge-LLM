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

import copy
import importlib.util
import pathlib
import unittest

PATH = pathlib.Path(__file__).parents[
    2] / 'benchmarks/phase_serving/service_normalized_shadow.py'
SPEC = importlib.util.spec_from_file_location('service_shadow', PATH)
SHADOW = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SHADOW)


def snapshot():
    """Synthetic equal-work P/D alternatives; not measured speedup evidence."""
    requests = [
        dict(request_id=1,
             phase='decode',
             milestone='next_token',
             elapsed_us=100,
             reference_us=10,
             reference_source='isolated_execution',
             reference_cohort_signature='d'),
        dict(request_id=2,
             phase='prefill',
             milestone='first_token',
             elapsed_us=0,
             reference_us=20,
             reference_source='isolated_execution',
             reference_cohort_signature='p')
    ]
    candidates = []
    for identity, delays in [(1, (30, 20)), (2, (10, 30))]:
        candidates.append(
            dict(action_id=identity,
                 hard_feasible=True,
                 target_work_signature='p+d',
                 horizon_us=30,
                 services=[
                     dict(request_id=r['request_id'],
                          milestone=r['milestone'],
                          completion_us=t,
                          uncertainty_us=0) for r, t in zip(requests, delays)
                 ]))
    return dict(schema_version=1,
                clock_domain='host_monotonic',
                target_work_signature='p+d',
                requests=requests,
                candidates=candidates)


class ServiceNormalizedShadowTest(unittest.TestCase):

    def test_runtime_event_reconstructs_mechanism_only_alternatives(self):
        protected = [
            dict(kind='prefill',
                 request_id=2,
                 predicted_completion_us=20,
                 uncertainty_us=1,
                 reference_us=10,
                 reference_source='runtime_exact'),
            dict(kind='decode',
                 request_id=1,
                 predicted_completion_us=30,
                 uncertainty_us=2,
                 reference_us=5,
                 reference_source='static_profile')
        ]
        event = dict(host_monotonic_ns=100000,
                     selected_action_id=11,
                     ready_prefill_request_ids=[2],
                     ready_decode_request_ids=[1],
                     service_clocks=[
                         dict(request_id=1,
                              submitted_host_ns=1000,
                              last_token_committed_host_ns=80000),
                         dict(request_id=2,
                              submitted_host_ns=70000,
                              last_token_committed_host_ns=0)
                     ],
                     mechanism_candidates=[
                         dict(action_id=11,
                              legal=True,
                              predicted_completion_us=35,
                              protected_services=protected),
                         dict(action_id=12,
                              legal=True,
                              predicted_completion_us=40,
                              protected_services=[
                                  dict(row,
                                       predicted_completion_us=row[
                                           'predicted_completion_us'] + 5)
                                  for row in protected
                              ])
                     ])
        reconstructed = SHADOW.snapshot_from_event(event)
        self.assertEqual(len(reconstructed['candidates']), 2)
        self.assertEqual(
            {
                row['request_id']: row['elapsed_us']
                for row in reconstructed['requests']
            }, {
                1: 20,
                2: 30
            })
        self.assertFalse(
            SHADOW.evaluate(reconstructed)['production_authority'])

    def test_runtime_event_does_not_duplicate_aggregate_projection(self):
        event = dict(host_monotonic_ns=100000,
                     selected_action_id=11,
                     ready_decode_request_ids=[1],
                     service_clocks=[
                         dict(request_id=1,
                              submitted_host_ns=1000,
                              last_token_committed_host_ns=80000)
                     ],
                     mechanism_candidates=[
                         dict(action_id=11,
                              legal=True,
                              predicted_completion_us=20,
                              protected_services=[
                                  dict(kind='decode',
                                       request_id=0,
                                       predicted_completion_us=20,
                                       uncertainty_us=0,
                                       reference_us=5,
                                       reference_source='runtime_exact')
                              ])
                     ])
        with self.assertRaises(ValueError):
            SHADOW.snapshot_from_event(event)

    def test_runtime_event_rejects_cold_reference(self):
        event = dict(host_monotonic_ns=100000,
                     selected_action_id=11,
                     ready_prefill_request_ids=[2],
                     service_clocks=[
                         dict(request_id=2,
                              submitted_host_ns=70000,
                              last_token_committed_host_ns=0)
                     ],
                     mechanism_candidates=[
                         dict(action_id=11,
                              legal=True,
                              predicted_completion_us=20,
                              protected_services=[
                                  dict(kind='prefill',
                                       request_id=2,
                                       predicted_completion_us=20,
                                       uncertainty_us=0,
                                       reference_us=5,
                                       reference_source='cold_fallback')
                              ])
                     ])
        with self.assertRaises(ValueError):
            SHADOW.snapshot_from_event(event)

    def test_runtime_clock_uses_commit_not_queue_readiness(self):
        event = dict(host_monotonic_ns=100000,
                     ready_decode_request_ids=[1],
                     ready_prefill_request_ids=[2],
                     service_clocks=[
                         dict(request_id=1,
                              submitted_host_ns=1000,
                              last_token_committed_host_ns=40000),
                         dict(request_id=2,
                              submitted_host_ns=2000,
                              last_token_committed_host_ns=0)
                     ])
        rows = {row['request_id']: row for row in SHADOW.service_clocks(event)}
        self.assertEqual(rows[1]['elapsed_us'], 60)
        self.assertEqual(rows[2]['elapsed_us'], 98)
        event['ready_encoder_request_ids'] = event.pop(
            'ready_prefill_request_ids')
        rows = {row['request_id']: row for row in SHADOW.service_clocks(event)}
        self.assertEqual(rows[2]['elapsed_us'], 98)

    def test_runtime_clock_rejects_missing_or_future_commit(self):
        event = dict(host_monotonic_ns=100000,
                     ready_decode_request_ids=[1],
                     service_clocks=[
                         dict(request_id=1,
                              submitted_host_ns=1000,
                              last_token_committed_host_ns=0)
                     ])
        with self.assertRaises(ValueError):
            SHADOW.service_clocks(event)
        event['service_clocks'][0]['last_token_committed_host_ns'] = 100001
        with self.assertRaises(ValueError):
            SHADOW.service_clocks(event)
        self.assertIsNone(SHADOW.service_clocks({}))

    def test_old_decode_is_not_lost_in_increment(self):
        result = SHADOW.evaluate(snapshot())
        self.assertEqual(result['diagnostic_lexicographic_action_id'], 2)
        row = result['candidates'][0]['requests'][0]
        self.assertEqual(
            (row['current_age'], row['additional_age'], row['age_at_service']),
            (10, 3, 13))
        self.assertFalse(result['production_authority'])

    def test_old_prefill_can_reverse_preference(self):
        data = snapshot()
        data['requests'][0]['elapsed_us'] = 0
        data['requests'][1]['elapsed_us'] = 400
        self.assertEqual(
            SHADOW.evaluate(data)['diagnostic_lexicographic_action_id'], 1)

    def test_uniform_time_scaling_preserves_ranking(self):
        data = snapshot()
        for row in data['requests']:
            row['elapsed_us'] *= 17
            row['reference_us'] *= 17
        for action in data['candidates']:
            action['horizon_us'] *= 17
            for row in action['services']:
                row['completion_us'] *= 17
                row['uncertainty_us'] *= 17
        original, scaled = SHADOW.evaluate(snapshot()), SHADOW.evaluate(data)
        self.assertEqual(original['pareto_action_ids'],
                         scaled['pareto_action_ids'])
        self.assertEqual(original['diagnostic_lexicographic_action_id'],
                         scaled['diagnostic_lexicographic_action_id'])
        self.assertEqual(original['candidates'][0]['max_age_at_service'],
                         scaled['candidates'][0]['max_age_at_service'])

    def test_unselected_request_must_have_projection(self):
        data = snapshot()
        data['candidates'][0]['services'].pop()
        with self.assertRaises(ValueError):
            SHADOW.evaluate(data)

    def test_action_cannot_change_reference(self):
        data = snapshot()
        data['candidates'][0]['services'][0]['reference_us'] = 100
        with self.assertRaises(ValueError):
            SHADOW.evaluate(data)

    def test_bad_reference_and_mixed_clock_rejected(self):
        for key, value in [('reference_us', 0), ('reference_us', float('nan')),
                           ('reference_source', 'queued_execution')]:
            data = snapshot()
            data['requests'][0][key] = value
            with self.assertRaises(ValueError):
                SHADOW.evaluate(data)
        data = snapshot()
        data['clock_domain'] = 'mixed_host_gpu'
        with self.assertRaises(ValueError):
            SHADOW.evaluate(data)

    def test_duplicate_ids_rejected(self):
        data = snapshot()
        data['requests'].append(copy.deepcopy(data['requests'][0]))
        with self.assertRaises(ValueError):
            SHADOW.evaluate(data)

    def test_same_work_and_finite_horizon_required(self):
        for key, value in [('target_work_signature', 'p-only'),
                           ('horizon_us', 5), ('horizon_us', float('inf'))]:
            data = snapshot()
            data['candidates'][0][key] = value
            with self.assertRaises(ValueError):
                SHADOW.evaluate(data)

    def test_uncertainty_is_part_of_service_delay(self):
        data = snapshot()
        data['candidates'][1]['services'][0]['uncertainty_us'] = 2
        self.assertEqual(
            SHADOW.evaluate(data)['candidates'][1]['requests'][0]
            ['age_at_service'], 11.2)

    def test_first_token_milestone_survives_encoder_transition(self):
        data = snapshot()
        data['requests'][1]['phase'] = 'encoder'
        before = SHADOW.evaluate(data)
        data['requests'][1]['phase'] = 'prefill'
        after = SHADOW.evaluate(data)
        self.assertEqual(before['diagnostic_lexicographic_action_id'],
                         after['diagnostic_lexicographic_action_id'])
        data['candidates'][0]['services'][1]['milestone'] = 'encoder_complete'
        with self.assertRaises(ValueError):
            SHADOW.evaluate(data)

    def test_input_is_not_mutated(self):
        data = snapshot()
        before = copy.deepcopy(data)
        SHADOW.evaluate(data)
        self.assertEqual(data, before)


if __name__ == '__main__':
    unittest.main()
