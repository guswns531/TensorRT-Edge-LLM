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

import importlib.util
import json
from pathlib import Path


SCRIPT = (Path(__file__).parents[2] / 'benchmarks' / 'phase_serving' /
          'analyze_v3_service_scale.py')
SPEC = importlib.util.spec_from_file_location('analyze_v3_service_scale', SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_reports_service_scale_and_pseudo_expiration(tmp_path):
    event = {
        'event_kind': 'decision',
        'host_monotonic_ns': 2_000_000,
        'action_kind': 'prefill',
        'encoder_service': {
            'request_id': 3,
            'reference_us': 20_000.0,
            'reference_source': 'static_profile',
            'ready_wait_us': 10_000.0,
            'reference_valid': True,
            'has_explicit_slo': True,
            'absolute_slack_us': 400_000.0,
            'service_epoch': 1,
            'service_age_quanta': 0.5,
        },
        'ready': {
            'decode_rows': 4
        },
        'decode_guard_audit': {
            'candidate_suppressed': True,
            'candidate_restored': False,
        },
        'mechanism_candidates': [{
            'protected_services': [{
                'kind': 'prefill',
                'request_id': 1,
                'slack_us': -100.0,
                'reference_us': 10_000.0,
                'reference_source': 'runtime_exact',
                'elapsed_service_us': 12_000.0,
                'has_explicit_slo': False,
                'absolute_slack_us': None,
                'service_epoch': 0,
            }, {
                'kind': 'decode',
                'request_id': 2,
                'slack_us': 70_000.0,
                'reference_us': 8_000.0,
                'reference_source': 'static_profile',
                'elapsed_service_us': 4_000.0,
                'has_explicit_slo': True,
                'absolute_slack_us': 70_000.0,
                'service_epoch': 0,
            }]
        }]
    }
    path = tmp_path / 'events.log'
    lines = [
        'PHASE_EPOCH\t' + json.dumps({'kind': 'measurement'}),
        'PHASE_SCHEDULER_EVENT\t' + json.dumps(event),
    ]
    path.write_text('\n'.join(lines) + '\n')

    result = MODULE.analyze(path)

    assert result['counts']['prefill_pseudo_expired'] == 1
    assert result['counts']['decode_candidate_suppressed'] == 1
    assert result['counts']['prefill_service_age_over_one'] == 1
    assert result['service_age_quanta']['encoder']['p50'] == 0.5
    assert result['fixed_scale_to_service_ratio']['prefill']['p50'] == 0.5
    assert result['fixed_scale_to_service_ratio']['decode']['p50'] == 0.25
    assert result['p_only_streak']['maximum'] == 1
