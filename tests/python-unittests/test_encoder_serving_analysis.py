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
import pathlib

import pytest

SPEC = importlib.util.spec_from_file_location(
    'encoder_analysis',
    pathlib.Path(__file__).parents[2] /
    'benchmarks/phase_serving/analyze_encoder_serving_pair.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_coverage_clips_and_merges():
    assert MODULE.covered_duration(0, 10, [(-2, 2), (1, 5), (8, 20)]) == 7
    assert MODULE.covered_duration(0, 10, [(10, 20)]) == 0
    assert MODULE.covered_duration(0, 10, []) == 0


def test_distribution_preserves_invalid_negative_gap():
    result = MODULE.distribution([-1, 1, 3])
    assert result['negative_count'] == 1
    assert result['median'] == 1
    assert result['p95'] == pytest.approx(2.8)


def test_requires_measurement_epoch(tmp_path):
    source = tmp_path / 'events.jsonl'
    source.write_text('')
    with pytest.raises(ValueError, match='measurement epoch'):
        MODULE.analyze(source)


def test_producer_ticket_joins_next_decode_and_keeps_terminal_sample(tmp_path):
    source = tmp_path / 'ready.jsonl'
    records = [('PHASE_EPOCH', {'kind': 'measurement'})]
    stages = [('prefill_sampling_submit', 0, 10),
              ('prefill_sampling_ready', 10, 10),
              ('prefill_sampling_collected', 12, 10),
              ('prefill_token_committed', 15, 10), ('decode_ready', 20, 10),
              ('decode_start', 1020, 42), ('decode_sampling_submit', 2000, 11),
              ('decode_sampling_ready', 2010, 11),
              ('decode_sampling_collected', 2012, 11),
              ('decode_token_committed', 2015, 11)]
    for stage, timestamp, index in stages:
        records.append(('PHASE_TIMELINE',
                        dict(stage=stage,
                             timestamp_us=timestamp,
                             dispatch_index=index,
                             request_index=1,
                             batch_size=4)))
    source.write_text(''.join(kind + '\t' + json.dumps(value) + '\n'
                              for kind, value in records))
    result = MODULE.analyze_ready_path(source)
    assert len(result['rows']) == 2
    assert result['rows'][0]['ready_to_decode_start_ms'] == 1
    assert result['rows'][0]['next_decode_dispatch'] == 42
    assert 'ready_to_decode_start_ms' not in result['rows'][1]
    source.write_text(source.read_text().replace('"timestamp_us": 20',
                                                 '"timestamp_us": 5', 1))
    with pytest.raises(ValueError, match='Ready precedes'):
        MODULE.analyze_ready_path(source)
