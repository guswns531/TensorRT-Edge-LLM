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

import json

from benchmarks.phase_serving.analyze_completion_sample_efficiency import \
    summarize


def test_summarizes_direction_authority_by_budget(tmp_path) -> None:
    calibration = tmp_path / "calibration.json"
    calibration.write_text(json.dumps([{
        "completed_warmup_requests": 424,
        "contextual_policy_calibration": {
            "prefill_decode": {
                "directions": [{
                    "direction":
                    "prefill_to_decode",
                    "completion_posterior_observations":
                    12,
                    "completion_authority_window_observations":
                    8,
                    "completion_authority_evidence_ready":
                    True,
                    "completion_authority_validated":
                    True,
                    "completion_authority_incumbent_blend_weight":
                    0.5,
                    "completion_authority_newcomer_blend_weight":
                    0.25,
                    "completion_authority_incumbent_completion_absolute_error_us":
                    4.0,
                    "completion_authority_newcomer_completion_absolute_error_us":
                    2.0,
                    "completion_authority_incumbent_reference_absolute_error_us":
                    8.0,
                    "completion_authority_newcomer_reference_absolute_error_us":
                    4.0,
                    "completion_authority_false_safe":
                    0,
                }]
            }
        }
    }]),
                           encoding="utf-8")
    result = summarize([calibration])
    point = result["points"][0]
    assert point["budget"] == 424
    assert point["authority_ready_fraction"] == 1.0
    assert point["model_to_reference_absolute_error_median"] == 0.5
