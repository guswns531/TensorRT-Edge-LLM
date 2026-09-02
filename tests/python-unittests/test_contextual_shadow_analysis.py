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
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks" / "phase_serving"))

from analyze_contextual_shadow import analyze  # noqa: E402


def test_analyze_contextual_shadow(tmp_path: Path) -> None:
    run = tmp_path / "balanced" / "worker-4" / "run-001"
    run.mkdir(parents=True)
    records = []
    for index, decision_us in enumerate((10.0, 20.0), start=1):
        record = {
            "host_scheduler_decision_us": decision_us,
            "global_action_fidelity_violations": 0,
        }
        for family in ("pd", "ep", "ed"):
            record.update({
                f"contextual_{family}_calibration_observations":
                index,
                f"contextual_{family}_ready_calibration_observations":
                index,
                f"contextual_{family}_interval_covered":
                index,
                f"contextual_{family}_ready_interval_covered":
                index,
                f"contextual_{family}_predicted_safe":
                index,
                f"contextual_{family}_false_safe":
                0,
                f"contextual_{family}_absolute_error_sum":
                0.1 * index,
                f"contextual_{family}_squared_error_sum":
                0.01 * index,
                f"contextual_{family}_ready_absolute_error_sum":
                0.1 * index,
                f"contextual_{family}_ready_squared_error_sum":
                0.01 * index,
            })
        record["contextual_completion"] = {
            direction: {
                "predictions": index,
                "observations": index,
                "ready_calibration_observations": index,
                "incumbent_interval_covered": index,
                "newcomer_interval_covered": index,
                "predicted_safe": index,
                "false_safe": 0,
                "incumbent_absolute_error_us": 10.0 * index,
                "incumbent_squared_error_us": 100.0 * index,
                "newcomer_absolute_error_us": 20.0 * index,
                "newcomer_squared_error_us": 400.0 * index,
            }
            for direction in ("prefill_to_decode", "decode_to_prefill",
                              "encoder_to_prefill", "prefill_to_encoder",
                              "encoder_to_decode", "decode_to_encoder")
        }
        record["contextual_completion_pair"] = {
            pair: {
                "predictions": index,
                "observations": index,
                "ready_calibration_observations": index,
                "incumbent_interval_covered": index,
                "newcomer_interval_covered": index,
                "predicted_safe": index,
                "false_safe": 0,
                "incumbent_absolute_error_us": 5.0 * index,
                "incumbent_squared_error_us": 25.0 * index,
                "newcomer_absolute_error_us": 6.0 * index,
                "newcomer_squared_error_us": 36.0 * index,
            }
            for pair in ("prefill_decode", "encoder_prefill", "encoder_decode")
        }
        record["contextual_completion_conformal"] = {
            pair: {
                "scale": 1.0 + 0.25 * index,
                "observations": 8 * index,
                "ready": index >= 2,
            }
            for pair in ("prefill_decode", "encoder_prefill", "encoder_decode")
        }
        records.append("PHASE_METRIC\t" + json.dumps(record))
    (run / "gateway.log").write_text("\n".join(records), encoding="utf-8")

    result = analyze(tmp_path)

    assert result["completion_calibration_evaluable"]
    assert not result["ranking_regret_evaluable"]
    assert not result["gate_c_evaluable"]
    assert result["scheduler_decision_us"]["p95"] == 20.0
    assert result["families"]["pd"]["observations"] == 2
    assert result["families"]["pd"]["interval_coverage"] == 1.0
    assert result["completion_directions"]["prefill_to_decode"][
        "incumbent_mae_us"] == 10.0
    assert result["completion_pairs"]["prefill_decode"][
        "incumbent_mae_us"] == 5.0
    assert result["completion_conformal"]["prefill_decode"]["ready_logs"] == 1
    assert result["completion_conformal"]["prefill_decode"][
        "median_scale"] == 1.5

    ranking_result = analyze(
        tmp_path, {
            "summary": {
                "contextual_ranking_regret_evaluable": True,
            },
            "contextual_ranking": {
                "evaluable_frontiers": 4,
                "top1_agreement_ratio": 0.75,
                "mean_normalized_h1_regret": 0.1,
            },
        })
    assert ranking_result["ranking_regret_evaluable"]
    assert ranking_result["gate_c_evaluable"]
    assert ranking_result["ranking"]["top1_agreement_ratio"] == 0.75
