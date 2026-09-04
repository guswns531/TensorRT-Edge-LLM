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

from pathlib import Path

from benchmarks.phase_serving.run_oracle_h1_policy_matrix import \
    materialize_command


def environment(command: list[str]) -> dict[str, str]:
    result = {}
    for index, token in enumerate(command[:-1]):
        if token == "-e":
            name, value = command[index + 1].split("=", 1)
            result[name] = value
    return result


def entry() -> dict:
    return {
        "case":
        "balanced",
        "variant":
        "worker-4",
        "command": [
            "python3", "bench.py", "--trace", ".local/trace.json",
            "--output-dir", "/old", "--repeats", "3", "--", "docker", "run",
            "-e", "TRT_EDGELLM_GLOBAL_LEGACY_COMPATIBILITY=1", "-e",
            "EDGELLM_PLUGIN_PATH=/old/plugin.so",
            "nvcr.io/nvidia/tensorrt:26.06-py3",
            "/old/llm_phase_context_smoke", "/engine", "/model"
        ],
    }


def test_materializes_safe_probe_with_common_epoch_instrumentation() -> None:
    result = materialize_command(entry(), "safe_probe", Path("/host/out"),
                                 Path("/host"), Path("/workspace"),
                                 "/workspace/plugin.so",
                                 "/workspace/build/llm_phase_context_smoke")
    command = result["command"]
    values = environment(command)
    assert command[command.index("--repeats") + 1] == "1"
    assert command[command.index("--trace") + 1] == "/host/.local/trace.json"
    assert values["TRT_EDGELLM_EMIT_PHASE_METRICS"] == "1"
    assert values["TRT_EDGELLM_PHASE_ACTIVITY_PREFIX"] == (
        "/workspace/out/balanced/worker-4/activity/run-001")
    assert values["TRT_EDGELLM_PHASE_RUN_ID"] == (
        "oracle-h1-safe_probe-balanced-r001")
    assert values["TRT_EDGELLM_GLOBAL_OVERLAP_MIN_SAMPLES"] == "1"
    assert "TRT_EDGELLM_GLOBAL_LEGACY_COMPATIBILITY" not in values
    assert "/workspace/build/llm_phase_context_smoke" in command


def test_materializes_legacy_without_probe_controls() -> None:
    result = materialize_command(entry(), "legacy", Path("/host/out"),
                                 Path("/host"), Path("/workspace"),
                                 "/workspace/plugin.so",
                                 "/workspace/build/llm_phase_context_smoke")
    values = environment(result["command"])
    assert values["TRT_EDGELLM_GLOBAL_LEGACY_COMPATIBILITY"] == "1"
    assert values["TRT_EDGELLM_GLOBAL_SAFE_PROBE_SLACK_MULTIPLIER"] == "0"
    assert "TRT_EDGELLM_DISABLE_GLOBAL_OVERLAP_WARMUP" not in values


def test_materializes_independent_matrix_repeat_lineage() -> None:
    result = materialize_command(entry(), "myopic", Path("/host/repeat-002"),
                                 Path("/host"), Path("/workspace"),
                                 "/workspace/plugin.so",
                                 "/workspace/build/llm_phase_context_smoke", 2)
    values = environment(result["command"])
    assert result["matrix_repeat"] == 2
    assert result["output_dir"] == "/host/repeat-002/balanced/worker-4"
    assert values["TRT_EDGELLM_PHASE_ACTIVITY_PREFIX"] == (
        "/workspace/repeat-002/balanced/worker-4/activity/run-001")
    assert values["TRT_EDGELLM_PHASE_RUN_ID"] == (
        "oracle-h1-myopic-balanced-r002")


def test_materializes_m6_shadow_without_policy_authority() -> None:
    result = materialize_command(entry(), "m6_shadow", Path("/host/out"),
                                 Path("/host"), Path("/workspace"),
                                 "/workspace/plugin.so",
                                 "/workspace/build/llm_phase_context_smoke")
    values = environment(result["command"])
    assert values["TRT_EDGELLM_CONTEXTUAL_PD"] == "shadow"
    assert values["TRT_EDGELLM_CONTEXTUAL_EP"] == "shadow"
    assert values["TRT_EDGELLM_CONTEXTUAL_ED"] == "shadow"
    assert values["TRT_EDGELLM_GLOBAL_SAFE_PROBE_SLACK_MULTIPLIER"] == "0"
    assert values["TRT_EDGELLM_CONTEXTUAL_PD_CONFIDENCE_BETA"] == "1.96"


def test_materializes_pd_only_authority() -> None:
    result = materialize_command(entry(), "pd_only", Path("/host/out"),
                                 Path("/host"), Path("/workspace"),
                                 "/workspace/plugin.so",
                                 "/workspace/build/llm_phase_context_smoke")
    values = environment(result["command"])
    assert values["TRT_EDGELLM_CONTEXTUAL_PD"] == "active"
    assert values["TRT_EDGELLM_CONTEXTUAL_EP"] == "shadow"
    assert values["TRT_EDGELLM_CONTEXTUAL_ED"] == "shadow"


def test_materializes_pd_causal_branch_without_forcing_encoder_pairs() -> None:
    result = materialize_command(entry(), "pd_force_overlap",
                                 Path("/host/out"), Path("/host"),
                                 Path("/workspace"), "/workspace/plugin.so",
                                 "/workspace/build/llm_phase_context_smoke")
    values = environment(result["command"])
    assert values["TRT_EDGELLM_EXPERIMENTAL_OVERLAP_PERCENT"] == "100"
    assert "TRT_EDGELLM_EXPERIMENTAL_ENCODER_PREFILL_OVERLAP_PERCENT" not in values
    assert "TRT_EDGELLM_EXPERIMENTAL_ENCODER_DECODE_OVERLAP_PERCENT" not in values


def test_materializes_full_request_timeline() -> None:
    result = materialize_command(entry(), "current", Path("/host/out"),
                                 Path("/host"), Path("/workspace"),
                                 "/workspace/plugin.so",
                                 "/workspace/build/llm_phase_context_smoke", 1,
                                 "full")
    values = environment(result["command"])
    assert values["TRT_EDGELLM_PHASE_TELEMETRY_LEVEL"] == "full"
