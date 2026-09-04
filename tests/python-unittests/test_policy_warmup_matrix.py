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

SCRIPT = (Path(__file__).parents[2] / "benchmarks" / "phase_serving" /
          "run_policy_warmup_matrix.py")
SPEC = importlib.util.spec_from_file_location("warmup_matrix", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MATRIX = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MATRIX)


def _trace(path: Path, vision: bool) -> None:
    content: str | list[dict] = "hello"
    if vision:
        content = [{"type": "image_url", "image_url": {"url": "file:///x"}}]
    path.write_text(json.dumps(
        {"requests": [{
            "messages": [{
                "content": content
            }]
        }]}),
                    encoding="utf-8")


def test_prepare_generic_command_selects_vlm_calibration(
        tmp_path: Path) -> None:
    measured = tmp_path / "measured.json"
    generic_text = tmp_path / "text.json"
    generic_vlm = tmp_path / "vlm.json"
    _trace(measured, True)
    _trace(generic_text, False)
    _trace(generic_vlm, True)
    command = [
        "python3",
        "bench.py",
        "--trace",
        str(measured),
        "--output-dir",
        "old",
        "--repeats",
        "1",
        "--warmup-requests",
        "10",
        "--",
        "docker",
        "run",
        "--rm",
        "nvcr.io/nvidia/tensorrt:26.06-py3",
        "binary",
    ]
    result = MATRIX.prepare_command({"command": command}, "generic",
                                    tmp_path / "out", 3, generic_text,
                                    generic_vlm)

    assert result[result.index("--generic-warmup-trace") +
                  1] == str(generic_vlm)
    assert result[result.index("--warmup-requests") + 1] == "1"
    assert result[result.index("--phase-calibration-round-requests") +
                  1] == "1"
    assert "TRT_EDGELLM_POLICY_WARMUP_MODE={policy_warmup_mode}" in result


def test_prepare_replaces_stale_backend_build(tmp_path: Path) -> None:
    measured = tmp_path / "measured.json"
    generic = tmp_path / "generic.json"
    _trace(measured, False)
    _trace(generic, False)
    command = [
        "python3",
        "bench.py",
        "--trace",
        str(measured),
        "--output-dir",
        "old",
        "--repeats",
        "1",
        "--",
        "docker",
        "run",
        "--rm",
        "-e",
        "EDGELLM_PLUGIN_PATH=/workspace/old/libNvInfer_edgellm_plugin.so.1.0",
        "-e",
        "LD_LIBRARY_PATH=/opt/tensorrt/lib:/workspace/old/examples/llm",
        "nvcr.io/nvidia/tensorrt:26.06-py3",
        "/workspace/old/examples/llm/llm_phase_context_smoke",
    ]

    result = MATRIX.prepare_command({"command": command}, "zero_start",
                                    tmp_path / "out", 1, generic, generic,
                                    "/workspace/new-build")

    assert "/workspace/new-build/examples/llm/llm_phase_context_smoke" in result
    assert ("EDGELLM_PLUGIN_PATH=/workspace/new-build/"
            "libNvInfer_edgellm_plugin.so.1.0") in result
    assert "LD_LIBRARY_PATH=/opt/tensorrt/lib:/workspace/new-build/examples/llm" in result


def test_prepare_replaces_backend_engine(tmp_path: Path) -> None:
    measured = tmp_path / "measured.json"
    generic = tmp_path / "generic.json"
    _trace(measured, False)
    _trace(generic, False)
    command = [
        "python3", "bench.py", "--trace",
        str(measured), "--output-dir", "old", "--repeats", "1", "--", "docker",
        "run", "--rm", "nvcr.io/nvidia/tensorrt:26.06-py3",
        "/workspace/build/examples/llm/llm_phase_context_smoke",
        "/workspace/old-engine", "/workspace/model"
    ]

    result = MATRIX.prepare_command({"command": command},
                                    "zero_start",
                                    tmp_path / "out",
                                    1,
                                    generic,
                                    generic,
                                    backend_engine_dir="/workspace/new-engine")

    binary = result.index(
        "/workspace/build/examples/llm/llm_phase_context_smoke")
    assert result[binary + 1] == "/workspace/new-engine"
    assert result[binary + 2] == "/workspace/model"


def test_prepare_pd_only_changes_authority_without_changing_mode(
        tmp_path: Path) -> None:
    measured = tmp_path / "measured.json"
    generic = tmp_path / "generic.json"
    _trace(measured, False)
    _trace(generic, False)
    command = [
        "python3",
        "bench.py",
        "--trace",
        str(measured),
        "--output-dir",
        "old",
        "--repeats",
        "1",
        "--",
        "docker",
        "run",
        "--rm",
        "-e",
        "TRT_EDGELLM_CONTEXTUAL_PD=shadow",
        "-e",
        "TRT_EDGELLM_CONTEXTUAL_EP=active",
        "-e",
        "TRT_EDGELLM_CONTEXTUAL_ED=active",
        "nvcr.io/nvidia/tensorrt:26.06-py3",
        "binary",
    ]

    result = MATRIX.prepare_command({"command": command},
                                    "generic",
                                    tmp_path / "out",
                                    1,
                                    generic,
                                    generic,
                                    policy_variant="pd_only")
    environments = {
        result[index + 1].split("=", 1)[0]: result[index + 1].split("=", 1)[1]
        for index, token in enumerate(result[:-1]) if token == "-e"
    }

    assert environments["TRT_EDGELLM_CONTEXTUAL_PD"] == "active"
    assert environments["TRT_EDGELLM_CONTEXTUAL_EP"] == "shadow"
    assert environments["TRT_EDGELLM_CONTEXTUAL_ED"] == "shadow"
    assert "TRT_EDGELLM_POLICY_WARMUP_MODE={policy_warmup_mode}" in result


def test_prepare_exact_only_disables_learned_authority(tmp_path: Path) -> None:
    measured = tmp_path / "measured.json"
    generic = tmp_path / "generic.json"
    _trace(measured, False)
    _trace(generic, False)
    command = [
        "python3", "bench.py", "--trace",
        str(measured), "--output-dir", "old", "--repeats", "1", "--", "docker",
        "run", "--rm", "nvcr.io/nvidia/tensorrt:26.06-py3", "binary"
    ]

    result = MATRIX.prepare_command({"command": command},
                                    "generic",
                                    tmp_path / "out",
                                    1,
                                    generic,
                                    generic,
                                    policy_variant="exact_only")
    environments = {
        result[index + 1].split("=", 1)[0]: result[index + 1].split("=", 1)[1]
        for index, token in enumerate(result[:-1]) if token == "-e"
    }

    assert environments["TRT_EDGELLM_CONTEXTUAL_PD"] == "disabled"
    assert environments["TRT_EDGELLM_CONTEXTUAL_EP"] == "disabled"
    assert environments["TRT_EDGELLM_CONTEXTUAL_ED"] == "disabled"
    assert environments["TRT_EDGELLM_ENABLE_GLOBAL_FORMATION_AWARE"] == "0"


def test_prepare_scalar_transition_enables_only_formation_reasoning(
        tmp_path: Path) -> None:
    measured = tmp_path / "measured.json"
    generic = tmp_path / "generic.json"
    _trace(measured, False)
    _trace(generic, False)
    command = [
        "python3", "bench.py", "--trace",
        str(measured), "--output-dir", "old", "--repeats", "1", "--", "docker",
        "run", "--rm", "nvcr.io/nvidia/tensorrt:26.06-py3", "binary"
    ]

    result = MATRIX.prepare_command({"command": command},
                                    "generic",
                                    tmp_path / "out",
                                    1,
                                    generic,
                                    generic,
                                    policy_variant="scalar_transition")
    environments = {
        result[index + 1].split("=", 1)[0]: result[index + 1].split("=", 1)[1]
        for index, token in enumerate(result[:-1]) if token == "-e"
    }

    assert environments["TRT_EDGELLM_CONTEXTUAL_PD"] == "active"
    assert environments["TRT_EDGELLM_CONTEXTUAL_EP"] == "active"
    assert environments["TRT_EDGELLM_CONTEXTUAL_ED"] == "active"
    assert environments["TRT_EDGELLM_ENABLE_GLOBAL_FORMATION_AWARE"] == "1"
    assert environments["TRT_EDGELLM_COMPLETION_CONFORMAL_ACTIVE"] == "0"


def test_prepare_overrides_backend_environment(tmp_path: Path) -> None:
    measured = tmp_path / "measured.json"
    generic = tmp_path / "generic.json"
    _trace(measured, False)
    _trace(generic, False)
    command = [
        "python3", "bench.py", "--trace",
        str(measured), "--output-dir", "old", "--repeats", "1", "--", "docker",
        "run", "--rm", "-e", "TRT_EDGELLM_GLOBAL_SCHEDULER=active",
        "nvcr.io/nvidia/tensorrt:26.06-py3", "binary"
    ]

    result = MATRIX.prepare_command(
        {"command": command},
        "zero_start",
        tmp_path / "out",
        1,
        generic,
        generic,
        backend_environment=("TRT_EDGELLM_GLOBAL_SCHEDULER=disabled", ))

    assert "TRT_EDGELLM_GLOBAL_SCHEDULER=disabled" in result
    assert "TRT_EDGELLM_GLOBAL_SCHEDULER=active" not in result


def test_prepare_overrides_client_max_in_flight(tmp_path: Path) -> None:
    measured = tmp_path / "measured.json"
    generic = tmp_path / "generic.json"
    _trace(measured, False)
    _trace(generic, False)
    command = [
        "python3", "bench.py", "--trace",
        str(measured), "--output-dir", "old", "--repeats", "1",
        "--max-in-flight", "48", "--", "docker", "run", "--rm",
        "nvcr.io/nvidia/tensorrt:26.06-py3", "binary"
    ]

    result = MATRIX.prepare_command({"command": command},
                                    "zero_start",
                                    tmp_path / "out",
                                    1,
                                    generic,
                                    generic,
                                    client_max_in_flight=80)

    assert result[result.index("--max-in-flight") + 1] == "80"


def test_prepare_maps_phase_telemetry_to_writable_workspace(
        tmp_path: Path) -> None:
    measured = tmp_path / "measured.json"
    generic = tmp_path / "generic.json"
    _trace(measured, False)
    _trace(generic, False)
    command = [
        "python3", "bench.py", "--trace",
        str(measured), "--output-dir", "old", "--repeats", "1", "--", "docker",
        "run", "--rm", "-v", f"{tmp_path}:/workspace",
        "nvcr.io/nvidia/tensorrt:26.06-py3", "binary"
    ]
    output = tmp_path / "matrix" / "balanced"

    result = MATRIX.prepare_command({"command": command},
                                    "zero_start",
                                    output,
                                    1,
                                    generic,
                                    generic,
                                    capture_phase_telemetry=True)

    environments = {
        result[index + 1].split("=", 1)[0]: result[index + 1].split("=", 1)[1]
        for index, token in enumerate(result[:-1]) if token == "-e"
    }
    expected = "/workspace/matrix/balanced/activity/run-{run}"
    assert environments["TRT_EDGELLM_PHASE_ACTIVITY_PREFIX"] == expected
    assert environments["TRT_EDGELLM_PHASE_TELEMETRY_PATH"] == expected + \
        "-events.jsonl"
    assert environments["TRT_EDGELLM_PHASE_TELEMETRY_LEVEL"] == \
        "counterfactual"
