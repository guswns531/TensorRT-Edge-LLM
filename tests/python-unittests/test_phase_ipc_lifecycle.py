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
"""CPU-only cancellation protocol tests, without inference or GPU dependencies."""

import copy
import hashlib
import importlib.util
import pathlib
import tempfile
import unittest
import unittest.mock


def load_tool():
    """Load the standalone lifecycle script without model dependencies."""
    path = (pathlib.Path(__file__).parents[2] /
            "benchmarks/phase_serving/check_phase_ipc_lifecycle.py")
    spec = importlib.util.spec_from_file_location("phase_ipc_lifecycle", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


G_TOOL = load_tool()


class SanitizerDiagnosticTest(unittest.TestCase):
    """Instrumentation changes are explicit and restricted to one Docker launch."""

    def test_default_memcheck_has_no_forced_synchronization(self):
        self.assertEqual(G_TOOL.sanitizer_prefix(), [
            "compute-sanitizer", "--tool", "memcheck", "--error-exitcode", "99"
        ])

    def test_positive_limit_is_forwarded_only_to_sanitizer(self):
        self.assertEqual(
            G_TOOL.sanitizer_prefix(1)[-2:],
            ["--force-synchronization-limit", "1"])
        for invalid in (0, -1):
            with self.assertRaises(ValueError):
                G_TOOL.sanitizer_prefix(invalid)

    def test_debug_wrapper_preserves_ipc_tail_network_and_runtime_limits(self):
        runtime = [
            "/opt/edgellm/examples/llm/llm_phase_context_smoke", "/model"
        ]
        command = [
            "docker", "run", "--rm", "--network", "none", "--cap-drop", "ALL",
            "--read-only", "-i", "-e", "TRT_EDGELLM_MAX_DECODE_BATCH=24", "-e",
            "CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1", "image:fixed"
        ] + G_TOOL.sanitizer_prefix(1) + runtime
        original = command.copy()
        result = G_TOOL.enable_debug_backtrace(command)
        self.assertEqual(command, original)
        self.assertEqual(result[result.index("compute-sanitizer"):],
                         G_TOOL.sanitizer_prefix(1) + runtime)
        self.assertEqual(result[result.index("--network") + 1], "none")
        for preserved in ("-i", "--read-only", "ALL",
                          "TRT_EDGELLM_MAX_DECODE_BATCH=24"):
            self.assertIn(preserved, result)
        for scoped in ("--cap-add=SYS_PTRACE",
                       "--security-opt=seccomp=unconfined",
                       "--ulimit=core=0:0"):
            self.assertLess(result.index(scoped), result.index("image:fixed"))
        self.assertIn("CUDA_ENABLE_COREDUMP_ON_EXCEPTION=0", result)
        self.assertNotIn("CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1", result)
        self.assertLess(result.index("cuda-gdb-minimal"),
                        result.index("compute-sanitizer"))
        for instruction in ("set startup-with-shell off",
                            "set follow-fork-mode child", "info proc exe",
                            "thread apply all -c bt 24"):
            self.assertIn(instruction, result)
        self.assertNotIn("--pid=host", result)
        self.assertNotIn("--privileged", result)

    def test_debug_requires_isolated_network(self):
        command = ["docker", "run", "image"] + G_TOOL.sanitizer_prefix()
        with self.assertRaises(ValueError):
            G_TOOL.enable_debug_backtrace(command)


class RequestFixtureTest(unittest.TestCase):
    """Text-only diagnosis excludes vision requests, not vision engine loading."""

    def test_text_only_drops_images_and_all_source_request_metadata(self):
        source = [{
            "messages": [{
                "role":
                "user",
                "content": [{
                    "type": "image_url",
                    "image_url": "image.png"
                }],
            }],
            "images": ["another.png"],
            "max_output_tokens":
            999,
            "metadata": {
                "vision": True
            },
        }]
        original = copy.deepcopy(source)
        fixture = G_TOOL.make_request_fixture(source, text_only=True)
        self.assertEqual(source, original)
        self.assertEqual(len(fixture), 2)
        for request in fixture:
            self.assertEqual(set(request), {"messages"})
            self.assertEqual(len(request["messages"]), 1)
            message = request["messages"][0]
            self.assertEqual(message["role"], "user")
            self.assertIsInstance(message["content"], str)
            self.assertTrue(message["content"])

    def test_default_preserves_source_payload_without_aliasing(self):
        source = [{"messages": [{"role": "user", "content": "hello"}]}]
        fixture = G_TOOL.make_request_fixture(source, text_only=False)
        self.assertEqual(fixture, source)
        fixture[0]["messages"][0]["content"] = "changed"
        self.assertEqual(source[0]["messages"][0]["content"], "hello")

    def test_text_fixture_does_not_require_source_images(self):
        first = G_TOOL.make_request_fixture([], text_only=True)
        first[0]["messages"][0]["content"] = "changed"
        second = G_TOOL.make_request_fixture([], text_only=True)
        self.assertNotEqual(first, second)
        with self.assertRaises(ValueError):
            G_TOOL.make_request_fixture([], text_only=False)


class LaunchedIdentityTest(unittest.TestCase):
    """The launch hashes describe mounted files, not the source campaign's build."""

    def test_hashes_mounted_files_with_sanitizer_and_current_checkout(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            executable = root / "examples/llm/llm_phase_context_smoke"
            executable.parent.mkdir(parents=True)
            executable.write_bytes(b"current binary")
            plugin = root / "current-plugin.so"
            plugin.write_bytes(b"changed plugin")
            command = [
                "docker",
                "run",
                "-v",
                str(root) + ":/opt/edgellm:ro",
                "-e",
                "EDGELLM_PLUGIN_PATH=/opt/edgellm/current-plugin.so",
                "image",
                "compute-sanitizer",
                "--tool",
                "memcheck",
                "--error-exitcode",
                "99",
                "/opt/edgellm/examples/llm/llm_phase_context_smoke",
                "/opt/model",
            ]
            original = command.copy()
            with unittest.mock.patch.object(
                    G_TOOL.subprocess,
                    "check_output",
                    side_effect=["current-head\n", " M source.cpp\n"]):
                result = G_TOOL.collect_launched_identity(command, root)
            self.assertEqual(command, original)
            self.assertEqual(result["binary"]["sha256"],
                             hashlib.sha256(b"current binary").hexdigest())
            self.assertEqual(result["plugin"]["sha256"],
                             hashlib.sha256(b"changed plugin").hexdigest())
            self.assertEqual(result["plugin"]["size_bytes"], 14)
            self.assertEqual(result["source_checkout"]["commit"],
                             "current-head")
            self.assertTrue(result["source_checkout"]["dirty"])
            self.assertFalse(
                result["source_checkout"]["proves_artifact_build_commit"])

    def test_missing_build_mount_is_not_inferred_from_parent(self):
        with self.assertRaises(ValueError):
            G_TOOL.collect_launched_identity(["docker", "run", "image"],
                                             pathlib.Path("/tmp"))


class DockerEnvironmentTest(unittest.TestCase):
    """Minimal startup changes only the explicitly recorded environment keys."""

    def test_replaces_duplicates_adds_missing_and_preserves_tail(self):
        command = [
            "docker",
            "run",
            "--rm",
            "-e",
            "WARMUP=generic",
            "--env",
            "WARMUP=trace_derived",
            "--env=WARMUP=generic",
            "-e",
            "CAPACITY=24",
            "image:fixed",
            "/binary",
            "--env=WARMUP=argument",
        ]
        result = G_TOOL.replace_docker_environment(
            command, {
                "WARMUP": "zero_start",
                "GRAPHS": "0"
            }, command.index("image:fixed"))
        self.assertEqual(result, [
            "docker",
            "run",
            "--rm",
            "-e",
            "CAPACITY=24",
            "-e",
            "WARMUP=zero_start",
            "-e",
            "GRAPHS=0",
            "image:fixed",
            "/binary",
            "--env=WARMUP=argument",
        ])
        self.assertIn("WARMUP=generic", command)

    def test_unsets_explicit_shape_override_without_empty_assignment(self):
        command = [
            "docker", "run", "-e", "SHAPES=1:128,2:512", "image", "/bin"
        ]
        result = G_TOOL.replace_docker_environment(command, {"SHAPES": None},
                                                   command.index("image"))
        self.assertEqual(result, ["docker", "run", "image", "/bin"])

    def test_empty_overrides_preserve_normal_startup(self):
        command = ["docker", "run", "-e", "MODE=generic", "image", "/bin"]
        self.assertEqual(
            G_TOOL.replace_docker_environment(command, {},
                                              command.index("image")), command)

    def test_minimal_startup_preserves_resource_contract(self):
        protected = {
            "TRT_EDGELLM_MAX_STABLE_SLOTS": "24",
            "TRT_EDGELLM_MAX_INFLIGHT": "24",
            "TRT_EDGELLM_MAX_PREFILL_BATCH": "8",
            "TRT_EDGELLM_MAX_DECODE_BATCH": "24",
            "TRT_EDGELLM_VISION_ENCODER_BATCH_SIZE": "4",
            "TRT_EDGELLM_FIXED_PREFILL_CHUNK": "128",
            "TRT_EDGELLM_PHASE_WORKSPACE_MODE": "shared_ep",
        }
        command = ["docker", "run"]
        for name, value in protected.items():
            command.extend(["-e", name + "=" + value])
        command.extend([
            "-e", "TRT_EDGELLM_CAPTURE_PHASE_GRAPHS=1", "-e",
            "TRT_EDGELLM_IPC_WARMUP_DECODE_SHAPES=1:128", "image", "/bin"
        ])
        result = G_TOOL.replace_docker_environment(
            command, G_TOOL.MINIMAL_STARTUP_OVERRIDES, command.index("image"))
        for name, value in protected.items():
            self.assertIn(name + "=" + value, result)
        self.assertIn("TRT_EDGELLM_DISABLE_IPC_SHAPE_WARMUP=1", result)
        self.assertIn("TRT_EDGELLM_CAPTURE_PHASE_GRAPHS=0", result)
        self.assertIn("TRT_EDGELLM_POLICY_WARMUP_MODE=zero_start", result)
        self.assertNotIn("TRT_EDGELLM_IPC_WARMUP_DECODE_SHAPES=1:128", result)

    def test_rejects_malformed_docker_environment(self):
        with self.assertRaises(ValueError):
            G_TOOL.replace_docker_environment(
                ["docker", "run", "-e", "image", "/bin"], {"A": "1"}, 3)


class CancellationAttemptsTest(unittest.TestCase):
    """Only accepted cancellation permits readmission in the lifecycle probe."""

    def test_busy_rejection_retries_at_next_token(self):
        tracker = G_TOOL.CancellationAttempts()
        self.assertTrue(tracker.at_token(0))
        self.assertFalse(tracker.acknowledge(False))
        self.assertFalse(tracker.at_token(0))
        self.assertTrue(tracker.at_token(1))
        self.assertTrue(tracker.acknowledge(True))
        self.assertEqual(tracker.report()["attempts"], 2)
        self.assertEqual(tracker.report()["busy_rejections"], 1)
        self.assertFalse(tracker.report()["acknowledgement_pending"])

    def test_does_not_enqueue_duplicate_pending_attempt(self):
        tracker = G_TOOL.CancellationAttempts()
        self.assertTrue(tracker.at_token(0))
        self.assertFalse(tracker.at_token(1))
        self.assertFalse(tracker.at_token(2))
        self.assertEqual(tracker.report()["attempts"], 1)
        self.assertFalse(tracker.acknowledge(False))
        self.assertTrue(tracker.at_token(3))

    def test_accepted_cancellation_is_terminal(self):
        tracker = G_TOOL.CancellationAttempts()
        self.assertTrue(tracker.at_token(0))
        self.assertTrue(tracker.acknowledge(True))
        self.assertFalse(tracker.at_token(1))
        self.assertEqual(tracker.report()["attempts"], 1)

    def test_rejects_unsolicited_or_malformed_acknowledgement(self):
        tracker = G_TOOL.CancellationAttempts()
        with self.assertRaises(AssertionError):
            tracker.acknowledge(True)
        self.assertTrue(tracker.at_token(0))
        with self.assertRaises(AssertionError):
            tracker.acknowledge(None)
        self.assertTrue(tracker.report()["acknowledgement_pending"])


if __name__ == "__main__":
    unittest.main()
