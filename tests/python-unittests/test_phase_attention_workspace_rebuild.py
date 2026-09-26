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
"""CPU-only checks for fresh engine plans without GPU or Docker execution."""

import importlib.util
import json
import pathlib
import tempfile
import unittest


def load_tool():
    """Import the dry-run planner independently of model dependencies."""
    path = pathlib.Path(
        __file__).parents[2] / "scripts/rebuild_phase_attention_workspace.py"
    spec = importlib.util.spec_from_file_location(
        "attention_workspace_rebuild", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


G_TOOL = load_tool()


class WorkspaceEngineRebuildTest(unittest.TestCase):
    """Capabilities, immutability, and no-overwrite rules survive rebuilding."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = pathlib.Path(self.temporary.name)
        self.build = self.root / ".local/builds/validation"
        self.artifacts = self.root / ".local/artifacts/v0101-forward-port"
        self.output = self.artifacts / "fresh"
        self.results = self.root / ".local/results/fresh"
        self.config = {
            "max_batch_size": 80,
            "max_prefill_batch_size": 8,
            "max_decode_batch_size": 64,
            "max_kv_pool_pages": 256,
            "max_prefill_chunk_tokens": 128,
            "max_vision_prefill_batch_size": 4,
            "max_vision_prefill_chunk_tokens": 1024,
            "allow_kv_pool_undercommit": True,
            "profile_local_packed_prefill_chunk_limit": True,
            "tp_size": 1,
        }
        onnx_relative, reference_relative = G_TOOL.LINEAGES["cosmos"]
        self.onnx = self.artifacts / onnx_relative
        reference = self.artifacts / reference_relative
        self.onnx.mkdir(parents=True)
        reference.mkdir(parents=True)
        (reference / "config.json").write_text(
            json.dumps({"builder_config": self.config}))
        for name in ("model.onnx", "model.onnx.data", "config.json",
                     "embedding.safetensors"):
            (self.onnx / name).write_bytes(b"fixture")
        for name in ("examples/llm/llm_build",
                     "libNvInfer_edgellm_plugin.so.1.0"):
            path = self.build / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"fixture")

    def plan(self):
        return G_TOOL.plan_model(self.root, self.build, self.output,
                                 self.results, "cosmos")

    def test_preserves_capabilities_without_creating_output(self):
        plan = self.plan()
        self.assertEqual(plan["builder_config"], self.config)
        command = plan["build_command"]
        for name, value in self.config.items():
            if name in G_TOOL.FLAGS:
                self.assertEqual(
                    command[command.index(G_TOOL.FLAGS[name]) + 1], str(value))
        self.assertIn("--allowKVPoolUndercommit", command)
        self.assertFalse(self.output.exists())
        self.assertFalse(self.results.exists())

    def test_links_only_weight_sidecars_with_one_mount(self):
        plan = self.plan()
        link = plan["sidecar_link_command"]
        self.assertEqual(link.count("-v"), 1)
        self.assertEqual(link[link.index("-v") + 1],
                         str(self.artifacts) + ":/opt/artifacts:rw")
        self.assertNotIn("--gpus", link)
        self.assertEqual(plan["sidecars"], ["embedding.safetensors"])
        self.assertEqual(
            link[-2],
            "/opt/artifacts/" + str(self.onnx.relative_to(self.artifacts)) +
            "/embedding.safetensors")
        self.assertEqual(link[-1], "/opt/artifacts/fresh/cosmos/")
        self.assertIn(str(self.onnx) + ":/opt/onnx:ro", plan["build_command"])

    def test_existing_artifact_or_dangling_pointer_is_never_overwritten(self):
        target = self.output / "cosmos"
        target.parent.mkdir(parents=True)
        target.symlink_to(self.root / "missing")
        with self.assertRaises(FileExistsError):
            self.plan()
        target.unlink()
        target.mkdir()
        with self.assertRaises(FileExistsError):
            self.plan()

    def test_rejects_artifacts_outside_shared_store(self):
        with self.assertRaises(ValueError):
            G_TOOL.plan_model(self.root, self.build, self.root / "outside",
                              self.results, "cosmos")


if __name__ == "__main__":
    unittest.main()
