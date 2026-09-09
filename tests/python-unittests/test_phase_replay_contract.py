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

import hashlib
import importlib.util
import json
import pathlib
import tempfile
import unittest


def load_tool(name):
    path = pathlib.Path(
        __file__).parents[2] / "benchmarks/phase_serving" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GUARD = load_tool("guarded_trace_client")
REPLAY = load_tool("replay_retained_policy_commands")
DISTRIBUTIONS = load_tool("report_request_distributions")
OUTPUTS = load_tool("compare_engine_outputs")
DISPATCH = load_tool("summarize_compact_dispatch")
SELECTOR = load_tool("analyze_selector_audit")
SHADOW = load_tool("analyze_all_late_shadow")
BRANCH = load_tool("inspect_branch_replay")


class ReplayContractTest(unittest.TestCase):

    def test_branch_metadata_does_not_accept_zero_ownership(self):
        event = dict(ready=dict(prefill_rows=2, decode_rows=48), candidates=[])
        errors = BRANCH.metadata_errors(event)
        self.assertIn('missing_kv_ownership_signature', errors)
        self.assertIn('incomplete_prefill_rows', errors)
        self.assertIn('incomplete_decode_rows', errors)
        self.assertIn('missing_candidate_snapshots', errors)

    def test_branch_metadata_requires_request_aligned_lengths(self):
        event = dict(ready=dict(prefill_rows=1, decode_rows=1),
                     candidates=[dict(action_id=1)],
                     kv_ownership_signature=1,
                     vision_lease_signature=1,
                     strict_snapshot_signature=1,
                     scalar_policy_state_signature=1,
                     ready_prefill_request_ids=[1],
                     ready_prefill_token_counts=[128],
                     ready_decode_request_ids=[2],
                     ready_decode_context_lengths=[512])
        self.assertEqual(BRANCH.metadata_errors(event), [])
        event['ready_decode_context_lengths'] = []
        self.assertIn('incomplete_decode_rows', BRANCH.metadata_errors(event))

    def test_complete_branch_metadata_is_not_a_physical_checkpoint(self):
        single = dict(action_id=1,
                      action_kind='prefill',
                      primary_batch=1,
                      secondary_batch=0,
                      hard_feasible=True,
                      frontier_eligible=True,
                      dominated=False)
        pair = dict(single,
                    action_id=2,
                    action_kind='prefill_decode',
                    secondary_batch=1)
        event = dict(event_kind='decision',
                     decision_id=1,
                     action_kind='prefill',
                     ready=dict(prefill_rows=1, decode_rows=1),
                     kv_ownership_signature=1,
                     vision_lease_signature=1,
                     strict_snapshot_signature=1,
                     scalar_policy_state_signature=1,
                     ready_prefill_request_ids=[1],
                     ready_prefill_token_counts=[128],
                     ready_decode_request_ids=[2],
                     ready_decode_context_lengths=[512],
                     candidates=[
                         dict(action_id=1, request_ids=[1]),
                         dict(action_id=2, request_ids=[1, 2])
                     ],
                     selector_audit=dict(inputs=[single, pair]))
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / 'events.jsonl'
            path.write_text('PHASE_EPOCH\t{"kind":"measurement"}\n' +
                            'PHASE_SCHEDULER_EVENT\t' + json.dumps(event) +
                            '\n')
            result = BRANCH.inspect(path, 1, 1)['matches'][0]
            self.assertEqual(result['errors'], [])
            self.assertTrue(result['metadata_fingerprint'])
            self.assertFalse(result['runtime_replay_ready'])
            self.assertEqual(BRANCH.inspect(path, 1, 2)['matches'], [])
            self.assertEqual(len(BRANCH.inspect(path, 1, None)['matches']), 1)

    def test_all_late_shadow_uses_only_the_actual_eligible_frontier(self):
        actual = dict(action_id=1,
                      hard_feasible=True,
                      frontier_eligible=True,
                      dominated=False,
                      additional_violation_us=200.0,
                      service_compression=1.2,
                      selection_horizon_us=200.0)
        lower_delay = dict(actual,
                           action_id=2,
                           additional_violation_us=100.0,
                           service_compression=1.0)
        pruned = dict(lower_delay,
                      action_id=3,
                      additional_violation_us=0.0,
                      dominated=True)
        audit = dict(inputs=[actual, lower_delay, pruned],
                     selected_action_id=1)
        result = SHADOW.compare(audit)
        self.assertEqual(result['shadow']['action_id'], 2)
        self.assertEqual(result['predicted_added_delay_saved_us'], 100.0)
        self.assertAlmostEqual(result['compression_delta'], -0.2)
        self.assertEqual(audit['selected_action_id'], 1)
        lower_delay['additional_violation_us'] = float('nan')
        with self.assertRaises(ValueError):
            SHADOW.compare(audit)

    def test_restored_decode_selection_is_not_final_dispatch(self):
        candidate = dict(action_id=1,
                         action_kind='decode',
                         hard_feasible=True,
                         max_slo_violation_us=1)
        encoder = dict(action_id=2,
                       action_kind='encoder',
                       hard_feasible=True,
                       max_slo_violation_us=2)
        event = dict(event_kind='decision',
                     decision_id=1,
                     action_kind='encoder',
                     ready=dict(decode_rows=1),
                     decode_guard_audit=dict(prefill_expired=True,
                                             decode_expired=True,
                                             candidate_restored=True,
                                             candidate_suppressed=False),
                     pd_selector_audit=dict(inputs=[candidate],
                                            selected_action_id=1,
                                            reason='minimum_violation'),
                     selector_audit=dict(inputs=[encoder],
                                         selected_action_id=2,
                                         reason='minimum_violation',
                                         post_select_override=False))
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / 'events.jsonl'
            path.write_text('PHASE_EPOCH\t{"kind":"measurement"}\n' +
                            'PHASE_SCHEDULER_EVENT\t' + json.dumps(event) +
                            '\n')
            counts = SELECTOR.analyze(path)['counts']
            self.assertEqual(counts['guard_restored'], 1)
            self.assertEqual(counts['restored_local_d_selected'], 1)
            self.assertEqual(counts['restored_local_d_then_non_d'], 1)
            self.assertEqual(counts['restored_final_encoder'], 1)
            self.assertEqual(counts['restored_without_dispatch'], 1)
            with path.open('a') as output:
                output.write('PHASE_SCHEDULER_EVENT\t' + json.dumps(
                    dict(event_kind='dispatch',
                         decision_id=1,
                         action_kind='encoder')) + '\n')
            counts = SELECTOR.analyze(path)['counts']
            self.assertEqual(counts['restored_without_dispatch'], 0)
            self.assertEqual(counts['restored_dispatch_encoder'], 1)
            event['pd_selector_audit']['inputs'] = []
            path.write_text('PHASE_EPOCH\t{"kind":"measurement"}\n' +
                            'PHASE_SCHEDULER_EVENT\t' + json.dumps(event) +
                            '\n')
            with self.assertRaises(ValueError):
                SELECTOR.analyze(path)

    def test_dispatch_epoch_is_required(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "dispatch.jsonl"
            path.write_text('PHASE_METRIC\t{}\n')
            with self.assertRaises(ValueError):
                DISPATCH.analyze(path)

    def test_negative_host_intervals_are_preserved(self):
        result = DISPATCH.stats([-1.0, 2.0])
        self.assertEqual(result["negative_count"], 1)
        self.assertEqual(result["mean"], 0.5)

    def test_eos_identity_does_not_hide_strict_difference(self):
        result = OUTPUTS.compare({"a": [1, 9, 2]}, {"a": [1, 9, 3]}, {9})
        self.assertEqual(result["full_equal"], 0)
        self.assertEqual(result["through_eos_equal"], 1)
        result = OUTPUTS.compare({"a": [1, 9]}, {"a": [2, 9]}, {9})
        self.assertEqual(result["through_eos_equal"], 0)
        with self.assertRaises(ValueError):
            OUTPUTS.compare({"a": [1]}, {"b": [1]}, {9})

    def test_eos_contract_removes_both_overrides(self):
        command = [
            "client", "--ignore-eos", "--", "docker", "-e",
            "TRT_EDGELLM_IGNORE_EOS=1", "-e", "KEEP=1", "image"
        ]
        changed = REPLAY.enable_eos_termination(command)
        self.assertEqual(changed,
                         ["client", "--", "docker", "-e", "KEEP=1", "image"])
        self.assertIn("--ignore-eos", command)
        self.assertEqual(REPLAY.enable_eos_termination(changed), changed)
        with self.assertRaises(ValueError):
            REPLAY.enable_eos_termination(["TRT_EDGELLM_IGNORE_EOS=1"])

    def test_no_slo_contract_removes_only_composition_targets(self):
        command = [
            "docker", "run", "-e", "TRT_EDGELLM_VISION_TTFT_TARGET_MS=500", "-e",
            "TRT_EDGELLM_VISION_DECODE_TPOT_TARGET_MS=80", "-e", "KEEP=1", "image"
        ]

        changed = REPLAY.remove_explicit_slo_contract(command)

        self.assertEqual(changed, ["docker", "run", "-e", "KEEP=1", "image"])
        with self.assertRaises(ValueError):
            REPLAY.remove_explicit_slo_contract(
                ["TRT_EDGELLM_GLOBAL_DECODE_TPOT_TARGET_US=80000"])

    def test_runtime_build_remap_preserves_the_workload_command(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            build = root / "canonical"
            build.mkdir()
            cache = build / "CMakeCache.txt"
            cache.touch()
            command = [
                "docker", "-v", f"{root}:/workspace",
                "EDGELLM_PLUGIN_PATH=/workspace/old/lib.so",
                "LD_LIBRARY_PATH=/workspace/old/examples/llm:/opt/tensorrt/lib",
                "image", "/workspace/old/examples/llm/llm_phase_context_smoke",
                "/workspace/engine",
            ]

            changed = REPLAY.remap_runtime_build(command, cache)

            self.assertIn(
                "EDGELLM_PLUGIN_PATH=/workspace/canonical/lib.so", changed)
            self.assertIn(
                "LD_LIBRARY_PATH=/workspace/canonical/examples/llm:/opt/tensorrt/lib",
                changed)
            self.assertIn(
                "/workspace/canonical/examples/llm/llm_phase_context_smoke",
                changed)
            self.assertEqual(command[-1], changed[-1])

    def test_latency_statistics(self):
        mean, p95 = DISTRIBUTIONS.latency_statistics([1.0, 2.0, 3.0])
        self.assertEqual(mean, 2.0)
        self.assertAlmostEqual(p95, 2.9)
        for values in ([], [float("nan")], [float("inf")]):
            with self.assertRaises(ValueError):
                DISTRIBUTIONS.latency_statistics(values)

    def test_successful_batch(self):
        GUARD.validate_rows([{"http_status": 200, "error": ""}], 1)

    def test_http_failure(self):
        with self.assertRaises(RuntimeError):
            GUARD.validate_rows([{"http_status": 500, "error": "failed"}], 1)

    def test_stream_failure(self):
        with self.assertRaises(RuntimeError):
            GUARD.validate_rows([{
                "http_status": 200,
                "error": "SSE error"
            }], 1)

    def test_missing_response(self):
        with self.assertRaises(RuntimeError):
            GUARD.validate_rows([], 1)

    def test_asset_remap_and_missing_media(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            old = root / "old"
            new = root / "new"
            new.mkdir()
            asset = new / "image.bin"
            asset.write_bytes(b"unchanged fixture")
            trace = root / "trace.json"
            content = {
                "requests": [{
                    "messages": [{
                        "content": [{
                            "type": "image_url",
                            "image_url": {
                                "url": (old / "image.bin").as_uri()
                            }
                        }]
                    }]
                }]
            }
            trace.write_text(json.dumps(content))
            commands = [{"command": ["--trace", str(trace)]}]
            with self.assertRaises(FileNotFoundError):
                REPLAY.materialize_inputs(commands, root / "output", [])
            documents, assets = REPLAY.materialize_inputs(
                commands, root / "output", [(str(old), str(new))])
            self.assertEqual(assets[str(asset)],
                             hashlib.sha256(asset.read_bytes()).hexdigest())
            rewritten = documents[trace][1]["requests"][0]["messages"][0][
                "content"][0]["image_url"]["url"]
            self.assertEqual(rewritten, asset.as_uri())
            self.assertEqual(json.loads(trace.read_text()), content)
            self.assertFalse((root / "output").exists())


if __name__ == "__main__":
    unittest.main()
