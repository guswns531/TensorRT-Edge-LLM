# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""CPU-only contracts for paired lifetime-admission HTTP experiments."""

import gzip
import importlib.util
import json
import pathlib
import tempfile
import unittest


def load_tool(name):
    """Load a standalone benchmark without importing model dependencies."""
    path = pathlib.Path(__file__).parents[2] / "benchmarks/phase_serving" / (
        name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


G_RUNNER = load_tool("run_lifetime_encoded_admission")
G_ANALYZER = load_tool("analyze_lifetime_encoded_admission")
G_SERVICE_ANALYZER = load_tool("analyze_decode_service_admission")


def environment(command):
    """Extract exact Docker environment values from a backend command."""
    return {
        command[index + 1].split("=", 1)[0]: command[index + 1].split("=",
                                                                      1)[1]
        for index, value in enumerate(command) if value == "-e"
    }


class LifetimeEncodedAdmissionContractTest(unittest.TestCase):

    def setUp(self):
        self.repo = pathlib.Path("/tmp/lifetime-test-repo")
        self.config = {
            "model": "test-model",
            "engine": self.repo / "engine",
            "vision": self.repo / "vision",
            "hf": self.repo / "hf",
            "traces": {
                "mixed": self.repo / "mixed.json"
            },
            "calibration": self.repo / "generic.json",
            "calibration_requests": 49,
            "stable_slots": 24,
            "in_flight": 24,
            "decode_batch": 24,
            "encoder_input_tokens": 1120,
            "initial_capacity": 4,
            "larger_capacity": 12
        }

    def command(self, variant, budget=0):
        return G_RUNNER.command_for(self.repo, self.config, self.repo / "cell",
                                    "mixed", variant, budget)

    def test_variants_have_identical_startup_except_boundary_activation(self):
        variants = [
            environment(self.command(v))
            for v in ("static-base", "static-large", "lifetime", "ownership")
        ]
        for variant in variants:
            self.assertEqual(variant["TRT_EDGELLM_MAX_ENCODED_VISION"], "4")
            self.assertNotIn("TRT_EDGELLM_LIFETIME_ENCODED_ADMISSION", variant)
            variant.pop("TRT_EDGELLM_MEASUREMENT_ENCODED_ADMISSION")
            variant.pop("TRT_EDGELLM_MEASUREMENT_ENCODED_CAPACITY")
        self.assertEqual(variants[0], variants[1])
        self.assertEqual(variants[1], variants[2])
        self.assertEqual(variants[2], variants[3])

    def test_lifetime_is_activated_only_at_measurement(self):
        values = environment(self.command("lifetime"))
        self.assertEqual(values["TRT_EDGELLM_MEASUREMENT_ENCODED_ADMISSION"],
                         "lifetime")
        self.assertNotIn("TRT_EDGELLM_MAX_ENCODED_VISION_BYTES", values)
        self.assertEqual(values["TRT_EDGELLM_PHASE_POLICY"],
                         "service-scaled-transition")

    def test_chunking_changes_only_after_common_calibration(self):
        lifetime = environment(self.command("lifetime"))
        chunked = environment(self.command("chunked"))
        self.assertEqual(
            chunked.pop("TRT_EDGELLM_MEASUREMENT_CHUNKED_VISION_PREFILL"), "1")
        self.assertEqual(lifetime, chunked)

    def test_small_encoder_preserves_engine_and_calibration_capabilities(self):
        lifetime = environment(self.command("lifetime"))
        for variant, rows in (("e1", "1"), ("e2", "2")):
            values = environment(self.command(variant))
            self.assertEqual(
                values.pop("TRT_EDGELLM_MEASUREMENT_ENCODER_PREPARATION_ROWS"),
                rows)
            self.assertEqual(values, lifetime)
            self.assertEqual(values["TRT_EDGELLM_VISION_ENCODER_BATCH_SIZE"],
                             "4")

    def test_dynamic_encoder_policy_changes_only_post_calibration_authority(
            self):
        lifetime = environment(self.command("lifetime"))
        for variant, mode in (("e-dynamic-shadow", "shadow"),
                              ("e-transition-shadow", "transition-shadow"),
                              ("e-dynamic-active", "active")):
            values = environment(self.command(variant))
            self.assertEqual(
                values.pop(
                    "TRT_EDGELLM_MEASUREMENT_ENCODER_PREPARATION_POLICY"),
                mode)
            self.assertEqual(values, lifetime)
            self.assertNotIn(
                "TRT_EDGELLM_MEASUREMENT_ENCODER_PREPARATION_ROWS", values)

    def test_static_large_does_not_change_encoder_batch_or_warmup_window(self):
        values = environment(self.command("static-large"))
        self.assertEqual(values["TRT_EDGELLM_MAX_ENCODED_VISION"], "4")
        self.assertEqual(values["TRT_EDGELLM_MEASUREMENT_ENCODED_CAPACITY"],
                         "12")
        self.assertEqual(values["TRT_EDGELLM_VISION_ENCODER_BATCH_SIZE"], "4")

    def test_byte_ceiling_is_common_to_all_variants(self):
        for variant in ("static-base", "static-large", "lifetime"):
            self.assertEqual(
                environment(self.command(
                    variant, 1000))["TRT_EDGELLM_MAX_ENCODED_VISION_BYTES"],
                "1000")

    def test_static_slot_comparator_uses_engine_capability_without_changing_calibration(
            self):
        values = environment(self.command("static-slot"))
        self.assertEqual(values["TRT_EDGELLM_MEASUREMENT_ENCODED_CAPACITY"],
                         str(self.config["stable_slots"]))
        self.assertEqual(values["TRT_EDGELLM_MAX_ENCODED_VISION"], "4")
        self.assertEqual(values["TRT_EDGELLM_MEASUREMENT_ENCODED_ADMISSION"],
                         "static")

    def test_runtime_is_network_free_and_has_no_added_slo_contract(self):
        command = self.command("lifetime")
        self.assertEqual(command[command.index("--network") + 1], "none")
        self.assertIn("--read-only", command)
        self.assertFalse(
            any("TPOT_TARGET" in value or "TTFT_TARGET" in value
                for value in command))

    def test_description_does_not_require_an_exact_key_or_model_dependency(
            self):
        self.assertEqual(G_ANALYZER.distribution([]), {"samples": 0})
        self.assertAlmostEqual(G_ANALYZER.distribution([2, 1])["p95"], 1.95)
        self.assertEqual(G_ANALYZER.distribution([3])["mean"], 3)

    def test_encoder_overlap_counts_copy_and_triple_intervals_once(self):
        result = G_ANALYZER.encoder_overlap_percent({
            "0000": 50,
            "0001": 10,
            "0011": 10,
            "1101": 10,
            "1111": 20
        })
        self.assertEqual(result["e_active_epoch_percent"], 50)
        self.assertEqual(result["ep_epoch_percent"], 30)
        self.assertEqual(result["ed_epoch_percent"], 30)
        self.assertEqual(result["e_overlapped_fraction_percent"], 80)
        self.assertEqual(
            G_ANALYZER.encoder_overlap_percent(
                {})["e_overlapped_fraction_percent"], 0)

    def test_compressed_log_reads_the_same_raw_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            path = pathlib.Path(folder) / "gateway.log"
            with gzip.open(str(path) + ".gz", "wt") as stream:
                stream.write("PHASE_EPOCH\t{\"epoch\":1}\n")
            with G_ANALYZER.log_stream(path) as stream:
                self.assertEqual(stream.read(), "PHASE_EPOCH\t{\"epoch\":1}\n")

    def test_decode_cycle_attribution_matches_request_and_measurement_epoch(
            self):
        with tempfile.TemporaryDirectory() as folder:
            path = pathlib.Path(folder) / "gateway.log"
            lines = ["PHASE_EPOCH\t{\"epoch\":1}\n", "PHASE_METRIC\t{}\n"]
            for stage, timestamp in (("vision_queued",
                                      0), ("decode_ready",
                                           1000), ("decode_start", 4000),
                                     ("decode_done",
                                      9000), ("decode_sampling_submit", 9001),
                                     ("decode_sampling_ready",
                                      9500), ("decode_sampling_collected",
                                              9600), ("decode_token_committed",
                                                      10000)):
                lines.append("PHASE_TIMELINE\t" +
                             json.dumps({
                                 "request_index": 1,
                                 "stage": stage,
                                 "timestamp_us": timestamp
                             }) + "\n")
            path.write_text("".join(lines))
            result = G_SERVICE_ANALYZER.analyze(path)
            components = result["per_request_mean_components"]["vision"]
            self.assertEqual(components["ready_wait_ms"]["mean"], 3)
            self.assertEqual(components["host_execution_ms"]["mean"], 5)
            self.assertEqual(components["completion_to_commit_ms"]["mean"], 1)
            self.assertEqual(
                components["done_to_sampling_observed_ms"]["mean"], 0.5)
            self.assertEqual(components["done_to_sampling_submit_ms"]["mean"],
                             0.001)
            self.assertEqual(components["sampling_submit_to_ready_ms"]["mean"],
                             0.499)
            self.assertEqual(components["sampling_collect_ms"]["mean"], 0.1)
            self.assertEqual(components["collect_to_commit_ms"]["mean"], 0.4)
            self.assertEqual(result["unmatched_stages"], {})
            self.assertEqual(result["sampled_decode_age_quanta"]["count"], 0)
            self.assertIsNone(result["sampled_decode_age_quanta"]["mean"])
            self.assertEqual(result["outstanding_at_end"], {
                "ready": 0,
                "running": 0,
                "sampling": 0,
                "sampling_submitted": 0
            })

    def test_explicit_replacement_preserves_origins_and_excludes_only_primary_key(
            self):

        def report(value):
            return {
                "serving": {
                    metric: value
                    for metric in G_ANALYZER.SERVING_METRICS
                },
                "aggregate_sha256": str(value)
            }

        key = "test/work/static-base/3"
        primary = {
            "test/work/static-base/1": report(10),
            "test/work/static-base/2": report(20),
            key: report(100),
            "test/work/lifetime/1": report(25)
        }
        supplemental = {key: report(30)}
        rows = G_ANALYZER.summarize_reports(
            [(pathlib.Path("/tmp/primary"), primary),
             (pathlib.Path("/tmp/supplemental"), supplemental)], [key])
        base = rows["test/work/static-base"]
        metric = "generated_token_s_median"
        self.assertEqual(base["success_runs"], 3)
        self.assertEqual(base["metrics"][metric]["mean"], 20)
        self.assertEqual(base["origins"][-1]["root"], "/tmp/supplemental")
        self.assertAlmostEqual(
            rows["test/work/lifetime"]["delta_percent"]["static-base"][metric],
            25)
        self.assertAlmostEqual(base["delta_percent"]["lifetime"][metric], -20)


if __name__ == "__main__":
    unittest.main()
