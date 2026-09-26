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
import hashlib
import importlib.util
import json
import pathlib
import tempfile
import unittest
import unittest.mock


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
G_REPORTER = load_tool("report_workspace_revalidation")
G_FROZEN_RECOVERY = load_tool("rederive_frozen_vllm")


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

    def test_serving_probe_ablation_keeps_calibration_contract(self):
        default = environment(self.command("independent"))
        disabled = environment(
            G_RUNNER.command_for(self.repo, self.config, self.repo / "cell",
                                 "mixed", "independent", 0,
                                 {"serving_overlap_probes": False}))
        self.assertEqual(
            disabled.pop("TRT_EDGELLM_DISABLE_SERVING_OVERLAP_PROBES"), "1")
        self.assertEqual(default, disabled)

    def test_compact_telemetry_changes_only_instrumentation(self):
        default = environment(self.command("independent"))
        compact = environment(
            G_RUNNER.command_for(self.repo, self.config, self.repo / "cell",
                                 "mixed", "independent", 0,
                                 {"telemetry_level": "dispatch"}))
        self.assertEqual(default.pop("TRT_EDGELLM_PHASE_TELEMETRY_LEVEL"),
                         "full")
        self.assertEqual(compact.pop("TRT_EDGELLM_PHASE_TELEMETRY_LEVEL"),
                         "dispatch")
        self.assertEqual(default, compact)

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

    def test_gemma_defaults_never_select_newer_engine_by_path_existence(self):
        with unittest.mock.patch.dict("os.environ", {}, clear=True):
            with unittest.mock.patch.object(pathlib.Path,
                                            "exists",
                                            return_value=True):
                config = G_RUNNER.model_config(self.repo, "gemma")
        self.assertEqual(config["engine"].name,
                         "engine-packed-p8-d24-kv2048-p192")
        self.assertEqual(config["vision"].parent.name, "visual-e4-soft280")
        self.assertEqual(config["vision_batch_size"], 4)
        self.assertEqual(config["calibration_requests"], 49)

    def test_cosmos_defaults_use_raw_corrected_frozen_baseline(self):
        commands = [{
            "case":
            "mixed",
            "command": [
                "client", "--trace", "mixed.json", "--generic-warmup-trace",
                "generic.json"
            ]
        }]
        with unittest.mock.patch.object(pathlib.Path,
                                        "read_text",
                                        return_value=json.dumps(commands)):
            config = G_RUNNER.model_config(self.repo, "cosmos")
        expected = pathlib.Path(".local/results/review-correction-20260926/"
                                "cosmos-vllm-frozen-raw-corrected.json")
        self.assertEqual(config["frozen_vllm"], self.repo / expected)
        self.assertEqual(G_REPORTER.DEFAULT_COSMOS_VLLM, expected)
        self.assertEqual(config["calibration_requests"], 239)
        self.assertEqual(config["traces"]["mixed"], self.repo / "mixed.json")

    def test_calibration_shapes_preserve_retained_sparse_contract(self):
        self.assertEqual(G_RUNNER.warmup_decode_batches(24),
                         "1,2,4,8,12,16,20,24")
        self.assertEqual(G_RUNNER.warmup_decode_batches(64),
                         "1,2,4,8,12,16,20,24,28,32,36,40,44,48,52,56,60,64")
        with unittest.mock.patch.dict("os.environ", {}, clear=True):
            command = G_RUNNER.command_for(self.repo, self.config,
                                           self.repo / "cell", "mixed",
                                           "independent", 0,
                                           {"cuda_graphs": True})
        self.assertEqual(
            environment(command)["TRT_EDGELLM_IPC_WARMUP_DECODE_BATCHES"],
            "1,2,4,8,12,16,20,24")

    def test_resident_decode_shadow_is_opt_in_and_only_forwards_diagnostics(
            self):
        name = "TRT_EDGELLM_RESIDENT_DECODE_SHADOW"
        with unittest.mock.patch.dict("os.environ", {}, clear=True):
            baseline = environment(self.command("shared_ep"))
            self.assertNotIn(name, baseline)
        for value in ("0", "1"):
            with unittest.mock.patch.dict("os.environ", {name: value},
                                          clear=True):
                command = self.command("shared_ep")
                enabled = environment(command)
            self.assertEqual(enabled.pop(name), value)
            self.assertEqual(enabled, baseline)
            self.assertEqual(
                G_RUNNER.command_environment(command)[name], value)

    def test_predictor_ablation_changes_one_environment_field(self):
        commands = []
        for enabled in ("0", "1"):
            commands.append(
                G_RUNNER.command_for(
                    self.repo, self.config, self.repo / "cell", "mixed",
                    "independent", 0, {
                        "environment": {
                            "TRT_EDGELLM_ENABLE_TRANSITION_PREDICTOR": enabled
                        }
                    }))
        off, on = [environment(command) for command in commands]
        self.assertEqual(off.pop("TRT_EDGELLM_ENABLE_TRANSITION_PREDICTOR"),
                         "0")
        self.assertEqual(on.pop("TRT_EDGELLM_ENABLE_TRANSITION_PREDICTOR"),
                         "1")
        self.assertEqual(off, on)

    def test_graphs_off_cannot_be_overridden_by_inherited_environment(self):
        with unittest.mock.patch.dict("os.environ",
                                      {"ENABLE_CUDA_GRAPHS": "1"}):
            command = G_RUNNER.command_for(self.repo, self.config,
                                           self.repo / "cell", "mixed",
                                           "independent", 0,
                                           {"cuda_graphs": False})
        self.assertNotIn("TRT_EDGELLM_CAPTURE_PHASE_GRAPHS",
                         environment(command))

    def test_contract_rejects_stale_or_unidentified_aggregate(self):
        with tempfile.TemporaryDirectory() as folder:
            cell = pathlib.Path(folder)
            with self.assertRaises(ValueError):
                G_RUNNER.validate_cell_contract(cell, {"binary": "a"}, True)
            G_RUNNER.validate_cell_contract(cell, {"binary": "a"}, False)
            G_RUNNER.validate_cell_contract(cell, {"binary": "a"}, True)
            with self.assertRaises(ValueError):
                G_RUNNER.validate_cell_contract(cell, {"binary": "b"}, True)

    def test_singleton_determinism_flag_is_not_repeatability(self):
        single = {
            "token_trace_deterministic": True,
            "token_trace_sha256_per_run": ["abc"]
        }
        self.assertEqual(
            G_RUNNER.token_repeatability([single])["status"], "not_tested")
        self.assertEqual(
            G_REPORTER.repeatability([single])["status"], "not_tested")
        self.assertEqual(
            G_RUNNER.token_repeatability([single, single])["status"],
            "observed_equal")
        self.assertEqual(
            G_RUNNER.token_repeatability(
                [single, {
                    "token_trace_sha256_per_run": ["xyz"]
                }])["status"], "observed_different")

    def test_campaign_partial_failure_is_nonzero_even_with_some_success(self):
        manifest = {
            "commands": [{
                "cell": "a"
            }, {
                "cell": "b"
            }],
            "completed": [{
                "cell": "a"
            }],
            "failures": [{
                "cell": "b"
            }]
        }
        result = G_RUNNER.campaign_completion(manifest, finished=True)
        self.assertEqual(result["completion_status"], "partial")
        self.assertEqual(result["exit_code"], 1)
        self.assertEqual(result["missing_cells"], 1)
        self.assertEqual(result["failed_cells"], 1)
        manifest["completed"].append({"cell": "b"})
        result = G_RUNNER.campaign_completion(manifest, finished=True)
        self.assertEqual(result["completion_status"], "complete")
        self.assertEqual(result["exit_code"], 0)
        self.assertEqual(result["failed_cells"], 0)

    def test_campaign_unattempted_cells_are_not_success(self):
        result = G_RUNNER.campaign_completion({"commands": [{
            "cell": "a"
        }]},
                                              finished=True)
        self.assertEqual(result["completion_status"], "partial")
        self.assertEqual(result["exit_code"], 1)
        self.assertEqual(result["failed_cells"], 0)

    def test_report_keeps_throughput_and_latency_wins_separate(self):
        baseline = {"metrics": {metric: 10.0 for metric in G_REPORTER.METRICS}}
        run = {metric: 20.0 for metric in G_REPORTER.METRICS}
        row = G_REPORTER.compare_runs([run], baseline)
        row.update(model="test", variant="independent", workload="mixed")
        summary = G_REPORTER.summary_by_model(
            [row])["test/independent"]["metrics"]
        self.assertEqual(summary["generated_token_s_median"]["wins"], 1)
        self.assertEqual(summary["ttft_p95_median_ms"]["wins"], 0)
        self.assertIsNone(row["metrics"]["ttft_p95_median_ms"]["stddev"])
        coverage = G_REPORTER.summary_by_model([row])["test/independent"]
        self.assertFalse(coverage["full12_three_repeat_coverage"])
        self.assertEqual(coverage["minimum_repeats"], 1)
        self.assertEqual(len(coverage["missing_full12_workloads"]), 11)

    def test_report_uses_raw_aggregates_not_existing_summary(self):
        with tempfile.TemporaryDirectory() as folder:
            root = pathlib.Path(folder)
            cell = root / "gemma/shared_ep/repeat-001/mixed"
            cell.mkdir(parents=True)
            aggregate = {metric: 20.0 for metric in G_REPORTER.METRICS}
            aggregate.update(generated_tokens_per_run_min=4,
                             requested_output_tokens_per_run=4,
                             trace_sha256="trace")
            (cell / "aggregate.json").write_text(json.dumps(aggregate))
            (root / "summary.json").write_text('{"fabricated": 9999}')
            baseline = {
                "gemma": {
                    "mixed": {
                        "metrics": {
                            metric: 10.0
                            for metric in G_REPORTER.METRICS
                        },
                        "trace_sha256": "trace"
                    }
                }
            }
            report = G_REPORTER.collect_campaign(root, baseline)
            self.assertEqual(report["retained_cells"], 1)
            self.assertEqual(
                report["rows"][0]["metrics"]["generated_token_s_median"]
                ["current"], 20.0)
            baseline["gemma"]["mixed"]["trace_sha256"] = "another"
            with self.assertRaises(ValueError):
                G_REPORTER.collect_campaign(root, baseline)

    def test_report_aggregation_matches_frozen_metric_names(self):
        baseline = {"metrics": {metric: 10.0 for metric in G_REPORTER.METRICS}}
        runs = [{
            metric: value
            for metric in G_REPORTER.METRICS
        } for value in (10, 20, 90)]
        row = G_REPORTER.compare_runs(runs, baseline)
        self.assertEqual(
            row["metrics"]["ttft_mean_of_run_means_ms"]["current"], 40)
        self.assertEqual(
            row["metrics"]["tpot_mean_of_run_means_ms"]["current"], 40)
        self.assertEqual(row["metrics"]["e2e_mean_of_run_means_ms"]["current"],
                         40)
        self.assertEqual(row["metrics"]["generated_token_s_median"]["current"],
                         20)
        self.assertEqual(row["metrics"]["e2e_p95_median_ms"]["current"], 20)

    def test_failed_aggregate_cannot_make_a_campaign_complete(self):
        with tempfile.TemporaryDirectory() as folder:
            root = pathlib.Path(folder)
            cell = root / "gemma/shared_ep/repeat-001/mixed"
            cell.mkdir(parents=True)
            aggregate = {metric: 20.0 for metric in G_REPORTER.METRICS}
            aggregate.update(generated_tokens_per_run_min=4,
                             requested_output_tokens_per_run=4)
            (cell / "aggregate.json").write_text(json.dumps(aggregate))
            record = {
                "model": "gemma",
                "variant": "shared_ep",
                "repeat": 1,
                "workload": "mixed"
            }
            (root / "manifest.json").write_text(
                json.dumps({
                    "commands": [record],
                    "failures": [record]
                }))
            report = G_REPORTER.collect_campaign(root, {})
            self.assertEqual(report["rows"], [])
            self.assertEqual(report["retained_cells"], 0)
            self.assertFalse(report["requested_cells_complete"])
            self.assertEqual(report["missing_cells"], 1)
            self.assertEqual(report["failed_cells"], 1)
            self.assertEqual(report["excluded_aggregates"][0]["reason"],
                             "failed_cell")
            self.assertIn("not a complete campaign",
                          G_REPORTER.markdown({"campaigns": {
                              "test": report
                          }}))

    def test_frozen_missing_trace_hash_is_prominently_disclosed(self):
        baseline = {"metrics": {metric: 10.0 for metric in G_REPORTER.METRICS}}
        row = G_REPORTER.compare_runs([{
            metric: 10.0
            for metric in G_REPORTER.METRICS
        }], baseline)
        row.update(model="cosmos",
                   variant="shared_ep",
                   workload="mixed",
                   baseline_trace_identity="not_available_in_frozen_summary")
        report = {"rows": [row], "retained_cells": 1, "expected_cells": 1}
        self.assertIn("cannot verify byte-identical traces",
                      G_REPORTER.markdown({"campaigns": {
                          "test": report
                      }}))

    def test_historical_frozen_schema_remains_readable(self):
        with tempfile.TemporaryDirectory() as folder:
            path = pathlib.Path(folder) / "historical.json"
            row = {"vllm_" + metric: 10.0 for metric in G_REPORTER.METRICS}
            row["workload"] = "mixed"
            path.write_text(json.dumps({"rows": [row]}))
            loaded = G_REPORTER.load_frozen(path)["mixed"]
        self.assertEqual(loaded["metrics"],
                         {metric: 10.0
                          for metric in G_REPORTER.METRICS})
        self.assertIsNone(loaded["repeat_count"])
        self.assertIsNone(loaded["trace_sha256"])
        self.assertEqual(loaded["raw_origins"], [])

    def test_frozen_raw_recovery_checks_hashes_coverage_and_mean_contract(
            self):
        with tempfile.TemporaryDirectory() as folder:
            root = pathlib.Path(folder)
            trace = root / "trace.json"
            trace.write_text("[]\n")
            trace_hash = hashlib.sha256(trace.read_bytes()).hexdigest()
            commands = root / "commands.json"
            commands.write_text(
                json.dumps([{
                    "case": "mixed",
                    "command": ["client", "--trace",
                                str(trace)]
                }]))
            summary = root / "historical.json"
            summary.write_text(
                json.dumps({
                    "rows": [{
                        "workload": "mixed",
                        "success_runs": 3,
                        "attempted_runs": 4,
                        "failed_runs": 1
                    }]
                }))
            raw_root = root / "raw"
            for index, value in enumerate((10, 20, 90)):
                path = raw_root / "mixed" / str(
                    index) / "client" / "aggregate.json"
                path.parent.mkdir(parents=True)
                data = {metric: value for metric in G_FROZEN_RECOVERY.METRICS}
                data.update(repeats=1, trace_sha256=trace_hash)
                path.write_text(json.dumps(data))
            result = G_FROZEN_RECOVERY.reconstruct(summary, [raw_root],
                                                   commands)
            row = result["rows"][0]
            self.assertEqual(row["vllm_ttft_mean_of_run_means_ms"], 40)
            self.assertEqual(row["vllm_ttft_p95_median_ms"], 20)
            self.assertEqual(row["vllm_generated_token_s_median"], 20)
            self.assertEqual(row["failed_runs"], 1)
            self.assertEqual(len(row["raw_origins"]), 3)
            corrected = root / "corrected.json"
            corrected.write_text(json.dumps(result))
            loaded = G_REPORTER.load_frozen(corrected)["mixed"]
            self.assertEqual(loaded["trace_sha256"], trace_hash)
            self.assertEqual(loaded["repeat_count"], 3)
            self.assertEqual(len(loaded["raw_origins"]), 3)
            with self.assertRaisesRegex(ValueError, "Repeated raw baseline"):
                G_FROZEN_RECOVERY.reconstruct(summary, [raw_root, raw_root],
                                              commands)
            trace.write_text("[1]\n")
            with self.assertRaisesRegex(ValueError, "workload hash differs"):
                G_FROZEN_RECOVERY.reconstruct(summary, [raw_root], commands)
            trace.write_text("[]\n")
            path.unlink()
            with self.assertRaisesRegex(ValueError, "success count differs"):
                G_FROZEN_RECOVERY.reconstruct(summary, [raw_root], commands)

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
