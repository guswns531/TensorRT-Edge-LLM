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
"""CPU-only same-runtime repeat merging and provenance contracts."""

import importlib.util
import json
import pathlib
import tempfile
import unittest


def load_reporter():
    path = pathlib.Path(__file__).parents[
        2] / "benchmarks/phase_serving/report_workspace_revalidation.py"
    spec = importlib.util.spec_from_file_location("workspace_reporter", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


G_REPORTER = load_reporter()


class WorkspaceReportMergeTest(unittest.TestCase):

    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.roots = [
            pathlib.Path(self.folder.name) / name
            for name in ("first", "additional")
        ]
        self.baselines = {
            "gemma": {
                "mixed": {
                    "metrics": {
                        metric: 10.0
                        for metric in G_REPORTER.METRICS
                    },
                    "trace_sha256": "trace",
                }
            }
        }
        self.write_campaign(self.roots[0], [10])
        self.write_campaign(self.roots[1], [20, 90])

    def write_campaign(self, root, values):
        records = []
        for repeat, value in enumerate(values, 1):
            cell = root / "gemma/independent" / ("repeat-%03d" %
                                                 repeat) / "mixed"
            cell.mkdir(parents=True)
            aggregate = {metric: value for metric in G_REPORTER.METRICS}
            aggregate.update(repeats=1,
                             generated_tokens_per_run_min=4,
                             requested_output_tokens_per_run=4,
                             trace_sha256="trace")
            (cell / "aggregate.json").write_text(json.dumps(aggregate))
            records.append({
                "model":
                "gemma",
                "variant":
                "independent",
                "workload":
                "mixed",
                "repeat":
                repeat,
                "cell":
                str(cell),
                "command": [
                    "python3", "client.py", "--output-dir",
                    str(cell), "--warmup-requests", "49", "--", "docker",
                    "run", "-v",
                    str(cell) + ":/opt/results:rw", "-e", "PROBES=on", "image",
                    "binary"
                ],
                "effective_environment": {
                    "PROBES": "on"
                }
            })
        model = {
            field: field + "-hash"
            for field in ("engine_sha256", "config_sha256", "vision_sha256",
                          "vision_config_sha256", "calibration_sha256")
        }
        model.update(engine_sidecars_sha256={"weights": "hash"},
                     effective_model_config={"slots": 24},
                     engine_builder_config={"batch": 24},
                     vision_builder_config={"batch": 4},
                     traces={"mixed": "trace"})
        identity = {
            field: field + "-hash"
            for field in ("binary_source_commit", "binary_sha256",
                          "plugin_sha256", "container", "runner_sha256")
        }
        identity.update(replay_tools_sha256={"client": "hash"},
                        models={"gemma": model})
        manifest = {
            "identity": identity,
            "commands": records,
            "completed": records,
            "failures": [],
            "calibration": {
                "gemma": 49
            },
            "activation": "generic",
            "byte_budget": 0
        }
        (root / "manifest.json").write_text(json.dumps(manifest))

    def mutate(self, mutation, root_index=1):
        path = self.roots[root_index] / "manifest.json"
        manifest = json.loads(path.read_text())
        mutation(manifest)
        path.write_text(json.dumps(manifest))

    def merge(self):
        return G_REPORTER.collect_merged_campaign(self.roots, self.baselines)

    def test_aggregates_three_raw_runs_not_two_summary_medians(self):
        (self.roots[0] / "summary.json").write_text('{"fake": 99999}')
        merged = self.merge()
        self.assertEqual(merged["retained_cells"], 3)
        self.assertEqual(merged["expected_cells"], 3)
        self.assertTrue(merged["requested_cells_complete"])
        row = merged["rows"][0]
        self.assertEqual(row["run_count"], 3)
        self.assertEqual(row["metrics"]["generated_token_s_median"]["current"],
                         20)
        self.assertEqual(row["metrics"]["e2e_mean_of_run_means_ms"]["current"],
                         40)
        self.assertEqual([origin["repeat"] for origin in row["origins"]], [
            "source-01:first/repeat-001", "source-02:additional/repeat-001",
            "source-02:additional/repeat-002"
        ])
        self.assertEqual(
            [origin["source_repeat"] for origin in row["origins"]],
            ["repeat-001", "repeat-001", "repeat-002"])
        self.assertEqual(len(merged["manifest"]), 2)

    def test_binary_mismatch_is_rejected(self):
        for field in ("binary_sha256", "binary_source_commit", "plugin_sha256",
                      "container", "runner_sha256"):
            with self.subTest(field=field):
                original = (self.roots[1] / "manifest.json").read_text()
                self.mutate(lambda manifest: manifest["identity"].update(
                    {field: "other"}))
                with self.assertRaisesRegex(ValueError,
                                            "runtime identity mismatch"):
                    self.merge()
                (self.roots[1] / "manifest.json").write_text(original)

    def test_source_checkout_metadata_does_not_replace_frozen_binary_identity(
            self):
        self.mutate(lambda manifest: manifest["identity"].update(
            source_commit="docs-only-commit",
            tracked_diff_sha256="other",
            source_status=[" M notes/report.md"]))
        self.assertTrue(self.merge()["requested_cells_complete"])

    def test_model_calibration_and_sidecar_mismatches_are_rejected(self):
        for field in ("engine_sha256", "vision_sha256",
                      "engine_sidecars_sha256", "calibration_sha256"):
            with self.subTest(field=field):
                original = (self.roots[1] / "manifest.json").read_text()
                self.mutate(lambda manifest: manifest["identity"]["models"][
                    "gemma"].update({field: "other"}))
                with self.assertRaisesRegex(ValueError,
                                            "model identity mismatch"):
                    self.merge()
                (self.roots[1] / "manifest.json").write_text(original)

    def test_consistent_command_and_environment_change_is_rejected(self):

        def change(manifest):
            for record in manifest["commands"]:
                record["effective_environment"]["PROBES"] = "off"
                record["command"][record["command"].index(
                    "PROBES=on")] = "PROBES=off"

        self.mutate(change)
        with self.assertRaisesRegex(ValueError, "cell contract mismatch"):
            self.merge()

    def test_command_environment_disagreement_is_rejected(self):
        self.mutate(lambda manifest: manifest["commands"][0][
            "effective_environment"].update(PROBES="off"))
        with self.assertRaisesRegex(ValueError, "environment differs"):
            self.merge()

    def test_warmup_command_change_is_rejected(self):

        def change(manifest):
            command = manifest["commands"][0]["command"]
            command[command.index("--warmup-requests") + 1] = "0"

        self.mutate(change)
        with self.assertRaisesRegex(ValueError, "cell contract mismatch"):
            self.merge()

    def test_trace_mismatch_is_rejected_without_frozen_trace_hash(self):
        del self.baselines["gemma"]["mixed"]["trace_sha256"]
        self.mutate(lambda manifest: manifest["identity"]["models"]["gemma"][
            "traces"].update(mixed="other"))
        with self.assertRaisesRegex(ValueError, "cell contract mismatch"):
            self.merge()

    def test_aggregate_trace_must_match_recorded_workload(self):
        del self.baselines["gemma"]["mixed"]["trace_sha256"]
        path = next(self.roots[1].glob("*/*/repeat-*/*/aggregate.json"))
        aggregate = json.loads(path.read_text())
        aggregate["trace_sha256"] = "wrong"
        path.write_text(json.dumps(aggregate))
        with self.assertRaisesRegex(ValueError, "Aggregate trace differs"):
            self.merge()

    def test_duplicate_root_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "distinct source roots"):
            G_REPORTER.collect_merged_campaign(
                [self.roots[0], self.roots[0] / "."], self.baselines)

    def test_duplicate_source_cell_is_rejected(self):
        self.mutate(lambda manifest: manifest["commands"].append(manifest[
            "commands"][0]))
        with self.assertRaisesRegex(ValueError,
                                    "Duplicate requested source cell"):
            self.merge()

    def test_uncompleted_aggregate_cannot_count_as_success(self):
        self.mutate(lambda manifest: manifest.update(completed=[]))
        merged = self.merge()
        self.assertEqual(merged["retained_cells"], 1)
        self.assertEqual(merged["missing_cells"], 2)
        self.assertFalse(merged["requested_cells_complete"])
        self.assertEqual(
            [item["reason"] for item in merged["excluded_aggregates"]],
            ["not_completed", "not_completed"])

    def test_failed_cell_is_excluded_even_if_marked_complete(self):
        self.mutate(lambda manifest: manifest.update(
            failures=[manifest["commands"][0]]))
        merged = self.merge()
        self.assertEqual(merged["retained_cells"], 2)
        self.assertEqual(merged["failed_cells"], 1)
        self.assertEqual(merged["missing_cells"], 1)
        self.assertFalse(merged["requested_cells_complete"])

    def test_multi_run_aggregate_cannot_be_treated_as_one_observation(self):
        path = next(self.roots[1].glob("*/*/repeat-*/*/aggregate.json"))
        aggregate = json.loads(path.read_text())
        aggregate["repeats"] = 3
        path.write_text(json.dumps(aggregate))
        with self.assertRaisesRegex(ValueError, "exactly one run"):
            self.merge()


if __name__ == "__main__":
    unittest.main()
