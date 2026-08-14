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

import io
import unittest
import unittest.mock
import urllib.error

from scripts.cosmos_reason2 import run_phase_http_trace_bench


class FakeProcess:

    def poll(self) -> None:
        return None


class HealthyResponse:

    status = 200

    def __enter__(self) -> "HealthyResponse":
        return self

    def __exit__(self, *args: object) -> None:
        del args


def http_error(code: int, details: str) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("http://127.0.0.1/health", code, details,
                                  {}, io.BytesIO(details.encode()))


class PhaseHTTPTraceBenchTest(unittest.TestCase):

    def test_wait_for_health_retries_loading(self) -> None:
        responses = iter([http_error(503, "loading"), HealthyResponse()])

        def urlopen(*args: object, **kwargs: object) -> HealthyResponse:
            del args, kwargs
            response = next(responses)
            if isinstance(response, Exception):
                raise response
            return response

        with unittest.mock.patch.object(
                run_phase_http_trace_bench.urllib.request,
                "urlopen",
                side_effect=urlopen), unittest.mock.patch.object(
                    run_phase_http_trace_bench.time,
                    "sleep",
                    return_value=None):
            run_phase_http_trace_bench.wait_for_health(
                "http://127.0.0.1", FakeProcess(), 1.0)

    def test_wait_for_health_reports_backend_failure(self) -> None:

        def urlopen(*args: object, **kwargs: object) -> HealthyResponse:
            del args, kwargs
            raise http_error(500, "backend exited with code 139")

        with unittest.mock.patch.object(
                run_phase_http_trace_bench.urllib.request,
                "urlopen",
                side_effect=urlopen), self.assertRaisesRegex(
                    RuntimeError, "backend exited with code 139"):
            run_phase_http_trace_bench.wait_for_health(
                "http://127.0.0.1", FakeProcess(), 1.0)

    def test_replace_run_formats_a_stable_directory(self) -> None:
        self.assertEqual(
            run_phase_http_trace_bench.replace_run(
                ["--traceCsv", "output/run-{run}/requests.csv"], 7),
            ["--traceCsv", "output/run-007/requests.csv"])


if __name__ == "__main__":
    unittest.main()
