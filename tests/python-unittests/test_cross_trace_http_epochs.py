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

from benchmarks.phase_serving.run_cross_trace_http_epochs import (
    parse_epoch, replace_placeholders)


def test_parses_named_epoch() -> None:
    name, path = parse_epoch("vision-heavy=/tmp/trace.json")
    assert name == "vision-heavy"
    assert str(path) == "/tmp/trace.json"


def test_resolves_persistent_backend_placeholders() -> None:
    assert replace_placeholders(
        ["--run={run}", "--warmup={policy_warmup_mode}"],
        "generic") == ["--run=001", "--warmup=generic"]
