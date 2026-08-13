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

import runpy
from pathlib import Path

SCRIPT = (Path(__file__).resolve().parents[2] / "scripts" / "cosmos_reason2" /
          "run_real_request_kv_matrix.py")
GLOBALS = runpy.run_path(str(SCRIPT))


def test_workload_preset_selects_validated_prefill_budget():
    resolve = GLOBALS["resolve_prefill_token_budget"]

    assert resolve("short", 0) == 512
    assert resolve("balanced", 0) == 256
    assert resolve("decode-heavy", 0) == 256
    assert resolve("short", 768) == 768
    assert resolve("none", 0) == 0
