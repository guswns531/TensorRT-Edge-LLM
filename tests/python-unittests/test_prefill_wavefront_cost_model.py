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
          "build_prefill_wavefront_cost_model.py")
GLOBALS = runpy.run_path(str(SCRIPT))


def test_decode_cost_prefers_isolated_samples():
    samples, scope = GLOBALS["select_decode_cost_samples"]([6.0, 6.1, 20.0],
                                                           [6.0, 6.1])

    assert samples == [6.0, 6.1]
    assert scope == "decode_only"


def test_decode_cost_falls_back_when_isolated_shape_is_missing():
    samples, scope = GLOBALS["select_decode_cost_samples"]([7.0, 7.1], [])

    assert samples == [7.0, 7.1]
    assert scope == "all_dispatch_fallback"
