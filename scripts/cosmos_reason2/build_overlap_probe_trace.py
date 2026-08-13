#!/usr/bin/env python3
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
"""Build a controlled late-prefill trace for direct overlap cost coverage."""

import argparse
import json
from pathlib import Path


def request(prompt: str, output_tokens: int, arrival_offset_us: int) -> dict:
    return {
        "messages": [{
            "role": "user",
            "content": prompt
        }],
        "max_generate_length": output_tokens,
        "arrival_offset_us": arrival_offset_us,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--decode-requests", type=int, default=64)
    parser.add_argument("--probe-requests", type=int, default=16)
    parser.add_argument("--decode-output-tokens", type=int, default=256)
    parser.add_argument("--probe-output-tokens", type=int, default=8)
    parser.add_argument("--probe-arrival-ms", type=float, default=1000.0)
    parser.add_argument("--probe-prompt-repeats", type=int, default=96)
    args = parser.parse_args()
    if (args.decode_requests <= 0 or args.probe_requests <= 0
            or args.decode_output_tokens <= 0 or args.probe_output_tokens <= 0
            or args.probe_arrival_ms < 0.0 or args.probe_prompt_repeats <= 0):
        parser.error(
            "request counts, token lengths, and prompt repeats must be positive"
        )

    decode_prompt = "Give one practical tip for reducing online inference latency."
    probe_prompt = "Explain one inference scheduling tradeoff clearly. " * args.probe_prompt_repeats
    requests = [
        request(decode_prompt, args.decode_output_tokens, 0)
        for _ in range(args.decode_requests)
    ]
    requests.extend(
        request(probe_prompt, args.probe_output_tokens,
                round(args.probe_arrival_ms * 1000.0))
        for _ in range(args.probe_requests))
    root = {
        "requests":
        requests,
        "max_generate_length":
        max(args.decode_output_tokens, args.probe_output_tokens),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(root, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
