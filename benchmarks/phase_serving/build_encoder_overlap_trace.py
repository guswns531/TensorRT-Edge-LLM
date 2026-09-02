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
"""Build deterministic real-request traces with controlled E/P/D overlap opportunity."""

import argparse
import json
from pathlib import Path
from typing import Any


def text_request(request_class: str, output_tokens: int,
                 arrival_us: int) -> dict[str, Any]:
    return {
        "request_class":
        request_class,
        "messages": [{
            "role":
            "user",
            "content":
            "Explain one practical method for reducing online GPU inference latency while preserving correctness."
        }],
        "max_generate_length":
        output_tokens,
        "arrival_offset_us":
        arrival_us,
    }


def vision_request(image: Path, arrival_us: int) -> dict[str, Any]:
    return {
        "request_class":
        "vision_overlap",
        "messages": [{
            "role":
            "user",
            "content": [{
                "type": "image_url",
                "image_url": {
                    "url": f"file://{image}"
                }
            }, {
                "type": "text",
                "text": "Name the animal in this image in one word."
            }]
        }],
        "max_generate_length":
        1,
        "arrival_offset_us":
        arrival_us,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--mode", choices=("ep", "ed", "mixed"), required=True)
    parser.add_argument("--encoder-requests", type=int, default=8)
    parser.add_argument("--prefill-requests", type=int, default=16)
    parser.add_argument("--decode-requests", type=int, default=8)
    parser.add_argument("--decode-output-tokens", type=int, default=192)
    parser.add_argument("--late-arrival-us", type=int, default=300000)
    parser.add_argument("--phase-order",
                        choices=("phase_first", "encoder_first",
                                 "interleaved"),
                        default="phase_first")
    args = parser.parse_args()
    if (args.encoder_requests <= 0 or args.prefill_requests <= 0
            or args.decode_requests <= 0 or args.decode_output_tokens <= 0
            or args.late_arrival_us < 0):
        parser.error("request counts and token counts must be positive")
    image = args.image.resolve()
    if not image.is_file():
        parser.error(f"image does not exist: {image}")

    phase_requests: list[dict[str, Any]] = []
    if args.mode in {"ed", "mixed"}:
        phase_requests.extend(
            text_request("resident_decode", args.decode_output_tokens, 0)
            for _ in range(args.decode_requests))
    late_arrival_us = 0 if args.mode == "ep" else args.late_arrival_us
    encoder_requests = [
        vision_request(image, late_arrival_us)
        for _ in range(args.encoder_requests)
    ]
    if args.mode in {"ep", "mixed"}:
        phase_requests.extend(
            text_request("late_prefill", 1, late_arrival_us)
            for _ in range(args.prefill_requests))
    if args.phase_order == "interleaved":
        requests = []
        while phase_requests or encoder_requests:
            if phase_requests:
                requests.append(phase_requests.pop(0))
            if encoder_requests:
                requests.append(encoder_requests.pop(0))
    else:
        requests = (encoder_requests + phase_requests if args.phase_order
                    == "encoder_first" else phase_requests + encoder_requests)
    payload = {
        "schema_version": 1,
        "workload": f"cosmos-controlled-{args.mode}-overlap",
        "requests": requests,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n",
                           encoding="utf-8")


if __name__ == "__main__":
    main()
