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
"""Materialize the 12 serving workloads (EXPERIMENT_METHODOLOGY.md section 5) for one tokenizer.

Request mixes, output-length distributions and arrival shapes follow the retained Gemma suite;
prompt lengths are measured with the target tokenizer. Every prompt is unique so no server can
benefit from prefix reuse. Images are ``file://`` URLs of the repository's example pictures.
"""

import argparse
import hashlib
import json
import pathlib
import random

WORDS = (
    "scheduler kernel tensor memory batch decode prefill latency throughput engine vision token "
    "attention cache page stream graph capture warp block thread register occupancy bandwidth "
    "compute model layer weight activation precision gradient sample policy request queue "
    "admission phase encoder image pixel patch window sliding global local head dimension "
    "vector matrix multiply reduce softmax norm residual embedding vocabulary logit sequence"
).split()


class PromptFactory:

    def __init__(self, tokenizer, seed):
        self.tokenizer = tokenizer
        self.random = random.Random(seed)
        self.serial = 0

    def text(self, tokens):
        self.serial += 1
        header = "Request %d. Summarize the following notes in plain language. " % self.serial
        words = [self.random.choice(WORDS) for _ in range(tokens)]
        ids = self.tokenizer(header + " ".join(words),
                             add_special_tokens=False)["input_ids"][:tokens]
        return self.tokenizer.decode(ids)


def text_request(factory, rng, prompt_range, output, arrival_us, label):
    return {
        "messages": [{
            "role": "user",
            "content": factory.text(rng.randint(*prompt_range))
        }],
        "max_generate_length":
        output,
        "arrival_offset_us":
        int(arrival_us),
        "request_class":
        label,
    }


def vision_request(factory, rng, images, image_count, output, arrival_us,
                   label):
    content = [{
        "type": "image_url",
        "image_url": {
            "url": rng.choice(images)
        }
    } for _ in range(image_count)]
    content.append({
        "type":
        "text",
        "text":
        factory.text(rng.randint(24, 64)) + " Describe the image briefly."
    })
    return {
        "messages": [{
            "role": "user",
            "content": content
        }],
        "max_generate_length": output,
        "arrival_offset_us": int(arrival_us),
        "request_class": label,
    }


def burst(count, span_us):
    return [span_us * index / max(1, count - 1) for index in range(count)]


def mixed_suite(factory,
                rng,
                images,
                text,
                single,
                double,
                outputs,
                arrivals,
                prompt_range=(256, 768)):
    kinds = ["text"] * text + ["single"] * single + ["double"] * double
    rng.shuffle(kinds)
    requests = []
    for kind, arrival in zip(kinds, arrivals):
        output = rng.choice(outputs)
        if kind == "text":
            requests.append(
                text_request(factory, rng, prompt_range, output, arrival,
                             "text"))
        else:
            requests.append(
                vision_request(factory, rng, images,
                               1 if kind == "single" else 2, output, arrival,
                               "vision"))
    return requests


def build(tokenizer, images, seed, bulk_requests=64, mix_scale=1):
    suites = {}
    for name in ("short", "balanced", "long-prefill", "bimodal",
                 "decode-heavy", "text-heavy", "mixed", "vision-heavy",
                 "multi-image", "poisson", "wave-drain", "late-vision"):
        rng = random.Random("%s-%d" % (name, seed))
        factory = PromptFactory(tokenizer, "%s-prompt-%d" % (name, seed))
        if name == "short":
            requests = [
                text_request(factory, rng, (64, 192),
                             rng.choice((8, 16, 24, 24, 32)), t, "text")
                for t in burst(48, 30_000)
            ]
        elif name in ("balanced", "long-prefill"):
            prompts = (256, 768) if name == "balanced" else (1024, 1536)
            requests = [
                text_request(factory, rng, prompts,
                             rng.choice((32, 64, 96, 96, 128, 128)), t, "text")
                for t in burst(bulk_requests, 100_000 * bulk_requests // 64)
            ]
        elif name == "bimodal":
            requests = [
                text_request(
                    factory, rng, (128, 512),
                    rng.choice((16, 24, 32)) if index % 2 else rng.choice(
                        (192, 256, 320, 384)), t, "text")
                for index, t in enumerate(
                    burst(bulk_requests, 100_000 * bulk_requests // 64))
            ]
        elif name == "decode-heavy":
            requests = [
                text_request(factory, rng, (128, 384),
                             rng.choice((96, 192, 288, 288, 384)), t, "text")
                for t in burst(bulk_requests, 100_000 * bulk_requests // 64)
            ]
        elif name == "text-heavy":
            requests = mixed_suite(factory, rng, images, 48 * mix_scale,
                                   13 * mix_scale, 3 * mix_scale, (32, 64, 64),
                                   burst(64 * mix_scale, 100_000 * mix_scale))
        elif name == "mixed":
            requests = mixed_suite(factory, rng, images, 32 * mix_scale,
                                   26 * mix_scale, 6 * mix_scale,
                                   (32, 32, 48, 64),
                                   burst(64 * mix_scale, 100_000 * mix_scale))
        elif name == "vision-heavy":
            requests = mixed_suite(factory, rng, images, 16 * mix_scale,
                                   39 * mix_scale, 9 * mix_scale,
                                   (32, 32, 32, 48, 64),
                                   burst(64 * mix_scale, 100_000 * mix_scale))
        elif name == "multi-image":
            arrivals = [
                wave * 200_000 + slot * 1_000 for wave in range(4)
                for slot in range(5)
            ]
            requests = mixed_suite(factory, rng, images, 0, 16, 4, (32, ),
                                   arrivals)
        elif name == "poisson":
            arrivals, now = [], 0.0
            for _ in range(64 * mix_scale):
                arrivals.append(now)
                now += rng.expovariate(16.0 * mix_scale) * 1e6
            requests = mixed_suite(factory, rng, images, 48 * mix_scale,
                                   13 * mix_scale, 3 * mix_scale,
                                   (32, 64, 64, 128), arrivals)
        elif name == "wave-drain":
            arrivals = [
                wave * 1_500_000 + slot * 1_000 for wave in range(5)
                for slot in range(4)
            ]
            requests = mixed_suite(factory, rng, images, 0, 16, 4, (32, ),
                                   arrivals)
        else:
            requests = [
                text_request(factory, rng, (256, 512), 192, t, "text")
                for t in burst(24, 20_000)
            ]
            requests += [
                vision_request(factory, rng, images, 1, 1, 500_000 + t,
                               "vision") for t in burst(8, 20_000)
            ]
        requests.sort(key=lambda request: request["arrival_offset_us"])
        suites[name] = {"workload": name, "seed": seed, "requests": requests}
    return suites


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer",
                        required=True,
                        help="HF checkpoint directory of the served model")
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--seed", type=int, default=20261005)
    parser.add_argument(
        "--bulk-requests",
        type=int,
        default=64,
        help=
        "Requests in balanced, long-prefill, bimodal and decode-heavy (288 for D64 engines)"
    )
    parser.add_argument(
        "--mix-scale",
        type=int,
        default=1,
        help=
        "Request-count multiplier for text-heavy, mixed, vision-heavy and poisson"
    )
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    pics = pathlib.Path(
        __file__).resolve().parents[2] / "examples/multimodal/pics"
    images = ["file://" + str(path) for path in sorted(pics.glob("*.jpeg"))]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "bulk_requests": args.bulk_requests,
        "mix_scale": args.mix_scale,
        "seed": args.seed,
        "tokenizer": args.tokenizer,
        "images": images,
        "workloads": {}
    }
    for name, suite in build(tokenizer, images, args.seed, args.bulk_requests,
                             args.mix_scale).items():
        path = args.output_dir / (name + ".json")
        path.write_text(json.dumps(suite, indent=1) + "\n")
        requests = suite["requests"]
        outputs = sorted(request["max_generate_length"]
                         for request in requests)
        manifest["workloads"][name] = {
            "sha256":
            hashlib.sha256(path.read_bytes()).hexdigest(),
            "requests":
            len(requests),
            "vision_requests":
            sum(request["request_class"] == "vision" for request in requests),
            "output_tokens":
            [outputs[0], outputs[len(outputs) // 2], outputs[-1]],
            "span_s":
            requests[-1]["arrival_offset_us"] / 1e6,
        }
    (args.output_dir /
     "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest["workloads"], indent=1))


if __name__ == "__main__":
    main()
