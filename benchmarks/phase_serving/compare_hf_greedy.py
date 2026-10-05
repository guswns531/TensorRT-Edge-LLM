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
"""Greedy output gate: HF transformers reference versus an OpenAI-compatible server.

``reference`` generates greedy continuations with transformers; ``serve`` queries a server with
the same messages; ``score`` reports exact matches and the common token-prefix length.
"""

import argparse
import json
import pathlib

PICS = pathlib.Path(__file__).resolve().parents[2] / "examples/multimodal/pics"

PROMPTS = [
    "What is the capital of France? Answer in one sentence.",
    "Write a short haiku about GPUs.",
    "Explain in two sentences why the sky is blue.",
    "List three prime numbers greater than 50.",
    "Translate 'good morning, how are you?' into Korean.",
    "What is 17 multiplied by 23? Show the result only.",
]
IMAGE_PROMPTS = [("giant_panda.jpeg", "Describe this image in one sentence."),
                 ("database_er.jpeg", "What kind of diagram is this?"),
                 ("woman_and_dog.jpeg",
                  "How many living beings are in this picture?")]


def cases():
    result = [{
        "messages": [{
            "role": "user",
            "content": text
        }]
    } for text in PROMPTS]
    for image, text in IMAGE_PROMPTS:
        result.append({
            "messages": [{
                "role":
                "user",
                "content": [{
                    "type": "image_url",
                    "image_url": {
                        "url": "file://" + str(PICS / image)
                    }
                }, {
                    "type": "text",
                    "text": text
                }]
            }]
        })
    return result


def to_hf(messages):
    converted = []
    for message in messages:
        content = message["content"]
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        parts = []
        for part in content:
            if part["type"] == "image_url":
                parts.append({
                    "type": "image",
                    "url": part["image_url"]["url"][len("file://"):]
                })
            else:
                parts.append(part)
        converted.append({"role": message["role"], "content": parts})
    return converted


def reference(args):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(args.hf_dir)
    model = AutoModelForImageTextToText.from_pretrained(
        args.hf_dir, dtype=getattr(torch,
                                   args.dtype), device_map="cuda").eval()
    outputs = []
    for case in cases():
        inputs = processor.apply_chat_template(to_hf(case["messages"]),
                                               add_generation_prompt=True,
                                               tokenize=True,
                                               return_dict=True,
                                               return_tensors="pt").to("cuda")
        with torch.inference_mode():
            generated = model.generate(**inputs,
                                       max_new_tokens=args.max_tokens,
                                       do_sample=False)
        tokens = generated[0, inputs["input_ids"].shape[1]:].tolist()
        outputs.append({
            "token_ids":
            tokens,
            "text":
            processor.decode(tokens, skip_special_tokens=True)
        })
        print(json.dumps(outputs[-1]["text"])[:160], flush=True)
    args.output.write_text(
        json.dumps({
            "dtype": args.dtype,
            "outputs": outputs
        }, indent=1) + "\n")


def serve(args):
    from concurrent.futures import ThreadPoolExecutor

    import httpx

    def query(case):
        response = httpx.post(args.endpoint + "/v1/chat/completions",
                              json={
                                  "model": args.model,
                                  "messages": case["messages"],
                                  "max_tokens": args.max_tokens,
                                  "temperature": 0.0
                              },
                              timeout=600)
        response.raise_for_status()
        return {"text": response.json()["choices"][0]["message"]["content"]}

    # Concurrent submission makes the server batch prompts of different lengths together.
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        outputs = list(pool.map(query, cases()))
    for output in outputs:
        print(json.dumps(output["text"])[:160], flush=True)
    args.output.write_text(json.dumps({"outputs": outputs}, indent=1) + "\n")


def score(args):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.hf_dir)
    reference_outputs = json.loads(args.reference.read_text())["outputs"]
    rows = []
    for candidate_path in args.candidates:
        candidate = json.loads(candidate_path.read_text())["outputs"]
        exact = 0
        prefixes = []
        for ref, got in zip(reference_outputs, candidate):
            ref_ids = tokenizer(ref["text"],
                                add_special_tokens=False)["input_ids"]
            got_text = got["text"]
            # Fixed-output servers continue past EOS; compare only the reference span.
            got_ids = tokenizer(
                got_text, add_special_tokens=False)["input_ids"][:len(ref_ids)]
            common = 0
            for a, b in zip(ref_ids, got_ids):
                if a != b:
                    break
                common += 1
            exact += int(common == len(ref_ids))
            prefixes.append(common / max(1, len(ref_ids)))
        rows.append({
            "candidate":
            str(candidate_path),
            "exact_cases":
            exact,
            "cases":
            len(reference_outputs),
            "mean_common_prefix_fraction":
            sum(prefixes) / len(prefixes),
            "per_case_prefix_fraction":
            [round(value, 3) for value in prefixes]
        })
    print(json.dumps(rows, indent=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    ref = sub.add_parser("reference")
    ref.add_argument("--hf-dir", required=True)
    ref.add_argument("--dtype", default="bfloat16")
    ref.add_argument("--max-tokens", type=int, default=48)
    ref.add_argument("--output", type=pathlib.Path, required=True)
    srv = sub.add_parser("serve")
    srv.add_argument("--endpoint", default="http://127.0.0.1:8001")
    srv.add_argument("--model", required=True)
    srv.add_argument("--max-tokens", type=int, default=48)
    srv.add_argument("--concurrency", type=int, default=1)
    srv.add_argument("--output", type=pathlib.Path, required=True)
    sc = sub.add_parser("score")
    sc.add_argument("--hf-dir", required=True)
    sc.add_argument("--reference", type=pathlib.Path, required=True)
    sc.add_argument("candidates", type=pathlib.Path, nargs="+")
    args = parser.parse_args()
    {"reference": reference, "serve": serve, "score": score}[args.mode](args)


if __name__ == "__main__":
    main()
