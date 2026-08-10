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
"""Extract teacher-forcing tokens and compare legacy/indexed logits dumps."""

import argparse
import math
import re
from pathlib import Path

import torch
from safetensors import safe_open

ROUND_LOGITS = re.compile(r"^round_(\d+)\.logits$")


def load_dump(path: Path) -> dict[str, torch.Tensor]:
    tensors = {}
    with safe_open(path, framework="pt") as handle:
        for name in handle.keys():
            tensors[name] = handle.get_tensor(name)
    return tensors


def rounds(tensors: dict[str, torch.Tensor]) -> list[int]:
    result = []
    for name in tensors:
        match = ROUND_LOGITS.match(name)
        if match:
            result.append(int(match.group(1)))
    return sorted(result)


def extract_tokens(args: argparse.Namespace) -> int:
    tensors = load_dump(args.dump)
    dump_rounds = rounds(tensors)
    if not dump_rounds or dump_rounds != list(range(len(dump_rounds))):
        raise RuntimeError(f"non-contiguous dump rounds: {dump_rounds}")
    first = tensors["round_0.generated_token_ids"]
    batch_size = first.numel()
    token_rows = [[] for _ in range(batch_size)]
    for round_index in dump_rounds:
        generated = tensors[
            f"round_{round_index}.generated_token_ids"].reshape(-1)
        if generated.numel() != batch_size:
            raise RuntimeError(
                f"round {round_index}: active batch changed during token extraction"
            )
        for row, token in enumerate(generated.tolist()):
            token_rows[row].append(token)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for row in token_rows:
            stream.write(" ".join(str(token) for token in row) + "\n")
    print(
        f"wrote {len(dump_rounds)} forced tokens for {batch_size} rows to {args.output}"
    )
    return 0


def cosine(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    reference = reference.double().flatten()
    candidate = candidate.double().flatten()
    denominator = float(
        torch.linalg.vector_norm(reference) *
        torch.linalg.vector_norm(candidate))
    if denominator == 0.0:
        return 1.0 if torch.equal(reference, candidate) else 0.0
    return float(torch.dot(reference, candidate)) / denominator


def compare(args: argparse.Namespace) -> int:
    reference = load_dump(args.reference)
    candidate = load_dump(args.candidate)
    reference_rounds = rounds(reference)
    candidate_rounds = rounds(candidate)
    if reference_rounds != candidate_rounds or not reference_rounds:
        print(f"FAIL: round mismatch: reference={reference_rounds}, "
              f"candidate={candidate_rounds}")
        return 1

    passed = True
    worst_cosine = math.inf
    largest_absolute_error = 0.0
    greedy_matches = 0
    greedy_total = 0
    for round_index in reference_rounds:
        prefix = f"round_{round_index}"
        reference_lengths = reference[f"{prefix}.context_lengths"]
        candidate_lengths = candidate[f"{prefix}.context_lengths"]
        if not torch.equal(reference_lengths, candidate_lengths):
            print(
                f"FAIL round {round_index}: context lengths "
                f"{reference_lengths.tolist()} != {candidate_lengths.tolist()}"
            )
            passed = False

        reference_logits = reference[f"{prefix}.logits"]
        candidate_logits = candidate[f"{prefix}.logits"]
        if reference_logits.shape != candidate_logits.shape:
            print(f"FAIL round {round_index}: logits shape "
                  f"{tuple(reference_logits.shape)} != "
                  f"{tuple(candidate_logits.shape)}")
            passed = False
            continue

        round_cosines = []
        round_max_abs = 0.0
        for row in range(reference_logits.shape[0]):
            row_reference = reference_logits[row].float()
            row_candidate = candidate_logits[row].float()
            row_cosine = cosine(row_reference, row_candidate)
            row_max_abs = float(
                torch.max(torch.abs(row_reference - row_candidate)))
            round_cosines.append(row_cosine)
            round_max_abs = max(round_max_abs, row_max_abs)
            passed = passed and row_cosine >= args.min_cosine
        worst_cosine = min(worst_cosine, min(round_cosines))
        largest_absolute_error = max(largest_absolute_error, round_max_abs)

        reference_tokens = reference[f"{prefix}.generated_token_ids"].reshape(
            -1)
        candidate_tokens = candidate[f"{prefix}.generated_token_ids"].reshape(
            -1)
        matches = int(torch.sum(reference_tokens == candidate_tokens))
        greedy_matches += matches
        greedy_total += reference_tokens.numel()
        status = "PASS" if min(round_cosines) >= args.min_cosine else "FAIL"
        print(f"{status} round {round_index}: "
              f"min_cosine={min(round_cosines):.8f}, "
              f"max_abs={round_max_abs:.6g}, "
              f"greedy={matches}/{reference_tokens.numel()}")

    print(f"{'PASS' if passed else 'FAIL'}: "
          f"worst_cosine={worst_cosine:.8f} "
          f"(threshold={args.min_cosine}), "
          f"max_abs={largest_absolute_error:.6g}, "
          f"greedy={greedy_matches}/{greedy_total}")
    return 0 if passed else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract_parser = subparsers.add_parser("extract-tokens")
    extract_parser.add_argument("--dump", type=Path, required=True)
    extract_parser.add_argument("--output", type=Path, required=True)
    extract_parser.set_defaults(function=extract_tokens)

    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--reference", type=Path, required=True)
    compare_parser.add_argument("--candidate", type=Path, required=True)
    compare_parser.add_argument("--min-cosine", type=float, default=0.999)
    compare_parser.set_defaults(function=compare)
    args = parser.parse_args()
    return args.function(args)


if __name__ == "__main__":
    raise SystemExit(main())
