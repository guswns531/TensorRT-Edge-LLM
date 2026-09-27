<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# 359. Fixed-KV D24 replay: row permutation alone

## Goal and fixture

The HTTP comparison in [358](358-row-order-http-ablation-confound-20260927.md)
changed ready membership before it reached the token split. This experiment uses
the existing `TRT_EDGELLM_DECODE_EQUAL_WORK_TRIAL` path to execute the same D batch
from the same stored KV state under two physical row permutations.

Gemma AWQ, RTX 3080 10 GiB, TensorRT 26.06, 24 rows, independent TensorRT decode
context, captured CUDA graph. Final repeated result `fixed-kv-row-order-v4` uses
binary SHA256 `3ca18aeceaf76e5f0951e595384bce50c4b00fad8fb71f330df4537221663b13`.
Every row first receives a synthetic, deterministic
128-token prefix through the actual prefill engine. Row contexts span 128, 256,
…, 1024 tokens in eight page-aligned classes, repeated three times. Each row owns
the same stable slot and page IDs in both branches. Its decode token is fixed by
row identity. Before each branch, the logical KV length for every slot is reset to
its prefix length; the decode overwrites position `past`, while all prefix pages
remain unchanged. The active page table presents those same slots in identity or
reversed order, and selected tokens are mapped back to stable row identity.

One pair consists of the same 24 one-token decode operations. No scheduler,
admission, request arrival, prefill membership, E phase, or serving policy runs in
this fixture. It isolates the TensorRT decode row permutation at a fixed pre-state;
it is not an HTTP throughput measurement or a test of cumulative live scheduling.

## Results

Five blocks each contain 20 warmup pairs and 50 measured pairs. The order within
each pair alternates. The run completed 250 measured branch pairs, counted 6000
request-row token comparisons, saw zero greedy-token differences, and had 700 CUDA
graph hits with zero misses. All 24 stable slots were returned after the run.

| Row order | Decode GPU mean / median / p95 (ms) | Host drain mean / median / p95 (ms) |
|---|---:|---:|
| Identity | 13.6727 / 13.6711 / 13.6868 | 14.3982 / 14.3940 / 14.4308 |
| Reversed | 13.6739 / 13.6727 / 13.6886 | 14.4013 / 14.3961 / 14.4308 |
| Reversed vs identity | +0.009% / +0.012% / +0.013% | +0.022% / +0.015% / -0.0002% |

This controlled fixture found neither a token change nor a material timing change
from row permutation alone. It narrows the live HTTP divergence: a different row
order by itself is insufficient to reproduce it under the same prefix state and
request membership. The synthetic token prefixes are not the real balanced prompts,
and this test does not replay the cumulative decode history preceding the live split.
It therefore does not establish bitwise invariance for all model inputs or explain
all cross-policy output differences.

## Artifacts and serving comparison

The diagnostic runner was extended with `--mode row_order`, `--rows`,
`--context-step-tokens` and `--context-buckets`. The existing dense/split default
is unchanged. The first v1 attempt used a stale binary and emitted dense/split
variants despite the row-order config; its manifest is marked failed and excluded.
The v2 and v3 result roots are separate, and v3 is the relevant D24 varied-context
run.

- `.local/results/fixed-kv-row-order-20260927/`: failed stale-binary attempt.
- `.local/results/fixed-kv-row-order-v2-20260927/`: successful D8 homogeneous-context replay.
- `.local/results/fixed-kv-row-order-v3-20260927/`: preliminary D24 varied-context replay before the final validation edit.
- `.local/results/fixed-kv-row-order-v4-20260927/`: final D24, eight-context replay reported above.

The frozen vLLM balanced trace is a serving workload and cannot be compared to this
single-step synthetic replay. No new serving or vLLM throughput claim follows from
this result. Both modes in the earlier HTTP balanced cell were already faster than
the frozen vLLM anchor; the remaining 12-workload gate is still open.

## Next move

Row permutation is no longer the leading explanation for the HTTP token split.
The next correctness replay should use actual tokenized prompts and compare the
same request's decode at the same step after independently building the same
prefill and preceding decode prefix. If those inputs and KV pages match yet logits
differ, inspect TensorRT binding/page-table consistency and GPU numeric behavior.
If logits match, attribute the HTTP split to its earlier action/membership
trajectory. Continue throughput work against the current 12-workload/vLLM gap data;
this isolated experiment does not justify promoting canonical row ordering.
