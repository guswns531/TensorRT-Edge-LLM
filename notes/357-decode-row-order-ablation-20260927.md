<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# 357. Canonical D row ordering: balanced paired diagnostic

## Ablation

[356](356-greedy-divergence-before-prefill-membership-change-20260927.md)
showed that different D cohort histories coincide with greedy-token divergence,
but did not isolate the physical row order. This diagnostic adds an opt-in
`PhaseDecodeRowOrderMode::kCanonicalEveryDispatch`. The normal mode retains the
previous request-to-row affinity. The diagnostic mode sorts each selected D batch
by the existing `(context bucket, request id)` comparator immediately before
engine staging. It does not alter selected request IDs, stable KV slot IDs, cache
ownership, or TensorRT shapes. The option is exposed to the serving runner as
`--canonical-decode-rows` and is off by default.

## Measurement contract

Gemma 4 E2B AWQ balanced, 64 HTTP requests, 6035 prompt tokens, 5440 fixed output
tokens, RTX 3080 10 GiB, TensorRT 26.06, generic startup service calibration,
independent E/P/D contexts, max 24 stable slots, ordered backend ingress, full
dispatch telemetry. Each row-order mode ran two independent processes. Both
manifests use the same smoke binary SHA256
`c936d9c170c1f6d6d1c0f75d7443338dc16beb278e7e6dfd0c87deff30a78bb5`, engine,
trace SHA256 `fb5bfd84ad542221a0975ef4aeec3dae6fe4f04bcc1ce527a52100b7a0dc012a`,
startup contract and backend request order. The only runner environment difference
is the canonical row-order flag.

- `.local/results/decode-row-order-20260927-affinity/manifest.json`
- `.local/results/decode-row-order-20260927-canonical/manifest.json`
- Paired token/cohort reports: `.local/results/decode-row-order-20260927-canonical/`

All four runs completed 64/64 requests and 5440 tokens. They all peaked at 9393 MiB.
The backend submit indices remained ordered 0→63. Actual scheduler decisions and
GPU timings still varied across processes; this is not a same-snapshot replay.

## Serving results

Values are arithmetic means of two independent processes; TTFT/TPOT/E2E are
HTTP-send based, in milliseconds.

| D row order | tokens/s | TTFT mean / p95 | TPOT mean / p95 | E2E mean / p95 | D dispatches | P dispatches |
|---|---:|---:|---:|---:|---:|---:|
| Retain affinity | 1236.67 | 1190.79 / 2739.70 | 15.60 / 16.87 | 2490.32 / 4019.98 | 294 / 306 | 40 / 41 |
| Canonical each D | 1232.06 | 1198.99 / 2737.13 | 15.65 / 17.14 | 2500.87 / 4037.29 | 308 / 306 | 40 / 39 |

Canonical ordering was 0.37% lower in mean throughput across these two runs. Its
TTFT/E2E tail differences were small and mixed; there is no demonstrated
performance win. The frozen same-trace vLLM result is 771.53 generated tokens/s
(one historical run), so both current modes remain ahead on this balanced cell;
the row-order ablation does not explain or improve that advantage. No new vLLM
run was required because this row-order option does not change its runtime or
workload contract.

## Output identity and why this is not a causal row-order result

Affinity vs canonical output identity was 62/64 requests in repeat 1 and 61/64 in
repeat 2. Repeat 1 diverged at request 28 token 66 and request 43 token 64. Repeat
2 diverged at requests 1 and 13 token 18, plus request 31 token 64. Requests 1 and
13 have identical prompts and produce identical outputs within each process; their
cross-mode mismatch recurred together in repeat 2.

The first D membership difference in repeat 2 was already at the second D
dispatch: affinity selected eight rows while canonical selected four. P row order
first differed at dispatch index 17 and P membership first differed at index 23.
Thus the later
request-1 divergence occurred after distinct decode histories had accumulated.
The row-order mode can perturb completion timing and then affect admission and
cohort formation; these runs do not hold those later states fixed. The experiment
does not show that physical row order alone caused or fixed the token split.

## Decision and next experiment

Keep row-affinity retention as the serving default. The canonical option remains a
diagnostic because it provides a reproducible alternate mechanism, but this run
does not justify promoting it for throughput or output consistency. To isolate
numerics, the next experiment must replay a fixed pre-decode state with identical
request membership, per-request token prefix, stable KV slot/page ownership and
engine graph path, and change only the row permutation. The current asynchronous
HTTP trace does not satisfy that state-equivalence requirement.
