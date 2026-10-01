---
id: W013
status: active
updated: 2026-10-01
notes: [356, 357, 358, 359, 368]
---

# Gemma batch-shape output divergence

## Goal
Explain why Gemma 4 E2B serving output differs from its batch-1 reference (and between serving runs) while Cosmos is bit-exact, and define a correctness gate that separates scheduler bugs from kernel numerics.

## Current state
- Cause of the bulk divergence: the V1 `Int4GroupwiseGemmPlugin` uses a CUDA-core GEMV for M <= 6 and a tensor-core GEMM above; both accumulate in FP16 in different orders, so a row's output depends on whether its batch exceeded six rows. Each path is M-invariant internally. Cosmos uses FP16 TensorRT GEMMs and never enters the plugin; XQA attention is batch-invariant (note 368).
- Exactness vs own batch-1 reference: default 1137-1140/1896 (60%); `TRT_EDGELLM_INT4_GEMV_MAX_M=24` 576/632 (91.1%); `=0` (forced GEMM) 610/632 (96.5%, text 468/472). Forced GEMM costs 4-8% tok/s, so it is an audit mode, not a serving default (note 368).
- Vision encoder batch shape (encoder batch 4 in serving vs per-request in the reference) causes the deterministic vision divergences: with forced GEMM plus `TRT_EDGELLM_VISION_ENCODER_BATCH_SIZE=1`, 293/296 (99.0%) on the six vision workloads (note 368).
- Remaining ~1% moves between runs (not a fixed per-prompt source) and is not localized (note 368).
- Earlier hypotheses are superseded: the 61/64 greedy split on requests 1/13 (token 18, `506` vs `1156`, top-2 margins 0.087 / 0.054) came from different sampling logits, not HTTP assembly (note 356). It occurred before the first P membership difference (P dispatch 25, about 1.75 s after the split step), so initial P5/P6 membership does not explain it (note 356). Canonical D row ordering shows no throughput win (1232.06 vs 1236.67 tok/s) and the HTTP A/B is confounded by differing scheduler state (notes 357, 358). A fixed-KV D24 replay of 6000 row comparisons under identity vs reversed row order found 0 token differences and +0.009% GPU time, so row permutation alone is not the cause (note 359). The INT4 M-threshold switch (note 368) is the explanation actually established.
- Accuracy is unaffected: zero-shot MMLU 14031 questions, batch-1 7065 (50.35%) vs serving 7066 (50.36%) twice; predicted letters agree 14030/14031 (note 368).
- vLLM v0.28.0 is neither run-to-run deterministic (562/632 serving vs serving) nor batch-invariant (1032/1264 vs own batch-1) on this contract; the fork default is 588/632 run-to-run (notes 364, 368). Bit-equality to batch-1 is therefore not an achievable gate for this INT4 engine.
- Proposed gate (note 368): (1) scheduler correctness is bit-exact in forced-GEMM mode apart from named batch-dependent kernels; (2) default serving matches batch-1 on an accuracy benchmark within noise. Neither is recorded as adopted; note 369 treats the MMLU result as satisfying part 2.

## Conclusions
- 356 — Greedy split at token 18 comes from different sampling logits, before the first P membership change; D row order accompanied it but was not isolated (single-step logit capture; every-step capture perturbs the trajectory). Cause attribution superseded by 359/368.
- 357 — Canonical D row ordering (opt-in `--canonical-decode-rows`): no throughput gain (-0.37%), outputs still 61-62/64 between arms; kept as diagnostic only.
- 358 — The HTTP row-order A/B is not a pre-decode-state-equivalent experiment (D membership differs from the second D dispatch); a fixed-KV replay is required.
- 359 — Fixed-KV D24 replay: row permutation alone changes neither tokens nor time; row order is no longer the leading explanation.
- 368 — Divergence is the INT4 GEMV/GEMM path switch (bulk) plus vision-encoder batch shape; MMLU gate passes; vLLM is also non-deterministic; gate proposal.

## Open questions
- Localize the run-varying ~1% residual (other FP16 TensorRT layers by M, such as PLE projection or LM head). Closing evidence: per-layer activation dump at batch 1 vs 8 for a divergent prompt (note 368).
- Identify the batch-dependent layer or TensorRT tactic in the vision encoder if batch-invariant vision output is required (note 368).
- Adopt (or reject) the two-part gate; note 369 already calls the tip a promotion candidate on the strength of the MMLU result, pending approval → W010.
- Upstream remark: FP16 accumulation in both INT4 kernels loses precision versus FP32 accumulation; whether to adopt V2 INT4 is raised in → W016.
- Not carried forward: the real-prompt same-step replay proposed in note 359 was not performed; note 368 superseded its motivation.

## Artifacts
- `.local/baselines/int4-force-gemm-863a6d4-20260929` (present)
- `.local/baselines/int4-gemv-knob-62f58ff-20260929` (present)
- `.local/results/vision-encoder-batch1-20260929` (present)
- `.local/results/vllm-determinism-20260929` (present)
- `.local/results/mmlu-serving-accuracy-20260929` (present)
- `.local/results/gemma-serial-reference-20260928` (present; note 364 reference)
- `.local/results/fixed-kv-row-order-v4-20260927` (deleted; note 370)
- `.local/results/decode-row-order-20260927-canonical` and `-affinity` (deleted; note 370)
- `.local/results/prefill-logit-step18-20260927-filtered` (deleted; note 370)
