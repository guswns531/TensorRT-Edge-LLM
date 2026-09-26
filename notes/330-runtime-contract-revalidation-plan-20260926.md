<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# 330. Runtime contract repair and dual-model revalidation

Date: 2026-09-26. Branch: `codex/v0101-phase-forward-port`.

Status: the fixed-contract dual-model Full12 x3 campaign is complete (72/72 runs).
Implementation, validation failures, and promotion limits are recorded in
[331](331-runtime-contract-memory-and-full24-revalidation-20260926.md).
Workspace controls and rejected memory-mode alternatives are separated in
[334](334-workspace-memory-mode-final-screen-20260926.md). Completion of the campaign does not imply
all correctness or performance gates passed; in particular, Gemma exact output and headroom remain unresolved.

## Scope

Preserve the existing contextual scalar controller and stable indexed-paged KV ownership. Repair observable-state
contracts before adding a new learned controller. No workload-label policies or new quantization are part of this
campaign. The initial no-rebuild plan was revised after finding an undersized packed-decode plugin workspace:
correctness required new plans from the same ONNX and builder configuration. Existing engines and current pointers
are preserved; this changes the artifact contract, so final measurements are not source-only performance A/B.
See note 331 sections 13–14 for the defect, evidence, and retained artifacts.

## Frozen reference

- Source baseline: `949334c` (clean at campaign start).
- Binary: `.local/baselines/runtime-review-20260926/bin/examples/llm/llm_phase_context_smoke`.
- Binary SHA256: `3217f49f76f07faa0e91f65bc9926dbdd43212a7bb7ba3250863a165879f3146`.
- Plugin SHA256: `d3c3c679794814e83bfd9468b05bc81ab08adc8c7966f3317102b020b67af2e5`.
- The binary matches the retained September 21 second full24 campaign. It is an artifact reference, not proof that
  its dirty-source build is identical to the final source baseline.

## Work sequence

1. Explicit, immutable experiment contracts and raw-derived seven-metric reporting; correct historical overclaims.
2. Unify transition-observation feature semantics. Queue residence is not physical E/P/D handoff latency.
3. Share runtime configuration resolution and decode graph initialization between smoke and production.
4. Separate E/P activation exclusivity from retained vision-output storage. Default stays one slab. An opt-in,
   bounded two-slab experiment may prepare a successor while the predecessor is retained, without allowing E/P
   workspace overlap. Independent workspace remains a separate control.
5. Build and run CPU/GPU ownership, scheduler, and configuration tests. Screen predictor on/off and workspace modes.
6. Run Gemma and Cosmos twelve-workload campaigns with three repetitions of one fixed contract. Reuse frozen vLLM
   only when the request/token/image contract is unchanged.

## Evaluation rules

- Report throughput and TTFT/TPOT/E2E mean and p95 separately; no all-metric win claims from throughput alone.
- Single-run token hashes are not repeatability evidence. Report cross-repeat and cross-variant hashes separately.
- Actual engine configuration determines KV capacity, not a directory name or a note heading.
- Keep graph preparation, batch limits, generic calibration, and encoded admission explicit.
- Compare memory at identical KV capacity before reinvesting savings in larger capacity.
- No source/engine deletion; temporary E8 OOM artifacts remain diagnostic, not completed results.
- Promotion requires correctness and repeated comparison; any failed gate remains visible.
