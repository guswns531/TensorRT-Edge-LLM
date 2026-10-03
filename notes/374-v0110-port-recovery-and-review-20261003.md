# 374 — v0.11.0 port: Cosmos recovery fixes, true packed prefill, and external review (2026-10-03)

## Outcome
- The port with the W016 recovery fixes is faster than frozen vLLM in all 24 full24 cells (0/72 runs below vLLM, minimum run ratio 1.003; geomean +27.9%). Against the v0.10.1 tip it is +0.7% overall: Gemma +2.7%, Cosmos −1.3%.
- MMLU zero-shot through serving: 51.27% (7193/14031), equal to the batch-1 reference on the port (51.27%); same prediction on 14028/14031.
- The recovery fixes recover Gemma and Cosmos prefill-heavy cells but not the Cosmos decode-heavy gap. Against the pre-fix port, Gemma is +1.1% and Cosmos +0.2%; Cosmos balanced, bimodal, decode-heavy and mixed stay 1.7–2.7% below v0.10.1, and Cosmos balanced is only +0.4% above vLLM.
- An external read-only review (Codex GPT-6.1-Sol, five areas) found no defect on the validated phase-serving path, but six port-introduced defects on other paths. All six are fixed. Four defects inherited from the fork are recorded and left open by decision.

## Commits (branch `codex/v0110-phase-forward-port`, on top of pushed `f6c2f094`)
| Commit | Change |
|---|---|
| `88845223` | Full-mode SWA selected at init for phase serving; dead host `generateVisionBlockIds` and PipelineIO's throwaway dense `inputsEmbeds` removed |
| `ee4bc148` | KV page-table upload: three-slot pinned staging ring with per-slot events (full and dirty-index uploads) |
| `c88d2f27` | Phase ragged metadata: one pinned block, one H2D copy and a device unpack (7 copies → 2 + 1 kernel); steady decode updates only advancing fields |
| `91feff6c` | True packed prefill (T = Σq_i, prefix-sum rows) in the host builder, both staging paths and the attention plugin; RoPE row fix for sequences after the first; N = 1 detection; shared-KV Q+K+V stride; SWA workspace sized as the maximum over dense and packed paths with an enqueue-time bound check |
| `c80fee7f` | Ordinary `LLMRankRuntime` prefill emits the packed layout and dims on packed engines (over-cap rows and multimodal input on that path are rejected explicitly); phase prefill metadata capacity on non-packed engines |
| `93083645` | CMake default multi-SM list restored while an explicit `CMAKE_CUDA_ARCHITECTURES` still wins; Gemma4 ragged `skip_softmax_scale_factor` back to upstream `0.0` |

The six commits were regrouped from 14 commits and 6 merges with an identical final tree (`924e7126`, kept as local branch `codex/v0110-port-pre-squash-924e7126`).

## Evidence
### Throughput (full24 ×3, binary `9f911c16`, generated tok/s, median of 3)
Contract unchanged from the pre-fix port run: same engines, workloads, repeat count, V3 independent E/P/D with predictors, CUDA graphs and serving probes.

| Comparison | Gemma | Cosmos | All 24 |
|---|---:|---:|---:|
| vs frozen vLLM | +40.6% | +16.3% | +27.9% |
| vs v0.10.1 tip (`tip-full24-3x-20260929`) | +2.7% | −1.3% | +0.7% |
| vs pre-fix port `f6c2f094` | +1.1% | +0.2% | — |

Largest movements against the pre-fix port: Gemma short +4.4%, text-heavy +3.2%, mixed +2.3%, multi-image −2.2%; Cosmos text-heavy +1.5%, short +1.3%, long-prefill +1.2%, bimodal −1.4%. The v0.10.1 tip was measured on 2026-09-29, so the version comparison carries a day-to-day term.

### Output quality
- MMLU (`v0110-port-fix-mmlu-20261001`, serving only, binary `9f911c16`) against the port batch-1 reference from `v0110-port-mmlu-20261001`: 51.27% vs 51.27%, 14028/14031 same prediction. The pre-fix port served 51.25%; v0.10.1 served 50.36% (note 368), the difference being the doubled Gemma `<bos>` fixed in the port.
- Fixed-output smoke, final tree `924e7126` (tree identical to `93083645`): Gemma short 48/48, Cosmos short 48/48 and Cosmos multi-image 5/5 token-exact against the fix full24 run; Gemma multi-image 17/20, inside its run-to-run spread (17–18/20 between repeats of the same binary).

### Tests (final tree)
- 12 C++ unit-test executables: 2300 tests, 0 failures.
- `tests/python-unittests/test_attention_plugin.py`: 181 passed, 13 skipped, 1 failed — the known upstream failure `test_head512_shared_vision_prefill_uses_token_aligned_rope`.
- `tests/python-unittests/test_gemma4_ragged_export.py`: 19 passed. Each new regression test was shown to fail with its fix reverted.
- Ordinary `llm_inference` on a fully committed rebuild of the packed Gemma engine: two unequal prompts in one batch matched their batch-1 outputs after the fix; before it, the second row diverged.

### Codex review
Five read-only reviews (attention plugin and kernels; host ragged metadata; KV, memory and SWA; merge fidelity of upstream-owned files; export and engine-build contract) produced 19 deduplicated findings: 10 high, 8 medium, 1 low. None reproduces on the validated phase-serving path. Fixed here: ordinary-path packed layout (#1), shared-KV Q+K+V stride (#4), SWA packed workspace (#5), non-packed phase capacity (#6), Gemma4 skip-softmax export (#11), CMake default architectures (#12).

Left open by decision (inherited from fork `fca7bd0`): batched system-prompt restore reuses pinned descriptor scratch without a fence (#2); shared-KV context prefill indexes token-aligned RoPE with tree position ids on speculative bases (#3); packed prefill rejected when the Blackwell FMHA backend is selected on SM100/101/110 (#16); M-RoPE row cache keyed only by request id (#17). Remaining port-path items not fixed: late `enablePhaseServing()` keeps bounded SWA (#9), non-semantic smoke warm-ups (#10), auxiliary-profile buffer sizing (#8), phase staging of `vision_block_ids` (#7), validator upper bound (#14), RoPE broadcast helper contract (#15), tiered Gemma4 ViT profile on the ordinary path (#13), rebuild script paths (#18), broker telemetry (#19).

## Limits
- Throughput and MMLU were measured on `9f911c16`. The six review fixes after it were validated by unit tests, pytest and four fixed-output smoke cells, not by a full24 or MMLU rerun.
- The Cosmos decode-heavy gap to v0.10.1 is unexplained; the regression analysis that motivated the KV upload and metadata fixes did not predict it correctly.
- Ordinary `llm_inference` on packed engines rejects rows longer than the chunk cap and multimodal input instead of chunking them.

## Retained paths
- `.local/results/v0110-port-fix-full24-3x-20261001` (diagnostic)
- `.local/results/v0110-port-fix-mmlu-20261001` (diagnostic)
- `.local/results/v0110-port-full24-3x-20261001`, `.local/results/v0110-port-mmlu-20261001` (diagnostic, pre-fix port)
- `.local/baselines/v0110-port-9f911c16-20261001` (frozen binary with manifest)
- `.local/results/v0110-port-codex-review-20261003` (diagnostic; review prompts, five reports, triage)
