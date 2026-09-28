# Fix Revalidation, Activity-Observer Overhead, and Batch-1 Output Reference

## Summary

This note covers the realized-label fix binary `1fab188` (`ddc1f4e` and follow-ups):

- **Revalidation.** A full24 x3 run under the note 362 contract completed 72/72 cells, and every run beat frozen vLLM.
  Gemma long-prefill initially looked 5% slower than 9db3.
- **Long-prefill attribution.** A 4-arm, 8-block ablation shows this is not caused by the fix. Every post-9db3 binary is
  bimodal on that trace, while 9db3 is not.
- **Observer overhead.** Neither observer mode costs throughput. With both observers off, cost labels fall back to
  planned state and throughput drops.
- **Output gate.**
  - A new batch-1 serial reference is fully deterministic.
  - Cosmos serving matches it bit for bit.
  - Gemma serving matches it on 60% of requests. Every sampled divergence is a top-2 near-tie flipped by
    Gemma-specific batch-shape logit variation.
  - The gate is narrowed to that Gemma numeric source; it is not approved.

All campaigns used binary `b2e6bdaa`
(`.local/baselines/realized-label-fix-1fab188-20260928`), the note 362 engines, generic calibration, and dispatch
telemetry. All are `diagnostic` unless stated otherwise.

## Revalidation

Campaign: `.local/results/realized-label-fix-full24-3x-20260928`. The per-cell environment and mounts are identical
to the note 362 campaigns except for the binary mount.

- 72/72 cells, 0 failures, 0 integrity issues. All 72 runs exceed the frozen vLLM anchor.
- Versus 9db3 medians:
  - Cosmos: -0.1% to +3.7%.
  - Gemma: -0.2% to +4.5%, except long-prefill at -5.0% (588.7 tok/s, runs 586.0-588.9).
- Output repeatability is unchanged from note 362. Per request across the three runs, Gemma is 588/632 exact and
  Cosmos is 1513/1513.

### Gemma long-prefill attribution

Two interleaved A/B campaigns ran on Gemma long-prefill:

- `gemma-long-prefill-binary-ab-20260928`: 6 blocks, arms 1fab188 / 5a9b417 / 9db3.
- `gemma-long-prefill-ablation-ab-20260928`: 8 blocks, arms 1fab188 / 5a9b417 / `noref` / `nocal`.
  - `noref` reverts only the serial-reference guard.
  - `nocal` reverts only the encoder-agnostic calibration key.
  - Both patches are recorded in their baseline directories.

| Binary | Runs | Median tok/s | Runs >= 610 |
|---|---:|---:|---:|
| 9db3 | 6 | 621.7 | 4/6 (P+D dispatches 105-146 in every run) |
| 5a9b417 (pre-fix) | 14 | 608.8 | 7/14 |
| 1fab188 (fix; includes revalidation) | 17 | 598.3 | 7/17 |
| `nocal` | 8 | 603.0 | 3/8 |
| `noref` | 8 | 593.3 | 3/8 |

- Every post-9db3 arm alternates between two modes:
  - Low: about 582-600 tok/s with 27-63 P+D dispatches and many global `wait` selections.
  - High: about 615-635 tok/s with 81-101 P+D dispatches.
- In the second campaign, the fix arm had the highest median (615.3).
- The earlier 3/3 low revalidation runs were a draw from this distribution; neither change moves the mode frequency
  beyond noise.
- The regression relative to 9db3 is therefore a pre-existing bistability introduced somewhere in notes 340-362. It
  remains open (see Next work).

## Activity-observer overhead

Campaign: `.local/results/activity-observer-overhead-20260928`, driver
`benchmarks/phase_serving/run_activity_observer_overhead.py`.

- Modes compared:
  - `off`: no recorder; labels fall back to planned state.
  - `encoder`: the `PhaseServingRuntime` default, exposed as `TRT_EDGELLM_PHASE_ENCODER_OBSERVER`.
  - `full`: E/P/D/C export.
- Design: 5 VLM workloads x 2 models x 3 blocks, with mode order rotated per block.

| Mode vs off | Throughput geomean | Largest cell |
|---|---:|---|
| encoder | +0.75% | Gemma long-prefill +5.1% |
| full | +1.69% | Gemma long-prefill +8.8% |

Gemma mixed is the only material negative cell (encoder -3.9%, full -1.2%). The observer's host and event cost is
therefore not a throughput concern. The realized label improves learned costs by more than the recording costs. No
recorder should be removed for performance reasons.

## Output gate: batch-1 reference

- Reference campaigns:
  - `.local/results/gemma-serial-reference-20260928`: 12 traces x 2.
  - `.local/results/cosmos-serial-reference-20260928`: 12 traces x 1.
- Both use `--client-max-in-flight 1 --ordered-backend-ingress` on the same binary, so every request runs alone at
  decode batch 1.

| Comparison | Exact requests |
|---|---:|
| Gemma reference repeat 1 vs 2 | 632/632 |
| Cosmos serving vs Cosmos reference | 1513/1513 |
| Gemma serving vs Gemma reference (3 runs) | 1137/1896 (60%); first divergence median token 30, min 1 |
| Gemma serving vs itself across runs | 588/632 |

Worst Gemma traces against the reference are decode-heavy (12/64) and late-vision (8/32); wave-drain is 20/20.

The frozen vLLM Gemma run kept only a text prefix (no token IDs; the prefix is typically tens of characters). On that
prefix, agreement is:

| Pair | Prefix agreement |
|---|---:|
| vLLM vs our reference | 186/632 |
| vLLM vs our serving | 194/632 |
| Our serving vs our reference | 544/632 |

A different engine on the same checkpoint is therefore much further from our batch-1 output than our own serving is.

### Divergence logits

Campaign: `.local/results/gemma-divergence-logits-20260928`. The steps 0-63 logits were captured for one stable
divergent request per workload, in both serial and serving mode:

| Trace/request | Step | Serial top-2 margin | Serving top-2 margin | Previous-step max abs logit delta |
|---|---:|---:|---:|---:|
| balanced 4, mixed 38, poisson 6, text-heavy 6, vision-heavy 6 (same prompt) | 13 | 0.109 | 0.0073 | 0.263 |
| bimodal 32 | 4 | 0.083 | 0.051 | 0.265 |
| long-prefill 10 | 36 | 0.035 | 0.035 | 0.208 |
| multi-image 9 | 20 | 0.074 | 0.019 | 0.473 |

- In every case, the serial and serving top-1/top-2 tokens are the same pair in swapped order.
- Every margin is smaller than the batch-shape logit delta already present one step earlier.
- No sampled divergence has a large margin.

### Gate status

- Cosmos passes the batch-1 invariance check exactly.
- The sampled Gemma divergences all fall in the near-tie class. However, the underlying 0.2-0.5 logit batch-shape
  variation is Gemma-specific, because Cosmos on the same runtime is bit-exact.
- Candidates for the Gemma-only numeric source:
  - AWQ int4 GEMM tactic or split selection by batch size.
  - The Gemma4 PLE preprocessor path.
  - Sliding-window or soft-cap attention variants.
- The gate stays open until that source is localized and either shown to be reduction-order-only or fixed.
- Proposed acceptance rule once localized: every serving-vs-batch-1 divergence occurs at a top-2 margin below the
  measured batch-shape logit delta, and no layer shows a batch-dependent error beyond FP16 reduction-order bounds.

## Next work

1. **Gemma long-prefill bistability.** Compare 9db3 and current global `wait` decisions on paired high/low runs, and
   find the post-9db3 change that makes wait selection history-dependent.
2. **Gemma batch-shape numerics.** Dump per-layer activations for one divergent prompt at decode batch 1 versus 8 and
   find the first layer whose difference exceeds FP16 reduction-order scale.
3. **Startup SIGSEGV with tokenizer DOM corruption** (note 363). Still unreproduced.
