# Gemma Long-Prefill Bistability: Mechanism Search and Residual-Compression Transfer

## Outcome

The Gemma long-prefill throughput gap against 9db3 (note 364) is a variance difference, not a confirmed mean
regression. Pooled over 77 dispatch-telemetry runs, 9db3 is 622.6 tok/s median (n=14, sd 8.4) and the post-9db3
lineage is 606.2 (n=63, sd about 16); Mann-Whitney z=1.65, P(9db3 > current)=0.64. 9db3 lands in the high mode
10/14 times; every later binary about half the time. Under full telemetry, 9db3 itself produced 593 and 604.

One policy change was implemented and A/B tested (`19d8aa8`, complete P+D priced from residual compression of the
same shape). It is correct and fires as designed, but it does not change the mode distribution (z=-0.92 versus the
pre-change binary). Its retention is undecided pending a full24 check; see Next work.

No single scheduler-state variable found so far determines the mode. The investigation is recorded so it is not
repeated.

## What was measured

All runs: Gemma long-prefill, note 362 contract, `--telemetry-level dispatch` unless stated.

| Campaign | Arms x blocks | Result |
|---|---|---|
| `gemma-long-prefill-binary-ab-20260928` | 1fab188 / 5a9b417 / 9db3 x6 | 9db3 621.7 median, others bimodal 585-635 |
| `gemma-long-prefill-ablation-ab-20260928` | 1fab188 / noref / nocal / 5a9b417 x8 | all four arms bimodal; neither ddc1f4e change is causal |
| `gemma-long-prefill-residual-transfer-ab-20260928` | 1fab188 / 19d8aa8 / 9db3 x8 | 19d8aa8 615.4 median, 5/8 high; 1fab188 616.5, 5/8; 9db3 622.5, 6/8 |
| `gemma-long-prefill-full-telemetry-20260928` | 1fab188 x6, 9db3 x2 (full) | 1fab188 5/6 high; 9db3 0/2 |
| `gemma-long-prefill-transfer-full-telemetry-20260928` | 19d8aa8 x3 (full) | 596, 591, 618 |

## Per-run accounting

Across 27 runs with dispatch metrics (three binaries), idle GPU time is 1.5-1.6% everywhere and decode steps are
fixed (292-299). The two quantities that move throughput are prefill GPU cost per token (101-102 us in fast runs,
108-110 in slow) and P+D overlap savings. 9db3 has poor prefill efficiency (105-113 us/token) but overlaps 108-124
times; the fast post-9db3 runs overlap 85-95 times with 101 us/token; slow runs get neither (36-63 overlaps, 100-110
us/token). Overlap count alone does not predict throughput: one 19d8aa8 run overlapped 110 times at 596 tok/s because
its prefill cost was 110 us/token.

## Mechanisms examined

1. **Complete P+D cold pricing.** A complete P+D candidate with no samples for its exact key is priced as serial
   P+D (`derived_isolated`) and loses to prefill-only on every decision (0/176 selected). Almost all executed P+D
   comes from residual augmentation (`unknown` cost source, 24-110 per run), whose samples are stored under
   residual-flagged keys that the complete candidate never reads. `19d8aa8` transfers the resolved residual
   compression to the complete candidate (`runtime_residual`). Full telemetry confirms it fires (45-70 candidates
   per run) but those candidates are still selected 0-2 times per run.
2. **Contextual PD authority.** When the contextual P+D head is ready, the candidate's decision cost is
   `reference * (1 - LCB)` regardless of the measured price. On this trace the head's mean reward is -0.03 to -0.11
   (LCB -0.08 to -0.16) on every binary including 9db3, because its 126-169 prefill-to-decode observations come from
   residual augmentations whose reference is scaled to the remaining launched-phase work, so late augmentations are
   structurally near-zero reward. The head therefore prices complete P+D at about 30 ms against a 19 ms prefill and
   the measured 18.6 ms overlap. This is a real bias, but it does not select the mode: the slowest run observed
   (573 tok/s) had a positive contextual mean (+0.34).
3. **Encoder-context key split, calibration key, reference guard** (ddc1f4e): ruled out by the 4-arm ablation.
4. **Wait selections, prefill fragmentation, decode batch shape**: all vary with the mode but none separates 9db3
   (fragmented prefill, many overlaps, fast) from the fast and slow post-9db3 runs consistently.

## Next work

1. Decide `19d8aa8`: revert, or keep only after a full24 x3 shows no regression on the other 23 workloads.
2. If the contextual head is to price complete P+D, its residual observations need the full-action reference (not
   the remaining-work reference) or a separate direction for complete actions. Not attempted.
3. The mode switch itself is timing-dependent. A paired replay with identical ingress timestamps (`--ordered-backend-
   ingress`) on two binaries would show whether it is scheduler state or arrival timing. Not attempted.
