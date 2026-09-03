# Minimal Physical-Outcome Controller: 12-Workload Gate

## 1. Scope

This report evaluates commit `8c592ea` (`feat: extend transition-aware phase scheduling`) on the complete 12-workload HTTP trace gate.

The implementation keeps the policy architecture intentionally bounded:

```text
ready E/P/D snapshot
        |
        v
hard feasibility
  DAG / TRT profile / memory / single-inflight
        |
        v
physical outcome estimates
  exact CUDA cost + contextual completion vector
        |
        v
deterministic transition
  final-P -> next-D cohort + ownership state
        |
        v
SLO-safe action evaluator
        |
        v
dispatch and CUDA observation feedback
```

The changes under test are:

- Reuse the existing contextual completion RLS rather than adding another policy model.
- Include final-prefill rows in the deterministic one-step decode-formation horizon.
- Permit contextual P+D evaluation after an external producer has completed E -> P.
- Drain asynchronous request-adapter completions out of order while preserving submission identity.
- Extend generic calibration coverage and expose independent feature gates.

Feature gates used for causal checks:

```text
TRT_EDGELLM_DISABLE_DECODE_FORMATION_HORIZON=1
TRT_EDGELLM_DISABLE_EXTERNAL_CONTEXTUAL_PD=1
TRT_EDGELLM_DISABLE_IPC_ADAPTER_OUT_OF_ORDER=1
```

## 2. Correctness and Build Validation

- Changed-scope C++ tests: 205/205 passed.
- Python tests: 7/7 passed.
- Pre-commit checks passed.
- Full C++ suite: 1211 passed, 42 skipped, and two pre-existing/changed-scope-unrelated SM86 numerical-tolerance failures remained in `InitializeYarnRopeCosSin.Accuracy` and `InitializeMRopeCosSin.Accuracy`.
- Commit: `8c592eac13f9bf8ef8329b4c64d00a3ee88bee88` with DCO sign-off.

## 3. Experimental Contracts

### 3.1 Invalid-for-code-attribution calibration experiment

The first full run used the new v7 calibration traces and shorter warmup:

- text warmup: 239 requests instead of p10's 260;
- VLM warmup: 319 requests instead of p10's 424;
- new small-D calibration trace instead of the p10 trace.

Results are retained at:

```text
.local/minimal-completion-full12-3x-20260903
```

This run is useful for evaluating the new calibration proposal, but it is not a code-only A/B against p10. It produced severe instability, including `multi-image` throughput ranging by 49% across three runs.

### 3.2 Fair code-only gate

The fair run used the new binary with p10's exact commands, warmup lengths, calibration traces, trace inputs, concurrency, and repetition counts.

```text
Current:
.local/minimal-completion-fair-p10warmup-full12-20260903

p10 baseline:
.local/p10-final-small-d-12x3-20260903

frozen vLLM baseline:
.local/profile-free-global-20260827/r4-vllm-12x3
```

The frozen vLLM result is reused because the request traces and vLLM configuration did not change. It reports median/p95 latency but not mean latency.

## 4. Fair 12-Workload Throughput, Determinism, and Memory

Positive deltas mean higher throughput.

| Workload | Current tok/s | vs p10 | vs vLLM | Exact token repeat | Peak MiB | Current run range |
|---|---:|---:|---:|:---:|---:|---:|
| short | 2505.5 | +0.18% | +26.32% | yes | 9237 | 1.1% |
| balanced | 4530.3 | -0.75% | +4.87% | yes | 9237 | 1.8% |
| decode-heavy | 5335.6 | +0.38% | +9.91% | yes | 9237 | 0.8% |
| long-prefill | 1194.4 | -0.36% | +6.56% | yes | 9237 | 1.1% |
| bimodal | 1975.0 | +1.18% | +5.71% | yes | 9237 | 1.7% |
| text-heavy | 2080.8 | -1.43% | +27.28% | no | 9385 | 1.4% |
| mixed | 1100.7 | -1.66% | +19.45% | no | 9435 | 6.5% |
| poisson | 1908.0 | -2.78% | +6.00% | yes | 9417 | 3.6% |
| vision-heavy | 682.6 | **-3.94%** | +17.85% | no | 9427 | 9.2% |
| wave-drain | 98.1 | +0.07% | +2.39% | no | 9385 | 2.5% |
| late-vision | 2553.8 | -0.04% | +8.25% | yes | 9379 | 1.2% |
| multi-image | 238.6 | **-26.12%** | **-2.41%** | yes | 9379 | 20.2% |

Aggregate interpretation:

- p10 throughput gate: 10/12 workloads are within 3%.
- vLLM throughput: Current wins 11/12 workloads.
- Geometric-mean throughput versus p10: -3.25%, dominated by multi-image.
- Excluding multi-image: -0.84% versus p10.
- Geometric-mean throughput versus vLLM: +10.65%.
- Excluding multi-image: +11.92% versus vLLM.
- Memory did not materially regress: text stays at 9237 MiB and VLM medians remain 9379--9435 MiB.

The implementation is therefore not promotable over p10 yet. `vision-heavy` narrowly misses the throughput gate and `multi-image` is both slower and too variable.

## 5. Absolute Current Latency

All values are milliseconds. Means are the median of per-run means and p95 is the median of per-run p95 values.

| Workload | TTFT mean | TTFT p95 | TPOT mean | TPOT p95 | E2E mean | E2E p95 |
|---|---:|---:|---:|---:|---:|---:|
| short | 83.1 | 169.8 | 13.65 | 26.95 | 330.3 | 409.0 |
| balanced | 61.3 | 167.1 | 12.19 | 13.83 | 1099.5 | 1699.0 |
| decode-heavy | 63.7 | 172.5 | 10.42 | 11.12 | 2754.5 | 4188.4 |
| long-prefill | 2080.7 | 2610.9 | 26.51 | 31.62 | 4342.9 | 6105.5 |
| bimodal | 1858.0 | 3736.3 | 17.63 | 29.33 | 4252.7 | 8688.1 |
| text-heavy | 308.6 | 931.9 | 23.39 | 35.77 | 1511.4 | 1608.2 |
| mixed | 689.1 | 2001.8 | 35.67 | 48.60 | 2381.7 | 2617.3 |
| poisson | 233.4 | 847.2 | 21.98 | 41.02 | 1642.2 | 2082.6 |
| vision-heavy | 1380.9 | 3213.8 | 41.70 | 59.60 | 2980.1 | 3523.3 |
| wave-drain | 243.5 | 291.4 | 7.95 | 9.76 | 490.5 | 499.7 |
| late-vision | 119.5 | 420.0 | 9.23 | 9.29 | 1434.9 | 1806.1 |
| multi-image | 330.3 | 466.9 | 9.38 | 13.23 | 616.7 | 670.1 |

## 6. Latency Delta Against p10

Negative values are improvements.

| Workload | TTFT mean | TTFT p95 | TPOT mean | TPOT p95 | E2E mean | E2E p95 |
|---|---:|---:|---:|---:|---:|---:|
| short | -3.2% | -0.7% | +1.3% | +1.8% | +0.5% | +0.4% |
| balanced | -4.9% | -0.3% | +1.1% | +2.4% | +0.8% | +0.2% |
| decode-heavy | -0.0% | -4.9% | -0.3% | +0.4% | -0.4% | -0.7% |
| long-prefill | -0.4% | +1.4% | +0.8% | +2.9% | +0.4% | +1.3% |
| bimodal | +0.0% | -2.9% | -2.5% | +0.7% | -1.2% | -2.3% |
| text-heavy | +2.0% | -5.4% | +1.7% | -1.4% | +1.5% | +1.3% |
| mixed | -0.0% | +2.4% | -5.0% | -9.1% | -0.3% | +2.2% |
| poisson | +19.5% | +15.5% | +1.9% | +2.9% | +4.5% | +4.0% |
| vision-heavy | +5.0% | +9.7% | -10.8% | -13.2% | -4.5% | +3.3% |
| wave-drain | -5.2% | -31.8% | -6.2% | -17.8% | -4.6% | -20.9% |
| late-vision | -1.3% | +0.2% | +0.4% | +0.6% | -0.2% | +0.3% |
| multi-image | +32.5% | +64.7% | +20.7% | +42.6% | +26.0% | +35.3% |

Important trade-offs:

- `vision-heavy` improves TPOT and mean E2E while worsening TTFT and E2E p95. The controller advances resident decode work but delays part of the E -> P first-token path.
- `poisson` remains inside the throughput gate but worsens TTFT/E2E tails. This is a request-ordering/SLO issue, not a throughput-only success.
- `wave-drain` improves nearly every latency measure despite flat throughput.
- `multi-image` regresses every latency family and remains the hard blocker.

## 7. Latency Delta Against Frozen vLLM

The frozen vLLM artifact does not contain means, so this table compares median and p95 only. Negative values favor Current.

| Workload | TTFT median | TTFT p95 | TPOT median | TPOT p95 | E2E median | E2E p95 |
|---|---:|---:|---:|---:|---:|---:|
| short | -65.6% | -35.7% | -4.6% | +8.3% | -23.0% | -18.8% |
| balanced | -61.4% | -54.2% | -21.2% | -20.5% | -16.8% | -24.3% |
| decode-heavy | -58.9% | -56.2% | -26.9% | -26.0% | -25.2% | -27.9% |
| long-prefill | -28.2% | -38.7% | -18.6% | -16.0% | -22.5% | -22.3% |
| bimodal | -17.4% | -12.9% | -22.4% | -31.6% | -15.4% | -15.9% |
| text-heavy | -27.6% | -24.4% | -24.3% | -24.5% | -22.1% | -21.1% |
| mixed | +9.6% | -21.2% | -17.5% | -42.2% | -14.7% | -16.7% |
| poisson | -79.3% | -6.1% | +4.6% | -10.2% | -7.9% | -8.1% |
| vision-heavy | -10.4% | -12.9% | -28.3% | -50.2% | -22.2% | -16.7% |
| wave-drain | +8.3% | -30.4% | -40.3% | -43.5% | -23.1% | -23.1% |
| late-vision | -26.2% | -33.5% | -7.1% | -6.5% | -7.7% | -7.6% |
| multi-image | +13.5% | +16.0% | -21.0% | -18.9% | -8.8% | +2.5% |

Win counts versus vLLM:

- TTFT median: 9/12;
- TTFT p95: 11/12;
- TPOT median: 11/12;
- TPOT p95: 11/12;
- E2E median: 12/12;
- E2E p95: 11/12.

## 8. Feature-Gate A/B

### 8.1 External contextual P+D disabled

| Workload | Active tok/s | External P+D off | Off vs p10 |
|---|---:|---:|---:|
| vision-heavy | 682.6 | 655.7 | -7.73% |
| multi-image | 238.6 | 323.7 | +0.20% |

At first sight this appears to show opposite requirements. However, a one-shot full-telemetry rerun reversed the multi-image direction:

```text
active       323.0 tok/s
external-off 236.2 tok/s
```

Therefore the three-run result is not sufficient to claim that external contextual P+D causes the multi-image regression. The stronger finding is that the five-request trace has an unstable E/P formation boundary after calibration. A workload-name or request-count special case would merely select one side of this unstable regime and is rejected.

### 8.2 Decode-formation horizon disabled

| Workload | Active tok/s | Horizon off | Off vs p10 |
|---|---:|---:|---:|
| vision-heavy | 682.6 | 671.8 | -5.46% |
| multi-image | 238.6 | 204.9 | -36.57% |

The one-step final-P -> next-D horizon is beneficial in both traces and is not the regression source. The multi-image horizon-off result was consistently slow across all three runs and retained exact token repeatability.

## 9. What the Results Say About the Architecture

### 9.1 The bounded architecture is still appropriate

The new completion/transition path stays within 3% of p10 on 10 workloads and within 0.84% geometric mean when the unstable five-request multi-image trace is excluded. No additional neural policy, workload classifier, external registry, or arbitrary future-arrival predictor is required.

The separation remains sound:

```text
learn physical completion/interference
              +
compute formation/ownership transitions deterministically
              +
apply request SLOs lexicographically
```

### 9.2 The new calibration trace is not promotable

Reducing VLM warmup from 424 to 319 and broadening small-D coverage increased calibration non-convergence and run variance. The proposal should stay as an experimental artifact, not replace the p10 calibration contract.

### 9.3 Completion authority has not actually promoted

The completion model receives observations, but the final calibration records show:

```text
completion_authority_evidence_ready = false
completion_authority_validated      = false
completion_authority_window_observations = 0
```

Consequently the production choice still obtains most of its authority from the scalar contextual advantage and exact cost path. This is the main semantic gap between the intended physical-outcome architecture and the observed run.

### 9.4 Tiny traces expose decision realization variance

On multi-image, a single encoder formation shift changes total runtime by hundreds of milliseconds. Three repeats are insufficient when the run range is 20--49%. This is not a reason to add a `multi-image` heuristic. It is a reason to make the calibration-to-serving transition and E formation reproducible, then evaluate the completion model.

## 10. Promotion Decision

**Do not promote commit `8c592ea` as the new best configuration yet.**

Keep p10 as the performance baseline. Keep `8c592ea` as the implementation branch because:

- the deterministic successor horizon is validated;
- the code is test-clean in changed scope;
- 10/12 throughput gates pass;
- Current remains substantially ahead of frozen vLLM on the broad suite;
- the remaining failures are localized to VLM calibration/formation stability rather than a broad text regression.

## 11. Next Plan

### P0 — Stabilize the calibration-to-serving boundary

- Freeze the p10 warmup contract.
- Assign a canonical E-ready order before the measured trace begins.
- Drain all warmup GPU events, sampling completions, request-adapter tasks, and memory leases before resetting policy telemetry.
- Assert an empty outstanding E/P/D/Copy set at the measurement epoch boundary.

### P1 — Make physical completion authority observable

- Record completion-authority validation samples during calibration rather than leaving the evidence window at zero.
- Report requested versus realized start skew, incumbent/newcomer completion error, false-safe count, and authority blend weight in the final summary.
- Do not grant authority merely from scalar sample count.

### P2 — Controlled multi-image repetition

- Run at least 10 repeats after P0 with one frozen calibration snapshot.
- Compare E dispatch count, E batch histogram, E GPU duration, first E start, first P start, and E -> P critical-path completion.
- Require run-range <=5% before using this trace as a promotion gate.

### P3 — Vision-heavy causal gate

- Preserve external P+D and decode horizon, since disabling either was not beneficial.
- Compare scalar-only, completion-shadow, and completion-active on the same frozen calibration state.
- Require throughput within 3% of p10 without worsening vision TTFT p95 or E2E p95 by more than 3%.

### P4 — Exact-output follow-up

- Diagnose row-order/tactic numerical divergence in text-heavy, mixed, vision-heavy, and wave-drain.
- Keep production promotion blocked on exact token identity even when semantic outputs agree.

### P5 — Full gate

- Rerun all 12 workloads with the p10 warmup contract.
- Reuse frozen vLLM only while the request trace and vLLM setup remain unchanged.
- Promote only if all 12 throughput scenarios are within 3%, exact-output gates pass, and no TTFT/TPOT/E2E p95 regression exceeds the workload-specific SLO gate.

