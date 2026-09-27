# External-Encoder Cost Context, Async-State Fix, and Full-Workload Validation

## Summary

This step added execution context to P/D cost keys so a P/D sample collected while an independent vision encoder is
running cannot be reused as an isolated P/D estimate. The full-24 screen initially exposed a major long-prefill
regression against the previous 9db3 champion. Full dispatch audit then found that the E-active scheduler bit can outlive
the measured E GPU interval: the three-phase coordinator's P/D-only fast path could bypass encoder-state retirement,
and a direct-event handoff returned before clearing E-active after the CUDA event was already complete.

The coordinator now routes stale E-active state through its completion path and clears the flag immediately after the
direct-event handoff when the encoder completion event is ready. This substantially recovered Cosmos long-prefill
overlap and service-cost coverage in a controlled full audit. It did not completely resolve the mismatch: some
text-only long-prefill dispatch records still show E-active while the measured E stream has no active interval. Exact
physical E/P/D interval attribution remains an open item; this change is a partial state-lifetime correction, not a
promotion of the new binary.

Final full24 throughput screen plus two repeats completed 72/72 cells with no throughput run below the frozen vLLM
anchor. Every workload's 3-run median beats that anchor. This does not establish universal latency superiority or
output-quality parity. Ten of 24 medians remain below the historical 9db3 Current result, most by less than 1%; the
largest is Gemma long-prefill at -7.0% against the historical result, while a same-day paired 9db3 control showed a
smaller -2.5% difference. The existing 9db3 serving default is therefore not promoted or replaced here.

## Implementation

- `PhaseGlobalActionKey.externalEncoderBackground` partitions direct and overlap cost observations by whether the
  independent E actor was active outside the P/D action. Equality, hashing, candidate identity, interpolation, and
  covering-estimate semantic filters include this field.
- When the field is false, hashing and candidate-ID construction retain their pre-change sequence. This avoids
  needlessly perturbing legacy no-E hash iteration and tie behavior.
- `PhaseQueueSnapshot` carries the current coordinator E-active state. Candidate generation, decode/prefill measured
  cost lookup, and observed action keys use the same bit.
- The P/D-only fast path now requires no coordinator E-active state. In the direct-event handoff path, after payload
  publication it queries the encoder completion event and clears the E-active bit immediately if the E GPU work has
  ended. This avoids retaining E contention merely because the coordinator has not yet retired the event's metadata.
- No workload-name branch, new user SLO, batch-size cap, or startup probe was added. The only ignored diagnostic option
  from the earlier D-reference experiment was removed after it showed no serving improvement.

## Why the issue mattered

The initial cost-key validation binary used a single key space for isolated P/D and E-contended P/D observations.
On Cosmos long-prefill, the three-phase coordinator reported 819 overlap opportunities, of which only 192 had a
known compatible cost and 472 had no sample. The P+D action fraction was 46.5% (684 of 1,470 dispatches).

In the final full audit after the coordinator state fix, the corresponding one-run Cosmos long-prefill audit reported
769 P+D dispatches out of 1,446 (53.2%), 202 known overlap costs, and 287 no-sample opportunities. The old 9db3
paired audit had 945/1,396 P+D dispatches (67.7%), 581 known costs, and 86 no-sample opportunities. The recovery is
real in this audit, but it does not close the old/new gap.

For Gemma, the final audit remained mixed: P+D was 111/658 dispatches (16.9%), versus 144/641 (22.5%) in the paired
old binary. The final targeted 3-run campaign did not show a statistically stable improvement against the historical
champion, although it remained above frozen vLLM throughput.

The audit also found E-active metric values in a text-only long-prefill trace while the captured E stream had zero
active duration. After the new event-clear path, the one-run full audit's E-active dispatch count fell from 50 to 15
for Cosmos and from 14 to 3 for Gemma; it did not reach zero. This is why the code must not yet treat the bool as a
fully validated physical-overlap label. The follow-up should compare planned E state with actual E CUDA-event
intervals and train observations on realized interval overlap, while preserving the planned state for candidate
prediction.

## Final 24-workload results

The same engine, request traces, generic calibration, fixed-output contract, and dispatch telemetry were used. Frozen
vLLM data was reused because its engine/runtime/workload contract did not change. Throughput columns show the 3-run
median; latency columns show percent change from frozen vLLM. Negative latency change favors Current. Small differences
against the older Current are not causal because the historical and paired campaigns were not a fully interleaved
cross-over.

| Model / workload | Current tok/s | vs vLLM | vs historical Current | TTFT mean / p95 Δ | TPOT mean / p95 Δ | E2E mean / p95 Δ |
|---|---:|---:|---:|---:|---:|---:|
| Cosmos / balanced | 4475.8 | +3.7% | +0.7% | -48.0 / -42.2% | +2.1 / +2.3% | -3.4 / -1.6% |
| Cosmos / bimodal | 1981.1 | +5.8% | +1.6% | +19.6 / +55.6% | -23.9 / -20.6% | -9.5 / -2.4% |
| Cosmos / decode-heavy | 5246.4 | +6.3% | +0.1% | -47.6 / -50.2% | -3.3 / -3.4% | -5.4 / -4.3% |
| Cosmos / late-vision | 2517.3 | +16.3% | -0.1% | -49.6 / -48.8% | -13.2 / -12.9% | -18.3 / -14.0% |
| Cosmos / long-prefill | 1301.1 | +15.8% | -0.8% | +3.7 / -9.6% | -26.7 / -26.9% | -14.1 / -14.9% |
| Cosmos / mixed | 1155.6 | +25.2% | +1.4% | -2.5 / -21.3% | -30.1 / -29.6% | -20.5 / -19.5% |
| Cosmos / multi-image | 313.2 | +28.4% | +0.6% | +1.1 / -24.2% | -36.7 / -42.9% | -21.3 / -22.0% |
| Cosmos / poisson | 2031.9 | +14.1% | +0.4% | -36.2 / -29.5% | -12.3 / -17.5% | -17.1 / -13.9% |
| Cosmos / short | 2386.9 | +16.7% | -0.9% | -48.5 / -22.2% | +0.3 / +6.6% | -21.7 / -15.5% |
| Cosmos / text-heavy | 2054.1 | +58.9% | +3.2% | -52.5 / -49.6% | +12.7 / -16.1% | -15.3 / -35.4% |
| Cosmos / vision-heavy | 719.6 | +24.7% | +0.9% | -9.4 / -12.6% | -31.1 / -34.4% | -21.3 / -20.5% |
| Cosmos / wave-drain | 98.0 | +2.2% | -0.1% | -3.3 / -27.6% | -33.2 / -35.1% | -21.3 / -21.6% |
| Gemma / balanced | 1268.4 | +64.4% | +0.5% | -35.1 / -14.3% | -35.2 / -30.0% | -35.8 / -35.0% |
| Gemma / bimodal | 843.5 | +40.5% | -0.4% | +22.6 / +14.3% | -18.2 / +12.1% | -24.6 / -27.0% |
| Gemma / decode-heavy | 1376.5 | +69.4% | -0.1% | -42.6 / -15.8% | -37.5 / -36.2% | -37.9 / -37.5% |
| Gemma / late-vision | 1523.1 | +53.7% | -0.6% | -0.5 / +53.6% | -39.8 / -39.4% | -38.2 / -39.6% |
| Gemma / long-prefill | 575.9 | +15.1% | -7.0% | +19.3 / +13.5% | -14.3 / -9.4% | -8.7 / -3.4% |
| Gemma / mixed | 746.0 | +6.0% | +0.2% | -9.7 / +20.0% | -3.9 / +10.8% | -4.9 / +0.8% |
| Gemma / multi-image | 402.5 | +5.6% | +3.9% | +97.9 / +211.7% | -12.4 / +18.4% | +6.4 / +17.5% |
| Gemma / poisson | 946.5 | +38.8% | +1.8% | +2.7 / +82.4% | -22.4 / -17.3% | -21.8 / -20.4% |
| Gemma / short | 854.1 | +50.5% | -0.0% | -38.8 / -12.0% | -21.0 / -1.4% | -27.4 / -23.5% |
| Gemma / text-heavy | 900.3 | +122.5% | +0.7% | -89.3 / -90.3% | -6.6 / -5.6% | -54.7 / -72.4% |
| Gemma / vision-heavy | 570.9 | +2.0% | -0.1% | +38.3 / +91.6% | -2.1 / +17.0% | +6.0 / -4.9% |
| Gemma / wave-drain | 97.3 | +5.0% | +0.0% | +39.4 / +52.1% | -57.6 / -57.0% | -37.9 / -37.5% |

Across these 24 rows, throughput is higher than frozen vLLM in 24/24 3-run medians and all 72 individual runs. Latency
is not universally better: mean/p95 wins against vLLM are 15/24 TTFT, 21/24 TPOT mean, and 22/24 E2E mean; corresponding
p95 wins are 15/24, 18/24, and 22/24. Positive latency deltas identify the remaining trade-offs.

The fixed-output bench generated the requested token counts in all cells. It is not a semantic-quality gate: only 14/24
workload traces had identical output-token hashes across the three runs, while 10/24 differed. Model output-quality
approval remains unresolved.

## Validation and retained evidence

- Pinned TensorRT 26.06 container; scheduler/runtime and `llm_phase_context_smoke` rebuilt.
- `PhaseGlobalCostModelTest.*` + `PhaseQueueSchedulerTest.*`: 184/184 passed after the final coordinator change.
- `test_lifetime_encoded_admission.py`: 53/53 unittest cases passed; Python syntax compilation passed.
- `git diff --check` passed. The commit-time hook chain ran and passed, including CRLF normalization, license insertion,
  clang-format, and codespell. The standalone `pre-commit` command was not on the host/container PATH.
- Final 3x full24 screen:
  - `.local/results/external-background-key-fastpath-fix-full12-screen-20260927`
  - `.local/results/external-background-key-fastpath-fix-full12-additional-20260927`
- Selected 3x control and detailed causal audit:
  - `.local/results/external-background-key-fastpath-fix-targeted-20260927`
  - `.local/results/champion9db3-paired-controls-20260927`
  - `.local/results/long-prefill-full-audit-champion-20260927`
  - `.local/results/long-prefill-full-audit-current-20260927`
  - `.local/results/long-prefill-full-audit-fastpath-fixed-20260927`
  - `.local/results/long-prefill-full-audit-final-20260927`
- Frozen vLLM source paths and identities are in each campaign manifest. The vLLM engine and request contract did not
  change, so it was not rerun.

## Next work

1. Replace the coarse planned E-active bit for measured labels with physical E-interval overlap. Preserve a planned
   context feature for candidate prediction, but update the observation key from actual E CUDA-event start/end overlap.
2. Add an audit invariant: a text-only dispatch with no E interval must not increment the E-background observation
   bucket. Test both event-complete-before-dispatch and E-finishes-during-P/D cases.
3. Repeat full24 after the physical-interval label correction; compare against the frozen vLLM anchor and the
   immutable 9db3 binary. Keep output quality separate from fixed-output throughput.
4. Do not promote the local default until physical E/P/D label fidelity, output-quality gates, and repeat stability are
   resolved. The 9db3 binary remains the recorded serving default in `.local/registry/current.json`.
