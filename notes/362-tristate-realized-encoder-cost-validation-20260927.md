# Tri-State Realized E/P/D Cost Labels and Full24 Revalidation

## Outcome

The physical E-overlap observer now distinguishes three cases: realized E/P/D interval overlap, resolved no-overlap,
and unresolved overlap while an E interval is still open. Unresolved labels are deferred; they are not coerced to either
planned state or no-overlap and do not update GPU-cost or contextual pair models. The measured cost key uses the
realized interval state when valid. Candidate prediction continues to use the planned E-active state.

The tri-state binary was validated on all 24 Gemma/Cosmos workloads with three runs per workload. Every workload's
median throughput beat frozen vLLM and all 72 individual runs exceeded it. Compared with historical 9db3 Current,
the largest median regression is Gemma long-prefill at -2.6% (603.6 vs 619.5 tok/s; runs 594.7/626.7/603.6); every
other workload is within -0.3% or better. Gemma long-prefill also fell from 629.4 tok/s in the note 361 campaign.
The three-run spread (about 5%) exceeds the gap, so this is not yet a resolved regression, but it is the one cell
that must be rechecked before promotion.
This is a stronger throughput result than the preceding planned-state key iteration, but it is not a universal latency
win or output-quality approval.

## Label evidence

In measurement epoch 1 across the three-run full24 campaign:

- 27,491 P/D dispatch records were observed.
- 27,491 had resolved interval labels; 0 were deferred.
- Physical E interval overlap was present on 354 dispatches; the P/D candidate snapshot predicted E-active on 16.
- 338 cases therefore had a planned/realized E-context mismatch, concentrated in vision/mixed traces where an E
  stream interval overlapped an already selected P/D action.
- Text-only balanced and long-prefill actions were labelled no-E when no E interval overlapped them.

This is the mechanism the earlier Boolean-only key could not represent: E can join after P/D action formation. Cost
observations now use the physical interval that actually influenced GPU execution. The candidate's planned context is
still available separately for decision audit.

The observer compares epoch-relative CUDA event intervals recorded by `PhaseActivityTimelineRecorder`. It measures
stream-work span intersection; it does not establish concurrent SM residency. Also, `globalReferenceWorkMs` still comes
from the planned candidate. When planned and realized E contexts differ, the denominator used to compute normalized
compression may therefore describe the planned rather than realized interference context. Future work must validate or
reconstruct that reference before making a stronger causal claim.

## Three-run full24 comparison

Throughput is generated token/s median over three runs. `vs vLLM / vs 9db3` is percentage change from the frozen vLLM
anchor and the historical 9db3 Current median. Latency columns are percentage change from frozen vLLM for mean/p95;
negative means lower latency.

| Model / workload | tok/s | vs vLLM / 9db3 | TTFT mean / p95 | TPOT mean / p95 | E2E mean / p95 |
|---|---:|---:|---:|---:|---:|
| Cosmos / balanced | 4478.9 | +3.8 / +0.8% | -47.7 / -40.2% | +1.5 / +0.9% | -3.9 / -1.7% |
| Cosmos / bimodal | 2008.1 | +7.2 / +3.0% | +17.3 / +62.0% | -23.6 / -18.0% | -10.9 / -2.0% |
| Cosmos / decode-heavy | 5260.7 | +6.5 / +0.4% | -47.4 / -44.7% | -3.4 / -3.9% | -5.6 / -5.0% |
| Cosmos / late-vision | 2531.2 | +16.9 / +0.4% | -50.3 / -49.6% | -13.4 / -12.9% | -18.6 / -14.4% |
| Cosmos / long-prefill | 1319.5 | +17.4 / +0.6% | +2.6 / -6.7% | -28.4 / -27.1% | -15.6 / -16.9% |
| Cosmos / mixed | 1182.5 | +28.1 / +3.8% | -11.4 / -27.3% | -28.8 / -28.3% | -22.6 / -20.9% |
| Cosmos / multi-image | 314.0 | +28.8 / +0.8% | -0.4 / -24.6% | -35.8 / -42.9% | -21.4 / -22.2% |
| Cosmos / poisson | 2045.5 | +14.8 / +1.1% | -42.2 / -15.4% | -11.0 / -12.4% | -17.6 / -15.0% |
| Cosmos / short | 2404.4 | +17.5 / -0.2% | -49.5 / -23.6% | -0.4 / -4.4% | -22.2 / -17.3% |
| Cosmos / text-heavy | 2054.9 | +59.0 / +3.2% | -55.4 / -50.1% | +16.1 / -16.0% | -15.1 / -34.2% |
| Cosmos / vision-heavy | 734.2 | +27.2 / +3.0% | -10.9 / -14.3% | -38.8 / -37.0% | -26.0 / -21.8% |
| Cosmos / wave-drain | 98.0 | +2.2 / -0.1% | +0.3 / -27.4% | -34.9 / -34.7% | -20.9 / -21.4% |
| Gemma / balanced | 1269.9 | +64.6 / +0.6% | -35.4 / -13.2% | -35.2 / -30.8% | -35.8 / -35.1% |
| Gemma / bimodal | 844.3 | +40.7 / -0.3% | +23.1 / +13.8% | -18.6 / +12.4% | -24.8 / -28.6% |
| Gemma / decode-heavy | 1378.3 | +69.7 / +0.1% | -41.6 / -15.6% | -37.7 / -36.4% | -38.1 / -37.6% |
| Gemma / late-vision | 1527.1 | +54.1 / -0.3% | -0.6 / +53.8% | -39.9 / -39.6% | -38.3 / -39.9% |
| Gemma / long-prefill | 603.6 | +20.6 / -2.6% | +26.2 / +12.9% | -20.3 / -18.0% | -12.7 / -8.2% |
| Gemma / mixed | 763.2 | +8.4 / +2.6% | -11.3 / +20.0% | -6.6 / +4.0% | -7.1 / -2.6% |
| Gemma / multi-image | 412.3 | +8.1 / +6.4% | +72.9 / +131.3% | -10.5 / +17.8% | +3.7 / +16.7% |
| Gemma / poisson | 934.8 | +37.1 / +0.6% | +5.7 / +108.3% | -21.8 / -16.9% | -20.7 / -18.6% |
| Gemma / short | 856.3 | +50.9 / +0.3% | -38.9 / -11.7% | -21.2 / -4.5% | -27.6 / -23.9% |
| Gemma / text-heavy | 905.5 | +123.8 / +1.3% | -89.3 / -89.6% | -6.6 / -6.6% | -54.9 / -72.4% |
| Gemma / vision-heavy | 584.2 | +4.4 / +2.3% | +41.3 / +76.4% | -7.9 / +8.8% | +2.4 / -7.2% |
| Gemma / wave-drain | 97.1 | +4.9 / -0.2% | +42.8 / +54.1% | -58.6 / -56.9% | -37.9 / -36.9% |

Latency wins over vLLM across the 24 rows are TTFT mean/p95 15/24 and 15/24, TPOT mean/p95 22/24 and 19/24, and E2E
mean/p95 22/24 and 23/24. Throughput is the universal win in this fixed-output set; latency is not.

Across three runs, output-token hashes are identical for 13/24 workload traces and differ for 11/24. Token-count
completion is verified, but model output-quality approval remains unresolved.

## Validation and artifacts

- `.local/results/tri-state-encoder-label-full12-screen-20260927`
- `.local/results/tri-state-encoder-label-full12-additional-20260927`
- C++ `PhaseActivityTimelineTest.*`, `PhaseGlobalCostModelTest.*`, `PhaseQueueSchedulerTest.*`: 190/190 passed.
- Earlier Python runner contract suite: 53/53 passed. No Python source changed in the tri-state implementation.
- Frozen vLLM baseline reused because its engine/runtime/request contract is unchanged.

`PhaseServingRuntime` now creates an encoder-only interval recorder automatically for VLM serving when the caller did
not provide a full activity recorder. It records E intervals only; P/D use their existing CUDA start/done events, and
the encoder-only recorder prunes completed intervals after they can no longer overlap a later P/D action. The manual
`llm_phase_context_smoke` benchmark path still uses its explicit full activity recorder. The new production-facade
default is compiled and unit-tested, but has not had a performance run.

The existing HTTP benchmark is not a `PhaseServingRuntime` HTTP test: its backend uses the research
`llm_phase_context_smoke` IPC adapter. `PhaseServingRuntime` is a direct asynchronous C++ API, currently exercised by
`llm_inference --phaseServing`; it is not itself an HTTP service. A valid observer-overhead comparison must therefore
use the same direct API request path and engine, with an explicit control that disables realized-label consumption.
Otherwise the comparison combines event-recording overhead with the scheduler-policy change caused by the new labels.
The HTTP/IPC results remain a separate end-to-end serving measurement, not proof of the production facade's observer
overhead.

## Correction (2026-09-28)

An earlier revision listed Gemma long-prefill as +0.7% versus 9db3 and called Cosmos short (-0.2%) the largest median
change. Recomputing from both cited campaign summaries gives -2.6% for Gemma long-prefill; the other 23 rows and the
72/72 vLLM result were confirmed unchanged.

Two label-path defects found after this validation are fixed in the follow-up commit and are described in note 363:
the encoder-only recorder could retire an E interval during the prefill query of a D-anchored augmented dispatch
before the decode query saw it, and the planned serial reference was reused when the realized E context differed.
The full-mask recorder used by these campaigns never pruned intervals, so the first defect does not affect the table.

## Next work

1. Add a diagnostic A/B mode for the direct `PhaseServingRuntime` path that can record realized E intervals without
   feeding those labels to cost learning; compare disabled, shadow-recording, and active-label modes on the same VLM
   request batch. Keep the existing HTTP/IPC serving benchmark separate.
2. Measure event-recorder cost and history size as E interval count grows, including workloads where E progresses ahead
   of the first P/D dispatch. The current per-interval CUDA-event create/destroy path has not been shown to be cheap.
3. When planned/realized E context differs, validate the serial-reference denominator used by normalized compression;
   the current candidate reference may have been predicted under the planned context.
4. Once observer overhead, label authority, and reference semantics are validated, rerun the 3x full24 through the
   existing HTTP/IPC adapter versus frozen vLLM; report it separately from direct-API results and retain the
   semantic-output gate.
