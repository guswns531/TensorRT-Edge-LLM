# Runtime service clocks and V2 diagnostic integration

## Implemented

- `cpp/runtime/phase/mechanism/phaseServiceClock.h` defines request ID, original
  submission and last token host-commit timestamps. Zero explicitly means unknown.
- `independentPhaseAsyncServer` records last-token commit when a collected token
  is appended to request state. Detailed arbitration snapshots expose service
  clocks for active server requests. Submission uses original scheduling admission
  when available, not the later vision-to-LLM admission time.
- `phaseThreeCoordinator` forwards those clocks and adds pending encoder and
  encoded-ready prefill request clocks without resetting first-token age at E→P.
  Prefix-admitted requests retain their server clock instead of a duplicate.
- Detailed `PHASE_SCHEDULER_EVENT` decisions serialize `service_clocks` alongside
  existing candidate, ownership and `scalar_formation` metadata.
- `service_normalized_shadow.py` validates ready-row coverage and timestamp
  ordering, derives first-token/next-token elapsed time, and reports existing V2
  evaluation coverage. It still refuses to score missing per-request alternatives.

These are host-monotonic **token commit** clocks, not CUDA end events, sampling
queue-ready times or HTTP delivery times. The decision event timestamp follows
snapshot collection on the same host clock. Submission may mean backend admission,
not a client's scheduled arrival. Request-level HTTP metrics remain separate.

The detailed clock vector is not a complete census of all requests outside the
server: unadmitted queues and encoder-inflight requests without a server entry are
not universally represented. The validator checks the explicitly advertised ready
rows. A complete policy producer must handle other protected states separately.

No selector, SLO, RLS feature/reward, KV allocation, engine or CUDA dependency was
changed. The extra host timestamp/store and diagnostic serialization have nonzero
overhead; policy-neutral does not imply timing-neutral. Clock fields are excluded
from existing logical snapshot hashes.

## Build and tests

Release SM86 build targets `llm_phase_context_smoke` and `unitTestRuntime` passed.
The initial host-user build failed to write existing container-owned dependency
files; rerunning with container write authority succeeded. No artifacts deleted.

- C++ filter `Phase*:*Independent*`: 327 run,326 pass,1 optional metadata benchmark skip.
- Python service-normalized tests:13 pass (including new runtime clock tests).
- Existing Python replay-contract tests:15 pass.
- Source pre-commit checks passed; no unrelated smoke edits were discarded.

Build environment: `nvcr.io/nvidia/tensorrt:26.06-py3`, `TRT_PACKAGE_DIR=/opt/tensorrt`.
Runtime library path includes `/opt/tensorrt/lib`, `/usr/local/cuda/lib64`, and
the build's `examples/llm` directory. Build base is `acc3a36` plus the documented
working changes; the two preexisting smoke telemetry edits remain present.

Binary SHA256:
`80a9625e1ba2291b42651d9ef8241ef0f8bfe1081163303829da12eafba9c181`.
Plugin SHA256:
`2a31e79d0d729202a005ccf08ad457bae9808ebe3d0b53e28505fb5322661566`.

## Fresh V2 HTTP validation

Retained result root: `.local/results/v0101-forward-port/service-clocks-20260909/`.
Exact commands and repaired asset identities are in `commands.json` and
`input-contract.json`; `coverage.json` is the post-run clock audit.

Same Cosmos FP16/native vision, P8/D64/E4, chunk128, KV256x128, full reservation,
graphsOFF, legacy pair eligibility, expired-D preservationOFF. One full-telemetry
mixed HTTP run with **V2 scalar-transition**; no new service-normalized selection.
Generic319 warmup responses all succeeded. Measurement64 requests,2928 tokens.
Output matches retained V1/old-version output exactly:
`73155475f432ffd2f98e5840347b702466850a69581fa1b9698b675b22908085`.

| Observability result | Count |
|---|---:|
| Decisions with service-clock snapshot | 115/115 |
| Ready-row clock checks passed | 5075 |
| Of those, ready-D token-commit clocks | 4041 |
| V2 formation evaluated / valid | 111/111 |
| Explicit guard audit present | 114 |
| Standalone D suppressed | 31 |
| D suppressed while both deadlines expired | 21 |
| Complete normalized counterfactual snapshot | 0 |
| New normalized-policy choices executed | 0 |

5075/4041 are repeated snapshot rows, not unique requests or token counts.
Existing V2's `valid` flag describes its aggregate bounded evaluation, not complete
request-by-request service projection coverage for the new prototype.

Final action counts: P28, P+D6, D71, E5, E+D2, E+P3. Final action ID differs from
recorded Scalar H1 ID16 times; this count alone is not proof of16 beneficial or
exclusively transition-caused changes.

Among ready-D snapshot observations, elapsed time since host token commit has mean
73.072ms and max520.353ms. These are decision-sampled ages, not request TPOT
percentiles. The maximum is at a D decision; preceding P decisions observe maxima
486.192ms and459.742ms. Thus a long D service-age episode also exists in this V2
run; V2 being evaluated frequently does not itself ensure service continuity.
It does not establish a persistent regression without repeated controlled runs.

## Diagnostic performance, not a promotion gate

| Metric | Fresh instrumented V2 |
|---|---:|
| output token/s | 1092.763 |
| TTFT mean / p95,ms | 676.681 /2200.363 |
| TPOT mean / p95,ms | 41.799 /67.553 |
| E2E mean / p95,ms | 2540.390 /2669.790 |

Historical same-engine V1 in note268:1072.421 token/s, TTFT714.789/2109.467,
TPOT39.490/58.464,E2E2497.229/2686.234. That was a different binary and separate
single run, so differences cannot isolate the V2 policy or instrumentation cost.
Frozen vLLM mixed reference remains921.48 token/s, TTFT874.56/2541.43,
TPOT46.97/84.02,E2E3008.38/3140.85 (note264 contract caveats apply).
vLLM was not rerun and these numbers are not a new fair headline comparison.

## Still missing / next

1. Produce a bounded, immutable mechanism-only shadow frontier before policy
   pruning. Today's clocks do not restore omitted D actions.
2. Attach canonical isolated reference costs with provenance. Do not substitute
   SLO targets, queue delays or chosen-action timings for unobserved references.
3. Generate valid request-local successor service estimates. Existing V2 aggregate
   oldest-D/protected estimates cannot simply be copied to every request. Preserve
   missing outcomes for unsupported horizons and residual paths.
4. Only then run the service-normalized evaluator over real decision snapshots.
5. Alternate repeated V1/V2 runs with and without diagnostic collection before
   drawing performance conclusions, followed by controlled alternatives and full12.

The runtime clock connection is complete for advertised ready rows in this smoke.
The full service-normalized runtime adapter, candidate preservation, counterfactual
validation and active policy remain unfinished. No claimed normalized-policy
speedup follows from this diagnostic run.

## Follow-up

The runtime adapter, mechanism-only candidate preservation, canonical reference
provenance, real shadow scoring, alternating-order overhead comparison, and fresh
full12 screen are completed in
[note272](272-service-normalized-shadow-and-full12-gate-20260909.md). A physical
same-state branch and any resulting active policy remain deliberately incomplete;
predicted shadow disagreement is not counterfactual speedup evidence.
