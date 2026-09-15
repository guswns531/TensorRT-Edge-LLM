# Lifetime-based encoded admission: Gemma and Cosmos

## Goal and scope

Replace the normal-path static encoded-request credit with an opt-in byte reservation based on actual
vision ownership. Preserve V3, fixed P128 chunks, engine profiles, KV reservation semantics and global
action selection. No workload labels, new SLO targets, capacity hysteresis or offline registry are added.

The previous 4/8/12 experiment is promising but is not an isolated lifetime result: capacity also changed
generic calibration acquisition. This campaign activates the experimental mode only after calibration
drains, and changes the static comparator's capacity at that same boundary.

## Mechanism

- E candidate admission reserves its known per-request payload estimate before asynchronous preparation.
- Completed E output is charged by retained physical storage, counting shared batch slabs once.
- Final P completion releases embeddings through the existing GPU-safe callback. M-RoPE storage still
  needed by D remains charged. Request metadata and KV leases remain alive independently.
- Pending, prepared and running encoder rows retain reservations until ownership transfers or cancellation
  completes. Cancellation must not reclaim a running GPU consumer's bytes early.
- Static mode is unchanged. Lifetime mode removes normal-path request-count credit; a separately derived
  metadata bound remains a capability limit, not a workload-selected window.
- An explicit byte ceiling remains supported. Otherwise a drained boundary samples free device memory,
  leaving one observed encoder-storage burst as reserve. This is a local admission budget, not an elastic
  CUDA allocator or a guarantee against external processes and unobserved temporary allocations.
- Unknown output sizes remain on the existing single-request bootstrap path. A candidate that cannot fit
  is not granted the legacy oversized-single-request exception in lifetime mode.

## Validation and comparison

Use one newly built binary per pair, unchanged model/engine/image identities, the same requests and generic
calibration. Run fresh HTTP real-request traces for Gemma INT4-AWQ and Cosmos FP16. Start with mixed,
vision-heavy and multi-image; add text-only controls when the VLM mechanism is validated. Compare the same
binary's static baseline, a static larger window, and lifetime admission; do not splice previous binaries.

Report output tokens/s, TTFT/TPOT/E2E mean and p95, peak memory, encoder wait, batch distributions, reservation
pressure and retained storage. Frozen vLLM is contextual and is reused only with unchanged request contracts.
First runs are diagnostic; selected pairs require repeats before promotion. Warmup counts alone do not
certify identical posteriors across runs, even when the activation procedure is identical.

## Trade-off selection

Evaluate each global configuration across both models without choosing settings by workload name. Faster
TTFT and E2E may justify higher TPOT; report the cost explicitly rather than imposing all-metric monotonicity.
Reject ownership corruption, OOM and persistent no-progress. Repeated E2E mean/p95 improvements and useful
throughput are primary evidence; streaming TPOT increases remain visible, not hidden in an aggregate score.
Do not invent a universal acceptable percentage before the two-model frontier is measured.

## Implementation order

1. Pure admission and retained-storage accounting tests.
2. Coordinator lifetime admission and drained activation; production config plumbing.
3. Build and GPU-safe completion regression tests.
4. Gemma and Cosmos static/lifetime HTTP comparisons with manifests.
5. Repeat useful trade-off points, document negatives, and decide whether a full-12 promotion is warranted.

Implementation and repeated results are recorded in
[302: two-model results and trade-off assessment](302-lifetime-encoded-admission-two-model-results-20260913.md).
The additional static-slot comparator uses the engine's stable-slot limit, not a fitted workload window.

## Artifact retention

The active filesystem has approximately 2 GiB free, so no new engines are built for this campaign.
The only compression allowlist is `run-001/gateway.log` in cells already listed in the campaign manifest's
`completed` array and containing a successful aggregate. These logs may be losslessly compressed to `.gz`
after analysis; running logs, models, engines and source worktrees are never touched. The analyzer supports
both forms. This preserves raw evidence while avoiding multi-gigabyte full-telemetry duplication.

The interrupted preliminary campaign's closed failure log at
`.local/results/lifetime-encoded-admission-20260913/primary-3x/cosmos/static-large/repeat-001/vision-heavy/run-001/gateway.log`
is also explicitly allowed for lossless compression. Its OOM and exit139 conclusion remains in `status.json`;
none of that preliminary binary's results are spliced into the corrected paired campaign.

In the corrected campaign, a log listed in the manifest's terminal `failures` array is also eligible for
lossless compression after its error lines are retained in `failure-analysis.json`. Failed cells are never
included in successful latency/throughput averages; report attempted and successful repeat counts separately.
