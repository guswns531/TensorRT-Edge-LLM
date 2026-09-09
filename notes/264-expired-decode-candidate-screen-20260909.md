# 264. Expired decode candidate: guarded feasibility screen

## Hypothesis

Note263 found that faster isolated encoder execution did not reliably improve resident decode service: vision-heavy ready→next-D host latency increased while first-D latency improved. This is evidence for investigating scheduling, not proof of a single cause.

The global P/D candidate generator suppresses standalone D whenever the prefill TTFT hard guard is enabled and the minimum P slack is nonpositive. It does so even if D TPOT slack is also nonpositive. A later global selector cannot recover a candidate removed upstream.

## Minimal opt-in experiment

`PhaseQueueSchedulerConfig::preserveExpiredDecodeCandidate` retains standalone D when both deadlines have expired. It does not force D selection. Existing protected completion estimates, ownership/memory constraints, overlap eligibility, batching, and global ranking still apply. Legacy queue policy is unchanged; the default is false.

- Implementation: `cpp/runtime/scheduling/phaseQueueScheduler.{h,cpp}`.
- Example configuration: `TRT_EDGELLM_PRESERVE_EXPIRED_DECODE_CANDIDATE=1` in `examples/llm/phaseSchedulerOptions.inc`. Like the existing guard, presence enables it; omit it to disable.
- Replay: `--preserve-expired-decode-candidate` records the explicit environment in retained commands.
- Tests: enabled/disabled × expired/nonexpired D with expired P; previews must not consume queue members; legacy hard guard remains intact.

No workload label, new learned estimator, deadline threshold, KV change, engine change, or admission change is introduced. The optional outer P/D frontier remains disabled, matching the baseline.

## Validation contract

Same newly built Release binary for flag OFF/ON, same native vision/text engines and generic warmup as note262; V1 scalar, P8/D64/E4, chunk128, FP16 KV3584MiB, fixed output, graph OFF. Warmup HTTP responses must validate. Begin with mixed and vision-heavy, three repeats per flag; report throughput and TTFT/TPOT/E2E mean and median-of-run-p95. Frozen vLLM is retained only under the unchanged contract.

This is a screening experiment, not a full12 promotion. Candidate eligibility alone may not improve global decisions; a failed screen must not be promoted or described as a resolved regression.

## Results and disposition

Completed 12 HTTP runs: two workloads × OFF/ON × three fresh processes. All 12 warmup receipts validated; each used 319 requests. Full fixed-output token hashes match across all three repeats and both settings for each workload. Mixed generated2928 tokens/run; vision-heavy generated2464 tokens/run (verify per-run summaries when reusing the trace). The option is **not promoted** and remains false by default. No full12 candidate campaign was run after this failed latency screen.

Artifacts: `.local/results/v0101-forward-port/expired-decode-20260909/`, with OFF/ON command manifests, input hashes, client results and provenance. The new binary SHA256 is `3d18b3340f136ae8c6eec431a5e4c4cc3d896335ac7426996c9004fd3e6912a5`; plugin unchanged. Existing unrelated compact-telemetry edits in `llm_phase_context_smoke.cpp` were preserved in both measurements. Parent commit is `c54b934`, plus the patch documented here.

Unit validation: Release build succeeded; Phase/Independent C++ tests324 passed, one optional metadata benchmark skipped. Python replay tests10 passed. Changed-file pre-commit and diff whitespace checks passed.

### Serving comparison

Throughput and p95 are medians over three run statistics; mean latency is the arithmetic mean of run means, not a pooled p95. Old V1 has one retained historical repeat, frozen vLLM three. Historical rows are context, not a fresh same-binary causal control.

| Workload | Variant | token/s | TTFT mean/p95 ms | TPOT mean/p95 ms | E2E mean/p95 ms |
|---|---|---:|---:|---:|---:|
| mixed | old v0.10.0 V1 | 1185.12 | 692.51/1949.30 | 30.39/39.54 | 2123.84/2404.71 |
| mixed | OFF, fresh | 1115.26 | 766.81/2149.76 | 36.17/54.72 | 2435.14/2580.39 |
| mixed | ON, fresh | 1096.20 | 778.05/2119.78 | 36.68/55.30 | 2454.73/2609.69 |
| mixed | frozen vLLM | 921.48 | 874.56/2541.43 | 46.97/84.02 | 3008.38/3140.85 |
| vision-heavy | old v0.10.0 V1 | 687.95 | 1334.86/2890.16 | 41.45/65.29 | 2955.64/3402.72 |
| vision-heavy | OFF, fresh | 646.24 | 1407.91/3313.33 | 34.63/55.05 | 2814.33/3652.17 |
| vision-heavy | ON, fresh | 660.44 | 1426.07/3227.50 | 38.04/62.49 | 2952.84/3613.98 |
| vision-heavy | frozen vLLM | 579.20 | 1710.70/3691.37 | 63.70/119.58 | 4119.14/4229.37 |

| ON relative to OFF | mixed | vision-heavy |
|---|---:|---:|
| Throughput | -1.71% | +2.20% |
| TTFT mean / p95 | +1.47% / -1.39% | +1.29% / -2.59% |
| TPOT mean / p95 | +1.40% / +1.07% | +9.87% / +13.50% |
| E2E mean / p95 | +0.80% / +1.14% | +4.92% / -1.05% |

Peak GPU memory medians: mixed OFF9601/ON9623MiB, vision-heavy OFF9659/ON9649MiB. KV reservation and engine identities are identical; these small peak differences do not establish a KV allocation change.

### Interpretation and limits

1. Candidate suppression is a verified code behavior; its correction alone is **not** a verified performance fix. The unit test proves D preview eligibility under both expired deadlines, not that production dispatch always chooses it.
2. Vision-heavy throughput improves slightly while resident-token latency becomes worse. Throughput-only promotion would hide the TPOT regression. Mixed does not gain throughput.
3. All six reported serving metrics remain better than retained vLLM on these two traces, but this was already largely true with OFF. It does not establish a gain from this patch or an all12 win.
4. Runs were grouped OFF then ON, not interleaved ABBA; three repeats are screening evidence, not a confidence interval or attribution proof. OFF mixed throughput ranged1016.07–1127.90token/s, ON1080.98–1104.14token/s. Small changes may be run variance.
5. Both variants receive identical warmup requests, but the option is active during calibration too. Their resulting posterior and overlap observations need not match. This is an end-to-end policy-contract comparison, not a frozen-posterior selector-only experiment.
6. These non-instrumented primary runs do not count how often the new eligibility condition fires, how often the recovered D candidate wins, or why it loses. Do not infer a specific formation/overlap causal chain from the serving numbers alone.

## Next bounded investigation

Keep the production default OFF. Add diagnostic counters for both-expired encounters, recovered D candidates, selected D, and post-selector dispatch outcome; retain request membership and residual E/P state. Use matched OFF/ON diagnostic runs separately from primary timings. Split D waiting into active E/P blocking versus candidate selection versus completion/host submission. Only after determining which mechanism dominates should another policy change be proposed. If calibration feedback is suspected, compare a common frozen posterior as a separate contract; do not silently mix it into the generic-learning benchmark.
