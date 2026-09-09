# 267. Same-state branch replay: readiness gate and full capture

## Outcome

The intended P2 versus P2+D48 physical comparison is **not completed**. We implemented the prerequisite validation and full capture, then ran mixed three times. The original lightweight audit cannot restore its state, and the exact eligible P2/P2+D48 frontier did not recur in any of the three full captures. No alternative branch was executed; there is no measured4.77ms improvement.

This is a concrete reproducibility limitation, not evidence that the alternative is slow or the shadow model is wrong. Production selection and SLO values remain unchanged. No new forced action, model or scheduler threshold was added.

## Why the old target cannot be replayed

Note266's promising ON mixed decision9223372036854776328 was captured in audit mode. Its state summary was P-ready3, D-ready48, E-ready11, D context tokens12713, outstanding mask0; the actual P request IDs were28 and44. However the candidate shape is not the full state.

The new checker reports six deficiencies:

1. KV ownership signature missing/zero.
2. Vision lease signature missing/zero.
3. P-ready request/length rows incomplete.
4. D-ready request/context rows incomplete.
5. Full candidate snapshots absent.
6. The actual selector inputs cannot be joined to materialized candidate snapshots.

Its nonzero strict snapshot signature does not repair the missing ownership/row data. Zero signatures are treated as unavailable, not as equal empty ownership. The old readiness report is retained at `.local/results/v0101-forward-port/all-late-shadow-20260909/replay-readiness.json`.

## What was implemented

- `replay_retained_policy_commands.py`: exposes existing runtime `--telemetry-level full`, recording full candidate/ownership/ready metadata plus request timelines. Previously the wrapper allowed only dispatch/audit even though the runtime supported full.
- `inspect_branch_replay.py`: inspects actual eligible, hard-feasible, non-dominated selector inputs, matches P/P+D shapes, joins candidate snapshots by action ID, validates request membership and row/length alignment, and groups complete metadata fingerprints across files.
- Default target is P2/D48. `--any-decode-batch` is an explicitly broader diagnostic; its output is not passed off as the original D48 target.
- Every result remains `runtime_replay_ready=false`: metadata capture alone cannot restore GPU KV/vision tensor contents, the complete posterior, or phase-local bindings. It never treats matching metadata as a physical checkpoint.
- Tests cover unavailable ownership, row alignment, shape filtering, and complete metadata still failing to constitute a physical checkpoint. Python15 tests passed. This turn changed Python tooling only; no C++ source or engine rebuild was required.

## Three-run experiment

Artifacts: `.local/results/v0101-forward-port/branch-readiness-20260909/`.

Same note266 binary SHA256 `ca42e909772ed8ae18537888058f53759bc78ef8d622d608eb82bca7159379b6`, native vision/text engines, V1 scalar, P8/D64/E4, chunk128, FP16 KV3584MiB, graph OFF, legacy pair eligibility ON, expired-D restoration ON, full telemetry. Each fresh process receives319 generic warmup requests, then64 mixed measurement requests. The underlying code remains parent `b0f9def` with previously retained compact telemetry worktree edits; new changes are replay/analysis tooling.

All three warmups validated. All three runs generated2928 tokens with the identical full output hash `73155475f432ffd2f98e5840347b702466850a69581fa1b9698b675b22908085`.

| Repeat | Measurement decisions | Exact eligible P2/D48 opportunity | Broader eligible P2/D batch |
|---|---:|---:|---|
| 1 | 118 | 0 | D28, one final-selector opportunity |
| 2 | 114 | 0 | D43, D44, D41, three final-selector opportunities |
| 3 | 137 | 0 | none |

The four broader opportunities passed the implemented metadata checks. There were zero repeated metadata fingerprints across runs. This checks equality of the recorded metadata, not similarity of shapes. Since full mode adds observation overhead and the posterior/arrival realization is not frozen, nonrecurrence does not establish that P2/D48 is naturally rare under uninstrumented serving.

Reports: `readiness.json` for strict D48 and `broader-p2-readiness.json` for the explicitly broader query. Full raw captures occupy about97MiB and are retained because these conclusions depend on their candidate and ownership records. No engine/model deletion or duplication was performed.

## Serving numbers: capture validation, not branch performance

Full capture uses the actual existing policy. Means are arithmetic run means; p95 and throughput are medians of run statistics. Different instrumentation means these rows must not replace the primary benchmark.

| Mixed dataset | token/s | TTFT mean/p95 ms | TPOT mean/p95 ms | E2E mean/p95 ms |
|---|---:|---:|---:|---:|
| Full capture,3 runs | 1082.01 | 761.26/2117.87 | 35.99/54.66 | 2434.88/2638.67 |
| Retained primary Current OFF,3 runs | 1115.26 | 766.81/2149.76 | 36.17/54.72 | 2435.14/2580.39 |
| Retained primary frozen vLLM,3 runs | 921.48 | 874.56/2541.43 | 46.97/84.02 | 3008.38/3140.85 |

Full-capture peak memory median9635MiB; throughput range1043.31–1094.91token/s. Frozen vLLM is unchanged context from note264, not a measurement of the unexecuted shadow branch. No fresh vLLM or full12 run occurred.

## Required next implementation

Independent repeats alone do not provide the desired controlled alternative. A runtime barrier/checkpoint facility is still required:

1. Choose a quiescent decision boundary with the actual requested candidate frontier. Freeze admission and completion visibility before branching; store pending-arrival order and remaining relative deadlines.
2. Capture request state, per-phase canonical bindings, KV slot/page leases and generations, live KV data, vision payload data/leases, and all cost/policy/probe state. Hashes alone are insufficient; add corresponding restore operations.
3. Execute branch A through a defined equal-work/reconvergence boundary, drain its outstanding GPU consumers, restore the captured state, verify ownership/binding/content identities, then execute B. Rotate A/B order over repeats.
4. Record request-local completion, cohort sequence, D dispatch count, GPU work, and ownership lifetime; reject runs with a different pre-branch state. Do not reinterpret merely similar metadata as equality.

Prefer a bounded in-memory checkpoint of the live selected working set, not duplicating a full engine or writing several GiB into the nearly full `.local` directory. This design must preserve live request ownership; it is not safe to bolt a forced action onto the production loop and label the result same-state replay. An isolated deterministic fixture can establish a controlled shape effect first, but would remain distinct from reproducing note266's natural state.

The current main gain is an explicit fail-closed experiment gate and working full capture. The intended physical counterfactual and promotion remain open.
