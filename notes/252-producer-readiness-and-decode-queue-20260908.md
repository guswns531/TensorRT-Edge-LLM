# 252. Producer readiness versus decode queue delay

Date: 2026-09-08. Parent: `a524c73`, branch `codex/v0101-phase-forward-port`.
Artifacts: `.local/results/v0101-forward-port/ready-path-20260908/`.

## Scope and implementation

Note251 found that unified `completion_visible_host_ns` is an upper-level snapshot notice and
cannot measure first CUDA completion visibility. The producer already has a separate request timeline
in `independentPhaseAsyncServer.cpp`. Reuse it rather than introduce another runtime controller or
duplicate events. No runtime scheduling, KV, model, engine or binary changes were made this turn.

```text
sampling submitted [ticket sequence]
  -> readiness detected by cudaEventQuery / optional event synchronize
  -> sampling_ready (CPU begins processing the ready ticket)
  -> sampling_collected
  -> token_committed [per request]
  -> decode_ready [same producer ticket sequence]
  -> decode_start [next request-local dispatch]
```

Source contracts:

- `enqueueSamplingTicket()` assigns ticket sequence and records submission.
- `processSamplingTickets()` detects ready CUDA events, then drains the ready-ticket list.
- `completeSamplingTicket()` records `sampling_ready` just before processing, **not at the exact
  query instruction and not at GPU completion**. Multiple ready tickets can precede this callback.
- `processTicket()` records collect and per-row token commit. Finished requests do not reenter D.
- `enqueueDecodeOrWait()` ensures page capacity and records ready after enqueueing the request.
- `phaseDispatchWorker.cpp` records `decode_start` before its CUDA dispatch event/enqueue.

`analyze_encoder_serving_pair.py --request-timeline` now joins `(request ID, ticket sequence)` for
producer events and matches each ready transition to exactly one following request-local D start.
It rejects missing/duplicate/nonmonotonic stages and unmatched starts rather than silently clamping
or pairing unrelated rows. All durations are in the same monotonic host clock. Existing common-GPU-epoch
analysis remains separate. The terminal sampling tickets are retained without inventing another D start.

Phase host-span coverage unions intersecting intervals, avoiding double counting within a phase.
E/P/D spans can overlap each other, so do not add their coverage percentages. Coverage is **not GPU
utilization**: these host intervals include preparation, queued GPU work and callback observation.

## Experiment contract

Two fresh-process diagnostic runs, one native exact and one explicit erf, with the unchanged
Vision-heavy 64-request trace, 319-request generic calibration, V1 scalar, P8/D64/E4, slots80,
FP16 KV256 and client cap64. Engine hashes and no-instrumentation performance references are in note251.
Both runs completed 64 requests and 2464 generated/captured tokens, with no dropped producer stages.

Use the same `run_policy_warmup_matrix.py` command as note251's diagnostic runs, but set
`TRT_EDGELLM_PHASE_TELEMETRY_LEVEL=full` to enable request producer timelines; research mode does not
emit them. Also set `TRT_EDGELLM_EMIT_PHASE_METRICS=1`, distinct telemetry output and activity prefixes.
Resolved commands: `native/commands.json` and `erf/commands.json` under this result directory.

Full mode also collects detailed candidate snapshots. It perturbs the host path, so these are
diagnostic measurements, **not new production performance baselines** or a full12 promotion gate.
The per-engine log is about 25 MiB including calibration; analysis explicitly resets at the one
`PHASE_EPOCH` measurement boundary. No calibration or reused warmup request IDs enter the statistics.

Reproduce:

```bash
python3 benchmarks/phase_serving/analyze_encoder_serving_pair.py \
  --events <results>/native-events.jsonl --events <results>/erf-events.jsonl \
  --request-timeline --output <new-output.json>
```

The retained final output is `ready-analysis-coverage.json`; `ready-analysis.json` is the earlier
producer-only analysis of the same logs. No new GPU runs were used for the coverage extension.

## Producer versus queue delay

All numbers below are milliseconds. These are **request-token-row weighted** distributions, not
batch-weighted service costs. Shared ticket handling time appears once per row in that ticket.

| Stage | Native mean | Native median | Native p95 | Erf mean | Erf median | Erf p95 |
|---|---:|---:|---:|---:|---:|---:|
| Sampling submit -> CPU handling | 5.66791 | 2.55256 | 10.40546 | 3.53599 | 2.54512 | 8.86706 |
| Handling -> collected | 0.00591 | 0.00620 | 0.00830 | 0.00575 | 0.00606 | 0.00734 |
| Collected -> token committed | 0.03296 | 0.02333 | 0.08699 | 0.30245 | 0.02387 | 0.10673 |
| Token committed -> decode ready | 0.00133 | 0.00106 | 0.00336 | 0.00144 | 0.00108 | 0.00346 |
| Decode ready -> next D start | 23.87747 | 0.79670 | 179.86142 | 23.30470 | 0.79821 | 189.19711 |

Each run has 2464 complete sampling rows, of which 2400 reenter decode. The 64 terminal rows correctly
have no ready/start pair. All paired host durations are nonnegative. Excluding prefill-produced first
D rows leaves 2336 resident-decode transitions, with ready-queue mean/p95 of 21.858/151.387 ms for
native and 22.265/189.196 ms for erf: the queue effect is not only first-token admission.

The approximately 1.3–1.4 microsecond commit->ready mean rules out a large ordinary ready publication
cost in these traces. It does not establish zero page-pressure cost under a different memory regime.
Collection is also only about 6 microseconds after handling begins. The larger submit->handling
duration mixes sampling GPU work, D2H/event readiness, polling cadence and prior ready-ticket processing;
do not label it GPU-only sampling or GPU->CPU visibility latency.

Erf has a 42.06 ms maximum collected->committed interval, versus native 0.313 ms. Its median remains
about 24 microseconds. This interval includes per-row processing of preceding ticket rows and possible
host descheduling; it is not necessarily one token append taking 42 ms. It deserves a focused repeat,
but is not sufficient evidence for moving sampling onto another thread.

## What happens while a request is already decode-ready?

| Request-ready interval coverage, mean ms | Native | Erf |
|---|---:|---:|
| Encoder host spans | 17.272 | 13.468 |
| Prefill host spans | 11.311 | 10.658 |
| Other decode host spans | 0.325 | 0.012 |
| Uncovered by any E/P/D host span | 0.994 | 1.029 |
| Entire ready->start interval | 23.877 | 23.305 |

The union of E/P/D host spans covers about 95.84% / 95.58% of the row-weighted ready waiting time.
The individual phase means exceed the union when E and P spans overlap; they must not be summed.
This supports examining phase arbitration and outstanding work, rather than assuming a missing
ready-queue connection. It does not tell us which blocked requests could have safely overlapped.

Selected actions while the snapshot already has ready D rows:

| Action | Native decisions | Erf decisions |
|---|---:|---:|
| E | 14 | 14 |
| P | 41 | 42 |
| D | 107 | 102 |
| P+D | 2 | 3 |

These counts are observations, not automatically policy errors. E/P may be servicing tighter TTFT
deadlines; a non-preemptible phase and legal overlap frontier can prevent immediate D dispatch.
We must compare feasible alternatives and protected-request slack before forcing D priority.

## Diagnostic serving and frozen references

For transparency, full-telemetry runs below are kept separate from the uninstrumented three-repeat
reference from note251. Latencies are mean / p95 in ms; frozen vLLM contract is unchanged and reused.

| Variant | Runs | token/s | TTFT mean / p95 | TPOT mean / p95 | E2E mean / p95 |
|---|---:|---:|---:|---:|---:|
| Native, no telemetry (note251) | 3 | 632.67 | 1422.04 / 3373.02 | 32.82 / 39.22 | 2700.53 / 3819.44 |
| Erf, no telemetry (note251) | 3 | 642.30 | 1415.59 / 3350.61 | 31.65 / 39.01 | 2636.20 / 3761.69 |
| Native, full telemetry | 1 | 611.59 | 1518.44 / 3538.75 | 37.75 / 63.81 | 2967.37 / 3966.51 |
| Erf, full telemetry | 1 | 607.48 | 1487.09 / 3553.54 | 34.95 / 41.87 | 2839.42 / 3983.25 |
| Frozen vLLM | 3 | 579.20 | 1710.70 / 3691.37 | 63.70 / 119.58 | 4119.14 / 4229.37 |

Do not use the full-telemetry ordering to reverse the production conclusion from note251. Logging
and event collection perturb trajectory; their exact overhead is not isolated by a single run.

## Remaining instrumentation trap and next implementation

Detailed candidate snapshots also contain `max_slo_violation_us=0` by default. The current
`appendCandidate` in `phaseThreeCoordinator.cpp` copies protected completions but does not populate
that derived violation field. Therefore zero there is **not evidence that the candidate is SLO-safe**.
The global selector calculates violations separately. Furthermore, the unified frontier can contain
preview candidates outside the final selector input set. Do not infer a selector bug from these fields alone.

The next bounded step is to record the **actual selector input set, decision reason, and computed
protected violations** with lightweight producer timelines, without the full candidate/snapshot dump.
Then replay the high ready-wait decisions and determine whether D was infeasible, legitimately less
urgent, or lost through a cost/selection error. Only after that should scheduling authority change.
No vision-heavy-specific rule, hard D preference, sampling thread, TTL or new RLS model was added.

Current conclusion: ready publication is fast; much larger time is spent waiting after readiness
while other phases are active. The remedy, if feasible, belongs primarily in global phase-action
arbitration rather than KV resizing or rewriting the sampling completion transport.

Validation: both instrumented HTTP runs completed; all 12 analysis/timeline unit tests passed.
Project format/static checks passed after formatting. The unrelated C++ telemetry edit and draft
note248 remain separate. This turn makes no production speedup or new full12-workload claim.
