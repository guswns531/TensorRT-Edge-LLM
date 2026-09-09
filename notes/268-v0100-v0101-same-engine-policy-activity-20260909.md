# v0.10.0 / v0.10.1 V1 Scalar: same-engine HTTP and CUDA activity

Follow-up: [note269](269-version-policy-activity-decomposition-20260909.md)
decomposes overlap accounting, actual selector decisions, D gap coverage and paired
request-class outcomes. It clarifies that lower calibration coverage did not block
Scalar authority and confirms direct encoder output for every measured E batch.

## 1. Question and scope

User requested a fresh v0.10.0 execution alongside current, including the existing
E/P/D/Copy masks. Both retained binaries were executed successfully on 2026-09-09.
This is **one diagnostic mixed HTTP run per version**, not a new full12 promotion
gate and not a reconstruction of the historical champion engine.

Retained results: `.local/results/v0101-forward-port/version-activity-20260909/`.
Commands, binary SHA256, repaired media paths/hashes and compatibility config are
in `commands.json` and `contract.json`. `comparison.json` holds metrics and action
counts. `version-timeline.svg` uses a common elapsed-time scale for both runs.

## 2. Exact comparison contract

- Cosmos-Reason2-2B FP16; V1 Scalar; independent E/P/D contexts.
- Same current text engine, native vision engine, embedding and checkpoint.
- P8/D64/E4, chunk128, stable slots80, KV256 pages of128 tokens (3584MiB).
- Same mixed32text/32vision HTTP trace: 64 requests, 22118 prompt tokens,
  2928 fixed output tokens; ignoreEOS, max client concurrency64.
- Same319 generic warmup requests; both319/319 successful, zero failed responses.
- Full phase telemetry plus CUDA activity recorder in both runs. Graphs OFF.
- Expired-decode candidate experiment OFF; legacy pair eligibility ON.
- Old binary: `.local/v010-forward-build/examples/llm/llm_phase_context_smoke`.
- Current binary: `.local/v0101-forward-build-make/examples/llm/llm_phase_context_smoke`.
- Source comparison: root `f4eb53e` versus forward-port `acc3a36`; binary identities
  are authoritative because the retained builds are not newly rebuilt from these
  source checkouts. Current build includes the previously documented local smoke
  telemetry edits. This is not a source-clean policy-only causal A/B.

Old text engine was deleted; its directory retains sidecars only. Initial old
runtime execution read the current engine but stopped at the missing
`tied_engine_contract` validation. A separate engine-view directory symlinks the
unchanged current files and supplies a freshly computed size/sample CRC and actual
FP16 embedding byte count. No validation was disabled, no engine rebuilt, and no
large model/engine copied. Both versions use this same compatibility configuration.
The v0.10.1 export version remains unchanged; old runtime emits a version warning.
Successful loading alone was not treated as sufficient: HTTP inference and output
identity were subsequently validated.

Both output SHA256 values match exactly:
`73155475f432ffd2f98e5840347b702466850a69581fa1b9698b675b22908085`.

## 3. Request-level results

Latencies are milliseconds. Each fresh column has n=1; not confidence intervals.
Frozen columns are historical references with a different engine/instrumentation
contract, not members of the fresh same-engine A/B. Frozen vLLM is reused, not rerun.

| Metric | Historical v0.10.0 V1 (note264) | Fresh old runtime | Fresh current | Frozen vLLM (note264) |
|---|---:|---:|---:|---:|
| output token/s | 1185.12 | 1089.37 | 1072.42 | 921.48 |
| TTFT mean | 692.51 | 724.23 | 714.79 | 874.56 |
| TTFT p95 | 1949.30 | 2262.66 | 2109.47 | 2541.43 |
| TPOT mean | 30.39 | 37.47 | 39.49 | 46.97 |
| TPOT p95 | 39.54 | 53.38 | 58.46 | 84.02 |
| E2E mean | 2123.84 | 2434.71 | 2497.23 | 3008.38 |
| E2E p95 | 2404.71 | 2631.29 | 2686.23 | 3140.85 |

Fresh current vs fresh old: throughput -1.56%, TTFT mean -1.30%, p95 -6.77%,
TPOT mean +5.39%, p95 +9.53%, E2E mean +2.57%, p95 +2.09%.
These are observations, not proof of a persistent regression. In particular the
old runtime on the current engine does not recover historical1185 token/s.
Engine, instrumentation, calibration realization and run variance remain mixed
in any comparison to that historical number.

## 4. Policy choices versus physical intervals

Final coordinator **decision events**, counted once per action after measurement
epoch (a pair has two dispatch events, so dispatch-event counts must not be used
as action counts):

| Selected action | Old runtime | Current |
|---|---:|---:|
| E | 7 | 8 |
| P | 25 | 31 |
| D | 77 | 81 |
| E+P | 3 | 3 |
| P+D | 9 | 3 |
| E+D | 0 | 0 |

Actual E dispatches10→11, P37→37, D86→84. E batch sequences:

- Old: 4,3,4,3,3,4,3,4,1,3 (sum32).
- Current: 4,3,3,3,3,4,3,3,3,1,2 (sum32).

All9 old and all3 current planned P+D actions have same-dispatch CUDA overlap;
missed planned P+D=0 on both. E+D has no explicit selected pair, yet current has
0.346ms E/D interval intersection. A tiny interval intersection is not equivalent
to a profitable selected overlap action or proof of a policy-invariant violation.

## 5. All sixteen masks

Bits: E=0001, P=0010, D=0100, C=1000. Percentages use first-to-last selected CUDA
work span: old2685.011ms, current2727.609ms. They exclude process initialization and
warmup and are not fractions of HTTP server lifetime or offered-load idle time.

| Mask | Recorded state | Old % | Current % |
|---|---|---:|---:|
| 0000 | no recorded E/P/D/C work | 3.384 | 3.638 |
| 0001 | E | 25.277 | 26.164 |
| 0010 | P | 32.039 | 34.557 |
| 0011 | E+P | 10.336 | 9.183 |
| 0100 | D | 23.712 | 25.166 |
| 0101 | E+D | 0 | 0.013 |
| 0110 | P+D | 5.252 | 1.280 |
| 0111 | E+P+D | 0 | 0 |
| 1000 | C | 0 | 0 |
| 1001 | C+E | 0 | 0 |
| 1010 | C+P | 0 | 0 |
| 1011 | C+E+P | 0 | 0 |
| 1100 | C+D | 0 | 0 |
| 1101 | C+E+D | 0 | 0 |
| 1110 | C+P+D | 0 | 0 |
| 1111 | C+E+P+D | 0 | 0 |

**Copy coverage caveat:** raw CSV has no copy intervals in either run, not merely
an analyzer filtering issue. The recorder covers `encoder_output_copy` when the
vision adapter cannot bind direct output storage; direct-output operation bypasses
that copy. It does not instrument every CUDA H2D/D2H/D2D transfer. Therefore C-bit
zeros mean no recorded dedicated output-copy interval, NOT no memory operations.
Likewise0000 is an instrumentation-defined idle state, not proven device-wide idle.
CUDA event-bounded intervals can contain internal waits; these masks do not measure
SM occupancy, kernel-level utilization, or bandwidth saturation.

Recorded E/P/D overlap418.533→285.747ms, or15.59%→10.48%.
D completion→next D start gap mean19.60→24.00ms, p95172.79→179.70ms,
max283.13→424.73ms. D interval total777.67→721.69ms nevertheless decreases.
Thus lower D GPU service sum does not guarantee better request TPOT: service
placement and inter-dispatch gaps also matter. This is a diagnostic association,
not a causal assignment of all TPOT loss to one mechanism.

## 6. What actually differs in policy architecture?

Both use V1 Scalar RLS, the same feasibility→SLO→efficiency selector family, and
the existing all-late efficiency recovery rule. The source-level contextual model
implementation `phaseContextualPdModel.cpp` is identical between the inspected
trees. The global selector diff is primarily audit collection; its ranking was
not replaced by the new additional-delay shadow metric.

Current queue/coordinator adds decode-dispatch blocking for exclusive encoder
execution and additional audits. The expired-decode preservation experiment is
opt-in and OFF here. These execution mechanisms, engine costs, and when completions
become visible can change the snapshots seen by an otherwise unchanged selector.
Do not describe this comparison as old static policy versus new RLS policy.

Most importantly, equal warmup inputs produced **unequal learned evidence**:

| Calibration status after319 successful requests | Old | Current |
|---|---:|---:|
| converged flag | true | false |
| calibrated / required exact keys | 5 / 5 | 1 / 3 |
| P+D safe probes | 35 | 25 |
| E+P probes | 23 | 18 |
| E+D probes | 0 | 1 |

The convergence flag is the existing coverage criterion, not a theorem about RLS
parameter convergence. Same successful warmup count does not freeze the posterior
or decision frontier. This is a strong confound for explaining9→3 P+D choices.

## 7. Reproduction and next steps

Run `benchmarks/phase_serving/compare_version_activity.py --output ABS_NEW_ROOT`.
The script saves the exact commands, materializes media repairs, creates only
small compatibility metadata and symlinks, then runs the old and new binary
sequentially. It must use a new output directory.

For each version run `analyze_phase_activity.py` with the recorded
`run-001/activity-events.jsonl`, `run-001/activity-intervals.csv`, `--request-count64`
(CLI spelling: `--request-count 64`) and an analysis output directory. Then run
`render_version_activity.py --root ABS_NEW_ROOT`.

Next useful controlled experiment: repeat fresh pairs in alternating order and
separately compare fixed319 warmup versus equal validated calibration coverage.
Keep the existing SLO values and engine constant. First separate evidence supply
from decision realization before changing policy thresholds. Extend to text-only
balanced/decode-heavy and vision-heavy after the mixed diagnostic is reproducible.
V0/V2 and the other11 workloads have not been rerun in this task.

No production policy, SLO setting, KV allocation, engine contents or old source
was modified. No artifacts were deleted. The saved command and plotting helpers
are research tooling; the new measurements do not promote a policy.
