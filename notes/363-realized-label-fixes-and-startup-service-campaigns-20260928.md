# Realized-Label Fixes, Startup Decode-Service Campaigns, and Record Corrections

## Summary

A read-only review of the 2026-09-27/28 commit series found one confirmed label defect in the encoder-only activity
recorder, several planned-versus-realized context mismatches, and analyzer defects that could silently undercount.
They are fixed with unit tests. The 2026-09-28 startup decode-service campaigns are recorded here for the first time:
measured host decode-service selection, whether active from startup (`5a9b417`) or activated only at the calibration
boundary (`1ae84b3`), does not beat the plan-only/observe-only control, and it regresses Cosmos balanced by 5-9%.
Neither variant is promoted. The serving default remains the 9db3 binary.

## Runtime fixes

1. **Encoder-interval retirement order.** `PhaseActivityTimelineRecorder::overlaps()` used to erase, in encoder-only
   mode, every completed E interval ending before the queried action start. `PhaseDispatchWorker` queries prefill
   first. For a D-anchored augmented dispatch (D starts, E runs and ends, residual P starts), the prefill query erased
   the E interval before the decode query ran, so an E-contended action was labelled a valid no-overlap sample.
   Queries no longer mutate state. A separate overlap window is retired once per dispatch, behind the earlier of the
   two phase starts. This is safe because each coordinator has one dispatch worker, and it serializes actions.
   Full-mask recorders now use the same pruned window for queries while keeping full export history, so query cost no
   longer grows with run length. The full24 campaigns in notes 361/362 used the full-mask recorder, which never
   pruned, so their tables are unaffected. The default encoder-only VLM path (`5a9b417`) was affected but had not been
   performance-run.
2. **Decode-service isolation.** Decode host-service samples were filtered on the planned E state only. A realized E
   overlap (E starting after selection and finishing before sampling) could enter the no-E service bucket. Realized
   overlap now rejects the sample too. The service key comparison uses action shape, ignoring the E flag, so a
   planned-state toggle between selection and dispatch no longer drops samples silently.
3. **Calibration tracking.** Warmup calibration keys took the planned E flag, while observations are stored under the
   realized flag. A shape planned with E but realized without it never became calibrated and occupied a second key
   slot. Calibration keys are now E-agnostic. Coverage counts a resolved diagnostic from either realized context.
4. **Serial reference under context mismatch.** The planned `globalReferenceWorkMs` is used only when the realized
   key, including the E context, equals the selected key. Otherwise the observation falls back to the measured
   component sum. This closes note 362 open item 3 by construction; it was not separately A/B validated.
5. **Diagnostics and contracts.**
   - The decode-candidate diagnostic now reads the same unflagged service bucket as selection.
   - `selectPrefillBatchSize` is no longer `noexcept`, because trace capture allocates.
   - Delayed measured-decode activation (`TRT_EDGELLM_MEASUREMENT_DECODE{,_SERVICE}`) now requires IPC mode instead
     of silently doing nothing.

"Deferred" E-label observations are discarded, not replayed. The field keeps its name for telemetry compatibility and
is now documented as such.

## Analyzer fixes

- `analyze_prefill_arrival_coupling.py` reported no first mismatch when one side had extra prefill dispatches after
  an identical prefix. It now reports a `length_mismatch`.
- `analyze_decode_partition_realization.py` now sorts by dispatch start and windows only decode-bearing metrics.
  Previously, log order and prefill-only records could undercount realized partitions.
- `run_ordered_phase_gateway.py` stalled every successor for the full timeout after one failed or missing request.
  Failures now fail successors immediately until the next control boundary.
- Analyzers prefer `gateway.log` over `.gz`. `gzip` removes the plain log only after a complete write, so a surviving
  plain log is authoritative.
- `run_async_decode_trials.py` validates `branch_turn` and flags branch timing that includes logit inspection.
- `run_decode_equal_work_trials.py` records the mode and mode-specific note references.
- Duplicate SPDX blocks were removed from ten files.

Validation:
- C++ `unitTestRuntime`: 726 passed, 2 optional skipped. Four tests are new or updated:
  - `PhaseActivityTimelineTest.OverlapQueriesKeepEncoderIntervalsUntilActionRetires`
  - `PhaseActivityTimelineTest.EncoderOnlyCaptureSkipsPDCEventAllocation`
  - `PhaseRuntimeCostTrackerTest.OverlapCalibrationAcceptsEitherRealizedEncoderContext`
  - `PhaseGlobalCostModelTest.ActionShapeIgnoresOnlyEncoderBackground`
- `unitTestRuntimeState` 82/82 and `unitTestExamples` 2/2 passed.
- Python phase-serving unit tests: 90 passed.
- No GPU serving campaign has been run on the fixed binary yet.

## 2026-09-28 campaigns

All four campaigns use the same Gemma/Cosmos engines, generic calibration, traces, and frozen vLLM references as note
362. All are `diagnostic`.

| Campaign | Binary | Comparison | Result |
|---|---|---|---|
| `startup-service-holistic-20260928/full24-screen` | `5a9b417`, `6ff2e703` | startup measured service vs plan-only shadow, 24 cells x 1 | geomean Cosmos -0.50%, Gemma -0.35% |
| `startup-service-holistic-clean-20260928/full24-3x` (+ `gemma-short-supplement`) | `5a9b417`, `e76b924c` | same, x3 | geomean Cosmos -0.85%, Gemma -0.28%; Cosmos balanced median -8.0% (-8.0/-8.8/-5.4), Gemma vision-heavy -4.1%, multi-image -4.1% |
| `startup-service-boundary-20260928/balanced-2x` | `1ae84b3`, `aa0178c4` | observe at startup, activate at boundary vs observe-only | Cosmos -5.4% (TPOT +6-9%), Gemma -0.65% |
| `startup-service-boundary-20260928/gemma-selected-2x` | `1ae84b3` | same, Gemma mixed/vision-heavy/multi-image | interrupted by user request at 8/12 cells; `scratch`, no conclusion |

Using measured host decode service for selection gives a consistent Cosmos balanced regression, with TPOT rising
more than TTFT falls. The observation/selection split in `1ae84b3` remains useful as an instrument, but the boundary
activation variant should not be pursued without first explaining the Cosmos balanced partition change.

### Provenance

- Both `5a9b417` binaries report a clean tree, yet their hashes differ. Their only differences are the ELF build ID
  and four `__LINE__` bytes in `phaseActivityTimeline.cpp` (`ELLM_CHECK` at 419 versus 418 in the commit). The screen
  binary was therefore built from a tree one line longer than the commit, most likely before formatting. It is
  behavior-equivalent, but its result manifest overstates cleanliness. Backfilled baseline manifests record this.
- The 1/144 failed cell (`gemma/shadow/repeat-002/short`) was a backend SIGSEGV (exit 139) during startup, before
  calibration:
  - The kernel log shows `segfault at 3`, resolved to `std::vector<nlohmann::json>::emplace_back` in
    `llm_phase_context_smoke`.
  - The same process logged `Loaded 514905 BPE merge priorities` with `Unrecognized merge entry format at index
    95652`. All 80 other inspected Gemma runs loaded 514906 merges with no warning.
  - The already-parsed tokenizer JSON DOM was therefore corrupted in memory, not misread.
  - The tokenizer code path is single-threaded and benign. The corruption source (a stray asynchronous device-to-host
    write into reused heap memory, another thread, or a host fault) is unresolved.
  - Next step: repeat readiness under `compute-sanitizer --tool memcheck` and an ASan build until the merge count
    diverges again.
  - Two supplementary Gemma short repeats completed normally.

## Record corrections

- Note 361 cited `observed-encoder-interval-full12-*`. Its table actually matches
  `realized-encoder-label-full12-{screen,additional}-20260927`. The citations are corrected, and the stale
  "not always-on" caveat now points to note 362.
- In note 362, Gemma long-prefill versus 9db3 is -2.6%, not +0.7% (603.6 vs 619.5 tok/s; runs 594.7/626.7/603.6). It
  is the largest median regression; every other workload is within -0.3%. The three-run spread exceeds the gap, so
  this cell must be rechecked before any promotion.

## Open items

1. Run full24 x3 on the fixed binary (encoder-only default recorder path) against frozen vLLM and 9db3. Report the
   Gemma long-prefill cell explicitly.
2. Measure the observer overhead: disabled vs encoder-only vs full-mask on the same VLM batch.
3. Output-quality gate: 11/24 traces still have differing output hashes across runs (note 362). Promotion remains
   blocked.
4. The startup SIGSEGV and tokenizer DOM corruption above.
5. Greedy divergence real-prompt replay (note 359), trusted-P forced-branch comparison (note 355).
