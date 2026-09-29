# Scheduler Redesign: Overlap by Default, Event-Driven Decisions, One Pricing Path

## Status

The head-off full24 x3 falsified the central premise below ("overlap always pays; the learned head is net
harmful"). Disabling the contextual P+D head wins only on Gemma long-prefill (+5.1%) and loses on 12 of 24 workloads
with non-overlapping run ranges (geomean -1.82%), driven by higher TPOT. Overlap-by-default and the head-removal
stages are withdrawn; see "Head-off full24 x3". Stage 1 (decision elision, `9ba95bc`) is independent of that premise
and is being validated. The overlap-first override (Stage 2a) was implemented, found to be blocked by the same
unknown-cost gate in the feasibility oracle, and reverted without a run because the head-off result already shows
that admitting more overlap raises TPOT on most workloads.

## Evidence that motivates the redesign

1. **The learned P+D head is net harmful on Gemma long-prefill.** Same binary (`c7a1b71`), 8 interleaved blocks,
   `TRT_EDGELLM_CONTEXTUAL_MIN_OBSERVATIONS=1000000` (head never ready):

   | Arm | Median tok/s | Range | P+D dispatches | Decisions | TTFT mean | TPOT mean |
   |---|---:|---|---:|---:|---:|---:|
   | base | 620.8 | 612-625 | 98 | 58,469 | 668 ms | 28.3 ms |
   | head off | **638.0** | 636-649 | **167** | 11,579 | 698 ms | 26.8 ms |

   +2.8% over base and +2.5% over 9db3's 622.6, with a fifth of the variance. Overlap count rises 70%. The head was
   suppressing measured-profitable overlap (notes 365/366). Full24 confirmation pending.
2. **Overlap is a hardware fact, not an unknown to learn.** Measured P 19 ms + D 18 ms run together in 18.6 ms
   (note 365). The current design spends most of its machinery (per-shape sample windows, probes, calibration, the
   contextual head, authority arbitration) deciding whether overlap pays. It always does on this hardware unless E is
   competing.
3. **The host decision loop is a spin loop.** 2.0M polls and 58-77k selector calls per 9 s run for ~500 dispatches;
   `previewMechanismPlan` deep-copies the whole scheduler two or three times per selector call. GPU idle is 1.5%, so
   this costs TPOT and augmentation latency rather than throughput directly.
4. **Actions complete atomically.** `PhaseDispatchWorker::poll` completes a P+D action only when both phase events
   are done, so after P finishes its stream idles until D finishes and nothing can be enqueued. Residual augmentation
   only handles single-phase -> two-phase. This bounds the overlap count and is the likely reason 9db3's 110-124
   overlaps were never exceeded until the head was disabled.
5. **Provenance audit (notes 300-366).** Mechanisms with a recorded failure behind them: probing in some form
   (self-lock, 337), common-frontier residual accounting (337), realized E labels and E-agnostic calibration keys
   (361-363), measured cost outranking any learned price (366), producer-class P compatibility and cohort refill
   (300/308), physical ownership admission (302/333). Mechanisms with no motivating failure or negative evidence:
   probe interval, TPOT hysteresis/decode guard (304/306 negative), TTFT hard guard (bypassed in V3),
   `producerCriticalPath` veto, transition-predictor controller, static D table and clamp, headroom admission,
   planned E-active key, measured host-service path (363), contextual head as authority.

## Target design (superseded; kept for the record)

- **Overlap by default.** When P and D work are both runnable, form P+D. Not overlapping needs a reason: predicted
  TTFT/TPOT violation under an explicit SLO, memory infeasibility, or E active with an interference coefficient that
  makes the pair slower than serial. Probing exists only to measure interference coefficients, never to discover
  whether overlap pays.
- **Low-dimensional cost model.** P cost = f(tokens, context); D cost = g(batch, context); pair makespan =
  max(P, D) * (1 + k_E?), with two interference coefficients (E idle, E active). Fitted online from every dispatch;
  no cold start per shape, no sample windows per key. Static tables are removed except a single largest-available
  fallback used before the first fit.
- **Explicit prefill merge policy.** Merge-wait tau is a first-class knob (fraction of the TTFT budget), decided
  independently of the P+D decision. Prefill us/token was 101 in the best runs versus 105-110 elsewhere.
- **One objective.** Candidates (D-only, P-only, P+D, merge-wait) are scored by one function: expected token
  throughput until the next decision point, with hard penalties for predicted SLO violation, all priced from the
  same model. No authority layers. Decision records carry the four scores.
- **Event-driven decisions.** Re-decide on arrival, completion, E completion, page release, or merge-wait expiry.
  Stage 1 approximates this by eliding repeated no-action decisions until state changes.
- **Kept invariants.** Physical ownership decides feasibility, the policy alone decides selection (304); queue
  residence is never a physical cost (331); unresolved labels are discarded (362); one dispatch worker per
  coordinator (363); missing coverage never becomes a confident decision (307).

## Stages

1. **Decision elision (implemented, pending validation).** `PhaseQueueScheduler::markStateChanged()` bumps
   `mStateEpoch` on every mutation the selector observes. `next()` reuses a no-action outcome while the epoch is
   unchanged and `noActionRevisitUs` (default 1 ms) has not elapsed; telemetry `globalElidedDecisionCount`, summary
   field `elided=`. Env `TRT_EDGELLM_DISABLE_DECISION_ELISION`, `TRT_EDGELLM_DECISION_REVISIT_US`. Behavior is
   intended to be identical; validation is unit tests plus a long-prefill A/B where `decisions=` must fall and tok/s
   must not.
2. **Per-phase completion.** `PhaseDispatchWorker` already has separable `completePrefillInFlight` and
   `completeDecodeInFlight` (the deferred-decode path uses one alone), but it holds a single `mInFlight` plan and
   one `mCurrentMetrics`, and `poll()` waits for both events. Retiring P early is a few lines; letting the idle
   stream start new work needs a second in-flight slot (one per phase stream) with its own events and metrics, and
   the action-level P+D observation (makespan, reference work, realized E label) has to become an overlap-window
   record derived from the two slot records. `augmentNext` then fires on "slot became idle" instead of on
   queue-count change. This is the structural ceiling on overlap count; it also changes the `observeMetrics`
   contract that every cost model consumes, so it lands after the policy seam, not before.
   **Stage 2a (cheap, in-binary experiment):** an `overlapFirst` override after legacy candidate construction: if
   E is idle and the P+D candidate passes the same feasibility/deadline oracle (`PhaseGlobalScheduler::select` on
   a one-candidate frontier), select it regardless of price. This tests overlap-by-default directly against the
   head-off arm without a new cost model.
3. **`FormationPolicy` seam.** Extract the body of `selectGlobalQueueAction` behind an interface; wrap the current
   logic as `LegacyFormationPolicy`; add the target policy behind `PhaseQueueSchedulerConfig::formationPolicy`
   so both run in one binary for A/B. Cache the two `previewMechanismPlan` results per state epoch to remove the
   scheduler deep copies.
4. **Removal.** After the target policy wins full24 x3 with the output audit unchanged, delete the mechanisms listed
   as unmotivated above.

## Head-off full24 x3 (`headoff-full24-3x-20260929`)

Same binary as note 366 (`c7a1b71`), `TRT_EDGELLM_CONTEXTUAL_MIN_OBSERVATIONS=1000000`, 72/72 cells. Compared with the
note 366 campaign (head active):

- Geomean -1.82% (Gemma -1.39%, Cosmos -2.25%); -0.73% versus 9db3.
- 12 workloads lose with every head-off run below every head-on run: Cosmos balanced -6.5%, poisson -5.2%, mixed
  -3.1%, text-heavy -2.6%, decode-heavy -2.4%, vision-heavy -2.0%, bimodal -1.0%; Gemma mixed -6.6%, vision-heavy
  -5.8%, balanced -2.4%, short -1.5%, poisson -1.8%.
- One workload wins with separated ranges: Gemma long-prefill +5.1% (644.9 tok/s, TPOT -9%).
- Two cells fall below frozen vLLM: Cosmos balanced (-2.1%) and Gemma vision-heavy (-2.4%).
- The losses carry TPOT +3..+9%; the head was declining P+D where prefill interference on decode costs more than the
  overlap saves. That is exactly the case "overlap is a hardware fact" ignored: a P+D makespan of max(P, D) is
  cheaper for the pair but stretches every decode row's token time, and decode-heavy mixes pay that on many rows.

Conclusion: the head's authority is justified outside Gemma long-prefill. Its failure mode there (note 365:
remaining-work reference on residual observations) is a bias to fix, not evidence for removing the head. The
correct objective must price decode-row slowdown, not just pair makespan.

## Decision-loop fix: epoch-gated residual previews (`41f287f`)

Stage 1 as first written (`9ba95bc`, reuse a no-action outcome in `next()` until the state epoch changes) never
fired: a same-binary A/B (`decision-elision-ab-20260929`, 6 blocks) showed 0 elided decisions and unchanged
selector counts. A temporary per-caller counter on Gemma long-prefill attributed the 56k selector calls:
`next()` found no action 0 times; 57k were `previewGlobalDecodeAction` calls from
`PhaseThreeCoordinator::dispatchGlobalPrefillDecodeResidual`, which runs on every host poll while a prefill action
is in flight, deep-copies the scheduler for mechanism previews, and then discards the result because its candidate
id equals the last one. The state epoch was also bumped about 4M times per run because `dispatchReady` and the
coordinator fast path call `setPendingPrefillProducerRows` and `setExternalDrainPreference` every poll.

`41f287f` skips the residual preview while the scheduler state epoch is unchanged, makes those setters bump the
epoch only on a real value change (producer wait estimates, which drift with wall time, do not count), and removes
the unused `next()` elision. Same-workload A/B against the parent `c7a1b71` (`epoch-preview-ab-20260929`, 6
interleaved blocks):

| Workload | Selector calls new / old | tok/s change | Outputs |
|---|---:|---:|---|
| Cosmos balanced | 599 / 4,809 | +1.2% (ranges overlap) | 288/288 identical to parent |
| Gemma long-prefill | 614 / 62,082 | +0.4% (overlap) | self-stability 57/64 vs 58/64 |
| Gemma vision-heavy | 222 / 2,716 | +0.2% (overlap), TTFT p95 -13% | self-stability 47/64 vs 47/64 |

P+D counts are unchanged (155/151, 98/92, 35/34). The host poll loop still spins (`yield`), so wall-clock poll time
is unchanged; the saving is work per poll. Replacing the yield loop with an event wait is the remaining part of
event-driven decisions.

## Separate complete-P+D contextual model (`1055f0e`, reverted)

Hypothesis (note 365): complete P+D and D-attached-to-running-P residual augmentation share the P->D contextual
model; complete observations are rare, so complete P+D is priced from residual rewards that are low for late
attaches. `1055f0e` routed complete P+D to its own model behind `TRT_EDGELLM_CONTEXTUAL_SEPARATE_COMPLETE_PD`.
Same-binary A/B (`separate-complete-pd-ab-20260929`, 6 blocks):

| Workload | Separate vs shared | P+D dispatches |
|---|---:|---:|
| Gemma long-prefill | -4.9%, all 6 runs lower (575-593 vs 614-627) | 24 vs 100 |
| Cosmos balanced | +1.6% (ranges overlap) | 147 vs 151 |
| Gemma mixed | +0.1% | 26 vs 32 |
| Gemma vision-heavy | -0.5% | 30 vs 30 |

Full telemetry (`separate-complete-pd-fulltel-20260929`) falsifies the hypothesis: the separate complete model becomes
ready from 88 warmup calibration observations of complete P+D and still refuses complete P+D (2 of 166 authority
candidates selected; 3 of 181 in the shared arm). The pessimism is learned from complete samples, not borrowed from
residual ones. Every offered residual augmentation is selected in both arms (20/20, 90/90; measured cost has
authority after `c7a1b71`); the loss comes from fewer residual opportunities, which is the low mode of note 365.

Remaining contradiction: measured complete P+D on long-prefill compresses about 2x (note 365), yet the head's
complete-P+D posterior is non-positive. Most of its observations come from the generic VLM warmup trace, so the
likely issue is distribution shift between warmup and serving shapes, not reward contamination. Reverted.

## Revised next steps

1. Validate Stage 1 elision (A/B on/off, same binary): decisions must fall, throughput and output must not move.
2. Fix the head's reference bias rather than its authority: residual observations should be rewarded against the
   full-action reference, or split into their own direction. Target: recover the head-off Gemma long-prefill gain
   (+5%) without the 12 losses.
3. Any future pair-cost model must include decode-row slowdown (TPOT) in the objective; makespan alone chose badly.
4. Per-phase completion (Stage 2) remains a plausible overlap-ceiling improvement but is now lower priority, since
   more overlap is not uniformly beneficial.

## Open questions

- E-active interference: the 337 probe-off losses were all VLM workloads. The two-coefficient model must be
  validated on Gemma multi-image and vision-heavy before overlap-by-default is allowed under active E.
- WAIT is still required for E formation (waiting for the next image to enlarge the E batch). Text P/D wait is
  removed; E wait stays with the formation policy.
- Whether `19d8aa8` (residual compression transfer) survives: unnecessary once the pair model prices every shape.
