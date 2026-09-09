# Same-engine v0.10.0 / v0.10.1: deeper policy and request attribution

This extends note268 using only its two existing mixed HTTP runs. No new GPU run,
policy change, SLO change or vLLM run was performed. Each version still has n=1.
Results and reproducible computations are in
`.local/results/v0101-forward-port/version-activity-20260909/detail.json` and
`benchmarks/phase_serving/analyze_version_activity_detail.py`.

## 1. Refine the calibration interpretation

Current's319-request warmup had lower exact-key coverage, but that does NOT mean
its Scalar RLS could not score measured P+D work:

| Measured serving observation | Old runtime | Current |
|---|---:|---:|
| Decision records with P+D in preview | 40 | 36 |
| Preview P+D marked scalar authority applied | 40 | 36 |
| Preview P+D marked scalar cost known | 40 | 36 |
| P+D selected by final coordinator | 9 | 3 |
| Selected fraction among those previews | 22.50% | 8.33% |
| P selected instead | 23 | 29 |
| New observed P+D executions | 9 | 3 |

These preview records are not an assertion that every candidate survived the
actual selector's frontier and SLO filtering. Actual selector audits are available
only in current. All36 recorded current opportunities have usable Scalar output;
there is no evidence here of a universal cold-key authority block.

The first measured metric shows168 versus106 cumulative P+D calibration
observations, ending177 versus109. This counter is not the same quantity as35/25
warmup safe probes. Avoid conflating evidence observations, exact keys, executions,
and probes. Current has less evidence, but data alone does not establish that more
warmup would restore the missing six selections or improve performance.

The measured contextual decision disagreement counter increases12 versus4. This
compares contextual LCB sign with the actual local decision when contextual control
applies; it is not prediction error or a count of unsafe requests.

## 2. An exact timeline accounting identity

Use the union of recorded intervals within each phase; sum E+P+D union durations,
not all nested kernel/stream timings. There is no recorded triple overlap or Copy
interval here, so:

`recorded span = E union + P union + D union - pair-overlap time + idle time`.

| Term, ms | Old | Current | Current minus old |
|---|---:|---:|---:|
| E/P/D union durations summed | 3012.678 | 2914.136 | -98.541 |
| Subtracted overlap duration | 418.533 | 285.747 | -132.786 |
| Added unrecorded-work interval | 90.866 | 99.220 | +8.353 |
| First-to-last recorded span | 2685.011 | 2727.609 | +42.598 |

Thus `-98.541 +132.786 +8.353 = +42.598ms`.
The recorded phases collectively take less time but overlap less, leaving a longer
span. This is an exact accounting decomposition, NOT a counterfactual prediction:
forcing the old overlap pattern could change contention, phase costs and cohorts.
The lost132.786ms cannot be claimed as independently recoverable speedup.

## 3. Neither P packing nor D fragmentation explains this pair of runs

| Phase-level metric | Old | Current |
|---|---:|---:|
| P dispatches | 37 | 37 |
| P row-chunk instances | 68 | 68 |
| Mean P batch | 1.838 | 1.838 |
| Mean P engine GPU time, ms | 34.518 | 32.995 |
| D dispatches | 86 | 84 |
| Total D row steps | 2864 | 2864 |
| Mean D batch | 33.302 | 34.095 |
| Mean D engine GPU time, ms | 8.981 | 8.409 |

There are64 initial output tokens from prefill and2864 subsequent decode row
steps, giving2928 total output tokens. Current has slightly larger D batches and
fewer D dispatches. It does not suffer aggregate D micro-batch fragmentation in
this diagnostic. P packing counts are equal, although request membership/order
and contention differ. Engine-only times differ from phase unions that also
include sampling; do not interchange those timing definitions.

## 4. The longest D pause is occupied by useful E/P work

Current D dispatch816 completes at302.459ms relative to first measured work.
The next D dispatch825 starts at727.192ms: a424.732ms gap.

| Recorded state inside this gap | ms |
|---|---:|
| P only | 239.001 |
| E+P | 165.198 |
| E only | 11.148 |
| E+D sampling-tail intersection | 0.098 |
| D sampling tail | 0.103 |
| No recorded phase work | 9.185 |

Only2.16% is unrecorded-work time. This does not support treating the entire425ms
as a host launch/polling failure. It also does not establish that all9ms is host
overhead: dependency waits, uninstrumented work and scheduling delays can coexist.

The execution sequence is:

```text
D32 (dispatch816)
  -> P1 request12  96.15ms
  -> P1 request4  121.87ms
  -> P1 request16  36.52ms
  -> P1 request32  26.45ms
  -> P1 request24  26.60ms
  -> P1 request43  26.04ms
  -> P2 requests0,20 33.97ms
  -> P2 requests8,28 35.70ms
  -> D40 (dispatch825)
```

Some P work overlaps E, so E durations cannot be added again to this sequence.
For six later P decisions, actual snapshots show34→38 D rows ready, both P and D
deadlines expired, and the P hard guard suppressing the standalone D candidate.
The existing all-late recovery then favors P over P+D by estimated compression.

Examples from **actual local P/D selector inputs**, not preview lists:

| P request(s) | Ready D | P compression | P+D compression | Reason |
|---|---:|---:|---:|---|
| 16 | 34 | 1.088 | 0.660 | all-late efficiency |
| 32 | 34 | 1.028 | 0.851 | all-late efficiency |
| 24 | 35 | 1.046 | 0.881 | all-late efficiency |
| 43 | 36 | 1.278 | 0.631 | all-late efficiency |
| 0,20 | 37 | 1.207 | 0.609 | all-late efficiency |
| 8,28 | 38 | 1.225 | 0.627 | all-late efficiency |

This supports a concrete mechanism: resident D work is available, but standalone
D is absent from the compared choices; estimated unprofitable P+D does not restore
D service. The older source also contains the same P-expiry exclusion and all-late
rule. This is not evidence of a newly introduced SLO setting or new ranking rule.
Different trajectories can expose the same rule differently.

Across current's36 preview-P+D decisions, the final audit reason is all-late
efficiency30 times, minimum-violation5, deadline-safe efficiency1. Those counts are
not available symmetrically for old, so no invented old reason distribution is
reported. Note264 already tested retaining expired D candidates: its mixed result
regressed and vision TPOT regressed. Therefore simply enabling that switch is not
a validated solution to the mechanism observed here.

Old also has a large283.126ms D pause, but later (2134.403–2417.528ms). It contains
150.485ms E-only,121.620ms P-only and10.953ms unrecorded work. The important change
is service placement and affected requests, not the existence of pauses only in
the new version.

First measured D starts at246.622ms old versus29.330ms current. Current starts D
earlier but subsequently interrupts it longer. Mean gap statistics span different
resident-decode windows and cannot alone establish the number of requests harmed.

## 5. Paired requests reveal who pays the cost

Match requests by ID; positive delta is worse for current.

| Class | Metric | Mean delta, ms | Current improved requests |
|---|---|---:|---:|
| text | TTFT | +13.969 | 1/32 |
| text | TPOT | +1.045 | 0/32 |
| text | E2E | +65.088 | 0/32 |
| vision | TTFT | -32.856 | 18/32 |
| vision | TPOT | +2.993 | 9/32 |
| vision | E2E | +59.941 | 4/32 |

Aggregate TTFT improves because vision's improvement exceeds text's loss. It is
incorrect to say current improves first-token latency for both classes. Earlier
vision first tokens also do not imply earlier request completion: most vision
requests lose E2E performance in this run. This supports examining TTFT/TPOT jointly
rather than promoting an action from aggregate TTFT alone.

## 6. Resolve the Copy caveat more precisely

Measured deltas in direct-output telemetry are10 old and11 current batches,
matching all measured encoder executions. Both bind305856512 bytes of output
storage directly during serving. Consequently the dedicated encoder-output-copy
fallback is bypassed for every measured E batch. No extra output-copy traffic
explains this version difference. Other H2D/D2H transfers are outside that counter
and not ruled out by C=0. The byte count is directly-bound output volume, not a
measurement of all bytes saved on the memory bus.

## 7. What to test next, and what not to claim

1. Repeat old/current in alternating order under the same fixed319 warmup contract;
   check whether the early long D pause and9→3 P+D selection change recur.
2. Separately compare fixed-count warmup against a validated coverage condition.
   RLS authority is already available here; investigate predicted cost/uncertainty
   and candidate ordering, not just a binary learned/unlearned label.
3. Use the observed six P decisions as candidate diagnostic snapshots. Compare
   P, standalone D and P+D under a controlled equal-work experiment before changing
   the all-late objective. A natural log cannot reveal the unexecuted branch's cost.
4. Preserve E/P progress while measuring request-level D service interruptions.
   More overlap or shorter host gaps alone is not the objective.

Current source policy and SLO values remain unchanged. This analysis narrows the
likely mechanism but does not establish an engine-independent policy regression,
a statistically significant version effect or a proven replacement policy.

## Validation

The analyzer aligns absolute CUDA-epoch interval timestamps with relative mask
segment timestamps before intersections, and checks that each gap's mask durations
sum back to its duration. It also requires matching request ID sets. The alignment
check passed on both runs. The same data feeds the existing note268 figures;
no changes to their aggregate masks or serving metrics were necessary.
