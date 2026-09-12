# Gemma 4 expanded batch frontier, equal-24 comparison, and heuristic-removal plan

## Outcome

The RTX 3080 headroom requirement was relaxed from the former 512 MiB promotion gate to a 192 MiB exploratory
floor. This exposed the actual Gemma 4 E2B AWQ capacity frontier rather than stopping at the first conservative
configuration.

The largest buildable profile was not the best runnable operating point:

- P8/D16 and P4/D64 engines build, but the complete VLM process runs out of memory while loading its visual path.
- P4/D48/E8 runs with only 413 MiB of measured headroom, but is slower than D24 and D32 on the vision-heavy
  sentinel.
- P4/D24/E8 and P4/D32/E8 have effectively identical throughput. D24 has marginally better latency and lower
  memory use.
- The retained capacity point is therefore compiled P4/D32/E8 with an active D24 cap, 24 stable owners, 96 KV
  pages, and a tiered E/P arena.

This point completed all twelve real HTTP request traces. With both clients limited to 24 outstanding requests, it
beats the fresh optimized vLLM comparison in token throughput and arrival-inclusive E2E mean/p95 on 12/12 traces.
It does not win every latency dimension: wave-drain TTFT and six TPOT-p95 cases remain weaker. All measurements are
one-repeat diagnostics, not citable confidence intervals or a universal default.

## Fixed experimental contract

- source branch: `codex/v0101-phase-forward-port`
- measured source commit: `138dab1` and its committed predecessors; the worktree was clean during the runs
- model: `Chunity/gemma-4-E2B-it-AWQ-4bit`
- quantization: INT4-AWQ backbone; FP16 PLE, embedding, LM head, and KV
- GPU: RTX 3080, 10,240 MiB reported memory
- runtime: independent E/P/D TensorRT contexts, indexed-paged stable ownership, 128-token KV pages
- policy: V3 service-scaled transition controller with workload-independent generic calibration
- text limits: input 1,024 tokens, KV capacity 2,048 tokens
- vision contract: 280 model-declared soft tokens per image
- external comparison contract: identical traces, output lengths, `ignore_eos`, 24 client workers, and 24 maximum
  outstanding HTTP requests
- Current internal capacity: P4/D24/E8, 24 owners, 96 KV pages, tiered shared E/P arena
- vLLM internal capacity: vLLM 0.28, max 8 sequences, max model length 2,048, max batched tokens 1,024, 160 MiB
  KV cache, chunked prefill, async scheduling, CUDA graph sizes 1/2/4/8, and up to eight images per prompt

Current and vLLM use different internal capacities because the goal is an end-to-end comparison of the best
runnable contract found for each stack under the same 10 GiB GPU and input arrival stream. It is not a causal
comparison of one scheduler mechanism.

## Expanded build and memory frontier

| Compiled profile | P activation | D activation | Full VLM runtime | Interpretation |
|---|---:|---:|---|---|
| P4/D32/E8 | 734,579,712 B | 138,563,584 B | pass | retained engine capability |
| P4/D48/E8 | 734,579,712 B | 207,843,328 B | pass | 413 MiB minimum headroom; no speed gain |
| P4/D64/E2/E8 | 734,579,712 B | 277,123,584 B | OOM | engine build succeeds; visual runtime allocation fails |
| P8/D16/E2 | 1,469,156,352 B | 69,283,328 B | OOM | prefill workspace is the limiting allocation |
| E8 | n/a | n/a | pass with tiered E/P | 717,373,440 B visual activation and 521,256,960 B scratch |

The failure is not primarily a KV-cache-layout failure. The KV pool is finite and stable, but Gemma's large PLE
residency plus independent TensorRT activation/workspace allocations dominate the last GiB. Increasing D is much
cheaper than increasing P, but D64 still pushes the complete VLM process past 10 GiB. E8 only fits because P and E
borrow one tiered arena and are therefore mutually exclusive. That memory optimization deliberately gives up E+P
overlap while retaining E+D and P+D.

The exploratory 192 MiB floor did useful work: it established that D48 is runnable. It did not justify selecting
D48, because the service curve had already saturated at D24.

## Vision-heavy active-D frontier

All rows use P4/E8, the same 64-request trace, 24 owners where supported, generic calibration, and the tiered E/P
arena.

| Active D | token/s | TTFT mean/p95 ms | TPOT mean/p95 ms | E2E mean/p95 ms | Peak MiB | Headroom MiB |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 409.22 | 2,548/5,459 | 20.25/25.34 | 3,296/5,917 | 9,727 | 513 |
| 24 | **455.70** | **2,087/4,802** | 20.42/28.12 | 2,895/5,327 | 9,737 | 503 |
| 32 | 455.40 | 2,091/4,823 | **20.33/27.63** | **2,893/5,331** | 9,747 | 493 |
| 48 | 452.42 | 2,108/4,856 | 20.40/27.73 | 2,913/5,367 | 9,827 | 413 |

D24 raises throughput 11.36% over D16 and materially reduces TTFT/E2E. D32 changes throughput by -0.07%, and D48
changes it by -0.72% relative to D24. The useful knee is therefore D24, not the maximum legal batch. The D24/D32
difference is below one-run noise, so D24 is selected on lower memory and slightly better TTFT rather than claimed
as a statistically proven winner.

## Why one static configuration does not dominate

| Operating point | Context memory | Stable owners | Main strength | Main cost |
|---|---|---:|---|---|
| P2/D4/E2 | fully independent E/P/D | 4 | low resident decode TPOT | low throughput and substantial admission waiting |
| P4/D16/E4 | fully independent E/P/D | 16 | balanced latency/capacity; E+P remains legal | less aggregate capacity than D24/E8 |
| P4/D24/E8 | tiered E/P, independent D | 24 | best tested aggregate throughput and queue drain | higher TPOT tails; no E+P overlap |

On the same client-64 full12 contract, moving from P4/D16/E4 to P4/D24/E8 increases geometric-mean token
throughput by 25.55%, but raises TTFT mean by 40.87%, TTFT p95 by 35.32%, E2E mean by 14.24%, and E2E p95 by
29.50%. TPOT mean and p95 improve by 17.62% and 18.30%. These figures describe different internal concurrency
frontiers; they are evidence that maximum active ownership must become an online decision, not that one point is a
universal replacement.

The earlier P2/D4/E2 full12 used only 16 client-side outstanding requests, while the first expanded campaigns used
64. Service TTFT starts when the HTTP request is sent, so it excludes time waiting behind the client semaphore.
Cross-capacity claims based only on service latency were therefore confounded. The comparison below uses scheduled
trace arrival as time zero and fixes the external maximum at 24 for both systems.

## Equal-24 twelve-workload comparison

`Arrival TTFT/E2E` includes client admission waiting. TPOT is measured between generated tokens after first-token
service begins. Lower latency is better.

| Workload | System | token/s | Arrival TTFT mean/p95 ms | TPOT mean/p95 ms | Arrival E2E mean/p95 ms | Peak MiB |
|---|---|---:|---:|---:|---:|---:|
| short | Current | 660.70 | 503/1,138 | 27.47/37.50 | 1,052/1,492 | 9,727 |
|  | vLLM | 288.38 | 1,398/2,813 | 23.63/25.27 | 1,885/3,354 | 8,143 |
| balanced | Current | 1,107.78 | 1,468/3,517 | 17.73/20.24 | 2,945/4,704 | 9,727 |
|  | vLLM | 329.72 | 6,766/13,623 | 22.39/22.86 | 8,648/15,652 | 8,143 |
| decode-heavy | Current | 1,274.64 | 3,455/7,941 | 15.49/16.29 | 7,376/12,062 | 9,727 |
|  | vLLM | 330.17 | 19,750/40,797 | 22.24/22.42 | 25,401/46,086 | 8,143 |
| long-prefill | Current | 401.92 | 6,162/11,610 | 23.07/25.99 | 8,121/13,099 | 9,727 |
|  | vLLM | 261.31 | 8,795/17,667 | 27.40/29.02 | 11,089/19,632 | 8,229 |
| bimodal | Current | 575.58 | 4,883/11,861 | 21.25/39.05 | 7,556/16,278 | 9,727 |
|  | vLLM | 298.08 | 11,901/25,446 | 24.12/25.93 | 15,484/30,159 | 8,229 |
| text-heavy | Current | 835.60 | 1,440/3,007 | 22.72/28.99 | 2,621/3,947 | 9,735 |
|  | vLLM | 309.64 | 4,618/9,367 | 23.30/25.80 | 5,828/10,199 | 8,235 |
| mixed | Current | 645.58 | 1,606/3,942 | 23.62/29.53 | 2,691/4,468 | 9,735 |
|  | vLLM | 270.00 | 6,003/10,001 | 21.68/27.27 | 6,936/10,738 | 8,235 |
| vision-heavy | Current | 445.65 | 2,132/4,926 | 22.03/28.98 | 2,996/5,459 | 9,735 |
|  | vLLM | 285.40 | 4,088/7,728 | 25.25/27.18 | 5,021/8,482 | 8,235 |
| poisson | Current | 895.77 | 1,363/2,825 | 20.77/26.60 | 2,818/4,388 | 9,731 |
|  | vLLM | 310.41 | 5,449/10,815 | 22.89/23.49 | 7,099/13,211 | 8,237 |
| wave-drain | Current | 97.58 | 213/275 | 10.08/11.90 | 525/542 | 9,735 |
|  | vLLM | 92.43 | **161/212** | 23.00/24.41 | 874/894 | 8,237 |
| multi-image | Current | 292.05 | 589/1,154 | 16.67/21.34 | 1,106/1,474 | 9,737 |
|  | vLLM | 231.73 | 605/1,270 | 25.86/27.45 | 1,406/1,993 | 8,237 |
| late-vision | Current | 1,430.92 | 726/2,698 | 14.36/14.55 | 2,783/2,831 | 9,733 |
|  | vLLM | 353.00 | 6,399/12,556 | 22.03/22.06 | 9,555/12,838 | 8,237 |

Across the twelve traces, geometric means give:

| Metric | Current relative to vLLM | Current wins |
|---|---:|---:|
| token throughput | 2.21x, +121.36% | 12/12 |
| arrival TTFT mean | 63.17% lower | 11/12 |
| arrival TTFT p95 | 57.66% lower | 11/12 |
| TPOT mean | 19.54% lower | 10/12 |
| TPOT p95 | 6.05% lower | 6/12 |
| arrival E2E mean | 53.14% lower | 12/12 |
| arrival E2E p95 | 56.75% lower | 12/12 |

The remaining losses are diagnostic:

- `wave-drain` TTFT mean/p95 is 213/275 ms versus vLLM's 161/212 ms. E formation and launch latency remain the
  critical path when there is no admission backlog to amortize them.
- TPOT mean loses on `short` and `mixed`.
- TPOT p95 loses on `short`, `bimodal`, `text-heavy`, `mixed`, `vision-heavy`, and `poisson`. More active owners
  improve arrival completion but lengthen or jitter per-request decode service.
- Current consumes about 1.5--1.6 GiB more peak memory than vLLM. This campaign intentionally spends that memory on
  independent/tiered compiled contexts and larger ownership capacity.

## Correctness and evidence limits

- Every retained Current and vLLM run returned the requested number of output tokens without OOM or page/slot
  errors.
- The P8/D16 and P4/D64 engine artifacts are build-success/runtime-failure characterization points, not runnable
  candidates.
- Token hashes differ across TensorRT engine profiles. vLLM request logs do not expose the same token-ID contract,
  so exact cross-framework greedy identity is not established.
- The current comparison has one repeat. The differences are large enough to choose the next diagnostic frontier,
  but promotion requires at least three fresh repeats and confidence intervals.
- Current and vLLM apply their native tokenizer/chat/runtime paths. Generated lengths and request content are held,
  but prompt-token counts are not guaranteed bit-identical.

## Heuristic-removal plan across Cosmos and Gemma

The Cosmos reference ran P8/D64/E4 with 80 owners. Gemma's useful point is P4/D24/E8 with 24 owners and shared E/P
workspace. Copying absolute batch sizes, milliseconds, or a decode-cost table between them is demonstrably wrong.
The same controller must instead derive decisions from engine capability, measured service, ready state, and
ownership state.

### Keep as correctness or compiled capability

| Value | Why it remains |
|---|---|
| Gemma 280 image tokens | model semantic contract required for correct M-RoPE and embedding placement |
| P/D/E engine maxima | legal TensorRT profile shapes, not selected operating batches |
| max input/KV length and 128-token pages | compiled capacity/layout contract |
| one in-flight execution per TensorRT context | context correctness invariant |
| tiered E/P exclusion | memory-ownership invariant for the shared arena |
| stable slot/page validity and request DAG dependencies | correctness and lifetime invariants |

### Remove or demote as policy heuristics

| Current static value | Replacement |
|---|---|
| Cosmos-derived decode cost table, 6.2--9.8 ms | startup isolated D measurements on a capability-scaled logarithmic grid, monotone interpolation, then online CUDA updates |
| fixed active owners and active D cap | endogenous capacity frontier using marginal completion gain, decode-cycle growth, ready mass, and measured memory headroom |
| E formation wait = 25 ms | WAIT only for an already-outstanding event; compare observed batch gain against wait cost and downstream D continuity |
| E initial cost/margin = 50/5 ms | measured E1/E2 points plus uncertainty from generic calibration |
| E text guard = 250 ms and max defer = 500 ms | service-normalized request age and phase progress deficit; no model-specific absolute time |
| E decode-pressure limit = 0.9 | measured marginal decode-cycle stretch and incumbent readiness |
| prefix-before-vision minimum = 128 tokens | measured E-first versus prefix-P-first crossover; retain only engine legality as a hard bound |
| formation attribution horizon = 4 dispatches | terminate attribution at request-ready boundaries or trajectory reconvergence |
| hard prefill TTFT guard without an SLO | relative service deficit and starvation protection; use an explicit SLO only when the caller supplies one |
| encoder arbiter policy | mechanism-only E executor; the global scheduler chooses E-now, prepared-E, E+D, or event-backed WAIT |
| fixed generic calibration shape/count | capability-scaled coverage that stops when confidence/authority gates are met |

`maxOverlapPrefillTokens=128` is intentionally deferred. The current Gemma engine uses 128-token dense chunks and
the prior Cosmos winner also used 128, so changing this while expanding concurrency would confound two dimensions.
It remains an acknowledged performance heuristic/compiled-granularity coupling, not a solved item.

### No-explicit-SLO active-capacity controller

The next controller should distinguish legal capacity from active capacity:

```text
Engine capability: P4 / D32 / E8 / 24 owners
                         |
                         v
Snapshot: ready E/P/D, request ages, measured service, memory
                         |
                         v
Candidate active frontiers: D8, D16, D24; owners 8, 16, 24
                         |
                         v
Reject memory-unsafe or context-illegal candidates
                         |
                         v
Choose the knee before marginal batch gain is smaller than
decode-cycle growth + admission/formation cost
```

With no caller SLO, the objective must not silently invent fixed millisecond targets. It should use arrival-inclusive
age, relative service progress, queue growth, and online knee detection. If the caller supplies TTFT/TPOT targets,
they become additional hard protection rather than replacing the profile-free controller.

### Implementation and validation sequence

1. **H0 -- Audit and telemetry.** Log every policy threshold with provenance and classify it as correctness,
   engine capability, memory mechanism, or removable policy. Add arrival-inclusive TTFT/E2E as first-class output.
2. **H1 -- Decode knowledge.** Replace the inherited Cosmos decode table in shadow with generic isolated sampling
   and monotone interpolation. Compare prediction error, chosen D cap, and scheduler CPU cost on both models.
3. **H2 -- Endogenous active frontier.** Evaluate D8/D16/D24 and owners 8/16/24 from one P4/D32/E8 capability
   engine. Activate only after shadow choices reproduce the measured D24 knee without seeing workload names.
4. **H3 -- Encoder timing.** Replace fixed E cost and 25 ms wait with completion-event-backed WAIT and measured
   uncertainty. Use wave-drain and scaled multi-image as forced causal tests.
5. **H4 -- Global authority.** Reduce the encoder arbiter to an executor and move E timing into the global action
   frontier. Preserve context, DAG, and arena invariants.
6. **H5 -- Cross-model gate.** Run identical controller code on the retained Cosmos and Gemma engines without
   changing policy constants. Model-specific semantic/profile metadata may differ; policy values may not.
7. **H6 -- Full validation.** Run all twelve traces at low, knee, and overload, output lengths 64/128/256, zero and
   generic warmup, and at least three repeats. Report throughput, arrival/service TTFT, TPOT, arrival/service E2E,
   activity masks, memory, selected active frontier, and confidence intervals.

Promotion requires no correctness regression, no unexplained static model/workload rule, and a single controller
that selects different legal operating points from observed state. It does not require selecting the maximum batch
or winning every instantaneous TPOT sample.

## Retained artifacts

| Artifact | Path |
|---|---|
| build logs and replay scripts | `.local/results/gemma4-e2b-awq-batch-frontier-20260911` |
| equal-24 machine-readable comparison | `.local/results/gemma4-e2b-awq-batch-frontier-20260911/equal24-full12-comparison.json` |
| equal-24 readable table | `.local/results/gemma4-e2b-awq-batch-frontier-20260911/equal24-full12-comparison.md` |
| Current equal-24 raw results | `.local/results/gemma4-e2b-awq-batch-frontier-20260911/p4-d24-e8-tiered-full12-equal24` |
| vLLM equal-24 raw results | `.local/results/gemma4-e2b-awq-batch-frontier-20260911/vllm-full12-equal24` and `vllm-full12-equal24-image8` |
| D48 engine | `.local/artifacts/v0101-forward-port/gemma-4-e2b-it-awq/engine-asym-p4-d48-kv2048-p96-soft280` |
| D64 build-only engine | `.local/artifacts/v0101-forward-port/gemma-4-e2b-it-awq/engine-asym-p4-d64-kv2048-p128-soft280` |
| P8 build-only engine | `.local/artifacts/v0101-forward-port/gemma-4-e2b-it-awq/engine-asym-p8-d16-kv2048-p64-soft280` |
| E8 engine | `.local/artifacts/v0101-forward-port/gemma-4-e2b-it-awq/visual-e8-soft280` |

