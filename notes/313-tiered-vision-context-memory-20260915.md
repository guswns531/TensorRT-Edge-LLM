# Tiered vision context memory

## Outcome

This experiment tests whether Current can preserve independent E/P/D TensorRT execution while reducing the
permanently resident E/P activation workspace. The implementation is correct enough to build and execute Gemma 4
requests with two vision optimization profiles, but it is not promoted as the default memory architecture.

The E3/E4 arena reduces fresh ready memory from 9,445 MiB to 9,375 MiB, a 70 MiB reduction. In the initial
single-run screen it regresses mixed tail latency and multi-image throughput, so it does not pass the performance
gate. A later contract audit also found that this screen disabled lifetime vision admission while the retained
high-performance campaign enabled it. Performance must therefore be repeated under the same lifetime-admission
contract before any causal claim about the arena is made.

## Mechanism

TensorRT exposes a separate activation-memory requirement for each optimization profile. The new opt-in Gemma 4
vision engine contains a smaller profile followed by the existing full-capacity profile. Runtime preprocessing
selects the smallest profile that contains the actual raw patch count.

```text
                        one E/P arena

small E profile         P workspace                 large E profile
┌───────────────┐       ┌────────────────────┐      ┌─────────────────────────┐
│ disjoint tail │       │ arena base         │      │ aliases the arena base  │
│ may overlap P │       │ may overlap E-small│      │ must be exclusive of P  │
└───────────────┘       └────────────────────┘      └─────────────────────────┘

small E action: [ P workspace ][ small-E workspace ]
large E action: [          large-E workspace         ]
```

The arena size is:

```text
max(aligned(P bytes) + small-E bytes, large-E bytes)
```

This differs from fully shared E/P memory, which serializes every E action with P. Small E retains the E+P
frontier; only requests requiring the large profile are exclusive with P. Decode keeps its independent context and
workspace in both modes.

## Builder and runtime contract

The builder adds `small_profile_max_image_tokens`. Zero retains the legacy single-profile engine. A nonzero value
is currently accepted only for Gemma 4 vision engines and produces:

1. profile 0 with the smaller aggregate image-token capacity;
2. profile 1 with the existing full capacity.

Gemma's exported `maxImageTokens` is a soft-token count while the TensorRT visual input is raw patches. The
pooling-kernel area converts 280 soft tokens to 2,520 raw patches. This exposed an important interface bug during
the experiment: scheduling policy cost keys and physical profile capacity cannot share one ambiguous token field.

The final implementation keeps two values per pending vision request:

- `inputTokens`: the existing policy/cost-model input, unchanged from the production path;
- `profileInputTokens`: raw TensorRT visual-input patches, used only for profile selection and E/P workspace
  exclusivity.

This preserves the existing workload-independent scheduler trajectory while making the memory alias decision in
the physical engine's units.

Profile switching happens after preprocessing has materialized the concrete Gemma patch tensor and before enqueue:

```text
request images
    -> resize geometry
    -> raw patch count
    -> smallest containing TRT profile
    -> setOptimizationProfileAsync(E stream)
    -> bind that profile's assigned context memory
    -> enqueue E
```

For an exclusive large E action, ownership is acquired when the prepared E batch is selected for GPU dispatch,
not when asynchronous CPU/GPU preprocessing begins. This prevents a prepared batch from blocking P before the
global scheduler actually chooses E. If P is still outstanding, E waits for that completion boundary while future
P dispatch is blocked; the aliased memory is never concurrently used.

## Measured workspace sizes

The same Gemma 4 visual ONNX and TensorRT build were used for both engines.

| Profile | Maximum soft tokens | Maximum raw patches | TensorRT activation bytes | MiB |
|---|---:|---:|---:|---:|
| E1 | 280 | 2,520 | 67,092,480 | 63.98 |
| E3 | 840 | 7,560 | 226,679,296 | 216.18 |
| E4 | 1,120 | 10,080 | 313,528,832 | 299.00 |
| P | n/a | n/a | 183,647,744 | 175.14 |

The E1/E4 arena is 313,528,832 bytes because the large E allocation covers `P + E1`. It can recover the full
separate P workspace, approximately 175 MiB, but only E1 can overlap P. The E3/E4 arena is 410,327,040 bytes and
recovers approximately 83 MiB at the raw workspace level while allowing E1--E3 to overlap P.

The observed fresh ready-memory reduction for E3/E4 is 70 MiB. The difference from the raw 83 MiB includes
allocator granularity and profile/context metadata.

## Initial screen

The following values are one-run diagnostic results. Positive deltas mean tiered is better. Both sides use the
same newly built binary, P8/D24/E4 limits, P128, graphs, engine, trace, and generic calibration. They accidentally
share the same **non-lifetime** admission contract, so this table compares the tiered mechanism with its immediate
same-contract baseline but not with the retained champion.

| Workload | Ready MiB base/tier | req/s | TTFT mean/p95 | TPOT mean/p95 | E2E mean/p95 |
|---|---:|---:|---:|---:|---:|
| mixed | 9,445 / 9,375 | -1.15% | -2.35% / -5.67% | +1.41% / -3.40% | -0.06% / -4.06% |
| vision-heavy | 9,445 / 9,375 | +14.74% | +20.72% / +2.75% | -6.04% / -3.24% | +14.70% / +2.89% |
| multi-image | 9,445 / 9,375 | -16.56% | -41.84% / -33.57% | -20.76% / -35.87% | -36.38% / -28.63% |

All requests completed and generated the requested token count. The tiered logs report the physical exclusivity
threshold as 7,560 raw patches. Mixed, vision-heavy, and multi-image formed 3, 6, and 2 exclusive large-E batches,
respectively. No action-fidelity violation occurred.

This screen rejects immediate promotion: mixed p95 latency exceeds the 3% regression gate and multi-image is much
worse. Vision-heavy's gain shows that the arena is not intrinsically slower, but one run is insufficient to
separate changed action trajectories from variance.

## Reproducibility audit

The retained high-performance campaign in note 310 and the initial tiered screen did not use the same admission
contract:

| Field | Retained champion | Frozen-binary recheck | Tiered screen |
|---|---:|---:|---:|
| lifetime vision admission | enabled | disabled | disabled |
| initial encoded capacity | 4 | 4 | 4 |
| effective encoded capacity | 12 | 4 | 4 |
| vision admission budget | 446,603,264 bytes | 0 | 0 |

This explains the apparent failure to reproduce the old binary more directly than GPU clock drift. On mixed, the
retained run used 26 E batches with maximum E4, while the frozen recheck used 29 batches and reached only E3. The
fresh recheck delayed vision first-token progress and cannot be used as evidence that later source changes caused a
regression.

The canonical diagnostic script now defaults to lifetime admission with effective capacity 12 and writes
`contract.env` into every result root. An explicit `ENCODED_ADMISSION_MODE=off` remains available for mechanism
ablation. This closes the missing-contract failure for future campaigns.

## Failed intermediate experiments

The following result roots are diagnostic only and must not be cited as performance results:

- E1/E4 runs that compared soft-token policy estimates with a raw-patch profile limit;
- runs that replaced Gemma's existing policy token estimate with raw patches or soft tokens, changing contextual
  cost keys and action formation;
- E3/E4 runs that configured a 7,560-patch profile but compared exclusivity against a 840-soft-token estimate;
- the frozen-binary recheck that omitted lifetime admission.

These failures motivated the dual policy/physical input contract above.

## Validation status and next gates

Completed:

- E1/E4 and E3/E4 engines build successfully;
- the runtime switches profiles and binds profile-specific memory;
- E3/E4 executes mixed, vision-heavy, and multi-image HTTP traces without OOM, invalid profile, or action-fidelity
  errors;
- the default builder and runtime behavior remains single-profile and fully independent unless explicitly enabled;
- builder configuration has default, round-trip, and legacy-JSON unit coverage.

Pending because the NVIDIA device is currently unavailable to the host:

1. repeat same-binary independent and E3/E4 tiered cells with lifetime admission enabled, three times each;
2. require exact request/token-count completion and no profile, ownership, or action-fidelity error;
3. require every primary median and p95 regression to remain within 3%;
4. if E3/E4 fails, test E1/E4 only as a latency/memory frontier rather than reducing the safety threshold;
5. retain the capability as experimental unless a repeated tier passes all three VLM workloads;
6. separately instrument the approximately 572 MiB common base-text excess identified in note 312.

The first five gates are encoded in
`benchmarks/phase_serving/run_gemma_tiered_vision_comparison.sh`. It builds the E3/E4 engine if necessary, runs the
independent and tiered variants with identical graph and lifetime-admission contracts, and produces a structured
comparison JSON/CSV.

Artifacts are under `.local/results/memory-attribution-20260915/`,
`.local/results/gemma4-tiered-vision-engine-20260915/`, and
`.local/results/gemma4-tiered-vision-e3-engine-20260915/`.
