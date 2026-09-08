# v0.10.1 P128 Vision-Prefill Recovery

## 1. Outcome

The remaining v0.10.1 memory and VLM-path regression was not a KV-cache
regression. It was dominated by the dedicated packed vision-prefill TensorRT
optimization profile. That profile allowed 1,024 packed tokens and raised the
prefill context-memory frontier to about 302 MiB. Rebuilding the same v0.10.1
ONNX with only the fixed P128 text-prefill and D64 decode profiles lets packed
vision prefill reuse the P128 profile in chunks and its existing prefill arena.

The recovered engine:

- lowers ready memory from 9,281 MiB to 9,061 MiB; under the initial
  capacity-four diagnostic, observed VLM peak memory falls from
  9,325--9,333 MiB to 9,097--9,113 MiB, while the restored-capacity final
  matrix peaks at 9,113--9,229 MiB depending on workload;
- improves V1 throughput by `+5.13%` geometric mean over the previous v0.10.1
  engine and wins 10/12 traces;
- improves the most affected V1 traces by `+17.79%` on mixed, `+10.93%` on
  vision-heavy, and `+16.14%` on multi-image;
- keeps the same 256-page stable indexed-paged KV pool and the same P8/D64/E4
  scheduler limits;
- makes V1 Scalar the clear policy winner on this engine: `+1.88%` geometric
  mean over V0 with 10/12 throughput wins. V2 is `-0.44%` versus V0 and is not
  promoted.

This recovers the intended independent E/P/D memory frontier. The first
three-run result below still used a restrictive encoded-vision capacity of four
and was therefore not a fair comparison with the old v0.10.0 champion. Restoring
the workload-independent v0.10.0 ownership frontier (`16` normal, up to `80`
when its existing age/TPOT guard allows growth) improves V1 by another `+12.65%`
geometric mean over that restricted run. The corrected three-run V1 result is
`-11.22%` geometric mean versus the old v0.10.0 table and `+1.68%` versus the
frozen vLLM table, with 7/12 vLLM wins. The remaining old-version gap is not a
KV-pool regression.

## 2. Engine contract

Both v0.10.1 engines use the same:

- `nvidia/Cosmos-Reason2-2B` FP16 ONNX;
- tied embedding/LM-head externalization;
- maximum logical batch 80, P8, D64, fixed chunk 128;
- 256 physical KV pages and 2,048-token per-slot capacity;
- independent TensorRT P and D contexts in one CUDA context;
- separate encoder TensorRT context and E/P/D/Copy streams;
- generic workload-independent policy calibration.

The only engine-profile difference is:

```text
previous engine
  profile 0: packed text prefill, at most P8 x 128 tokens
  profile 1: decode, at most D64
  profile 2: packed vision prefill, at most E/P4 x 1024 tokens

recovered engine
  profile 0: packed text and vision prefill, at most P8 x 128 tokens
  profile 1: decode, at most D64
```

The recovered plan is:

```text
path    .local/v0101-forward-artifacts/cosmos-reason2-2b/
        engine-p8-d64-kv256-p128-no-vp/llm.engine
size    3,070,408,348 bytes
SHA256  c420acaa60ef65a0c76569afc12104e764290c0c4c7fd9d6af46c3e9a6154634
```

The plan itself is only 11.3 MB smaller. The material memory reduction comes
from eliminating the large profile-specific prefill execution-context arena,
not from plan size or KV allocation.

## 3. Why P128 profile reuse is valid

Packed vision prefill already enters the common prefill queue as external
embeddings. Fixed chunking limits each dispatch to 128 useful tokens. It does
not require a separate 1,024-token TensorRT shape merely because one image
produces more than 128 language-side visual tokens; the request can traverse
multiple P128 chunks while keeping its stable KV page lease.

The execution relationship is:

```text
vision request
    |
    v
E context produces visual embeddings
    |
    v
P queue owns an external-embedding request
    |
    +-- P128 chunk 0 --+
    +-- P128 chunk 1 --+--> same stable KV page lease
    +-- ... -----------+
    |
    v
D queue
```

Text P and external-embedding P need distinct TensorRT execution contexts so
their dynamic binding state cannot interfere, but they deliberately share one
P128 profile and one context-memory arena. The coordinator already serializes
the two prefill consumers of that arena. P can still be independently enqueued
from D, and E retains its own context/workspace.

The runtime and the phase smoke server now create this external-prefill sibling
automatically when a packed VLM has no dedicated vision-prefill profile. The
old environment variable remains an explicit override but is no longer needed
for this normal configuration.

## 4. Independent execution mechanism check

Before changing the plan, the existing v0.10.1 engine was run once with its
normal independent E/P/D mechanism and once with forced E/D exclusion. This is
a same-engine, same-policy V0 comparison.

| Workload | Mode | req/s | tok/s | P dispatches | mean P BS | P GPU activity |
|---|---|---:|---:|---:|---:|---:|
| mixed | independent | 12.385 | 566.6 | 177 | 2.605 | 2,174.0 ms |
| mixed | E/D exclusion | 11.741 | 537.1 | 200 | 2.305 | 2,451.3 ms |
| vision-heavy | independent | 8.043 | 309.7 | 264 | 1.989 | 3,470.1 ms |
| vision-heavy | E/D exclusion | 7.882 | 303.5 | 276 | 1.902 | 3,649.9 ms |

Independent execution improves mixed by `+5.49%` and vision-heavy by `+2.05%`.
Measured simultaneous CUDA kernel activity is almost zero: mixed has no pair
overlap and vision-heavy has only 0.068 ms of E+D activity. The gain comes from
execution--formation coupling: removing an exclusion boundary changes queue
drain timing, forms denser P cohorts, and reduces total P dispatch work. This
is useful independence even when the optimal schedule rarely overlaps kernels.

## 5. Dedicated vision profile versus P128 reuse

The clearest same-policy V1 mixed comparison is:

| Metric | dedicated P1024 profile | shared P128 profile | change |
|---|---:|---:|---:|
| request throughput | 12.093 req/s | 13.537 req/s | +11.94% |
| token throughput | 553.3 tok/s | 619.3 tok/s | +11.94% |
| TTFT mean | 1,307.7 ms | 1,215.2 ms | -92.5 ms |
| TTFT p95 | 4,572.8 ms | 4,094.3 ms | -478.5 ms |
| TPOT mean | 16.65 ms | 15.90 ms | -0.75 ms |
| TPOT p95 | 20.01 ms | 21.43 ms | +1.42 ms |
| E2E mean | 2,060.3 ms | 1,955.4 ms | -104.9 ms |
| E2E p95 | 5,160.0 ms | 4,569.8 ms | -590.2 ms |
| peak memory | 9,325 MiB | 9,103 MiB | -222 MiB |

CUDA activity attributes the recovered wall time primarily to P:

| Mixed activity | dedicated P1024 | shared P128 |
|---|---:|---:|
| E activity | 832.9 ms | 836.4 ms |
| P activity | 2,174.0 ms | 1,635.1 ms |
| D activity | 1,819.3 ms | 1,928.4 ms |
| idle ratio | 6.55% | 7.19% |
| measured P+D overlap | 0 ms | 15.15 ms |

The shared-P128 run is faster even though its measured idle fraction is
slightly larger: it completes the fixed work sooner because P consumes about
539 ms less GPU time. Idle percentage alone is therefore not an optimization
objective.

## 6. Same-engine V0/V1/V2 gate

These are one-run engineering measurements. All rows use the recovered engine
and differ only in policy authority. `hash` is V0/V1/V2 token-trace identity
against V0.

| workload | V0 tok/s | V1 tok/s | V2 tok/s | V1 vs V0 | V2 vs V0 | best | hash |
|---|---:|---:|---:|---:|---:|---|---|
| short | 2,321.9 | 2,371.9 | 2,301.2 | +2.16% | -0.89% | V1 | Y/Y/Y |
| balanced | 4,172.8 | 4,319.7 | 4,043.4 | +3.52% | -3.10% | V1 | Y/Y/Y |
| decode-heavy | 4,965.2 | 4,965.7 | 4,730.0 | +0.01% | -4.74% | V1 | Y/Y/Y |
| long-prefill | 1,124.2 | 1,127.0 | 1,087.6 | +0.24% | -3.26% | V1 | Y/Y/Y |
| bimodal | 1,821.6 | 1,880.9 | 1,871.3 | +3.25% | +2.73% | V1 | Y/Y/Y |
| text-heavy | 1,312.9 | 1,318.1 | 1,320.8 | +0.39% | +0.60% | V2 | Y/Y/N |
| mixed | 626.7 | 651.7 | 599.4 | +3.99% | -4.35% | V1 | Y/N/Y |
| poisson | 1,539.0 | 1,509.7 | 1,639.2 | -1.90% | +6.51% | V2 | Y/Y/Y |
| vision-heavy | 342.9 | 356.2 | 346.3 | +3.87% | +0.99% | V1 | Y/Y/Y |
| wave-drain | 94.1 | 94.1 | 94.1 | -0.01% | -0.04% | V0 | Y/Y/Y |
| late-vision | 2,310.7 | 2,449.1 | 2,351.2 | +5.99% | +1.76% | V1 | Y/Y/Y |
| multi-image | 192.4 | 195.0 | 190.7 | +1.35% | -0.90% | V1 | Y/Y/Y |

Aggregate:

| Variant | geometric mean vs V0 | throughput wins | interpretation |
|---|---:|---:|---|
| V1 Scalar | +1.88% | 10/12 | promotion candidate |
| V2 Scalar+Transition | -0.44% | 5/12 | retain for research, do not promote |

V2's earlier v0.10.1 long-prefill advantage depended on the previous engine's
physical cost surface. This is expected for an online physical controller: an
engine tactic/profile change changes the action costs even when the scheduler
source is unchanged. The result also demonstrates why the policy gate must be
rerun after rebuilding an engine.

## 7. Restrictive-capacity diagnostic

Each workload was repeated three times after generic calibration, but this run
fixed both normal and throughput encoded-vision capacity at four. It is retained
as a controlled diagnostic of ownership/admission pressure, not as the final
promotion result. Throughput and tail columns are medians; mean-latency columns
are the median of run means. Values are milliseconds except token throughput
and memory.

| workload | tok/s | TTFT mean/p95 | TPOT mean/p95 | E2E mean/p95 | peak MiB | vs frozen vLLM |
|---|---:|---:|---:|---:|---:|---:|
| short | 2,358.4 | 103.6 / 181.5 | 13.15 / 23.29 | 352.6 / 434.4 | 9,061 | +18.90% |
| balanced | 4,203.0 | 71.8 / 173.3 | 13.14 / 14.69 | 1,192.9 / 1,846.8 | 9,061 | -2.71% |
| decode-heavy | 4,991.2 | 74.7 / 189.0 | 11.14 / 11.80 | 2,954.9 / 4,527.3 | 9,061 | +2.82% |
| long-prefill | 1,125.6 | 2,198.2 / 2,961.7 | 28.19 / 33.44 | 4,599.2 / 6,523.4 | 9,061 | +0.42% |
| bimodal | 1,857.5 | 1,940.4 / 4,154.7 | 18.40 / 27.04 | 4,483.7 / 9,438.3 | 9,061 | -0.58% |
| text-heavy | 1,323.5 | 433.8 / 2,019.9 | 18.45 / 24.91 | 1,415.2 / 2,432.9 | 9,105 | -19.04% |
| mixed | 636.9 | 1,176.9 / 3,978.7 | 15.39 / 20.52 | 1,891.4 / 4,435.0 | 9,103 | -30.89% |
| poisson | 1,544.1 | 360.6 / 1,913.4 | 20.17 / 26.29 | 1,838.5 / 2,668.5 | 9,101 | -14.22% |
| vision-heavy | 370.0 | 2,446.2 / 6,092.4 | 14.09 / 17.40 | 2,988.8 / 6,516.7 | 9,101 | -36.11% |
| wave-drain | 94.1 | 293.5 / 568.5 | 8.16 / 11.31 | 545.5 / 775.7 | 9,105 | -1.80% |
| late-vision | 2,438.3 | 144.5 / 566.1 | 9.65 / 9.70 | 1,527.8 / 1,891.3 | 9,105 | +3.35% |
| multi-image | 193.5 | 323.5 / 552.9 | 9.95 / 11.47 | 631.9 / 786.3 | 9,097 | -20.88% |

The repeated run preserves deterministic captured tokens within every trace.
It wins 4/12 token-throughput rows versus the frozen vLLM table. Section 14
shows that most of this apparent VLM deficit came from the capacity-four test
contract rather than v0.10.1 execution itself.

## 8. Historical comparison

| workload | new v0.10.1 V1 | old-profile v0.10.1 V1 | delta | v0.10.0 V1 | delta | frozen vLLM | delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| short | 2,371.9 | 2,332.1 | +1.71% | 2,478.9 | -4.32% | 1,983.5 | +19.58% |
| balanced | 4,319.7 | 4,202.3 | +2.79% | 4,455.0 | -3.04% | 4,319.9 | -0.00% |
| decode-heavy | 4,965.7 | 4,968.3 | -0.05% | 5,319.2 | -6.65% | 4,854.3 | +2.29% |
| long-prefill | 1,127.0 | 1,118.2 | +0.79% | 1,220.4 | -7.65% | 1,120.9 | +0.54% |
| bimodal | 1,880.9 | 1,926.1 | -2.35% | 1,916.0 | -1.83% | 1,868.3 | +0.67% |
| text-heavy | 1,318.1 | 1,247.3 | +5.68% | 2,131.9 | -38.17% | 1,634.8 | -19.37% |
| mixed | 651.7 | 553.3 | +17.79% | 1,185.1 | -45.01% | 921.5 | -29.28% |
| poisson | 1,509.7 | 1,491.7 | +1.21% | 2,065.8 | -26.92% | 1,800.1 | -16.13% |
| vision-heavy | 356.2 | 321.1 | +10.93% | 687.9 | -48.22% | 579.2 | -38.50% |
| wave-drain | 94.1 | 93.2 | +0.96% | 98.0 | -3.99% | 95.8 | -1.78% |
| late-vision | 2,449.1 | 2,265.2 | +8.12% | 2,552.3 | -4.04% | 2,359.2 | +3.81% |
| multi-image | 195.0 | 167.9 | +16.14% | 312.1 | -37.52% | 244.5 | -20.25% |

The v0.10.0 and frozen vLLM rows are retained historical results, not fresh
runs. Their request/model/precision contract is unchanged, so they are useful
directional anchors; they are not confidence-interval claims for this commit.

## 9. Why this is not KV cache

The two v0.10.1 engine comparisons have identical:

```text
physical KV pages       256
logical stable slots     80
slot capacity          2048 tokens
ownership              indexed-paged stable lease
eviction               logical only; no KV compaction copy
```

No KV tensor shape, page size, lease rule, or allocator code changed. The
approximately 220--264 MiB resident-memory recovery tracks the removed
TensorRT prefill profile workspace. KV remains the largest fixed resident pool
at roughly 3,584 MiB, but it is not responsible for this delta.

## 10. Encoder calibration repair

The Qwen2/Cosmos multimodal runner inherited a zero-valued generic token
estimator. Consequently controlled encoder calibration rejected valid image
requests before executing E. The runner now estimates:

```text
ViT input tokens = ceil(frames / temporal_patch)
                 * resized_height / patch
                 * resized_width / patch

LLM visual tokens = ViT input tokens / merge_size^2
```

It uses the same model-specific smart-resize virtual method as real inference,
so Qwen2-VL, Qwen2.5-VL, and Cosmos retain their own resize contract. The
repaired calibration completed three encoder shapes with 12 isolated and 12
E+D samples. On the mixed trace, however, all 24 contextual E+D opportunities
still selected serial execution, and throughput did not improve. Missing E
samples were therefore a real instrumentation defect but not the primary
v0.10.1 VLM regression.

## 11. Validation

Completed:

- current ONNX -> recovered engine build -> HTTP inference;
- same-engine V0/V1/V2 12-workload gate;
- environment-free automatic external-prefill context mixed HTTP smoke;
- controlled Qwen/Cosmos encoder calibration with real images;
- full `unitTestRuntime`: 619 passed, one two-GPU NCCL test skipped on the
  single-GPU host;
- captured-token determinism within every repeated run and V0 identity on most
  cross-policy traces.

The old v0.10.0 ONNX could not be rebuilt directly by the current builder. Its
`last_token_ids` contract is `[B,1]`, while the v0.10.1 packed builder profiles
`[1,B]`; TensorRT rejects the profile before plan build. A compatibility shim
would change the tested software contract and was not added merely to obtain a
number.

Retained raw results:

```text
.local/results/v0101-forward-port/activity-current-independent-v0
.local/results/v0101-forward-port/activity-current-ed-exclusion-v0
.local/results/v0101-forward-port/encoder-calibrated-v1-smoke
.local/results/v0101-forward-port/no-vision-profile-v1-smoke-dedicated
.local/results/v0101-forward-port/no-vp-full12-v0
.local/results/v0101-forward-port/no-vp-full12-v1
.local/results/v0101-forward-port/no-vp-full12-v2
.local/results/v0101-forward-port/no-vp-full12-v0-v1-v2.{json,csv}
.local/results/v0101-forward-port/no-vp-auto-context-v1-smoke
.local/results/v0101-forward-port/no-vp-full12-v1-r3
```

## 12. Cleanup

Removed before the experiment:

- an incompatible 3.3 GiB v0.10 engine;
- an obsolete D80 engine;
- scratch and stale-worktree build products;
- old scratch/legacy build directories.

About 4.8 GiB was recovered before measurement. After the replacement passed
the full gate, its superseded dedicated-P1024 plan and a failed empty build
directory were also removed, recovering another 3.5 GiB. These ignored binary
artifacts were deleted permanently and are not recoverable from Git. The current ONNX, recovered
engine, visual engine, model, retained result summaries, and frozen comparison
tables remain.

## 13. Decision and next work

Promote this engine topology and V1 Scalar together for v0.10.1. Keep V0 as the
mechanism baseline and V2 as a research variant; do not select V2 globally on
the current cost surface.

Next work is ordered by the remaining measured gap:

1. decompose the remaining VLM gap into E execution, visual preprocessing and
   copy, P chunk/dispatch formation, D continuity, and host realization;
2. compare v0.10.0 and v0.10.1 request/embedding placement and P dispatch
   telemetry rather than tuning a workload-specific policy;
3. rerun vLLM only after the runtime or workload contract changes materially;
4. retain P128 and profile-free V0/V1/V2 evaluation until a controlled result
   proves that another engine shape has better cross-workload SLO goodput.

## 14. Encoded-vision capacity confound and recovery

The old v0.10.0 champion and the initial v0.10.1 promotion run did not have the
same downstream ownership frontier:

```text
old v0.10.0 champion
  normal encoded capacity       16 requests
  guarded throughput capacity   80 requests
  generic calibration           v7 text / v7 VLM small-D traces

initial v0.10.1 P128 result
  normal encoded capacity        4 requests
  guarded throughput capacity    4 requests
  generic calibration           earlier text / VLM traces
```

The capacity is not a workload label and does not directly force a batch size.
It bounds how many request-owned encoder results may wait for or execute through
P/D. The existing age, TPOT-pressure, byte, and stable-page guards still decide
when the larger frontier is safe. Setting both limits to four removed the queue
state needed to form useful E and external-P cohorts.

Restoring the old workload-independent contract on the recovered P128 engine
produced these one-run V0/V1/V2 results:

| workload | V0 tok/s | V1 tok/s | V2 tok/s | V1 vs V0 | V2 vs V0 |
|---|---:|---:|---:|---:|---:|
| short | 2,370.1 | 2,323.6 | 2,309.6 | -1.96% | -2.55% |
| balanced | 4,279.5 | 4,139.5 | 4,099.9 | -3.27% | -4.20% |
| decode-heavy | 4,957.0 | 4,998.5 | 4,680.7 | +0.84% | -5.58% |
| long-prefill | 1,044.1 | 1,171.8 | 1,174.6 | +12.23% | +12.50% |
| bimodal | 1,800.8 | 1,818.4 | 1,789.8 | +0.98% | -0.61% |
| text-heavy | 1,731.1 | 1,873.7 | 1,727.6 | +8.24% | -0.20% |
| mixed | 936.5 | 1,042.0 | 1,067.1 | +11.27% | +13.95% |
| poisson | 1,656.4 | 1,802.2 | 1,683.4 | +8.80% | +1.63% |
| vision-heavy | 622.7 | 563.7 | 629.8 | -9.48% | +1.14% |
| wave-drain | 95.1 | 96.2 | 95.5 | +1.19% | +0.42% |
| late-vision | 2,438.8 | 2,390.6 | 2,358.0 | -1.98% | -3.31% |
| multi-image | 219.5 | 262.6 | 219.1 | +19.64% | -0.19% |

Aggregate token throughput is `+3.58%` for V1 and `+0.92%` for V2 relative
to V0. V1 wins 8/12 rows and remains the promoted production policy. V2 wins
5/12 and remains a research-only transition variant. This gate uses one run per
cell and is the causal same-binary/same-engine policy comparison.

## 15. Corrected V1 three-run promotion result

The final V1 matrix uses the restored capacity/calibration contract. All values
are medians across three runs except that the mean columns are the median of run
means. Historical v0.10.0 and frozen vLLM values retain the same request, model,
precision, and output-length contract but are not fresh confidence-interval
runs.

| workload | tok/s | TTFT mean/p95 | TPOT mean/p95 | E2E mean/p95 | peak MiB | vs v0.10.0 | vs vLLM | hash stable |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| short | 2,325.2 | 110.4 / 192.8 | 12.89 / 23.36 | 357.1 / 438.8 | 9,061 | -6.20% | +17.23% | yes |
| balanced | 4,137.4 | 71.1 / 179.3 | 13.42 / 15.22 | 1,209.9 / 1,905.3 | 9,061 | -7.13% | -4.22% | yes |
| decode-heavy | 4,950.9 | 72.7 / 195.2 | 11.27 / 12.01 | 2,981.1 / 4,589.9 | 9,061 | -6.92% | +1.99% | yes |
| long-prefill | 1,181.4 | 2,135.4 / 2,712.6 | 26.48 / 32.22 | 4,381.9 / 6,246.5 | 9,061 | -3.20% | +5.40% | yes |
| bimodal | 1,815.5 | 2,037.3 / 4,221.6 | 18.81 / 26.73 | 4,660.2 / 9,677.9 | 9,061 | -5.25% | -2.83% | yes |
| text-heavy | 1,728.9 | 332.8 / 1,198.4 | 29.95 / 45.43 | 1,861.8 / 1,939.6 | 9,173 | -18.90% | +5.76% | no |
| mixed | 996.9 | 782.9 / 2,305.6 | 36.32 / 52.97 | 2,526.7 / 2,838.8 | 9,229 | -15.88% | +8.18% | no |
| poisson | 1,781.9 | 244.2 / 964.3 | 23.89 / 41.06 | 1,794.9 / 2,260.8 | 9,193 | -13.74% | -1.01% | yes |
| vision-heavy | 578.0 | 1,544.0 / 3,747.8 | 37.85 / 55.82 | 2,983.9 / 4,184.1 | 9,185 | -15.98% | -0.21% | no |
| wave-drain | 96.0 | 291.3 / 447.9 | 9.91 / 13.65 | 613.3 / 663.3 | 9,121 | -2.07% | +0.18% | yes |
| late-vision | 2,407.0 | 156.0 / 578.5 | 9.78 / 9.83 | 1,553.7 / 1,916.1 | 9,141 | -5.69% | +2.03% | yes |
| multi-image | 220.4 | 389.1 / 499.4 | 9.52 / 12.25 | 684.0 / 721.9 | 9,113 | -29.37% | -9.85% | yes |

The corrected V1 matrix is `+12.65%` geometric mean over the restrictive
capacity-four matrix. It is `+1.68%` geometric mean versus frozen vLLM with
7/12 wins, but remains `-11.22%` versus the old v0.10.0 V1 table. The old
version gap is largest in multi-image, text-heavy, mixed, and vision-heavy.

Text-heavy, mixed, and vision-heavy do not preserve the exact greedy token hash
across all three repeats. This is a numerical/order determinism failure, not a
request-count or completion failure: every run emits the requested number of
tokens. It blocks an exact-fidelity production promotion for those VLM traces
and must be debugged separately from performance.

## 16. Final architecture and remaining work

```text
one CUDA context
  E TensorRT context / E stream
      -> request-owned visual payload lease
      -> encoded-ready queue (normal 16, guarded frontier up to 80)
  P text context + external-P sibling / P stream
      -> one shared P128 context-memory arena
      -> fixed 128-token chunks, stable indexed-paged KV lease
  D TensorRT context / D stream
      -> D64 dynamic cohorts, sampling completion back to ready queue
  Copy stream
      -> pinned/async staging and explicit CUDA-event dependencies

Global V0/V1/V2 mechanism
  feasibility -> request SLO safety -> measured action value -> dispatch
```

The next work is no longer another broad capacity or KV redesign:

1. restore canonical VLM row/request ordering until the three failing traces
   have repeat-stable greedy hashes;
2. compare old/new E dispatch count, E input-token mass, external-P dispatch
   count, and P/D GPU service on multi-image and vision-heavy;
3. isolate whether the remaining 11.22% old-version gap is encoder export,
   external-embedding placement, or runtime realization;
4. repeat only the affected V1 traces after each controlled mechanism change;
5. rerun the frozen vLLM baseline only if model, requests, precision, memory
   budget, or server contract changes.

Retained corrected results:

```text
.local/results/v0101-forward-port/no-vp-v010-contract-full12-v0
.local/results/v0101-forward-port/no-vp-v010-contract-v1
.local/results/v0101-forward-port/no-vp-v010-contract-full12-v2
.local/results/v0101-forward-port/no-vp-v010-contract-v0-v1-v2.{json,csv}
.local/results/v0101-forward-port/no-vp-v010-contract-full12-v1-r3
```
