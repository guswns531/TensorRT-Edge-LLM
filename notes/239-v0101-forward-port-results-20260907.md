# v0.10.1 phase forward-port implementation and results

Date: 2026-09-07

## 1. Outcome

The V0/V1/V2 phase-serving system now runs on the official TensorRT Edge-LLM `v0.10.1` runtime layout. The port is
implemented in `.local/v0101-forward-port` on branch `codex/v0101-phase-forward-port`, based directly on upstream
commit `e8b29522938901f6df19ebeedd4b69bc8edbcd97`.

The completed path includes:

- independent TensorRT execution contexts and CUDA streams for prefill and decode;
- an independently queued vision encoder whose workspace is safely shared when 10 GiB cannot hold all contexts;
- the upstream canonical paged KV pool with stable request leases and row-to-page rebinding;
- packed 128-token prefill, continuous admission, asynchronous sampling completion, and text/VLM request adapters;
- public `llm_inference --phaseServing --phasePolicy=exact|scalar|scalar-transition` execution;
- identical greedy token traces across V0/V1/V2 for all 12 retained workloads;
- one-image and two-image semantic VLM inference through the public executable;
- three-repeat V0/V1/V2 measurements using one binary, engine, request set, calibration contract, and memory limit.
- request-owned Qwen/Cosmos encoder outputs bound directly as TensorRT output tensors, eliminating the retained-output
  device copy;
- a follow-up E4/P8/D32 single-run matrix that restores encoder batching while preserving the same binary, engine,
  traces, and policy calibration.

The port is functionally complete for the declared v1 scope. Direct encoder-output binding restores E4 when decode is
limited to D32, but it does not recover E4/D64: the runner's internal maximum-size output tensors remain resident even
when TensorRT writes into request-owned storage. E4/D32 is therefore a measured high-throughput frontier with only
23--35 MiB of framebuffer headroom, while E1/D32 remains the conservative configuration. V1 is the best general
default in both frontiers. V2 is valuable for long-prefill and bimodal traces but is not a universal winner.

## 2. Source and artifact identity

| Item | Identity |
|---|---|
| Official source base | `v0.10.1`, `e8b29522938901f6df19ebeedd4b69bc8edbcd97` |
| Forward-port branch | `codex/v0101-phase-forward-port` |
| Model | `nvidia/Cosmos-Reason2-2B`, FP16 |
| Export container | `nvcr.io/nvidia/pytorch:25.12-py3` |
| Build/runtime container | `nvcr.io/nvidia/tensorrt:26.06-py3` |
| TensorRT / CUDA | 11.0.0 / 13.3 |
| GPU | NVIDIA GeForce RTX 3080, 10 GiB, SM86 |
| Text engine | max B80, P8, D64, vision-P4, KV capacity 2048, 256 pages, packed chunk 128 |
| Conservative VLM frontier | E1, P8, D32, at most one encoded vision request resident |
| Measured high-throughput frontier | E4, P8, D32; 9,839--9,851 MiB peak, not a 512 MiB-headroom production setting |

Retained local artifacts are indexed by `.local/results/v0101-forward-port/build-manifest.txt`. The text ONNX and
engine hashes are recorded there. The 4 GiB ONNX external-data sidecar was deleted only after hashing and successful
engine construction because the filesystem had less than 5 GiB free. The retained engine remains reproducible from
the recorded checkpoint, revision, export image, and options.

The visual engine predates this worktree but was loaded and executed successfully by the v0.10.1 runtime. This is a
runtime-compatibility and end-to-end inference result, not a fresh v0.10.1 visual export/build claim. A fresh visual
export was intentionally not attempted with the remaining disk headroom.

## 3. Architecture

```text
LLMInferenceRuntime public facade
             |
             v
RuntimeCoordinator -------------------------- multi-rank boundary retained
             |
             v
LLMRankRuntime (phase-only construction)
             |
             +--> PhaseServingRuntime
                     |
                     +--> PhaseThreeCoordinator
                     |       E queue / request DAG / vision leases
                     |
                     +--> IndependentPhaseAsyncServer
                     |       P queue / D queue / continuous admission
                     |
                     +--> IndependentEngineExecutorPair
                     |       one deserialized TRT engine
                     |       +-- P execution context + P stream
                     |       +-- D execution context + D stream
                     |
                     +--> MultimodalRunner + E stream
                     +--> Copy stream
                     +--> StableKVPageManager
                             logical request -> stable page lease
                             active row -> page table
                             no KV compaction on row eviction
```

There is one CUDA primary context per process/GPU. TensorRT execution contexts are independent. Cross-stream data
dependencies use CUDA events; independent ready work may overlap only when workspace ownership and the selected
action permit it.

### 3.1 Memory-constrained workspace mode

The measured context workspaces are:

| Context | Bytes | MiB |
|---|---:|---:|
| Prefill | 197,136,384 | 188.00 |
| Decode | 539,497,472 | 514.50 |
| Vision encoder | 444,873,728 | 424.26 |

Allocating all three independently is not feasible together with model weights, KV pages, IO, logits, and vision
payloads on the 10 GiB card. The runtime therefore chooses a capability-preserving constrained mode:

```text
P workspace: independent 188 MiB

shared E/D arena: max(424, 515) MiB
  E owns arena -> D dispatch blocked
  D owns arena -> E dispatch blocked

available overlap: E+P and P+D
unavailable overlap: E+D
```

This is workspace aliasing, not CUDA-context merging. E and D retain distinct TensorRT execution contexts and streams,
but their device-memory arena has mutually exclusive lifetime. The engine supports D64; the runtime uses D32 in this
10 GiB configuration to keep phase IO and transient allocations below the measured limit.

### 3.2 KV ownership

The physical layout remains the v0.10.1 canonical paged pool. Stable ownership is layered above it:

```text
request admission
    -> reserve stable lease (generation checked)
    -> publish active-row page-table view asynchronously
    -> P/D append through the same physical pages
logical row compaction
    -> reorder request and slot IDs only
    -> no KV tensor copy
completion/cancel
    -> release only after all GPU consumers complete
```

Legacy `handleRequest()` is rejected for an undercommitted pool so that it cannot bypass stable lease accounting.

## 4. Code map

| Area | Primary files | Result |
|---|---|---|
| Public API and output metrics | `examples/llm/llm_inference.cpp` | Async text/VLM submit, poll, token completion, TTFT/TPOT/E2E JSON |
| Runtime attachment | `cpp/runtime/llmRankRuntime.*`, `cpp/runtime/multiDevice/runtimeCoordinator.*` | Phase-only rank-local construction below the upstream facade |
| Phase runtime | `cpp/runtime/scheduling/phaseServingRuntime.*` | Contexts, IO sizing, workspace mode, stable ownership, callbacks |
| Three-phase DAG | `cpp/runtime/scheduling/phaseThreeCoordinator.*` | E queue, E-to-P transition, vision lifetime, E/P/D action arbitration |
| Independent contexts | `cpp/runtime/scheduling/independentEngineExecutorPair.*` | Sibling contexts, per-profile memory, shared E/D arena |
| Async server | `cpp/runtime/scheduling/independentPhaseAsyncServer.*` | Continuous admission, P/D queues, completion and cancellation |
| KV lease | `cpp/runtime/kvCacheManager.*` | Stable paged ownership and undercommit validation |
| External weights | `cpp/runtime/state/externalWeightManager.*` | One validation with multiple phase tensor maps |
| Export/build metadata | `tensorrt_edgellm/checkpoint/checkpoint_utils.py`, `cpp/builder/llmBuilder.cpp` | Packed-prefill and asymmetric profile metadata propagation |
| Comparison tool | `benchmarks/phase_serving/compare_http_policy_variants.py` | Contract checks, token-hash identity, absolute and relative CSV/JSON |
| Direct encoder output | `cpp/multimodal/qwen2/qwenViTRunner.*`, `cpp/multimodal/qwen3/qwen3vlViTRunner.*` | Main and deepstack outputs bind directly to request-owned leases |

The branch is split into reviewable stages:

| Commit | Scope |
|---|---|
| `fa058e5` | Phase scheduling, telemetry, ownership, benchmark and test substrate |
| `f8ad02e` | v0.10.1 executor/config/runtime interface integration |
| `dc8b7d0` | Packed-prefill profiles and indexed-paged contracts |
| `9ce089e` | Production asynchronous phase-serving surface |
| `9cf0b01` | Vision-to-prefill queue connection |

## 5. Functional validation

### 5.1 Export, build, inference

The validation sequence was executed in the required order:

```text
Cosmos FP16 checkpoint
  -> packed-prefill ONNX export
  -> TensorRT engine build for SM86
  -> public llm_inference text execution
  -> public llm_inference one-image and two-image VLM execution
  -> HTTP-style 12-workload phase execution
```

The retained two-image semantic run is
`.local/results/v0101-forward-port/public-phase-v2-vlm-output16.json`:

| Metric | Result |
|---|---:|
| Requests / generated tokens | 2 / 32 |
| Wall time | 441.743 ms |
| Request / token throughput | 4.528 req/s / 72.440 token/s |
| TTFT mean / p95 | 151.276 / 155.166 ms |
| TPOT mean / p95 | 12.760 / 15.351 ms |
| E2E mean / p95 | 342.671 / 385.438 ms |

The output refers to both images rather than asking the caller to provide an image again. This closes the previous
multi-image placement issue for the tested Cosmos contract.

A final rebuild and validation pass after the direct-output port produced:

| Check | Result |
|---|---:|
| `unitTestRuntime` | 619 passed, 1 two-GPU NCCL test skipped |
| `unitTestRuntimeState` | 74 passed |
| `test_export_config.py` | 16 passed |
| Pre-commit hooks | all passed |
| Result analyzer regeneration | byte-identical JSON to the retained comparison |

The direct-output integration was also exercised by the real HTTP multi-image path. The E1 run recorded five direct
encoder batches, 48,300,032 bytes of direct output, and zero retained-output D2D operations or bytes. The previous
fallback path performed 20 D2D operations for the same 48,300,032 bytes. The request-owned lease and its CUDA-ready
event remain unchanged: P cannot consume the tensors before E completion, and the storage pool cannot reuse them
until every P consumer has completed.

The rebuilt `llm_phase_context_smoke` then completed a 48-request short HTTP/IPC trace with 1,040 generated tokens,
94.293 req/s, deterministic token hash `9e44a8e...6726a`, 95.320/182.339 ms TTFT mean/p95,
15.240/28.245 ms TPOT mean/p95, and 374.191/479.868 ms E2E mean/p95. This run is a post-build smoke, not a
replacement for the three-repeat matrix below.

### 5.2 Public text policy smoke

Three repeats per policy, 32-token output, identical greedy tokens:

| Policy | req/s | token/s | TTFT mean/p95 ms | TPOT mean/p95 ms | E2E mean/p95 ms |
|---|---:|---:|---:|---:|---:|
| V0 Exact | 78.181 | 1,407.254 | 147.520 / 162.065 | 9.029 / 10.341 | 301.021 / 304.458 |
| V1 Scalar | 79.855 | 1,437.386 | 142.047 / 156.418 | 8.974 / 10.267 | 294.604 / 298.046 |
| V2 Scalar+Transition | 79.682 | 1,434.272 | 143.128 / 157.578 | 8.950 / 10.250 | 295.280 / 298.705 |

V1 is 2.14% faster than V0; V2 is 1.92% faster than V0 and 0.22% slower than V1 in this smoke.

## 6. Fair 12-workload V0/V1/V2 comparison

All variants use the same binary, engine, requests, E1/P8/D32 runtime frontier, packed chunk 128, calibration contract,
and memory limits. Each workload was executed three times. The comparison tool refuses to emit results unless the
per-run token trace hashes match V0; all 12 workloads passed that check.

### 6.1 Throughput

| Workload | V0 req/s | V1 req/s | V1 vs V0 | V2 req/s | V2 vs V0 | Winner |
|---|---:|---:|---:|---:|---:|---|
| balanced | 32.131 | 31.252 | -2.74% | 29.864 | -7.06% | V0 |
| bimodal | 10.173 | 10.157 | -0.16% | 11.203 | +10.12% | V2 |
| decode-heavy | 12.480 | 12.643 | +1.31% | 11.676 | -6.44% | V1 |
| late-vision | 15.204 | 15.568 | +2.40% | 15.119 | -0.56% | V1 |
| long-prefill | 10.429 | 12.897 | +23.67% | 13.683 | +31.20% | V2 |
| mixed | 6.766 | 6.782 | +0.23% | 6.800 | +0.50% | V2 |
| multi-image | 3.477 | 3.508 | +0.89% | 3.510 | +0.94% | V2 |
| poisson | 11.218 | 11.465 | +2.20% | 11.198 | -0.18% | V1 |
| short | 90.871 | 94.939 | +4.48% | 88.696 | -2.39% | V1 |
| text-heavy | 12.336 | 12.452 | +0.94% | 12.291 | -0.37% | V1 |
| vision-heavy | 4.656 | 4.623 | -0.70% | 4.653 | -0.05% | V0 |
| wave-drain | 2.683 | 2.683 | 0.00% | 2.684 | +0.02% | V2 |

Geometric-mean request throughput relative to V0:

- V1: **+2.52%**, faster on 8 of 12 workloads;
- V2: **+1.74%**, faster on 5 of 12 workloads.

V1 is therefore the current promotion candidate. V2 demonstrates a real transition benefit in long-prefill and
bimodal execution, but its current rollout objective overvalues that benefit in balanced and decode-heavy traces.

### 6.2 Complete request latency

Each cell is `mean / p95` in milliseconds.

| Workload | Policy | TTFT | TPOT | E2E |
|---|---|---:|---:|---:|
| balanced | V0 | 60.431 / 190.053 | 20.992 / 23.727 | 1846.464 / 2961.704 |
|  | V1 | 57.049 / 185.484 | 21.551 / 25.810 | 1897.487 / 3169.481 |
|  | V2 | 57.265 / 191.404 | 22.607 / 26.910 | 1989.011 / 3275.784 |
| bimodal | V0 | 2374.450 / 5288.700 | 22.621 / 35.682 | 5491.821 / 11546.334 |
|  | V1 | 2356.659 / 5387.602 | 22.957 / 36.043 | 5490.981 / 11801.107 |
|  | V2 | 2155.179 / 4467.753 | 19.788 / 29.064 | 4913.684 / 10145.824 |
| decode-heavy | V0 | 54.114 / 198.132 | 18.085 / 19.719 | 4722.150 / 7542.477 |
|  | V1 | 61.730 / 205.056 | 17.773 / 19.361 | 4649.956 / 7389.714 |
|  | V2 | 57.994 / 208.338 | 19.304 / 21.284 | 5044.158 / 8086.114 |
| late-vision | V0 | 174.026 / 738.731 | 10.747 / 10.797 | 1713.571 / 2102.785 |
|  | V1 | 163.710 / 692.864 | 10.495 / 10.546 | 1667.138 / 2053.034 |
|  | V2 | 163.353 / 690.424 | 10.832 / 10.902 | 1715.048 / 2114.320 |
| long-prefill | V0 | 2674.427 / 3445.793 | 37.067 / 51.508 | 5792.170 / 8327.419 |
|  | V1 | 2256.352 / 2983.803 | 28.417 / 37.489 | 4656.509 / 6792.108 |
|  | V2 | 2149.763 / 3054.402 | 26.396 / 34.780 | 4376.632 / 6532.464 |
| mixed | V0 | 2452.472 / 8287.723 | 9.435 / 13.293 | 2907.906 / 8490.991 |
|  | V1 | 2448.388 / 8253.891 | 9.244 / 13.001 | 2894.399 / 8461.142 |
|  | V2 | 2412.705 / 8225.972 | 9.336 / 12.851 | 2862.742 / 8433.292 |
| multi-image | V0 | 647.304 / 1164.553 | 6.570 / 6.674 | 850.988 / 1371.020 |
|  | V1 | 635.968 / 1151.897 | 6.556 / 6.653 | 839.198 / 1358.130 |
|  | V2 | 636.224 / 1151.565 | 6.543 / 6.648 | 839.062 / 1357.646 |
| poisson | V0 | 775.294 / 4128.658 | 13.867 / 21.998 | 1790.148 / 4332.066 |
|  | V1 | 746.624 / 4007.303 | 13.202 / 19.927 | 1718.028 / 4210.424 |
|  | V2 | 781.064 / 4138.818 | 13.526 / 20.504 | 1774.662 / 4341.996 |
| short | V0 | 109.861 / 194.470 | 15.283 / 24.824 | 396.316 / 498.240 |
|  | V1 | 98.664 / 192.673 | 14.997 / 28.344 | 372.766 / 469.121 |
|  | V2 | 102.789 / 196.313 | 16.659 / 32.327 | 404.692 / 508.538 |
| text-heavy | V0 | 790.862 / 3973.755 | 13.093 / 18.345 | 1503.709 / 4176.505 |
|  | V1 | 785.085 / 3923.961 | 12.794 / 17.684 | 1480.432 / 4126.652 |
|  | V2 | 786.034 / 3991.029 | 13.607 / 18.873 | 1528.821 / 4194.136 |
| vision-heavy | V0 | 5147.601 / 12570.261 | 7.703 / 11.713 | 5455.349 / 12777.554 |
|  | V1 | 5228.577 / 12676.493 | 7.867 / 12.032 | 5545.640 / 12879.936 |
|  | V2 | 5142.996 / 12585.247 | 7.778 / 11.953 | 5454.861 / 12788.483 |
| wave-drain | V0 | 634.950 / 1214.559 | 6.569 / 6.700 | 838.588 / 1422.207 |
|  | V1 | 635.591 / 1215.432 | 6.571 / 6.695 | 839.300 / 1422.864 |
|  | V2 | 634.567 / 1214.056 | 6.565 / 6.697 | 838.082 / 1421.573 |

Peak memory is 9,803 MiB on text-only traces, 9,813 MiB on late-vision, and 9,821 MiB on the other VLM traces.

## 7. vLLM and retained v0.10.0 interpretation

The frozen vLLM results are useful as a directional external reference, but they are not a like-for-like performance
claim for this port. The old comparison used the retained v0.10.0 E4/P8/D64 Current configuration; the v0.10.1 port
uses E1/P8/D32 because of its larger context workspaces. CUDA graph coverage also differs.

Against the frozen vLLM token-throughput table, geometric means are:

| Variant | Relative token throughput | Workload wins |
|---|---:|---:|
| V0 | -40.26% | 0 / 12 |
| V1 | -38.76% | 1 / 12 |
| V2 | -39.22% | 1 / 12 |

The retained v0.10.0 report had V2 at +15.16% geometric mean and 12/12 wins over that same frozen vLLM table. The
sign reversal must not be attributed to policy quality: the dominant contract change is D64 to D32 plus E4 to E1,
with less graph/replay coverage. A fair clean-v0.10.1/current/vLLM headline comparison requires either restoring a
larger v0.10.1 frontier or memory-normalizing all three systems.

No separate clean-v0.10.1 engine was built. Doing so would require another roughly 4.1 GiB while only about 4.0 GiB
was free. The source baseline and source delta are documented in `237-upstream-v0101-code-comparison-20260907.md`;
performance claims are deliberately deferred instead of using the Current engine as a clean-upstream proxy.

## 8. Direct-output memory frontier

The following are real allocation outcomes, not scheduler estimates:

| Configuration | Outcome |
|---|---|
| E1/P8/D32 | success, 9,821 MiB in the retained matrix and 9,821 MiB in the direct-output E1 check |
| E2/P8/D32 | success, 9,825 MiB peak; multi-image 5.107 req/s |
| E4/P8/D32 | success, 9,839 MiB in the focused run and up to 9,851 MiB in the 12-workload matrix |
| E1/P8/D64 | success, 9,871 MiB peak and approximately 3 MiB allocatable headroom |
| E2/P8/D64 | CUDA OOM while allocating measurement-request storage |
| E4/P8/D64 | CUDA OOM while allocating measurement-request storage |

The new binding removes the transient runner-output-to-retained-payload transfer, but it does not release the Qwen3
runner's internal maximum output and three deepstack buffers. This distinction matters: the data path is copy-free,
but the static memory footprint is not yet single-buffered. Recovering D64 together with E4 requires eliminating or
right-sizing that internal backing, reducing another persistent pool, or changing the engine/runtime memory contract.
E1/D64 is an allocation boundary probe, not a deployable configuration.

## 9. Promotion decision

The conservative E1 port passes functional and output-fidelity gates for its supported scope:

- single-GPU vanilla text and vision requests;
- continuous admission and asynchronous token/completion polling;
- stable paged KV ownership with no logical-eviction KV copy;
- one-image and two-image placement;
- V0/V1/V2 identical greedy token traces on all retained workloads.

The E4 follow-up passes completion and semantic VLM checks, but exact cross-policy token identity is 9/12. The three
exceptions are mixed, text-heavy, and vision-heavy; 3/64, 4/64, and 4/64 requests respectively differ, all in vision
rows. The outputs remain semantically image-conditioned, but E4 cannot replace the conservative exact-identity gate
until canonical execution/row-order sensitivity is resolved or a stated numerical-fidelity contract replaces exact
greedy identity.

Neither frontier passes the performance portability gate relative to the retained v0.10.0 E4/P8/D64 system or frozen
vLLM. V1 Scalar remains the default policy candidate, while V2 remains an experimental transition ablation.

Unsupported paths remain explicit: speculative decoding, audio, tensor parallel phase serving, context reuse, DART,
and calling legacy `handleRequest()` on an undercommitted KV pool.

## 10. Next ordered work

1. **Make direct output single-buffered.** Remove or lazily allocate the now-unused Qwen/Cosmos internal output and
   deepstack backing while retaining the request-owned double-buffering needed for E/P pipelining.
2. **Recover D64 with headroom.** Right-size decode logits/sampling/IO buffers and expand CUDA graph buckets only after
   proving at least 512 MiB of deployable headroom or explicitly adopting a lower memory target.
3. **Resolve E4 numerical sensitivity.** Canonicalize row/cohort ordering and repeat the three divergent VLM traces;
   keep the strict comparison tool unchanged.
4. **Make E cohort formation state-driven.** Preserve E4's burst benefit without using workload names, and bound E
   wait by first-token slack and observable ready/event state.
5. **Re-evaluate V2.** Fix balanced/decode-heavy transition overvaluation without workload-name rules; retain V1 as
   fallback until V2 wins the gate.
6. **Fair external comparison.** Run clean v0.10.1, Current, and vLLM with equal model, precision, request arrival,
   output contract, memory budget, E/P/D limits, and graph mode.
7. **Fresh visual pipeline.** When disk permits, execute visual export, build, and inference entirely from this branch.
8. **Broaden scope only after promotion.** Add context reuse and multi-rank support after the single-GPU gate passes.

The research conclusion is narrower but stronger than an aggregate speedup claim: the v0.10.1 execution substrate can
host the independent phase architecture and direct request-owned vision output. The limiting resource is now static
context/IO/output storage rather than the E-to-P copy itself. Scheduler comparisons remain meaningful only when the
memory-induced E/D frontier and numerical-order contract are reported explicitly.

## 11. E4/P8/D32 follow-up matrix

This matrix is a one-run engineering sweep performed after direct output was enabled. It reuses one binary, one engine,
the same 12 traces, the same 319-request VLM/260-request text generic calibration, fixed P128, P8, D32, and identical
memory settings. Only the policy authority changes. It complements rather than replaces the three-repeat E1 matrix.

| Workload | Policy | req/s | token/s | TTFT mean/p95 ms | TPOT mean/p95 ms | E2E mean/p95 ms | peak MiB |
|---|---|---:|---:|---:|---:|---:|---:|
| short | V0 | 92.864 | 2012.1 | 98.7 / 186.4 | 15.66 / 29.49 | 384.4 / 484.2 | 9803 |
|  | V1 | 93.499 | 2025.8 | 101.2 / 186.5 | 15.20 / 25.23 | 381.7 / 483.6 | 9803 |
|  | V2 | 91.052 | 1972.8 | 106.0 / 202.7 | 15.59 / 29.27 | 390.3 / 494.3 | 9803 |
| balanced | V0 | 31.278 | 2710.8 | 54.5 / 186.3 | 21.63 / 25.59 | 1900.1 / 3161.8 | 9803 |
|  | V1 | 31.859 | 2761.2 | 59.0 / 189.9 | 21.12 / 24.95 | 1858.2 / 3085.1 | 9803 |
|  | V2 | 30.551 | 2647.8 | 60.0 / 190.5 | 22.01 / 26.25 | 1936.5 / 3212.2 | 9803 |
| decode-heavy | V0 | 12.538 | 3259.9 | 56.1 / 202.8 | 18.01 / 19.72 | 4705.4 / 7503.0 | 9803 |
|  | V1 | 12.692 | 3300.0 | 60.0 / 202.5 | 17.76 / 19.16 | 4642.7 / 7316.6 | 9803 |
|  | V2 | 11.979 | 3114.5 | 61.3 / 205.3 | 18.89 / 20.86 | 4937.0 / 7926.6 | 9803 |
| long-prefill | V0 | 10.477 | 908.0 | 2664.8 / 3294.1 | 36.84 / 50.24 | 5770.9 / 8233.9 | 9803 |
|  | V1 | 12.639 | 1095.4 | 2303.3 / 3282.2 | 28.92 / 40.43 | 4753.1 / 7047.9 | 9803 |
|  | V2 | 13.966 | 1210.4 | 2109.8 / 2788.3 | 25.51 / 33.13 | 4272.7 / 6003.2 | 9803 |
| bimodal | V0 | 10.268 | 1574.4 | 2339.3 / 5389.8 | 22.54 / 39.65 | 5431.2 / 11469.2 | 9803 |
|  | V1 | 11.190 | 1715.8 | 2173.4 / 4561.2 | 20.83 / 33.60 | 4984.7 / 10221.1 | 9803 |
|  | V2 | 11.412 | 1749.8 | 2119.3 / 4765.1 | 19.69 / 30.22 | 4825.0 / 10091.9 | 9803 |
| text-heavy | V0 | 24.473 | 1297.1 | 481.9 / 2022.2 | 17.32 / 20.47 | 1397.1 / 2494.4 | 9847 |
|  | V1 | 24.359 | 1291.0 | 491.6 / 2042.6 | 16.61 / 20.93 | 1371.7 / 2500.3 | 9851 |
|  | V2 | 22.570 | 1196.2 | 475.5 / 2165.3 | 19.07 / 23.28 | 1485.4 / 2685.7 | 9841 |
| mixed | V0 | 14.220 | 650.6 | 1175.2 / 3867.1 | 14.26 / 17.02 | 1816.0 / 4342.2 | 9849 |
|  | V1 | 14.196 | 649.5 | 1192.9 / 3809.8 | 13.95 / 18.15 | 1829.0 / 4313.4 | 9849 |
|  | V2 | 14.192 | 649.3 | 1152.8 / 3911.2 | 14.70 / 18.31 | 1830.5 / 4350.6 | 9849 |
| poisson | V0 | 20.541 | 1499.5 | 391.9 / 2024.1 | 19.66 / 26.43 | 1808.2 / 2657.9 | 9847 |
|  | V1 | 20.652 | 1507.6 | 396.5 / 2047.4 | 18.92 / 24.37 | 1752.6 / 2598.4 | 9847 |
|  | V2 | 20.205 | 1474.9 | 385.9 / 2069.4 | 20.32 / 25.82 | 1850.7 / 2682.6 | 9843 |
| vision-heavy | V0 | 10.026 | 386.0 | 2322.2 / 5796.5 | 13.82 / 15.80 | 2847.6 / 6254.9 | 9851 |
|  | V1 | 10.333 | 397.8 | 2305.8 / 5668.5 | 12.22 / 16.23 | 2776.1 / 6046.7 | 9851 |
|  | V2 | 10.070 | 387.7 | 2337.5 / 5770.9 | 13.20 / 16.48 | 2846.5 / 6217.7 | 9849 |
| wave-drain | V0 | 2.948 | 94.3 | 288.6 / 555.2 | 7.99 / 10.66 | 536.3 / 761.6 | 9847 |
|  | V1 | 2.948 | 94.3 | 285.7 / 548.2 | 8.01 / 10.85 | 533.9 / 754.7 | 9847 |
|  | V2 | 2.947 | 94.3 | 287.7 / 551.4 | 8.00 / 10.67 | 535.7 / 757.6 | 9847 |
| late-vision | V0 | 16.485 | 2378.0 | 149.0 / 603.6 | 9.90 / 9.95 | 1566.9 / 1938.8 | 9849 |
|  | V1 | 16.497 | 2379.7 | 148.5 / 597.2 | 9.86 / 9.91 | 1561.7 / 1935.0 | 9849 |
|  | V2 | 16.345 | 2357.7 | 144.9 / 577.5 | 9.95 / 10.01 | 1570.0 / 1954.7 | 9849 |
| multi-image | V0 | 6.219 | 199.0 | 317.6 / 531.9 | 9.48 / 10.80 | 611.3 / 763.8 | 9839 |
|  | V1 | 6.364 | 203.6 | 290.7 / 512.4 | 7.93 / 9.74 | 536.5 / 741.7 | 9847 |
|  | V2 | 6.294 | 201.4 | 307.6 / 521.5 | 9.47 / 10.77 | 601.3 / 753.3 | 9839 |

Geometric-mean request throughput relative to V0 is +3.08% for V1 with 9/12 wins and +1.79% for V2 with 4/12
wins. Compared with each policy's retained E1 matrix, E4 is +34.64%/+35.38%/+34.71% for V0/V1/V2, but that is not
a pure batch-size ablation: the E1 results predate direct output and are three-run medians, whereas E4 is one run.
The focused direct-output E1 run changed multi-image throughput by less than one percent, so the large VLM increase is
principally encoder cohort formation rather than copy elimination.

Against the unchanged frozen vLLM token-throughput table, E4 V0/V1/V2 are -19.57%/-17.09%/-18.13% geometric mean.
That improves substantially on the E1 port's -40.26%/-38.76%/-39.22%, but remains below the retained v0.10.0 E4/D64
configuration, which was +15.16% for V2 on the same frozen table. D32 remains the dominant portability difference.

## 12. Single-buffered vision output and the recovered D64 frontier

The remaining reusable v0.10.0 mechanisms were audited before adding another policy change. Stable indexed page
ownership, independent E/P/D contexts, packed P128, async preparation, direct encoder output, early vision-storage
release, shared E/D context memory, canonical row ordering, CUDA graphs, and the V0/V1/V2 policy surface were already
present in this forward port. The missing memory property was narrower: direct output removed the device-to-device
copy but still kept the Qwen/Cosmos runner's maximum-size internal outputs alive.

The implementation now separates output metadata from output backing:

```text
preprocess
   |
   +--> active output specs (shape, dtype)
   |
PhaseVisionAdapter
   +--> request-owned main embedding
   +--> request-owned deepstack[0..2]
   |
bindExternalOutputStorage()
   |
TensorRT vision context writes directly into request-owned storage
   |
E completion event --> P-ready queue --> release after P consumption

Qwen/Cosmos internal maximum outputs: released once at adapter construction
```

For Cosmos-Reason2-2B, the released maximum backing is 2,048 rows x 2,048 hidden x FP16 for the main output and each
of three deepstack outputs: 8 MiB x 4 = 32 MiB. The runner retains only active shape/type metadata. Legacy runners
that cannot bind external output storage keep their internal outputs and the copy fallback, so the optimization does
not change their contract.

The phase smoke runtime also constructs the vision runner at the declared encoder batch limit instead of the LLM's
80-slot capacity. At E4 this reduces the Qwen M-RoPE position-ID device tensor from B80 to B4, saving about 3.6 MiB of
device memory and the same amount of pinned host backing. These two changes are enough to cross the allocation cliff:

| Configuration after change | Result |
|---|---|
| E4/P8/D64, encoded capacity 4 | succeeds on all 12 workloads |
| E4/P8/D64, encoded capacity 8 | CUDA OOM at the first vision-heavy measurement wave |
| E4/P8/D64, encoded capacity 16/throughput cap 80 | CUDA OOM at the first vision-heavy measurement wave |
| E4/P8/D64 with 48 MiB byte admission gate | no progress; count/byte lifetime coupling over-constrains admission |

The byte-gate experiment was rejected rather than promoted. It prevented the OOM but produced a scheduler progress
cycle, so it is not a safe substitute for lifetime-aware byte admission. Capacity 4 is therefore a measured hardware
frontier for this 10 GB configuration, not a workload-tuned preference.

### 12.1 D64 V0/V1/V2 one-run gate

All variants below use the same v0.10.1 binary and engine, E4, P8, fixed P128, D64, encoded-vision capacity 4, generic
workload-agnostic calibration, traces, request ordering, and memory settings. Only `TRT_EDGELLM_PHASE_POLICY` changes.
The table reports milliseconds for latency and one-run engineering results rather than confidence intervals.

| Workload | Variant | req/s | token/s | TTFT mean / p95 | TPOT mean / p95 | E2E mean / p95 | Peak MiB |
|---|---|---:|---:|---:|---:|---:|---:|
| short | V0 | 108.771 | 2356.7 | 104.1 / 189.2 | 13.71 / 27.97 | 354.8 / 435.1 | 9819 |
|  | V1 | 106.688 | 2311.6 | 110.7 / 194.4 | 13.17 / 22.32 | 358.3 / 441.0 | 9819 |
|  | V2 | 104.490 | 2264.0 | 112.1 / 198.2 | 13.69 / 23.02 | 368.9 / 453.3 | 9819 |
| balanced | V0 | 46.300 | 4012.7 | 69.8 / 181.1 | 13.91 / 16.08 | 1255.7 / 1957.6 | 9819 |
|  | V1 | 48.414 | 4195.9 | 70.7 / 177.3 | 13.18 / 15.50 | 1192.6 / 1925.7 | 9819 |
|  | V2 | 46.264 | 4009.5 | 78.5 / 189.0 | 13.76 / 15.53 | 1250.1 / 1951.0 | 9819 |
| decode-heavy | V0 | 19.009 | 4942.4 | 74.9 / 199.4 | 11.29 / 12.05 | 2991.1 / 4615.6 | 9819 |
|  | V1 | 19.135 | 4975.2 | 76.5 / 197.8 | 11.17 / 11.85 | 2962.5 / 4507.2 | 9819 |
|  | V2 | 18.071 | 4698.5 | 76.3 / 209.1 | 11.91 / 12.62 | 3149.1 / 4819.7 | 9819 |
| long-prefill | V0 | 12.385 | 1073.4 | 2280.4 / 3095.4 | 29.93 / 34.90 | 4851.1 / 7045.1 | 9819 |
|  | V1 | 13.806 | 1196.5 | 2129.1 / 2657.4 | 25.84 / 30.83 | 4325.1 / 6072.8 | 9819 |
|  | V2 | 11.973 | 1037.6 | 2403.1 / 3331.1 | 30.75 / 38.07 | 5026.6 / 7603.5 | 9819 |
| bimodal | V0 | 11.740 | 1800.1 | 2026.8 / 4457.3 | 19.04 / 29.20 | 4675.3 / 9668.8 | 9819 |
|  | V1 | 11.918 | 1827.4 | 2035.8 / 4364.9 | 18.12 / 26.52 | 4599.5 / 9900.1 | 9819 |
|  | V2 | 11.734 | 1799.3 | 2059.3 / 4411.6 | 18.90 / 28.39 | 4673.2 / 9755.2 | 9819 |
| text-heavy | V0 | 27.434 | 1454.0 | 431.7 / 1771.7 | 15.52 / 21.53 | 1258.0 / 2218.3 | 9869 |
|  | V1 | 27.373 | 1450.8 | 430.9 / 1841.8 | 14.94 / 20.31 | 1224.8 / 2216.7 | 9863 |
|  | V2 | 26.027 | 1379.5 | 422.2 / 1855.9 | 17.19 / 21.58 | 1343.4 / 2334.4 | 9871 |
| mixed | V0 | 14.655 | 670.4 | 1138.8 / 3766.4 | 14.14 / 17.44 | 1784.4 / 4242.8 | 9869 |
|  | V1 | 14.788 | 676.6 | 1156.3 / 3764.0 | 13.23 / 16.84 | 1770.7 / 4198.7 | 9865 |
|  | V2 | 15.066 | 689.3 | 1068.5 / 3639.6 | 14.05 / 18.39 | 1724.1 / 4106.4 | 9867 |
| poisson | V0 | 22.307 | 1628.4 | 336.7 / 1720.6 | 18.88 / 24.06 | 1718.5 / 2468.6 | 9863 |
|  | V1 | 22.735 | 1659.7 | 338.4 / 1694.8 | 17.37 / 21.92 | 1623.6 / 2357.0 | 9873 |
|  | V2 | 22.065 | 1610.8 | 336.8 / 1756.5 | 19.84 / 26.28 | 1786.8 / 2513.7 | 9859 |
| vision-heavy | V0 | 9.964 | 383.6 | 2358.5 / 5835.5 | 13.74 / 16.17 | 2878.5 / 6294.6 | 9869 |
|  | V1 | 10.017 | 385.6 | 2358.1 / 5793.2 | 13.38 / 16.42 | 2869.9 / 6242.6 | 9873 |
|  | V2 | 10.424 | 401.3 | 2271.6 / 5573.8 | 12.66 / 16.10 | 2764.3 / 5995.3 | 9865 |
| wave-drain | V0 | 2.942 | 94.1 | 286.8 / 561.8 | 8.04 / 10.72 | 536.1 / 769.0 | 9865 |
|  | V1 | 2.948 | 94.3 | 290.5 / 583.6 | 8.15 / 10.85 | 543.0 / 790.6 | 9863 |
|  | V2 | 2.947 | 94.3 | 287.9 / 550.6 | 8.01 / 10.73 | 536.2 / 757.6 | 9865 |
| late-vision | V0 | 16.326 | 2355.1 | 143.1 / 584.3 | 10.00 / 10.05 | 1575.0 / 1957.9 | 9859 |
|  | V1 | 16.433 | 2370.5 | 146.7 / 599.1 | 9.93 / 9.98 | 1568.9 / 1945.4 | 9863 |
|  | V2 | 16.319 | 2354.1 | 147.0 / 593.6 | 10.00 / 10.06 | 1579.0 / 1956.8 | 9863 |
| multi-image | V0 | 6.503 | 208.1 | 290.7 / 494.7 | 7.24 / 8.74 | 515.2 / 717.4 | 9855 |
|  | V1 | 6.188 | 198.0 | 316.1 / 534.9 | 9.72 / 11.31 | 617.4 / 767.7 | 9855 |
|  | V2 | 6.275 | 200.8 | 309.4 / 524.4 | 9.50 / 10.82 | 603.8 / 756.7 | 9855 |

V1 is the D64 aggregate winner: +1.22% token-throughput geometric mean over V0 with 9/12 wins. V2 is -1.25% with
3/12 wins, although it is best on mixed and vision-heavy. V1 improves 8/12 workloads over its D32 result and gains
+11.47% geometric-mean request throughput; balanced and decode-heavy improve by 51.96% and 50.77%. Multi-image and
vision-heavy regress by 2.76% and 3.06%, showing that a wider D frontier is not useful when E dominates or the cohort
is tiny.

The strict comparison tool rejects this matrix because cross-policy greedy identity is 9/12. Mixed, text-heavy, and
vision-heavy differ; all other traces are exact. Therefore V1 is the performance candidate, not a production
promotion. Relative to the unchanged frozen vLLM token-throughput table, V0/V1/V2 are -8.70%/-7.58%/-9.84%
geometric mean, with V1 winning 4/12. D64 removes most of the text-path gap, but v0.10.1 still cannot sustain the
v0.10.0 C16/C80 vision admission frontier within 10 GB. The remaining external gap is primarily VLM capacity, not
evidence that V2 transition reasoning is universally beneficial.

After formatting, the final source was rebuilt and both runtime suites were rerun: `unitTestRuntime` passed 619 tests
with the single two-GPU NCCL test skipped, and `unitTestRuntimeState` passed all 74 tests. A final D64 multi-image V1
inference reproduced the retained token SHA-256 `cd7c5ff866b12e39907ec334e1ccf81dc41f1d98a0aebf67653b3889a8a1da9f`.

### 12.2 Revised next work

1. Keep V1 as the default research candidate and V0 as the conservative exact-cost baseline; V2 remains an ablation.
2. Resolve the three E4 cross-policy numerical divergences before production promotion.
3. Recover roughly 0.6 GiB of v0.10.0's memory advantage. The leading architectural difference is the v0.10.1
   standard ONNX/embedded-weight engine versus the retained v0.10.0 external-weight path; the memory delta prevents
   C8/C16 vision admission even after output-backing reclamation.
4. Add lifetime-aware byte admission only with an explicit progress invariant; the tested byte gate deadlocks.
5. Repeat the decisive D64 points for confidence intervals before any paper headline claim.
6. Run a fresh vLLM comparison only after the model/engine/request/memory contract is made genuinely equal.
