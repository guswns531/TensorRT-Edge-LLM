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

The port is functionally complete for the declared v1 scope. It is not yet a performance replacement for the retained
v0.10.0 configuration: v0.10.1 workspace growth forces the 10 GiB GPU from the old E4/P8/D64 frontier to E1/P8/D32.
Within this new frontier V1 is the best general default. V2 is valuable for long-prefill and bimodal traces but is not
a universal winner.

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
| Runtime-safe VLM frontier | E1, P8, D32, at most one encoded vision request resident |

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

A final rebuild and validation pass produced:

| Check | Result |
|---|---:|
| `unitTestRuntime` | 619 passed, 1 two-GPU NCCL test skipped |
| `unitTestRuntimeState` | 74 passed |
| `test_export_config.py` | 16 passed |
| Pre-commit hooks | all passed |
| Result analyzer regeneration | byte-identical JSON to the retained comparison |

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

## 8. Memory frontier and failed alternatives

The following are real allocation outcomes, not scheduler estimates:

| Configuration | Outcome |
|---|---|
| E2/P8/D32, encoded vision 2 | CUDA OOM |
| E2/P4/D16, KV/runtime cap 64 | CUDA OOM |
| E1/P8/D32, encoded vision 1 | success, ready 9,803 MiB, peak 9,821 MiB |
| E1/P8/D16, runtime cap 16 | success, but needlessly reduces D concurrency |

E2 still fails because the visual runner output and request-owned persistent vision payload coexist during handoff.
Workspace aliasing removed hundreds of MiB but did not remove this duplicate payload lifetime. The next memory task is
a completion-gated zero-copy E-to-P lease or a preallocated shared vision slab. That must preserve the invariant that
E cannot overwrite a feature buffer until every P consumer has completed.

## 9. Promotion decision

The port passes functional and output-fidelity gates for its supported scope:

- single-GPU vanilla text and vision requests;
- continuous admission and asynchronous token/completion polling;
- stable paged KV ownership with no logical-eviction KV copy;
- one-image and two-image placement;
- V0/V1/V2 identical greedy token traces on all retained workloads.

It does not pass the performance portability gate relative to the retained v0.10.0 system or frozen vLLM. The default
policy should therefore be V1 Scalar for the current branch, while V2 remains an experimental variant.

Unsupported paths remain explicit: speculative decoding, audio, tensor parallel phase serving, context reuse, DART,
and calling legacy `handleRequest()` on an undercommitted KV pool.

## 10. Next ordered work

1. **Zero-copy vision lease.** Remove the runner-output to persistent-payload duplication and re-test E2/E4 memory.
2. **Recover D64.** Right-size remaining decode logits/sampling/IO buffers and expand CUDA graph buckets only after
   memory validation.
3. **Re-run the same 12 workloads.** Require identical token hashes and no regression outside the intended frontier.
4. **Re-evaluate V2.** Fix balanced/decode-heavy transition overvaluation without workload-name rules; retain V1 as
   fallback until V2 wins the gate.
5. **Fair external comparison.** Run clean v0.10.1, Current, and vLLM with equal model, precision, request arrival,
   output contract, memory budget, E/P/D limits, and graph mode.
6. **Fresh visual pipeline.** When disk permits, execute visual export, build, and inference entirely from this branch.
7. **Broaden scope only after promotion.** Add context reuse and multi-rank support after the single-GPU gate passes.

The research conclusion is narrower but stronger than an aggregate speedup claim: the v0.10.1 execution substrate can
host the independent phase architecture with full output fidelity, but context workspace and payload lifetimes are now
the limiting resource. Scheduler comparisons are meaningful only after that memory-induced E/D frontier change is
controlled.
