<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# 380 — Cosmos-Reason2-2B on A100: batch tuning and optimized vLLM (2026-10-06)

## Scope and evidence

Resumes the Cosmos capacity sweep left unmeasured in note 379. Only Cosmos-Reason2-2B was downloaded,
exported, built and tested. This campaign searches throughput across batch capacities and representative
traffic; it does not establish a global optimum or a production latency SLO.

- Source: `dd773f05a7010634fa49e18543531bf9eff40c35`, branch `codex/v0110-phase-forward-port`, clean during builds and measurements.
- GPU: NVIDIA A100-SXM4-40GB, 40960 MiB, driver 580.82.07; CUDA compiler 13.0.88.
- Checkpoint: `nvidia/Cosmos-Reason2-2B`, revision `9ce19a195e423419c349abfc86fd07178b230561`.
- TensorRT 11.0.0.114; SDK headers extracted from CUDA 13.2 development packages, matching shared libraries from
  the TensorRT Python wheel. Export: Python 3.12.3, torch 2.13.0+cu130, transformers 5.14.1.
- vLLM 0.31.0 in its own environment, torch 2.13.0+cu132 and transformers 5.17.0; native BF16 checkpoint serving.
- Runtime frozen at `.local/artifacts/runtimes/cosmos-a100-dd773f05a701-20261006/`, rather than a mutable CMake build.
- Retained campaign: `.local/results/a100-cosmos-reason2-2b-fp16/capacity-resume/`; environment/e2e evidence:
  `.local/results/cosmos-restart-20261006/`. Manifests identify binaries, engines, commands, configurations,
  workload hashes, repeat counts and summary paths. Models, tokens, engines and result data remain untracked.

## Export, engine contracts and validation

The FP16 export uses packed prefill with a 1024-token export chunk budget. All runtime engines use 512-token
uniform chunks, input limit 1024, KV capacity 8192 and 1536 KV pool pages with undercommit explicitly enabled.
The visual engine has 22528 total image tokens and 2816 tokens per image; its runtime directory ends in `/visual`.
The `vision-e4` directory name does not impose an E4 runtime limit.

| LLM engine | Build batch | Build prefill | Build decode | Purpose |
|---|---:|---:|---:|---|
| `engine-p8-d64-b80-c512-kv8192-pool1536` | 80 | 8 | 64 | Restart/e2e validation |
| `engine-p32-d256-b256-c512-kv8192-pool1536` | 256 | 32 | 256 | Capacity sweep, E/P grid and final candidate |
| `engine-p32-d512-b512-c512-kv8192-pool1536` | 512 | 32 | 512 | Upper-capacity diagnostics |

All engines came from the same export and run through the same frozen binary. Different engine contracts are
reported separately. Validation exercised export → engine build → runtime inference in that order. The common,
runtime, scheduler and runtime-state C++ test groups passed. No runtime source was changed during this campaign.

V3 independent encoder/prefill/decode serving uses serving probes, dispatch telemetry and measured decode
batching (`TRT_EDGELLM_ENABLE_MEASURED_DECODE_BATCHING=1`). Calibration traces were built explicitly with eight
cycles, covering decode rows through each campaign's upper bound, prefill batches through P32 and encoder
batches through E8. The capacity sweep helper's default two-cycle calibration would not cover these decode
capacities; generating and retaining the eight-cycle input before the sweep is necessary.

## Decode capacity: load scales with D

Each point uses client concurrency D, 4D text requests and 2D mixed requests. P16/E4 and stable slots 256 are
fixed for the D256 engine. Calibration is common within this sweep. One repeat per point; generated tok/s.

| D | Balanced | Decode-heavy | Mixed | Three-workload mean | Worst TPOT p95 (ms) |
|---:|---:|---:|---:|---:|---:|
| 64 | 3629 | 5723 | 885 | 2640 | 82.6 |
| 128 | 4723 | 8194 | 958 | 3335 | 145.7 |
| 192 | 5095 | 9339 | 843 | 3424 | 252.3 |
| 256 | 5348 | 10037 | 960 | 3721 | 273.4 |

The memory probe estimated a 14 MiB KV page and a p95-length decode capacity of about 131, versus about 213
at the mean request length. These are planning bounds, not hard admission limits: undercommit and variable
lifetimes let the D256 engine complete the measured traces. The planner is not a guarantee for every 8192-token
request at these admission counts.

The D512 engine completed text workloads at D256 and D384 but failed every mixed request with CUDA OOM. At
D512 it completed all three workloads after CUDA graph capture OOM reduced cached graphs to 27. Its three-workload
throughput geometric mean was 3479.9 tok/s, below the D256 engine's 3721.4 tok/s. This nonmonotonic fallback is a
diagnostic result, not a safe recommendation to raise capacity to 512. The D256 engine was favored by this screen; the larger final protocol still needed separate validation.
The sweep's 50 ms TPOT p95 SLO was not satisfied by any point; this note selects throughput, not SLO compliance.

## E/P search and repeated confirmation

The nine-point E1/E4/E8 × P8/P16/P32 screen fixes D256, the D256 engine, clients256, 512 text requests and 256
mixed/vision requests. All nine points completed all fixed-length requests. The E8/P32 point led the four-workload
geometric mean in the single-run screen. Five finalists were then measured three times on the same traces,
calibration, binary, engine and memory limits; the table gives median generated tok/s.

| E/P/D | Balanced | Decode-heavy | Mixed | Vision-heavy | Four-workload mean |
|---|---:|---:|---:|---:|---:|
| E8-P32-D256 | 5295 | 9586 | 1004 | 657 | 2405 |
| E4-P32-D256 | 5300 | 9372 | 994 | 637 | 2368 |
| E8-P32-D192 | 5136 | 8840 | 1006 | 645 | 2330 |
| E8-P32-D128 | 4431 | 6841 | 995 | 642 | 2098 |
| E4-P8-D64 | 3895 | 6023 | 962 | 637 | 1947 |

The D64 row is a runtime batch ablation on the D256 engine, not the original b80/D64 engine.

E8/P32/D256 led the repeated four-workload mean by about 24% over E4/P8/D64. E4/P32/D256 was close and slightly
faster on balanced text; E8 is not universally best. The larger decode batch drives the text improvement, while
encoder/prefill settings make smaller differences on these image traces. The screening and confirmation traces
are shorter than the final comparison, so their throughput must not be mixed into the final system ratios.

## vLLM optimization and matched final comparison

The initial vLLM screen varies `--max-num-seqs` 128/256/512 at a 4096-token scheduler budget, then tests
8192/16384-token budgets at the best sequence capacity and a 0.95 memory utilization candidate. Chunked
prefill and asynchronous scheduling are enabled, model length is 8192, and at most two images are admitted
per prompt. Native checkpoint precision is BF16 for vLLM versus FP16 for TensorRT; this is a deployment
comparison, not a numerical precision equivalence test.

The final protocol fixes client concurrency at 512 for both systems and uses identical workload files:
1024 balanced/decode-heavy requests and 512 mixed/vision-heavy requests. TensorRT internal admission remains
256 (`TRT_EDGELLM_MAX_INFLIGHT=256`), matching its engine stable-slot limit; the gateway queues extra clients.
The workload output length is fixed and EOS is ignored on both systems. Measured throughput excludes
server startup and warmup, so this is warmed serving rather than a cold-start comparison. Generic calibration is shared,
with 1119 requests and eight cycles; later workloads have 16 unmeasured workload warmup requests. Four
representative workloads determine the geometric mean; the other available workload files were not included
in this optimization objective. Results are not portable defaults for other GPUs or model families.

Cache conditions require care. `--no-enable-prefix-caching --mm-processor-cache-gb 0` disables text prefix
and CPU multimodal processor caching. **It does not disable vLLM's internal encoder embedding cache.** In
vLLM 0.31.0, `v1/core/encoder_cache_manager.py` shares same-hash embeddings across requests, independently of
those two switches. These traces repeat four image files, so internal encoder reuse can benefit even this
restricted-cache configuration. TensorRT encodes images per request. The result therefore does not isolate
encoder kernel speed. Result directory/config names containing `uncached` are historical labels for prefix/
processor caching off, not a claim that every cache is disabled.

The cache-enabled variant additionally uses prefix caching and a 4 GiB CPU multimodal processor cache.
Its three confirmations each start a fresh server and run each workload once; this avoids turning repeated
full-workload passes into an all-prompt prefix-cache benchmark. Calibration and the 16-request workload warmup
still warm caches, and the measured workloads still reuse images. This variant describes that repeated-image
traffic and should not be extrapolated to unique-image traffic.

Single-run search results (selection screen, not the final medians):

| Candidate | Balanced | Decode-heavy | Mixed | Vision-heavy | Four-workload mean | GPU peak (MiB) |
|---|---:|---:|---:|---:|---:|---:|
| S512-T8192-U095-cache-enabled | 5395 | 10563 | 4432 | 3588 | 5487 | 40240 |
| S512-T8192-U095 | 5376 | 10896 | 920 | 620 | 2404 | 40364 |
| S512-T8192 | 5316 | 10653 | 919 | 618 | 2382 | 40432 |
| S512-T16384 | 5572 | 9906 | 910 | 614 | 2357 | 39124 |
| S512-T4096 | 5204 | 10389 | 898 | 609 | 2332 | 40346 |
| S128-T4096 | 4989 | 10137 | 891 | 605 | 2285 | 38484 |
| S256-T4096 | 4901 | 9822 | 897 | 609 | 2265 | 38536 |

S512/T8192/U0.95 led the restricted-cache screen, but the improvement over U0.92 was only about 1%; this
single-run search does not establish a statistically unique optimum. The selected settings are confirmed
three times below. Cache enablement changed prefix and processor caching together, so its gain cannot be
attributed to only one of them. vLLM shutdown logs sometimes report `EngineDeadError` after the harness sends
SIGINT and engine shutdown finishes; every measured request in this screen completed before shutdown.

The initial E8/P32/D256 candidate failed the larger final mixed workload: 3/512 requests completed in
repeat 1, none in repeats 2/3, and the backend exited with code -6. The mixed-cell GPU sample peaked at 40408
MiB. The runner restarted the server for the next cell and overwrote the original server log, so this final
failure cannot be conclusively attributed to CUDA OOM from the retained log; the request CSVs substantiate
the backend abort. This candidate is excluded from the final comparison despite its short-trace screen lead.
The initial final directory retains its valid text numbers and failed-cell evidence.

A `fixed_output_complete` flag alone is insufficient: the mixed summary marked it true for its three
successful requests while 509 failed. Final acceptance additionally requires `failed == 0` and
`succeeded == requests` on every repeat. Lower E/P candidates are screened on the exact final protocol,
and the surviving candidate is confirmed three times before the final table.

The E8/P16/D256 and E4/P32/D256 recovery screens also failed mixed traffic; their preserved server logs
explicitly report `CUDA runtime error in cudaMalloc(&data, memoryCapacity): out of memory`. Reducing only
P or E did not resolve the larger protocol. An E4/P16 screen was interrupted after this evidence to test
an explicit memory ceiling.

An explicit encoded-result budget needs a separate global memory contract on this binary. Setting only
`TRT_EDGELLM_MAX_ENCODED_VISION_BYTES=2147483648` stalled generic calibration with the backend CPU busy and
GPU utilization zero. Source inspection shows the encoder action counts committed KV pages in managed
bytes and falls back to `maxEncodedBytes` for its global action budget when `memoryBroker.maxManagedBytes`
is unset. The 21 GiB physical KV pool therefore cannot fit inside a 2 GiB global budget. Pairing the encoded
ceiling with `TRT_EDGELLM_PHASE_MEMORY_MAX_BYTES=25769803776` (24 GiB managed bytes, including KV) restored
progress. This is a configuration constraint inferred from that code path and the paired-setting rerun,
not a source fix. The 24 GiB figure excludes model weights and other unmanaged device allocations; it is
not the total GPU memory limit.

The paired budget screen keeps E8/P32/D256 and tests encoded ceilings 2 GiB and 2.5 GiB. The common final
traces, engine, binary, calibration and 40 GiB physical GPU remain the same. The selected budget is then
confirmed three times.

| Encoded ceiling | Balanced | Decode-heavy | Mixed | Vision-heavy | Four-workload mean | GPU peak (MiB) |
|---|---:|---:|---:|---:|---:|---:|
| 2GiB | 5457 | 10366 | 948 | 661 | 2440 | 39292 |
| 2.5GiB | 5293 | 10670 | 979 | 676 | 2473 | 39788 |

Both candidates completed all requests. The 2.5 GiB candidate led this single-run screen by about 1.4%, and
its final repeated confirmation retains that encoded ceiling plus the 24 GiB managed-byte contract. These
limits apply to this local engine/GPU/workload lineage; they are not portable automatic defaults.

Final accepted configurations: median generated tok/s of three runs; every one of the 27648 measured
requests completed at its requested output length, with zero failures. Cached vLLM uses three fresh servers.

| Workload | TRT | vLLM restricted-cache | vLLM cache-enabled | Cache-enabled vLLM/TRT |
|---|---:|---:|---:|---:|
| balanced | 5337 | 5265 | 5337 | 1.00x |
| decode-heavy | 10271 | 10661 | 10593 | 1.03x |
| mixed | 963 | 919 | 4389 | 4.56x |
| vision-heavy | 671 | 621 | 3506 | 5.22x |
| Geometric mean | 2439 | 2379 | 5431 | 2.23x |

TRT and restricted-cache vLLM are close overall (+2.5% TRT geometric mean); this small gap is not evidence
of a statistically unique winner. Native cache-enabled vLLM is 2.23x faster overall on this repeated-image
traffic, with 4.56x mixed and 5.22x vision-heavy throughput. Text throughput is similar. The optimized native
vLLM settings are S512/T8192/U0.95, chunked prefill, asynchronous scheduling, prefix caching and a 4 GiB CPU
multimodal processor cache. Neither system is quantized. The exact optimized arguments are:

```text
--dtype bfloat16 --max-model-len 8192
--max-num-seqs 512 --max-num-batched-tokens 8192
--gpu-memory-utilization 0.95 --seed 0
--enable-chunked-prefill --async-scheduling
--enable-prefix-caching --mm-processor-cache-gb 4
--limit-mm-per-prompt '{"image":2,"video":0}'
```

The restricted-cache comparison replaces the prefix switch with `--no-enable-prefix-caching` and uses
`--mm-processor-cache-gb 0`; other optimized arguments stay the same.

Latency: each entry is the median of the three per-run p95 values, not a percentile pooled across runs.
TTFT is measured from client submission, including queue time; TPOT is generation time per token after
the first token. Throughput tuning supplies no latency SLO.

| Workload | TRT TTFT p95 (ms) | TRT TPOT p95 (ms) | Restricted vLLM TTFT p95 (ms) | Restricted vLLM TPOT p95 (ms) | Cached vLLM TTFT p95 (ms) | Cached vLLM TPOT p95 (ms) |
|---|---:|---:|---:|---:|---:|---:|
| balanced | 5572 | 53.2 | 3546 | 105.5 | 3550 | 105.2 |
| decode-heavy | 6935 | 25.5 | 5511 | 53.4 | 5269 | 54.9 |
| mixed | 20775 | 215.9 | 20491 | 341.0 | 2893 | 89.1 |
| vision-heavy | 29090 | 245.2 | 30551 | 360.8 | 3987 | 45.3 |

TRT has lower TPOT than restricted-cache vLLM, but text TTFT is higher at the 512-client protocol. The
cache-enabled image workloads change both throughput and latency substantially; the restricted-cache
latencies do not describe that optimized cached setting.

| GPU peak across workloads (MiB) | TRT | Restricted vLLM | Cached vLLM |
|---|---:|---:|---:|
| Device memory sampled every 200 ms | 39824 | 40216 | 40398 |

The physical limit is 40960 MiB. These sampled peaks and three successful passes establish completion for
these retained traces; they do not guarantee allocation safety on longer contexts or unseen image geometry.

## Output quality and promotion

Nine-case greedy reference-span gate (six text, three image cases, 48-token maximum):

| Candidate | HF FP16 exact spans | HF BF16 exact spans | Image exact spans |
|---|---:|---:|---:|
| TRT sequential | 8/9 | 6/9 | 3/3 |
| TRT concurrent9 | 8/9 | 6/9 | 3/3 |
| Restricted vLLM sequential | 8/9 | 8/9 | 3/3 |
| Restricted vLLM concurrent9 | 7/9 | 9/9 | 3/3 |

TRT sequential and concurrent outputs are identical. The remaining FP16-reference difference is the
sky-color explanation wording; it is also present in the restart D64-engine gate. The batch/memory tuning
did not resolve it. vLLM sequential/concurrent responses differ on a text case as reflected above; the
nine prompts do not establish broad model accuracy for either system. This gate used the restricted-cache
vLLM configuration; the cache-enabled configuration has request/output-length validation from the
performance runs. The scoring compares only the HF
reference span because fixed-output TensorRT continues beyond EOS.

The registry records a validated throughput candidate with the paired memory budgets, but the quality
gate remains open and the default serving lineage has not been promoted on throughput evidence alone.

The three final bulk repeats had identical per-request text hashes on TensorRT for all four workloads.
vLLM restricted-cache bulk repeats had different hashes: versus repeat0, repeat1/repeat2 differed on
620/616 of 1024 balanced requests, 862/916 of 1024 decode-heavy requests, 187/194 of 512 mixed requests and
187/170 of 512 vision-heavy requests. Requests use temperature0 and fixed output ignoring EOS; these
hash differences do not by themselves establish semantic errors or identify a numerical cause. Output
length completion is therefore a separate check from exact output identity and semantic quality.


## Reproduction and retained results

Start from the environment and build recipe in note 379, with the CUDA compiler explicitly set when CMake
cannot discover it. Source `.local/env.sh` before binaries; it sets `TRT_PACKAGE_DIR`, `LLM_SDK_DIR` and
`LD_LIBRARY_PATH`. The token is read from a mode-600 local secret file and is never checked in.

Build the candidate from the packed export:

```bash
source .local/env.sh
"$BUILD_DIR/examples/llm/llm_build" \
  --onnxDir .local/artifacts/colab-a100/cosmos-reason2-2b/onnx-fp16-packed-p1024/llm \
  --engineDir .local/artifacts/colab-a100/cosmos-reason2-2b/engine-p32-d256-b256-c512-kv8192-pool1536 \
  --maxInputLen 1024 --maxKVCacheCapacity 8192 --maxBatchSize 256 \
  --maxPrefillBatchSize 32 --maxDecodeBatchSize 256 --maxPrefillChunkTokens 512 \
  --maxKVPoolPages 1536 --allowKVPoolUndercommit
```

Generate the final inputs/calibration and use the retained configuration (whose absolute artifact paths must
be adapted if reproducing elsewhere):

```bash
R=.local/results/a100-cosmos-reason2-2b-fp16/capacity-resume
"$EXPORT_VENV/bin/python" benchmarks/phase_serving/build_serving_workloads.py \
  --tokenizer .local/artifacts/models/Cosmos-Reason2-2B \
  --bulk-requests 1024 --mix-scale 8 --seed 20261005 --output-dir "$R/optimal-inputs"
"$EXPORT_VENV/bin/python" benchmarks/phase_serving/build_generic_policy_calibration_trace.py \
  --output "$R/optimal-inputs/calibration.json" \
  --image-url "file://$REPO/examples/multimodal/pics/giant_panda.jpeg" \
  --cycles 8 --max-prefill-batch 32 --max-decode-batch 512 \
  --max-encoder-batch 8 --prefill-tokens 1024
"$EXPORT_VENV/bin/python" benchmarks/phase_serving/run_serving_comparison.py \
  --config "$R/vllm-best-uncached-config.json" --systems trt vllm \
  --workloads balanced decode-heavy mixed vision-heavy --reuse-server \
  --repeats 3 --output-dir "$R/reproduction/uncached"
for i in 1 2 3; do
  "$EXPORT_VENV/bin/python" benchmarks/phase_serving/run_serving_comparison.py \
    --config "$R/vllm-cache-enabled-config.json" --systems vllm \
    --workloads balanced decode-heavy mixed vision-heavy --reuse-server \
    --repeats 1 --output-dir "$R/reproduction/cache-fresh-$i"
done
```

The final candidate also requires both byte limits:

```text
TRT_EDGELLM_ENABLE_MEASURED_DECODE_BATCHING=1
TRT_EDGELLM_MAX_INFLIGHT=256
TRT_EDGELLM_MAX_ENCODED_VISION_BYTES=2684354560
TRT_EDGELLM_PHASE_MEMORY_MAX_BYTES=25769803776
```

The other configuration fields are stable slots256, client in-flight512, prefill batch32, decode batch256,
prefill chunk512, prefill batch tokens16384, vision batch8, encoder input tokens22528 and initial capacity8.
The runner enables V3 probes, dispatch telemetry, asynchronous image preparation and independent contexts;
it primes at most64 decode graphs and disables prefill graph caching. Calibrated dispatches above64 rows
still execute; the 64 graph-cache limit is not a 64-row decode capacity limit. Final retained dispatch
telemetry observed decode batch256 and prefill batch32.

The retained configurations include the complete phase contract and vLLM arguments. The comparison runner
records server commands, filtered runtime environment and workload hashes beside each summary. A frozen
runtime and its plugin shared library are required for retained engine identities; point `EDGELLM_PLUGIN_LIB`
and `EDGELLM_PLUGIN_PATH` at that immutable artifact, not a later mutable build.

- `decode-sweep/sweep.json`: D64–D256, load scaled with D, one repeat.
- `upper-decode-sweep/sweep.json`: engine512 failure/fallback diagnostics, one repeat.
- `ep-grid/grid.json`: all nine E/P combinations, one repeat.
- `confirmation/ranking.json`: five fixed-load finalists, three repeats.
- `vllm-grid/grid.json`: settings searched and per-workload metrics.
- `final/summary.json`: final median rates, latency percentiles, peak memory and exact trace contracts.
- `final/quality/`: sequential/concurrent output files and scores against both HF precision references.
- `manifest.json` and per-campaign manifests: retention states, hashes, commands and this note reference.

The default current engine remains the restart-validated lineage; candidate links and registry evidence
protect the throughput candidate without silently promoting it across an unresolved output-quality gate.
