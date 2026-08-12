SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

# Cosmos upstream-main baseline comparison

## 결론

NVIDIA public `main`을 `.local/upstream-main`에 detached worktree로 고정하고 Cosmos Reason2-2B
FP16의 `export -> LLM/visual build -> greedy inference -> phase cost`를 다시 수행했다.

- upstream commit: `7f061f21f0a581ba234a1e233c9315b89d8e47d6` (`v0.9.1`)
- current commit: `0882f5f5e0292612df3a2e8eb0704fa20af754c5`
- current는 upstream보다 48 commits, 30,064 insertions 앞선 상태다.
- upstream fixed-linear B16은 RTX 3080 10GB에서 OOM이고 B8은 peak `7,348 MiB`로 실행됐다.
- 512-token 고립 비용은 current indexed-linear/indexed-paged가 upstream fixed-linear와 대체로
  `2%` 이내다. 지금까지의 큰 service-level 개선은 attention kernel 자체의 가속이 아니라 KV
  ownership/page pool, 비대칭 batch, 독립 TensorRT context와 queue overlap에서 발생한다.

재현 manifest는
`.local/upstream-baseline/cosmos-reason2-2b/baseline-manifest.json`, 원시 CSV는 같은 디렉터리의
`bench-*` 아래에 있다.

## 고정 환경과 디렉터리

| item | value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 3080, 10,240 MiB |
| driver | 610.43.02 |
| TensorRT/CUDA | TensorRT 11.0.0, CUDA 13.3 |
| runtime image | `nvcr.io/nvidia/tensorrt@sha256:7cd94ee...ceac3` |
| export image | `nvcr.io/nvidia/pytorch@sha256:1dc787...cff8` |
| model | `nvidia/Cosmos-Reason2-2B`, HF revision `9ce19a1...` |
| model weights SHA256 | `fa5a6e6e...e6ede` |

```text
.local/
  upstream-main/                         # clean public main worktree
  upstream-build/                        # clean public main C++ build
  upstream-baseline/cosmos-reason2-2b/
    baseline-manifest.json
    onnx-fp16/                            # thinker + visual export
    engine-fp16-fixed-b8-i1024-kv2048/   # runnable upstream bundle
    llm-basic-output.json
    llm-basic-profile.csv
    bench-upstream-fixed/
    bench-current-indexed/
    bench-current-paged/
```

`.local`은 git-ignored이므로 upstream checkout과 약 9.2GiB의 ONNX/engine은 현재 branch의 tracked
상태를 오염시키지 않는다. upstream source와 current source는 같은 HF checkpoint를 공유하지만
ONNX, engine, plugin, 실행 바이너리는 섞지 않았다.

## Upstream pipeline

upstream source는 다음과 같이 고정했다.

```bash
git fetch https://github.com/NVIDIA/TensorRT-Edge-LLM.git main
git worktree add --detach .local/upstream-main 7f061f21f0a581ba234a1e233c9315b89d8e47d6
git -C .local/upstream-main submodule update --init
```

PyTorch 25.12 container에서 upstream Python package를 사용해 양자화 없이 두 component를
export했다.

```bash
python -m tensorrt_edgellm.scripts.export \
  .local/cosmos-reason2-2b/hf \
  .local/upstream-baseline/cosmos-reason2-2b/onnx-fp16 \
  --dtype float16 --components thinker

python -m tensorrt_edgellm.scripts.export \
  .local/cosmos-reason2-2b/hf \
  .local/upstream-baseline/cosmos-reason2-2b/onnx-fp16 \
  --dtype float16 --components visual
```

upstream C++는 TRT 26.06 container에서 `TRT_PACKAGE_DIR=/opt/tensorrt`, CUDA arch 86으로
빌드했다. LLM engine의 최종 실행 가능 설정은 다음과 같다.

```bash
.local/upstream-build/examples/llm/llm_build \
  --onnxDir .local/upstream-baseline/cosmos-reason2-2b/onnx-fp16/llm \
  --engineDir .local/upstream-baseline/cosmos-reason2-2b/engine-fp16-fixed-b8-i1024-kv2048 \
  --maxBatchSize 8 --maxInputLen 1024 --maxKVCacheCapacity 2048

.local/upstream-build/examples/multimodal/visual_build \
  --onnxDir .local/upstream-baseline/cosmos-reason2-2b/onnx-fp16/visual \
  --engineDir .local/upstream-baseline/cosmos-reason2-2b/engine-fp16-fixed-b8-i1024-kv2048
```

Cosmos config는 `num_deepstack_features=3`이므로 public runtime은 text-only request에도
`--multimodalEngineDir`를 요구한다. visual engine 없이 시작하면 초기화가 거부되며, visual engine을
같이 전달한 B8 `llm_basic` greedy inference는 성공했다.

| upstream greedy result | value |
| --- | ---: |
| prompt tokens | 17 |
| prefill GPU time | 14.4948 ms |
| generated tokens | 128 |
| decode average | 6.03 ms/token |
| peak GPU memory | 7,348 MiB |

artifact SHA256는 manifest에 모두 기록했다. LLM engine은 약 3.45GB, FP16 embedding은 622MB,
visual engine은 약 827MB다.

## B16 OOM과 KV capacity

동일 upstream engine을 B16으로 먼저 build했지만 실제 runtime은 LLM context와 visual context를
적재한 뒤 `cudaMalloc` OOM으로 종료됐다. 실패 직전 TensorRT-managed allocation은 약 4,063MiB였고,
그 뒤 fixed KV/runtime tensor가 추가된다.

Cosmos의 FP16 KV 한 token은 다음과 같다.

```text
28 layers * 2(K,V) * 8 KV heads * 128 head dim * 2 bytes = 114,688 bytes = 112 KiB
```

| configuration | addressable requests | physical token cells | raw FP16 KV |
| --- | ---: | ---: | ---: |
| upstream fixed-linear B8 | 8 | 8 * 2048 = 16,384 | 1,792 MiB |
| upstream/current fixed-linear B16 | 16 | 16 * 2048 = 32,768 | 3,584 MiB |
| current indexed-paged pool80 | stable slots 16 | 80 * 128 = 10,240 | 1,120 MiB |
| current indexed-paged pool256/maxBatch80 | stable slots 80 | 256 * 128 = 32,768 | 3,584 MiB |

중요한 차이는 maxBatch80 paged가 `80 * 2048`을 예약하지 않는다는 점이다. fixed B16과 같은
3,584MiB raw KV budget으로 최대 80개의 stable request identity를 표현하고, 실제 살아 있는 token에
page bundle을 붙인다. page pool exhaustion은 CUDA OOM 대신 admission backpressure로 처리된다.

기존 end-to-end 측정에서는 indexed-linear `9,294MiB`에서 pool80 indexed-paged `6,828MiB`로
`2,466MiB` 감소했다. upstream B8 peak와 직접 비교하면 paged가 `520MiB` 낮지만, 이 둘은
max batch와 visual runner 적재 정책이 달라 절대값만으로 같은-capacity 절감률을 주장하지 않는다.

## 512-token isolated phase cost

GPU idle 상태에서 warmup 20회, 측정 100회, CUDA graph 비활성으로 측정했다.

- upstream: clean `llm_bench`, fixed-linear, 한 TensorRT context
- current indexed-linear: `llm_phase_bench` sequential sample, stable slot mapping
- current indexed-paged: `llm_phase_bench` sequential sample, page table mapping
- input length: 512, chunk disabled(한 번의 512-token prefill)
- decode past KV: upstream/indexed-linear 512

upstream `llm_bench`는 원시 sample을 저장하지 않아 upstream 열은 100회 CUDA-event mean이다.
current 열은 `mean / median / p95`다. 따라서 delta는 mean 대 mean으로 계산했다.

### Prefill

| BS | upstream fixed mean | current indexed-linear mean/med/p95 | mean delta | current indexed-paged mean/med/p95 | mean delta |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 21.8803 | 21.9554 / 21.9325 / 22.2173 | +0.34% | 21.8939 / 21.8839 / 22.1887 | +0.06% |
| 2 | 35.5879 | 35.1853 / 35.2353 / 35.6054 | -1.13% | 35.5204 / 35.5523 / 35.9046 | -0.19% |
| 4 | 69.1995 | 68.7834 / 68.7452 / 69.1684 | -0.60% | 69.2777 / 69.2588 / 69.7177 | +0.11% |
| 8 | 134.7067 | 134.5383 / 134.5295 / 135.0991 | -0.13% | 134.9469 / 135.0293 / 135.3587 | +0.18% |

indexed-paged의 indexed-linear 대비 prefill mean 차이는 BS1/2/4/8에서 각각
`-0.28%/+0.95%/+0.72%/+0.30%`다. 모두 기존 3% gate 안이다.

### Decode

| BS | upstream fixed mean | current indexed-linear mean/med/p95 | mean delta |
| ---: | ---: | ---: | ---: |
| 1 | 6.3757 | 6.3538 / 6.3539 / 6.3877 | -0.34% |
| 2 | 6.3835 | 6.4520 / 6.4502 / 6.4841 | +1.07% |
| 4 | 6.5241 | 6.6056 / 6.6028 / 6.6427 | +1.25% |
| 8 | 6.8514 | 6.9709 / 6.9673 / 7.0021 | +1.74% |

paged v1은 arbitrary `pastKVLen=512`를 이미 존재하는 system-prompt cache로 주입하는 synthetic
benchmark를 거부한다. 그래서 paged decode를 이 표에 억지로 넣지 않았다. 동일 real-request
trajectory에서 측정한 별도 20-case matrix는 indexed-linear 대비 indexed-paged decode engine
median이 shared `6.307 -> 6.348ms`, independent `6.310 -> 6.341ms`로 모두 1% 이내였다.

## Independent context overlap

current indexed-linear에서 같은 P/D batch를 sequential과 independent TensorRT context concurrent로
실행한 결과다. CUDA primary context는 하나를 공유한다.

| P/D BS | sequential makespan median | concurrent makespan median | speedup |
| ---: | ---: | ---: | ---: |
| 1/1 | 28.2931 | 24.2761 | 1.165x |
| 2/2 | 41.6860 | 38.7343 | 1.076x |
| 4/4 | 75.3490 | 72.2611 | 1.043x |
| 8/8 | 141.5061 | 138.0541 | 1.025x |

큰 prefill batch에서는 한 prefill이 SM을 대부분 점유하므로 동시 decode가 느려지고 overlap 이득이
작아진다. 실제 scheduler의 기본값을 “항상 최대 prefill batch”로 두면 안 되는 근거다. 이전
fixed-128 real-request 결과에서 independent가 shared(serialized) 대비 TTFT 약 39~40%, TPOT/E2E
약 25% 개선된 것은 작은 chunk 사이에 decode를 자주 삽입하고 queue batching을 유지했기 때문이다.

## 해석과 다음 비교 원칙

1. upstream fixed-linear은 correctness/per-kernel 기준선으로 유지한다.
2. current indexed-linear은 stable ownership/compaction 제거의 단독 비용을 분리한다.
3. current indexed-paged는 memory/admission 효과를 분리하며 synthetic past-KV 대신 실제 lifecycle
   trace 또는 allocator가 page를 실제로 채우는 benchmark를 사용한다.
4. independent-context 성능은 sequential 고립 비용과 concurrent makespan을 항상 함께 기록한다.
5. service-level 비교는 같은 JSON arrival trace, output distribution, fixed-128 chunk, observed batch로
   해석하고 requested cap만으로 speedup을 주장하지 않는다.

현재 포화 trace의 best single-run throughput은 maxBatch80/page256에서 `P16/D48`의
`3,646 tok/s`였지만, `P8/D48`보다 약 1.18%뿐이고 tail은 더 나빴다. 따라서 현재 기본 후보는
낮은/중간 load에서 `P8/D32`, decode 포화 시 `P8/D48`, P16은 queue/cost-table 기반 선택으로
남기는 것이 맞다. 세부 결과는 `notes/51-cosmos-independent-paged-batch-sweep-20260811.md`를 따른다.

후속 3-run 최대 향상 탐색에서는 saturation throughput 후보가 `P10/D56 = 3,605.55 token/s`로
수렴했다. clean upstream B8의 optimistic active-GPU ceiling `1,177.27 token/s` 대비 `3.063x`다.
비교 방법과 전체 finalist 표는 `notes/53-cosmos-upstream-max-speedup-search-20260811.md`에 있다.
