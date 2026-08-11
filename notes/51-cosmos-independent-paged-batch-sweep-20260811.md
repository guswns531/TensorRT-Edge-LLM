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

# Cosmos independent indexed-paged batch/page-pool sweep

## 실험 범위

이번 sweep은 CUDA primary context 하나와 독립 TensorRT prefill/decode execution context를
사용하는 `independent` 경로만 측정한다. `shared` context는 이번 단계의 성능 지표에서 제외한다.

| 축 | 값 |
| --- | --- |
| model | `nvidia/Cosmos-Reason2-2B`, FP16 text decoder |
| prefill cap | 1, 2, 4, 8 |
| decode cap | 1, 2, 4, 8, 16, 24, 32 |
| page pool | 80, 128, 160, 256 bundles |
| stable slots | 16, 24, 32 |
| page size | 128 tokens |
| KV capacity | 2048 tokens/request |
| chunk | fixed 128 tokens |
| workload | real JSON trace, Poisson 120 req/s, seed 7, 48 requests |

실제 phase batch가 cap에 도달하는지 확인하기 위해 request-level CSV와 dispatch-level CSV를 함께
보존한다. `--slotCount`는 decode cap 이상이어야 한다. 따라서 slot16에서는 decode 1/2/4/8/16,
slot24에서는 decode 1/2/4/8/16/24, slot32에서는 전체 decode 조합을 실행한다.

## 엔진

`maxBatchSize=32`, `maxPrefillBatchSize=8`, `maxDecodeBatchSize=32`로 page pool별 엔진을
build했다. engine weight와 TensorRT profile은 동일하고, `kv_cache_page_bundles`만 다르다.

```text
.local/cosmos-reason2-2b/sweep-engines/
  engine-fp16-paged-p8-d32-b80
  engine-fp16-paged-p8-d32-b128
  engine-fp16-paged-p8-d32-b160
  engine-fp16-paged-p8-d32-b256
```

대표 build command:

```bash
build/examples/llm/llm_build \
  --onnxDir .local/cosmos-reason2-2b/onnx-fp16-paged/llm \
  --engineDir .local/cosmos-reason2-2b/sweep-engines/engine-fp16-paged-p8-d32-b256 \
  --maxBatchSize 32 --maxPrefillBatchSize 8 --maxDecodeBatchSize 32 \
  --maxInputLen 1024 --maxKVCacheCapacity 2048 --kvCachePageBundles 256
```

## 현재 실행 결과

### slot dimension

- `page80/slot16`: decode 24/32는 실행 전 `Physical slot count is smaller than a phase batch limit`로
  거부됐다. decode cap을 키우려면 stable slot도 함께 키워야 한다.
- `page80/slot24`: prefill 1/2/4/8 × decode 8/16/24, 12/12 성공.
- `page80/slot32`: prefill 1/2/4/8 × decode 8/16/24/32, 15/16 성공.
  `p8/d24`는 실제 고부하에서 `KV page-bundle pool is exhausted`로 실패했다.

이는 page pool exhaustion이 CUDA OOM과 별개의 admission 경계로 동작한다는 것을 확인한다.

### page-pool dimension

slot32에서 high-decode slice(각 12 cases)는 다음과 같이 실행됐다.

| page pool | 성공 cases | p8/d16 observed max decode | p8/d24 observed max decode | p8/d32 observed max decode |
| ---: | ---: | ---: | ---: | ---: |
| 80 | 15/16 전체 sweep | 16 | 20 | 실패/20 |
| 128 | 12/12 | 16 | 22 | 22 |
| 160 | 12/12 | 16 | 20 | 20 |
| 256 | 12/12 | 16 | 20 | 20 |

여기서 `decode cap`은 상한이며 실제 observed batch와 다르다. arrival trace와 page admission이
실제 batch 형성을 제한한다. 따라서 cap만 보고 성능을 해석하지 않고 `requests-dispatch.csv`의
observed batch 분포를 기본 지표로 사용한다.

대표 p8 결과의 generated token/s는 다음과 같다.

| page pool | d16 | d24 | d32 |
| ---: | ---: | ---: | ---: |
| 80, slot32 | 768.6 | - | 771.7 |
| 128, slot32 | 808.8 | 815.8 | 815.6 |
| 160, slot32 | 792.9 | 795.8 | 795.7 |
| 256, slot32 | 773.1 | 778.4 | 776.1 |

이 값은 각 조합 1회 실행의 초기 cost table이며 최종 결론이 아니다. pool 크기를 키웠을 때
성공률은 올라갔지만, 128 이상에서 throughput이 단조 증가하지 않는다. 다음 단계에서 각 case를
3회 이상 반복하고 GPU clock/temperature와 peak VRAM을 기록해야 한다.

## 재현 명령

예를 들어 slot32/page256 high-decode slice는 다음과 같다.

```bash
python3 scripts/cosmos_reason2/run_real_request_kv_matrix.py \
  --bench build/examples/llm/llm_phase_bench \
  --source-trace notes/results/gemma4-llm-real-request-poisson-seed1-20260804/fixed128_p2d2_30rps/trace.json \
  --output-dir .local/cosmos-reason2-2b/sweep-results/saturation-b256-s32 \
  --engine paged=.local/cosmos-reason2-2b/sweep-engines/engine-fp16-paged-p8-d32-b256 \
  --prefill-batches 1 2 4 8 --decode-batches 16 24 32 \
  --context-modes independent --arrival-rate 120 --seed 7 --repeat-count 4 \
  --slot-count 32 --page-bundles 256 --tokens-per-page 128 --continue-on-error
```

원자료는 다음 디렉토리에 있다.

```text
.local/cosmos-reason2-2b/sweep-results/
  p8d32-b80-s16/
  saturation-b80-s24/
  saturation-b80-s32/
  saturation-b128-s32/
  saturation-b160-s32/
  saturation-b256-s32/
```

각 결과에는 `status.csv`, `request-summary.csv`, `dispatch-cost-table.csv`,
`kernel-cost-table.csv`, `page-pressure-model.csv`가 있다.

## 긴 output saturation과 latency

실제 decode BS24/32를 만들기 위해 source trace를 8회 반복해 96 request로 만들고, request별
`max_generate_length`를 2배로 늘렸다. arrival rate는 240 req/s로 설정했다.

```bash
--repeat-count 8 --output-multiplier 2.0 --arrival-rate 240
```

새 `--output-multiplier`는 원래 trace의 request별 길이 분포를 유지하면서 output 상한만 배율로
늘린다. `page128/slot32`와 `page256/slot32` 모두 p1/p4/p8 × d16/d24/d32의 9/9 case가
성공했고, p8 case에서 requested decode cap과 observed decode batch가 모두 일치했다.

| page pool | p8/d16 observed | p8/d24 observed | p8/d32 observed |
| ---: | ---: | ---: | ---: |
| 128 | BS16 | BS24 | BS32 |
| 256 | BS16 | BS24 | BS32 |

request-level latency는 다음과 같다. 각 값은 `p8` case의 median/p95다.

| page pool / decode | TTFT (ms) | TPOT (ms) | E2E (ms) | generated tok/s |
| --- | ---: | ---: | ---: |
| 128 / d16 | 1,113 / 2,052 | 17.4 / 24.1 | 1,999 / 2,680 | 1,328 |
| 128 / d24 | 1,114 / 1,833 | 14.4 / 18.3 | 1,807 / 2,331 | 1,481 |
| 128 / d32 | 1,113 / 1,768 | 12.8 / 16.2 | 1,724 / 2,253 | 1,523 |
| 256 / d16 | 1,112 / 2,055 | 17.5 / 24.2 | 2,000 / 2,684 | 1,327 |
| 256 / d24 | 1,109 / 1,829 | 14.4 / 18.3 | 1,802 / 2,327 | 1,484 |
| 256 / d32 | 1,112 / 1,766 | 12.8 / 16.1 | 1,723 / 2,251 | 1,523 |

decode batch를 16→24→32로 키우면 TPOT median은 약 17.5→14.4→12.8ms로 감소하고,
TTFT/E2E tail도 이 workload에서는 함께 줄었다. 이는 decode queue가 충분히 포화되어 batch
효율이 latency를 지배한 결과다. 다만 arrival rate가 낮은 workload에서는 큰 decode batch를
기다리는 시간이 TTFT를 다시 악화시킬 수 있으므로 이 표를 고정 정책으로 사용하지 않는다.

page128의 modelled peak pressure는 `0.453`, page256은 `0.227`이었고 두 경우 모두 page
exhaustion은 발생하지 않았다. 관측된 admission pending queue는 약 61~62 request까지
늘었으며, 이 긴 trace의 latency tail은 page pool보다 stable-slot/phase queue 대기 영향이
더 컸다. Kernel-group median은 page128/page256에서 prefill engine `17.47/17.48ms`,
decode engine `6.72/6.71ms`로 거의 동일했다.

원자료는 다음에 있다.

```text
.local/cosmos-reason2-2b/sweep-results/
  long-saturation-b128-s32/
  long-saturation-b256-s32/
```

## 다음 측정

1. 위 성공 case를 warmup 20회와 측정 100회, process 3회로 반복한다.
2. page80/128/160/256 × slot16/24/32의 교차 조합은 slot limit을 넘는 decode case를 자동으로
   제외하되, 동일한 observed batch 조건을 맞춰 비교한다.
3. peak VRAM headroom 512 MiB를 gate로 사용하고, page exhaustion은 failure가 아니라 admission
   backpressure latency로 기록한다.
4. page80에서도 동일한 긴 trace를 실행해 page exhaustion이 발생하는 시점의 TTFT/E2E tail과
   pending admission latency를 별도로 측정한다.
