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

동일한 긴 trace를 page80에도 적용했다. p8 case는 d16과 d32에서 성공했지만 d24는 page pool
exhaustion으로 종료됐다.

| page80 / p8 | TTFT (ms) | TPOT (ms) | E2E (ms) | generated tok/s |
| --- | ---: | ---: | ---: | ---: |
| d16 | 1,157 / 2,179 | 18.3 / 25.8 | 2,094 / 2,823 | 1,270 |
| d24 | admission failure | admission failure | admission failure | - |
| d32 | 1,157 / 1,872 | 13.5 / 17.3 | 1,802 / 2,368 | 1,460 |

page80의 d24 실패와 d32 성공처럼 cap이 커질수록 항상 실패하는 단조 관계는 아직 관측되지
않았다. request arrival과 phase admission 시점에 따라 page lifetime이 달라지기 때문이다.
따라서 page exhaustion은 단순 pass/fail만 보지 않고 allocator snapshot, pending admission
시간, retry 횟수를 함께 기록해야 하며, 동일 case를 반복해 deterministic backpressure인지
확인해야 한다.

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
4. page80의 d24 exhaustion을 동일 seed로 3회 이상 반복해 allocator lifetime과 pending admission
   latency가 deterministic한지 확인한다.

## Prefill 16 / Decode 64 후보 측정

prefill과 decode 상한을 함께 확장한 별도 엔진도 build했다.

```text
maxBatchSize=64
maxPrefillBatchSize=16
maxDecodeBatchSize=64
page pool=128 또는 256
stable slots=64
```

P8/P12/P16 × D32/D48/D64를 96-request trace에서 실행했다. page128에서는 일부 P12/P16
조합이 page exhaustion으로 중단됐고, page256에서는 9/9 case가 성공했다. 그러나 arrival
240 req/s에서는 D64 cap도 실제 observed BS46까지밖에 형성되지 않았다.

page256, 144-request trace에서 observed batch와 request latency는 다음과 같다.

| prefill / decode cap | observed P/D | TTFT med/p95 (ms) | TPOT med/p95 (ms) | E2E med/p95 (ms) | tok/s |
| --- | --- | ---: | ---: | ---: | ---: |
| P8 / D32 | 7 / 32 | 1,252 / 2,131 | 11.0 / 14.5 | 1,695 / 2,553 | 2,040 |
| P8 / D48 | 8 / 46 | 1,217 / 2,113 | 9.8 / 12.3 | 1,637 / 2,513 | 1,994 |
| P8 / D64 | 8 / 46 | 1,216 / 2,110 | 9.7 / 12.3 | 1,636 / 2,509 | 1,996 |
| P12 / D32 | 7 / 32 | 1,250 / 2,127 | 11.0 / 14.6 | 1,694 / 2,548 | 2,043 |
| P12 / D48 | 8 / 46 | 1,213 / 2,113 | 9.8 / 12.2 | 1,634 / 2,513 | 1,993 |
| P12 / D64 | 8 / 46 | 1,222 / 2,121 | 9.8 / 12.3 | 1,643 / 2,521 | 1,988 |
| P16 / D32 | 7 / 32 | 1,245 / 2,129 | 11.0 / 14.2 | 1,692 / 2,552 | 2,041 |
| P16 / D48 | 8 / 46 | 1,215 / 2,116 | 9.8 / 12.3 | 1,637 / 2,515 | 1,991 |
| P16 / D64 | 8 / 46 | 1,214 / 2,116 | 9.8 / 12.3 | 1,636 / 2,515 | 1,992 |

표의 첫 번째 행은 열 수가 밀리지 않도록 `TTFT`, `TPOT`, `E2E`가 각각 median/p95 순서다.
실행 결과는 P8/P12/P16 간 차이가 0.2% 안팎으로 작았고, D48→D64도 throughput 개선이
거의 없었다. 반면 D32→D48은 TPOT을 약 11% 줄였다. 따라서 RTX 3080 기준 기본 후보는
`P8/D32`, TPOT 목표가 더 엄격한 경우 `P8/D48`로 두고, P16/D64는 page budget과 queue
saturation을 더 확인한 뒤 선택한다.

P16/D64 builder의 activation memory는 prefill 약 1.14 GiB, decode 약 0.54 GiB로 P8/D32의
약 0.54/0.26 GiB보다 크다. page128 pressure는 `0.828~0.859`, page256은 `0.434`였으므로
P16/D64에는 page256이 더 안전하다. 이 측정 시점에는 실제 BS64가 형성되지 않았으므로,
다음 절의 더 높은 arrival rate와 request 수를 사용한 전용 queue-saturation case를 추가했다.

## Max batch 80에서 실제 Decode BS64 포화 측정

Independent 실행 중 prefill과 decode가 서로 다른 physical slot 집합을 동시에 사용하므로,
`P16 + D64`를 완전히 허용하려면 `maxBatchSize=64`보다 큰 slot capacity가 필요하다. 이를 확인하기
위해 다음 capability로 indexed-paged 엔진을 다시 build했다.

```text
maxBatchSize=80
maxPrefillBatchSize=16
maxDecodeBatchSize=64
page pool=256
stable slots=80
prefill/decode TensorRT execution context=independent
CUDA primary context=shared
```

실제 request workload는 288 requests, arrival rate 1,000 req/s, output multiplier 4, fixed 128-token
prefill chunk, queue-default scheduler로 구성했다. 모든 요청은 text-only이고 output 상한은 최대
128 tokens다. `P8/P16 x D32/D48/D64` 여섯 조합이 모두 성공했으며, D64 case에서는 요청한
decode batch 64가 실제로 형성됐다.

| prefill / decode cap | observed P/D | TTFT med/p95 (ms) | TPOT med/p95 (ms) | E2E med/p95 (ms) | tok/s |
| --- | --- | ---: | ---: | ---: | ---: |
| P8 / D32 | 8 / 32 | 3,240 / 6,776 | 21.7 / 29.0 | 4,698 / 8,051 | 2,828 |
| P8 / D48 | 8 / 48 | 3,020 / 5,527 | 11.1 / 13.0 | 3,853 / 6,292 | 3,603 |
| P8 / D64 | 8 / 64 | 3,000 / 5,450 | 10.7 / 11.6 | 3,857 / 6,189 | 3,602 |
| P16 / D32 | 9 / 32 | 3,098 / 6,832 | 23.4 / 33.5 | 5,474 / 8,227 | 2,776 |
| P16 / D48 | 9 / 48 | 3,022 / 5,308 | 11.0 / 14.1 | 3,744 / 6,204 | 3,646 |
| P16 / D64 | 9 / 64 | 2,928 / 5,465 | 10.7 / 12.6 | 3,756 / 6,302 | 3,578 |

이 포화 workload에서는 `P16/D48`이 `3,646 tok/s`로 가장 높은 처리량과 가장 낮은 E2E
median을 보였다. `P8/D48` 대비 처리량 차이는 약 `+1.18%`로 작고 TPOT p95는 오히려
약 `+8.1%`였으므로, 이 한 번의 trace만으로 항상 P16을 고정하는 근거로 사용하지 않는다.
P16에서 D48을 D64로 늘리면 TPOT median/p95는 개선됐지만 처리량은 약 `-1.85%`였고 E2E
median도 개선되지 않았다. P8에서도 D48과 D64의 처리량 차이는 `0.05%` 이내였다.

`observed P=9`는 엔진이 P9까지만 지원한다는 의미가 아니다. `maxPrefillBatchSize=16`은 상한이고,
현재 scheduler는 queue의 seed request와 `dispatchedPrefillTokens`가 같으며 initial/continuation
상태가 같은 row만 한 batch로 묶는다. 실제 P9 dispatch에는 `9 x 128`, `9 x 96`, `9 x 117`,
`9 x 21` token bucket이 각각 존재했다. 따라서 전체 prefill queue가 커도 dispatch 시점에 같은
bucket에 있던 row가 최대 9개였다. 현재 정책은 가장 큰 bucket을 찾지 않고 queue seed의 bucket을
사용하며, decode burst 사이에 prefill을 즉시 dispatch하므로 P16까지 기다리지 않는다.

### 현재 권장 버전

엔진 capability와 runtime scheduler 기본값을 분리한다.

```text
Engine capability:
  independent TensorRT execution contexts
  indexed-paged KV cache
  maxBatchSize=80
  maxPrefillBatchSize=16
  maxDecodeBatchSize=64
  pagePool=256
  stableSlots=80

Runtime default:
  fixed prefill chunk=128
  prefill cap=8
  decode cap=48

Saturation policy candidate:
  compatible prefill bucket이 충분하면 P16/D48
  TPOT pressure가 우선이고 slot/page headroom이 충분하면 일시적으로 D64
```

따라서 현재 종합 권장안은 P16/D64-capable independent indexed-paged 엔진을 유지하면서,
보수적인 운영 기본값은 `P8/D48`, 충분히 포화된 queue에서는 `P16/D48`을 선택하는 것이다.
D64는 실제 형성 가능함을 검증했지만 처리량 기본값이 아니라 TPOT tail을 줄이는 reserve cap으로
사용한다. 다음 scheduler 개선은 compatible bucket 크기, queue wait, SLO pressure를 함께 보고
P8/P16과 D48/D64를 선택해야 한다.

원자료는 다음에 있다.

```text
.local/cosmos-reason2-2b/sweep-results/stress288-p16d64-b256-m80-s80/
.local/cosmos-reason2-2b/sweep-engines/engine-fp16-paged-p16-d64-b256-m80/
```
