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

# Cosmos upstream 보수적 공정 비교

> 동일 P8/D8에서 independent pipeline만 분리한 real-request E2E 비교는
> `notes/55-cosmos-upstream-p8d8-e2e-independent-comparison-20260811.md`에 정리했다.

## 결론

현재 구현의 kernel 자체가 upstream보다 빨라진 것은 아니다. 동일 decode batch cap `D8`까지
강제하면 현재 real-request runtime은 upstream의 낙관적인 GPU-only 상한보다 처리량이 `13.9%`
낮다. 반면 raw FP16 KV 용량을 정확히 `1,792 MiB`로 맞추고 indexed-paged가 확보한 request
capacity를 사용해 `D32`를 허용하면 처리량은 `2.062x`, TTFT p95는 `54.4%`, E2E p95는
`51.7%` 개선된다.

즉 확인된 향상은 kernel speedup이 아니라 같은 KV byte에서 더 많은 request를 안정적으로
유지하고 큰 decode dynamic batch를 만드는 데서 발생한다.

| comparison | raw KV | decode cap | throughput | upstream 대비 | TTFT med/p95 | E2E med/p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| upstream optimistic ceiling | 1,792 MiB | 8 | 1,177.27 token/s | 1.000x | 6,818 / 18,297 ms | 7,400 / 19,083 ms |
| current, same D8 | 1,792 MiB | 8 | 1,013.12 token/s | 0.861x | 10,221 / 20,905 ms | 12,905 / 22,994 ms |
| current, same raw KV | 1,792 MiB | 32 | 2,427.00 token/s | 2.062x | 4,234 / 8,345 ms | 5,298 / 9,225 ms |
| current, max-throughput | 3,584 MiB | 56 | 3,605.55 token/s | 3.063x | 3,036 / 5,425 ms | 3,875 / 6,246 ms |

## 왜 upstream에 유리한 비교인가

upstream public runtime은 stable indexed slot과 continuous admission을 지원하지 않아 동일한 queue
scheduler를 그대로 연결할 수 없다. upstream 코드를 수정하면 더 이상 clean upstream 비교가
아니므로 다음과 같은 optimistic oracle을 만들었다.

1. 두 구현 모두 동일한 288개 prompt와 output-length 분포를 사용한다.
2. upstream request는 output max별로 묶어 모든 batch를 최대 BS8로 완전히 채운다.
3. upstream 시간에는 TensorRT CUDA-event로 측정한 prefill과 generation GPU time만 포함한다.
4. upstream에서 tokenizer, sampling, host dispatch, queue overhead, partial batch 비용을 제외한다.
5. latency oracle은 실제 arrival time을 보존하되, ready batch 중 GPU 시간이 가장 짧은 batch를
   먼저 실행하는 single-GPU SPT(shortest processing time) 순서를 사용한다.
6. current 값은 arrival부터 마지막 완료까지의 실제 wall-clock trace이며 queue, sampling,
   synchronization 비용을 포함한다.

따라서 upstream `1,177.27 token/s`는 production server 처리량이 아니라 실현하기 어려운
active-GPU 상한이다. latency 역시 online SPT가 미래 request를 알지는 못하지만, batch 실행
시간을 미리 알고 가장 짧은 작업을 선택하는 upstream 우대 oracle이다.

## 동일 메모리 조건

직접 공정하게 일치시킬 수 있는 항목은 raw KV allocation이다.

| item | upstream fixed-linear | current indexed-paged |
| --- | ---: | ---: |
| FP16 KV token cells | `8 x 2048 = 16,384` | `128 pages x 128 = 16,384` |
| raw KV bytes | 1,792 MiB | 1,792 MiB |
| stable request identities | 8 | 32 |
| max decode batch | 8 | 32 |
| request 종료 시 KV row compaction | fixed batch 경로 | 없음 |

동일 KV byte에서 stable request identity는 `4x`다. page pool은 sequence의 실제 길이만큼 page를
lease하므로 32개 request를 유지하면서 D32를 형성할 수 있다. 반면 fixed-linear는 각 row가
처음부터 2,048-token capacity를 독점하므로 GPU에 KV 여유 공간이 남아 있어도 아홉 번째 request를
수용할 row가 없다.

이 표는 **전체 peak VRAM이 동일하다는 뜻이 아니다**. current는 independent TensorRT execution
context, phase별 workspace와 scheduler buffer를 추가로 사용한다. upstream production peak
`7,348 MiB`에는 visual runtime도 포함되어 current text-only phase harness와 구성요소가 다르다.
따라서 total VRAM 수치를 억지로 직접 비교하지 않고, 구조적으로 동일한 raw KV byte만 맞췄다.

## 실험 1: decode batch까지 동일하게 제한

current engine은 page pool 128, stable slots 32를 사용하지만 scheduler를 `P8/D8`로 제한했다.
세 번 모두 성공했다.

| metric | current 3-run median | upstream oracle | delta |
| --- | ---: | ---: | ---: |
| throughput | 1,013.12 token/s | 1,177.27 token/s | `-13.9%` |
| TTFT median | 10,220.7 ms | 6,818.2 ms | `+49.9%` |
| TTFT p95 | 20,905.1 ms | 18,297.2 ms | `+14.3%` |
| E2E median | 12,904.7 ms | 7,400.1 ms | `+74.4%` |
| E2E p95 | 22,993.9 ms | 19,082.6 ms | `+20.5%` |

이 결과는 indexed-paged mapping과 independent context 자체가 공짜가 아님을 보여준다. current는
실제 wall-clock overhead를 지불하고도 더 큰 batch를 사용하지 못하므로 손해다. isolated 512-token
kernel 비용도 upstream과 current가 2% 이내였으므로 scheduler의 core kernel이 빨라졌다는 주장은
할 수 없다.

## 실험 2: raw KV는 같고 capacity 이점만 허용

동일 1,792 MiB raw KV에서 current의 scheduler를 `P8/D32`로 사용했다. 세 번의 throughput은
`2,422.65`, `2,427.00`, `2,427.91 token/s`였고 모든 run에서 observed D32를 형성했다.

| metric | current 3-run median | upstream oracle | delta |
| --- | ---: | ---: | ---: |
| throughput | 2,427.00 token/s | 1,177.27 token/s | `2.062x`, `+106.2%` |
| TTFT median | 4,233.9 ms | 6,818.2 ms | `-37.9%` |
| TTFT p95 | 8,344.7 ms | 18,297.2 ms | `-54.4%` |
| E2E median | 5,298.1 ms | 7,400.1 ms | `-28.4%` |
| E2E p95 | 9,224.5 ms | 19,082.6 ms | `-51.7%` |

TPOT median은 current `12.26 ms`, upstream 평균 decode-step GPU time은 `6.15 ms`다. 서로 다른
통계와 계측 범위이므로 완전한 apples-to-apples 값은 아니지만 current가 약 `2x` 느리다. aggregate
throughput과 queue tail은 개선되지만, 개별 active request의 inter-token latency는 희생한다는 뜻이다.

## 실험 3: current가 사용할 수 있는 추가 KV 허용

page pool 256은 raw KV `3,584 MiB`, 즉 upstream의 `2x`다. `P10/D56`은 세 번의 median
`3,605.55 token/s`로 upstream 상한의 `3.063x`다.

메모리 증가까지 감안한 raw-KV efficiency는 다음과 같다.

```text
(3.063x throughput) / (2x raw KV) = 1.531x throughput per raw-KV byte
```

stable slot은 upstream 8개에서 current 80개로 `10x`, KV byte당 stable-slot density는 `5x`다.
TTFT p95는 `70.3%`, E2E p95는 `67.3%` 낮아진다. 다만 TPOT median은 current `10.70 ms`로
upstream의 GPU-only decode-step 평균 `6.15 ms`보다 약 `74%` 길다.

## 메모리와 latency 해석

- upstream fixed-linear B16은 RTX 3080 10GB에서 OOM이고 B8이 실행 가능한 최대다.
- current indexed-paged는 page allocation으로 request의 미사용 tail capacity를 공유 pool로 돌린다.
- 같은 raw KV에서도 D32가 가능해 aggregate token throughput과 queue TTFT/E2E가 개선된다.
- 큰 decode batch는 한 step의 GPU 시간을 늘리므로 active request TPOT은 나빠질 수 있다.
- current 내부의 indexed-linear 대 indexed-paged peak 비교는 `9,294 -> 6,828 MiB`, `-26.5%`다.
  이는 current 두 KV 방식 사이의 공정한 비교이며 upstream peak와 직접 혼합하면 안 된다.
- 최대-throughput 구성은 raw KV를 2배 사용하지만 throughput을 3.063배 올려 KV byte당 처리량도
  1.531배 개선한다.

production policy는 단순히 항상 D56을 선택하면 안 된다. queue가 얕거나 TPOT SLO가 엄격하면
D8~D24를 선택하고, queue pressure와 request age가 증가할 때 D32~D56으로 승격하는 방식이 맞다.

## 재현 자료

```text
.local/upstream-baseline/cosmos-reason2-2b/real-trace-bs8/
.local/cosmos-reason2-2b/upstream-fair-loss-comparison/
  page128-repeat1/
  page128-repeat2/
  page128-repeat3/
  same-batch-repeat1/
  same-batch-repeat2/
  same-batch-repeat3/
  summary.csv
.local/cosmos-reason2-2b/upstream-max-speedup-search/repeat-summary.csv
```

검증한 engine은 다음과 같다.

```text
same raw KV:
  .local/cosmos-reason2-2b/sweep-engines/engine-fp16-paged-p8-d32-b128
maximum throughput:
  .local/cosmos-reason2-2b/sweep-engines/engine-fp16-paged-p16-d64-b256-m80
```

원자료의 18개 same-KV sweep case와 3개 same-D8 case는 모두 return code 0이었다. 최대-throughput
finalist도 36/36 성공했다. 집계 CSV는 metric basis를 함께 기록해 GPU-only oracle과 real wall-clock
값을 구분한다.
