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

# Cosmos upstream P8/D8 E2E independent 비교

> 30/200/1,000 req/s에서 P8/D8과 P8/D32를 shared/independent로 분리한 후속 결과는
> `notes/56-cosmos-independent-load-p8d8-p8d32-sweep-20260811.md`에 정리했다.

## 결론

동일 raw KV `1,792 MiB`와 동일 scheduler cap `P8/D8`에서 independent prefill/decode pipeline은
효과가 있다. current shared-serialized 대비 처리량은 `3.10%` 증가하고 trace makespan과 request
latency는 약 `3%` 감소했다. 그러나 clean upstream의 full-BS8 E2E oracle보다 처리량은 여전히
`13.3%` 낮다.

| implementation | throughput | makespan | TTFT med/p95 | TPOT med/p95 | E2E med/p95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| upstream fixed-linear P8/D8 | 1,169.85 token/s | 20,269 ms | 6,858 / 18,374 ms | 6.19 / 6.27 ms | 7,443 / 19,163 ms |
| current shared P8/D8 | 984.25 token/s | 24,530 ms | 10,562 / 21,565 ms | 38.23 / 39.19 ms | 13,329 / 23,693 ms |
| current independent P8/D8 | 1,014.74 token/s | 23,793 ms | 10,201 / 20,870 ms | 36.95 / 37.89 ms | 12,879 / 22,956 ms |

current independent와 shared는 같은 engine, 같은 indexed-paged cache, 같은 288-request
materialized trace, 같은 P8/D8 cap을 사용한다. 달라지는 것은 TensorRT execution context와 phase
stream을 독립시켜 concurrent enqueue를 허용하는지 여부뿐이다.

## Independent pipeline 자체의 효과

| metric | independent 대 shared |
| --- | ---: |
| generated throughput | `+3.10%` |
| trace makespan | `-3.01%` |
| TTFT median / p95 | `-3.41%` / `-3.23%` |
| TPOT median / p95 | `-3.36%` / `-3.30%` |
| E2E median / p95 | `-3.37%` / `-3.11%` |

동시 prefill+decode dispatch 519개를 세 run에서 합쳐 보면 다음과 같다.

| joint dispatch metric | shared | independent | change |
| --- | ---: | ---: | ---: |
| CUDA-event makespan median | 21.48 ms | 17.34 ms | `-19.3%` |
| prefill interval median | 14.78 ms | 17.33 ms | `+17.3%` |
| decode interval median | 6.71 ms | 9.55 ms | `+42.2%` |
| overlap ratio median | 0 | 0.355 | concurrent |

두 kernel group은 SM과 memory bandwidth를 경쟁하므로 각각은 느려진다. 그럼에도 두 interval을
겹쳐 joint dispatch makespan은 19.3% 감소한다. 전체 trace 개선이 3%에 그치는 이유는 joint
dispatch가 각 run `173 / 3,153`, 약 `5.49%`뿐이기 때문이다.

또한 requested cap은 P8이지만 real trace의 observed prefill batch는 최대 P3이었다. 32개 stable
slot 대부분이 long-lived decode request로 차 있고 종료로 반환된 slot만 새 prefill admission에
사용되기 때문이다. decode는 실제 D8을 계속 형성했다. 따라서 이 결과는 forced P8+D8 kernel
microbenchmark가 아니라 production-like queue trajectory에서의 효과다.

## Clean upstream E2E 기준

upstream public fixed-linear runtime에는 continuous admission과 independent prefill/decode context가
없다. clean upstream을 수정하지 않기 위해 다음 방식으로 E2E service loop를 구성했다.

1. current와 동일한 288개 request와 arrival offset을 사용한다.
2. output max 32/48/64/96/128별로 request를 묶어 모든 upstream batch를 BS8로 완전히 채운다.
3. unmodified upstream `llm_inference`에서 각 BS8 batch의 실제 wall completion을 측정한다.
4. engine/tokenizer 초기화와 warmup은 service-loop E2E에서 제외한다.
5. 측정된 batch wall cost를 original arrival 위에 replay한다.
6. ready batch 중 실제 wall cost가 가장 짧은 작업을 먼저 실행하는 clairvoyant SPT를 사용한다.

clairvoyant SPT, full BS8, partial batch 없음은 모두 upstream에 유리한 조건이다. 세 번의 upstream
throughput은 `1,169.51`, `1,169.85`, `1,170.49 token/s`다. 기존 CUDA-event GPU-only ceiling
`1,177.27 token/s`보다 0.6% 정도 낮아 측정 범위도 일관된다.

upstream E2E completion은 batch wall time을 사용한 값이다. public API가 first-token을 반환하지
않으므로 TTFT는 batch start와 prefill CUDA time으로, TPOT은 batch wall에서 prefill CUDA time을
제외한 뒤 generation step 수로 나눈 추정치다. current TTFT/TPOT/E2E는 real-request trace에서 직접
측정한 wall 값이다. 따라서 upstream 세부 latency는 optimistic estimate이고 E2E completion이 가장
신뢰할 수 있는 비교 지표다.

## Upstream 대비 current independent

| metric | delta |
| --- | ---: |
| throughput | `-13.3%` |
| makespan | `+17.4%` |
| TTFT median / p95 | `+48.7%` / `+13.6%` |
| E2E median / p95 | `+73.0%` / `+19.8%` |

current는 24,144 tokens, upstream profiler는 23,712 tokens를 생성했다. current가 `1.82%` 더 많은
token을 실제로 수행했으므로 current에 조금 불리하다. 그러나 이를 보정해도 P8/D8에서의 결론은
바뀌지 않는다.

independent context는 upstream과의 P8/D8 gap을 없애지는 못한다. shared current의 upstream 대비
throughput 손실 `15.9%`를 independent가 `13.3%`로 줄여 약 2.6 percentage point를 회복한다.
주요 throughput 향상은 D8 제약을 풀고 indexed-paged capacity로 D32 이상을 구성할 때 발생한다.

## 재현 자료

```text
scripts/cosmos_reason2/run_upstream_e2e_replay.py
.local/cosmos-reason2-2b/upstream-fair-loss-comparison/
  p8d8-pipeline-repeat1/
  p8d8-pipeline-repeat2/
  p8d8-pipeline-repeat3/
  upstream-e2e-replay-repeat1/
  upstream-e2e-replay-repeat2/
  upstream-e2e-replay-repeat3/
  p8d8-e2e-summary.csv
```

current의 세 run은 모두 동일 SHA256 materialized trace
`339c03622c22ca6b820fc1960d579a5319b9a32a61fc856bbe4c0e55038654d2`를 사용했다. current 여섯
case와 upstream 15개 grouped inference 실행은 모두 성공했다.
