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

# Cosmos upstream 대비 최대 향상 탐색

> 최대값뿐 아니라 동일 decode batch에서의 손해와 동일 raw KV 조건을 포함한 보수적 비교는
> `notes/54-cosmos-upstream-conservative-fair-comparison-20260811.md`에 정리했다.

## 결론

재현 가능한 최대-throughput 후보는 `P10/D56`, fixed-128 chunk, indexed-paged KV,
independent TensorRT context다.

| item | result |
| --- | ---: |
| requested prefill/decode cap | P10 / D56 |
| observed max prefill/decode batch | P9 / D56 |
| 3-run generated throughput median | 3,605.55 token/s |
| 3-run range | 3,600.17 ~ 3,607.13 token/s |
| upstream fixed-linear B8 optimistic ceiling | 1,177.27 token/s |
| speedup | 3.063x |
| improvement | +206.3% |

기존 단일 최고는 `P16/D48 = 3,645.84 token/s`, upstream 대비 `3.097x` 또는 `+209.7%`다.
그러나 같은 조건의 새 3-run median은 `3,476.74 token/s`였으므로 단발 최대치로만 보존하고
기본 scheduler 후보로 사용하지 않는다.

## 비교의 의미와 한계

public upstream fixed-linear engine은 stable slot이 없어 continuous admission/eviction을 지원하지
않는다. 현재 `llm_phase_bench`도 non-indexed engine의 real-request trace를 명시적으로 거부한다.
따라서 upstream에 현재 queue scheduler를 연결하면 더 이상 clean upstream 비교가 아니다.

대신 upstream에 유리한 포화 상한을 만들었다.

1. current와 동일한 288개 text prompt와 output-length 분포를 사용했다.
2. output max가 32/48/64/96/128인 request를 각각 모았다.
3. 각 그룹의 request 수가 8의 배수이므로 모든 upstream batch를 BS8로 완전히 채웠다.
4. arrival gap, queue 대기, partial batch가 전혀 없는 순차 실행으로 만들었다.
5. TensorRT CUDA-event의 prefill+generation GPU time만 합산했다. runtime 초기화, tokenizer,
   sampling, host overhead도 upstream 시간에서 제외했다.

따라서 `1,177 token/s`는 upstream production service 측정치가 아니라 fixed-linear B8의
낙관적인 active-GPU ceiling이다. current의 `3,606 token/s`는 실제 trace 시작부터 마지막 request
완료까지의 wall-clock throughput이므로 비교가 오히려 upstream에 유리하다. 다만 두 runtime은
batch trajectory가 달라 EOS 종료 수가 조금 달랐고, throughput은 각자가 실제 생성한 token 수로
정규화했다.

## Upstream ceiling

upstream은 `.local/upstream-main`의 `7f061f2`와 다음 engine을 사용했다.

```text
.local/upstream-baseline/cosmos-reason2-2b/
  engine-fp16-fixed-b8-i1024-kv2048/
  real-trace-bs8/
    input-output{32,48,64,96,128}.json
    output-output{32,48,64,96,128}.json
    profile-output{32,48,64,96,128}.csv
```

| output max | requests | full BS8 batches | generated tokens | prefill GPU ms | generation GPU ms |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 24 | 3 | 432 | 50.19 | 305.81 |
| 48 | 24 | 3 | 1,152 | 51.86 | 848.59 |
| 64 | 72 | 9 | 4,608 | 550.93 | 3,505.14 |
| 96 | 96 | 12 | 8,304 | 380.31 | 6,982.12 |
| 128 | 72 | 9 | 9,216 | 398.00 | 7,068.48 |
| total | 288 | 36 | 23,712 | 1,431.29 | 18,710.14 |

```text
23,712 tokens / (1,431.29 + 18,710.14) ms = 1,177.27 token/s
```

upstream fixed-linear B16은 10GB GPU에서 OOM이므로 BS8이 실행 가능한 최대 upstream batch다.
BS1/2/4/8 decode microbench에서도 batch가 클수록 aggregate token throughput이 증가하므로, 완전
BS8 구성은 upstream의 합리적인 최선 조건이다.

## Current search space

current engine은 다음 capability를 사용했다.

```text
maxBatchSize=80
maxPrefillBatchSize=16
maxDecodeBatchSize=64
indexed-paged KV, page size=128, page pool=256 bundles
stable slots=80
CUDA primary context=shared
prefill/decode TensorRT execution context=independent
```

workload는 288 requests, Poisson 1,000 req/s, output multiplier 4, seed 7, fixed-128 chunk다.
output max 분포는 32×24, 48×24, 64×72, 96×96, 128×72다. 첫 request 도착 뒤 약 294ms에
모든 request가 제출되므로 queue를 충분히 포화시킨다.

탐색은 다음 순서로 수행했다.

1. 48-request coarse: P1/2/4/6/8/10/12/14/16 × D40/48/56, 27 cases
2. 288-request saturation: P4/6/8/10/12/16 × D40/48/56, 18 cases
3. finalist: P8/10/12/16 × D48/56/64, 실행 순서를 회전해 3회 반복

모든 case가 성공했고 finalist에서는 page exhaustion이 없었다. 원자료와 집계표는 다음에 있다.

```text
.local/cosmos-reason2-2b/upstream-max-speedup-search/
  coarse/
  saturated/
  repeat1/
  repeat2/
  repeat3/
  repeat-summary.csv
```

## Finalist 3-run median

각 latency 값도 세 번의 scenario 값에 대한 median이다.

| rank | case | token/s median (range) | TTFT med/p95 ms | TPOT med/p95 ms | E2E med/p95 ms |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | P10/D56 | 3,605.55 (3,600.17~3,607.13) | 3,035.7 / 5,425.2 | 10.70 / 13.01 | 3,875.2 / 6,245.9 |
| 2 | P12/D56 | 3,597.88 (3,513.86~3,602.08) | 3,040.0 / 5,437.4 | 10.71 / 13.03 | 3,879.8 / 6,260.4 |
| 3 | P16/D56 | 3,596.05 (3,596.02~3,599.90) | 3,047.1 / 5,440.0 | 10.72 / 13.07 | 3,887.9 / 6,263.3 |
| 4 | P8/D64 | 3,591.87 (3,587.88~3,603.76) | 2,852.3 / 5,460.2 | 10.52 / 12.21 | 3,757.1 / 6,175.9 |
| 5 | P8/D56 | 3,589.55 (3,583.00~3,594.20) | 2,857.7 / 5,506.9 | 10.49 / 12.52 | 3,747.6 / 6,205.6 |
| 6 | P8/D48 | 3,580.16 (3,575.02~3,592.62) | 2,940.1 / 5,494.1 | 11.28 / 13.49 | 3,856.0 / 6,269.9 |
| 7 | P10/D64 | 3,517.84 (3,516.96~3,520.13) | 3,094.9 / 5,530.2 | 10.36 / 12.61 | 3,873.2 / 6,415.1 |
| 8 | P12/D64 | 3,514.85 (3,511.63~3,516.88) | 3,098.5 / 5,535.7 | 10.37 / 12.62 | 3,878.2 / 6,420.6 |
| 9 | P16/D64 | 3,511.55 (3,508.91~3,513.48) | 3,102.0 / 5,542.2 | 10.38 / 12.65 | 3,882.0 / 6,427.5 |

P10/D56은 세 번 모두 observed max P9/D56이었다. P10보다 큰 prefill cap은 실제 batch를 더 키우지
못하면서 작은 scheduling/competition 차이만 만들었다. D64는 P8에서 tail이 가장 좋았지만
P10 이상에서는 prefill+decode 동시 slot/SM pressure가 증가해 throughput이 D56보다 낮았다.

## 목적별 선택

| objective | candidate | reason |
| --- | --- | --- |
| 최대 재현 throughput | P10/D56 | 3,605.55 token/s, upstream 대비 3.063x |
| latency median 균형 | P8/D56 | TPOT median 10.49ms, E2E median 3,747.6ms |
| tail latency | P8/D64 | TPOT p95 12.21ms, E2E p95 6,175.9ms |
| 단발 최대 기록 | P16/D48 | 3,645.84 token/s지만 반복 재현 실패 |

현재 queue-default scheduler의 saturation preset은 `P10/D56`으로 둘 수 있다. 다만 낮은 load에서
이 cap을 채우려고 기다리면 TTFT가 악화될 수 있으므로, production 정책은 observed queue depth와
decode age를 보고 `P8/D48 -> P8/D56 -> P10/D56`으로 승격하는 형태가 적절하다.

## 왜 3배인가

512-token isolated kernel 비용은 upstream과 current가 2% 이내이므로 3배 향상은 단일 kernel
가속이 아니다.

- upstream은 fixed KV 때문에 B8이 메모리 상한이고 request batch도 8을 넘지 못한다.
- paged KV는 maxBatch80에서 실제 D56을 구성한다.
- decode 한 step의 latency는 batch에 비례해 7배 증가하지 않으므로 aggregate token/s가 커진다.
- stable slot 때문에 request 종료 시 KV row compaction/copy가 없다.
- independent TensorRT context가 fixed-128 prefill chunk와 decode를 겹친다.
- prefill cap을 P10으로 제한해 큰 prefill이 decode stream을 과도하게 막는 것을 피한다.

따라서 최대 향상은 `indexed-paged + 큰 decode dynamic batch + 제한된 prefill batch + independent
context overlap`의 조합 효과다.
