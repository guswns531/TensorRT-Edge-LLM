SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

# Cosmos 다섯 workload Current, vLLM, clean upstream 비교

## 결론

balanced 하나만으로 내렸던 판단을 short, balanced, decode-heavy, long-prefill, bimodal-mixed 다섯
real-request trace로 확장했다. Current와 vLLM은 같은 localhost OpenAI-compatible HTTP/SSE client를 사용했고,
각 workload를 세 번 완주했다.

- Current generated token/s는 vLLM보다 다섯 workload에서 각각 `+20.96%`, `+7.70%`, `+9.82%`,
  `+29.91%`, `+13.04%` 높았다.
- Current가 vLLM보다 나쁜 주요 지표는 balanced TPOT p95 `+1.69%`와 decode-heavy TTFT median `+7.13%`다.
  나머지 workload/latency median·p95 조합은 Current가 더 낮았다.
- Current의 세 run 처리량 CV는 `0.08--0.71%`였다. 직전 balanced-only 재측정의 큰 batch-trajectory 편차는
  재현되지 않았다.
- clean upstream fixed-BS8 oracle 대비 Current 처리량은 workload별 `1.73--4.84x`다. 이는 KV 형식 하나의
  효과가 아니라 continuous admission, P8/D64 비대칭 batch, independent TensorRT contexts, phase overlap을 합친
  serving 구조 차이다.
- long-prefill과 bimodal은 page pool을 각각 240/256, 228/256 bundle까지 사용했다. 큰 decode batch를 더
  형성하려면 raw page 수 증설보다 reservation/admission 정책 개선이 먼저다.

## 비교 조건

| 항목 | Current | vLLM | clean upstream |
| --- | --- | --- | --- |
| 모델 | `nvidia/Cosmos-Reason2-2B`, FP16 | 동일 | 동일 |
| KV | indexed-paged FP16, 128-token page, 256 bundles | paged FP16, 3.5GiB | fixed-linear FP16 |
| active capacity | 80 stable slots | max 80 sequences | fixed BS8 |
| prefill/decode | P8/D64, fixed chunk 128, packed prefill | max sequences 80, chunked prefill | 한 BS8 batch를 완료할 때까지 직렬 |
| 실행 | CUDA context 하나, independent TensorRT contexts/streams | vLLM 0.27.1 production engine | public upstream `llm_inference` |
| graph | phase별 CUDA graph, workload shape priming | vLLM full/piecewise CUDA graph | 없음 |
| 반복 | workload마다 fresh backend 3회 | 한 server lifecycle, workload마다 warm-up 뒤 3회 | homogeneous output group 측정 3회 |

Current는 `prefillTokenBudget=1024`, `maxOverlapPrefillTokens=1024`로 고정했다. 이 두 값을 누락한 진단
실행에서는 P8 prefill이 decode와 거의 겹치지 않아 short가 1,724 token/s까지 떨어졌다. production 값을 복원한
최종 short는 2,464 token/s로 이전 결과를 재현했다. 진단 실행은 아래 집계에 포함하지 않았다.

Current의 graph profile은 기존 workload dispatch에서 만들었다. bimodal은 측정 전에 별도 1회로 shape를 수집한 뒤
세 측정 run을 시작했다. long-prefill 전용 256-token engine은 디스크 정리에서 삭제됐으므로 이번 Current는 보존된
공통 128-token engine을 사용했다.

clean upstream에는 continuous admission과 HTTP streaming이 없다. 따라서 출력 길이별 homogeneous BS8 batch의
실제 wall cost를 측정하고, original arrival 위에 upstream에 유리한 clairvoyant shortest-processing-time 순서로
재생했다. TTFT와 TPOT은 client-observed 값이 아니라 이 replay의 optimistic estimate다. short의 네 partial batch는
padding 없이 실행했다. balanced upstream 값은 같은 날 같은 binary/engine으로 완료한 직전 세 run을 재사용했다.

## Workload 범위

| Workload | Requests | Prompt tokens | prompt p50/p95/max | requested output | output p50/p95/max | arrival span |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| short | 48 | 4,312 | 93/236/236 | 1,040 | 24/32/32 | 33ms |
| balanced | 288 | 25,872 | 93/236/236 | 24,960 | 96/128/128 | 295ms |
| decode-heavy | 288 | 25,872 | 93/236/236 | 74,880 | 288/384/384 | 278ms |
| long-prefill | 288 | 213,024 | 711/964/964 | 24,960 | 96/128/128 | 295ms |
| bimodal-mixed | 288 | 117,864 | 422/847/847 | 44,160 | 104/384/384 | 295ms |

Trace SHA-256은 short `3f689bc3...294e459`, balanced `290d3406...49d6538`, decode-heavy
`68f523e9...a349af`, long-prefill `7d713d97...79886ec3`, bimodal-mixed `b3e8ab21...e0d29c`다.

## HTTP E2E 처리량

값은 세 complete run의 중앙값이다. upstream은 앞에서 설명한 replay oracle이다.

| Workload | Current tok/s | vLLM tok/s | Current 변화 | Current req/s | vLLM req/s | upstream tok/s | Current/upstream |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| short | **2,463.9** | 2,036.9 | **+20.96%** | **113.72** | 94.01 | 838.6 | 2.94x |
| balanced | **4,413.4** | 4,097.9 | **+7.70%** | **53.60** | 49.77 | 1,167.4 | 3.78x |
| decode-heavy | **4,848.9** | 4,415.5 | **+9.82%** | **24.29** | 22.38 | 1,000.9 | 4.84x |
| long-prefill | **1,418.4** | 1,091.8 | **+29.91%** | **16.57** | 12.97 | 819.9 | 1.73x |
| bimodal-mixed | **1,997.2** | 1,766.9 | **+13.04%** | **14.57** | 13.06 | 930.3 | 2.15x |

EOS 위치가 backend별로 조금 다르다. 실제 생성 token은 Current/vLLM/upstream 순으로 short
`1,040/1,040/1,040`, balanced `23,712/23,712/23,712`, decode-heavy `57,480/56,814/57,936`,
long-prefill `24,648/24,240/24,432`, bimodal `39,480/38,966/38,952`다. 따라서 뒤의 세 workload는
token/s뿐 아니라 request/s와 latency를 함께 판단해야 한다.

## Current와 vLLM latency

절대값은 `median / p95`다.

| Workload | TTFT Current | TTFT vLLM | TPOT Current | TPOT vLLM | E2E Current | E2E vLLM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| short | **148.7/204.7ms** | 209.3/258.6ms | **8.960/14.622ms** | 10.867/24.372ms | **333.5/405.1ms** | 428.0/492.0ms |
| balanced | **1,630/3,854ms** | 1,831/4,154ms | **16.287**/19.116ms | 17.377/**18.798ms** | **3,051/4,813ms** | 3,386/5,176ms |
| decode-heavy | 3,690/**8,144ms** | **3,444**/8,591ms | **12.420/13.212ms** | 15.555/16.945ms | **5,751/10,680ms** | 6,553/11,635ms |
| long-prefill | **7,727/15,336ms** | 10,021/19,679ms | **21.980/22.883ms** | 35.239/39.771ms | **9,563/16,595ms** | 12,895/21,418ms |
| bimodal-mixed | **7,278/15,635ms** | 7,839/17,209ms | **15.447/18.124ms** | 22.148/38.674ms | **9,571/17,800ms** | 10,855/20,185ms |

Current의 vLLM 대비 변화율은 다음과 같다. latency는 음수가 개선이다.

| Workload | TTFT med/p95 | TPOT med/p95 | E2E med/p95 |
| --- | ---: | ---: | ---: |
| short | -28.95% / -20.83% | -17.54% / -40.00% | -22.08% / -17.66% |
| balanced | -10.96% / -7.22% | -6.27% / **+1.69%** | -9.87% / -7.03% |
| decode-heavy | **+7.13%** / -5.21% | -20.16% / -22.03% | -12.23% / -8.21% |
| long-prefill | -22.89% / -22.07% | -37.62% / -42.46% | -25.84% / -22.52% |
| bimodal-mixed | -7.15% / -9.15% | -30.26% / -53.14% | -11.83% / -11.82% |

decode-heavy의 TTFT median 손실은 decode가 커진 뒤 들어온 prefill이 기다리는 정책 trade-off다. 하지만 TTFT p95,
TPOT, E2E는 모두 Current가 낫다. balanced TPOT p95 손실은 0.318ms로 작지만 유일하게 반복해서 남는 tail gate다.

## 실제 dynamic batch, overlap, page pressure

세 run의 dispatch를 합쳤다. overlap은 같은 scheduler dispatch에 prefill과 decode가 모두 존재한 비율이다.

| Workload | observed max P/D | overlap dispatch | page peak | prefill engine med/p95 | decode engine med/p95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| short | 8/47 | 18.9% | 56/256 | 26.69/37.88ms | 7.19/19.51ms |
| balanced | 8/64 | 32.6% | 144/256 | 14.19/29.72ms | 8.33/14.25ms |
| decode-heavy | 8/64 | 11.1% | 190/256 | 14.93/30.17ms | 8.93/12.64ms |
| long-prefill | 8/36 | 73.3% | 240/256 | 20.62/37.73ms | 14.71/20.80ms |
| bimodal-mixed | 8/53 | 36.2% | 228/256 | 18.69/33.08ms | 10.41/15.99ms |

balanced/decode-heavy에서 D64가 실제 형성됐다. long-prefill은 overlap이 가장 많지만 page pressure 93.75% 때문에
D36에서 끝났다. bimodal은 짧은 요청과 긴 요청이 섞여 D53까지 형성됐고 page pressure는 89.06%였다.

CUDA graph의 prefill cache는 모든 workload에서 3 entries, 14MiB뿐이다. hit rate는 short/balanced/decode-heavy/
long-prefill/bimodal 순으로 `37.5/5.4/23.5/27.7/56.3%`다. decode는 `73.9/88.2/88.5/95.9/96.9%`로
높다. 다음 graph 최적화는 decode entry 증설보다 prefill shape bucketing이 우선이라는 기존 결론을 다시 확인했다.

## 메모리와 반복 안정성

Current의 workload 종료 시 device 사용량은 short 9,059MiB, balanced/decode-heavy 9,185MiB, long-prefill
9,117MiB, bimodal 9,161MiB다. 최소 headroom은 689MiB로 512MiB gate를 통과했다. 같은 vLLM 설정의 보존된
device 사용량은 약 8,154MiB, clean upstream peak는 7,348MiB다.

tied embedding/LM-head opt-in은 앞선 검증에서 Current보다 574MiB를 절감했지만 exact greedy identity gate가 남아
이번 기본 비교에는 넣지 않았다.

| Workload | Current token/s range / CV | vLLM token/s range / CV |
| --- | ---: | ---: |
| short | 2,463.7--2,467.9 / 0.08% | 1,985.3--2,063.8 / 1.61% |
| balanced | 4,371.7--4,448.8 / 0.71% | 4,093.3--4,136.0 / 0.47% |
| decode-heavy | 4,834.0--4,877.8 / 0.37% | 4,414.5--4,416.2 / 0.02% |
| long-prefill | 1,414.0--1,419.2 / 0.16% | 1,091.2--1,093.2 / 0.08% |
| bimodal-mixed | 1,991.9--2,005.1 / 0.27% | 1,764.7--1,770.9 / 0.14% |

## 해석과 다음 순서

1. Current는 한 balanced run의 우연한 trajectory가 아니라 다섯 workload 전체에서 vLLM보다 높은 system
   throughput을 보였다.
2. short/long-prefill/bimodal의 큰 이득은 phase overlap과 packed chunked prefill이 실제 serving latency도
   줄인 결과다. decode kernel 자체가 빨라졌다고만 해석하면 안 된다.
3. balanced TPOT p95와 decode-heavy TTFT median은 서로 반대 방향의 scheduler trade-off다. 다음 정책은 workload를
   미리 분류하기보다 online decode pressure와 prefill age로 이 두 지표를 동시에 guard해야 한다.
4. long/bimodal의 다음 병목은 slot 수가 아니라 page reservation이다. bounded overcommit/growth lease를 EOS-enabled
   HTTP trace로 다시 검증하고, page wait와 D batch 증가가 TPOT을 해치지 않는지 확인한다.
5. mixed-load low/burst/recovery trace와 arrival-rate/output-length sweep은 별도 부하 축이다. 다음 회귀는 이 다섯
   workload를 고정 gate로 유지하면서 rate `10/30/200/1000 req/s`와 output scale `1x/2x/4x`를 교차한다.
6. clean upstream은 fixed-batch kernel/correctness oracle로 유지한다. production 성능 목표와 기능 격차는 같은 HTTP
   surface를 사용하는 vLLM을 기준으로 판단한다.

## Artifact

- Current: `.local/cosmos-reason2-2b/all-workload-recheck-20260814/current/`
- vLLM: `.local/cosmos-reason2-2b/all-workload-recheck-20260814/vllm/`
- clean upstream 새 측정: `.local/cosmos-reason2-2b/all-workload-recheck-20260814/upstream/`
- graph profiles: `.local/cosmos-reason2-2b/all-workload-recheck-20260814/*-shapes.json`
- balanced upstream 재사용 원본: `.local/cosmos-reason2-2b/three-way-recheck-20260814/upstream-balanced-r1..r3/`

