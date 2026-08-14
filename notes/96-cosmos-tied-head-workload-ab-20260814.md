# Cosmos tied LM-head workload 확장 A/B

## 결론

Tied embedding/LM-head GPU buffer 공유를 short, balanced, decode-heavy, long-prefill real-request workload로
확장 검증했다. graph-off, P8/D64, 80 slots, indexed-paged KV 256 bundles, fixed chunk 128, independent TensorRT
contexts와 shared CUDA primary context를 고정했다.

EOS가 서로 다른 TensorRT build의 수치 차이로 성능 비교를 왜곡하지 않도록 decode-heavy와 long-prefill은
`--ignoreTraceEos`로 모든 요청을 지정 output 길이까지 실행했다. short는 원래 모든 요청이 max-length였고,
balanced는 baseline/layout의 요청별 output 길이가 모두 같았다. 각 variant는 세 번 새 process에서 실행했다.

- GPU 사용량은 모든 workload에서 실행 전 `-576MiB`, 실행 후 `-574MiB`였다.
- 동일 작업량에서 generated token throughput의 최악 회귀는 decode-heavy의 `-0.445%`였다.
- 주요 E2E median/p95 최악 회귀는 decode-heavy의 `+0.604% / +0.494%`였다.
- 네 workload 모두 3% performance gate를 충분히 통과했다.
- 다만 decode-heavy의 overlap 중 prefill engine event는 `+9.24%` 증가했다. tied engine 전용 overlap cost
  table을 다시 생성해야 한다.

## Real-request 결과

변화율은 `(shared-layout / baseline - 1)`이다. latency는 양수가 회귀다.

| Workload | Output tokens | Throughput 변화 | TTFT med / p95 | TPOT med / p95 | E2E med / p95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| short | 1,040 | **+0.196%** | +0.041% / -0.178% | +0.003% / -0.294% | -0.188% / -0.228% |
| balanced | 23,712 | -0.206% | +0.185% / +0.211% | +0.349% / +0.405% | +0.128% / +0.227% |
| decode-heavy, fixed length | 74,880 | **-0.445%** | +0.576% / +0.578% | +0.525% / +0.579% | +0.604% / +0.494% |
| long-prefill, fixed length | 24,960 | **-0.376%** | +0.442% / +0.409% | +0.344% / +0.526% | +0.435% / +0.395% |

절대값은 다음과 같다.

| Workload | Baseline / shared tok/s | Baseline / shared TTFT median | Baseline / shared TPOT median | Baseline / shared E2E median |
| --- | ---: | ---: | ---: | ---: |
| short | 1,686.47 / 1,689.77 | 287.96 / 288.08ms | 8.720 / 8.720ms | 490.56 / 489.64ms |
| balanced | 3,623.86 / 3,616.41 | 2,723.73 / 2,728.76ms | 10.567 / 10.603ms | 3,620.85 / 3,625.50ms |
| decode-heavy | 4,939.17 / 4,917.21 | 5,005.76 / 5,034.60ms | 12.844 / 12.911ms | 8,390.03 / 8,440.73ms |
| long-prefill | 1,176.43 / 1,172.01 | 10,716.30 / 10,763.71ms | 14.031 / 14.079ms | 11,623.18 / 11,673.76ms |

## 작업량과 scheduler 동등성

성능 비교에 사용한 run은 양쪽의 output token, scheduler dispatch 수, observed batch, page-pool peak가 같았다.

| Workload | Dispatch / overlap | Observed max P / D | Page peak | Baseline/shared output |
| --- | ---: | ---: | ---: | ---: |
| short | 70 / 0 | 8 / 28 | 40 / 256 | 1,040 / 1,040 |
| balanced | 727 / 16 | 8 / 64 | 118 / 256 | 23,712 / 23,712 |
| decode-heavy | 1,484 / 91 | 8 / 64 | 195 / 256 | 74,880 / 74,880 |
| long-prefill | 1,697 / 71 | 8 / 35 | 229 / 256 | 24,960 / 24,960 |

따라서 최종 표의 차이는 EOS 조기 종료, 다른 page pressure, 다른 batch 형성 때문에 생긴 것이 아니다.

## Kernel-group 분석

세 process의 각 group 평균 중앙값을 비교했다.

| Workload | Prefill engine 변화 | Decode engine 변화 |
| --- | ---: | ---: |
| short | -0.326% | -0.213% |
| balanced | +0.372% | +0.089% |
| decode-heavy | **+5.674%** | +0.314% |
| long-prefill | +0.786% | -0.101% |

decode-heavy의 prefill을 동시 decode 유무로 다시 나누면 원인이 더 분명하다.

| Decode-heavy prefill | Baseline | Shared layout | 변화 |
| --- | ---: | ---: | ---: |
| Prefill-only event | 15.245ms | 15.336ms | +0.597% |
| Prefill+decode overlap event | 13.538ms | 14.788ms | **+9.237%** |

decode engine은 solo `+0.200%`, overlap `+0.869%`였다. 전체 E2E 회귀가 0.45% 수준인 이유는 1,484개
scheduler dispatch 중 overlap이 91개뿐이기 때문이다. overlap 비율이 훨씬 높은 workload에서는 tied engine용
cost model 없이 공격적으로 겹치면 손실이 커질 수 있다.

## Fixed-shape sequential/overlap 실험

Real-request scheduler 효과를 제거하기 위해 P1/D64와 P8/D64, input 128, past KV 0을 각 mode 50회 교차
실행했다. 이 fixed path는 random `inputsEmbeds`를 engine에 직접 넣으므로 transposed embedding lookup kernel을
실행하지 않는다. 따라서 여기서 관측한 차이는 TensorRT LM-head constant를 external input binding으로 바꾼
효과와 별도 build tactic 차이의 합이다.

| Shape | Mode | Baseline makespan | Shared makespan | Shared 변화 |
| --- | --- | ---: | ---: | ---: |
| P1/D64 | sequential | 18.591ms | 18.587ms | -0.025% |
| P1/D64 | independent overlap | 15.237ms | 15.623ms | **+2.539%** |
| P8/D64 | sequential | 42.989ms | 43.098ms | +0.255% |
| P8/D64 | independent overlap | 39.496ms | 40.198ms | **+1.777%** |

P1/D64 overlap speedup은 baseline `1.220x`, shared `1.190x`이고 P8/D64는 `1.088x`, `1.072x`다. 공유
경로도 sequential보다 빠르지만 overlap 이득이 각각 3.05, 1.63 percentage points 줄었다.

이 실험만으로 같은 physical buffer의 동시 읽기와 TensorRT tactic 변경을 분리할 수는 없다. 다음 정확한 원인
분리는 baseline/shared ONNX에 같은 GEMM tactic을 고정한 engine을 만들거나, 두 engine의 layer profiling을 비교해야
한다.

## 잘못된 비교가 만든 가짜 향상

처음 long-prefill을 EOS 활성 상태로 실행했을 때 shared throughput은 `+7.81%`, request/s는 `+9.62%`로 보였다.
하지만 baseline/shared output은 `24,648 / 24,240`, dispatch는 `1,768 / 1,453`으로 서로 달랐다. 반면 kernel
event는 shared prefill `+6.90%`, decode `+12.76%`로 더 느렸다.

즉 별도 build의 작은 logits 차이가 EOS 시점, slot release, admission과 batch 형성을 바꾼 결과였다. online serving
A/B에서는 단순 token/s만 보면 잘못된 결론을 낼 수 있으며, 최소한 요청별 output length, finish reason, dispatch
shape, page pressure를 함께 고정하거나 검증해야 한다.

## 판단과 다음 단계

현재 tied sharing은 다음 상태다.

1. 메모리: 반복 가능하게 574--576MiB 절감, 통과.
2. graph-off real-request 성능: 네 workload 모두 0.65% 이내, 통과.
3. independent overlap: 기능은 유지되지만 고정 shape에서 최대 2.54% makespan 회귀, 조건부 통과.
4. exact greedy identity: 별도 TensorRT build 사이에서 불일치하므로 아직 미통과.

따라서 experimental opt-in은 유지할 수 있지만 기본값으로 전환하지 않는다. 다음에는 tied engine으로 kernel-group
cost table을 새로 만들고 scheduler가 그 table을 선택하도록 engine fingerprint를 연결한다. 동시에 teacher-forced
logits 비교와 accuracy suite를 완료해야 한다.

결과 artifact는 다음 위치에 있다.

- real-request 요약: `.local/cosmos-reason2-2b/tied-head-workload-ab-20260814/fair-summary.json`
- real-request CSV: `.local/cosmos-reason2-2b/tied-head-workload-ab-20260814/fair-summary.csv`
- fixed overlap 요약: `.local/cosmos-reason2-2b/tied-head-workload-ab-20260814/fixed-overlap/summary.json`
