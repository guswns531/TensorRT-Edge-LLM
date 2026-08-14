# Cosmos tied LM-head CUDA Graph workload 재검증

## 결론

Cosmos-Reason2-2B의 tied embedding/LM-head shared layout을 independent prefill/decode context와 CUDA Graph를
함께 사용해 네 종류의 real-request trace로 다시 측정했다.

- 모든 run에서 CUDA Graph capture/launch failure와 graph budget rejection은 0이었다.
- baseline과 shared layout의 cached shape, capture, launch, enqueue, graph byte 수가 workload별로 정확히 같았다.
- shared layout은 graph 적용 후에도 baseline 대비 GPU 사용량을 574MiB 줄였다.
- shared layout의 최악 처리량 회귀는 balanced의 `-0.582%`, 최악 TPOT p95 회귀는 balanced의 `+0.857%`였다.
- 따라서 이번 CUDA Graph 조건에서도 3% performance gate를 통과한다.
- CUDA Graph 자체는 충분히 긴 balanced/decode-heavy/long-prefill에서 처리량을 `+1.0--2.7%` 높였다. short는
  cold dynamic capture 비용이 실행 절감보다 커서 shared layout 기준 `-0.774%`였다.

기능은 계속 experimental opt-in이다. 별도 TensorRT build 사이 exact greedy token identity가 아직 성립하지 않기
때문이다.

## 고정 조건

- GPU: RTX 3080 10GB
- model: `nvidia/Cosmos-Reason2-2B`, FP16
- KV: indexed-paged FP16, 128-token page, 256 bundles
- topology: CUDA context 공유, independent TensorRT prefill/decode execution context
- engine: P8/D64, 80 slots, max input 1,024, max KV 2,048
- prefill: fixed 128-token chunk, ragged batching, packed token layout
- scheduler: `queue_default`
- admission: full page reservation
- arrival rate: 30 request/s
- output: `--ignoreTraceEos`로 trace의 요청 output 길이를 고정
- 반복: engine variant와 workload마다 새 process lifecycle 3회, 중앙값 비교

CUDA Graph budget은 prefill 4 entries/16MiB, decode 64 entries/220MiB이고 shape당 최소 charge는 4MiB다.
warm-up profile을 넣지 않았으므로 이 결과는 production priming 이후의 steady state가 아니라 cold dynamic capture를
포함한다.

## Shared layout A/B

아래 변화율은 `(shared layout / baseline - 1) * 100`이다. 처리량은 양수가 좋고 latency는 음수가 좋다.

| Workload | Output tokens | Baseline token/s | Shared token/s | 처리량 변화 | TTFT med / p95 | TPOT med / p95 | E2E med / p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| short | 1,040 | 1,678.4 | 1,676.7 | -0.100% | -0.142% / +0.047% | +0.094% / +0.019% | +0.051% / +0.090% |
| balanced | 24,960 | 4,532.4 | 4,506.1 | -0.582% | +0.550% / +0.735% | +0.571% / +0.857% | +0.699% / +0.653% |
| decode-heavy | 74,880 | 5,068.2 | 5,050.7 | -0.345% | +0.277% / +0.453% | +0.406% / +0.711% | +0.299% / +0.400% |
| long-prefill | 24,960 | 1,204.2 | 1,200.8 | -0.285% | +0.358% / +0.314% | +0.184% / +0.621% | +0.363% / +0.304% |

Graph 상태가 두 engine 사이에서 정확히 같으므로 이전 smoke처럼 graph hit 차이가 A/B를 왜곡하지 않았다. 네
workload 모두 fixed-output이므로 EOS 시점이 batch shape와 측정량을 바꾸는 문제도 제거했다.

## Graph cache와 GPU headroom

`prefill cached/captures/launches/hit`, `decode cached/captures/launches/hit` 순서다. 아래 값은 두 engine에서 같다.

| Workload | Prefill graph | Decode graph | Graph bytes 합 | Baseline free | Shared free |
| --- | ---: | ---: | ---: | ---: | ---: |
| short | 1/1/7/43.75% | 10/10/59/84.29% | 46MiB | 877MiB | 1,451MiB |
| balanced | 4/4/12/12.24% | 35/35/459/91.25% | 162MiB | 761MiB | 1,335MiB |
| decode-heavy | 4/4/46/29.11% | 52/52/1,332/92.95% | 234MiB | 689MiB | 1,263MiB |
| long-prefill | 3/3/147/50.34% | 36/36/1,450/97.18% | 162MiB | 761MiB | 1,335MiB |

graph-off 대비 실제 device memory 증가는 표의 graph byte 합과 정확히 일치했다. shared layout은 graph allocation
정책을 바꾸지 않으므로 graph byte 자체를 줄이지는 않지만, weight sharing으로 확보한 574MiB가 그대로 추가
headroom이 된다. 가장 빡빡한 decode-heavy에서 baseline 689MiB 대비 shared layout은 1,263MiB를 남긴다.

Prefill hit가 낮은 이유는 fixed-128 chunk여도 마지막 chunk 길이, batch row 수, initial/continuation 조합이 달라져
shape가 분산되기 때문이다. decode는 active row와 context shape가 반복되어 91--97% hit를 얻는다. 다음 graph
최적화는 decode entry를 더 늘리는 것이 아니라 prefill shape bucketing 또는 warm-up profile 개선이 우선이다.

## CUDA Graph 자체의 효과

아래는 같은 engine에서 graph-off 대비 graph-on의 `token/s 변화 / TPOT p95 변화`다. balanced는 이번에 같은
fixed-output graph-off 대조를 추가했고, 나머지는 앞선 fixed-output 3회 결과를 재사용했다.

| Workload | Baseline | Shared layout | Graph memory |
| --- | ---: | ---: | ---: |
| short | -0.480% / -0.186% | -0.774% / +0.127% | +46MiB |
| balanced | +1.134% / -2.856% | +1.038% / -2.635% | +162MiB |
| decode-heavy | +2.613% / -3.717% | +2.716% / -3.590% | +234MiB |
| long-prefill | +2.361% / +0.019% | +2.454% / +0.113% | +162MiB |

짧은 trace에서는 새 lifecycle의 capture 비용을 amortize하지 못한다. 반대로 decode-heavy는 1,332번 decode graph
launch가 발생해 host enqueue 절감 효과가 가장 크다. long-prefill은 처리량이 개선되지만 TPOT p95는 거의 같고,
prefill 처리량 개선이 주효하다.

## Kernel-group 해석

shared layout의 decode engine 평균 변화는 모든 sample 기준 short `-0.063%`, balanced `+0.113%`, decode-heavy
`-0.106%`, long-prefill `-0.127%`로 사실상 동일하다. 반면 prefill engine은 balanced `+0.915%`, decode-heavy
`+1.142%`, long-prefill `+0.514%`였다.

overlap sample만 보면 shared prefill engine 회귀가 balanced `+1.363%`, decode-heavy `+1.569%`, long-prefill
`+2.018%`까지 커진다. 그래도 요청 TPOT p95는 1% 안쪽이다. 현재 수치는 shared layout의 external LM-head input과
TensorRT tactic 차이가 prefill/overlap에서 작은 비용을 만든다는 기존 분석과 일치한다.

## 정확성 상태

각 engine 안에서는 동일 workload 3회 token sequence가 모두 재현됐다. 그러나 baseline engine과 shared-layout
engine 사이 exact request sequence 일치율은 short 44/48, balanced 96/288, decode-heavy 24/288,
long-prefill 96/288이다. constant GEMM을 external-input GEMM으로 바꾸면서 tactic과 FP16 누산 순서가 달라진 기존
문제가 CUDA Graph에서도 그대로 보인다.

따라서 이번 결과는 fixed work 성능과 graph 호환성 gate이지 정확성 승인 결과가 아니다. 기본값 전환 전에
teacher-forced logits allclose와 task/accuracy suite가 필요하다.

## TPOT hard guard 범위

이번 실험은 shared-layout 자체와 CUDA Graph의 영향을 분리하려고 `queue_default`를 유지했다. TPOT hard guard는
scheduler cost table을 필요로 하지만 현재 table은 baseline engine에서 생성됐다. shared prefill overlap cost가 최대
약 2% 달라진다는 이번 결과에서 그 table을 그대로 쓰면 guard 검증이 아니라 stale cost-model 검증이 된다.

그러므로 hard guard A/B는 이번 단계에 억지로 섞지 않는다. shared-layout 전용 kernel-group cost table을 만든 뒤
engine/layout fingerprint로 table 선택을 강제하고, balanced와 decode-heavy에서 guard off/on을 비교하는 것이 다음
순서다.

## Artifact

- 전체 A/B: `.local/cosmos-reason2-2b/tied-head-graph-workload-ab-20260814/summary.json`
- 전체 CSV: `.local/cosmos-reason2-2b/tied-head-graph-workload-ab-20260814/summary.csv`
- graph 효과: `.local/cosmos-reason2-2b/tied-head-graph-workload-ab-20260814/graph-effect-summary.json`
- balanced graph-off: `.local/cosmos-reason2-2b/tied-head-graph-workload-ab-20260814/balanced-nograph-summary.json`
- 각 run의 request/token/prefill/dispatch/kernel-group CSV와 log는 같은 디렉터리 아래에 있다.

## 다음 순서

1. shared-layout engine으로 isolated/overlap kernel-group cost table을 다시 생성한다.
2. table에 engine SHA와 embedding layout fingerprint를 넣고 mismatch를 runtime에서 거부한다.
3. 새 table로 TPOT hard guard와 throughput-balanced scheduler를 balanced/decode-heavy에서 재검증한다.
4. teacher-forced logits와 Cosmos accuracy suite로 exact-token 불일치의 허용 가능성을 판정한다.
5. prefill graph shape bucketing과 production warm-up profile로 낮은 prefill hit를 개선한다.
