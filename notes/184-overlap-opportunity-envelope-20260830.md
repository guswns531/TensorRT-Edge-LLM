# Overlap Opportunity Envelope

## 결론

이번 단계에서는 production selector를 튜닝하지 않고, 독립 P/D context가 제공하는 실행 자유도의 **최대 성능 가능 범위**를 실제 HTTP request trace에서 측정했다. 그 결과 P+D overlap은 불필요한 기능이 아니라 다음과 같은 명확한 Pareto trade-off를 만든다는 것을 확인했다.

- 실제 P+D active overlap 비율을 0%에서 6.79%로 높이면 3회 중앙 처리량은 1.69% 증가했다.
- resident decode TPOT 평균은 2.11%, 전체 E2E p95는 1.73% 감소했다.
- 같은 변화가 late-prefill TTFT 평균을 16.27% 증가시켰다.
- 최대-overlap Current는 fresh vLLM과 처리량 -0.30%, TPOT 평균 +0.03%, 전체 E2E p95 +0.08%로 사실상 같은 범위였다.
- Current peak GPU memory는 7,875 MiB로 vLLM의 9,039 MiB보다 1,164 MiB, 12.88% 적었다.

따라서 다음 연구 문제는 overlap을 켤지 말지가 아니다. **관측된 opportunity 중 어떤 것을 선택해야 decode drain 이득을 얻으면서 protected prefill slack을 침범하지 않는가**가 핵심이다.

## 이번 구현

### Opt-in overlap envelope controller

`TRT_EDGELLM_EXPERIMENTAL_OVERLAP_PERCENT`를 추가했다.

- `-1`: production 동작. 기존 cost, interference, debt, SLO policy를 그대로 사용한다.
- `0..100`: 연구용 envelope. 실행 가능한 P+D opportunity stream에서 지정한 장기 비율만큼 deterministic하게 선택한다.
- `0`도 같은 experimental candidate mechanism을 통과하되 overlap action만 선택하지 않는다. 따라서 0%와 100%는 같은 hard-feasibility 경계 위에서 비교된다.
- accumulator 기반이므로 난수 seed와 무관하고 동일 opportunity 순서에서 선택이 재현된다.

Experimental mode에서는 production의 predicted interference, overlap debt, consecutive-overlap soft guard를 건너뛴다. 다음 hard constraint는 그대로 유지한다.

- request dependency
- execution-context ownership
- TensorRT binding/profile shape
- memory feasibility
- 이미 outstanding인 context와의 action compatibility

즉 이 옵션은 안전 조건을 제거한 강제 동시 실행이 아니다. production policy가 아직 선택하지 않는 **feasible action-space envelope**를 노출한다.

### Initial 및 residual P+D 경로

두 dispatch 경로에 동일한 controller를 연결했다.

1. `PhaseQueueScheduler`가 처음부터 P와 D를 함께 선택하는 경로
2. `PhaseThreeCoordinator`가 이미 outstanding인 phase에 residual P 또는 D를 추가하는 경로

각 경로는 독립 accumulator를 가진다. 다음 telemetry도 추가했다.

- `global_experimental_overlap_opportunities`
- `global_experimental_overlap_selections`
- `vision_global_experimental_residual_prefill_decode_opportunities`
- `vision_global_experimental_residual_prefill_decode_selections`

### Action fidelity

모든 sweep에서 planned action과 실제 stream interval을 함께 검사했다.

- missed planned overlap: 0
- unplanned overlap: 0

따라서 아래 결과는 `P`를 선택했는데 다음 poll에서 우연히 `D`가 겹친 결과가 아니다. selector가 선택한 P+D outstanding set과 실제 CUDA event interval이 일치한다.

## 측정 정의

Overlap에는 서로 다른 세 숫자가 존재한다.

1. **Opportunity count**: 같은 decision boundary에서 P와 D가 모두 ready이고 hard-feasible한 횟수
2. **Selection count**: 그 opportunity에서 P+D action을 선택한 횟수
3. **Actual overlap duration**: P와 D의 CUDA event interval 교집합

`requested 100%`는 opportunity를 전부 선택한다는 뜻이지 trace 전체 GPU 시간을 100% 동시 실행한다는 뜻이 아니다. 예를 들어 maximum-exposure trace에서 16개 opportunity를 전부 선택해도 실제 P+D active overlap은 전체 active span의 6.79%였다. 나머지 시간에는 P가 없거나 D가 없으므로 겹칠 작업 자체가 없다.

## Workload 설계

재현 가능한 trace 생성기를 `benchmarks/phase_serving/build_overlap_opportunity_trace.py`에 추가했다. 두 request class를 만든다.

- `resident_decode`: 먼저 도착해 장시간 D queue를 유지한다.
- `late_prefill`: decode가 이미 진행되는 동안 wave 또는 burst로 도착한다.

Maximum-exposure trace는 다음과 같다.

- resident decode: 8 requests, output 256 tokens
- late prefill: 48 requests, prompt 약 64 tokens, output 1 token
- late arrival: 300 ms
- max prefill batch: 2
- fixed prefill chunk: 128
- P formation window: 5 ms
- 총 요청: 56, 생성 token: 2,096

Output 1인 late request를 사용한 이유는 prefill 완료 후 D cohort에 장기간 합류하여 측정을 오염시키지 않기 위해서다. P2 formation window는 HTTP thread arrival staggering 때문에 같은 burst가 P1로 갈라지는 것을 줄인다.

## 실험 1: KV headroom이 opportunity를 결정한다

처음 사용한 D64/output384 trace는 resident request가 page pool 256개를 모두 예약했다.

```text
64 resident requests x 4 pages = 256 pages
late P admission headroom       =   0 pages
```

그 결과 late request는 약 3초간 admission에서 기다렸고, P와 D가 동시에 ready인 상태가 만들어지지 않았다. 0%와 100%는 각각 약 6,159 및 6,114 token/s였지만 이는 overlap policy의 비교가 아니다.

이 결과는 중요한 구조적 제약을 보여준다.

```text
request arrival
    -> memory/page admission
    -> P ready
    -> P+D opportunity
```

독립 context가 있어도 stable KV ownership이 pool을 모두 예약하면 scheduler는 overlap 후보를 생성할 수 없다. Overlap opportunity는 compute readiness뿐 아니라 memory admission의 함수다.

## 실험 2: D64 headroom curve

Resident output을 256으로 줄여 64개 page headroom을 남겼다. 단일 반복 결과는 다음과 같다.

| Requested | Selected/opportunity | Actual P+D span | token/s | Late E2E mean | Late E2E p95 | Resident D TPOT mean |
|---:|---:|---:|---:|---:|---:|---:|
| 0% | 0/41 | 0.07% | 6030.34 | 256.46 ms | 342.56 ms | 10.450 ms |
| 25% | 9/38 | 3.55% | 5957.96 | 257.56 ms | 349.02 ms | 10.696 ms |
| 50% | 17/35 | 6.69% | 5997.13 | 234.50 ms | 290.59 ms | 10.561 ms |
| 75% | 23/31 | 8.83% | 6010.88 | 243.22 ms | 311.96 ms | 10.526 ms |
| 100% | 30/30 | 11.22% | 5998.07 | 217.89 ms | 267.00 ms | 10.547 ms |

D64에서는 100% overlap이 late request completion을 평균 15.0%, p95 22.1% 줄였지만 전체 처리량은 0%보다 0.54% 낮고 resident D TPOT는 0.92% 높았다. 이미 큰 decode batch는 단독 실행 효율이 높다. 이 경우 overlap은 총처리량 향상보다 짧은 downstream work의 latency를 줄이는 service redistribution으로 작동한다.

## 실험 3: Maximum-exposure curve

Decode batch를 D8로 낮추고 P2 burst를 형성해 더 많은 unused execution capacity가 존재하는 구간을 만들었다.

### 단일 반복 전체 곡선

| Requested | Selected/opportunity | Actual P+D span | token/s | TPOT mean | E2E p95 | Late TTFT mean |
|---:|---:|---:|---:|---:|---:|---:|
| 0% | 0/27 | 0.00% | 1062.26 | 7.585 ms | 1970.07 ms | 140.03 ms |
| 25% | 6/24 | 2.52% | 1064.08 | 7.567 ms | 1965.89 ms | 146.93 ms |
| 50% | 9/18 | 3.79% | 1071.31 | 7.494 ms | 1947.57 ms | 152.33 ms |
| 75% | 12/16 | 5.14% | 1078.79 | 7.458 ms | 1939.39 ms | 157.87 ms |
| 100% | 16/16 | 6.79% | 1077.53 | 7.426 ms | 1940.14 ms | 163.96 ms |

Opportunity denominator가 percentage마다 다른 것은 오류가 아니다. 선택한 overlap이 P queue를 더 빨리 drain하여 이후 decision boundary 자체를 바꾸기 때문이다. 각 percentage는 자신이 관측한 opportunity stream에 대해 정확한 장기 선택률을 적용한다.

### 3회 반복 안정성 확인

| Setting | token/s | TTFT mean / p95 | TPOT mean / p95 | E2E mean / p95 | Late TTFT mean / p95 |
|---|---:|---:|---:|---:|---:|
| Current 0% | 1059.01 | 125.70 / 248.39 ms | 7.605 / 7.667 ms | 402.85 / 1975.30 ms | 140.71 / 249.03 ms |
| Current 75% | 1072.56 | 142.31 / 284.20 ms | 7.480 / 7.542 ms | 414.64 / 1949.02 ms | 159.71 / 284.28 ms |
| Current 100% | 1076.86 | 146.43 / 289.93 ms | 7.445 / 7.511 ms | 419.37 / 1941.14 ms | 163.60 / 292.39 ms |

100%를 0%와 비교하면 다음과 같다.

- throughput: **+1.69%**
- decode TPOT mean: **-2.11%**
- E2E p95: **-1.73%**
- resident-decode E2E mean: **-1.85%**
- late-prefill TTFT mean: **+16.27%**
- overall E2E mean: **+4.10%**

즉 GPU drain과 tail completion은 개선되지만, overlap interference를 late-prefill first-token path가 지불한다. 평균 하나만 보면 이 구조를 잘못 평가하게 된다.

## Fresh vLLM 비교

동일 trace, 동일 56 max in-flight, ignore-EOS, 실제 HTTP arrival로 vLLM을 fresh 3회 실행했다.

| Metric | Current 0% | Current 100% | Fresh vLLM |
|---|---:|---:|---:|
| token/s | 1059.01 | 1076.86 | 1080.09 |
| TTFT mean | 125.70 ms | 146.43 ms | 164.55 ms |
| TTFT median | 120.61 ms | 140.54 ms | 171.71 ms |
| TTFT p95 | 248.39 ms | 289.93 ms | 234.91 ms |
| TPOT mean | 7.605 ms | 7.445 ms | 7.443 ms |
| TPOT p95 | 7.667 ms | 7.511 ms | 7.445 ms |
| E2E mean | 402.85 ms | 419.37 ms | 436.28 ms |
| E2E median | 161.64 ms | 188.12 ms | 196.13 ms |
| E2E p95 | 1975.30 ms | 1941.14 ms | 1939.68 ms |
| Peak GPU memory | 7,875 MiB | 7,875 MiB | 9,039 MiB |

Current 100%는 vLLM 대비 다음 수준이다.

- throughput: -0.30%
- TPOT mean: +0.03%
- E2E p95: +0.08%
- E2E mean: -3.88%
- peak memory: -1,164 MiB, -12.88%

Late-prefill TTFT는 평균과 tail의 방향이 다르다. 세 run의 request-class 중앙값 기준으로 Current 100% 평균은 약 11.6% 짧지만 p95는 약 24.3% 길다. Current의 P2 FIFO burst가 앞쪽 request는 빨리 끝내지만 뒤쪽 wave에 긴 formation/drain tail을 만든다.

## 실패 실험이 알려준 것

### HTTP arrival staggering

P2를 의도한 작은 wave가 formation wait 없이 P1로 분리되었다. overlap percentage를 바꾸면 dispatch timing도 바뀌어 P1/P2 shape 분포가 달라졌다. 이 상태의 비교는 policy-only A/B가 아니다. 동일한 5 ms P formation window와 output 1을 적용해 candidate formation을 고정했다.

### Production soft guard에 가려진 opportunity

처음 controller는 production overlap candidate가 생성된 뒤에만 percentage를 적용했다. 24개 P wave 중 실제 opportunity가 2~8개뿐이라 maximum potential을 볼 수 없었다. Experimental envelope가 soft guard 이전의 executable P+D shape를 hard-feasibility filter로 직접 전달하도록 바꾼 이유다.

### 큰 batch만으로 overlap 가치를 일반화할 수 없다

D64에서는 throughput 이득이 없었고 D8에서는 있었다. 이는 workload label용 heuristic이 아니라 현재 action shape의 isolated efficiency와 interference 차이다. Scheduler 입력은 `vision-heavy`, `throughput` 같은 profile이 아니라 다음 observable state여야 한다.

- ready P/D rows와 chunk/context shape
- current outstanding contexts
- measured isolated and overlap makespan
- protected request slack
- memory/page ownership horizon

## 연구 해석

Independent context의 가치는 항상 동시 실행하는 데 있지 않다. 실행 가능한 action space를 다음처럼 확장하는 데 있다.

```text
serial P
serial D
P+D overlap
bounded WAIT
```

이번 결과는 그 확장된 공간 안에 실제 성능 이득이 있음을 보였다. 동시에 한 workload에서 가장 높은 throughput point가 모든 request class에 최적인 것은 아님도 보였다.

따라서 production selector는 workload별 fine-tuning이나 static mode를 사용하지 않고 다음 lexicographic rule을 따라야 한다.

1. dependency, ownership, TRT shape, memory hard feasibility
2. protected request의 robust slack 위반 최소화
3. deadline-equivalent 후보 사이에서 measured service compression 최대화
4. batch gain이 wait cost보다 클 때만 bounded WAIT

오버랩의 online value는 다음처럼 측정할 수 있다.

```text
compression(P,D) = (isolated_P + isolated_D) / overlap_makespan
```

하지만 이 값만 최대화하면 maximum-exposure 100%처럼 P TTFT tail을 희생한다. 각 observation에는 어떤 protected request가 얼마나 지연됐는지도 함께 귀속해야 한다.

## 다음 단계

1. `action key -> overlap makespan, P slowdown, D slowdown`을 process-local runtime observation으로 축적한다.
2. overlap opportunity마다 isolated serial counterfactual과 observed makespan을 비교한다.
3. first-token critical path의 robust slack보다 예상 P slowdown이 작은 후보만 overlap eligible로 만든다.
4. D8처럼 slack이 넉넉하고 compression이 큰 shape는 적극 선택하고, D64처럼 throughput gain이 없는 shape는 completion urgency가 있을 때만 선택한다.
5. 같은 12-workload gate에서 production `-1`과 새 selector를 비교한다. Experimental 0/25/50/75/100은 연구 envelope와 regression 진단용으로만 유지한다.

이 접근은 workload마다 percentage를 고르는 fine-tuning이 아니다. Percentage sweep은 action-space의 경계를 측정하는 실험 도구이고, production은 현재 ready state, online CUDA observation, request slack만 사용하여 그 경계 안의 action을 선택한다.

## 결과 파일

- 원시 요약표: `benchmarks/phase_serving/results/overlap-opportunity-envelope-20260830.csv`
- trace 생성기: `benchmarks/phase_serving/build_overlap_opportunity_trace.py`
- 이전 kernel-level controlled sweep: `notes/183-controlled-phase-overlap-sweep-20260830.md`
