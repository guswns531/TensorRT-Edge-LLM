# E/P/D/Copy activity: 12-workload real-request analysis

## 결론

`E=0001`, `P=0010`, `D=0100`, `C=1000` CUDA-event activity recorder를 Cosmos Reason2-2B의 기존
12개 real-request HTTP workload에 연결했다. warmup과 measured request가 동일한 request ID를 재사용하는
경우까지 분리해 production lifecycle만 분석했다.

결과는 다음과 같다.

1. host scheduler의 planned outstanding mask와 dispatch 시점의 observed outstanding mask 분포는 12/12에서
   정확히 일치했다. 누적 action-fidelity violation도 증가하지 않았다.
2. 실제 CUDA-event timeline에서 P+D overlap은 long-prefill의 한 dispatch, `30.283 ms`뿐이었다. 계획한
   P+D 한 번이 실제 같은 dispatch의 `0110`으로 나타났으므로 그 action의 fidelity는 `1/1`이다.
3. 나머지 11개 workload는 E/P/D overlap이 정확히 `0 ms`였다. independent TensorRT execution context와
   CUDA stream은 존재하지만, 현재 global selector가 measured 구간에서 거의 전부 serial E/P/D action을
   선택한다.
4. burst workload의 active-span 내부 idle은 대체로 `2.8--5.6%`다. 즉 GPU service timeline은 잘 채워져
   있지만 concurrency로 압축된 것이 아니라 E/P/D를 직렬로 이어 붙여 채운 상태다.
5. wave/drain의 idle `69.8%`는 스케줄러의 즉시 dispatch gap이 아니라 trace의 의도적인 arrival wave 사이
   drain 구간이다. 다른 burst workload와 같은 의미로 해석하면 안 된다.
6. Copy bit는 12개 모두 `0`이다. Cosmos visual runner가 retained vision slab을 external output binding으로
   직접 사용하므로 별도 D2D output copy를 만들지 않는 것이 정상이다.
7. CUDA timing event 두 개를 dispatch마다 추가한 진단 실행과 직전 request-attribution 진단 실행의 token/s
   변화 중앙값은 `-0.02%`였다. 단일 실행 노이즈 안이며 production 기본값에서는 recorder가 생성되지 않는다.

따라서 다음 개선은 stream이나 context를 더 만드는 일이 아니다. 현재 warmup에서 관측된 작은 overlap cost
bucket이 measured candidate shape를 충분히 덮지 못해 unknown overlap이 safe fallback으로 제거되는 문제를
해결해야 한다. workload label이나 workload별 knob를 추가하지 않고, bounded safe probe와 canonical cost-key
coverage로 online overlap eligibility를 확보해야 한다.

## 실행 계약

- model: `nvidia/Cosmos-Reason2-2B`, FP16
- engine: `.local/profile-local-prefill-20260827/engine-p128-v1024-kv256-tied`
- CUDA context: 하나를 E/P/D/C가 공유
- TensorRT execution context: E/P/D 독립
- maximum batch: P8 / D64 / E4
- fixed prefill chunk: 128
- stable slots: 80, maximum in-flight: 64
- KV page pool: 256 pages
- HTTP adapter: production IPC async adapter, workers 4
- workload trace, arrival schedule, requested output length: Note 179와 동일
- warmup: 64 requests, maximum output 32
- measured repeat: workload별 1회
- activity opt-in: `TRT_EDGELLM_PHASE_ACTIVITY_PREFIX`
- dispatch/timeline log: `TRT_EDGELLM_EMIT_PHASE_METRICS=1`

이번 실행은 activity attribution을 위한 1회 진단이다. 성능 promotion headline은 비계측 3회 결과를 계속
사용한다.

## 구현한 분석 경로

`benchmarks/phase_serving/analyze_phase_activity.py`를 추가했다.

```text
gateway PHASE_TIMELINE
        |
        | completion으로 lifecycle 분리
        | request ID마다 마지막 completed lifecycle 선택
        v
 measured P/D dispatch IDs --------------------+
                                                |
runtime activity-intervals.csv                  |
        |                                       |
        +-- P/D correlation = dispatch ID ------+
        |
        +-- E/C correlation = encoder batch 대표 request ID
        |     마지막 warmup P/D 종료 이후의 E/C만 선택
        v
 measured E/P/D/C intervals
        |
        | CUDA epoch timestamp boundary sweep
        v
 measured-segments.csv: 0000 ... 1111
        |
        +-- duty / idle / overlap
        +-- planned P+D dispatch와 same-dispatch CUDA overlap 교집합
        +-- planned/observed outstanding mask 분포
        v
 activity-summary.json
```

단순히 같은 request ID의 마지막 E interval을 고르면 안 된다. encoder batch의 correlation ID는 batch의 첫
request인데, measured batch에서 첫 row가 아니었던 ID의 마지막 E가 warmup에 남을 수 있다. 분석기는 다음
경계를 사용한다.

```text
last non-measured P/D end
          |
          v
----------|----- measured E/C ----- measured P/D -------------------|
                                                                  last measured P/D end
```

이 경계로 warmup E/C를 제거한 뒤 active-span이 client duration과 같은 범위로 수렴하는지 검증했다.

## 12-workload 결과

`E/P/D/idle`은 measured CUDA active-span에 대한 wall-time 비율이다. overlap이 있으면 phase duty 합은
100%를 넘을 수 있다. latency 단위는 ms다.

| workload | req | tok/s | TTFT mean/p95 | TPOT mean/p95 | E2E mean/p95 | E% | P% | D% | idle% | EPD overlap% | planned/actual P+D |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| short | 48 | 2,450.2 | 85.4 / 169.3 | 13.9 / 27.2 | 336.8 / 418.8 | 0.0 | 43.7 | 52.6 | 3.7 | 0.00 | 0 / 0 |
| balanced | 288 | 4,479.7 | 66.5 / 164.6 | 12.3 / 13.5 | 1,115.2 / 1,733.6 | 0.0 | 28.0 | 66.7 | 5.3 | 0.00 | 0 / 0 |
| decode-heavy | 288 | 5,116.4 | 66.0 / 180.6 | 10.9 / 11.4 | 2,878.8 / 4,391.8 | 0.0 | 10.9 | 83.4 | 5.6 | 0.00 | 0 / 0 |
| long-prefill | 288 | 1,170.8 | 2,074.5 / 2,975.3 | 27.4 / 31.9 | 4,427.8 / 6,392.4 | 0.0 | 59.3 | 38.1 | 2.8 | 0.14 | 1 / 1 |
| bimodal | 288 | 1,904.6 | 1,896.0 / 3,554.3 | 18.1 / 29.8 | 4,375.0 / 8,684.3 | 0.0 | 32.0 | 64.6 | 3.4 | 0.00 | 0 / 0 |
| text-heavy | 64 | 1,913.1 | 330.1 / 1,121.8 | 25.7 / 38.6 | 1,660.3 / 1,762.0 | 23.7 | 39.5 | 32.0 | 4.8 | 0.00 | 0 / 0 |
| mixed | 64 | 1,120.9 | 728.8 / 2,161.6 | 32.2 / 42.1 | 2,256.1 / 2,541.8 | 32.3 | 38.7 | 24.3 | 4.7 | 0.00 | 0 / 0 |
| vision-heavy | 64 | 681.6 | 1,381.5 / 3,159.7 | 29.3 / 37.7 | 2,536.5 / 3,550.2 | 34.9 | 37.5 | 23.3 | 4.4 | 0.00 | 0 / 0 |
| poisson | 64 | 1,947.5 | 197.6 / 755.9 | 21.9 / 40.5 | 1,597.7 / 2,044.7 | 17.3 | 31.9 | 46.4 | 4.5 | 0.00 | 0 / 0 |
| wave/drain | 20 | 97.4 | 217.1 / 328.3 | 9.7 / 11.7 | 516.7 / 541.2 | 8.0 | 7.4 | 14.8 | 69.8 | 0.00 | 0 / 0 |
| multi-image | 5 | 299.8 | 219.6 / 327.0 | 9.6 / 13.1 | 516.0 / 533.4 | 24.4 | 23.3 | 49.4 | 2.8 | 0.00 | 0 / 0 |
| late-vision D24 | 32 | 2,490.6 | 115.2 / 457.6 | 9.5 / 9.5 | 1,472.9 / 1,851.4 | 9.3 | 13.7 | 71.7 | 5.4 | 0.00 | 0 / 0 |

모든 workload가 requested output token을 전부 생성했다. 11개 workload는 직전 실행의 token hash와 exact
match했다. vision-heavy 첫 실행만 다른 hash를 냈으며, 동일 설정 재실행은 직전 canonical hash
`cfc7a828...`와 다시 exact match했다. measured production request는 기본 12개 실행 `1,473`개와
vision-heavy 재검증 64개다. 이 한 번의 divergence는 아래 correctness 절에서 별도로 다룬다.

## workload별 activity 해석

### Short

P와 D가 `43.7/52.6%`로 거의 전체 active span을 직렬 점유한다. 내부 gap은 `3.7%`뿐이다. P+D overlap을
만들어도 줄일 수 있는 상한은 P와 D의 단순 합이 아니라 contention을 포함한 makespan gain이므로, short에서
무조건 overlap을 강제할 근거는 없다. 기존 비계측 Current는 cached vLLM보다 throughput `+26.8%`, E2E p95
`-19.1%`이므로 low-risk probe 외의 short 전용 정책은 불필요하다.

### Balanced

D가 `66.7%`, P가 `28.0%`, idle이 `5.3%`다. 기존 request attribution에서 평균 D batch는 61.1로 D64에
가깝다. 따라서 D formation 부족이 아니라 P와 D가 서로 양보하며 직렬 실행되는 구조다. cached vLLM 대비
비계측 Current는 throughput `+5.8%`, E2E p95 `-4.5%`다. overlap을 추가할 때 이 margin과 TPOT를
보호해야 한다.

### Decode-heavy

D duty `83.4%`이고 idle `5.6%`다. D64 cohort가 이미 잘 형성된 상태여서 P+D overlap이 D kernel을 조금만
느리게 해도 TPOT tail에 직접 나타난다. cached vLLM 대비 throughput `+6.4%`, TPOT p95 `-4.2%`라는 현재
장점을 deadline guard로 보존해야 한다.

### Long-prefill

유일한 actual overlap workload다. P `59.3%`, D `38.1%`이며 dispatch 407에서 계획한 P+D가 `30.283 ms`
동안 실제 `0110`으로 실행됐다. planned/actual은 `1/1`이고 unplanned same-dispatch overlap은 0이다. 다만
전체 active span의 `0.14%`라 성능을 좌우할 양은 아니다. TTFT 주원인은 Note 179에서 확인한 KV-page-gated
admission이며 overlap만 늘려 해결할 수 없다.

### Bimodal

P/D가 `32.0/64.6%`, idle `3.4%`지만 overlap은 없다. 평균 GPU service density는 높고 TTFT tail은 KV page
admission과 short/long request ordering이 지배한다. cached vLLM보다 throughput/E2E는 좋지만 TTFT mean/p95가
`+19.7/+54.8%` 느린 기존 약점은 activity idle이 아니라 admission/fairness 문제라는 결론이 강화된다.

### Text-heavy VLM

E/P/D가 `23.7/39.5/32.0%`로 합계 `95.2%`이며 실제 overlap은 없다. 세 독립 context가 순차적으로 GPU를
채운다. cached vLLM 대비 비계측 Current는 throughput `+20.3%`, E2E p95 `-15.9%`이므로 E overlap probe가
text D tail을 악화시키지 않아야 한다.

### Mixed VLM

E/P/D가 `32.3/38.7/24.3%`, idle `4.7%`다. 세 phase 모두 충분한 wall-time을 갖기 때문에 이론적으로
overlap action의 후보가 가장 많지만 measured 실행에서는 모두 serial이었다. cached vLLM 대비 비계측 Current는
throughput `+22.5%`, E2E p95 `-19.5%`다. bounded E+D/P+D probe의 첫 positive-control workload로 적합하다.

### Vision-heavy

E/P가 `34.9/37.5%`로 대부분이고 D는 `23.3%`다. idle은 `4.4%`뿐이라 vision-heavy의 느린 TTFT를
`GPU가 놀아서`라고 설명할 수 없다. E queue와 직렬 E->P critical path를 줄여야 한다. E+P는 둘 다
compute-heavy라 RTX 3080에서 interference가 클 수 있으므로 첫 구현은 E+D부터 검증한다.

### Poisson

E/P/D가 `17.3/31.9/46.4%`, idle `4.5%`다. online arrival에서도 active-span 내부 service gap은 작지만
concurrency는 없다. arrival가 분산된 workload이므로 WAIT와 probe가 요청 slack을 침범하는지 검증하기 좋다.

### Wave/drain

idle `69.8%`는 의도적인 wave 간 arrival 공백이다. active phase의 총합이 작다고 scheduler가 underfill한
것이 아니다. 이 workload에서 overlap보다 중요한 것은 각 wave의 oldest vision TTFT이며, idle period를
다음 batch를 위한 WAIT 이득으로 오인하면 안 된다.

### Multi-image

요청 5개의 작은 cohort다. E/P/D가 `24.4/23.3/49.4%`, idle `2.8%`이며 overlap은 없다. small-D refill과
E first-token critical-path guard를 검증하기 적합하다. 단일 실행 variance가 크므로 성능 promotion은 3회
결과를 사용해야 한다.

### Late-vision D24

D duty `71.7%`, E/P `9.3/13.7%`, idle `5.4%`다. 이미 실행 중인 D24에 late E가 도착하지만 현재 selector는
E+D를 고르지 않았다. contention placement를 고립시키는 trace이므로 첫 E+D safe probe의 핵심 gate다.

## planned action과 actual execution

현재 결과는 두 fidelity 층을 구분한다.

```text
selector action
    |
    v
planned outstanding mask -- coordinator/worker --> observed outstanding mask
                                                |
                                                v
                                      CUDA-event activity mask
```

- 첫 번째 층은 12/12에서 일치한다.
- 두 번째 층에서 계획된 P+D 한 번은 실제 `0110`으로 나타났다.
- serial로 계획된 workload에서 unplanned E/P/D overlap은 없었다.

따라서 이전에 우려했던 `choose(E)` 직후 별도 poll이 D를 enqueue해 암묵 E+D가 생기는 action-fidelity 문제는
이번 current 구현/실행에서는 관측되지 않았다. 반대로 문제는 너무 보수적이라는 것이다.

## 왜 overlap이 거의 없는가

현재 global scheduler의 안전 규칙은 다음과 같다.

```text
direct/eligible overlap cost가 있음
        -> deadline/efficiency 비교 후보

cost가 unknown
        -> calibration probe 또는 safe probe만 후보

probe도 불가
        -> serial fallback
```

각 컨테이너는 runtime cost model을 빈 상태로 시작한다. 64-request calibration warmup이 일부 canonical P+D,
E+P cost를 수집하지만 measured workload의 batch/context/chunk key를 모두 덮지 못한다. 실행 환경은 일반
aggressive exploration을 막기 위해 `TRT_EDGELLM_GLOBAL_SAFE_PROBE_SLACK_MULTIPLIER=0.0`을 사용한다. 따라서
미지의 overlap shape는 measured 구간에서 serial fallback으로 수렴한다.

이 동작은 correctness 측면에서는 안전하지만 independent context의 성능 가능성을 거의 사용하지 못한다.
해결은 workload별 static cost table이나 profile이 아니다.

## Copy stream 해석

모든 workload에서 C=0이다.

```text
현재 Cosmos direct binding

encoder stream:  visual infer -> retained vision slab
                                      |
                                      +-> P consumes the same storage

별도 D2D copy 없음 -> C bit 0
```

fallback runner에서는 다음이 이미 구현돼 있다.

```text
E stream: infer -> encoderDone event
                         |
C stream:             wait -> D2D copy -> ready event
                                                  |
P stream:                                      consume
```

따라서 C=0은 계측 누락이 아니라 memory operation을 제거한 최적 경로의 증거다. KV offload, staging upload,
별도 compaction copy를 추가할 때만 C bit가 나타나야 한다.

## 진단 오버헤드

이번 activity run과 직전 Note 179 request-attribution run은 동일 engine/trace/HTTP 계약을 사용한다. token/s
변화 중앙값은 `-0.02%`다.

| workload | activity run token/s vs Note 179 |
|---|---:|
| short | -0.6% |
| balanced | +0.4% |
| decode-heavy | -0.8% |
| long-prefill | +2.7% |
| bimodal | +1.6% |
| text-heavy | -0.2% |
| mixed | +0.7% |
| vision-heavy | -0.2% |
| poisson | -0.6% |
| wave/drain | -0.4% |
| multi-image | +7.7% |
| late-vision | +0.1% |

multi-image는 요청이 5개뿐이라 formation 경계의 1회 variance가 크다. 이 표로 속도 향상을 주장하지 않는다.
systematic regression 신호가 없다는 smoke evidence로만 사용한다. production default에서는 activity recorder가
disabled라 CUDA timing event 비용이 없다.

## vLLM 비교의 위치

model, trace SHA, arrival, output contract가 바뀌지 않았으므로 vLLM은 다시 실행하지 않고 Note 179의 fresh
3회 cache를 유지한다. 비계측 Current는 cached fresh vLLM 대비 다음 상태다.

- throughput: 12/12 우세
- E2E mean/p95: 12/12 우세
- 남은 cross-system 약점: long-prefill TTFT mean, bimodal TTFT mean/p95
- 원인: activity idle이나 context concurrency 부족 하나가 아니라 KV-page-gated admission과 fairness

이번 activity 결과는 vLLM 대비 headline을 교체하지 않는다. 다만 Current가 E/P/D independent context를
보유하면서도 현재는 거의 직렬로 사용한다는 새로운 최적화 여지를 확인한다.

## 다음 구현 순서

### 1. Canonical overlap coverage telemetry

후보를 제거할 때 다음 이유를 dispatch aggregate로 남긴다.

```text
P+D opportunity
  overlap cost eligible
  insufficient samples
  canonical key mismatch
  slack reject
  memory reject
  selected serial candidate was better
```

E+D와 E+P도 같은 rejection taxonomy를 사용한다. workload 이름은 입력하지 않는다.

### 2. Bounded profile-free safe probe

다음 조건을 모두 만족할 때 unknown overlap을 제한적으로 한 번 실행한다.

```text
oldest protected slack
  > robust serial critical path + probe guard

memory horizon safe
no context already in flight twice
small canonical action key
same key probe budget not exhausted
```

TTL이나 외부 registry는 사용하지 않는다. process lifetime 동안 canonical bucket의 direct observations만
축적한다.

### 3. 우선 E+D, 이후 P+D

- mixed: E+D positive control
- late-vision D24: D TPOT interference control
- vision-heavy: E critical-path benefit
- balanced/decode-heavy: P+D가 기존 D advantage를 훼손하지 않는지 guard
- long-prefill: P+D sample은 있지만 admission 문제와 분리

E+P는 두 heavy phase의 interference가 클 가능성이 있어 그 뒤에 검증한다.

### 4. Activity-based promotion gate

각 workload에서 다음 세 invariant를 함께 본다.

```text
planned action == observed outstanding phase set
planned overlap dispatch has the corresponding non-zero CUDA mask
overlap gain > wait/interference cost without TTFT/TPOT/E2E SLO regression
```

overlap ratio 자체를 목표로 하지 않는다. overlap이 0이어도 serial이 더 빠르면 정상이고, overlap이 많아도
tail을 악화시키면 reject한다.

## 산출물과 검증

Raw artifact:

```text
.local/phase-activity-12x1-20260830/<workload>/
├── activity-intervals.csv
├── activity-segments.csv
├── activity-summary.csv
├── bench/
│   ├── aggregate.json
│   └── run-001/gateway.log
└── analysis/
    ├── measured-intervals.csv
    ├── measured-segments.csv
    └── activity-summary.json
```

vision-heavy deterministic 재검증은 같은 root의 `vision-heavy-r2/`에 있다.

검증 결과:

- 12/12 real-request trace와 vision-heavy 재검증 완료
- 기본 suite 1,473/1,473와 재검증 64/64 measured lifecycle 완료
- 모든 requested output token 완료
- 11/12 첫 실행 token hash exact match
- vision-heavy 첫 실행은 hash divergence, 동일 설정 재실행은 canonical hash exact match
- Python activity/timeline analyzer tests: 9/9 pass
- synthetic four-way `1111`, internal `0000`, missed planned overlap, warmup lifecycle filtering 검증
- 이전 C++ activity recorder/worker GPU tests: 8/8 pass
- 이전 최종 C++ build와 pre-commit 통과 상태 유지

이 계측은 SM occupancy가 아니라 E/P/D/C stream service interval이다. 실제 SM utilization과 copy-engine
bandwidth가 필요하면 Nsight Systems/CUPTI counter를 별도로 결합해야 한다.

### Vision-heavy hash divergence 해석

첫 vision-heavy activity run과 deterministic 재검증은 모두 다음을 만족했다.

- 64/64 request 완료
- 2,464/2,464 output token 완료
- trace SHA 동일
- planned/observed outstanding mask 일치
- E/P/D overlap 0, invalid ownership/fidelity telemetry 0

재검증 hash는 Note 179의 canonical hash와 일치했다. 따라서 stable ownership corruption이나 overlap action
오류의 지속적 증거는 아니다. 다만 동일 serial action 계열에서도 batch row formation/order 또는 FP16 수치
경계가 한 실행의 greedy branch를 바꿀 수 있다는 기존 determinism 문제는 남는다. activity 기능의 promotion과
별개로 canonical row ordering 및 per-request token agreement artifact가 필요하다. 전체-run SHA 하나만
저장하면 어느 request/token에서 갈라졌는지 역추적할 수 없으므로 향후 client harness가 request별 hash를
보존해야 한다.
