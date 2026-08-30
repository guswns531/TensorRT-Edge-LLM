# Sampling completion-ready handoff 실험

## 결론

48.8 offered req/s, admission 64 경계에서 sampling CUDA event polling을 CUDA stream host callback 기반
completion-ready handoff로 교체해 5회 반복했다. 구현은 correctness를 지켰지만 성능 승격 기준을
통과하지 못했다.

- 5/5 run에서 288/288 request와 24,960/24,960 output token을 완료했다.
- 5회 token hash는 모두 기존 Current와 동일한 `f51d...75ca`였다.
- callback handoff의 raw throughput point estimate는 기존 event polling보다 `0.82%` 낮았다.
- joint-SLO goodput point estimate는 `6.43%` 낮고 TTFT p95는 `8.22%` 길었다.
- CUDA callback이 signal을 publish한 뒤 serving loop가 ticket을 소비하기까지 평균 `1.249 ms`, run별
  최대값 평균 `7.112 ms`가 걸렸다.
- raw throughput과 goodput의 95% confidence interval은 서로 겹친다. 따라서 callback이 통계적으로 더
  느리다고 단정하지는 않지만, 더 빠르다는 증거가 없고 tail point estimate도 나빠 promotion할 근거가
  없다.

실험용 callback 구현은 결과 확인 후 제거했다. Current production source는 기존 non-blocking
`cudaEventQuery()` 경로를 유지하며, 불필요한 opt-in/env/config 분기를 남기지 않았다. 이 문서와 raw run만
negative result로 보존한다.

## 검증한 구현

실험 구현은 workload, request class, batch size를 보지 않는 공통 runtime mechanism이었다.

```text
sampling stream
  -> top-k sampling
  -> token D2H
  -> CUDA event record
  -> cudaLaunchHostFunc
       -> ticket-local ready timestamp
       -> release-store ready=true

serving thread
  -> acquire-load ready
  -> collect host token
  -> request state commit
  -> enqueue decode-ready row
```

mutex queue를 매 poll에서 drain하면 수백만 회의 hot-loop lock을 추가하므로 사용하지 않았다. 대신 ticket별
atomic signal을 사용했다. callback은 CUDA API를 호출하지 않고 timestamp 기록과 atomic publish만 수행했다.
server는 기존 ticket 순서를 유지하며 ready signal만 소비했다.

다음 계측도 함께 추가해 실험했다.

- callback submit/completion 수
- callback publish -> serving-thread consume 지연 mean/max
- callback timestamp를 sampling-ready timeline에 전달

실험 뒤 위 코드와 telemetry는 모두 제거했다. performance-negative mechanism을 production API에 남기지 않는
것이 현재 프로젝트의 current-only 정리 원칙과 맞는다.

## 실험 계약

- model: `nvidia/Cosmos-Reason2-2B`, FP16, 비양자화
- engine/runtime: Current P8/D64, fixed prefill chunk 128, 80 stable indexed slots
- server admission: 64
- client maximum in-flight: 80
- execution: independent P/D TensorRT contexts와 Global selector
- trace: P5--P7과 동일한 materialized real-request trace, offered 48.8 req/s
- trace SHA-256: `c9d64ed7e84fea87352c9b4acedb309932c8f0e9cae71d527fc2f868ec90849d`
- measured work: run당 288 requests, prompt 25,872 tokens, output 24,960 tokens
- warmup: 64 requests, output 32 tokens
- joint SLO: scheduled TTFT <= 500 ms, TPOT <= 50 ms, scheduled E2E <= 2500 ms
- 반복: 각 구성 5회 fresh backend
- 통계: 5회 산술평균과 Student-t 95% confidence interval half-width, 자유도 4
- Current event poll: Note 189의 동일 cap64 5-run baseline 재사용
- vLLM: workload/contract가 같으므로 Note 187의 fresh 5-run frozen baseline 재사용

## End-to-end 결과

`mean +/- CI95`이고 latency 단위는 ms다.

| Runtime | Raw req/s | SLO pass | Goodput req/s | Scheduled TTFT mean/p95 | TPOT mean/p95 | Scheduled E2E mean/p95 | Peak MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| Current, event poll | 37.865 +/- 0.122 | 75.97 +/- 4.97 pp | 28.769 +/- 1.930 | 240.95 / 618.52 | 15.82 / 17.87 | 1587.27 / 2356.13 | 9081 |
| Current, host handoff | 37.553 +/- 0.248 | 71.67 +/- 3.93 pp | 26.918 +/- 1.643 | 247.15 / 669.36 | 15.94 / 18.10 | 1601.63 / 2388.57 | 9081 |
| vLLM, frozen | 40.908 +/- 0.029 | 100% | 40.908 +/- 0.029 | 51.44 / 84.17 | 13.53 / 17.65 | 1205.87 / 2119.43 | 9039 |

Host handoff의 event-poll 대비 변화는 다음과 같다.

| Metric | 변화 | 판정 |
|---|---:|---|
| Raw request throughput | -0.82% | 개선 없음 |
| Joint-SLO pass | -4.31 percentage points | 개선 없음 |
| Joint-SLO goodput | -6.43% | 개선 없음 |
| Scheduled TTFT mean | +2.57% | 회귀 방향 |
| Scheduled TTFT p95 | +8.22% | 회귀 방향 |
| TPOT mean | +0.72% | 회귀 방향 |
| TPOT p95 | +1.30% | 회귀 방향 |
| Scheduled E2E mean | +0.91% | 회귀 방향 |
| Scheduled E2E p95 | +1.38% | 회귀 방향 |

동일 trace의 vLLM goodput은 host handoff Current보다 `52.0%` 높다. callback notification은 Current-vLLM
saturation 차이를 줄이지 못했다.

## Direct handoff 자체의 계측

| Run | Callback submits/completions | Publish -> consume mean | Max |
|---|---:|---:|---:|
| 1 | 982 / 982 | 1255.468 us | 7534.918 us |
| 2 | 984 / 984 | 1262.449 us | 6995.717 us |
| 3 | 994 / 994 | 1244.343 us | 6948.737 us |
| 4 | 948 / 948 | 1242.274 us | 6787.617 us |
| 5 | 989 / 989 | 1239.733 us | 7292.493 us |
| Mean | 979.4 / 979.4 | 1248.853 us | 7111.896 us |

callback loss는 없었다. 문제는 전달 누락이 아니라 callback publish와 request state commit 사이에 남은 host
decision boundary다.

## 왜 빨라지지 않았는가

### 1. CUDA host callback은 GPU event보다 빠른 notification primitive가 아니다

기존 경로는 serving thread가 이미 hot loop에서 `cudaEventQuery()`를 수행한다. host callback은 같은 stream의
sampling/D2H 뒤에 실행되지만 CUDA driver가 host callback을 schedule하는 별도 경계를 추가한다. callback이
실행됐다는 사실만으로 main serving thread가 즉시 request state를 변경할 수는 없다.

### 2. callback thread에서 state commit을 실행할 수 없다

CUDA host callback 안에서는 CUDA API 호출을 피해야 하고, `IndependentPhaseAsyncServer`, stable KV ownership,
scheduler queue는 single-owner serving thread 계약을 사용한다. callback에서 직접 `processTicket()`을 호출하면
request map, slot release, token callback, decode enqueue에 data race가 생긴다.

따라서 안전한 callback은 atomic readiness까지만 전달할 수 있다. 실제 token collect와 state commit은 다음
serving-thread boundary까지 기다린다. 실측 `1.249 ms`가 바로 이 간격이다.

### 3. notification 최적화와 decision realization 최적화는 다르다

Note 188에서 관측한 sampling residual을 단순히 event polling 비용이라고 해석하면 안 된다. 이번 실험으로
다음이 분리됐다.

```text
GPU sampling completion visibility
        !=
request state commit과 next-D materialization
```

Current의 주 문제는 event API 선택 하나보다 completion을 소비하고 다음 TensorRT D action을 materialize하는
전체 iteration 경로다.

## Promotion 판단

### 유지

- 기존 `cudaEventQuery()` 기반 non-blocking sampling completion
- Note 189의 admission64 기본값
- callback experiment raw runs와 negative result 문서

### 제거

- `TRT_EDGELLM_ENABLE_SAMPLING_READY_HANDOFF`
- server config/API/telemetry의 sampling handoff 분기
- ticket atomic signal과 `cudaLaunchHostFunc` callback

## 다음 권장 단계

다음 단계는 notification primitive를 다시 바꾸는 것이 아니라 D64 iteration realization을 vLLM과 같은 단위로
분해하는 것이다.

1. Current의 `D TensorRT enqueue + sampling + state commit + next D enqueue` cycle을 cohort shape별로 측정한다.
2. 이번 실행에서 decode CUDA graph hit가 여전히 0인 이유를 binding/profile/context-length key 관점에서 찾는다.
3. D64 stable cohort에서 graph replay가 가능한 binding subset과 매 iteration 변하는 KV/page metadata를
   분리한다.
4. graph replay 또는 persistent device-side sampling/token staging 후보를 각각 opt-in A/B한다.
5. cap64 48.8 trace를 5회 반복하고 Note 189 Current 및 frozen vLLM과 비교한다.

이 순서는 workload별 heuristic을 추가하지 않는다. 동일한 completion/launch mechanism을 모든 text/VLM
workload가 공유하게 한다.

## 검증 및 산출물

- C++ scheduler 회귀 테스트: 187/187 passed
- TensorRT 11.0/CUDA 13.3 build: `unitTest`, `llm_phase_context_smoke` passed
- real-request smoke: 1/1 complete, exact token hash 유지
- real-request repeated gate: 5/5 complete, exact token hash 유지
- raw experiment: `.local/transition-aware-20260830/p8-sampling-ready-handoff/`
- 결과 CSV: `benchmarks/phase_serving/results/sampling-ready-handoff-ab-20260830.csv`
