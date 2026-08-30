# Capacity-safe D64 cohort와 admission 64/72/80 검증

## 결론

48.8 offered req/s saturation 경계에서 발견한 `active requests > max decode batch` 오류를 수정했다.
이제 80 stable slot과 D64 engine을 함께 사용해도 active 65/72/80에서 invariant abort 없이 요청을
완료하며, 5회 반복 모두 기존과 동일한 exact token hash를 냈다.

하지만 더 큰 admission은 성능 향상이 아니었다.

- admission 64가 raw throughput `37.865 req/s`, joint-SLO goodput `28.769 req/s`로 세 Current 구성 중 가장
  좋았다.
- admission 72는 goodput `16.906 req/s`, admission 80은 `18.900 req/s`로 admission 64보다 각각
  `41.24%`, `34.30%` 낮았다.
- admission 80은 send-relative TTFT는 매우 짧지만 긴 request lifetime 때문에 client in-flight queue가
  커지고 scheduled TTFT와 E2E가 다시 악화됐다.
- cap 72/80은 cap 64보다 decode prepare가 많고 stable page mapping 재사용이 크게 줄었다. 빈 KV slot에
  request를 더 넣는 것과 D64 cohort를 더 효율적으로 실행하는 것은 같은 문제가 아니다.
- 같은 trace의 frozen vLLM은 goodput `40.908 req/s`이고 Current cap64보다 `42.20%` 높다. overflow crash가
  saturation gap의 원인은 아니었고, cap을 무조건 늘리는 방식으로는 격차를 줄일 수 없다.

따라서 기본 decode-aligned admission 64는 유지한다. 이번 수정은 `active > D batch`를 허용해야 하는 다른
동적 admission/memory 상태의 correctness substrate로 승격하지만, cap 72/80을 기본 성능 정책으로
승격하지 않는다.

## 실패 원인과 수정한 실행 계약

기존 scheduler에는 두 종류의 decode row가 섞여 있었다.

```text
decode queue
  |-- 현재 persistent D64 cohort에 속하고 지금 dispatch 가능한 row
  `-- stable slot은 가졌지만 D64 cohort 밖에서 기다리는 overflow row
```

기존 `snapshot()`과 `popBatch()`의 initial count는 두 집합을 모두 eligible decode work로 셌다. 하지만 실제
selection loop는 persistent cohort ID만 허용했다. active 65에서 snapshot/count는 decode work가 있다고
판단해 D dispatch를 선택했지만 cohort와 교집합인 row가 부족해 다음 invariant에서 종료됐다.

```text
Eligible decode work disappeared during batch selection
```

수정 후 계약은 다음과 같다.

1. `PhaseQueueSnapshot::decodeQueued`는 queue 전체가 아니라 `decodeCandidateRows()`가 반환하는 실제
   cohort-dispatchable row 수다.
2. `popBatch()`는 cohort를 구성한 뒤 `eligible && cohort member`의 실제 교집합으로 selection count를 다시
   계산한다.
3. 현재 cohort row가 모두 sampling/in-flight 상태이면 overflow row를 잘못 dispatch하지 않는다.
4. 알려진 sampling completion event가 있으면 Global scheduler는 concrete `WAIT(event)`를 기록하고 해당
   cohort가 다시 ready가 될 때까지 기다린다.
5. overflow request의 stable KV ownership은 그대로 유지한다. KV tensor compaction이나 slot-to-slot
   D2D copy는 추가하지 않았다.

코드 위치:

- `cpp/runtime/scheduling/phaseQueueScheduler.cpp`: runnable snapshot, cohort intersection count,
  zero-runnable-row WAIT
- `unittests/phaseQueueSchedulerTest.cpp`: overflow/in-flight/partial-refill/65·72·80 active 회귀 테스트

## 실험 계약

- model: `nvidia/Cosmos-Reason2-2B`, FP16, 비양자화
- engine/runtime: Current P8/D64, fixed prefill chunk 128, 80 stable indexed slots
- execution: independent P/D TensorRT contexts와 Global selector
- trace: P5/P6와 동일한 materialized real-request trace, offered 48.8 req/s
- trace SHA-256: `c9d64ed7e84fea87352c9b4acedb309932c8f0e9cae71d527fc2f868ec90849d`
- warmup: 64 requests, output 32 tokens
- measured work: run당 288 requests, prompt 25,872 tokens, output 24,960 tokens
- client maximum in-flight: 세 Current 실험 모두 80
- server admission: 64, 72, 80만 변경
- joint SLO: scheduled TTFT <= 500 ms, TPOT <= 50 ms, scheduled E2E <= 2500 ms
- 반복: 각 capacity 5회 fresh backend
- 통계: 5회 산술평균과 Student-t 95% confidence interval half-width, 자유도 4
- vLLM: model/trace/arrival/output/SLO 계약이 같으므로 Note 187의 fresh 5-run frozen baseline 재사용

## End-to-end 결과

`mean +/- CI95`이고 latency 단위는 ms다.

| Runtime | Raw req/s | SLO pass | Goodput req/s | Scheduled TTFT mean/p95 | TPOT mean/p95 | Scheduled E2E mean/p95 | Peak MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| Current, cap64 | 37.865 +/- 0.122 | 76.0 +/- 5.0 pp | 28.769 +/- 1.930 | 240.95 / 618.52 | 15.82 / 17.87 | 1587.27 / 2356.13 | 9081 |
| Current, cap72 | 33.948 +/- 0.291 | 49.8 +/- 1.9 pp | 16.906 +/- 0.795 | 442.61 / 1344.37 | 19.64 / 24.01 | 2114.86 / 3310.11 | 9081 |
| Current, cap80 | 35.810 +/- 0.167 | 52.8 +/- 1.4 pp | 18.900 +/- 0.552 | 291.62 / 865.65 | 19.81 / 25.35 | 1963.02 / 3042.51 | 9081 |
| vLLM, frozen | 40.908 +/- 0.029 | 100% | 40.908 +/- 0.029 | 51.44 / 84.17 | 13.53 / 17.65 | 1205.87 / 2119.43 | 9039 |

Current 세 구성은 모두 5/5 run에서 288/288 request와 24,960/24,960 output token을 완료했다. 모든
Current run의 token hash는
`f51d448d5824038f5237cdee55dc50799de5c5cb5a228fcf1166ff1ed30875ca`로 동일하다.

cap64의 새 5회 결과는 Note 187의 frozen Current와 confidence interval이 겹치는 같은 성능 영역이다.
따라서 correctness 수정이 기본 cap64 경로에 유의미한 성능 회귀를 만들었다는 증거는 없다.

## Admission을 늘렸을 때 latency가 이동한 위치

| Capacity | Client dispatch mean/p95 | Send-relative TTFT mean/p95 | TTFT failures/run | E2E failures/run | Joint failures/run |
|---|---:|---:|---:|---:|---:|
| 64 | 55.48 / 219.63 | 185.48 / 412.63 | 62.8 | 10.8 | 69.2 |
| 72 | 308.68 / 1103.28 | 133.94 / 302.63 | 111.8 | 95.4 | 144.6 |
| 80 | 266.00 / 840.68 | 25.63 / 40.53 | 97.4 | 68.0 | 136.0 |

cap80은 server가 즉시 admission하므로 send-relative TTFT는 크게 줄었다. 그러나 각 request가 더 오래
active 상태로 남아 client의 80-request concurrency window가 늦게 열린다. 그 결과 이후 scheduled arrival가
client에서 오래 기다리고 scheduled TTFT mean은 `291.62 ms`, p95는 `865.65 ms`가 된다. 즉 queue가
server admission 앞에서 client admission 앞으로 이동했을 뿐 end-to-end SLO capacity는 회복되지 않았다.

TPOT threshold를 넘긴 request는 모든 Current capacity에서 0개다. 그러나 cap72/80의 TPOT가 약
`19.6--19.8 ms`로 늘면서 slot lifetime과 client dispatch delay를 간접 증가시키고 TTFT/E2E failure를
만든다.

## KV/page mapping과 decode work 변화

아래 값은 5회 평균 runtime counter다.

| Capacity | D prepares | D page-copy bytes | Device-reuse batches | Device-reuse rows | CUDA graph D hits/misses |
|---|---:|---:|---:|---:|---:|
| 64 | 694.6 | 293,990 | 355.4 | 12,003.4 | 0 / 973.6 |
| 72 | 845.2 | 3,457,254 | 243.4 | 5,690.8 | 0 / 1124.0 |
| 80 | 782.2 | 2,471,859 | 246.6 | 5,666.0 | 0 / 1061.0 |

stable indexed ownership은 overflow request의 KV를 움직이지 않지만, active set이 D64보다 커지면 현재
cohort가 끝나고 replacement row가 들어오는 selection shape가 자주 바뀐다. cap72/80에서 device-side
selection reuse row는 절반 이하로 줄고 page metadata copy bytes는 크게 증가했다. 이것은 KV payload
compaction이 아니라 page binding/selection metadata 비용이다.

또한 이번 engine에서 CUDA graph는 세 capacity 모두 decode hit가 0이다. 커진 active set이 만들어낸 더
많은 decode dispatch와 shape churn을 eager launch가 그대로 부담한다.

## Promotion 판단

### 승격

- active 65/72/80에서 capacity-safe persistent decode cohort
- dispatchable row와 total queued overflow row의 의미 분리
- cohort completion을 기다리는 explicit WAIT semantics
- exact token identity와 stable indexed ownership 유지

### 승격하지 않음

- server admission 72 또는 80을 기본값으로 변경
- free stable slot 수만 보고 admission을 D batch보다 크게 증가
- 이번 결과를 admission expansion 성능 향상으로 해석

기본 정책은 D64에 맞춘 cap64를 유지한다. 향후 admission을 64보다 늘리려면 단순 capacity가 아니라
predicted decode service와 page-selection churn이 SLO-safe한지를 같이 확인해야 한다.

## 검증

- 수정 전 신규 cohort test 3개: 3/3 expected failure, production invariant 재현
- 수정 후 `PhaseQueueSchedulerTest.*`: 137/137 passed
- scheduler 전체 targeted gate: 187/187 passed
  (`IndependentPhaseAsyncServerTest.*`, `PhaseGlobalSchedulerTest.*`, `PhaseQueueSchedulerTest.*`)
- TensorRT 11.0/CUDA 13.3 incremental build: `unitTest`, `llm_phase_context_smoke` passed
- real-request cap80 smoke: 기존 abort 대신 288/288 complete
- real-request 64/72/80: 각 5/5 complete, exact token hash 동일

## 다음 권장 단계

다음 단계는 admission을 더 늘리는 것이 아니라 Note 188에서 확인한 sampling completion visibility를 줄이는
것이다.

1. sampling CUDA event를 generic server poll에 의존하지 않고 completion-ready queue로 직접 전달한다.
2. `sampling submit -> CUDA complete -> CPU visible -> state commit -> decode ready` timestamp를 유지한다.
3. cap64, 48.8 trace를 5회 재실행해 raw throughput, scheduled TTFT, joint-SLO goodput을 이번 cap64와 비교한다.
4. 같은 workload이므로 vLLM은 Note 187의 frozen baseline을 계속 사용한다.
5. 개선이 없으면 D64 page-selection/graph miss와 vLLM iteration service 차이를 다음 병목으로 분해한다.

## 산출물

- 집계 CSV: `benchmarks/phase_serving/results/capacity-safe-admission-sweep-20260830.csv`
- raw Current runs: `.local/transition-aware-20260830/p7-capacity-safe-cohort/`
- joint-SLO per-run 결과: `.local/transition-aware-20260830/p7-capacity-safe-cohort/slo-goodput.json`
- frozen vLLM runs: `.local/transition-aware-20260830/p6-repeat-baseline/vllm/vllm-48.8/`
