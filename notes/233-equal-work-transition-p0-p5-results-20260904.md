# Equal-work transition P0--P5 구현 및 검증

## 결론

이번 단계에서 production default는 바꾸지 않았다. 최종 production 후보는 여전히 **V1 Contextual Scalar
Immediate**다.

- strict decision-to-dispatch identity는 새 telemetry에서 `3,159/3,159`, mismatch `0`이다.
- V2 Scalar+Transition은 mixed와 multi-image에서 successor formation을 개선하는 신호가 있지만 mixed TTFT p95
  regression이 `+4.94%`여서 3% promotion gate를 통과하지 못했다.
- V4 Decomposed Effect와 V5 Completion은 물리 예측 MAE가 Scalar보다 작은 경우가 있어도 false-safe `0` 조건과
  sign-agreement 조건을 동시에 만족하지 못했다.
- process-restart forced replay는 async 준비를 끈 뒤에도 공통 full strict pre-branch snapshot을 만들지 못했다.
  descriptive branch 결과를 causal evidence로 사용하지 않는다.
- 최신 V1의 full12 1회는 frozen fresh vLLM보다 token throughput, request throughput, joint-SLO goodput,
  E2E mean/p95에서 모두 `12/12` 우세다. token throughput geometric mean은 `+16.34%`다.
- long-prefill 3회 confirmation은 `1,163.71 tok/s`다. 동일 `64 + v7-small-d` frozen Current의
  `1,190.5 tok/s` 대비 `-2.25%`로 3% gate 안이다.

핵심 architecture는 다음과 같다.

```text
Deterministic mechanism
  DAG legality / TRT shape / single inflight / KV+vision ownership
                 |
                 v
Common candidate frontier + strict pre-dispatch snapshot
                 |
       +---------+----------+
       |         |          |
    Scalar     Effect    Completion
   authority   shadow      shadow
       |         |          |
       +---------+----------+
                 |
     same-frontier H=2 formation replay
                 |
     held-out physical + SLO promotion gate
                 |
        production dispatch (V1 only)
```

`모두 진행`은 gate를 실패한 richer model까지 production에 강제로 켠다는 의미로 처리하지 않았다. 이는
workload별 fine tuning을 없애려는 목표와 correctness/SLO invariant를 지키기 위한 선택이다.

## 1. P0 -- strict paired correctness

### 1.1 Dispatch signature

기존 `strict_snapshot_signature`는 action을 선택하기 직전의 공통 상태를 식별한다. 새
`dispatch_signature`는 그 위에 실제 선택/실행 identity를 더한다.

```text
strict snapshot
+ selected candidate ID
+ incremental action ID
+ action kind/direction/dispatch mode
+ outstanding-before/planned-outstanding masks
= dispatch signature
```

Candidate ID에는 primary/secondary request IDs와 row order, stable slot identity가 이미 포함된다. 따라서
completion event가 decision과 같은 dispatch를 끝낸 것인지 exact하게 검사할 수 있다.

구현 위치:

- `cpp/runtime/phase/mechanism/phaseUnifiedEvent.h`
- `cpp/runtime/scheduling/phaseThreeCoordinator.cpp`
- `examples/llm/llm_phase_context_smoke.cpp`
- `benchmarks/phase_serving/manifests/phase_event_schema_v1.json`

관측 결과:

| Dataset | decision signatures | completion signatures | matched | mismatched |
|---|---:|---:|---:|---:|
| controlled four workloads, 1x | 4,178 | 1,091 | 1,091 | 0 |
| mixed + multi-image, 3x | 3,578 | 2,068 | 2,068 | 0 |
| 합계 | 7,756 | 3,159 | 3,159 | 0 |

completion이 없는 decision은 아직 GPU completion을 기다리는 logical decision이거나 paired action의 다른 member일
수 있으므로 mismatch로 세지 않는다.

### 1.2 Counterfactual analyzer 수정

기존 analyzer는 E와 E+P처럼 서로 다른 branch를 비교하면서 selected request set까지 같아야 한다고 요구했다.
이는 정상적인 counterfactual을 제거하는 잘못된 조건이었다.

새 계약은 다음과 같다.

```text
branch 사이: strict pre-branch signature가 같아야 함
branch 내부: decision dispatch signature == 모든 completion dispatch signature
selected request/work: branch action에 따라 달라질 수 있음
```

`--forced-only`를 추가해 자연 decision과 강제된 causal branch를 섞지 않는다.

## 2. P1 -- equal-work successor rollout

### 2.1 적용 범위 확장

기존 active H=2 formation은 `E + P/D`가 동시에 보이는 VLM 상태에만 들어갔다. 이번에는 current candidate
frontier에서 두 phase 이상이 실제로 노출되면 실행하도록 바꿨다.

```text
target E rows = max E rows in current candidates
target P rows = max P rows in current candidates
target D rows = max D rows in current candidates
```

따라서 P+D-only state도 같은 bounded H=2 evaluator를 사용한다. 임의의 future arrival, workload label, trace ID는
입력에 포함하지 않는다. reference work도 phase-local single action의 합으로 계산한다.

### 2.2 동일 frontier에서 V2/V4/V5 비교

같은 candidate vector를 복사한 뒤 각 physical model의 decision cost만 교체한다.

```text
V2 Scalar+T      scalar action cost      -> H2
V4 Effect+T      compression LCB         -> H2
V5 Completion+T incumbent/newcomer time -> H2
```

세 모델 모두 candidate construction, batch membership, row ordering, SLO guard와 transition mechanism은 같다.
각 decision event는 model별 selected action ID, robust horizon, decode violation, protected-request violation을 남긴다.

현재 H=2는 **same-frontier bounded successor evaluator**다. full live request graph를 임의로 복제하거나 미래
arrival을 예측하지 않는다. frozen request-lineage replay는 별도 shadow telemetry이며 live allocator를 변경하지 않는다.

## 3. P2 -- controlled counterfactual 결과

5-request multi-image trace에서 decision 2를 `encoder`와 `encoder_prefill`로 각각 5회 강제했다. 이후 async image
preparation과 request adapter를 끈 통제 계약에서도 각각 3회 반복했다.

### 3.1 Production-like forced branches

| Branch | runs | token/s 중앙 경향 | TTFT mean 경향 | TPOT p95 경향 |
|---|---:|---:|---:|---:|
| forced E | 5 | 약 322 | 약 252 ms | 약 8.73 ms |
| forced E+P | 5 | 약 315, 1 outlier 268 | 약 220 ms | 약 11.92 ms |

E+P는 first-token progress를 당기지만 resident decode tail을 늘리는 descriptive trade-off를 보였다.

### 3.2 Synchronous controlled branches

`TRT_EDGELLM_VISION_ASYNC_PREPARATION=0` 및 `TRT_EDGELLM_IPC_ASYNC_REQUEST_ADAPTER=0`에서도 공통 full strict
snapshot은 `0`이었다. 일부 ready row가 같아도 직전 outstanding state와 scalar CUDA history가 달랐다.

따라서 다음은 주장하지 않는다.

```text
forced E+P가 E보다 causal하게 빠르다/느리다  (증명 안 됨)
```

같은 live GPU/allocator/TensorRT execution context를 process branch별로 fork하는 것은 현재 runtime에서 불가능하다.
현재 in-process immutable replay는 같은 snapshot에 measured/model envelope를 주입해 두 ready boundary를 재생하는
correctness/decision-regret 도구이지, 두 실제 GPU trajectory를 동시에 실행하는 도구가 아니다.

## 4. P3 -- model promotion 재평가

### 4.1 Controlled four-workload 1x physical gate

| Model | ready samples | sign agreement | false-safe | makespan MAE us | Gate |
|---|---:|---:|---:|---:|---|
| Scalar | 1,042 | 86.95% | 10 | 9,919 | FAIL |
| Effect | 978 | 74.03% | 2 | 6,908 | FAIL |
| Completion | 1,046 | 71.22% | 1 | 7,098 | FAIL |

### 4.2 Mixed + multi-image 3x physical gate

| Model | ready samples | sign agreement | false-safe | makespan MAE us | Gate |
|---|---:|---:|---:|---:|---|
| Scalar | 1,973 | 86.47% | 23 | 17,767 | FAIL |
| Effect | 1,944 | 66.92% | 14 | 10,981 | FAIL |
| Completion | 1,984 | 68.45% | 10 | 13,773 | FAIL |

Effect/Completion은 MAE만 보면 Scalar보다 나은 구간이 있다. 하지만 scheduler promotion에는 평균 오차보다
false-safe가 더 중요하다. 실제로 손해인 action을 안전하다고 판단한 표본이 있으므로 V4/V5는 shadow에 남긴다.

### 4.3 Formation action agreement

1x에서는 세 모델이 1,130 formation decision에서 모두 같은 action을 골랐다. 3x에서는:

| Model | Scalar와 공통 formation decisions | Scalar action agreement |
|---|---:|---:|
| Effect | 3,198 | 99.94% |
| Completion | 3,198 | 98.50% |

즉 richer prediction이 대부분 동일 action으로 collapse한다. Completion은 48개 정도의 action disagreement를
만들지만 false-safe gate를 통과하지 못했으므로 이 disagreement에 authority를 줄 근거가 없다.

## 5. V1 대 V2 반복 A/B

동일 새 binary, `max-in-flight=64`, v7 generic calibration, 같은 HTTP traces에서 비교했다.

| Workload | Metric | V1 Scalar immediate | V2 Scalar+Transition | V2 변화 |
|---|---|---:|---:|---:|
| mixed | token/s | 1,167.12 | 1,176.55 | +0.81% |
| mixed | TTFT mean / p95 ms | 706.93 / 1,970.00 | 741.61 / 2,067.35 | +4.91% / +4.94% |
| mixed | TPOT mean / p95 ms | 37.69 / 57.06 | 34.14 / 47.86 | -9.42% / -16.12% |
| mixed | E2E mean / p95 ms | 2,374.72 / 2,485.11 | 2,322.11 / 2,470.99 | -2.22% / -0.57% |
| multi-image | token/s | 315.28 | 321.08 | +1.84% |
| multi-image | TTFT mean / p95 ms | 258.75 / 298.91 | 239.65 / 288.99 | -7.38% / -3.32% |
| multi-image | TPOT mean / p95 ms | 8.63 / 10.96 | 8.55 / 11.15 | -0.91% / +1.77% |
| multi-image | E2E mean / p95 ms | 497.93 / 507.05 | 492.73 / 497.98 | -1.04% / -1.79% |

`multi-image`는 양쪽 모두 cross-repeat token hash가 비결정적이므로 promotion 근거로 약하다. `mixed`는 TPOT/E2E
개선이 명확하지만 protected first-token tail이 3% gate를 넘는다. 따라서 V2를 default로 바꾸지 않았다.

## 6. P4 -- Selective authority 판정

Selective V6의 prerequisite는 richer model의 검증된 state domain이다.

```text
samples >= 100
sign agreement >= 80%
false-safe == 0
same-frontier transition regret 개선
end-to-end throughput와 모든 protected tail regression <= 3%
```

V4/V5 모두 false-safe와 sign gate를 실패했다. V2도 mixed TTFT p95 gate를 실패했다. 따라서 selective authority를
활성화하지 않았다. workload 이름이나 `if E<=2`, `if multi-image` 같은 static rule을 추가하지 않았다.

## 7. P5 -- 최종 V1 full12 대 fresh frozen vLLM

계약:

```text
model: nvidia/Cosmos-Reason2-2B
Current: max-in-flight=64, P128, P8/D64/E4, v7 generic calibration
HTTP trace/output/timeout/ignore-EOS: frozen comparison과 동일
vLLM: 2026-09-04 fresh 12x1 artifact 재사용
```

Current는 최신 binary 1회다. vLLM은 trace와 vLLM runtime contract가 바뀌지 않아 새로 실행하지 않았다.

| Workload | Current tok/s | vLLM tok/s | Throughput | TTFT mean/p95 | TPOT mean/p95 | E2E mean/p95 |
|---|---:|---:|---:|---:|---:|---:|
| balanced | 4,549.2 | 4,312.7 | +5.48% | -43.69% / -42.22% | -0.86% / -1.24% | -5.30% / -5.63% |
| bimodal | 1,928.4 | 1,840.2 | +4.79% | +19.82% / +59.40% | -21.43% / -22.04% | -8.16% / -2.98% |
| decode-heavy | 5,311.1 | 4,960.4 | +7.07% | -44.91% / -42.15% | -4.48% / -4.92% | -6.44% / -5.95% |
| late-vision | 2,547.0 | 2,356.6 | +8.08% | -26.48% / -31.78% | -6.57% / -6.01% | -8.61% / -7.52% |
| long-prefill | 1,164.4 | 1,118.9 | +4.06% | +11.23% / -10.64% | -16.00% / -14.59% | -4.69% / -7.19% |
| mixed | 1,129.0 | 747.7 | +50.99% | -52.99% / -43.17% | +14.92% / -29.42% | -15.93% / -34.82% |
| multi-image | 321.7 | 237.6 | +35.40% | -20.92% / -32.03% | -29.75% / -31.59% | -26.17% / -26.31% |
| poisson | 1,973.8 | 1,792.9 | +10.09% | -57.19% / -17.12% | -2.86% / -13.27% | -13.61% / -12.50% |
| short | 2,489.4 | 1,987.4 | +25.26% | -49.21% / -32.02% | -2.32% / +0.03% | -23.11% / -19.46% |
| text-heavy | 2,132.2 | 1,636.5 | +30.29% | -34.99% / -26.91% | -18.41% / -21.12% | -23.55% / -22.76% |
| vision-heavy | 706.3 | 575.7 | +22.67% | -25.90% / -16.21% | -45.46% / -55.66% | -34.99% / -19.57% |
| wave-drain | 98.1 | 95.7 | +2.45% | -4.90% / -35.56% | -38.56% / -49.36% | -24.58% / -26.15% |

요약:

- token/request throughput: `12/12` Current 우세
- throughput geometric mean: `+16.34%`
- joint-SLO goodput: `12/12` Current 우세
- E2E mean/p95: `12/12` Current 우세
- TTFT mean: `10/12`, TTFT p95: `11/12` Current 우세
- TPOT mean/p95: 각각 `11/12` Current 우세
- peak memory: `8/12`에서 Current가 작거나 같음

### 7.1 같은 64 + v7-small-d frozen Current 재현

과거 260-request calibration을 쓴 P7 Scalar 표가 아니라, note 232의 동일 239/319-request v7-small-d 계약과
비교해야 한다.

| Workload | Latest V1 tok/s | Matching frozen V1 tok/s | 변화 |
|---|---:|---:|---:|
| short | 2,489.4 | 2,525.1 | -1.41% |
| balanced | 4,549.2 | 4,461.3 | +1.97% |
| decode-heavy | 5,311.1 | 5,255.1 | +1.07% |
| long-prefill | 1,163.7 (3x) | 1,190.5 | -2.25% |
| bimodal | 1,928.4 | 1,950.5 | -1.13% |
| text-heavy | 2,132.2 | 2,010.8 | +6.04% |
| mixed | 1,129.0 | 1,099.2 | +2.71% |
| poisson | 1,973.8 | 2,019.8 | -2.28% |
| vision-heavy | 706.3 | 644.4 | +9.60% |
| wave-drain | 98.08 | 98.1 | -0.02% |
| late-vision | 2,547.0 | 2,553.6 | -0.26% |
| multi-image | 321.7 | 236.2 | +36.20% |

하락한 workload는 모두 3% 안이다. multi-image의 큰 증가는 5-request run variance가 커서 성능 향상 claim으로
사용하지 않는다. 이 표는 최근 strict/frozen telemetry 구현이 production V1 hot path를 구조적으로 훼손하지
않았다는 regression check다.

한계:

- Current와 vLLM 모두 이 표에서는 1회다. confidence interval용 최종 paper 수치가 아니다.
- long-prefill을 과거 260-request calibration 결과 `1,237.6 tok/s`와 직접 비교하면 `-5.97%`로 보이지만 이는
  calibration 계약이 다르다. 같은 239-request v7-small-d frozen Current `1,190.5 tok/s`와 비교하면 3회
  confirmation은 `-2.25%`로 gate 안이다.
- mixed TPOT mean, bimodal TTFT mean/p95, long-prefill TTFT mean, short TPOT p95는 vLLM이 낫다. 모든 개별
  latency 축에서 이긴다는 주장은 하지 않는다.

## 8. Artifact

```text
.local/p11-full-transition-controlled4-20260904/
.local/p11-current-binary-immediate-controlled4-20260904/
.local/p11-immediate-mixed-multi-r3-20260904/
.local/p11-transition-mixed-multi-r3-20260904/
.local/p11-forced-{encoder,encoder-prefill}-v2-20260904/
.local/p11-forced-sync-{encoder,encoder-prefill}-20260904/
.local/p11-final-v1-full12-20260904/
```

분석 결과:

```text
.local/p11-full-transition-controlled4-20260904/transition-analysis.json
.local/p11-transition-mixed-multi-r3-20260904/transition-analysis.json
.local/p11-forced-branch-analysis-20260904.json
.local/p11-forced-sync-analysis-20260904.json
.local/p11-final-v1-full12-20260904/policy-matrix.{json,csv}
.local/p11-final-v1-full12-20260904/vllm-gate.{json,csv}
```

## 9. 다음 순서

1. process restart가 아닌 single-process measured-envelope fork를 controlled harness로 분리한다. live TensorRT/KV를
   두 번 실행했다고 부르지 않고 deterministic counterfactual evaluator임을 API 이름에 명시한다.
2. V2 mixed TTFT regression을 workload rule 없이 protected-request completion budget으로 해결한다.
3. V4/V5는 false-safe `0`을 만들기 전까지 shadow로 유지한다.
4. 최종 후보가 V1에서 실제로 바뀐 경우에만 full12 3--5회와 fresh vLLM을 다시 실행한다.

현재 결론은 단순하다.

```text
Production winner: V1 Contextual Scalar Immediate
Useful experimental signal: V2 Scalar + bounded transition
Not promotable: V4 Effect+T, V5 Completion+T, V6 Selective
Reason: physical false-safe and protected TTFT gates
```

## 10. 구현 검증

최종 소스 상태에서 다음을 통과했다.

| 검증 | 결과 |
|---|---:|
| TensorRT 26.06 전체 C++ build | PASS |
| transition/event/KV 집중 C++ unit tests | 44/44 PASS |
| policy/transition/counterfactual Python unit tests | 23/23 PASS |
| 수정 파일 pre-commit | PASS |

첫 Docker build는 GPU device를 전달하지 않아 `libcuda.so` 링크에서 실패했으며, 동일 빌드 디렉터리를
`--gpus all`로 실행한 최종 검증은 성공했다. 이는 소스 compile failure가 아니라 container device 전달 조건의
차이다.
