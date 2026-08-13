# Cosmos direct-overlap cost와 TPOT hard guard

## 결론

Cosmos-Reason2-2B FP16 indexed-paged P16/D64 엔진에서 fixed 128-token chunk를 유지하면서 다음 경로를
production scheduler와 independent TensorRT executor pair에 연결했다.

- 실제 CUDA-event dispatch에서 `Co(P,D,chunk,past-KV,decode-context)` p95를 직접 생성한다.
- prefill 선택기는 현재 계획된 decode batch/context에 맞는 direct overlap point를 우선 사용한다.
- 허용 decode slack, 연속 overlap turn, 누적 predicted decode debt를 넘으면 overlap을 decode-only로 바꾼다.
- decode cost의 D bucket은 상한이다. 예를 들어 active D=63은 D64 p95로 평가하면서 63개를 모두 실행한다.
- production profile은 `latency-safe`, `balanced`, `long-prefill`, `auto`, `custom`으로 구분한다.
- phase-local CUDA graph 정책은 benchmark가 아니라 `IndependentEngineExecutorPairConfig`가 소유한다.

현재 권장은 다음과 같다.

1. 일반 기본값은 여전히 fixed-P2이다. cost table이 없거나 GPU/engine shape가 달라지면 가장 예측 가능하다.
2. profiled balanced workload에서는 `balanced`가 더 좋았다. 기본 hard guard는 overlap streak 2, debt 25ms이다.
3. decode-heavy에서는 fixed-P2가 더 좋았다. profile 자동 선택은 workload classifier를 더 학습한 뒤 기본 활성화해야 한다.
4. `latency-safe`는 direct overlap coverage를 강제하므로 uncovered final-chunk shape가 많으면 처리량을 크게 잃을 수 있다.

## 구현 위치

- `cpp/runtime/scheduling/phaseQueueScheduler.{h,cpp}`
  - direct overlap cost 구조체와 profile
  - planned D/context 기반 cost lookup
  - TPOT hard guard, overlap streak, predicted debt
  - sparse context fallback과 decode upper-bucket 실행
- `cpp/runtime/scheduling/phaseDispatchWorker.cpp`
  - debt, streak, hard-guard deferral을 dispatch metric에 전달
- `cpp/runtime/scheduling/independentEngineExecutorPair.{h,cpp}`
  - independent prefill/decode context별 CUDA graph count/memory/reserve 정책
- `cpp/runtime/exec/engineExecutor.{h,cpp}`
  - observed, uncapturable, memory-rejected shape 수 telemetry
- `examples/llm/llm_phase_bench.cpp`
  - profile/guard CLI, schema-v3 loader, CSV telemetry, graph hit-rate 출력
- `scripts/cosmos_reason2/build_prefill_wavefront_cost_model.py`
  - schema v3 direct overlap table과 canonical decode upper buckets 생성
- `scripts/cosmos_reason2/run_real_request_kv_matrix.py`
  - real-request matrix에 profile/guard 설정 전달 및 manifest 저장

## 의사 결정 흐름

```text
prefill queue + decode queue
           |
           v
  planned decode D/context
           |
           v
direct Co(P,D,shape) point available? -- no --> indirect conservative prefill point
           |
           v
decode slack, streak, predicted debt 검사
        |                       |
      safe                    unsafe
        |                       |
 overlap P+D              decode-only dispatch
                                |
                                v
                  다음 turn에서 debt/streak reset
```

`requireDirectOverlapCost=true`이면 direct point가 없는 shape도 unsafe이다. `balanced`는 direct point를 우선하지만
coverage hole에서는 간접 p95를 사용한다. `latency-safe`만 direct coverage를 강제한다.

## Cost table 의미

schema v3는 세 배열을 가진다.

- `decode`: `(max batch, max context) -> p95 decode GPU ms`
- `prefill`: `(P, chunk, max past-KV, max concurrent D, initial) -> p95 prefill/indirect slowdown`
- `overlap`: `(P, max D, chunk, max prefill past-KV, max decode context, initial)`에 대한 prefill, decode,
  makespan, decode slowdown p95

D와 context는 모두 upper bucket이다. 실험에서 처음에는 관측된 D=37/50 등을 별도 action으로 사용했고 D8/D32
backlog 진동이 발생했다. canonical D bucket을 상한으로 해석하고 현재 active rows 전체를 실행하도록 바꾼 뒤 이
문제가 사라졌다. cost table은 GPU, engine, TensorRT context mode, chunk, KV page 설정이 바뀌면 재생성해야 한다.

이번 통합 table은 기존 real-suite의 dispatch CUDA events로 만들었으며 다음 coverage를 가졌다.

| table | points |
| --- | ---: |
| decode | 13 |
| prefill | 213 |
| direct overlap | 271 |

## 96-request 결과

동일 engine, 동일 materialized trace, P8/D64, 80 slots, 256 page bundles, independent contexts 조건이다. 수치는
각각 한 번의 paired process run이므로 최종 채택 전 3회 반복이 필요하다.

### Balanced trace

| metric | fixed-P2 | balanced profile | change |
| --- | ---: | ---: | ---: |
| request/s | 42.338 | 43.055 | +1.69% |
| generated token/s | 3584.6 | 3645.3 | +1.69% |
| TTFT p95 | 1041.8 ms | 1042.4 ms | +0.06% |
| TPOT p95 | 17.230 ms | 17.016 ms | -1.24% |
| E2E p95 | 1996.4 ms | 1934.2 ms | -3.12% |
| dispatches | 210 | 209 | -1 |

balanced profile은 20번 overlap을 decode-only로 미뤘고 연속 overlap은 최대 2였다.

### Decode-heavy trace

| metric | fixed-P2 | balanced profile | change |
| --- | ---: | ---: | ---: |
| request/s | 19.779 | 19.210 | -2.88% |
| generated token/s | 3982.2 | 3867.6 | -2.88% |
| TTFT p95 | 1422.2 ms | 1419.0 ms | -0.22% |
| TPOT p95 | 14.652 ms | 16.002 ms | +9.21% |
| E2E p95 | 4342.5 ms | 4413.8 ms | +1.64% |

따라서 profile 이름은 workload classifier가 아니라 명시적 operator 선택이다. decode-heavy에서는 fixed-P2가
production 권장값이다.

## CUDA graph와 메모리

P16/D64/80-slot engine은 graph 활성화 전에도 10GB GPU에서 9494.9MiB를 사용하고 379.4MiB만 남긴다. 이는
512MiB headroom gate 실패이며 scheduler 문제가 아니라 engine/context/KV/embedding 조합의 기본 메모리이다.

64MiB per-phase graph budget, graph당 최소 4MiB charge, 256MiB global free reserve smoke 결과는 다음과 같다.

| phase | cached | hit rate | graph bytes | failures |
| --- | ---: | ---: | ---: | ---: |
| prefill | 1 | 30.00% | 4MiB | 0 |
| decode | 2 | 98.28% | 8MiB | 0 |

종료 시 free memory는 361.4MiB였다. 새 telemetry는 observed shape, uncapturable shape, budget-rejected shape를
분리하므로 동적 shape churn과 메모리 부족을 구분할 수 있다.

## 안정성 검증

- 관련 unit tests 40/40 통과
- 전체 C++ suite는 665개 중 625 pass, 39 skip, 1 fail이었다. 단독 재현된 실패는 수정 범위 밖의
  `InitializeMRopeCosSin.Accuracy` 기존 수치 tolerance 차이(got -0.13458, expected -0.13607)이다.
- P8/D32 one-request, 16-output-token full lifecycle compute-sanitizer memcheck: 0 errors
- P8/D32 4-request Nsight Systems report 생성
- Nsight에서 indexed length gather/increment, paged page-list gather, RoPE/KV write kernel 실행 확인
- 288-request decode-heavy endurance: 288/288 완료, 최종 page pool `allocated=0`, 최대 observed allocation 180/256,
  adaptive growth owners 최대 8, overlap streak 최대 4, predicted debt 최대 35.1ms

288-request 부하는 1000 req/s open-loop overload이므로 성능 대표값이 아니라 backpressure/endurance 검증이다.
initial pending은 208, TTFT/TPOT tail은 크게 증가했지만 page leak이나 admission deadlock 없이 종료했다.

실험 artifact는 `.local/cosmos-reason2-2b/phase-hard-guard-20260813/`에 있다. cost model은
`.local/cosmos-reason2-2b/prefill-wavefront-20260812/cost-model-v3.{json,csv}`이다.

## 다음 단계

1. balanced/fixed-P2/decode-heavy를 각 3회 process repeat해 confidence interval과 3% gate를 판정한다.
2. workload classifier는 prompt remaining, continuation ratio, active decode rows, context 분포를 입력으로 하되,
   먼저 fixed-P2/decode-heavy fallback을 유지한다.
3. direct overlap table의 final-chunk coverage를 보강하고 GPU/engine hash를 schema metadata에 넣는다.
4. P8/D32처럼 최소 512MiB headroom을 확보한 production engine에서 CUDA graph on/off를 반복 비교한다.
5. Nsight SQLite에서 phase stream별 kernel interval을 자동 추출해 CUDA-event group 합과 교차 검증한다.
