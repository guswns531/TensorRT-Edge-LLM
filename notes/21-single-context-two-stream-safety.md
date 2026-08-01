# 단일 CUDA context / 분리 TensorRT context 실행 안전 계약

## 결론

여기서 공유하는 context는 프로세스가 사용하는 하나의 CUDA primary context다. prefill/decode를 실제로 겹치는
모드에서는 그 안에 CUDA stream 두 개와 TensorRT `IExecutionContext` 두 개를 둔다. CUDA stream이 다르다는
사실만으로 하나의 TensorRT context를 동시에 enqueue해도 안전해지는 것은 아니다. optimization profile, tensor
address, shape, workspace 상태가 host-side mutable state이기 때문이다.

```text
one CUDA primary context
├─ prefill stream ─ TensorRT context P ─ workspace P ─ I/O P
└─ decode stream  ─ TensorRT context D ─ workspace D ─ I/O D
```

호환용 shared TensorRT context 모드에서는 prefill CUDA event가 끝난 뒤 decode의 prepare와 enqueue를 시작한다.

```text
prefill stream: [prepare][enqueue]----[prefill done event]
                                                  |
                                                  | host event completion
                                                  v
decode stream:                              [prepare][enqueue]----[decode done event]
```

두 phase를 실제로 겹치는 independent TensorRT context 모드는 다음 identity가 모두 non-null이고 서로 달라야 한다.

- TensorRT execution context
- context workspace
- phase-local I/O buffer owner
- CUDA stream

```text
prefill: Context P + Workspace P + I/O P  [==============]
decode:  Context D + Workspace D + I/O D       [==============]
```

`PhaseExecutionSafetyContract`와 `PhaseDispatchWorker`가 이 조건을 construction time에 검증한다. 두 stream의
`cuStreamGetCtx()` 결과도 같아야 하고, 그 context는 현재 device의 primary context여야 한다. shared TensorRT 모드는
동일한 execution-context identity만 허용하고, independent 모드는 context/workspace/I/O/stream 중 하나라도 alias이면
실행 전에 거부한다.

## 왜 CUDA stream wait만으로 부족한가

`cudaStreamWaitEvent()`는 device work의 선후관계만 만든다. 두 host callback이 같은 `IExecutionContext`에 대해
profile 전환, input shape 설정, tensor binding을 동시에 수행하는 것은 막지 못한다. 그래서 shared 모드에서는
decode callback 자체를 prefill event completion 뒤로 미룬다. queue와 stream은 분리되어 있지만 TensorRT context
mutation은 host에서 직렬화된다.

## Gemma 4 E2B INT4 indexed 재검증

RTX 3080, TensorRT 11.0.0, BS2, input 128, prefill chunk 128, past KV 128, warmup 1, measurement 3 조건이다.

| 모드 | sequential median | scheduled median | makespan speedup |
|---|---:|---:|---:|
| independent TensorRT context | 30.6012 ms | 27.1404 ms | 1.1275x |
| shared TensorRT context | smoke 통과 | smoke 통과 | 1-sample 1.0245x |

shared 결과는 overlap 속도 향상을 주장하지 않는다. 두 queue/stream/event 구조를 유지하면서 하나의 TensorRT
context를 안전하게 사용하는 fallback이다. independent 결과도 작은 smoke sample이며 최종 throughput/latency
gate는 아니다. 이번 결과는 `/tmp/gemma4-e2b/perf/phase/shared-cuda-independent-trt.csv`에 저장했다.

## 검증 범위

- shared context의 decode callback이 prefill batch completion 전에 호출되지 않음
- independent context에서 두 callback을 event wait 전 enqueue할 수 있음
- aliased context/workspace/I/O contract가 거부됨
- prefill/decode stream이 하나의 CUDA primary context임을 runtime에서 검증
- Gemma PLE table은 공유하면서 mutable output buffer는 phase별 분리
- 실제 Gemma indexed engine의 shared/independent 실행 성공
