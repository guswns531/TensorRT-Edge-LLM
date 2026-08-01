# Single-context/two-stream 실행 안전 계약

## 결론

CUDA stream이 다르다는 사실만으로 하나의 TensorRT `IExecutionContext`를 동시에 enqueue해도 안전해지는 것은
아니다. execution context의 optimization profile, tensor address, shape, workspace 상태는 host-side mutable
state이므로 shared-context 모드에서는 prefill CUDA event가 끝난 뒤에 decode의 prepare와 enqueue를 시작한다.

```text
prefill stream: [prepare][enqueue]----[prefill done event]
                                                  |
                                                  | host event completion
                                                  v
decode stream:                              [prepare][enqueue]----[decode done event]
```

두 phase를 실제로 겹치는 independent-context 모드는 다음 세 identity가 모두 non-null이고 서로 달라야 한다.

- TensorRT execution context
- context workspace
- phase-local I/O buffer owner

```text
prefill: Context P + Workspace P + I/O P  [==============]
decode:  Context D + Workspace D + I/O D       [==============]
```

`PhaseExecutionSafetyContract`가 이 조건을 construction time에 검증한다. shared 모드는 동일한 execution-context
identity만 허용하고, independent 모드는 context/workspace/I/O 중 하나라도 alias이면 실행 전에 거부한다. identity는
주소 기반 선언이므로 caller는 실제 소유 object의 lifetime과 주소를 정확히 전달해야 한다.

## 왜 CUDA stream wait만으로 부족한가

`cudaStreamWaitEvent()`는 device work의 선후관계만 만든다. 두 host callback이 같은 `IExecutionContext`에 대해
profile 전환, input shape 설정, tensor binding을 동시에 수행하는 것은 막지 못한다. 그래서 shared 모드에서는
decode callback 자체를 prefill event completion 뒤로 미룬다. queue와 stream은 분리되어 있지만 TensorRT context
mutation은 host에서 직렬화된다.

## Gemma 4 E2B INT4 indexed smoke

RTX 3080, TensorRT 11.0.0, BS2, input 128, prefill chunk 128, past KV 128, warmup 1, measurement 3 조건이다.

| 모드 | sequential median | scheduled median | makespan speedup |
|---|---:|---:|---:|
| independent context | 30.9391 ms | 27.0954 ms | 1.1419x |
| shared context | 38.5700 ms | 38.4420 ms | 1.0033x |

shared 결과는 의도대로 overlap 속도 향상을 주장하지 않는다. 두 queue/stream/event 구조를 유지하면서 하나의 context를
안전하게 사용하는 fallback이다. independent 결과는 작은 smoke sample이며 최종 throughput/latency gate는 아니다.
raw samples는 `/tmp/gemma4-e2b/perf/phase/safety-independent.csv`와
`/tmp/gemma4-e2b/perf/phase/safety-shared.csv`에 있다.

## 검증 범위

- shared context의 decode callback이 prefill batch completion 전에 호출되지 않음
- independent context에서 두 callback을 event wait 전 enqueue할 수 있음
- aliased context/workspace/I/O contract가 거부됨
- 실제 Gemma indexed engine의 shared/independent 실행 성공

다음 단계는 별도 encoder queue/stream/event와 encoder-to-prefill handoff를 추가하고, encoder가 LLM 실행과 겹칠 때도
동일한 resource identity 원칙을 적용하는 것이다.
