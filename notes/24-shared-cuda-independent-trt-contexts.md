# 공유 CUDA context와 분리 TensorRT context 구현 결과

## 구현된 topology

```text
process / device
└─ CUDA primary context (공유)
   ├─ prefill CUDA stream
   │  ├─ TensorRT IExecutionContext P
   │  ├─ user workspace P
   │  ├─ phase-local input/output P
   │  └─ Gemma PLE mutable output P
   └─ decode CUDA stream
      ├─ TensorRT IExecutionContext D
      ├─ user workspace D
      ├─ phase-local input/output D
      └─ Gemma PLE mutable output D

shared immutable state
├─ ICudaEngine / weights
├─ indexed-linear KV allocation (stable slot ownership)
└─ Gemma PLE embedding table
```

`EngineExecutor::createSibling()`은 같은 engine으로 새 `IExecutionContext`를 만든다. benchmark와 safety contract는
더 이상 `EngineExecutor` wrapper 주소를 대리 identity로 사용하지 않고 실제 `IExecutionContext*` 주소를 비교한다.
`PhaseDispatchWorker`는 두 stream의 CUDA context가 같은지, 그것이 device primary context인지 검증한다.

## PLE 분리 이유

Gemma 4 PLE embedding table은 inference 중 바뀌지 않아 공유할 수 있다. 반면 gather 결과 buffer는 prefill과 decode가
동시에 쓰므로 공유하면 data race가 된다. `createSibling()`은 table allocation만 `shared_ptr`로 공유하고 각 phase에
다른 output buffer와 tensor view를 만든다. shared TensorRT context fallback은 host에서 직렬화되므로 기존 output을
재사용한다.

## 실제 Gemma smoke 결과

RTX 3080, Gemma 4 E2B INT4 indexed engine, BS2/BS2, prompt 128, past KV 128, warmup 1, measurement 3 조건에서:

- CUDA context: prefill/decode 동일 주소 확인
- TensorRT context: prefill/decode 서로 다른 주소 확인
- independent TensorRT context median: sequential 30.6012 ms, concurrent 27.1404 ms, 1.1275x
- actual greedy serving facade: prefill부터 반복 decode와 terminal까지 성공
- shared TensorRT context fallback도 동일 context 주소와 직렬화 실행을 확인

이 값은 구조 smoke이지 최종 성능 결론이 아니다. 다음 continuous-load 실험에서 실제 arrival, queue saturation,
phase별 batch-size 분포, TTFT/E2E를 측정해야 한다.

## 현재 한계

- prefill/decode만 같은 CUDA context 검증을 직접 수행한다. encoder까지 합치는 3-phase coordinator는 후속 단계다.
- 한 worker가 동시에 유지하는 dispatch plan은 하나다. plan 내부의 prefill/decode만 independent 모드에서 overlap한다.
- CUDA graph, SM mask/Green Context, engine segmentation은 아직 범위 밖이다.
