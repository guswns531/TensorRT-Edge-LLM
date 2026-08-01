# Encoder queue/stream/event와 prefill handoff

## 구현 경계

`PhaseEncoderDispatchWorker`는 encoder를 기존 prefill/decode 실행과 독립적으로 batching하고 CUDA event 완료 뒤
prefill queue로 넘긴다.

```text
request + stable KV slot lease
              |
              v
     bounded encoder queue
              |
         batch pop
              |
              v
 encoder context / workspace / I/O
              |
        encoder stream
              |
      encoder done event
              |
    host output finalization
              |
              v
 PhaseWorkItem(request, prompt, slot)
              |
              v
         prefill queue
```

encoder queue는 FIFO이고 configurable max batch 및 queue capacity를 가진다. 한 request ID는 queued 또는 in-flight
상태에서 중복 submit할 수 없다. queued request는 취소할 수 있지만 in-flight request는 output buffer와 slot
lifetime을 event 전까지 유지해야 하므로 취소를 연기한다.

## 실제 encoder 연결점

모델별 구현은 두 callback을 제공한다.

- `enqueueEncoder(batch, stream)`: input packing, tensor binding, TensorRT encoder enqueue
- `completeEncoder(item)`: event 이후 encoder output을 LLM input context에 연결하고 initial `PhaseWorkItem` 반환

worker는 반환된 request ID와 stable KV slot이 원래 encoder item과 같은지 검증한 후에만 prefill scheduler에
admit한다. 따라서 slot lease는 encoder부터 decode 종료까지 이동하지 않는다.

## 동시 실행 안전성

`PhaseEncoderExecutionSafetyContract`는 encoder의 context/workspace/I/O가 동시에 실행할 모든 LLM resource와
각각 다르고 non-null인지 construction time에 검사한다. stream만 분리하고 context workspace를 공유하는 구성은
거부한다.

```text
encoder: Context E + Workspace E + I/O E  [==========]
prefill: Context P + Workspace P + I/O P      [==========]
decode:  Context D + Workspace D + I/O D          [==========]
```

Gemma 4 E2B text-only engine에는 실행할 multimodal encoder가 없다. 따라서 이번 단계의 GPU 검증은 encoder callback에
실제 `cudaMemsetAsync` work를 enqueue하고 event 완료 전후의 batching, cancellation, metrics, prefill handoff를
검증한다. VLM engine 연결 시 worker나 scheduler를 바꾸지 않고 callback과 resource identity만 교체한다.

## v1 제한

- encoder worker 하나당 in-flight batch는 하나다.
- encoder batching은 FIFO이며 shape bucketing은 모델별 callback 또는 후속 batch policy가 담당한다.
- encoder output tensor lifetime은 callback owner가 prefill 소비 완료까지 보장한다.
- 세 phase의 통합 deadline policy와 SM partition backend는 별도 후속 단계다.
