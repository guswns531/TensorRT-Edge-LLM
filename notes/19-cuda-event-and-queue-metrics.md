# CUDA event와 queue metrics

## 측정 경계

각 `PhaseDispatchWorker` dispatch는 다음 CUDA event를 가진다.

```text
dispatchStart (prefill stream)
  |-- prefillStart -> pack/H2D/PLE/TRT/sampling -> prefillDone
  |-- decode stream wait(dispatchStart)
      decodeStart -> pack/H2D/PLE/TRT/sampling -> decodeDone
```

shared-context serialized mode에서는 prefill event completion과 host-side completion이 끝난 뒤 decodeStart를
기록한다. independent-context mode에서는 두 phase가 dispatchStart 이후 각 stream에서 바로 실행될 수 있다.
event는 timing-enabled이며 worker가 두 completion을 모두 확인한 후에만 elapsed time을 읽는다.

## `PhaseDispatchMetrics`

- `dispatchIndex`, `kind`
- prefill/decode batch size
- prefill token 합계와 decode token 수
- 선택된 batch 중 가장 오래 기다린 row의 queue wait microseconds
- prefill/decode GPU milliseconds
- dispatchStart에서 마지막 phaseDone까지의 makespan
- `1 - makespan / (prefill + decode)`로 계산하고 `[0,1]`로 clamp한 overlap ratio

metrics는 `onMetrics` callback과 `lastMetrics()`로 노출된다. lifecycle과 serving facade는 같은 record를
`onDispatchMetrics`까지 전달한다. callback은 CUDA event completion 이후 host thread에서 실행되므로 그 안에서
GPU buffer를 읽지 않아도 된다.

## Queue timestamp 의미

요청이 prefill/decode queue에 들어가거나 chunk/decode completion 후 다시 queue에 들어갈 때
`steady_clock::now()`를 저장한다. batch pop 시 timestamp를 제거하고 residence time을 계산한다. 따라서 이 값은
요청 전체 latency가 아니라 “이번 scheduling turn을 기다린 시간”이다. pending admission wait는 아직 별도이며,
다음 metrics-policy 단계에서 필요하면 admission metric으로 추가한다.

## 실제 Gemma smoke

RTX 3080, Gemma 4 E2B INT4 indexed engine, BS2, prompt 512, chunk 128 조건에서 실제 serving dispatch CSV를
생성했다.

```text
dispatch 1: prefill BS2 / 256 tokens / queue 6.196 us / GPU 127.129 ms
dispatch 2: prefill BS2 / 256 tokens / queue 8.914 us / GPU 28.025 ms
...
dispatch 5: decode BS1 / 1 token / queue 3.989 us / GPU 654.883 ms
```

첫 prefill/decode의 큰 시간은 engine/profile/kernel warmup이 포함된 smoke record다. scheduler 학습값으로 쓰기
전에 warmup sample을 버리거나 EWMA로 완화해야 한다. raw record는
`/tmp/gemma4-e2b/perf/phase/event-metrics-smoke-dispatch.csv`, 기존 phase summary는
`event-metrics-smoke.csv`에 저장했다. 동일 run의 timed synthetic workload speedup은 1.1305x였지만 1회
sample이므로 성능 gate 수치가 아니다.

## 주의점

- CUDA event 시간은 event 사이에 해당 stream에 enqueue된 PLE, TensorRT aux-stream join, sampling까지 포함한다.
- host adapter scatter와 callback 실행시간은 GPU phase 시간에 포함되지 않는다.
- queue wait은 host scheduling pressure, GPU time은 device work를 나타내므로 서로 대체하지 않는다.
- overlap ratio는 두 phase가 모두 있는 dispatch에서만 의미가 있다. 한 phase만 있으면 0이다.
- timing-enabled event는 disable-timing event보다 약간 비싸다. 최종 3% gate에서 계측 on/off 옵션 필요성을
  다시 판단한다.
