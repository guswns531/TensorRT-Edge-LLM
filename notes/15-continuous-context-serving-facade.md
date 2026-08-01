# Continuous request context serving facade

이번 단계는 request별 `DecodingInferenceContext`와 continuous phase queue 사이의 production 경계를 연결한다.
`PhaseContextServingFacade`가 source context row를 복사하지 않고 borrowed reference로 유지하며 stable physical KV slot lease,
prefill/decode 전이, packed decode, 완료 scatter와 terminal release를 한 lifecycle로 관리한다.

## 요청과 GPU 데이터 흐름

```text
submit(request A context row 0) -> free-list slot 0 --+
submit(request B context row 0) -> free-list slot 1 --+--> prefill queue
                                                       |
                                      first chunk: slot length만 0으로 clear
                                      later chunk: 누적 length 유지
                                                       |
                                         prefill stream + CUDA event
                                                       v
                                                   decode queue
                                                       |
                  source A row -- slot 0 --+            |
                  source B row -- slot 1 --+--> packed rows [0, 1]
                                                slot_ids [0, 1]
                                                lengths [LA, LB]
                                                       |
                                          decode stream / TensorRT XQA
                                                       |
                                             decode-done CUDA event
                                                       v
                                     token/finished state를 source row로 scatter
                                                       |
                                   unfinished: decode queue에 다시 삽입
                                   finished: slot을 free-list에 반환
```

prefill queue와 decode queue는 서로 다른 batch size와 정책을 가진다. `maxPrefillChunkTokens`가 설정되면 prompt는
여러 turn으로 나뉘며, 새 slot의 global length는 `tokenOffset == 0`인 첫 chunk 직전에만 같은 prefill stream에서
0으로 만든다. continuation chunk에서 다시 clear하지 않으므로 KV length가 누적된다.

한 decode dispatch는 vanilla autoregressive token 한 개를 생성하는 계약이다. adapter는 active request row를
임시 context에 pack하고 physical KV를 복사하지 않는다. CUDA event 완료 뒤 callback이 token 상태를 갱신하면
source context로 scatter하고, 요청별 종료 판단 후 slot을 반환한다. cancel과 submit 실패도 source registration과
slot lease를 함께 정리한다.

## Stable indexed-linear의 할당과 파편화

```text
고정 GPU 할당 (layer마다 한 번)
slot 0: [K/V, capacity 2048]  <- request C (A 종료 후 재사용)
slot 1: [K/V, capacity 2048]  <- request B
slot 2: [K/V, capacity 2048]  <- free hole
slot 3: [K/V, capacity 2048]  <- request D

logical decode rows: [B, D, C]
kv_slot_ids:         [1, 3, 0]
```

- runtime 중 KV tensor를 allocate/free하지 않으므로 CUDA allocator 외부 파편화가 생기지 않는다.
- 중간 slot이 비어도 logical row만 compact하고 `kv_slot_ids`가 hole을 건너뛴다. surviving KV D2D copy는 0이다.
- slot을 재사용할 때 length만 0으로 만들고 KV byte 전체는 지우지 않는다. 올바른 length 경계에서는 이전 byte에
  접근할 수 없고 새 token이 사용 구간을 덮어쓴다. 보안상 물리적 zeroization이 필요하면 별도 정책이 필요하다.
- fixed-linear의 내부 파편화는 남는다. 짧은 요청도 slot 하나의 최대 capacity를 예약하므로 남은 token 공간을 다른
  요청이 사용할 수 없다. 대신 주소 안정성, O(1) free-list 재사용, 예측 가능한 VRAM을 얻는다.
- 모든 slot이 사용 중이면 현재 facade는 대기 admission queue를 내부에 만들지 않고 submit을 실패시킨다. 상위
  서버가 backpressure를 걸거나 terminal release 뒤 재시도해야 한다.

따라서 이 구현은 요청 순서 변화와 batch hole에는 강하지만, paged KV cache처럼 token 단위로 빈 공간을 공유하지는
않는다. 다음 메모리 효율 단계는 facade의 stable ownership 계약을 유지한 채 page/block allocator backend를
추상화하는 방식이 적합하다.

## TensorRT context와 stream 선택

기본 `kSharedSerialized`는 TensorRT execution context 하나를 유지하면서 prefill/decode queue와 CUDA stream을
분리한다. profile과 binding host state를 동시에 바꾸지 않도록 CUDA event 완료 뒤 다음 phase를 enqueue하므로
kernel은 안전하게 직렬 실행된다. 사용자가 원한 “context 추가 없이 queue별 batching”에 해당하는 모드다.

`kIndependentConcurrent`는 같은 engine에서 sibling execution context와 phase별 workspace/I/O를 사용해 실제
kernel overlap을 허용하는 선택 모드다. queue, slot allocator와 facade 계약은 두 모드에서 동일하다.

## RTX 3080 검증과 성능

환경은 Gemma 4 E2B INT4-AWQ indexed engine, FP16 KV, TensorRT 11.0.0.114, CUDA 13.3이다.
BS2 prefill input 512와 BS2 decode past KV 512에서 warmup 20회 후 100회를 측정했다.
facade의 request admission smoke는 timed sample 전에 실제 TensorRT prefill과 한 token packed decode를 실행한다.

| 실행 방식 | Sequential median / p95 | Scheduled median / p95 | 변화 |
|---|---:|---:|---:|
| context 하나, 두 stream event 직렬화 | 101.4292 / 101.9689 ms | 101.0094 / 101.9023 ms | -0.41% / -0.07% |
| sibling context, 실제 overlap | 94.8552 / 95.3252 ms | 90.1714 / 90.5400 ms | -4.94% / -5.02% |

이전 context-adapter 측정과 비교하면 새 facade 연결 후 independent mode의 sequential median/p95 변화는
-0.02%/-0.02%, overlap은 +0.08%/-0.11%로 모두 3% gate 안이다. raw CSV는 다음에 있다.

- `/tmp/gemma4-e2b/perf/phase/serving-facade-final-b2_i512_k512.csv`
- `/tmp/gemma4-e2b/perf/phase/serving-facade-shared-final-b2_i512_k512.csv`

GPU 단위 테스트는 17개 phase 관련 test가 통과했다. facade test는 1-token chunked prefill, 서로 다른 decode 종료
길이, 중간 slot release/reuse, stale length clear, cancellation과 exact TensorMap binding restore를 함께 검증한다.
GPU 메모리를 직접 다루는 6개 batch/adapter/facade test의 Compute Sanitizer memcheck 결과는 0 error다.

최종 `llm_basic` greedy 출력 SHA256은
`1918f649c96695ea807985d3e7a98c4257d3d429f3b9557277f4957472dfcb2a`로 기존 indexed baseline과 같다.

## 현재 남은 경계

facade는 decode source context의 pack/scatter를 production 형태로 연결했지만 prefill input tensor fan-in은 아직
callback 책임이다. 실제 서버에 붙일 다음 단계는 prompt length/chunk bucket별 prefill context adapter와 slot이
가득 찼을 때의 pending admission/backpressure 정책이다. 기존 `handleRequest()`와 speculative/MTP, multimodal,
streaming callback, LoRA, logprobs 경로는 변경하지 않았다.
