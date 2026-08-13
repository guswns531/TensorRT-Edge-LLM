# Cosmos packed prefill engine E2E와 real-request 결과

## 결론

Cosmos-Reason2-2B FP16 indexed-paged engine에 true packed prefill I/O 계약을 export부터 TensorRT build,
independent-context serving까지 연결했다. 서로 다른 logical request의 fixed-128 chunk는 더 이상
`[P, 128, H]` dense batch가 아니라 `[1, sum(tokens), H]` token carrier로 실행된다. KV cache metadata와
stable slot mapping은 logical batch `P`를 그대로 유지한다.

RTX 3080 10GB에서 동일한 96-request mixed trace를 P4/D32와 P8/D32로 각각 3회 반복했다. 기존
indexed-paged와 packed engine은 `maxBatchSize=32`, `maxPrefillBatchSize=8`, `maxDecodeBatchSize=32`,
256 page bundles, independent TensorRT contexts, CUDA graph off 조건을 맞췄다. P8/D32 3회 median에서 packed는
생성 처리량을 5.92% 높였고 TTFT p95를 8.14%, TPOT p95를 8.33%, E2E p95를 6.38% 낮췄다.

## 실행 계약

```text
logical rows:       request A       request B       request C
chunk tokens:          128              96              64
stable KV slot:          7               2              11
                         |               |               |
                         +------- logical batch P=3 -----+

token carrier:      [ A:128 | B:96 | C:64 ]
TensorRT shape:     inputs_embeds [1, 288, H]
metadata shapes:    context_lengths [3]
                    kv_slot_ids [3]
                    kv_page_ids [3, pagesPerSequence]
last-token IDs:     [127, 223, 287] shaped [1, 3]
```

`InferenceDims`에 `tokenBatch`를 추가해 token tensor의 carrier batch와 KV metadata의 logical batch를 분리했다.
legacy와 dense-ragged recipe는 `tokenBatch=batch`이므로 기존 동작이 유지된다. packed recipe만
`batch=P`, `tokenBatch=1`, `seqLen=sum(tokens)`, `selectLen=P`를 사용한다.

prefill adapter는 host request row를 stable slot 순서와 무관하게 한 token carrier에 이어 붙이고, row별
`context_lengths`, global last-token offset, `kv_slot_ids`를 별도로 staging한다. 실행 후 원래 request tensor와
slot/length binding을 정확히 복원한다. output logits의 storage는 `[1,P,V]`에서 `[P,V]`로 reinterpret해 기존
greedy sampler가 그대로 사용한다.

attention plugin은 `enable_packed_prefill` 직렬화 필드를 가진다. packed prefill에서는 cumulative Q/KV sequence
length를 생성하고, row별 prefix length와 stable page table을 이용해 RoPE/KV write를 수행한다. 이후 필요한 KV
prefix만 compact workspace로 gather해 unpadded FMHA_v2에 전달한다. 실제 KV allocator는 여전히 indexed-paged이며
KV page를 재배치하거나 request 종료 시 compact하지 않는다.

## 구현 위치

- `tensorrt_edgellm/config.py`, `scripts/export.py`
  - opt-in `--packed-prefill` export 계약과 config 기록
- `tensorrt_edgellm/models/default/modeling_default.py`, `models/ops.py`
  - independent `token_batch` dynamic dimension과 attention op attribute
- `tensorrt_edgellm/onnx/{dynamo_translations.py,onnx_custom_schemas.py}`
  - ONNX custom-op에서 TensorRT plugin field 전달
- `cpp/builder/llmBuilder.cpp`
  - context profile을 `[1, 1..P*128, H]`, last-token profile을 `[1, 1..P]`로 생성
- `cpp/runtime/config/{inferenceDims,llmEngineConfig}.{h,cpp}`
  - logical/token batch dimension recipe 분리
- `cpp/runtime/exec/registryBuilder.cpp`
  - embeds/deepstack/last-token tensor만 token carrier dimension 사용
- `cpp/runtime/scheduling/phasePrefillContextBatchAdapter.{h,cpp}`
  - fixed-128 row concatenation, global select index, binding restore, logits view
- `cpp/runtime/scheduling/phaseQueueScheduler.{h,cpp}`
  - fixed-128 indexed-paged 조건을 강제하는 opt-in scheduler gate
- `examples/llm/llm_phase_bench.cpp`
  - dynamic serving callback과 fixed warmup 모두 packed recipe 사용
- `scripts/cosmos_reason2/run_real_request_kv_matrix.py`
  - engine config의 `packed_prefill`을 읽어 harness flag를 자동 선택

## export/build/E2E 검증

실제 Cosmos checkpoint로 다음 순서를 완료했다.

1. FP16 indexed-paged + packed ONNX export
2. ONNX의 28개 attention node가 모두 `enable_packed_prefill=1`인지 확인
3. P8/D32 TensorRT engine build
4. independent prefill/decode TensorRT context로 12-request E2E 실행
5. P1/P4 batch-invariance와 96-request 반복 성능 측정

ONNX dynamic shapes는 `inputs_embeds=[token_batch,seq_len,2048]`,
`context_lengths=[batch]`, `kv_slot_ids=[batch]`, `last_token_ids=[token_batch,num_selected]`로 확인했다.
TensorRT parser와 28 attention plugin 생성, deepstack profile, engine serialization이 모두 성공했다.

초기 E2E에서 real-trace warmup만 옛 `[P,128,H]` shape를 사용해 profile reject가 발생했다. fixed benchmark
warmup도 packed recipe를 사용하도록 고친 뒤 12/12 request, 5 prefill dispatch, 130 decode dispatch가 정상 완료됐다.

## 정확성 해석

동일 engine 내부의 batch-invariance 결과는 다음과 같다.

- 기존 indexed-paged: P1 결과와 P4 결과 12/12 text 일치
- packed: P1 결과와 P4 결과 12/12 text 일치
- 기존 engine과 packed engine 사이: 4/12 text 일치

마지막 차이는 packed row mapping 문제가 아니다. P1에서도 동일한 4/12만 일치하며, 각 engine 내부에서는 P를
바꿔도 출력이 전부 같다. 두 engine을 별도로 build하면서 profile/plugin contract가 달라져 TensorRT tactic과 FP16
reduction 순서가 바뀐 결과로 판단한다. 따라서 현재 correctness evidence는 packed P1 reference에 대한 P4
batch-invariance다. cross-engine greedy token exact gate를 다시 적용하려면 같은 timing cache/tactic replay를 사용한
재빌드 또는 layer/logit tolerance 비교를 추가해야 한다.

## 96-request 성능

trace는 prompt 길이가 섞인 96 request, output token 합 8,320, 고정 arrival offset이며 모든 실행에서 EOS를
무시해 output work를 동일하게 했다.

### P4/D32, token budget 512

| metric | indexed-paged | packed | change |
| --- | ---: | ---: | ---: |
| generated token/s | 2499.8 | 2645.4 | +5.82% |
| TTFT median | 1024.3 ms | 939.5 ms | -8.28% |
| TTFT p95 | 2142.4 ms | 1972.4 ms | -7.94% |
| TPOT median | 10.553 ms | 9.911 ms | -6.09% |
| TPOT p95 | 11.460 ms | 10.642 ms | -7.14% |
| E2E median | 1829.6 ms | 1707.7 ms | -6.66% |
| E2E p95 | 2951.3 ms | 2770.5 ms | -6.13% |

### P8/D32, token budget 1024

| metric | indexed-paged | packed | change |
| --- | ---: | ---: | ---: |
| generated token/s | 2539.5 | 2689.9 | +5.92% |
| TTFT median | 995.2 ms | 927.8 ms | -6.77% |
| TTFT p95 | 2083.2 ms | 1913.6 ms | -8.14% |
| TPOT median | 10.377 ms | 9.738 ms | -6.16% |
| TPOT p95 | 11.410 ms | 10.460 ms | -8.33% |
| E2E median | 1768.2 ms | 1632.5 ms | -7.67% |
| E2E p95 | 2891.8 ms | 2707.3 ms | -6.38% |

kernel-group median은 P8에서 prefill engine 18.792ms에서 13.273ms로 29.4% 감소했다. prefill prepare는
0.155ms에서 0.152ms로 사실상 같고 decode engine도 7.375ms와 7.401ms로 중립적이다. 즉 E2E 개선은 host packing
비용이나 decode 변화가 아니라 packed attention의 prefill GPU 비용 감소에서 나온다.

P8 packed가 P4 packed보다 처리량 약 1.7%, TTFT p95 약 3.0%, E2E p95 약 2.3% 추가 개선되어 이 trace의 현재
최선은 P8/D32다. 다만 더 큰 prefill overlap은 decode-heavy trace에서 TPOT을 악화시킬 수 있으므로 production
scheduler에는 기존 direct-cost/TPOT hard guard를 유지해야 한다.

## 메모리와 제한

동일 maxBatch=32 engine의 실행 직전 GPU 사용량은 기존 8696.9MiB, packed 8920.9MiB였다. packed가 224MiB
더 사용하고 free headroom은 1177.4MiB에서 953.4MiB로 줄었다. 두 경우 모두 실행 후 증가는 6MiB였고 request
진행 중 cache compaction이나 동적 대형 allocation은 없었다.

추가 사용량은 KV page pool 증가가 아니다. 두 engine 모두 256 page bundles를 사용한다. packed ONNX/profile에
대해 TensorRT가 생성한 engine plan/tactic 차이와 context resource 차이다. 현재 953MiB headroom은 목표 512MiB를
통과하지만, CUDA graph variant를 많이 capture하기 전에 별도 memory budget을 적용해야 한다.

v1 제한은 다음과 같다.

- fixed prefill chunk 128, FP16 indexed-paged KV, head dimension 128
- text-only attention; 실제 visual/deepstack feature row는 packed batch에서 거부
- sliding-window, speculative decoding, FP8 KV 미지원
- cross-engine greedy exact equality는 아직 gate 미통과

artifact는 `.local/cosmos-reason2-2b/packed-prefill-20260813/`에 있으며, 공정한 engine은
`.local/cosmos-reason2-2b/engine-fp16-packed-p8-d32-mb32-b256/`이다.

## 다음 단계

1. packed P1/P2/P4/P8 × prefix bucket별 prefill kernel cost table을 생성한다.
2. 기존 direct-overlap table에 `layout=dense|packed` 축을 추가한다.
3. P8 packed를 throughput-balanced scheduler에 넣고 48 short, 288 balanced/decode-heavy trace를 3회씩 gate한다.
4. timing cache/tactic replay 또는 selected-logit dump로 cross-engine correctness gate를 분리한다.
5. 953MiB headroom 안에서 phase CUDA graph capture budget을 다시 탐색한다.
