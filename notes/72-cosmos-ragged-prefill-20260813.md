# Cosmos ragged prefill 구현과 vLLM 격차 축소

## 결론

Cosmos-Reason2-2B FP16 indexed-paged independent-context 경로에 opt-in ragged prefill을 연결했다. 서로 다른
길이의 text chunk를 하나의 `[B, Smax]` TensorRT 입력으로 right-padding하고, attention에는 각 row의 실제 길이를
전달한다. KV cache 소유권과 page allocator는 바꾸지 않았다.

이 변경은 짧은 48-request 부하에서는 중립적이었지만, prompt 길이가 섞인 288-request balanced 부하에서는
uniform prefill 대비 다음 개선을 보였다.

| metric | uniform | ragged | change |
| --- | ---: | ---: | ---: |
| generated token/s | 3648.2 | 3941.2 | +8.03% |
| TTFT p95 | 5162.5 ms | 4658.8 ms | -9.76% |
| TPOT p95 | 16.705 ms | 15.633 ms | -6.42% |
| E2E p95 | 6093.2 ms | 5577.7 ms | -8.46% |

동일 trace의 vLLM median은 4109.8 token/s, TTFT p95 4156.2ms, TPOT p95 18.707ms, E2E p95
5200.5ms였다. 따라서 처리량 격차는 uniform의 -11.23%에서 ragged의 -4.10%로 줄었다. 현재 경로의 TPOT은
vLLM보다 16.4% 낮지만 TTFT와 E2E는 각각 약 12.1%, 7.3% 높다.

## 동작 방식

```text
prefill queue
   |
   | exact-length row 우선, 그 다음 긴 final chunk
   v
[128, 128, 96, 64] -- useful=416, padded=512, efficiency=81.25%
   |
   v
token_ids [B, 128]       context_lengths [128, 128, 96, 64]
   |                               |
   | right padding                 | attention logical bounds
   v                               v
independent prefill TensorRT context / prefill CUDA stream
   |
   +-- RoPE Q: padding row는 zero
   +-- KV write: padding token은 skip
   +-- FMHA/FFPA: row별 cu-seqlen으로 padding mask
   v
stable indexed-paged KV slots; commit length는 row별 실제 token 수
```

핵심은 TensorRT의 물리 shape는 여전히 dense라는 점이다. 이번 단계는 vLLM식 packed token layout으로 engine
contract 전체를 바꾼 것이 아니라, final chunk를 다른 full chunk와 합쳐 prefill dispatch 수와 queue 대기를 줄이는
안전한 중간 단계다.

## 구현 위치

- `cpp/runtime/scheduling/phaseQueueScheduler.{h,cpp}`
  - `enableRaggedPrefillBatching` opt-in
  - initial/continuation을 섞지 않는 호환성 규칙
  - exact-length 우선 정렬과 padded-token budget
  - dynamic cost 선택 시 padded shape의 GPU cost와 실제 useful-token efficiency 분리
- `cpp/runtime/scheduling/phasePrefillContextBatchAdapter.{h,cpp}`
  - row별 실제 token 복사, 나머지 zero-fill
  - row별 실제 commit length 유지
- `cpp/kernels/posEncoding/applyRopeWriteKV.{h,cu}`
  - row별 input length 이후의 Q를 zero 처리
  - padding token의 RoPE cache read와 K/V page write를 건너뜀
- `cpp/plugins/attentionPlugin/attentionPlugin.cpp`
  - own-KV prefill의 FFPA, CuTe DSL, FMHA 경로에 실제 input length 전달
- `cpp/runtime/scheduling/phaseDispatchWorker.cpp`
  - useful, padded, padding token과 packing efficiency 기록
- `examples/llm/llm_phase_bench.cpp`
  - `--raggedPrefillBatching`과 row별 context/select-index binding
- `scripts/cosmos_reason2/build_prefill_wavefront_cost_model.py`
  - schema v4에서 padded shape와 observed packing efficiency 저장

## 안전 계약

- 기본값은 off라 기존 uniform engine/runtime 동작은 유지한다.
- text chunk 중 `allowChunkedPrefill=true`인 row만 ragged batch에 포함한다.
- multimodal/vision atomic row는 padding하지 않고 BS1을 유지한다.
- initial chunk와 continuation chunk는 같은 batch에 넣지 않는다.
- token budget은 useful-token 합이 아니라 실제 실행 footprint인 `B * Smax`에 적용한다.
- KV 길이는 padded width가 아니라 row별 실제 token 수만 증가한다.
- paged KV의 미할당 다음 page를 padding token이 참조하면 테스트가 실패하도록 CUDA test를 추가했다.

최종 검증에서는 scheduler, packed adapter, dispatch worker, independent executor, paged RoPE/KV write를 포함한
50개 GPU/C++ test가 모두 통과했다. 재빌드한 plugin으로 실행한 Cosmos 12-request P4/D64 independent-context
smoke도 12/12 완료됐고, 이전 검증 run과 generated text가 12/12 정확히 일치했다. 이 결과는
`.local/cosmos-reason2-2b/ragged-prefill-20260813/final-safety-smoke/`에 저장했다.

## 실험 해석

### 96-request mixed trace

P8/D64, chunk 128, padded-token budget 256에서 3회 median은 다음과 같다.

| metric | uniform | ragged | change |
| --- | ---: | ---: | ---: |
| generated token/s | 2261.9 | 2408.0 | +6.46% |
| TTFT p95 | 2598.0 ms | 2349.0 ms | -9.58% |
| TPOT p95 | 10.611 ms | 9.462 ms | -10.83% |
| E2E p95 | 3518.0 ms | 3320.0 ms | -5.63% |
| dispatch count | 449 | 416 | -7.35% |
| prefill dispatch count | 44 | 41 | -6.82% |

ragged packing efficiency median은 97.86%였고 96/96 request의 generated token이 세 반복 모두 uniform과
일치했다.

### 48-request short trace

P4/D64에서는 uniform 1604.0 token/s, ragged 1595.6 token/s로 -0.52%였다. dispatch 수가 동일했고 packing
efficiency도 97.25%여서 합칠 final chunk가 거의 없었다. 이 기능은 항상 빨라지는 최적화가 아니라 길이 분산과
동시 queue depth가 있을 때 효과가 생긴다.

### overlap cap 상호작용

288-request trace에서 ragged token budget을 256으로 늘리고 기존 `maxOverlapPrefillTokens=128`을 유지하면 useful
token 합이 128을 넘는 ragged batch가 decode와 겹치지 못했다. 그 결과 overlap dispatch가 273에서 84로 줄고
처리량이 약 3207 token/s까지 하락했다. overlap cap을 같은 padded budget인 256으로 맞추자 3941.2 token/s로
회복했다.

따라서 두 설정을 무조건 자동으로 묶으면 안 된다. 큰 overlap은 TPOT을 해칠 수 있으므로 cost table과 direct-overlap
hard guard가 허용한 shape만 겹쳐야 한다. 현 단계 권장은 다음과 같다.

1. ragged off에서는 기존 cap 128을 유지한다.
2. ragged on, token budget 256에서는 direct overlap cost가 있는 경우에만 cap 256을 사용한다.
3. cost coverage가 없거나 decode pressure가 높으면 decode-only 또는 작은 prefill batch로 되돌린다.

## 남은 vLLM 격차와 다음 순서

ragged prefill은 balanced workload 처리량 격차의 약 7%p를 회수했지만 dense `[B,Smax]` 실행과 static TensorRT
profiles는 그대로다. 다음 우선순위는 다음과 같다.

1. schema-v4 direct overlap table을 ragged P1/2/4/8, D1/2/4/8/16/32/64에 대해 다시 생성한다.
2. overlap cap을 고정값이 아니라 direct-cost coverage, decode slack, packing efficiency로 제한한다.
3. prompt length bucket의 대기 시간을 짧게 두는 bounded batch formation window를 추가한다.
4. TensorRT attention contract의 truly packed/varlen 입력은 별도 실험 branch에서 구현하고 dense ragged와 A/B한다.
5. 48-request short, 96-request mixed, 288-request balanced/decode-heavy를 각 3회 이상 반복해 3% gate를 적용한다.

실험 artifact는 `.local/cosmos-reason2-2b/ragged-prefill-20260813/`에 있다.
