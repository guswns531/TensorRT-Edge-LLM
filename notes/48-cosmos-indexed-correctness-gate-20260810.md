# Cosmos indexed KV correctness gate (2026-08-10)

## 결론

Cosmos Reason2-2B FP16 text decoder의 legacy fixed-linear engine과 indexed-linear
engine을 같은 token trajectory에서 비교했다. CUDA graph를 끄고 TensorRT의 동일한
aux-stream 설정을 유지한 상태에서 BS2와 BS4 모두 8 round logits cosine gate
`0.999`를 통과했다.

stable slot이 요청 종료 순서에 따라 `4 -> 3 -> 2 -> 1`로 줄어드는 lifecycle도
별도 GPU test로 고정했다. Compute Sanitizer는 0 error였고, 해당 구간의 Nsight
Systems trace에는 KV compaction kernel과 device-to-device copy가 모두 없었다.

따라서 Cosmos FP16 attention-only indexed KV v1의 correctness gate는 통과로
판정한다. VLM maxBatch 16 구성의 512 MiB headroom 미달은 별도의 capacity/배포
gate이며 이 수치 정확성 판정에는 포함하지 않는다.

## 비교 방법

서로 별도로 build한 TensorRT engine은 tactic 차이 때문에 아주 가까운 logits에서
greedy token이 달라질 수 있다. token exact match만으로 KV 주소 오류를 판정하면
false negative가 생기므로 다음 순서를 사용했다.

1. legacy engine을 `--noCudaGraph`와 `EDGELLM_IGNORE_EOS=1`로 8 round 실행한다.
2. `EDGELLM_DUMP_LOGITS_KVCACHE_LAYERS=0`으로 logits, context length, sampled token만
   저장한다. 0은 이번 변경에서 추가한 logits-only 모드이며 큰 KV tensor를 저장하지
   않는다.
3. legacy sampled token을 `EDGELLM_FORCE_TOKENS_FILE` 형식으로 추출한다.
4. indexed engine에 같은 token을 teacher-forcing하고 동일한 round를 실행한다.
5. 각 round와 row의 전체 vocabulary logits cosine을 비교한다. context length와
   tensor shape는 exact match가 필수다.

비교 도구는 `scripts/cosmos_reason2/compare_indexed_logits.py`이다. `extract-tokens`와
`compare` 두 subcommand를 제공하며 기본 gate는 minimum cosine 0.999이다.

## 결과

### Legacy 대 indexed

| workload | round/row 비교 수 | worst cosine | max abs | 자체 greedy 일치 |
| --- | ---: | ---: | ---: | ---: |
| BS2, 8 rounds | 16 | 0.99997274 | 0.203125 | 14/16 |
| BS4, 8 rounds | 32 | 0.99997376 | 0.218750 | 31/32 |

두 workload 모두 prefill round 0 logits는 byte-exact였다. 이후 max absolute error는
별도 engine tactic의 FP16 누적 차이를 포함한다. greedy mismatch가 있던 row도
legacy token을 입력한 다음 round에서 cosine이 계속 0.9999 이상이므로 cache
trajectory가 분리되거나 잘못된 physical slot을 읽은 징후는 없었다.

### 동일 indexed engine 반복 결정성

동일 engine, 입력, forced token, CUDA graph off, aux stream 2개 조건으로 indexed
실행을 반복했다.

- 8 rounds x BS2의 logits max absolute error: `0`
- worst cosine: `1.0`
- 자체 greedy token: `16/16`

즉 현재 engine 내부 aux-stream 실행 자체는 이 workload에서 byte-exact하게
재현됐다. legacy/indexed의 작은 차이는 run-to-run race가 아니라 두 engine을
별도로 build할 때 선택된 tactic 또는 indexed graph 차이로 해석한다.

## `4 -> 3 -> 2 -> 1` stable-slot 검증

`PhaseContextServingFacadeTest.KeepsStableSlotsWhileDecodeBatchShrinksFourToOne`은
physical slot `[0, 1, 2, 3]`에 output budget `[1, 2, 3, 4]`인 요청을 넣는다.
관측된 decode slot batch는 다음과 정확히 일치한다.

```text
[0, 1, 2, 3] -> [1, 2, 3] -> [2, 3] -> [3]
```

종료한 row만 logical batch에서 빠지며 surviving slot ID는 바뀌지 않는다. 마지막
global physical KV length는 `[3, 4, 5, 6]`으로 남고 네 slot은 모두 free-list로
반환됐다. 이는 cache tensor를 앞쪽으로 당기지 않고 ownership metadata만 제거한
결과다.

관련 indexed KV/lifecycle 회귀 묶음은 11 suite, 21 test가 모두 통과했다.

## 메모리 안전성과 copy 검증

Compute Sanitizer 2026.2 memcheck에서 다음 8 test를 함께 실행했다.

- multi-head page-major RoPE write와 gather round trip
- non-identity stable-slot gather
- logical compaction 뒤 physical length 유지
- slot reserve/release, exhaustion, double-free, deterministic reuse
- `4 -> 3 -> 2 -> 1` serving lifecycle

결과는 `ERROR SUMMARY: 0 errors`다.

Nsight Systems 2026.3으로 정확히 같은 `4 -> 3 -> 2 -> 1` test를 캡처했다.

- 실행 kernel: indexed length gather 5회, increment 5회, clear 1회
- `compactKVCacheKernel` / `compactKVCacheBatchedKernel`: 0회
- CUDA D2D memcpy: 0회
- 관측된 copy: test 검증용 D2H 5회, metadata H2D 11회

따라서 eviction 자체가 KV payload copy를 발생시키지 않는다는 것을 trace로
확인했다.

## 구현 변경

- `cpp/runtime/debug/layerDebugger.cpp`
  - layer count 0을 logits-only dump로 허용한다.
- `examples/llm/llm_inference.cpp`
  - `--noCudaGraph`를 추가해 base/TTS decode graph capture를 opt-out할 수 있다.
- `scripts/cosmos_reason2/compare_indexed_logits.py`
  - reference token 추출과 teacher-forced logits cosine 비교를 제공한다.
- `unittests/phaseDispatchWorkerTest.cpp`
  - stable slot `[0,1,2,3] -> [1,2,3] -> [2,3] -> [3]` 회귀를 고정한다.

## 결과 artifact

모델과 측정 결과는 Git에 넣지 않고 `.local` 아래에 유지한다.

- `.local/cosmos-reason2-2b/correctness-20260810/legacy/edgellm_dump.safetensors`
- `.local/cosmos-reason2-2b/correctness-20260810/indexed/edgellm_dump.safetensors`
- `.local/cosmos-reason2-2b/correctness-20260810/indexed-repeat/edgellm_dump.safetensors`
- `.local/cosmos-reason2-2b/correctness-20260810/bs4-legacy/edgellm_dump.safetensors`
- `.local/cosmos-reason2-2b/correctness-20260810/bs4-indexed/edgellm_dump.safetensors`
- `.local/cosmos-reason2-2b/correctness-20260810/eviction-4to1.nsys-rep`
- `.local/cosmos-reason2-2b/correctness-20260810/eviction-4to1.sqlite`
