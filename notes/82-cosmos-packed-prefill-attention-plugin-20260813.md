# Cosmos packed prefill attention plugin 연결

## 이번 단계의 결론

`AttentionPlugin`에 직렬화 가능한 `enable_packed_prefill` opt-in을 추가하고, normal prefill에서 다음 두 CUDA
경로를 하나의 enqueue로 연결했다.

```text
packed Q/K/V [1,T,*]
        |
        | context_lengths [B] -> cuSeqLens [B+1]
        v
packed RoPE + indexed-paged KV write
        |
        v
compact causal FMHA_v2 (isSPadded=false)
        |
        v
attention output [1,T,Hq,D]
```

기존 decode는 계속 `[B,1,*]`과 XQA를 사용한다. `enable_packed_prefill=0`인 legacy/indexed 엔진의 shape와
실행 분기는 바뀌지 않는다.

## logical batch와 tensor batch 분리

packed prefill에서 Q/K/V의 첫 축 `1`은 실제 batch가 아니다. 모든 valid token을 total-token 축에 이어 붙였다는
carrier 차원이다. 실제 batch `B`는 `context_lengths`의 길이로 얻는다.

```text
request lengths = [8,5,3]
cuSeqLens       = [0,8,13,16]

Q/K/V binding   = [1,16,*]   <- inputBatchSize = 1
context_lengths = [3]        <- runtimeBatchSize = 3
kv_slot_ids     = [3,0,2]    <- stable physical ownership
```

workspace, indexed page-list, KV active view, FMHA batch는 모두 `runtimeBatchSize=3`을 사용한다. output은 packed
token 순서를 유지하므로 `[1,16,Hq,D]`이다.

## 지원 범위와 fail-fast 조건

v1 plugin capability는 다음 조합만 받는다.

- indexed-paged KV cache
- FP16 KV cache
- head dimension 128
- non-sliding standard causal attention
- owned KV normal prefill
- multi-request packed carrier (`Q.shape[0]=1`, `context_lengths.shape[0]>1`)

FP8 KV, D256/D512, sliding attention, vision-block/tree attention, shared donor KV와 cache prefix를 읽는 packed
continuation은 아직 허용하지 않는다. CLI/export에는 아직 노출하지 않았다. runtime packer와 last-token gather를
연결하기 전에 사용자가 잘못된 packed engine을 만들지 않게 하기 위한 단계적 제한이다.

## 변경 위치

- `cpp/plugins/attentionPlugin/attentionPlugin.{h,cpp}`
  - `enable_packed_prefill` parse/clone/serialize/plugin-field 등록
  - context-length 축을 사용한 logical batch 계산
  - compact Q/K/V tensor view 생성
  - `launchApplyRopeWriteKVPacked()`와 causal compact `ContextFMHARunner` 연결
  - decode/legacy 경로 보존
- `tests/python-unittests/test_attention_plugin.py`
  - indexed/paged/packed plugin field와 binding을 만들 수 있도록 test runner 확장
  - lengths `[8,5,3]`, slots `[3,0,2]`의 packed causal reference test 추가
  - 임대하지 않은 slot 1의 physical pages가 0인지 검사

## 검증 결과

- SM86 TensorRT 11 plugin target 빌드 통과
- `ContextAttentionTest.compactLayout_Causal` 통과
- `RopeWriteKvPrefill.PackedRaggedRowsWriteStableIndexedSlots` 통과
- `RopeWriteKvPrefill.PackedBenchmark` 통과
- 최종 재측정: dense 1024 tokens `0.0263627ms`, packed 678 tokens `0.0207616ms`, **21.25% 감소**
- Python test syntax와 pre-commit 전체 대상 통과

Torch와 TensorRT 11이 함께 든 Python image가 로컬에 없어 새 plugin-level Python integration test는 아직 실제
실행하지 못했다. 지정 PyTorch image 다운로드를 시도했으나 저장 공간 부족으로 실패했고, 기존 모델/engine은
삭제하지 않았다. C++에서 구성 요소 정확성과 plugin 컴파일은 통과했지만 export/build/inference E2E gate로
간주하지 않는다.

## 바로 다음 단계

1. packed continuation용 `endLengths[B]`/cache-prefix gather를 plugin input 계약에 연결한다.
2. fixed-128 prefill queue가 padded input을 `[1,T]`와 `cuSeqLens` 의미로 pack하도록 phase adapter를 구현한다.
3. default/Cosmos export에서 hidden-state total-token axis와 per-request last-token gather를 추가한다.
4. builder profile과 engine config에 opt-in을 노출한다.
5. Cosmos `export -> build -> inference` 출력 동일성 후 P1/P2/P4/P8 및 real-request E2E를 비교한다.
