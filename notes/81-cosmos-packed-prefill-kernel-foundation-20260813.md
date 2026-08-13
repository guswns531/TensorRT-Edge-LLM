# Cosmos true packed prefill CUDA 기반

## 이번 단계의 결론

TensorRT engine binding을 바꾸기 전에 true packed text prefill에 필요한 두 CUDA 계산 경로를 먼저 검증했다.

1. compact causal FMHA가 variable sequence length에서 reference와 일치한다.
2. packed Q/K/V의 RoPE와 stable indexed KV write가 padding 없이 동작한다.

아직 export/plugin/runtime adapter가 packed binding을 사용하지 않으므로 E2E 기능은 아니다. 기존 dense engine과
production preset에는 변화가 없다.

## packed RoPE/KV 계약

새 `launchApplyRopeWriteKVPacked()` 입력은 다음과 같다.

```text
Q          [totalTokens, Hq, D]
K/V        [totalTokens, Hkv, D]
cuSeqLens  [B + 1]
endLengths [B]             optional, insertion 뒤 global slot length
kvSlotIds  [B]             optional, active row -> stable slot
kvPageIds  [slots, 2, pages] optional, indexed-paged physical mapping
```

CUDA kernel은 `cuSeqLens`로 각 packed token의 logical row와 row-local offset을 얻는다. chunked row의 RoPE/KV
position은 `endLengths[row] - rowLength + localOffset`이다. 따라서 서로 다른 chunk length와 past KV length를 같은
packed tensor에 넣어도 각 stable slot의 올바른 위치에 기록된다.

테스트는 다음 mapping을 사용했다.

```text
cuSeqLens  = [0, 3, 4, 6]
row length = [3, 1, 2]
past KV    = [0, 128, 5]
slot       = [3, 0, 2]

packed token 0..2 -> slot 3, KV position 0..2
packed token 3    -> slot 0, KV position 128
packed token 4..5 -> slot 2, KV position 5..6
```

Q와 K의 RoPE 결과, K/V cache 위치를 CPU reference와 비교했고 모두 일치했다. compute-sanitizer memcheck도 오류
0개로 통과했다.

## compact causal FMHA

기존 `ContextFMHARunner`에는 `isSPadded=false`가 있었지만 테스트는 vision용 non-causal compact layout뿐이었다.
reference 호출이 causal flag를 전달하도록 수정하고 다음 text-prefill shape를 추가했다.

- B3, lengths `[17, 64, 31]`, Hq16/Hkv4/D128
- B4, lengths `[3, 32, 64, 29]`, Hq8/Hkv2/D256

두 경우 모두 `atol=1e-2`, `rtol=1e-2`를 통과했고 1e-3 기준 element pass rate도 1.0이었다.

## RTX 3080 microbenchmark

Cosmos head shape Hq16/Hkv8/D128, B8, max chunk 128에서 실제 short-trace와 비슷한 row lengths
`[128,117,109,102,96,90,33,3]`을 사용했다. CUDA event로 warmup 20회 후 200회 평균을 세 프로세스에서
측정했다.

| run | dense `[8,128]` | packed 678 tokens | 감소 |
| ---: | ---: | ---: | ---: |
| 1 | 0.026337ms | 0.020813ms | 20.97% |
| 2 | 0.026381ms | 0.020977ms | 20.49% |
| 3 | 0.026353ms | 0.020797ms | 21.08% |
| median | **0.026353ms** | **0.020813ms** | **20.97%** |

물리 token 수는 1024에서 678로 33.8% 줄었지만 kernel 시간은 약 21% 줄었다. packed row lookup과 고정 launch
비용이 남기 때문이다. 이 수치는 RoPE/KV write 하나에만 해당하며 model E2E speedup은 아니다.

## 회귀 검증

- Context attention 및 padded/packed RoPE 관련 GPU 테스트 20개 실행
- 기존 padded causal/non-causal, FP16 KV, vanilla/tree decode 통과
- 무작위 FP8 prefill test는 전체 묶음에서 기존 quantization 경계 한 원소가 한 번 실패했지만, 독립 프로세스
  재실행 3회는 모두 통과했다. packed API는 FP16 cache 정확성 test로 분리되어 있다.
- packed stable-slot test의 compute-sanitizer 오류 0개
- production engine binding과 scheduler preset 변경 없음

## 다음 연결 순서

1. attention plugin에 serialized opt-in `enable_packed_prefill`을 추가한다.
2. plugin input/output shape 계약을 compact Q/K/V와 `cuSeqLens`로 분기한다.
3. normal prefill은 compact FMHA를 직접 사용하고 chunked prefill은 packed Q와 stable cache prefix를 함께 읽는
   compact K/V gather를 추가한다.
4. default-model export의 hidden state와 last-token gather에 total-token axis를 추가한다.
5. builder profile, binding registry, prefill adapter를 연결한다.
6. Cosmos export → build → inference 출력 동일성 후 P1/P2/P4/P8 kernel-group와 real-request E2E를 재측정한다.

Cosmos D128 standard attention부터 연결한다. Gemma 4 D512 FFPA/CuTe split-QKV, sliding/vision-block attention은 별도
후속 범위다.
