# Cosmos packed prefill continuation 연결

## 결론

`enable_packed_prefill` attention 경로를 첫 chunk 전용에서 indexed-paged KV prefix를 읽는 continuation까지
확장했다. 요청마다 과거 길이와 현재 chunk 길이가 달라도 stable slot/page ownership을 유지하면서 하나의 compact
causal FMHA 호출로 처리한다.

```text
packed current Q/K/V [1,T,*]             stable indexed-paged KV cache
            |                            [physical pages, 128 tokens/page]
            | RoPE + page-aware KV write               |
            | at each row's past length                 |
            +----------------------------+---------------+
                                         |
                   read-only variable-prefix gather
                   K/V [sum(past + chunk), Hkv, D]
                                         |
          cuQ=[0, q0, q0+q1, ...]        | cuKV=[0, kv0, kv0+kv1, ...]
                            \             /
                             compact causal FMHA
                         (bottom-right alignment)
                                      |
                         packed attention output [1,T,Hq,D]
```

이 gather는 attention 한 번을 위한 임시 compact view를 만드는 read-only 작업이다. physical page, slot ownership,
global length tensor는 이동하지 않으며 KV compaction이나 D2D cache relocation을 수행하지 않는다.

## continuation 길이 의미

packed carrier의 `runtimeSeqLen=T`는 전체 batch의 token 합이다. 각 row의 KV 종료 위치로 사용하면 안 된다.

```text
chunk lengths       = [50, 30, 7]
past lengths        = [100, 0, 129]
cuQ                 = [0, 50, 80, 87]
per-row KV lengths  = [150, 30, 136]
cuKV                = [0, 150, 180, 316]
KV write end        = [150, 30, 136]
```

`calCuQCuKVSeqLensAndKVEndIdxs()`에 packed 의미를 명시하는 opt-in을 추가했다. legacy padded 경로는 이전처럼 모든
row를 `runtimeSeqLen`만큼 전진시키므로 동작이 바뀌지 않는다.

## attention causal 정렬

continuation row에서 `qLength < kvLength`다. query `q`가 볼 수 있는 마지막 key는 다음과 같다.

```text
pastLength = kvLength - qLength
lastVisibleKey(q) = pastLength + q
```

FMHA에는 별도의 `cu_q_seqlens`와 `cu_kv_seqlens`를 전달하고 compact causal bottom-right alignment를 사용한다.
새 GPU 테스트는 Q 길이 `[2,3,1]`, KV 길이 `[6,4,3]`을 CPU softmax 기준과 비교해 이 의미를 검증한다.

## prefix gather 커널

초기 구현은 `[B, capacity]` 전체 token grid를 띄운 뒤 유효 길이 밖 block이 return했다. 정확히 복사하는 token이
적어도 빈 block 수가 많아 Cosmos형 측정에서 padded-max gather보다 느렸다.

최종 구현은 다음 구조다.

- grid z: `(row, K/V, KV head)`
- grid y: 고정 8 token tile
- 각 block: 자기 tile의 유효 token만 stride 순회
- source: `kv_slot_ids -> kv_page_ids -> physical page/token`
- destination: `cuKVSeqLens[row] + token`

capacity 전체를 scan하지 않으며 각 row의 실제 prefix만 읽고 쓴다.

## 메모리 영향

attention plugin이 기존 chunked-prefill deinterleave를 위해 이미 예약하던
`[B,2,Hkv,capacity,D]` FP16 workspace를 K/V 두 영역으로 재해석한다. 따라서 이번 변경으로 plugin workspace의
peak 예약량은 증가하지 않는다. 다만 FMHA가 page table을 직접 읽는 것은 아니므로 attention 호출마다 유효 KV
prefix를 이 임시 workspace로 materialize하는 bandwidth 비용은 남는다.

장기 최적화 우선순위는 다음과 같다.

1. page-aware prefill FMHA로 gather 자체 제거
2. 또는 prefix가 긴 row의 gather와 이전 layer compute overlap
3. prefix-length bucket별 token-tile 수 튜닝

## RTX 3080 SM86 측정

조건은 FP16 KV, `B=8`, `Hkv=8`, `D=128`, capacity 2048이다. row prefix는
`[128,256,384,512,640,768,896,1024]`로 exact 4,608 tokens이며 padded 비교는 8 x 1,024 = 8,192 tokens이다.
warmup 20회 후 CUDA event 100회 평균을 한 프로세스에서 3번 반복했다.

| 반복 | padded-max gather | packed exact gather | 감소율 |
|---:|---:|---:|---:|
| 1 | 0.210012 ms | 0.082717 ms | 60.61% |
| 2 | 0.210727 ms | 0.082500 ms | 60.85% |
| 3 | 0.211200 ms | 0.082309 ms | 61.03% |
| median | **0.210727 ms** | **0.082500 ms** | **60.85%** |

이는 gather kernel-group만의 마이크로벤치마크다. 모델 E2E 또는 request throughput 향상으로 해석하면 안 된다.
각 transformer layer에서 prefix gather가 반복되므로 실제 효과는 attention/GEMM과 합친 E2E로 다시 측정해야 한다.

## 검증

- TensorRT 11 SM86 `NvInfer_edgellm_plugin`, `unitTest` 빌드 통과
- 집중 GPU 테스트 6개를 3회 반복, 매회 6/6 통과
  - packed row별 cuQ/cuKV/end length
  - stable indexed slot의 ragged RoPE/KV write
  - hole/custom page table의 variable-prefix gather
  - compact causal FMHA legacy 동작
  - 비대칭 cuQ/cuKV continuation bottom-right causal 정확성
  - Cosmos형 gather benchmark
- compute-sanitizer memcheck: continuation attention과 prefix gather 모두 오류 0건
- 대상 파일 pre-commit 통과

## 현재 지원 범위와 다음 단계

현재 plugin continuation 경로는 fixed-128 scheduler를 전제로 한 standard causal attention, indexed-paged FP16
KV, head dimension 128에 한정한다. decode/XQA와 `enable_packed_prefill=0` 경로는 바뀌지 않는다.

아직 runtime phase adapter가 여러 request chunk를 `[1,T]` carrier로 만들고 per-request last-token output을 다시
분리하는 연결, default/Cosmos export opt-in, builder profile, 실제 engine E2E는 남아 있다. 다음 구현 순서는 다음과
같다.

1. phase adapter의 fixed-128 token pack과 output row scatter
2. default/Cosmos ONNX export 및 builder profile에 packed prefill opt-in 노출
3. Cosmos `export -> build -> inference` 출력 동일성
4. P1/P2/P4/P8, D8/D16/D32와 real-request trace에서 sequential/independent E2E 비교
5. kernel-group table에 prefix gather, packed RoPE/write, compact FMHA를 별도 항목으로 기록
