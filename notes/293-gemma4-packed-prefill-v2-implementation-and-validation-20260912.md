# Gemma 4 packed-prefill v2 구현 및 검증

## 1. 결론

Gemma 4 E2B INT4-AWQ text decoder에 true packed/chunked prefill을 연결했다. 이번 구현은 Gemma 전용
스케줄러가 아니라, 기존 공통 packed-prefill contract가 요구하는 attention capability를 Gemma 4의
heterogeneous attention 구조까지 확장한 것이다.

- packed token carrier: `[1, sum(row_tokens), C]`
- logical prefill batch: `context_lengths.shape[0]`
- chunk upper bound: profile-local `packed_prefill_chunk_limit`, 현재 128
- stable indexed-paged KV ownership 유지
- Gemma sliding attention d256 지원
- Gemma global attention d512 지원
- owned-KV와 donor/shared-KV layer 모두 지원
- P8/D24 text engine과 E4 visual engine을 같은 phase runtime에 연결

짧은 text 요청, 128-token 경계를 여러 번 넘는 긴 text 요청, 실제 image 요청이 모두 성공했다. 긴 text
요청은 dense engine과 greedy output token이 정확히 일치했다. 다만 자연 arrival의 4-image wave에서는
encoder engine의 E4 capacity와 별개로 실제 E cohort가 E2까지만 형성됐고, 50 ms controlled wait에서도
E3까지만 형성됐다. 따라서 이번 단계가 증명한 것은 `E4/P8/D24 engine capability + E/P/D 연결`이지,
production arrival에서 E4 formation이 항상 달성된다는 것은 아니다.

## 2. 왜 이 구현은 과도하게 모델별이지 않은가

공통 mechanism과 Gemma-specific metadata를 분리했다.

| 구분 | 공통 구현 | Gemma 4에서 추가로 필요한 정보 |
|---|---|---|
| token packing | 여러 logical row를 하나의 token carrier로 결합 | 없음 |
| row recovery | `cu_q_seq_lens`로 packed token을 logical row에 매핑 | 없음 |
| KV ownership | stable request lease와 page table 사용 | donor layer 관계 |
| RoPE | row별 absolute position으로 적용 | d256/d512, partial rotary |
| attention | native paged causal FMHA-v2 | sliding/global layer 종류 |
| export | packed input/profile contract | dual RoPE와 PLE wiring |

즉 모델별 코드는 `어떤 layer가 d256/d512이고 어떤 cache를 공유하는가`를 전달한다. token packing,
logical-row mapping, page-table ownership, chunk limit, independent P/D context 실행은 공통 runtime/plugin
contract다.

## 3. 실행 구조

```text
request queues
    |
    +-- E queue -- visual TensorRT context -- vision lease --+
    |                                                       |
    +-- P queue ---------------------------------------------+
                                                            v
                                  logical rows [L0, L1, ...]
                                             |
                                             v
                            packed carrier [1, sum(Li), C]
                                             |
                                  cu_q_seq_lens / page table
                                             |
                         +-------------------+-------------------+
                         |                                       |
                         v                                       v
                  owned-KV layer                         shared-KV layer
           packed Q/K/V -> RoPE -> page write       packed Q -> dense row scratch
                         |                              -> RoPE -> donor pages
                         v                                       |
                  paged FMHA-v2                                 v
                         |                              paged FMHA-v2
                         |                                       |
                         +-------------------+-------------------+
                                             v
                              packed attention output carrier
                                             |
                                             v
                                  P completion -> D queue
                                             |
                                  decode TensorRT context
```

Owned-KV layer는 기존 packed Q/K/V split-and-write 경로를 재사용한다. Shared-KV layer는 K/V를 쓰지 않고
donor cache를 읽어야 한다. FMHA의 dense logical-row input contract를 만족시키기 위해 packed Q에 RoPE를
적용하며 dense scratch로 scatter하고, FMHA 출력에서 유효 row만 다시 packed carrier로 gather한다. dense
padding은 launch 전에 비동기 memset으로 0으로 만든다.

## 4. 코드 변경

### Export

- `tensorrt_edgellm/models/gemma4/modeling_gemma4_text.py`
  - Gemma flat wrapper에 `packed_prefill_chunk_limit`을 추가했다.
  - input embedding, PLE, last-token selection의 batch/token carrier 축을 packed contract에 맞췄다.
  - 각 attention custom op에 packed enable/max-chunk attribute와 profile-local chunk input을 전달한다.

### CUDA/kernel

- `cpp/kernels/posEncoding/applyRopeWriteKV.{h,cu}`
  - `launchApplyRopeQOnlyPackedToDense`를 추가했다.
  - `cu_q_seq_lens`로 token의 logical batch/row를 복원한다.
  - `kv_cache_end_lens - row_len + row`로 absolute RoPE position을 계산한다.
  - d256/d512 및 partial rotary를 같은 vectorized kernel로 처리한다.

### TensorRT plugin/runtime config

- `cpp/plugins/attentionPlugin/attentionPlugin.cpp`
  - packed prefill을 shared-KV와 sliding-window layer에 허용했다.
  - supported head dimension을 128/256/512로 확장했다.
  - shared-KV packed Q scatter, paged donor-cache FMHA, packed output gather를 연결했다.
- `cpp/runtime/config/llmEngineConfig.cpp`
  - 단일 `head_dim == 128` 검사 대신 모든 attention layer의 실제 head dimension을 검증한다.
  - 각 layer는 128/256/512 중 하나여야 하며 FP16 KV, vanilla autoregressive 등 기존 invariant는 유지한다.

### Tests

- CUDA ragged packed-Q RoPE: d256/d512
- runtime config: heterogeneous d256/d512 layer
- AttentionPlugin: Gemma owned/shared KV packed-vs-dense
- export source contract: wrapper, custom-op attribute, chunk-limit input

## 5. 산출물과 실행 contract

기준 source parent commit은 `83b337d80a5b87191570eff0a03e4bcc0d95a413`이다. 이 문서와 구현을
포함하는 최종 commit은 본 문서 작성 뒤 생성한다.

| 산출물 | 경로 | contract |
|---|---|---|
| packed ONNX | `.local/artifacts/v0101-forward-port/gemma-4-e2b-it-awq/onnx-int4-awq-packed-p128` | INT4 plugin v1, packed chunk 128 |
| correctness engine | `.local/artifacts/v0101-forward-port/gemma-4-e2b-it-awq/engine-packed-p2-d4-kv2048-p64` | P2/D4, KV capacity 2048, 64 pages |
| target text engine | `.local/artifacts/v0101-forward-port/gemma-4-e2b-it-awq/engine-packed-p8-d24-kv2048-p96` | P8/D24, KV capacity 2048, 96 pages |
| visual engine | `.local/artifacts/v0101-forward-port/gemma-4-e2b-it-awq/visual-e4-soft280/visual` | E4 capability |
| diagnostic logs | `.local/scratch/gemma4-packed-prefill-v2-20260912` | disposable raw logs |

Target engine은 profile의 모든 D24 sequence가 동시에 2048 token을 보유하는 worst case보다 작은 96-page
pool을 의도적으로 사용하므로 `allow_kv_pool_undercommit=true`다. 물리 pool은 12,288 token이다. 실제
admission과 page reservation이 이 한도를 강제해야 하며, profile maximum만 보고 49,152 token이 상주할
수 있다고 해석하면 안 된다.

ONNX contract 검사 결과는 다음과 같다.

- AttentionPlugin node 35개, 모두 packed enabled
- sliding d256 owned 12개
- global d512 owned 3개
- sliding d256 shared 16개
- global d512 shared 4개
- `Int4GroupwiseGemmPlugin` v1 223개, v2 0개
- ONNX checker 통과

SM86에서 처음 시도한 INT4 plugin v2 engine build는 실패했다. 이는 packed attention failure가 아니라
SM86에서 해당 GEMM plugin v2가 지원되지 않은 문제였고, export를 plugin v1으로 고정해 해결했다.

## 6. 검증 결과

### 6.1 정적/단위 검증

| 검증 | 결과 |
|---|---|
| `git diff --check` | PASS |
| pre-commit 전체 대상 파일 | PASS |
| Python `py_compile` | PASS |
| Gemma/export source contract | PASS |
| `LLMEngineConfigTest` packed 관련 | PASS |
| `RopeQOnlyPackedToDense.Gemma4HeadDimensionsAndRaggedRows` | PASS, 204 ms |
| plugin/runtime/phase-smoke build | PASS |

Python AttentionPlugin GPU test는 현재 TensorRT 11 container에 PyTorch가 없고, 별도 PyTorch container의
TensorRT ABI가 10.14라 이 조합에서는 실행하지 않았다. 대신 실제 exported Gemma engine의 end-to-end
inference가 owned/shared layer 전체를 실행했고, dense output identity를 확인했다.

### 6.2 text correctness

| case | packed result | dense reference |
|---|---|---|
| short, prompt 19, output 16 | success | 16/16 greedy token exact |
| long request A, prompt 275 | chunks 128+128+19 | 16/16 exact |
| long request B, prompt 154 | chunks 128+26 | 16/16 exact |

긴 두 요청의 packed E2E는 각각 274.99 ms, 272.24 ms였고 dense reference는 285.33 ms, 281.23 ms였다.
이는 단 1회 diagnostic이고 engine shape도 완전히 동일한 성능 캠페인이 아니므로 speedup 근거로 사용하지
않는다. 이 결과의 의미는 multi-chunk state/KV continuity와 greedy identity다.

### 6.3 actual TensorRT ragged P2 및 overlap capability

실제 P2/D4 packed engine에 `[96, 32]` ragged P2를 넣고 100 iteration으로 실행했다.

| controlled start | makespan | sequential 대비 |
|---:|---:|---:|
| sequential independent contexts | 22.137 ms | 1.000x |
| independent overlap | 18.177 ms | 1.218x |
| 25% | 21.139 ms | 1.047x |
| 50% | 20.200 ms | 1.096x |
| 75% | 19.290 ms | 1.148x |
| 100% | 18.280 ms | 1.211x |

이것은 zero embedding을 사용한 controlled kernel/plugin characterization이다. real-request E2E throughput이나
현재 online scheduler의 정책 우위를 의미하지 않는다. 다만 ragged P2가 실제 TensorRT plugin 경로에서
동작하며 P/D independent contexts가 같은 GPU에서 overlap 가능한 것을 증명한다.

### 6.4 E4/P8/D24 연결

Target P8/D24 engine은 real text request에서 16 token을 정상 생성했고 correctness engine과 같은 greedy
token sequence를 냈다. single-image VLM request도 성공했다.

| VLM case | 결과 |
|---|---|
| single image | E1, encoder 17.193 ms, TTFT 59.758 ms, E2E 162.442 ms |
| natural 4-image wave | 4/4 success, E batch 3회, max E2 |
| controlled 50 ms E wait | 4/4 success, E batch 2회, max E3 |

Single-image prompt는 282 token이므로 vision output placement 뒤의 P path가 128-token chunk boundary를
통과했다. Natural wave에서는 downstream packed prefill이 P3/384-token dispatch를 형성했고 decode도 D3까지
형성됐다. 즉 logical P batching은 실제 VLM path에서 작동했다.

E4가 형성되지 않은 직접 원인은 engine capacity 부족이 아니라 image adapter의 비동기 준비 완료 시각과
coordinator가 첫 ready encoder request를 즉시 dispatch하는 formation semantics다. E batch를 키우려면 kernel을
다시 바꾸는 것이 아니라, `기다림으로 얻는 cohort gain`과 oldest vision request의 critical-path delay를 비교하는
encoder formation policy가 필요하다.

## 7. 남은 제한

1. Legacy public `handleRequest()`는 긴 prompt를 자체적으로 128-token chunk로 분할하지 않는다. packed engine에
   전체 prompt를 한 번에 보내면 `logical batch * chunk limit` 검사에서 거부된다. 현재 검증된 production research
   path는 `PhaseAsyncServer/llm_phase_context_smoke`의 independent phase runtime이다.
2. Full 12-workload 성능 gate와 frozen vLLM 비교는 아직 실행하지 않았다. 이번 단계는 capability/correctness gate다.
3. P8와 D24는 engine maximum이며, 한 번의 diagnostic request가 해당 cohort를 실제로 형성했다는 뜻이 아니다.
4. Natural E4 formation은 아직 관측하지 못했다.
5. FP8 KV, tree/speculative attention, vision-block attention과 packed prefill의 조합은 계속 명시적으로 거부한다.

## 8. 다음 순서

### G4. 공정한 성능 gate

동일 packed engine/binary/request/calibration/memory contract로 V0/V1/V2를 실행한다. 먼저 text short,
balanced, decode-heavy, long-prefill과 VLM mixed/vision-heavy/multi-image를 3회 확인한 뒤 회귀가 없으면 전체
12-workload로 확대한다. vLLM request contract가 기존 frozen 결과와 같을 때는 재사용하고, model/engine/output
contract가 달라지면 fresh vLLM을 실행한다.

필수 지표는 throughput, TTFT mean/p95, TPOT mean/p95, E2E mean/p95, P/D batch distribution, packed useful-token
ratio, page pressure, stream busy/overlap/idle mask다.

### G5. E formation과 public API

- E4를 강제하는 static wait를 추가하지 않는다.
- already-outstanding adapter completion과 현재 E-ready mass로부터 bounded WAIT candidate를 생성한다.
- E cohort gain이 wait cost와 first-token critical-path loss보다 클 때만 기다린다.
- public `handleRequest()`를 계속 지원할 필요가 확인되면 동일 chunk state machine을 공통 request adapter 아래로
  내린다. phase runtime과 별도 chunk 구현을 만들지 않는다.

## 9. 판단

이번 작업은 과한 모델별 kernel fork가 아니다. 공통 packed-prefill mechanism이 기존에는 head 128, owned-KV,
non-sliding layer에만 닫혀 있었고, Gemma 4가 그 미지원 조합을 처음 모두 요구했기 때문에 capability frontier를
확장한 것이다. 향후 다른 모델이 d256/d512, sliding attention, layer KV sharing을 사용해도 같은 plugin/kernel
경로를 재사용할 수 있다.

성능 주장은 G4 이후에만 한다. 현재 확정된 성과는 `Gemma 4 true packed/chunked P + stable paged KV + independent
P/D contexts + visual E context`가 실제 export/build/inference에서 correctness를 유지한다는 것이다.
