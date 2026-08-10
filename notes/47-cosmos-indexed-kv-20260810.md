# Cosmos Reason2-2B indexed KV 지원 결과 (2026-08-10)

## 결론

`nvidia/Cosmos-Reason2-2B`의 Qwen3-VL text decoder를 FP16, indexed KV로
export/build하고 RTX 3080에서 다음 경로를 실행했다.

- text prefill/decode와 `kv_slot_ids` stable ownership
- fixed-128 chunked prefill과 decode dynamic batching
- 하나의 CUDA context 안의 독립 prefill/decode TensorRT execution context
- 서로 다른 output length의 request 종료, slot 반환 및 재사용
- 실제 image encoder -> deepstack/M-RoPE -> indexed prefill -> decode trace

생성된 decoder engine은
`.local/cosmos-reason2-2b/engine-fp16-indexed-asym-p8-d16-i1024-kv2048`이며,
`maxBatch=16`, `maxPrefillBatch=8`, `maxDecodeBatch=16`,
`maxInputLength=1024`, `maxKVCacheCapacity=2048`이다.

## 왜 Gemma 경로를 그대로 켜면 깨졌는가

기존 indexed 구현의 cache binding은 논리적으로
`[slot, 2, Hkv, capacity, D]`였다. write kernel도 이 BHSD 순서로 썼지만,
decode XQA는 같은 byte pool을 128-token page의 NHD 순서로 읽었다.

```text
legacy write: [slot][K/V][head][token][d]
XQA read:     [physical page][token in page][head][d]
```

Gemma 4 E2B는 `Hkv=1`이라 두 주소 계산이 우연히 같았다. Cosmos는
`Hq=16`, `Hkv=8`, `D=128`이므로 head와 token stride가 달라져 decode가
잘못된 K/V를 읽었다. 처음 실행이 정상 종료하면서도 다국어 garbage token을
생성한 이유다.

indexed mode는 binding shape와 총 allocation byte 수를 유지하되, 실제 저장
순서를 XQA와 같은 page-major NHD로 바꿨다.

```text
logical owner: slot s, K/V plane k, token t

pagesPerSequence = capacity / 128
physicalPage = (s * 2 + k) * pagesPerSequence + t / 128
offset = (((physicalPage * 128 + t % 128) * Hkv + head) * D + d)

slot 0: K page 0..N-1 | V page 0..N-1
slot 1: K page 0..N-1 | V page 0..N-1
...
```

따라서 page table은 metadata만으로 stable linear slot의 page를 가리키고,
request 종료 시 KV tensor를 옮기지 않는다. non-indexed engine에는
`kv_slot_ids`가 없으므로 기존 BHSD write/read가 그대로 유지된다.

## 구현 위치

- `tensorrt_edgellm/models/default/modeling_default.py`
  - default/Qwen3-VL export graph에 opt-in `kv_slot_ids: INT32[B]`를 전달한다.
  - physical slot 수와 active batch를 분리하기 위해 KV input의 batch dimension은
    `kv_slots`, attention input은 `batch`로 export한다.
  - indexed attention custom op를 모든 decoder layer에 연결한다.
- `tensorrt_edgellm/scripts/export.py`
  - `qwen3_vl`과 `qwen3_vl_text`에서 `--indexed-kv-cache`를 허용한다.
- `cpp/kernels/posEncoding/applyRopeWriteKV.cu`
  - indexed prefill/decode K/V write를 page-major NHD offset으로 계산한다.
- `cpp/kernels/contextAttentionKernels/utilKernels.cu`
  - chunked prefill용 cache gather가 stable slot과 page-major layout을 함께 해석한다.
- `cpp/plugins/attentionPlugin/attentionPlugin.cpp`
  - 기존 indexed page list를 XQA에 연결한다. allocator는 page allocator로
    바뀌지 않으며 slot의 연속 page view만 만든다.
- `cpp/runtime/config/llmEngineConfig.cpp`
  - `Hkv=1` 임시 제한을 제거했다. speculative/Mamba 등 v1 제한은 유지한다.

## export/build 검증

실행 순서는 실제로 `export -> build -> inference`를 지켰다.

```bash
tensorrt-edgellm-export .local/cosmos-reason2-2b/hf \
  .local/cosmos-reason2-2b/onnx-fp16-indexed \
  --dtype float16 --skip-visual --skip-audio --skip-code2wav --skip-action \
  --indexed-kv-cache

./build/examples/llm/llm_build \
  --inputModel .local/cosmos-reason2-2b/onnx-fp16-indexed/llm \
  --outputDir .local/cosmos-reason2-2b/engine-fp16-indexed-asym-p8-d16-i1024-kv2048 \
  --maxBatchSize 16 --maxPrefillBatchSize 8 --maxDecodeBatchSize 16 \
  --maxInputLength 1024 --maxKVCacheCapacity 2048 --profilingVerbosity detailed
```

ONNX에는 `kv_slot_ids ['batch']`, 28개의 indexed AttentionPlugin node,
`past_key_values_* ['kv_slots', 2, 8, 'past_len', 128]`가 존재한다.

## 정확성 및 lifecycle 검증

관련 GPU unit test 29개가 통과했다. 새 핵심 test는 다음을 포함한다.

- `Hkv=8`, capacity 256, slot `[3, 0]`의 RoPE write -> paged gather round trip
- slot `[3, 0, 2]`, 두 page, 모든 8 KV head의 K/V gather
- stable slot reserve/release, middle eviction, exhaustion, double-free, deterministic reuse
- global physical length와 독립 phase view, registry/config contract
- grouped-query attention engine config에서 indexed KV 허용

continuous load는 16 slot로 24 request를 drain했다. prompt 128--512,
output 8--64이며 prefill BS 1--4와 decode BS 1--12가 실제 dispatch되었다.
모든 request가 `stable admission -> greedy sampling -> repeated packed decode ->
scatter -> slot release`를 완료했다.

실제 두 image request도 성공했다.

| request | encoder-to-decode 결과 | E2E |
| --- | --- | ---: |
| woman and dog | `A woman with long, dark hair, dressed in a black...` | 126.245 ms |
| red panda | `A charming red panda with a fluffy white face, dark eyes...` | 125.816 ms |

Qwen3-VL image prefill은 deepstack과 M-RoPE placement를 원자적으로 유지해야 해서
현재 chunk size 1024, prefill BS1로 실행했다. text-only continuous load는
fixed-128 chunk를 사용한다.

## 공정한 fixed-shape 성능 비교

동일 RTX 3080, 같은 `P4/D12`, input 512, past KV 1536, fixed chunk 128,
독립 TensorRT context, warmup 5, 측정 30회 결과다. 별도 process에서 실행했지만
동일 runtime/plugin과 동일한 비대칭 profile을 사용했다.

| 모드/측정 | legacy median / p95 | indexed median / p95 | indexed 변화 |
| --- | ---: | ---: | ---: |
| sequential makespan | 139.315 / 139.700 ms | 141.046 / 141.713 ms | +1.24% / +1.44% |
| sequential prefill | 102.776 / 103.160 ms | 104.115 / 104.771 ms | +1.30% / +1.56% |
| sequential decode | 36.511 / 36.570 ms | 36.925 / 36.988 ms | +1.13% / +1.14% |
| concurrent makespan | 122.447 / 123.281 ms | 125.243 / 126.031 ms | +2.28% / +2.23% |

isolated phase 기준 median/p95 회귀는 모두 3% 이내다. indexed concurrent
실행 자체는 sequential 141.046 ms에서 125.243 ms로 줄어 1.126배 makespan
speedup을 보였다. dynamic load의 짧은 fixed case에서는 1.157배였다.

## 남은 제한과 해석

- 별도로 build한 legacy/indexed TensorRT engine의 greedy text는 둘 다 의미 있는
  문장을 생성하지만 token-for-token 완전 일치는 안정적으로 재현되지 않았다.
  동일 legacy engine의 과거/현재 반복 사이에도 첫 분기 token이 달라졌다.
  따라서 garbage가 제거되고 VLM semantic trace가 통과한 것은 확인했지만,
  엄격한 exact-token gate는 deterministic tactic/aux-stream 조건 또는 logit
  tolerance 비교 harness를 만든 뒤 다시 판정해야 한다.
- maxBatch 16 decoder와 visual engine을 함께 올린 `llm_inference` peak는
  9432 MiB로, 측정 시 총 9874 MiB 중 약 442 MiB만 남았다. 기능 검증에는
  성공했지만 512 MiB production headroom gate는 못 넘는다. 이미지 serving은
  slot 8--12 engine을 별도로 build하거나 workspace를 줄여야 한다.
- indexed cache는 외부 파편화와 eviction D2D copy를 없애지만 slot마다
  capacity 2048을 고정 예약하므로 짧은 sequence의 내부 낭비는 남는다.
- v1은 vanilla FP16 KV 경로다. speculative decoding, system-prompt cache,
  host offload, Mamba state는 지원 범위 밖이다.

## 결과 파일

- `.local/cosmos-reason2-2b/indexed-text-bs2-output.json`
- `.local/cosmos-reason2-2b/indexed-text-bs2-profile.json`
- `.local/cosmos-reason2-2b/indexed-load-p4d12-n24.csv`
- `.local/cosmos-reason2-2b/indexed-load-p4d12-n24-dispatch.csv`
- `.local/cosmos-reason2-2b/indexed-load-kernels.csv`
- `.local/cosmos-reason2-2b/legacy-fair-p4d12.csv`
- `.local/cosmos-reason2-2b/indexed-fair-p4d12.csv`
- `.local/cosmos-reason2-2b/indexed-vlm-trace.csv`
- `.local/cosmos-reason2-2b/indexed-vlm-trace-dispatch.csv`
- `.local/cosmos-reason2-2b/indexed-vlm-kernels.csv`
