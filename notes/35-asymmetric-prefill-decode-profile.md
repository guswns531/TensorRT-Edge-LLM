# Gemma 4 E2B 비대칭 prefill/decode profile

## 목적

RTX 3080 10GB에서 physical KV slot은 32개로 유지하면서 prefill activation/workspace만
8개 요청 기준으로 제한하고, decode는 32개 active row까지 허용한다.

## 구현

- `LLMBuilderConfig`에 `maxPrefillBatchSize`, `maxDecodeBatchSize`를 추가했다.
- `llm_build`에 `--maxPrefillBatchSize`, `--maxDecodeBatchSize`를 추가했다.
- engine JSON의 `builder_config`에 `max_prefill_batch_size`, `max_decode_batch_size`를 기록한다.
- legacy JSON은 두 값이 없으면 `max_batch_size`를 상속한다.
- Gemma 4 indexed export에서 past KV의 첫 축을 active `batch`와 별도인 `kv_slots` symbolic dimension으로 export한다.
- KV cache profile은 `maxBatchSize`를 사용해 physical slot capacity를 유지한다.
- context/generation 입력 profile은 각각 prefill/decode cap을 사용한다.
- `llm_phase_bench`는 요청 batch가 해당 phase cap을 넘으면 거부한다.

## 재현 경로

모델과 중간 산출물은 Git에 커밋하지 않고 다음 경로에 저장한다.

```text
.local/gemma4-e2b/hf
.local/gemma4-e2b/quant-int4-awq
.local/gemma4-e2b/onnx-indexed
.local/gemma4-e2b/engine-indexed-slots32-p8-d32-i128
```

고정 모델 revision:

```text
google/gemma-4-E2B-it
3e22461f65e89153144f8adb70e3b8c2cc9845a7
```

Build command:

```bash
build/examples/llm/llm_build \
  --onnxDir .local/gemma4-e2b/onnx-indexed \
  --engineDir .local/gemma4-e2b/engine-indexed-slots32-p8-d32-i128 \
  --maxBatchSize 32 --maxPrefillBatchSize 8 --maxDecodeBatchSize 32 \
  --maxInputLen 128 --maxKVCacheCapacity 2048
```

## 현재 검증
