# TensorRT Edge-LLM

## Set up

### Build

```bash
cd build
cmake .. \
    -DTRT_PACKAGE_DIR=/usr \
    -DCMAKE_TOOLCHAIN_FILE=cmake/aarch64_linux_toolchain.cmake \
    -DEMBEDDED_TARGET=jetson-thor
make -j$(nproc)
```

### Python environment

```bash
cd ~/Documents/TensorRT-Edge-LLM/
source .edge/bin/activate

export CUDA_HOME=/usr/local/cuda
export TRITON_PTXAS_PATH="${CUDA_HOME}/bin/ptxas"
export PATH="${CUDA_HOME}/bin:$PATH"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:$LD_LIBRARY_PATH"
```

### Model export (Qwen3-VL-2B example)

```bash
tensorrt-edgellm-quantize-llm \
    --model_dir Qwen/Qwen3-VL-2B-Instruct \
    --output_dir /home/sslab/nfs/thor/TensorRT-Edge-LLM/quantized/qwen3-vl-2b \
    --quantization nvfp4

tensorrt-edgellm-export-llm \
    --model_dir /home/sslab/nfs/thor/TensorRT-Edge-LLM/quantized/qwen3-vl-2b \
    --output_dir /home/sslab/nfs/thor/TensorRT-Edge-LLM/onnx_models/qwen3-vl-2b

tensorrt-edgellm-export-visual \
    --model_dir /home/sslab/nfs/thor/TensorRT-Edge-LLM/quantized/qwen3-vl-2b \
    --output_dir /home/sslab/nfs/thor/TensorRT-Edge-LLM/visual_enc_onnx/qwen3-vl-2b
```

### Engine build

```bash
./build/examples/llm/llm_build \
  --onnxDir /home/sslab/nfs/thor/TensorRT-Edge-LLM/onnx_models/qwen3-vl-2b \
  --engineDir /home/sslab/nfs/thor/TensorRT-Edge-LLM/engines/qwen3-vl-2b \
  --maxBatchSize 4 \
  --maxInputLen=1024 \
  --maxKVCacheCapacity=4096 \
  --vlm \
  --minImageTokens 128 \
  --maxImageTokens 512

./build/examples/multimodal/visual_build \
  --onnxDir /home/sslab/nfs/thor/TensorRT-Edge-LLM/visual_enc_onnx/qwen3-vl-2b \
  --engineDir /home/sslab/nfs/thor/TensorRT-Edge-LLM/visual_engines/qwen3-vl-2b
```

### Inference

```bash
./build/examples/llm/llm_inference \
  --engineDir engines/qwen3-vl-2b \
  --multimodalEngineDir visual_engines/qwen3-vl-2b \
  --inputFile input_with_images.json \
  --outputFile output.json
```

### Benchmark

```bash
./build/examples/llm/llm_benchmark \
  --engineDir engines/qwen3-vl-2b \
  --multimodalEngineDir visual_engines/qwen3-vl-2b \
  --inputFile input_with_images.json \
  --outputFile output.json \
  --benchmarkCount 200
```

---

## In-GPU Disaggregation

### Concept

LLM 추론은 성격이 다른 두 단계로 구성됩니다.

| 단계 | 특성 | 병목 |
|------|------|------|
| **Prefill** | 입력 토큰 전체를 한 번에 처리 | Compute-bound (GEMM 위주) |
| **Decode** | 토큰을 한 개씩 자기회귀적으로 생성 | Memory-bandwidth-bound (KV cache 접근) |

두 단계를 같은 GPU 자원에서 순차 실행하면 각 단계의 최적 실행 조건이 충돌합니다.
**In-GPU Disaggregation**은 단일 GPU 내의 compute 자원(GPC/TPC)을 단계별로 분리하여 prefill과 decode를 물리적으로 독립된 파티션에서 동시에 실행하는 것을 목표로 합니다.

### Jetson Thor GPU 토폴로지

Jetson Thor(AGX Thor)의 GPU는 3개의 GPC와 총 10개의 TPC로 구성됩니다.

```
$ ./libsmctrl/libsmctrl_test_gpc_info
GPU0 has 3 enabled GPCs.
  GPC0: mask 0x049 → TPC {0, 3, 6}   (3 TPCs)
  GPC1: mask 0x092 → TPC {1, 4, 7}   (3 TPCs)
  GPC2: mask 0x324 → TPC {2, 5, 8, 9} (4 TPCs)
Total: 10 TPCs
```

각 GPC를 독립적으로 스케줄링할 수 있으므로, 예를 들어 GPC0+GPC1을 decode 전용, GPC2를 prefill 전용으로 파티셔닝하는 구성이 가능합니다.

### 현재 구현 (`LLMInferenceDisaggregationRuntime`)

현재는 단일 GPU에서 스트림을 분리한 파이프라인 구조로 구현되어 있으며, 다음 두 가지 최적화가 포함됩니다.

**1. 3-stage worker pipeline (별도 CUDA stream)**

```
Thread:   multimodalWorker ──► prefillWorker ──► decodeWorker
Stream:   mMultimodalStream    mPrefillStream    mDecodeStream
동기화:         cudaEvent(multimodalDone)  cudaEvent(prefillDone)
```

각 스테이지가 독립 스레드와 CUDA 스트림에서 동작하며, CUDA event를 통해 CPU blocking 없이 스테이지 간 GPU 레벨 동기화를 수행합니다.

**2. System prompt KV cache 재사용**

동일한 system prompt가 반복되는 요청에서 prefill을 재실행하지 않고, 사전 계산된 KV cache를 GPU로 복원하여 해당 토큰만큼의 prefill 연산을 완전히 스킵합니다.

```cpp
// setUpForPrefillExecution() 내부
if (mSystemPromptKVCache.find(promptHash) != mSystemPromptKVCache.end())
{
    kernel::instantiateKVCacheFromTensor(kvCacheBuffer, kvCacheContent, i, stream);
    // system prompt 토큰 수만큼 입력에서 제외 → prefill 연산량 감소
    processedInputIds.emplace_back(batchedInputIds[i].begin() + reuseLength, ...);
}
```

### 벤치마크 결과 (Jetson Thor, Qwen3-VL-2B NVFP4, count=200)

```
=== Default ===
stage        samples    mean_ms  median_ms    p99_ms      throughput
--------------------------------------------------------------------
encoding         200    101.101    102.980   106.461   4807.063 imgtok/s
prefill          200     20.080     20.218    21.173  25497.863 tok/s
decode         25400      9.119      9.267     9.480    109.662 tok/s
request          200   1279.289   1300.079  1331.581      0.782 req/s

=== Disaggregation #1 ===
stage        samples    mean_ms  median_ms    p99_ms      throughput
--------------------------------------------------------------------
encoding         200     92.292     88.610   104.773   5265.872 imgtok/s
prefill          200     18.437     17.625    21.066  27770.920 tok/s
decode         25400      8.276      8.049     9.504    120.838 tok/s
request          200   1161.719   1128.481  1332.889      0.861 req/s
```

| 지표 | Default | Disaggregation #1 | 개선율 |
|------|---------|-------------------|--------|
| Encoding latency | 101.1 ms | 92.3 ms | **-8.7%** |
| Prefill latency | 20.1 ms | 18.4 ms | **-8.2%** |
| Decode latency | 9.12 ms/tok | 8.28 ms/tok | **-9.2%** |
| Request throughput | 0.782 req/s | 0.861 req/s | **+10.1%** |

### 향후 계획

현재 구현은 단일 GPU 내에서 스트림만 분리한 상태입니다. 다음 단계는 libsmctrl 등을 활용하여 GPC/TPC를 단계별로 실제 파티셔닝함으로써 prefill과 decode를 물리적으로 동시에 실행하는 것입니다.

```
[목표 구조]
GPC0           (3 TPCs) ─── Decode worker    (memory-BW 최적화)
GPC1+GPC2      (7 TPCs) ─── Prefill worker   (compute 최적화)
                         ↕ (CUDA event 동기화, KV cache 공유)
```

이를 통해 decode 중에 다음 요청의 prefill을 물리적으로 병렬 실행하여 end-to-end 처리량을 더 향상시키는 것이 최종 목표입니다.



```c

#define CU_13_0_MASK_OFF_JETSON 0x54c
// 13.0 tested on Jetson Thor (Feb 2026)

struct stream_sm_mask_v2 {
	uint32_t enabled;
	uint32_t mask[4];
};

// Should work for CUDA 8.0 through 12.8
// A cudaStream_t is a CUstream*. We use void* to avoid a cuda.h dependency in
// our header
void libsmctrl_set_stream_mask(void* stream, uint64_t mask) {
	// When the old API is used on GPUs with over 64 TPCs, disable all TPCs >64
	uint128_t full_mask = -1;
	full_mask <<= 64;
	full_mask |= mask;
	libsmctrl_set_stream_mask_ext(stream, full_mask);
}

void libsmctrl_set_stream_mask_ext(void* stream, uint128_t mask) {
	char* stream_struct_base = *(char**)stream;
	struct stream_sm_mask_v2* hw_mask_v2 = NULL;
  hw_mask_v2 = (void*)(stream_struct_base + CU_13_0_MASK_OFF_JETSON);

   if (hw_mask_v2) {
		hw_mask_v2->enabled = 1;
		hw_mask_v2->mask[0] = mask;
		hw_mask_v2->mask[1] = mask >> 32;
		hw_mask_v2->mask[2] = mask >> 64;
		hw_mask_v2->mask[3] = mask >> 96;
	} else {
		abort(1, 0, "Stream masking unsupported on this CUDA version (%d), and"
		            " no fallback MASK_OFF set!", ver);
	}
}


```
