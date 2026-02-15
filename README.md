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

### TPC 분할 실행 (`--disaggregation --tpcCount`)

현재 `llm_benchmark`의 disaggregation 경로에서는 `--tpcCount`를 **decode 전용 TPC 개수**로 사용합니다.

- `decode stream`에는 앞쪽 `tpcCount` TPC를 할당
- `prefill stream`과 `multimodal(encoding) stream`은 나머지 TPC를 공유
- 적용 예시:

```bash
./build/examples/llm/llm_benchmark \
  --engineDir engines/qwen3-vl-2b \
  --multimodalEngineDir visual_engines/qwen3-vl-2b \
  --inputFile input_with_images.json \
  --outputFile output.json \
  --benchmarkCount=20 \
  --disaggregation \
  --tpcCount 3
```

위 예시는 `decode=3 TPC`, `prefill+encoding=7 TPC` 구성이 됩니다.

제약 사항:
- Jetson Thor CUDA 13.x 기준 내부 stream mask 오프셋(`0x54c`)을 사용합니다.
- disaggregation 모드에서 `--tpcCount`는 `1 <= tpcCount < totalTPC` 범위를 만족해야 합니다.

### 최근 업데이트 (2026-02)

#### 1) Runner 실행 컨텍스트 메모리 분리

`LLMEngineRunner`에서 prefill/decode 실행 컨텍스트가 같은 device memory를 공유하지 않도록 분리했습니다.

- `mPrefillExecContextMemory`
- `mGenerationExecContextMemory`

이를 통해 prefill/decode 동시 실행 시 컨텍스트 메모리 충돌 가능성을 줄였습니다.

#### 2) KV cache 바인딩 경로 분리

KV cache 바인딩을 스테이지별로 분리했습니다.

- prefill 경로: `bindKVCacheToPrefillEngine()`
- decode 경로: `bindKVCacheToGenerationEngine()`

기존처럼 한 경로에서 두 execution context를 동시에 갱신하지 않으므로, 스테이지 간 setTensorAddress/setInputShape 간섭이 감소했습니다.

#### 3) Stage lock 분리

Disaggregation runtime에서 단일 runner lock 대신 stage별 lock을 사용합니다.

- `mRunnerPrefillMutex`
- `mRunnerDecodeMutex`

#### 4) Front 실행 고정 (queue 구조 유지)

큐는 기존처럼 3개를 유지합니다.

- `multimodalQueue`
- `prefillQueue`
- `decodeQueue`

다만 실행 정책은 front 파티션(encoding+prefill) 단위로 고정했습니다.

- 한 요청의 `encoding -> prefill`이 끝난 뒤 다음 요청 encoding을 시작
- 동시에 decode 파티션은 이전 요청을 계속 처리

즉, 목표 동작은 다음과 같습니다.

- `ReqA`가 decode 중일 때
- `ReqB`는 encoding+prefill 수행 가능
- 하지만 서로 다른 요청의 encoding과 prefill이 동시에 섞여 실행되지는 않음

#### 5) 병렬 동작 검증 로그 추가

병렬 동작 확인을 위해 stage 시작/종료 로그를 추가했습니다.

- `encoding(start/end)`
- `prefill(start/end)`
- `decode(start/end)`
- 요청 단위 식별자: `[Disagg][Req=N]`

예시:

```text
[Disagg][Req=1] prefill(end) ...
[Disagg][Req=2] encoding(start) ...
[Disagg][Req=1] decode(start) ...
```

위와 같이 front/back 파이프라인이 겹쳐 동작하는지 로그로 확인할 수 있습니다.

#### 6) 안정화 과정에서 반영된 주요 수정 사항

아래 이슈들은 `benchmarkCount=20` 반복 테스트 과정에서 실제로 재현되었고, 현재 코드에 반영되어 있습니다.

- `LinearKVCache::resetForNewSequences()`의 H2D 복사 크기를 tensor capacity 기준에서 `batchSize * sizeof(int32_t)` 기준으로 수정
  - 증상: `cudaMemcpyAsync ... invalid argument`
- `LLMEngineRunner`의 prefill/decode scratch tensor 수명을 `thread_local`로 보장
  - 증상: 비동기 커널 이후 `cudaFree` 시점의 illegal memory access
- `finalizeContext()`를 idempotent하게 변경 (`completionSet` 사용)
  - 증상: 예외 경로 중복 `set_value()`로 인한 recursive terminate
- decode CUDA graph는 disaggregation 경로에서 현재 비활성화
  - 안정화 우선 정책으로 운영 (`captureDecodingCUDAGraph()`는 warning 후 false 반환)

또한 동시성 안전화를 위해 다음 구조 변경이 함께 적용되었습니다.

- slot-aware KV cache API/호출 경로 사용 (`slotOffset` 기반)
- 요청별 `StageContext` tensor 분리 (runtime 공유 tensor 경합 감소)

#### 7) 검증 커맨드와 최신 결과 (이번 대화 기준)

검증 커맨드:

```bash
./build/examples/llm/llm_benchmark \
  --engineDir engines/qwen3-vl-2b \
  --multimodalEngineDir visual_engines/qwen3-vl-2b \
  --inputFile input_with_images.json \
  --outputFile output.json \
  --benchmarkCount=20 \
  --disaggregation \
  --tpcCount 3
```

최근 실행 결과 요약(`count=20`):

- encoding mean: `112.746 ms`
- prefill mean: `51.614 ms`
- decode mean: `13.941 ms/token`
- request mean: `1934.857 ms` (`0.517 req/s`)
- 처리 성공: `20/20`

#### 8) 현재 동작 요약 및 한계

- 현재 disaggregation은 **front(encoding+prefill) + back(decode)** 파이프라인을 목표로 동작
- front는 요청 단위로 순차 고정 (`encoding -> prefill` 완료 후 다음 요청 encoding 시작)
- back(decode)은 front와 병렬로 진행
- `--tpcCount`는 benchmark disaggregation 경로에서 decode 전용 TPC 개수로 해석

현재 알려진 한계:

- decode CUDA graph는 disaggregation 경로에서 비활성화 상태
- Jetson Thor CUDA 13.x 기준 mask 오프셋에 의존

#### tpcCount 스윕 결과 (qwen3-vl-2b, benchmarkCount=200)

- `tpc=1`은 측정 정체로 `N/A` 처리

**Mean(ms)**

| tpcCount | encoding | prefill | decode | request |
|---------:|---------:|--------:|-------:|--------:|
| 1 | N/A | N/A | N/A | N/A |
| 2 | 250.282 | 46.465 | 14.112 | 2088.927 |
| 3 | 172.174 | 33.610 | 11.279 | 1638.253 |
| 4 | 133.209 | 25.903 | 9.322 | 1342.947 |
| 5 | 115.236 | 23.331 | 8.573 | 1227.312 |
| 6 | 104.091 | 20.849 | 8.125 | 1156.833 |
| 7 | 99.398 | 19.852 | 8.106 | 1148.755 |
| 8 | 96.593 | 18.369 | 7.947 | 1124.194 |
| 9 | 95.272 | 18.098 | 7.958 | 1124.094 |
| 10 | 108.016 | 19.938 | 9.014 | 1272.716|

**P99(ms)**

| tpcCount | encoding | prefill | decode | request |
|---------:|---------:|--------:|-------:|--------:|
| 1 | N/A | N/A | N/A | N/A |
| 2 | 251.489 | 46.752 | 14.281 | 2111.922 |
| 3 | 174.420 | 34.300 | 11.519 | 1671.691 |
| 4 | 137.862 | 27.572 | 9.660 | 1392.228 |
| 5 | 118.229 | 23.807 | 8.704 | 1247.495 |
| 6 | 104.527 | 21.051 | 8.205 | 1167.573 |
| 7 | 102.316 | 20.379 | 8.231 | 1168.072 |
| 8 | 97.959 | 18.648 | 8.060 | 1140.234 |
| 9 | 95.981 | 18.370 | 8.092 | 1142.070 |
| 10 | 111.342 | 21.492 | 9.672 | 1361.234 |

#### 9) `--quiet` 3회 재측정 결과 (qwen3-vl-2b, benchmarkCount=200)

측정 목적:
- 동일 입력(`input_with_images.json`)에서 모드별 편차를 줄이기 위해 각 조건을 3회 반복 실행
- `--quiet`를 사용해 disaggregation 상세 로그 스팸을 제거하고 요약 지표만 비교

측정 커맨드:

```bash
# Baseline
./build/examples/llm/llm_benchmark \
  --engineDir engines/qwen3-vl-2b \
  --multimodalEngineDir visual_engines/qwen3-vl-2b \
  --inputFile input_with_images.json \
  --outputFile /tmp/llm_bench_q_base_1.json \
  --benchmarkCount=200 \
  --quiet

# Disaggregation (decode CUDA graph off; default)
./build/examples/llm/llm_benchmark \
  --engineDir engines/qwen3-vl-2b \
  --multimodalEngineDir visual_engines/qwen3-vl-2b \
  --inputFile input_with_images.json \
  --outputFile /tmp/llm_bench_q_disagg_1.json \
  --benchmarkCount=200 \
  --disaggregation \
  --quiet

# Disaggregation + decode CUDA graph on
./build/examples/llm/llm_benchmark \
  --engineDir engines/qwen3-vl-2b \
  --multimodalEngineDir visual_engines/qwen3-vl-2b \
  --inputFile input_with_images.json \
  --outputFile /tmp/llm_bench_q_disagg_graph_1.json \
  --benchmarkCount=200 \
  --disaggregation \
  --disaggDecodeCudaGraph \
  --quiet
```

3회 평균 결과:

| Mode | encoding mean (ms) | prefill mean (ms) | decode mean (ms/token) | request mean (ms) | request p99 (ms) | request throughput (req/s) |
|------|--------------------:|------------------:|------------------------:|------------------:|-----------------:|---------------------------:|
| Baseline | 100.823 | 18.714 | 8.468 | 1194.996 | 1257.935 | 0.840 |
| Disagg (graph off) | 90.147 | 40.591 | 10.307 | 1439.764 | 1509.307 | 0.695 |
| Disagg (graph on) | 96.545 | 44.171 | 10.993 | 1536.883 | 1651.789 | 0.653 |

요약:
- Baseline 대비 Disagg(graph off)에서는 `request mean +20.48%`, `throughput -17.27%`로 악화
- Disagg(graph off) 대비 Disagg(graph on)에서도 `request mean +6.75%`, `throughput -6.00%`로 추가 악화
- 즉, 현재 환경/설정에서는 disaggregation과 disagg decode CUDA graph 활성화가 모두 E2E 성능 개선으로 이어지지 않음
