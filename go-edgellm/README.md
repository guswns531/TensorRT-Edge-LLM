# Go EdgeLLM HTTP Server

This project allows you to serve **TensorRT-Edge-LLM** engines via an HTTP API using Go. It implements a **Dynamic Batching** mechanism where incoming HTTP requests are queued and processed in batches by C++ workers to maximize throughput.

## Architecture

*   **HTTP Server (`:8000`)**: Accepts requests via `/generate` (inference) and `/benchmark` (stress test).
*   **Request Queue**: Incoming requests are buffered in a channel.
*   **Dynamic Batcher**: A background routine collects requests from the queue and forms batches based on a configurable `BatchSize` or a time timeout (10ms).
*   **C++ Workers**: Multiple concurrent workers process batches using the underlying TensorRT-Edge-LLM C++ Runtime.

## Prerequisites

This environment is designed to run inside the **TensorRT-LLM Docker container**.

### 1. Start Docker Container
```bash
docker run --gpus all -it --name tensorrt --net=host \
    -v /home/sslab/nfs/TensorRT-Edge-LLM:/workspace \
    -v /home/sslab/nfs/cache:/root/.cache \
    -w /workspace \
    nvcr.io/nvidia/tensorrt:25.10-py3
```
#### Atatch 
```bash
docker exec -it 4f54eee2e87e /bin/bash
```

### 2. Install Dependencies (Inside Docker)
```bash
apt update ; apt install -y cmake build-essential
pip3 install .

# Install Go (if not already installed)
wget https://go.dev/dl/go1.25.6.linux-amd64.tar.gz && \
rm -rf /usr/local/go && \
tar -C /usr/local -xzf go1.25.6.linux-amd64.tar.gz && \
rm go1.25.6.linux-amd64.tar.gz

export PATH=$PATH:/usr/local/go/bin
```

## Model Preparation

Before running the server, you need to build the TensorRT engine.

### Text Model Qwen3-0.6B
```bash
# 1. Quantize (e.g., Qwen3-0.6B)
tensorrt-edgellm-quantize-llm \
    --model_dir Qwen/Qwen3-0.6B \
    --output_dir ./quantized/qwen3-0.6b \
    --quantization int4_awq

# 2. Export to ONNX
tensorrt-edgellm-export-llm \
    --model_dir ./quantized/qwen3-0.6b \
    --output_dir ./onnx_models/qwen3-0.6b

# 3. Build Engine
./build/examples/llm/llm_build \
    --onnxDir ./onnx_models/qwen3-0.6b \
    --engineDir ./engines/qwen3-0.6b
```
### Vision Model  
```bash

# 1. Quantize (e.g., Qwen3-VL-2B-Instruct)
tensorrt-edgellm-quantize-llm \
    --model_dir Qwen/Qwen3-VL-2B-Instruct \
    --output_dir ./quantized/qwen3-vl-2b_fp8 \
    --quantization fp8

# 2. Export to ONNX
tensorrt-edgellm-export-llm \
  --model_dir ./quantized/qwen3-vl-2b_fp8 \
  --output_dir onnx_models/qwen3-vl-2b_fp8

tensorrt-edgellm-export-visual \
  --model_dir Qwen/Qwen3-VL-2B-Instruct \
  --output_dir onnx_models/qwen3-vl-2b_fp8/visual_enc_onnx \
  --quantization fp8
  
# 3. Build Engine
./build/examples/llm/llm_build \
  --onnxDir onnx_models/qwen3-vl-2b_fp8 \
  --engineDir engines/qwen3-vl-2b_fp8 \
  --maxBatchSize 1 \
  --maxInputLen=1024 \
  --maxKVCacheCapacity=4096 \
  --vlm \
  --minImageTokens 128 \
  --maxImageTokens 512

./build/examples/multimodal/visual_build \ 
  --onnxDir onnx_models/qwen3-vl-2b_fp8/visual_enc_onnx \ 
  --engineDir engines/qwen3-vl-2b_fp8


tensorrt-edgellm-export-llm \
  --model_dir Qwen/Qwen3-VL-2B-Instruct \
  --output_dir onnx_models/qwen3-vl-2b

tensorrt-edgellm-export-visual \
  --model_dir Qwen/Qwen3-VL-2B-Instruct \
  --output_dir onnx_models/qwen3-vl-2b/visual_enc_onnx 

./build/examples/llm/llm_build \
  --onnxDir onnx_models/qwen3-vl-2b \
  --engineDir engines/qwen3-vl-2b \
  --vlm 

./build/examples/multimodal/visual_build --onnxDir onnx_models/qwen3-vl-2b/visual_enc_onnx --engineDir visual_engines/qwen3-vl-2b

./build/examples/llm/llm_inference --engineDir engines/qwen3-vl-2b --multimodalEngineDir visual_engines/qwen3-vl-2b --inputFile input_with_images.json --outputFile output.json



```

## Compilation

### Step 1: Build C++ Runtime Library
The Go application links against `libedge_llm_runtime.so`.

```bash
cd /workspace/build
cmake .. -DTRT_PACKAGE_DIR=/usr
make -j$(nproc) edge_llm_runtime
```

#### in Thor

```bash

mkdir build_thor
cd build_thor
cmake .. \
    -DTRT_PACKAGE_DIR=/usr \
    -DCMAKE_TOOLCHAIN_FILE=cmake/aarch64_linux_toolchain.cmake \
    -DEMBEDDED_TARGET=jetson-thor
make -j$(nproc)

```

### Step 2: Build Go Application

```bash
cd /workspace/go-edgellm
export PATH=$PATH:/usr/local/go/bin

go mod tidy
go build -o go-edgellm-app .
```

## Running the Server

You must set `LD_LIBRARY_PATH` to include the C++ build directory and `EDGELLM_PLUGIN_PATH` for TensorRT plugins.

```bash
cd /workspace/go-edgellm

# Set environment variables
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/workspace/build/cpp:/workspace/build
export EDGELLM_PLUGIN_PATH=/workspace/build/libNvInfer_edgellm_plugin.so

# Run the server (default port :8000, 4 workers)
./go-edgellm-app --workers 4

./go-edgellm-app --engine-dir=/workspace/engines/qwen3-vl-2b --multimodal-engine-dir=/workspace/visual_engines/qwen3-vl-2b --workers 1
```

## API Usage

### 1. Generate Text (`POST /generate`)

```bash
curl -X POST http://localhost:8000/generate \
  -d '{"prompt": "Hello via HTTP!", "max_tokens": 50, "temperature": 0.7}'

curl -X POST http://12.18.0.249:8000/generate \
  -d '{"prompt": "Hello via HTTP!", "max_tokens": 50, "temperature": 0.7}'

curl -X POST http://localhost:8000/generate -d '{
    "messages": [
        {"role": "user", "content": [
            {"type": "image", "image": "/workspace/examples/multimodal/pics/woman_and_dog.jpeg"},
            {"type": "text", "text": "Describe this image."}
        ]}
    ],
    "max_tokens": 1500
}'
```

### 2. Benchmark & Adjust Batch Size (`GET /benchmark`)

The server supports dynamic batch size adjustment and built-in benchmarking.

*   `requests`: Total number of requests to send.
*   `concurrency`: Number of concurrent clients (simulated).
*   `batch_size`: **Update** the server's max batch size (1-4) on the fly.

**Example: Run stress test with 20k requests, 16 concurrent clients, and set batch size to 4**

```bash
curl "http://localhost:8000/benchmark?requests=20000&concurrency=16&batch_size=4"

curl "http://localhost:8000/benchmark?requests=200&concurrency=16&batch_size=1&image=/workspace/examples/multimodal/pics/woman_and_dog.jpeg"
```

## Performance Results

Tested on NVIDIA Jetson / Edge device with Qwen3-0.6B (Int4):

```text
Starting Benchmark: Requests=20000, Concurrency=16, BatchSizeLimit=4

Benchmark Complete:
Total Time: 1m26.918s
Total Tokens: 180000
Throughput: 2070.90 tokens/sec
Avg Latency: 68.92ms
Avg TTFT: 22.58ms
Avg ITL: 5.15ms
Avg TPOT: 5.79ms
```

## Accuracy Test

### 1. Build Engine 
```
./build/examples/llm/llm_build \
  --onnxDir ./onnx_models/qwen3-0.6b \
  --engineDir ./engines/qwen3-0.6b_benchmark \
  --maxInputLen 8192 \
  --maxKVCacheCapacity 10240 \
  --maxBatchSize 1
```

### 2. Run MMLU Benchmark
```bash
# Prepare dataset
python3 scripts/prepare_dataset.py --dataset MMLU --output_dir datasets/mmlu_output

# Run inference
./build/examples/llm/llm_inference \
  --engineDir ./engines/qwen3-0.6b_benchmark \
  --inputFile ./datasets/mmlu_output/mmlu_dataset.json \
  --outputFile ./mmlu_predictions.json

# Calculate accuracy
python3 scripts/calculate_correctness.py \
  --predictions_file ./mmlu_predictions.json \
  --answers_file ./datasets/mmlu_output/mmlu_dataset.json
```
