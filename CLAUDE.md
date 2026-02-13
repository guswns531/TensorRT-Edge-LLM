# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TensorRT Edge-LLM is NVIDIA's high-performance C++ inference runtime for LLMs and VLMs on embedded platforms (Jetson, DRIVE). It implements a three-stage pipeline:

1. **Python Export Pipeline** — quantizes HuggingFace models and exports to ONNX
2. **Engine Builder** — compiles ONNX models to TensorRT engines (`llm_build`, `visual_build`)
3. **C++ Runtime** — executes TensorRT engines with CUDA graphs, KV cache, and speculative decoding

## Build Commands

### C++ Build
```bash
mkdir -p build && cd build
cmake .. \
    -DTRT_PACKAGE_DIR=/usr \
    -DCMAKE_TOOLCHAIN_FILE=cmake/aarch64_linux_toolchain.cmake \
    -DEMBEDDED_TARGET=jetson-thor \
    -DBUILD_UNIT_TESTS=ON
make -j$(nproc)
```

Required CMake variables: `-DTRT_PACKAGE_DIR` (TensorRT installation path).

### Python Package Build
```bash
pip install build
python -m build --wheel --outdir dist .
pip install dist/*.whl
```

### Environment Setup (required for CUDA/export tools)
```bash
export CUDA_HOME=/usr/local/cuda
export TRITON_PTXAS_PATH="${CUDA_HOME}/bin/ptxas"
export PATH="${CUDA_HOME}/bin:$PATH"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:$LD_LIBRARY_PATH"
```

## Testing

### Setup
```bash
export LLM_SDK_DIR=$(pwd)
export ONNX_DIR=/path/to/onnx/models
export ENGINE_DIR=/path/to/engine/outputs
export LLM_MODELS_DIR=/path/to/pytorch/models
export EDGELLM_DATA_DIR=/path/to/datasets
pip install -r tests/requirements.txt
```

### Run Tests
```bash
# Run a specific test suite (target GPU/platform)
pytest --priority=l0_pipeline_a30 -v
pytest --priority=l0_export_ampere -v
pytest --priority=l0_pipeline_orin -v

# Remote execution (for embedded targets)
pytest --priority=l0_pipeline_orin \
       --execution-mode=remote \
       --remote-host=192.168.55.1
```

Available test suite YAMLs in `tests/test_lists/`: `l0_export_ampere`, `l0_export_blackwell`, `l0_pipeline_a30`, `l0_pipeline_orin`, `l0_pipeline_rtx5080`, `l0_pipeline_thor_ferrix`.

## Linting and Formatting

Pre-commit hooks handle all formatting automatically:
```bash
pip install pre-commit
pre-commit install
# Runs on every commit; or run manually:
pre-commit run --all-files
```

**C++**: `clang-format` (≥14.0), column limit 120, Allman brace style
```bash
git-clang-format --style file [commit-id]
```

**Python**: `black` (≥23.0), line length 120
```bash
git diff --name-only | grep "*.py" | xargs black -l 120
```

If hook modifies files, re-stage and recommit.

## End-to-End Workflow

```bash
# 1. Quantize model
tensorrt-edgellm-quantize-llm \
    --model_dir Qwen/Qwen3-VL-2B-Instruct \
    --output_dir /path/to/quantized/qwen3-vl-2b \
    --quantization nvfp4

# 2. Export LLM + visual encoder to ONNX
tensorrt-edgellm-export-llm \
    --model_dir /path/to/quantized/qwen3-vl-2b \
    --output_dir /path/to/onnx/qwen3-vl-2b
tensorrt-edgellm-export-visual \
    --model_dir /path/to/quantized/qwen3-vl-2b \
    --output_dir /path/to/visual_onnx/qwen3-vl-2b

# 3. Build TensorRT engines
./build/examples/llm/llm_build \
    --onnxDir /path/to/onnx/qwen3-vl-2b \
    --engineDir /path/to/engines/qwen3-vl-2b \
    --maxBatchSize 4 --maxInputLen=1024 --maxKVCacheCapacity=4096 \
    --vlm --minImageTokens 128 --maxImageTokens 512
./build/examples/multimodal/visual_build \
    --onnxDir /path/to/visual_onnx/qwen3-vl-2b \
    --engineDir /path/to/visual_engines/qwen3-vl-2b

# 4. Run inference
./build/examples/llm/llm_inference \
    --engineDir engines/qwen3-vl-2b \
    --multimodalEngineDir visual_engines/qwen3-vl-2b \
    --inputFile input.json --outputFile output.json
```

## Architecture

### C++ Runtime (`cpp/`)

Two distinct runtime implementations in `cpp/runtime/`:
- **`LLMInferenceRuntime`** — standard inference (text-only and VLM)
- **`LLMInferenceSpecDecodeRuntime`** — EAGLE3 speculative decoding with two engine runners (base + draft)

Key components:
- `cpp/runtime/llmEngineRunner.*` — manages TensorRT engine execution and CUDA graphs
- `cpp/runtime/linearKVCache.*` — KV cache with eviction policies
- `cpp/kernels/contextAttentionKernels/` — prefill-phase FMHA (pre-compiled cubins)
- `cpp/kernels/decodeAttentionKernels/` — decode-phase XQA kernels (pre-compiled cubins)
- `cpp/common/tensor.*` — unified memory management for runtime data
- `cpp/plugins/` — TensorRT plugin (`NvInfer_edgellm_plugin.so`)

Build output libraries: `edgellmCore` (static), `edgellmTokenizer` (static), `edgellmBuilder` (static), `NvInfer_edgellm_plugin` (shared).

### Python Package (`tensorrt_edgellm/`)

- `quantization/` — FP8, INT4 AWQ, INT8 SQ, NVFP4 quantization via `nvidia-modelopt`
- `onnx_export/` — PyTorch → ONNX export for LLM, EAGLE3 draft, and visual encoders
- `llm_models/` — model definitions and attention/GEMM plugin layers
- `scripts/` — CLI entry points (installed as `tensorrt-edgellm-*` commands)

### Supported Quantization
`fp16`, `fp8` (SM89+), `int8_sq`, `int4_awq`, `int4_gptq`, `nvfp4` (SM100+ Blackwell only)

### Supported Models
LLMs: Llama 3.x, Qwen 2/2.5/3, DeepSeek-R1-Distilled
VLMs: Qwen2/2.5/3-VL, InternVL3, Phi-4-Multimodal

## Coding Style

**C++**: Google C++ Style Guide base, Allman braces, 120-char limit, C++17.
- Variables/functions: `camelCase` (lowercase first)
- Types/classes: `PascalCase`
- Constants: `kPrefixedUpperSnakeCase`
- Class members: `mPrefixedCamelCase`
- Static/global: `sPrefixedCamelCase`
- Use `#pragma once` for header guards
- Smart pointers required (`unique_ptr` preferred over `shared_ptr`)
- All new public APIs need Doxygen comments (`//!` style)

**Python**: snake_case functions/variables, PascalCase classes, UPPER_SNAKE_CASE constants, 4-space indent, Google-style docstrings, import as `from package.sub import module; module.Class()`.

All new files need the Apache 2.0 NVIDIA copyright header (see `CODING_GUIDELINES.md`).

## Contribution Requirements

- All commits must be signed off: `git commit -s -m "message"`
- PR titles follow Conventional Commits: `feat:`, `fix:`, `chore:`, `docs:`, `refactor:`, `test:`
- One concern per PR; target `main` branch (or release branch per nvbug if applicable)
- Open a GitHub issue before submitting non-trivial changes
