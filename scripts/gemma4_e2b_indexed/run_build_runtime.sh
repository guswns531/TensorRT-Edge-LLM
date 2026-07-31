#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

repo_dir=${REPO_DIR:-/workspace/TensorRT-Edge-LLM}
work_dir=${WORK_DIR:-/workspace/artifacts/gemma4-e2b}
mode=${MODE:-legacy}
: "${TRT_PACKAGE_DIR:?TRT_PACKAGE_DIR must point at TensorRT}"
cuda_dir=${CUDA_DIR:-/usr/local/cuda-13.3}
cuda_ctk_version=${CUDA_CTK_VERSION:-13.3}
cuda_architectures=${CUDA_ARCHITECTURES:-86}
enable_cute_dsl=${ENABLE_CUTE_DSL:-ffpa}
cute_dsl_artifact_tag=${CUTE_DSL_ARTIFACT_TAG:-sm_86}

git -C "${repo_dir}" submodule update --init
cmake -S "${repo_dir}" -B "${repo_dir}/build" \
    -DTRT_PACKAGE_DIR="${TRT_PACKAGE_DIR}" \
    -DCUDA_DIR="${cuda_dir}" \
    -DCUDA_CTK_VERSION="${cuda_ctk_version}" \
    -DCMAKE_CUDA_ARCHITECTURES="${cuda_architectures}" \
    -DAARCH64_BUILD=OFF \
    -DENABLE_CUTE_DSL="${enable_cute_dsl}" \
    -DCUTE_DSL_ARTIFACT_TAG="${cute_dsl_artifact_tag}" \
    -DBUILD_UNIT_TESTS=ON
cmake --build "${repo_dir}/build" --parallel

export LD_LIBRARY_PATH="${TRT_PACKAGE_DIR}/lib:${LD_LIBRARY_PATH:-}"
"${repo_dir}/build/unitTest" --gtest_filter="KVSlotAllocatorTest.*"

"${repo_dir}/build/examples/llm/llm_build" \
    --onnxDir "${work_dir}/onnx-${mode}/llm" \
    --engineDir "${work_dir}/engine-${mode}" \
    --maxBatchSize 4 \
    --maxInputLen 1024 \
    --maxKVCacheCapacity 2048

"${repo_dir}/build/examples/llm/llm_inference" \
    --engineDir "${work_dir}/engine-${mode}" \
    --inputFile "${repo_dir}/tests/test_cases/llm_basic.json" \
    --outputFile "${work_dir}/llm-basic-${mode}.json"
