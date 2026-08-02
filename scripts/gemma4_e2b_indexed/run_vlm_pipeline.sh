#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

repo_dir=${REPO_DIR:-$(git rev-parse --show-toplevel)}
work_dir=${WORK_DIR:-/tmp/gemma4-e2b}
pytorch_image=${PYTORCH_IMAGE:-nvcr.io/nvidia/pytorch:25.12-py3}
trt_image=${TRT_IMAGE:-nvcr.io/nvidia/tensorrt:26.06-py3}
stage=${1:-all}

container_repo=/workspace/TensorRT-Edge-LLM
container_work=/workspace/artifacts/gemma4-e2b

run_export()
{
    test -f "${work_dir}/hf/config.json"
    docker run --rm --gpus all \
        -v "${repo_dir}:${container_repo}" \
        -v "${work_dir}:${container_work}" \
        -w "${container_repo}" \
        "${pytorch_image}" bash -lc \
        "python3 -m pip install -e '${container_repo}[tools]' && \
         tensorrt-edgellm-export '${container_work}/hf' '${container_work}/onnx-multimodal' \
             --dtype float16 --skip-llm --components visual"
}

run_build()
{
    test -f "${work_dir}/onnx-multimodal/visual/model.onnx"
    docker run --rm --gpus all \
        -v "${repo_dir}:${container_repo}" \
        -v "${work_dir}:${container_work}" \
        -w "${container_repo}" \
        "${trt_image}" bash -lc \
        "cmake -S . -B build -DTRT_PACKAGE_DIR=/opt/tensorrt -DBUILD_UNIT_TESTS=ON && \
         cmake --build build --parallel --target visual_build llm_inference llm_phase_bench unitTest && \
         LD_LIBRARY_PATH='${container_repo}/build:/opt/tensorrt/lib:/usr/local/cuda/lib64' \
         ./build/examples/multimodal/visual_build \
             --onnxDir '${container_work}/onnx-multimodal/visual' \
             --engineDir '${container_work}/engine-multimodal' \
             --minImageTokens 4 --maxImageTokens 1120 --maxImageTokensPerImage 280 \
             --profilingDetailed"
}

run_inference()
{
    test -f "${work_dir}/engine-indexed/llm.engine"
    test -f "${work_dir}/engine-multimodal/visual/visual.engine"
    docker run --rm --gpus all \
        -v "${repo_dir}:${container_repo}" \
        -v "${work_dir}:${container_work}" \
        -w "${container_repo}" \
        -e "LD_LIBRARY_PATH=${container_repo}/build:/opt/tensorrt/lib:/usr/local/cuda/lib64" \
        "${trt_image}" ./build/examples/llm/llm_inference \
        --engineDir "${container_work}/engine-indexed" \
        --multimodalEngineDir "${container_work}/engine-multimodal" \
        --inputFile "${container_repo}/tests/test_cases/vlm_basic.json" \
        --maxGenerateLength 16 \
        --outputFile "${container_work}/vlm-indexed-output.json" \
        --dumpProfile \
        --profileOutputFile "${container_work}/vlm-indexed-profile.json"
}

case "${stage}" in
export)
    run_export
    ;;
build)
    run_build
    ;;
infer)
    run_inference
    ;;
all)
    run_export
    run_build
    run_inference
    ;;
*)
    echo "Usage: $0 [export|build|infer|all]" >&2
    exit 2
    ;;
esac
