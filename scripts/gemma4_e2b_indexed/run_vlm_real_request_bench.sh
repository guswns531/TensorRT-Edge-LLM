#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

repo_dir=${REPO_DIR:-$(git rev-parse --show-toplevel)}
work_dir=${WORK_DIR:-/tmp/gemma4-e2b}
result_dir=${RESULT_DIR:-${work_dir}/perf/vlm-real}
trt_image=${TRT_IMAGE:-nvcr.io/nvidia/tensorrt:26.06-py3}
repeats=${REPEATS:-3}
warmup=${WARMUP:-2}

container_repo=/workspace/TensorRT-Edge-LLM
container_work=/workspace/artifacts/gemma4-e2b
input_file=${container_repo}/tests/test_cases/gemma4_vlm_real_requests.json

test -f "${work_dir}/engine-legacy/llm.engine"
test -f "${work_dir}/engine-indexed/llm.engine"
test -f "${work_dir}/engine-multimodal/visual/visual.engine"
mkdir -p "${result_dir}"

run_one()
{
    local variant=$1
    local engine_mode=$2
    local batch_size=$3
    local repeat=$4

    docker run --rm --gpus all \
        -v "${repo_dir}:${container_repo}" \
        -v "${work_dir}:${container_work}" \
        -w "${container_repo}" \
        -e "LD_LIBRARY_PATH=${container_repo}/build:/opt/tensorrt/lib:/usr/local/cuda/lib64" \
        "${trt_image}" ./build/examples/llm/llm_inference \
        --engineDir "${container_work}/engine-${engine_mode}" \
        --multimodalEngineDir "${container_work}/engine-multimodal" \
        --inputFile "${input_file}" \
        --batchSize "${batch_size}" \
        --maxGenerateLength 96 \
        --warmup "${warmup}" \
        --dumpProfile \
        --outputFile "${container_work}/perf/vlm-real/${variant}-run-${repeat}-output.json" \
        --profileOutputFile "${container_work}/perf/vlm-real/${variant}-run-${repeat}-profile.json" \
        > "${result_dir}/${variant}-run-${repeat}.log" 2>&1
}

for ((repeat = 1; repeat <= repeats; ++repeat)); do
    if ((repeat % 2 == 1)); then
        modes=(legacy indexed)
    else
        modes=(indexed legacy)
    fi
    for mode in "${modes[@]}"; do
        run_one "${mode}" "${mode}" 2 "${repeat}"
        echo "completed ${mode} BS2 run ${repeat}"
    done
done

for ((repeat = 1; repeat <= repeats; ++repeat)); do
    run_one indexed-bs1 indexed 1 "${repeat}"
    echo "completed indexed BS1 run ${repeat}"
done

python3 "${repo_dir}/scripts/gemma4_e2b_indexed/summarize_vlm_real_requests.py" \
    "${result_dir}" "${result_dir}/legacy-indexed-summary.csv" --repeats "${repeats}"
python3 "${repo_dir}/scripts/gemma4_e2b_indexed/summarize_vlm_real_requests.py" \
    "${result_dir}" "${result_dir}/indexed-bs1-bs2-summary.csv" --repeats "${repeats}" \
    --baseline indexed-bs1 --candidate indexed
