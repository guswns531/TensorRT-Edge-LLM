#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

repo_dir=${REPO_DIR:-/workspace/TensorRT-Edge-LLM}
work_dir=${WORK_DIR:-/workspace/artifacts/gemma4-e2b}
mode=${MODE:?MODE must be legacy or indexed}
bench=${repo_dir}/build/examples/llm/llm_bench
engine=${work_dir}/engine-${mode}
output=${work_dir}/bench-${mode}

: "${TRT_PACKAGE_DIR:?TRT_PACKAGE_DIR must point at TensorRT}"
export LD_LIBRARY_PATH="${TRT_PACKAGE_DIR}/lib:${LD_LIBRARY_PATH:-}"
mkdir -p "${output}"

for repeat in 1 2 3; do
    for batch in 1 4; do
        for length in 128 512 1024; do
            "${bench}" --engineDir "${engine}" --mode prefill \
                --batchSize "${batch}" --inputLen "${length}" \
                --warmup 20 --iterations 100 --seed 0 --noCudaGraph \
                --outputDir "${output}/run-${repeat}/bs-${batch}"
        done
        for past in 128 512 1536; do
            "${bench}" --engineDir "${engine}" --mode decode \
                --batchSize "${batch}" --pastKVLen "${past}" \
                --warmup 20 --iterations 100 --seed 0 --noCudaGraph \
                --outputDir "${output}/run-${repeat}/bs-${batch}"
        done
    done
done

# CUDA-graph path is a smoke test only; primary gate uses --noCudaGraph above.
"${bench}" --engineDir "${engine}" --mode decode --batchSize 4 \
    --pastKVLen 512 --warmup 2 --iterations 3 --seed 0 \
    --outputDir "${output}/cuda-graph-smoke"
