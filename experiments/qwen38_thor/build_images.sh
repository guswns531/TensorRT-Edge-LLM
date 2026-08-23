#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
phase_source="${PHASE_SOURCE:-$(cd "${script_dir}/../.." && pwd)}"
upstream_source="${UPSTREAM_SOURCE:-/home/sslab/Desktop/TensorRT-Edge-LLM-upstream-v010}"
phase_image="${PHASE_IMAGE:-tensorrt-edge-llm:qwen38-phase}"
upstream_image="${UPSTREAM_IMAGE:-tensorrt-edge-llm:qwen38-upstream-v010}"
max_jobs="${MAX_JOBS:-12}"

build_image() {
    local source_dir="$1"
    local image="$2"
    local source_ref

    source_ref="$(git -C "${source_dir}" rev-parse HEAD)"
    docker build \
        --network=host \
        --shm-size=8g \
        --build-arg "MAX_JOBS=${max_jobs}" \
        --build-arg "SOURCE_REF=${source_ref}" \
        -f "${script_dir}/Dockerfile" \
        -t "${image}" \
        "${source_dir}"
}

build_image "${upstream_source}" "${upstream_image}"
build_image "${phase_source}" "${phase_image}"

