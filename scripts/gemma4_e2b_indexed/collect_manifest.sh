#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

repo_dir=${REPO_DIR:-$(git rev-parse --show-toplevel)}
output=${1:-baseline-manifest.txt}
: "${MODEL_DIR:?MODEL_DIR must point at the pinned HF snapshot}"

{
    git -C "${repo_dir}" rev-parse HEAD
    git -C "${repo_dir}" status --short
    nvidia-smi
    nvcc --version || true
    g++ --version || true
    cmake --version || true
    trtexec --version || true
    docker version || true
    docker image inspect nvcr.io/nvidia/pytorch:25.12-py3 \
        --format '{{index .RepoDigests 0}}' || true
    docker image inspect nvcr.io/nvidia/tensorrt:26.06-py3 \
        --format '{{index .RepoDigests 0}}' || true
    find "${MODEL_DIR}" -type f -print0 | sort -z | xargs -0 sha256sum
    if [[ -n "${ARTIFACT_DIR:-}" && -d "${ARTIFACT_DIR}" ]]; then
        find "${ARTIFACT_DIR}" -type f -printf "%s %p\n" | sort -n
    fi
} > "${output}"
