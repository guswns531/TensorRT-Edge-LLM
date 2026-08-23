# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

ARG BASE_IMAGE=tensorrt-edge-llm:gemma4-e2b
FROM ${BASE_IMAGE}

ARG MAX_JOBS=12
ARG SOURCE_REF=unknown

ENV EDGELLM_HOME=/opt/TensorRT-Edge-LLM-v010 \
    TRT_PACKAGE_DIR=/usr \
    CUDA_CTK_VERSION=13.2 \
    EMBEDDED_TARGET=jetson-thor \
    ENABLE_CUTE_DSL=ALL \
    CUTE_DSL_ARTIFACT_TAG=sm_110 \
    CUDA_CUBIN_ARCH=sm_110a \
    MAX_JOBS=${MAX_JOBS}

COPY . ${EDGELLM_HOME}
WORKDIR ${EDGELLM_HOME}

RUN experimental/docker/build.sh

RUN cmake -S . -B build -DBUILD_UNIT_TESTS=ON && \
    cmake --build build \
        --target unitTest llm_build llm_inference llm_bench \
        --parallel ${MAX_JOBS} && \
    if cmake --build build --target help | grep -q llm_phase_context_smoke; then \
        cmake --build build --target llm_phase_context_smoke --parallel ${MAX_JOBS}; \
    fi

RUN python3 -m pip install --break-system-packages --no-cache-dir \
    'nvidia-ml-py>=12' \
    'omegaconf==2.3.0' \
    'pulp<4' && \
    python3 -m pip install --break-system-packages --no-cache-dir --no-deps \
        'nvidia-modelopt==0.45.0'

LABEL org.opencontainers.image.revision=${SOURCE_REF}

ENV EDGELLM_PLUGIN_PATH=${EDGELLM_HOME}/build/libNvInfer_edgellm_plugin.so \
    PYTHONPATH=${EDGELLM_HOME}:${EDGELLM_HOME}/build/pybind \
    LD_LIBRARY_PATH=${EDGELLM_HOME}/build:${EDGELLM_HOME}/build/pybind:/usr/lib/aarch64-linux-gnu:/usr/local/cuda/lib64:/usr/local/cuda/targets/sbsa-linux/lib:${LD_LIBRARY_PATH}
