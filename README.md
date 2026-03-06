<div align="center">

# TensorRT Edge-LLM


```bash
cd build
cmake .. \
    -DTRT_PACKAGE_DIR=/usr \
    -DCMAKE_TOOLCHAIN_FILE=cmake/aarch64_linux_toolchain.cmake \
    -DEMBEDDED_TARGET=jetson-thor
make -j$(nproc)
```

### Docker environment

```bash
cd ~/Documents/TensorRT-Edge-LLM/

docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -it --rm \
    -v $(pwd):/workspace \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -w /workspace \
    nvcr.io/nvidia/pytorch:25.12-py3 \
    bash

python3 -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'Torch Version: {torch.__version__}')"

pip install . 

pip show tensorrt-edgellm
```

### Model export (Qwen3-VL-2B example)

```bash
tensorrt-edgellm-quantize-llm \
    --model_dir Qwen/Qwen3-VL-2B-Instruct \
    --output_dir /workspace/quantized/qwen3-vl-2b \
    --quantization nvfp4

tensorrt-edgellm-export-llm \
    --model_dir /workspace/quantized/qwen3-vl-2b \
    --output_dir /workspace/onnx_models/qwen3-vl-2b

tensorrt-edgellm-export-visual \
    --model_dir /workspace/quantized/qwen3-vl-2b \
    --output_dir /workspace/visual_enc_onnx/qwen3-vl-2b

```

### Engine build

```bash
./build/examples/llm/llm_build \
  --onnxDir /workspace/onnx_models/qwen3-vl-2b \
  --engineDir /workspace/engines/qwen3-vl-2b \
  --maxBatchSize 4 \
  --maxInputLen=1024 \
  --maxKVCacheCapacity=4096 

./build/examples/multimodal/visual_build \
  --onnxDir /workspace/visual_enc_onnx/qwen3-vl-2b \
  --engineDir /workspace/visual_engines/qwen3-vl-2b
```

### Inference

```bash
./build/examples/llm/llm_inference \
  --engineDir engines/qwen3-vl-2b \
  --multimodalEngineDir visual_engines/qwen3-vl-2b \
  --inputFile input_with_images.json \
  --outputFile output.json


docker run --gpus all --ipc=host \
    --ulimit memlock=-1 --ulimit stack=67108864 -it --rm \
    -v $(pwd):/workspace \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -w /workspace \
    nvcr.io/nvidia/pytorch:25.12-py3 \
    ./build/examples/llm/llm_inference \
    --engineDir engines/qwen3-vl-2b \
    --multimodalEngineDir visual_engines/qwen3-vl-2b \
    --inputFile input_with_images.json \
    --outputFile output.json

```



**High-Performance Large Language Model Inference Framework for NVIDIA Edge Platforms**

[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat)](https://nvidia.github.io/TensorRT-Edge-LLM/)
[![version](https://img.shields.io/badge/release-0.5.0-green)](https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/main/tensorrt_edgellm/version.py)
[![license](https://img.shields.io/badge/license-Apache%202-blue)](https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/main/LICENSE)

[Overview](https://nvidia.github.io/TensorRT-Edge-LLM/latest/developer_guide/getting-started/overview.html)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[Examples](https://nvidia.github.io/TensorRT-Edge-LLM/latest/developer_guide/getting-started/examples.html)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[Documentation](https://nvidia.github.io/TensorRT-Edge-LLM/)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[Roadmap](https://github.com/NVIDIA/TensorRT-Edge-LLM/issues?q=is%3Aissue%20state%3Aopen%20label%3ARoadmap)

---
<div align="left">

## Overview

TensorRT Edge-LLM is NVIDIA's high-performance C++ inference runtime for Large Language Models (LLMs) and Vision-Language Models (VLMs) on embedded platforms. It enables efficient deployment of state-of-the-art language models on resource-constrained devices such as NVIDIA Jetson and NVIDIA DRIVE platforms. TensorRT Edge-LLM provides convenient Python scripts to convert HuggingFace checkpoints to [ONNX](https://onnx.ai). Engine build and end-to-end inference runs entirely on Edge platforms.

---

## Getting Started

For the supported platforms, models and precisions, see the [**Overview**](https://nvidia.github.io/TensorRT-Edge-LLM/latest/developer_guide/getting-started/overview.html). Get started with TensorRT Edge-LLM in <15 minutes. For complete installation and usage instructions, see the [**Quick Start Guide**](https://nvidia.github.io/TensorRT-Edge-LLM/latest/developer_guide/getting-started/quick-start-guide.html).

---

## Documentation

### Introduction

- **[Overview](https://nvidia.github.io/TensorRT-Edge-LLM/latest/developer_guide/getting-started/overview.html)** - What is TensorRT Edge-LLM and key features
- **[Supported Models](https://nvidia.github.io/TensorRT-Edge-LLM/latest/developer_guide/getting-started/supported-models.html)** - Complete model compatibility matrix

### User Guide

- **[Installation](https://nvidia.github.io/TensorRT-Edge-LLM/latest/developer_guide/getting-started/installation.html)** - Set up Python export pipeline and C++ runtime
- **[Quick Start Guide](https://nvidia.github.io/TensorRT-Edge-LLM/latest/developer_guide/getting-started/quick-start-guide.html)** - Run your first inference in ~15 minutes
- **[Examples](https://nvidia.github.io/TensorRT-Edge-LLM/latest/developer_guide/getting-started/examples.html)** - End-to-end LLM, VLM, EAGLE, and LoRA workflows
- **[Input Format Guide](https://nvidia.github.io/TensorRT-Edge-LLM/latest/developer_guide/getting-started/input-format.html)** - Request format and specifications
- **[Chat Template Format](https://nvidia.github.io/TensorRT-Edge-LLM/latest/developer_guide/getting-started/chat-template-format.html)** - Chat template configuration

### Developer Guide

#### Software Design

- **[Python Export Pipeline](https://nvidia.github.io/TensorRT-Edge-LLM/latest/developer_guide/software-design/python-export-pipeline.html)** - Model export and quantization
- **[Engine Builder](https://nvidia.github.io/TensorRT-Edge-LLM/latest/developer_guide/software-design/engine-builder.html)** - Building TensorRT engines
- **[C++ Runtime Overview](https://nvidia.github.io/TensorRT-Edge-LLM/latest/developer_guide/software-design/cpp-runtime-overview.html)** - Runtime system architecture
  - [LLM Inference Runtime](https://nvidia.github.io/TensorRT-Edge-LLM/latest/developer_guide/software-design/llm-inference-runtime.html)
  - [LLM SpecDecode Runtime](https://nvidia.github.io/TensorRT-Edge-LLM/latest/developer_guide/software-design/llm-inference-specdecode-runtime.html)

#### Advanced Topics

- **[Customization Guide](https://nvidia.github.io/TensorRT-Edge-LLM/latest/developer_guide/customization/customization-guide.html)** - Customizing TensorRT Edge-LLM for your needs
- **[TensorRT Plugins](https://nvidia.github.io/TensorRT-Edge-LLM/latest/developer_guide/customization/tensorrt-plugins.html)** - Custom plugin development
- **[Tests](tests/)** - Comprehensive test suite for contributors

---

## Use Cases

**🚗 Automotive**
- In-vehicle AI assistants
- Voice-controlled interfaces
- Scene understanding
- Driver assistance systems

**🤖 Robotics**
- Natural language interaction
- Task planning and reasoning
- Visual question answering
- Human-robot collaboration

**🏭 Industrial IoT**
- Equipment monitoring with NLP
- Automated inspection
- Predictive maintenance
- Voice-controlled machinery

**📱 Edge Devices**
- On-device chatbots
- Offline language processing
- Privacy-preserving AI
- Low-latency inference

---

## Tech Blogs

*Coming soon*

Stay tuned for technical deep-dives, optimization guides, and deployment best practices.

---

## Latest News

* [01/05] 🚀 Accelerate AI Inference for Edge and Robotics with NVIDIA Jetson T4000 and NVIDIA JetPack 7.1 ✨ [➡️ link](https://developer.nvidia.com/blog/accelerate-ai-inference-for-edge-and-robotics-with-nvidia-jetson-t4000-and-nvidia-jetpack-7-1/)
* [01/05] 🚀 Accelerating LLM and VLM Inference for Automotive and Robotics with NVIDIA TensorRT Edge-LLM ✨ [➡️ link](https://developer.nvidia.com/blog/accelerating-llm-and-vlm-inference-for-automotive-and-robotics-with-nvidia-tensorrt-edge-llm/)

Follow our [GitHub repository](https://github.com/NVIDIA/TensorRT-Edge-LLM) for the latest updates, releases, and announcements.

---

## Support

- **Documentation**: [Full Documentation](https://nvidia.github.io/TensorRT-Edge-LLM/)
- **Examples**: [Code Examples](https://nvidia.github.io/TensorRT-Edge-LLM/latest/developer_guide/getting-started/examples.html)
- **Roadmap**: [Developer Roadmap](https://github.com/NVIDIA/TensorRT-Edge-LLM/issues?q=is%3Aissue%20state%3Aopen%20label%3ARoadmap)
- **Issues**: [GitHub Issues](https://github.com/NVIDIA/TensorRT-Edge-LLM/issues)
- **Discussions**: [GitHub Discussions](https://github.com/NVIDIA/TensorRT-Edge-LLM/discussions)
- **Forums**: [NVIDIA Developer Forums](https://forums.developer.nvidia.com/)

---

## License

[Apache License 2.0](LICENSE)

---

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

---
