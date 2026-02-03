# TensorRT Edge-LLM

**High-Performance Large Language Model Inference Framework for NVIDIA Edge Platforms**

---


docker run --gpus all -it --name tensorrt --net=host -v /home/sslab/nfs/TensorRT-Edge-LLM:/workspace -w /workspace nvcr.io/nvidia/tensorrt:25.10-py3

git submodule update --init --recursive

apt update ; apt install cmake build-essential
pip3 install .

mkdir build ; cd build
cmake ..     -DTRT_PACKAGE_DIR=/usr
make -j$(nproc)

tensorrt-edgellm-quantize-llm \
    --model_dir Qwen/Qwen3-0.6B \
    --output_dir ./quantized/qwen3-0.6b \
    --quantization int4_awq

tensorrt-edgellm-export-llm \
    --model_dir ./quantized/qwen3-0.6b \
    --output_dir ./onnx_models/qwen3-0.6b

./build/examples/llm/llm_build \
    --onnxDir ./onnx_models/qwen3-0.6b \
    --engineDir ./engines/qwen3-0.6b

./build/examples/llm/llm_inference \
    --engineDir ./engines/qwen3-0.6b \
    --inputFile input.json \
    --outputFile output.json

docker exec -it 4f54eee2e87e /bin/bash


wget https://go.dev/dl/go1.25.6.linux-amd64.tar.gz && rm -rf /usr/local/go && tar -C /usr/local -xzf go1.25.6.linux-amd64.tar.gz && rm go1.25.6.linux-amd64.tar.gz
export PATH=\$PATH:/usr/local/go/bin

Here are the commands to compile and run the project from the container.

1. Compile C++ Backend (edge_llm_runtime)
This builds the shared library (libedge_llm_runtime.so) that Go will link against.

```bash
cd /workspace/build
make -j$(nproc) edge_llm_runtime
```

2. Compile Go Application
This builds the Go executable.

```bash
export PATH=$PATH:/usr/local/go/bin

cd /workspace/go-edgellm

export PATH=$PATH:/usr/local/go/bin
go mod tidy
go build -o go-edgellm-app .
```

3. Run Application
You must set LD_LIBRARY_PATH so the system finds the C++ libraries, and EDGELLM_PLUGIN_PATH so TensorRT finds the custom plugins.

```bash
cd /workspace/go-edgellm
# Set environment variables
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/workspace/build/cpp:/workspace/build
export EDGELLM_PLUGIN_PATH=/workspace/build/libNvInfer_edgellm_plugin.so
# Run
./go-edgellm-app
```

c

4f54eee2e87e    nvcr.io/nvidia/tensorrt:25.10-py3                                                                                                      "/opt/nvidia/nvidia_…"    4 minutes ago     Up                 tensorrt



 이걸 이제 serving 관점으로 htttp 요청을 8000 으로 받아서 queue상태에 따라 worker가 꺼내서 dynamic batching 이 가능케 하는것까지 수정하고싶어. 그리고 지금처럼 benchmark 나 테스트가 되는것까지 고려하면 좋을거 같아

sslab@node3:~/nfs/TensorRT-Edge-LLM$ curl "http://localhost:8000/benchmark?requests=20000&concurrency=16&batch_size=4"
Server Batch Size Set to 4
Starting Benchmark: Requests=20000, Concurrency=16, BatchSizeLimit=4

Benchmark Complete:
Total Time: 1m26.918819803s
Total Tokens: 180000
Throughput: 2070.90 tokens/sec
Avg Latency: 68.917196ms
Avg TTFT: 22.580865ms
Avg ITL: 5.149451ms
Avg TPOT: 5.79204ms


## Overview

TensorRT Edge-LLM is NVIDIA's high-performance C++ inference runtime for Large Language Models (LLMs) and Vision-Language Models (VLMs) on embedded platforms. It enables efficient deployment of state-of-the-art language models on resource-constrained devices such as NVIDIA Jetson and NVIDIA DRIVE platforms. TensorRT Edge-LLM provides convenient Python scripts to convert HuggingFace checkpoints to [ONNX](https://onnx.ai). Engine build and end-to-end inference runs entirely on Edge platforms.

---

## Getting Started

For the supported platforms, models and precisions, see the [**Overview**](docs/source/developer_guide/01.1_Overview.md). Get started with TensorRT Edge-LLM in <15 minutes. For complete installation and usage instructions, see the [**Quick Start Guide**](docs/source/developer_guide/01.2_Quick_Start_Guide.md).

---

## Documentation

### Developer Guide

Complete documentation for installation, usage, and deployment:

- **[Overview](docs/source/developer_guide/01.1_Overview.md)** - What is TensorRT Edge-LLM and key features
- **[Quick Start Guide](docs/source/developer_guide/01.2_Quick_Start_Guide.md)** - Get started in ~15 minutes
- **[Installation](docs/source/developer_guide/01.3_Installation.md)** - Detailed installation instructions
- **[Supported Models](docs/source/developer_guide/02_Supported_Models.md)** - Complete model compatibility matrix
- **[Python Export Pipeline](docs/source/developer_guide/03.1_Python_Export_Pipeline.md)** - Model export and quantization
- **[Engine Builder](docs/source/developer_guide/03.2_Engine_Builder.md)** - Building TensorRT engines
- **[C++ Runtime Overview](docs/source/developer_guide/04.1_C++_Runtime_Overview.md)** - Runtime system architecture
  - [LLM Inference Runtime](docs/source/developer_guide/04.2_LLM_Inference_Runtime.md)
  - [LLM SpecDecode Runtime](docs/source/developer_guide/04.3_LLM_Inference_SpecDecode_Runtime.md)
  - [Advanced Runtime Features](docs/source/developer_guide/04.4_Advanced_Runtime_Features.md)
- **[Examples](docs/source/developer_guide/05_Examples.md)** - Working code examples
- **[Chat Template Format](docs/source/developer_guide/06_Chat_Template_Format.md)** - Chat template configuration
- **[TensorRT Plugins](docs/source/developer_guide/07_TensorRT_Plugins.md)** - Introduction for TensorRT plugins.


### Additional Resources

- **[Examples Directory](examples/)** - LLM and VLM inference examples
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

*Coming soon*

Follow our [GitHub repository](https://github.com/NVIDIA/TensorRT-Edge-LLM) for the latest updates, releases, and announcements.

---

## Support

- **Documentation**: [Developer Guide](docs/source/developer_guide/01.1_Overview.md)
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
