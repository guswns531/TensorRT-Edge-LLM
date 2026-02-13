# TensorRT Edge-LLM

```
cd ~/Documents/TensorRT-Edge-LLM/
source .edge/bin/activate

tensorrt-edgellm-quantize-llm \
    --model_dir Qwen/Qwen3-VL-2B-Instruct \
    --output_dir /home/sslab/nfs/thor/TensorRT-Edge-LLM/quantized/qwen3-vl-2b \
    --quantization nvfp4

export CUDA_HOME=/usr/local/cuda            # (환경에 맞게)
export TRITON_PTXAS_PATH="${CUDA_HOME}/bin/ptxas"
export PATH="${CUDA_HOME}/bin:$PATH"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:$LD_LIBRARY_PATH"

tensorrt-edgellm-export-llm \
    --model_dir /home/sslab/nfs/thor/TensorRT-Edge-LLM/quantized/qwen3-vl-2b \
    --output_dir /home/sslab/nfs/thor/TensorRT-Edge-LLM/onnx_models/qwen3-vl-2b

tensorrt-edgellm-export-visual \
    --model_dir /home/sslab/nfs/thor/TensorRT-Edge-LLM/quantized/qwen3-vl-2b \
    --output_dir /home/sslab/nfs/thor/TensorRT-Edge-LLM/visual_enc_onnx/qwen3-vl-2b


./build/examples/llm/llm_build \
  --onnxDir /home/sslab/nfs/thor/TensorRT-Edge-LLM/onnx_models/qwen3-vl-2b \
  --engineDir /home/sslab/nfs/thor/TensorRT-Edge-LLM/engines/qwen3-vl-2b \
  --maxBatchSize 4 \
  --maxInputLen=1024 \
  --maxKVCacheCapacity=4096 \
  --vlm \
  --minImageTokens 128 \
  --maxImageTokens 512

./build/examples/multimodal/visual_build \
  --onnxDir /home/sslab/nfs/thor/TensorRT-Edge-LLM/visual_enc_onnx/qwen3-vl-2b \
  --engineDir /home/sslab/nfs/thor/TensorRT-Edge-LLM/visual_engines/qwen3-vl-2b


./build/examples/llm/llm_inference \
  --engineDir engines/qwen3-vl-2b \
  --multimodalEngineDir visual_engines/qwen3-vl-2b \
  --inputFile input_with_images.json \
  --outputFile output.json

```