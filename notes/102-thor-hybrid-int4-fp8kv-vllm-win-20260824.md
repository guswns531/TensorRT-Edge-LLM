# Thor Qwen3.8 hybrid phase engine beats vLLM c64

## Result

The optimized independent-phase path beats the fair stock-vLLM c64 baseline on all three core HTTP workloads. Each
number is the median of three warmed streaming runs. Every run completed 128/128 requests with the exact configured
output-token count.

| workload | Phase hybrid | vLLM vanilla | Phase advantage |
| --- | ---: | ---: | ---: |
| short burst, 128->64 | 286.78 tok/s | 235.27 tok/s | +21.89% |
| wave burst, 256->96 | 275.04 tok/s | 256.56 tok/s | +7.20% |
| decode heavy, 128->256 | 341.33 tok/s | 333.19 tok/s | +2.45% |

Phase also wins median latency in every row:

| workload | Phase TTFT / TPOT / E2E | vLLM TTFT / TPOT / E2E |
| --- | --- | --- |
| short | 1,707 / 199 / 14,250 ms | 2,515 / 235 / 17,335 ms |
| wave | 1,839 / 203 / 22,279 ms | 2,606 / 220 / 23,272 ms |
| decode heavy | 1,663 / 181 / 47,949 ms | 2,172 / 184 / 49,103 ms |

Machine-readable results:

- Phase: `data/qwen38/results/next/phase-hybrid-int4-fp8kv-graphs-input256-final/summary.json`
- vLLM: `data/qwen38/results/next/vllm-vanilla-c64/summary.json`

## Winning architecture

The serving process uses two I/O-compatible TensorRT engines with independent contexts and streams:

- prefill: `phase-b64-p8-d64-fp8kv-input256`
  - FP16 GDN input projection, FP8 KV, dense P8, D64, max input 256, KV capacity 4096;
  - the narrowed input profile reduced prefill activation from about 2.03 GiB to 404 MiB;
- decode: `phase-b64-p8-d64-fp8kv-gdn-int4`
  - group-128 weight-only INT4 RTN for the fused GDN input projection, FP8 KV, D64;
  - weights fell from about 20.55 GiB to 14.95 GiB.

The executor pair now accepts a separately deserialized decode engine. Runtime-contract checks reject mismatched hidden
size, vocabulary, layer count, decode batch cap, KV dtype, or recurrent/conv state dtype before serving.

The INT4 projection keeps activations in FP16 and pads its 16,480 logical outputs to 16,512 for the legacy W4A16
kernel, then slices back before the canonical GDN split. A greedy smoke test still produces exactly `THOR READY`.
This is a performance gate, not a substitute for a full task-quality evaluation of the RTN projection.

## Kernel and serving changes

- Export a second GDN small-grid AOT symbol and select it for N>=16. The corrected standalone harness shows small-grid
  wins of 28% at N16 and 25% at N32; the old harness silently ignored `--small_batch` during tests.
- Cache single-token text decoding in the persistent IPC process.
- Permit production CUDA Graph capture after warmup. The final server recorded:
  - prefill: 4 entries, 108 hits, 4 captures, 0 evictions;
  - decode: 5 entries, 3,072 hits, 5 captures, 0 evictions.
- Use a 10 ms prefill queue target for the 128/256-token core matrix.

Production graph capture has a real first-request cost. The graph-populating campaign is a warmup and must not be
included in the three reported repetitions.

## Component gates

| candidate | D64 past-128 graph-off |
| --- | ---: |
| FP16 projection + FP16 KV, small GDN | 209.70 ms |
| FP16 projection + FP8 KV, small GDN | 208.34 ms |
| INT4 RTN projection + FP8 KV, small GDN | 186.98 ms |
| INT4 RTN projection + FP8 KV, CUDA Graph | 183.87 ms |

The INT4 decode engine improves the graph-off step by 10.25%. This raised the D64 component ceiling above the vLLM
decode-heavy service rate; scheduler-only tuning could not do that.

## FP8 KV checkpoint note

The local Inferact NVFP4 checkpoint did not ship KV calibration buffers. The experiment marks its KV scheme as FP8 and
uses the model's explicit default Q/K/V scale of 1.0. The FP8 engine passes the greedy smoke gate. A calibrated FP8-KV
checkpoint and downstream quality suite remain required before treating this as a general accuracy claim.

## Reproduction

Generate an FP8-KV ONNX export, then build the prefill engine with `maxInputLen=256`, P8, and D64. Export the decode
ONNX with `TRT_EDGELLM_GDN_INT4_RTN=1` and `--int4-gemm-plugin-version 1`, then build it with P8/D64. The launcher preset
sets the small GDN crossover, 10 ms prefill target, and production graphs:

```text
IGNORE_EOS=1 PHASE_IMAGE=<final-overlay-image> \
  experiments/qwen38_thor/run_phase_server.sh thor-throughput-core-hybrid
```

Run one complete warmup campaign to populate the graph cache, followed by the three measured repetitions:

```text
python3 experiments/qwen38_thor/run_http_matrix.py \
  --base-url http://127.0.0.1:8001 --backend phase-hybrid-warmup \
  --tokenizer data/qwen38/nvfp4/tokenizer.json --results-dir /tmp/phase-hybrid-warmup \
  --cases short_burst_c64,wave_burst_c64,decode_heavy_c64 --repeats 1 --mode stream

python3 experiments/qwen38_thor/run_http_matrix.py \
  --base-url http://127.0.0.1:8001 --backend phase-hybrid-int4-fp8kv-graphs-input256 \
  --tokenizer data/qwen38/nvfp4/tokenizer.json \
  --results-dir data/qwen38/results/next/phase-hybrid-int4-fp8kv-graphs-input256-final \
  --cases short_burst_c64,wave_burst_c64,decode_heavy_c64 --repeats 3 --mode stream
```

## Rejected candidates

- CUDA Graphs without production capture: almost no hits, therefore no HTTP gain.
- Direct padded-vocabulary sampling: neutral.
- GDN pre-normalization: a small regression after the extra launch and global Q/K traffic.
- Initial decode-cohort waits of 1.2-3.2 seconds: worse wave throughput.
- P16 prefill: halved dispatch count but increased compute time and TTFT.
- B128 stable slots: irrelevant to the fair c64 client, which never has more than 64 live requests.
- GDN V8/V32/V64 and reduced-unroll candidates: slower or numerically weaker than V16/8-CTA.
