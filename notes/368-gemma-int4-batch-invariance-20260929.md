# Gemma Batch-Shape Output Divergence Is the INT4 GEMV/GEMM Path Switch

## Outcome

Note 364 left the output gate open because Gemma serving matched the batch-1 serial reference on only 60% of
requests, while Cosmos was bit-exact. The source is the V1 `Int4GroupwiseGemmPlugin`: it runs a CUDA-core GEMV for
M <= 6 and a tensor-core GEMM above, and both accumulate in FP16 (GEMV `__hfma2` into `half` partial sums; GEMM
`mma.sync...f16.f16.f16.f16`) in different orders. A row's output therefore depends on whether its batch had more
than six rows. Within each path a row is M-invariant: GEMV computes each batch row independently, and the GEMM has
a fixed 64x128x64 tile with no split-K. Cosmos uses FP16 TensorRT GEMMs and never enters this plugin. XQA decode
attention has no multi-block launch (grid is heads x batch), so it is batch-invariant.

With `TRT_EDGELLM_INT4_GEMV_MAX_M=0` (`863a6d4`, every CTA-aligned INT4 GEMM on the tensor-core kernel), serving
matches its batch-1 reference on 610/632 requests (96.5%); text requests 468/472 (99.2%). Neither FP16 path is more
correct than the other, so bit-equality to the default batch-1 reference is not an achievable gate for this INT4
engine; the forced-GEMM mode is a batch-invariant audit mode, not a serving default.

## Campaigns

Gemma full12 x1, binary `.local/baselines/int4-force-gemm-863a6d4-20260929` (and `int4-gemv-knob-62f58ff-20260929`),
serving versus `--client-max-in-flight 1 --ordered-backend-ingress` reference on the same binary and knob.

| Mode | Exact requests vs own batch-1 reference |
|---|---:|
| Default (note 364, 3 runs) | 1137-1140/1896 (60%) |
| `TRT_EDGELLM_INT4_GEMV_MAX_M=24` (GEMV through decode batch 24) | 576/632 (91.1%) |
| `TRT_EDGELLM_INT4_GEMV_MAX_M=0` (GEMM for all aligned M) | 610/632 (96.5%) |

With the GEMV bound at 24, 26 of the 56 remaining divergences were the same switch on the prefill side: late-vision
20-token text prompts (GEMV alone, GEMM when packed) and 784-token long-prefill prompts whose final 16-token chunk
crossed the bound. Forcing GEMM makes decode-heavy, late-vision, and long-prefill 100% exact.

Remaining 22 under forced GEMM:

- 18 vision requests, all early (token 20-30). Two prompts account for 14 of them and diverge deterministically to
  the same token pair in every serving occurrence (567-token prompt: 7280 vs 2543 at step 20; 291-token prompt: 107
  vs 106 at step 22). Candidate: the Gemma vision encoder engine's image-batch shape; not yet tested.
- 4 text requests (bimodal 16/43/60, mixed 38); two share one 108-token prompt. Unexplained.

Forced GEMM costs 4-8% tok/s against the `c7a1b71` full24 medians (balanced -4.7%, poisson -7.2%, short -8.2%);
a GEMV bound of 24 costs 3-5%.

## Vision encoder batch shape (`vision-encoder-batch1-20260929`)

Forced-GEMM serving with `TRT_EDGELLM_VISION_ENCODER_BATCH_SIZE=1`, compared with the same forced-GEMM batch-1
reference on the six vision-bearing workloads: 293/296 (99.0%). vision-heavy 64/64, multi-image 20/20, wave-drain
20/20, text-heavy 64/64 (were 58, 18, 17, 61). The deterministic vision divergences come from the Gemma vision
encoder engine's image-batch shape (encoder batch 4 in serving, per-request in the reference). The remaining three
(bimodal 21 and 41 text, mixed 49 vision) are different requests from the four text divergences of the previous
forced-GEMM run (bimodal 16/43/60, mixed 38), so the last ~1% moves between runs and is not a fixed per-prompt
source.

Attribution of the default-mode 40% divergence: INT4 GEMV/GEMM switch (bulk), vision encoder batch shape (vision
requests), and a run-varying residual of about 1% not yet localized.

## vLLM run-to-run stability (`vllm-determinism-20260929`)

Same vLLM v0.28.0 server contract as the frozen Gemma baseline (CUDA graphs 1-24, async scheduling, seed 0,
temperature 0, ignore EOS), token IDs captured with `return_token_ids`. Twelve traces, serving x2 at 24 in flight and
one batch-1 run (`--max-in-flight 1`):

| Comparison | vLLM | TensorRT Edge-LLM (default) |
|---|---:|---:|
| Serving run 1 vs run 2 | 562/632 (88.9%) | 588/632 (93.0%, note 364) |
| Serving vs own batch-1 | 1032/1264 (81.6%) | 1137/1896 (60%, note 364) |

vLLM is neither run-to-run deterministic nor batch-invariant under serving load (decode-heavy 48/64 run-to-run,
39-41/64 against batch-1). Bit-equality to batch-1 is therefore not an industry-standard serving property; our serving
is more run-to-run stable than vLLM and further from its own batch-1 output because of the INT4 path switch above.

## MMLU accuracy gate (`mmlu-serving-accuracy-20260929`)

Zero-shot MMLU test split (14,031 of 14,042 questions; 11 over the 1024-token engine input limit dropped),
`benchmarks/phase_serving/mmlu_serving_accuracy.py`, 4 output tokens, default numerics (no INT4 override), binary
`int4-force-gemm-863a6d4`, harness `--trace-file short=...`:

| Run | Accuracy |
|---|---:|
| Batch-1 reference (`--client-max-in-flight 1 --ordered-backend-ingress`) | 7065/14031 (50.35%) |
| Serving repeat 1 | 7066/14031 (50.36%) |
| Serving repeat 2 | 7066/14031 (50.36%) |

Predicted letters agree on 14030/14031 (reference vs serving) and 14031/14031 (serving vs serving); exact token
agreement 13980-13987/14031. 1,336 outputs name no option within 4 tokens ("The correct option is", ...) identically
in every run, so the absolute accuracy is understated but the comparison is unaffected. Default serving passes the
accuracy gate: batch-shape numerics do not change task accuracy.

## Gate proposal

1. **Scheduler correctness (bit-exact):** in forced-GEMM mode, serving must match its batch-1 reference except for
   divergences attributed to a named batch-dependent kernel. This isolates scheduler bugs (request mixing, KV
   corruption) from kernel numerics.
2. **Output quality (accuracy):** default serving and batch-1 must score the same on an accuracy benchmark within
   noise, since both FP16 accumulation orders are equally valid.

## Next work

1. Localize the run-varying ~1% residual (other FP16 TensorRT layers such as PLE projection or LM head by M).
2. Vision encoder engine: identify the batch-dependent layer (TensorRT tactic by batch) if batch-invariant vision
   output is required.
4. Upstream note: FP16 accumulation in both INT4 kernels loses precision relative to FP32-accumulating kernels.
