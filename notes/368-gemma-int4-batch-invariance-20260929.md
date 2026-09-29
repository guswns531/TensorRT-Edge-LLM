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

## Gate proposal

1. **Scheduler correctness (bit-exact):** in forced-GEMM mode, serving must match its batch-1 reference except for
   divergences attributed to a named batch-dependent kernel. This isolates scheduler bugs (request mixing, KV
   corruption) from kernel numerics.
2. **Output quality (accuracy):** default serving and batch-1 must score the same on an accuracy benchmark within
   noise, since both FP16 accumulation orders are equally valid.

## Next work

1. Vision encoder batch-shape test for the two deterministic vision divergences.
2. Localize the 4 text divergences (other FP16 TensorRT layers such as PLE projection or LM head by M).
3. Measure vLLM run-to-run token-ID stability under the same traces to calibrate what a production gate can require.
4. Upstream note: FP16 accumulation in both INT4 kernels loses precision relative to FP32-accumulating kernels.
