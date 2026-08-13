SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

# Cosmos packed-prefill phase I/O memory reduction

## Result

Packed prefill has one token carrier whose sequence dimension contains all active rows. The phase harness previously
allocated token-shaped tensors as `[max prefill batch, max input length, hidden]`, even though TensorRT receives
`[1, total packed tokens, hidden]`. `PipelineIO::createForPackedPrefill()` now keeps logical-row metadata at P8 while
allocating input embeddings, deepstack embeddings, and output hidden states for the single token carrier.

Cosmos-Reason2-2B FP16 indexed-paged, P8/D64, 256 page bundles, independent contexts, and the 288-request balanced
trace produced:

| metric | previous phase I/O | packed phase I/O | change |
| --- | ---: | ---: | ---: |
| CUDA used before serving | 9,086.9 MiB | 8,946.9 MiB | **-140.0 MiB** |
| CUDA used after serving/graph priming | 9,312.9 MiB | 9,176.9 MiB | **-136.0 MiB** |
| generated throughput | 4,630.4 token/s | 4,645.2 token/s | +0.32% |
| E2E p95 | 4,831.1 ms | 4,814.2 ms | -0.35% |

All 288 outputs matched the previous run by request id, generated-token count, finish reason, and text. The observed
prefill deepstack allocation changed from three `[8, 1024, 2048]` tensors to three `[1, 1024, 2048]` tensors.

Independent TensorRT contexts already use profile-local workspace: 84.0 MiB for prefill and 513.3 MiB for decode,
instead of allocating the 513.3 MiB all-profile maximum twice. This change removes the remaining phase-I/O shape
over-allocation without changing the engine, KV page pool, weights, or graph budgets.

## Code and validation

- `cpp/runtime/state/pipelineIO.{h,cpp}`: packed-prefill factory with separate logical and token dimensions.
- `examples/llm/llm_phase_bench.cpp`: selects the packed factory for packed engines.
- `unittests/pipelineIOTest.cpp`: locks the token-carrier and logical-row shape contract.
- GPU E2E artifact: `.local/cosmos-reason2-2b/packed-phase-io-memory-20260813/balanced-r1/`.

