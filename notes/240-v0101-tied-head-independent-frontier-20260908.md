# v0.10.1 tied LM-head and independent E/P/D frontier recovery

Date: 2026-09-08

## 1. Objective

The v0.10.1 forward port used 582 MiB more device memory than the retained
v0.10.0 system and consequently fell from the E4/P8/D64 independent frontier
to memory-constrained E1/P8/D32 operation. The same v0.10.1 text engine also
regressed in prefill time. This change addresses the common engine-level cause
without changing KV capacity or ownership.

## 2. KV is not the source of the regression

Both engines use the same canonical paged-KV contract:

```text
layers              28
K/V planes           2
page bundles        256
tokens per page     128
KV heads              8
head dimension      128
dtype               FP16
```

The allocation is therefore identical:

```text
28 * 2 * 256 * 128 * 8 * 128 * 2 bytes
  = 3,758,096,384 bytes
  = 3,584 MiB
```

Stable request-to-page leases, active-row page-table publication, and deferred
release remain unchanged.

## 3. Root cause

The retained v0.10.0 ONNX and the v0.10.1 ONNX both contain 859 nodes and 28
attention plugins. Their phase topology is the same. The material difference
is the LM head:

```text
v0.10.0 retained engine
  LM head: external input [2048, 151936]
  embedding: one 622,329,856-byte FP16 allocation

v0.10.1 regression engine
  LM head: TensorRT-managed constant
  embedding: separate 622,329,856-byte FP16 allocation
```

The duplicate table is 593.5 MiB. It accounts for the observed device-memory
delta after allocator and workspace effects. It also changes TensorRT's GEMM
materialization and tactic contract. The measured profile workspaces were:

| Engine | Prefill workspace | Decode workspace |
|---|---:|---:|
| retained v0.10.0 external head | 301,993,472 B | 21,548,544 B |
| v0.10.1 embedded head | 197,136,384 B | 539,497,472 B |

Thus the extra 493.95 MiB decode workspace and the prefill regression are not
evidence of a KV-layout regression. They are consistent with the embedded
LM-head engine contract. A rebuilt external-input engine is required to prove
the exact workspace and latency recovery.

## 4. Implemented data path

```text
checkpoint tied weight [vocab, hidden]
                 |
                 | export-time transpose
                 v
embedding.safetensors
embedding_transposed [hidden, vocab]
                 |
                 | one GPU allocation
                 v
       +--------- shared FP16 storage ---------+
       |                                        |
       | strided token gather                   | non-owning Tensor alias
       v                                        v
embedding lookup                         TensorRT LM-head input
                                              [hidden, vocab]
```

The ONNX transpose is removed before building. No runtime transpose or extra
activation is introduced. The opt-in export flag is:

```text
--reuse-tied-lm-head
```

It is restricted to tied, scale-1, FP16 embeddings, TP1, vanilla decoding. FP8
embedding, reduced vocabulary, and speculative decoding are rejected.

Implementation locations:

- `tensorrt_edgellm/external_weights.py`: external LM-head input and embedding-source manifest
- `tensorrt_edgellm/onnx/export.py`: tied-layout validation and artifact connection
- `tensorrt_edgellm/checkpoint/checkpoint_utils.py`: transposed embedding sidecar
- `tensorrt_edgellm/scripts/export.py`: public CLI and scope checks
- `cpp/runtime/state/externalWeightManager.*`: non-owning engine-input alias
- `cpp/runtime/llmRuntimeUtils.*`: normal/transposed sidecar loading
- `cpp/kernels/embeddingKernels/embeddingKernels.*`: transposed FP16 gather
- rank runtime, artifact loader, smoke, and bench: load embedding before binding external weights

## 5. Independent workspace selection

The previous v0.10.1 phase runtime always shared the vision workspace with P or
D whenever a vision runner existed. It could therefore suppress one overlap
pair even when all three workspaces fit.

The workspace selector now follows this capability order:

```text
free memory >= E workspace + safety headroom
  -> allocate independent E, P, and D arenas
  -> E+P, E+D, and P+D remain feasible

otherwise, E/P sharing fits
  -> retain the existing profile-aware E/P sharing fallback

otherwise
  -> retain the existing E/D sharing fallback and constrained D capacity
```

This is based only on current device memory and measured TensorRT workspace
requirements. It does not use workload names or policy profiles.

## 6. Validation completed without a GPU

| Gate | Result |
|---|---:|
| external-weight and tied-layout Python tests | 9 passed |
| export-config regression tests | 16 passed |
| `llm_phase_context_smoke` build | passed |
| `llm_bench` build | passed |
| `unitTestKernelsMisc` build | passed |
| `unitTestRuntimeState` build | passed |
| changed-file pre-commit hooks | passed |

The GPU tests are pending because the NVIDIA kernel modules and PCI device are
visible but `/dev/nvidia*` device nodes are absent. The implementation does not
claim a measured recovery until the following gate completes.

## 7. GPU promotion gate

1. Export Cosmos with packed prefill 128 and `--reuse-tied-lm-head`.
2. Build B80/P8/D64/vision-P4/KV2048 with the v0.10.1 builder.
3. Verify the engine file loses one vocabulary table and inspect per-profile
   P/D workspace bytes.
4. Run greedy text and one-/two-image inference before benchmarking.
5. Verify startup selects `independent E/P/D arenas` and exercise E4/P8/D64.
6. Compare isolated P and D kernel-group time against both retained v0.10.0 and
   the v0.10.1 embedded-head engine.
7. Repeat V0/V1/V2 and the 12-workload HTTP gate with identical requests.

Primary success criteria:

- KV allocation remains exactly 3,584 MiB.
- At least 512 MiB of the v0.10.1 memory regression is removed.
- E4/P8/D64 initializes without workspace sharing or OOM.
- Greedy output and semantic VLM gates pass.
- Prefill and decode median/p95 are within 3% of the retained v0.10.0 engine
  before scheduler comparisons are interpreted.
