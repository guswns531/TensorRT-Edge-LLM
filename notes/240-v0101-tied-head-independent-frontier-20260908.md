# v0.10.1 tied LM-head and independent E/P/D frontier recovery

Date: 2026-09-08

## 1. Outcome

The v0.10.1 forward port now retains the v0.10.0 phase-workspace contract while
keeping the v0.10.1 runtime and plugin behavior:

| Profile | Retained v0.10.0 | v0.10.1 before | v0.10.1 fixed |
|---|---:|---:|---:|
| text prefill | 79,695,360 B | 197,136,384 B | 79,695,360 B |
| decode | 21,548,544 B | 539,497,472 B | 21,548,544 B |
| vision prefill | 301,993,472 B | 352,325,632 B | 301,993,472 B |

The text-P and vision-P/D arenas are byte-identical to the retained v0.10.0
engine. The runtime can therefore allocate independent E, P, and D context
arenas again instead of serializing E with D under memory pressure.

Two independent regressions had accumulated during the port:

1. the tied LM head was embedded separately from the embedding table, adding a
   second 593.5 MiB vocabulary matrix and changing the text GEMM contract;
2. packed attention lost its early workspace return and reserved a dense split
   K/V carrier that native-paged prefill and XQA decode never read.

The tied-head data path was fixed in commit `ce149c9`. The attention-workspace
fix is the source change documented here. Together they restore the memory and
text-P frontier. Neither issue changes the paged-KV pool.

## 2. KV is not the source of the regression

Both versions use the same canonical paged-KV contract:

```text
layers              28
K/V planes           2
page bundles        256
tokens per page     128
KV heads              8
head dimension      128
dtype               FP16
```

The allocation is identical:

```text
28 * 2 * 256 * 128 * 8 * 128 * 2 bytes
  = 3,758,096,384 bytes
  = 3,584 MiB
```

Stable request-to-page leases, active-row page-table publication, deferred
release, and the 256-page admission limit are unchanged. The regressed bytes
were a duplicated model weight and an activation workspace, not persistent KV.

## 3. Root cause of the 512 MiB decode workspace

The retained v0.10.0 `getAttentionWorkspaceSize()` returned after allocating
packed prefill's dense Q/output boundary scratch. During the v0.10.1 forward
port that return was lost, so every packed-attention profile also reserved:

```text
[batch, 2, numKVHeads, kvCacheCapacity, headSize] FP16
```

For D64 this is exactly:

```text
64 * 2 * 8 * 2048 * 128 * 2 bytes
  = 536,870,912 bytes
  = 512 MiB
```

This allocation resembles KV by shape, but it is not the persistent KV cache.
It is a temporary dense split-K/V workspace. Decode uses paged XQA, and packed
FP16 prefill uses the native-paged path, so neither consumes this carrier.

The repaired workspace decision is:

```text
packed attention
  |-- reserve sequence metadata
  |-- reserve dense Q boundary [1, S, Hq, D]
  |-- reserve dense output boundary [1, S, Hq, D]
  `-- return

dense non-packed attention
  |-- reserve split K/V carrier [B, 2, Hkv, capacity, D]
  |-- reserve Q/K/V scratch
  `-- continue
```

The compact packed carrier has physical batch one. Keeping this distinction in
the plugin workspace estimator avoids materializing the paged pool as a dense
per-row temporary tensor.

## 4. Tied LM-head data path

The retained v0.10.0 ONNX and the v0.10.1 ONNX have the same decoder topology,
but the initial v0.10.1 engine independently materialized the tied LM head and
embedding table. The fixed opt-in path is:

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

The ONNX transpose is removed before building. There is no runtime transpose or
second owning allocation. `--reuse-tied-lm-head` is restricted to tied, scale-1,
FP16 embeddings, TP1, and vanilla decoding.

Implementation locations include:

- `tensorrt_edgellm/external_weights.py`
- `tensorrt_edgellm/onnx/export.py`
- `tensorrt_edgellm/checkpoint/checkpoint_utils.py`
- `cpp/runtime/state/externalWeightManager.*`
- `cpp/runtime/llmRuntimeUtils.*`
- `cpp/kernels/embeddingKernels/embeddingKernels.*`
- `cpp/plugins/attentionPlugin/attentionPlugin.cpp`

## 5. Independent workspace selection

The runtime chooses arenas by current capability rather than workload name:

```text
free memory >= E workspace + safety headroom
  -> independent E, P, and D arenas
  -> E+P, E+D, and P+D remain feasible

otherwise, E/P sharing fits
  -> profile-aware E/P sharing fallback

otherwise
  -> E/D sharing fallback and constrained decode capacity
```

The workspace correction reduces the independent P+D arena requirement by
568,281,088 bytes: 50,332,160 bytes from the maximum P/vision-P arena and
517,948,928 bytes from D. The exact engine file can still vary in size because
TensorRT tactic selection and plan serialization are build-local, so observed
process peak should be interpreted separately from this invariant arena delta.

## 6. GPU validation

### 6.1 Build and reload

The fixed B80/P8/D64/vision-P4/KV2048 engine reports:

```text
text P workspace       79,695,360 B
D workspace            21,548,544 B
vision P workspace    301,993,472 B
```

Reloading the retained default engine selects a 301,993,472-byte P/vision arena
and a 21,548,544-byte D arena. The stable paged-KV smoke passes.

### 6.2 Controlled text execution

The saved engine produced:

| Metric | Result |
|---|---:|
| isolated packed P | 10.3044 ms |
| isolated D | 5.9882 ms |
| sequential P+D | 16.2926 ms |
| overlapped P+D | 13.6704 ms |
| speedup | 1.192x |

An earlier explicit D64 sweep with the same fixed workspace implementation
measured 18.6269 ms sequential and 15.1731 ms overlap, or 1.228x. Start-skew
results were 1.048x at 25%, 1.102x at 50%, 1.158x at 75%, and 1.223x at 100%.
The saved-engine smoke uses its built-in controlled decode shape, while the D64
sweep exercises the maximum decode profile.

The tied-head fix also recovers text prefill from approximately 12.21 ms in the
untied v0.10.1 engine to 10.30 ms, matching the retained v0.10.0 result of about
10.35 ms.

### 6.3 Correctness

Greedy text outputs are exact:

```text
Use asynchronous computation or offloading inference tasks
Dynamic batching improves GPU utilization by intelligently
1. **Kernel Timing Analysis**: Profile
```

One-image VLM inference with fully independent E/P/D arenas also passed. The
saved default engine produced `A woman sits on the sandy beach,` for the
woman-and-dog semantic smoke; an earlier bar-chart smoke produced `The bar
chart compares the performance scores of`.

### 6.4 Device-memory result

For the identical one-image request:

| Engine | Observed peak |
|---|---:|
| old high-workspace engine | 9,596 MiB |
| fixed default engine | 9,280 MiB |
| observed reduction | 316 MiB |

The corrected runtime retains approximately 960 MiB of headroom on the 10 GiB
device, exceeding the 512 MiB promotion requirement. The observed reduction is
smaller than the 568 MiB arena delta because this fresh TensorRT build serialized
a roughly 224 MiB larger engine plan than the previous saved engine. Workspace
bytes are nevertheless byte-identical to v0.10.0 and remain the reproducible
cross-build invariant.

## 7. Validation summary

| Gate | Result |
|---|---:|
| external-weight and tied-layout Python tests | 9 passed |
| export-config regression tests | 16 passed |
| plugin, `llm_build`, and phase smoke build | passed |
| B80/P8/D64/vision-P4 engine build | passed |
| saved-engine reload and stable paged-KV smoke | passed |
| independent P/D overlap | passed, 1.192x built-in / 1.228x D64 |
| greedy text identity | passed |
| one-image semantic VLM | passed |
| one-image peak headroom | passed, approximately 960 MiB |

The engine retained at the default experiment path is:

```text
size    3,081,676,788 bytes
SHA256  50493717b98f2c3b7dcbac4bb41b5bc395bd70ecfa5f9c5f577bdd20de49dad1
```

## 8. Remaining evaluation

The engine/runtime regression is resolved. The next experiment is a same-engine
V0/V1/V2 12-workload replay to isolate scheduler policy from this repaired
mechanism. Frozen vLLM results remain reusable because the requests, model,
precision, memory limit, and vLLM configuration have not changed. Those policy
results should not be mixed into the engine promotion claim above.
