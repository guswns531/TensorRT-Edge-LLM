# 321. Strategy D: PipelineIO buffer dynamic pooling and non-owning reshape support

Date: 2026-09-20. Branch: `codex/v0101-phase-forward-port`.

## 1. Outcome

Following Note 320, this change implements **Strategy D (PipelineIO Dynamic Pooling)** from the memory architecture
roadmap (`tensorrt-memory-architecture-analysis.md`).

**Key accomplishments**:
1. **`PipelineIOPool` unified buffer allocation**:
   - Instead of allocating separate maximum-shape GPU and CPU buffers for `mPrefillIO` and `mDecodeIO` (~50–100 MiB total),
     `PipelineIOPool` allocates unified backing buffers sized to the maximum capacity across prefill and decode
     (`max(prefillBatch, decodeBatch)` and `max(prefillSeq, decodeSeq)`).
   - `mPrefillIO` and `mDecodeIO` obtain non-owning phase views into this shared pool, eliminating memory duplication
     for `inputsEmbeds`, `outputLogits`, `deepstackEmbeds`, `mropeCosSin`, and token selection metadata.
2. **`Tensor` non-owning reshape capability**:
   - Added `Tensor(void* data, Coords const& extent, int64_t capacity, ...)` and `setAllowReshape(bool)`.
   - Updated `Tensor::reshape` to permit non-owning view reshaping within pre-allocated capacity when explicitly allowed,
     preserving existing safety invariants for unmanaged non-owning tensors.
3. **Unit test verification**:
   - Added `TensorTest.DeviceTensorNonOwnMemoryAllowReshape` in `rtTensorTest.cpp`.
   - Added `PipelineIOPoolTest.SharedViewsAndReshape` in `phaseKVActiveViewTest.cpp`.
   - All 161 unit tests pass (`unitTestCommon`, `unitTestRuntime`).
4. **Live serving validation**:
   - Verified on Cosmos `short` workload with `pooled_io` variant: 100% deterministic token agreement
     (SHA256: `c8d9092714caa68b2bcd518939da9a9247a9f45432742d556caabdcfdeebc163`), with peak memory reduced
     from 9,301 MiB to 9,293 MiB.

## 2. Architectural design

```
+-----------------------------------------------------------------------------+
|                               PipelineIOPool                                |
|  Backing GPU Buffers (capacity = max(prefill, decode)):                     |
|  - mInputsEmbeds:       [maxBatch, maxSeq, hiddenSize]                      |
|  - mOutputLogits:       [maxBatch, vocabSize]                               |
|  - mDeepstackEmbeds:    N x [maxBatch, maxSeq, hiddenSize]                  |
|  - mMRopeCosSin:        [maxBatch, maxKVCacheCapacity, rotaryDim]           |
|  - Metadata Tensors:    selectTokenIndices, phaseIsEncoder, contextLengths  |
+-----------------------------------------------------------------------------+
               |                                             |
               v (createPrefillView)                         v (createDecodeView)
+-------------------------------+             +-------------------------------+
|          mPrefillIO           |             |           mDecodeIO           |
| Non-owning view into Pool:    |             | Non-owning view into Pool:    |
| - shape: [P8, seqCap, H]      |             | - shape: [D64, 1, H]          |
| - allowReshape: true          |             | - allowReshape: true          |
+-------------------------------+             +-------------------------------+
```

## 3. Retained artifacts

- Validation results: `.local/results/pipeline-io-pooling-screen-20260920/cosmos/pooled_io/repeat-001/short/aggregate.json`
- Tests: `unittests/cpp/common/rtTensorTest.cpp`, `unittests/cpp/runtime/scheduling/phaseKVActiveViewTest.cpp`
