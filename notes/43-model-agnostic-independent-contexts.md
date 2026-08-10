# Model-agnostic independent TensorRT phase contexts

## What changed

`IndependentEngineExecutorPair` is a model-agnostic resource wrapper in
`cpp/runtime/scheduling/independentEngineExecutorPair.{h,cpp}`. A caller creates
the first `EngineExecutor` with its model-specific registry, then passes that
executor to the pair. The pair:

1. creates a sibling executor over the same deserialized `ICudaEngine`,
2. verifies that setup, prefill, and decode streams belong to one CUDA context,
3. verifies distinct TensorRT `IExecutionContext` objects, and
4. allocates and assigns separate profile-sized USER_MANAGED workspaces.

Weights and the serialized engine are shared; execution context state,
auxiliary streams, CUDA graph caches, and workspaces are not shared. Therefore
prefill/decode enqueue calls can overlap without relying on a model name or a
model-specific branch.

## Use

```cpp
auto executor = EngineExecutor::createForLLM(enginePath, config);
IndependentEngineExecutorPairConfig pairConfig;
pairConfig.prefillProfile = 0;
pairConfig.decodeProfile = 1;
pairConfig.setupStream = setupStream;
pairConfig.prefillStream = prefillStream;
pairConfig.decodeStream = decodeStream;
auto pair = IndependentEngineExecutorPair::create(std::move(executor), pairConfig);
```

`llm_phase_bench --trtContextMode independent` now uses this path. The shared
mode remains available for an apples-to-apples serialized-context comparison.
The phase scheduler and TensorRT executor only consume tensor maps, profiles,
and callbacks, so Llama/Gemma/Qwen/Cosmos-style text engines can use the same
pair. A multimodal encoder still needs its own `PhaseEncoderDispatchWorker`
resource identity and adapter to map encoder output into the decoder input;
that data path is intentionally outside this generic context primitive.

## Resource lifetime

The pair must outlive all phase callbacks and TensorRT enqueues. Destroy it
only after both phase streams have completed. The pair owns both workspace
tensors, so callers must not replace or free those buffers independently.

## RTX 3080 smoke

Using the existing Gemma indexed engine (`maxPrefillBatchSize=8`,
`maxDecodeBatchSize=32`) in the TensorRT 26.06 container, the new benchmark
logged two distinct TensorRT context addresses on one CUDA context. An earlier
`warmup=0`, one-iteration result was cold-start biased because sequential was
always measured first. After the benchmark primes both modes, the same
`prefillBatch=1`, `decodeBatch=4`, `slotCount=8` configuration measured:

- independent: sequential 24.3313 ms, concurrent 20.5418 ms, speedup 1.1845x
- phase timings: sequential prefill/decode 18.1821/6.1460 ms;
  concurrent prefill/decode 20.4892/9.3450 ms

The same-run shared-context comparison remains available, but must also be
interpreted only after the priming step.

`slotCount=32` exceeded the 10 GB RTX 3080 budget while allocating the KV/cache
resources alongside two TensorRT context workspaces; this is an admission
capacity limit, not a context-identity failure. The scheduler should derive a
model/device-specific slot limit from the memory budget before admitting a
larger workload.
