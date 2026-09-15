# Current versus vLLM Gemma 4 memory attribution

## Scope

This note compares the retained Gemma 4 E2B AWQ serving contracts before changing Current's memory architecture.
It separates directly reported values from derived attribution. It does not treat serialized engine or checkpoint
file sizes as device-resident bytes.

- Current: P8/D24/E4, P128, 96 stable indexed-paged KV pages, independent E/P/D TensorRT contexts, graph replay.
- vLLM 0.28: max sequences 24, 480 MiB FP16 KV, 4096 batched tokens, graph sizes 1/2/4/8/16/24.
- GPU: RTX 3080 10 GiB.

The Current result is the three-repeat full-12 campaign in note 310. The vLLM result is the retained one-repeat
capacity comparator. The different repeat counts limit statistical performance claims but do not affect the
startup allocation records used here.

## Observed process memory

| State or workload | Current MiB | vLLM MiB | Current - vLLM MiB |
|---|---:|---:|---:|
| balanced | 9,451 | 8,559 | +892 |
| decode-heavy | 9,449 | 8,621 | +828 |
| mixed | 9,461 | 8,843 | +618 |
| multi-image | 9,469 | 8,843 | +626 |
| vision-heavy | 9,469 | 8,843 | +626 |
| long-prefill | 9,449 | 8,841 | +608 |

vLLM grows from approximately 8,559 MiB to 8,843 MiB after exercising the multimodal path and then retains that
allocator high-water state. Current allocates its phase execution substrate at startup and stays near 9,445 MiB
before measured work. The stable VLM comparison is therefore approximately 608--626 MiB, while early text-only
cells can show an 828--892 MiB gap because vLLM has not yet reached its multimodal allocator high-water state.

## Direct vLLM attribution

The retained vLLM server log reports:

| Component | Reported memory |
|---|---:|
| Model load | 6.92 GiB |
| KV cache | 0.47 GiB / 480 MiB |
| CUDA graph capture | 0.31 GiB |
| GPU KV capacity | 27,025 tokens |

The remaining process memory includes the CUDA/PyTorch runtime, activation buffers, sampling, multimodal cache,
and allocator reserve. vLLM uses one token-centric model runner rather than permanently retaining three independent
TensorRT profile workspaces. Its server was configured with an eager compilation backend but still captured full
and piecewise CUDA graphs for decode sizes 1/2/4/8/16/24.

## Current attribution

### Immutable weights and PLE

The Current PLE table is 4,480 MiB FP16 and is shared by the prefill and decode preprocessors. Prefill and decode
only duplicate their small output buffers, approximately 17.5 MiB and 0.41 MiB. The sibling TensorRT executors share
one deserialized ICudaEngine and its immutable weights.

The PLE table is therefore a major absolute-memory target, but it does not explain why Current exceeds vLLM: the
vLLM 6.92 GiB model-load value already includes the same Gemma model parameters. Reducing or caching PLE can create
headroom for both systems, but it is not the first causal explanation of the relative gap.

### KV cache

Gemma has 35 attention layers, but `kv_sharing_donors` leaves only layers 0--14 as physical owners. Layers 15--34
reuse donor 13 or 14. With FP16 K/V and 128-token pages:

```text
physical bytes/page = 2.25 MiB
Current pool         = 96 * 2.25 MiB = 216 MiB
Current capacity     = 96 * 128 = 12,288 tokens
```

vLLM reports 480 MiB for 27,025 tokens. Both systems therefore use approximately the same 18 KiB/token physical KV
geometry. Current does not waste more bytes per KV token; it reserves fewer total tokens. Current is approximately
264 MiB smaller in physical KV while exposing only 45% of vLLM's token capacity.

If Current matched vLLM's 27,025-token capacity with the same geometry, it would require approximately 475 MiB,
adding about 259 MiB to its current process peak.

### TensorRT phase workspaces

The retained P128 engine reports:

| Workspace | MiB |
|---|---:|
| P128 | 175.14 |
| D24 | 27.09 |
| E | 299.00 |
| Sum if fully independent | 501.23 |

These workspaces are additional to shared immutable weights. They are what allow independently outstanding E/P/D
TensorRT execution contexts. The external/vision-prefill text context shares the P workspace and is serialized with
text P, so it does not add another 175 MiB allocation.

### CUDA graphs

The graph-disabled Current campaigns peak near 9,373--9,377 MiB; the retained graph-enabled campaign is generally
9,449--9,469 MiB. Current graph residency therefore adds approximately 76--92 MiB in these runs. vLLM explicitly
reports a 0.31 GiB graph-capture allocation. A fresh-process causal run measures a smaller 158 MiB net process
increment because vLLM releases or reuses other allocations while capturing. Current's corresponding fresh net
increment is 76 MiB. CUDA graph storage is not Current's relative excess under either accounting method.

## Fresh-process causal matrix

The direct experiment kept the Current binary, LLM engine, 96-page KV pool, P8/D24 limits, P128 chunk, balanced
HTTP trace, and 49-request text calibration fixed. Only execution-context, vision-engine, and graph residency were
changed. Each row started from a quiescent GPU in a fresh process.

| Current cell | Ready MiB | Peak MiB | Increment | Balanced req/s |
|---|---:|---:|---:|---:|
| shared P/D, no E, graph off | 8,547 | 8,547 | baseline | 12.329 |
| independent P/D, no E, graph off | 8,585 | 8,585 | +38 | 13.941 |
| independent E/P/D, graph off | 9,369 | 9,369 | +784 | 13.496 |
| independent E/P/D, graph on | 9,445 | 9,445 | +76 | 13.602 |

The P/D independence price is only 38 MiB, including the additional 27.09 MiB decode workspace and approximately
11 MiB of context/runtime state. In this diagnostic balanced run it also raises request throughput by 13.1% over
the shared-context cell. This is not a repeated performance result, but it is strong evidence against recovering
memory by merging P and D.

The dominant Current increment is the 784 MiB vision execution substrate. It includes the separately loaded vision
engine weights, 299 MiB E workspace, E context state, vision I/O, and preparation buffers. The entire increment is
resident before a vision request executes. Current's first full multi-image trace adds only 6 MiB with graphs off;
the retained graph-on multi-image campaign grows by 24 MiB from ready to peak.

The matching fresh vLLM processes fixed the model, 480 MiB KV pool, sequence limit 24, batched-token limit 4096,
and all server settings. Only `--enforce-eager` versus graph sizes 1/2/4/8/16/24 changed.

| vLLM cell | Ready MiB | First text peak MiB | First vision peak MiB | Increment |
|---|---:|---:|---:|---:|
| enforce-eager | 8,239 | 8,239 | 8,239 | baseline |
| graph enabled | 8,397 | 8,397 | 8,399 | +158 ready, +2 first vision |

vLLM integrates the vision module in the loaded model and reuses its common allocator, so a single image request
does not require another persistent engine/context allocation. Larger retained VLM workloads eventually raise its
allocator high-water mark from 8,397 to approximately 8,843 MiB, a 446 MiB dynamic increment. Current instead pays
most of the VLM execution cost at startup and has a much smaller request-time increment.

The fresh graph-on ready gap closes exactly into four measured or derived categories:

```text
Current independent E/P/D graph-on       9,445 MiB
vLLM graph-on                            8,397 MiB
observed gap                            +1,048 MiB

Current KV advantage                    -264 MiB
Current graph net-residency advantage    -82 MiB
Current base text substrate excess      +610 MiB
Current vision E substrate              +784 MiB
------------------------------------------------
accounted gap                          +1,048 MiB
```

The 610 MiB base-text excess is obtained from the independent-P/D graph-off gap after correcting for KV:
`8,585 - 8,239 + 264 = 610 MiB`. Only 38 MiB of it is caused by P/D independence. The remaining approximately
572 MiB exists even in shared-P/D mode and belongs to common TensorRT/PLE/model representation, base context,
phase I/O, sampling, and allocator state.

## Relative-gap accounting

Use the stable mixed VLM point after the fresh-process matrix:

```text
fresh graph-on ready gap                  +1,048 MiB
Current mixed request-time growth            +16 MiB
vLLM mixed allocator high-water growth      -446 MiB
---------------------------------------------------
observed warmed mixed peak gap               +618 MiB
```

The fresh 1,048 MiB gap consists of:

- -264 MiB from Current's smaller KV pool;
- -82 MiB from Current's smaller net graph residency;
- +610 MiB from Current's base text execution/model substrate;
- +784 MiB from Current's separately resident vision E substrate.

The evidence nevertheless rules out three incorrect explanations:

1. Current does not duplicate the 6.92 GiB model once per phase context.
2. Current KV is not larger; it is 264 MiB smaller but has much less token capacity.
3. Current CUDA graph cache is not larger; its fresh net increment is 82 MiB smaller.

The Current-specific target is therefore the independently materialized execution substrate, especially context
workspaces and phase-local maximum-shape I/O.

## Architectural comparison

```text
vLLM
  one model runner / allocator
    + 6.92 GiB model
    + 480 MiB paged KV
    + 317 MiB CUDA graphs
    + dynamically reused activation/MM allocator high-water

Current
  one shared ICudaEngine/weight set
    + 216 MiB stable paged KV
    + approximately 76 MiB CUDA graphs
    + P context and 175 MiB workspace
    + D context and 27 MiB workspace
    + E context and 299 MiB workspace
    + separate phase-local maximum-shape I/O and runtime state
```

vLLM obtains memory reuse by executing one token-centric iteration through a common allocator. Current intentionally
pays stable workspace residency to make E/P/D independently enqueueable. The memory question is therefore not
whether to merge the CUDA context; both already use one CUDA context. It is how much of the TensorRT workspace and
phase I/O can be safely aliased without removing the profitable P+D and E+D execution frontier.

## Remaining fine-grained attribution

The fresh-process matrix resolves the major architectural deltas. Fine-grained ownership counters are still useful
for splitting the remaining 572 MiB base-text excess. The next instrumentation should record `cudaMemGetInfo` plus
the sum of owned Tensor capacities at these boundaries:

1. CUDA runtime initialized;
2. shared engine and external weights loaded;
3. PLE and embedding loaded;
4. KV allocated;
5. shared P/D execution context;
6. independent P/D contexts;
7. vision engine and E context;
8. phase-local I/O and preprocessors;
9. each graph capture;
10. first text and first vision request.

The vLLM eager/graph/first-request cells and the primary Current shared-P/D/independent-P/D/independent-E/P/D cells
are now complete. The remaining counters should therefore target base-text TensorRT/PLE allocations rather than
repeat the already-resolved architectural matrix.

## Current conclusion

For equal configured process contracts, Current is approximately 0.61 GiB larger after heavy VLM allocator warmup
and 1.02 GiB larger at fresh graph-on readiness, despite holding 0.26 GiB less KV and 82 MiB less net graph
residency. The largest directly observed Current-specific increment is the 784 MiB separately resident vision E
substrate. Independent P/D costs only 38 MiB and preserves a valuable overlap frontier, so merging P/D is the wrong
first optimization.

The first architectural target is therefore lazy/tiered E residency and stronger E/P or E/D arena aliasing when an
overlap action is not outstanding. The second target is the approximately 572 MiB common base-text excess, using
owned-buffer accounting before changing representation. PLE remains the largest absolute allocation, but KV
resizing would be actively misleading: matching vLLM's KV token capacity would add roughly 259 MiB without fixing
either execution-substrate excess.

The first tiered E/P implementation and its non-promotable initial screen are documented in
[note 313](313-tiered-vision-context-memory-20260915.md). It recovers 70 MiB with an E3/E4 arena while preserving
small-E/P overlap, but repeated performance validation under the retained lifetime-admission contract remains
required.
