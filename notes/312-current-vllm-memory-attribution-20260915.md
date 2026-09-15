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
reports 0.31 GiB, approximately 317 MiB. CUDA graph storage is not Current's relative excess; Current retains about
225--241 MiB less graph memory under these contracts.

## Relative-gap accounting

Use the stable mixed VLM point as an example:

```text
observed Current - vLLM process peak       +618 MiB
Current - vLLM physical KV                 -264 MiB
Current - vLLM graph storage, approximate  -241 MiB
---------------------------------------------------
unattributed execution/model/runtime delta +1,123 MiB
```

The 1,123 MiB residual is not all TensorRT context memory. It contains:

- 501 MiB of known independent E/P/D context workspaces;
- phase-local PipelineIO, PLE output, logits, sampling and staging buffers;
- TensorRT versus PyTorch model-weight representation differences;
- TensorRT context/tactic state and CUDA auxiliary streams;
- allocator reserve and fragmentation differences;
- vLLM's retained multimodal allocator/cache high-water state.

The evidence nevertheless rules out three incorrect explanations:

1. Current does not duplicate the 6.92 GiB model once per phase context.
2. Current KV is not larger; it is 264 MiB smaller but has much less token capacity.
3. Current CUDA graph cache is not larger; it is approximately 0.23 GiB smaller.

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

## Required causal measurement before optimization

Peak process memory alone cannot split the remaining 1.1 GiB residual. The next experiment should use the same
Current engine/model/KV and record `cudaMemGetInfo` plus the sum of owned Tensor capacities at these boundaries:

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

The matching vLLM cells should be fresh processes with fixed 480 MiB KV:

1. eager/no-graph ready;
2. graph-enabled ready;
3. first text request;
4. first vision request and post-drain allocator state.

The primary Current A/B is shared-context versus independent P/D versus independent E/P/D. This directly prices
the execution freedom rather than inferring it from different engines or KV budgets. Only after this attribution
should workspace aliasing, graph trimming, PLE caching, or KV resizing be implemented.

## Current conclusion

For equal configured process contracts, Current is approximately 0.61 GiB larger in warmed VLM serving despite
holding 0.26 GiB less KV and approximately 0.23 GiB less CUDA graph memory. The known 0.49 GiB of independent phase
workspace and additional phase-local buffers make the execution substrate the first optimization target.

PLE remains the largest absolute allocation, but optimizing it first would reduce total memory without explaining
or isolating Current's relative inefficiency. KV resizing would be actively misleading: matching vLLM's KV token
capacity would increase Current's relative memory gap to roughly 0.87 GiB unless execution-substrate memory is
recovered first.
