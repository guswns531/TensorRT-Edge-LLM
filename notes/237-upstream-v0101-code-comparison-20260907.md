# Upstream v0.10.1 source installation and code comparison

Date: 2026-09-07

## 1. Scope and conclusion

The official NVIDIA `v0.10.1` tag was installed as a detached, read-only
comparison worktree at:

```text
.local/upstream-v0101
```

The checkout and its submodules are complete. It is not a Python virtual
environment and no TensorRT engine was built from it in this comparison. The
purpose of this installation is source-level baseline inspection and a future
forward-port target.

The current research branch and `v0.10.1` have the exact `v0.10.0` release as
their merge base. This makes the comparison unusually clean:

```text
                         v0.10.1  e8b2952
                        /
v0.10.0  71dd1ba ------+
                        \
                         Current  c03881b
```

The most important result is that upstream independently moved the runtime to
a canonical paged KV pool in `v0.10.1`. This aligns the physical KV substrate
with Current, but it does not replace Current's stable request-to-page
ownership, independent E/P/D contexts, phase queues, activity telemetry, or
online global scheduling. Upstream also made a large runtime/server refactor,
so a blind merge is inappropriate. A staged semantic forward-port is required.

## 2. Installed identities

| Item | Identity | Date |
|---|---|---|
| Official base | `v0.10.0`, `71dd1bae032e70771265917ec74d3ff4cad07a10` | 2026-08-12 |
| Official comparison | `v0.10.1`, `e8b29522938901f6df19ebeedd4b69bc8edbcd97` | 2026-09-03 |
| Current research branch | `codex/v010-phase-forward-port`, `c03881b6d7f2db773c9a6c7ed1654150a7c3b49c` | 2026-09-07 |

Initialized submodules in `.local/upstream-v0101`:

| Submodule | Commit |
|---|---|
| `3rdParty/NVTX` | `2fb879e512ed208f83c6aa4d4c96b958789edf49` |
| `3rdParty/googletest` | `f132c893119698e10daef8525d0ad7a3f05176f2` |
| `3rdParty/nlohmannJson` | `55f93686c01528224f448c19128836e7df245f72` |

The source checkout occupies about 63 MiB. Both the checkout and the release
delta pass `git diff --check`.

## 3. Quantitative comparison

### 3.1 Official v0.10.0 to v0.10.1

```text
659 files changed
61,126 insertions
21,391 deletions
```

The change is a substantial release rather than a small patch. The largest
areas by changed-file count are:

| Area | Changed files |
|---|---:|
| C++ unit tests | 118 |
| C++ runtime | 92 |
| Python tests | 67 |
| CUDA/C++ kernels | 55 |
| Python frontend | 48 |
| C++ multimodal | 40 |
| Experimental server | 33 |
| Kernel sources | 32 |
| Experimental builder | 32 |
| Documentation | 29 |

### 3.2 Current relative to the common v0.10.0 base

```text
197 Current-only commits versus 3 v0.10.1-side commits
274 files changed
71,267 insertions
404 deletions
```

The insertion count includes committed notes and benchmark results. Excluding
notes and phase-serving result/manifests, the Current delta is:

```text
183 files changed
47,564 insertions
404 deletions
```

### 3.3 Direct overlap and expected merge conflicts

Of 659 files changed by `v0.10.1`, 56 are also changed by Current. A three-way
merge preview finds 18 textual conflict files:

```text
CMakeLists.txt
cpp/builder/llmBuilder.h
cpp/kernels/posEncoding/applyRopeWriteKV.cu
cpp/kernels/posEncoding/applyRopeWriteKV.h
cpp/plugins/attentionPlugin/attentionPlugin.cpp
cpp/runtime/exec/engineExecutor.cpp
cpp/runtime/exec/engineExecutor.h
cpp/runtime/exec/registryBuilder.cpp
cpp/runtime/kvCacheManager.cpp
cpp/runtime/kvCacheManager.h
cpp/runtime/llmInferenceRuntime.cpp
cpp/runtime/llmInferenceRuntime.h
cpp/runtime/state/externalWeightManager.cpp
cpp/runtime/state/externalWeightManager.h
examples/llm/CMakeLists.txt
examples/llm/llm_build.cpp
tensorrt_edgellm/scripts/export.py
tests/python-unittests/test_external_weights.py
```

This list understates semantic migration work because upstream also moved most
multimodal runner files into model-specific subdirectories.

## 4. Official v0.10.1 changes relevant to this project

### 4.1 KV cache: strong architectural convergence

`v0.10.1` changes the attention cache from an active-batch-shaped linear cache
to a canonical per-layer pool:

```text
[2, numPages, 128, Hkv, D]
```

The page table now addresses K and V planes explicitly, RoPE writes directly
through that table, and the old single-layer tensor/cache copy helpers were
removed. Upstream also accepts a pool larger than the minimum active working
set so pages can survive across requests.

This is aligned with Current's indexed-paged kernel path. The remaining
difference is ownership:

```text
Upstream v0.10.1
  paged physical pool
  + active request page tables
  + context-cache records for cross-request reuse

Current
  paged physical pool
  + StableKVPageManager lease per logical request
  + active row -> stable lease rebinding
  + generation checks, deterministic reuse, prefix page sharing
  + phase scheduler memory/lifetime visibility
```

Therefore Current should adopt the upstream pool ABI and kernels, then retain a
thin stable-ownership layer above upstream's page table. Reintroducing the old
linear compatibility layout on `v0.10.1` would move backward.

### 4.2 Runtime: facade, coordinator, and rank-local execution

Upstream split the former large runtime into:

```text
LLMInferenceRuntime        public facade, 246 lines
  -> RuntimeCoordinator    SD/MD and rank coordination
       -> LLMRankRuntime   rank-local execution, 3,250 lines
```

Current still carries the phase integration around a 2,592-line
`LLMInferenceRuntime.cpp`. The Current phase code itself is already separated
under `cpp/runtime/phase/` and `cpp/runtime/scheduling/`, but its attachment
point must move from the old monolith to the rank-local runtime or to a sibling
serving coordinator.

The preferred boundary is:

```text
public LLMInferenceRuntime / RuntimeCoordinator
  -> one rank-local phase-serving runtime
       -> IndependentPhaseAsyncServer
       -> IndependentPhaseCoordinator
       -> E/P/D executors and ownership
```

This preserves upstream multi-rank routing while keeping phase scheduling
local to each GPU/rank.

### 4.3 EngineExecutor: useful abstraction, missing Current capabilities

Upstream converted `EngineExecutor` into an interface backed by an internal
`TrtEngineExecutor`. That is a good base for testing and alternative executors.
However Current's independent-context mechanism additionally relies on:

- creating sibling TensorRT execution contexts from one engine;
- querying and assigning context memory per optimization profile;
- checking execution-context identity;
- CUDA graph cache counters and bounded cache management.

These capabilities do not exist in the `v0.10.1` public executor interface.
They must be added deliberately to the interface/implementation, rather than
restoring Current's old concrete class wholesale.

### 4.4 Context and media reuse

Upstream adds speculative-decoding context reuse and an LRU GPU
`EncoderEmbeddingCache` keyed by media hashes. Current has stable KV/prefix
leases and explicit vision ownership tied to request DAG transitions.

The two approaches are complementary:

- reuse lookup, record formats, and media hashing can come from upstream;
- lease release must remain gated by Current's GPU completion and ownership
  invariants;
- cached encoder embeddings must become P-queue inputs, not a synchronous
  shortcut that reconnects E and P execution.

### 4.5 Multimodal layout and VLM optimization

Upstream moved runners from a flat `cpp/multimodal/` directory into
`common/`, `gemma4/`, `qwen2/`, `qwen3/`, and other model subdirectories. It
also adds DART visual-token pruning and end-to-end FP8 ViT attention.

Current changes to `multimodalRunner`, Qwen, and Qwen3-VL adapters must be
replayed at their new paths. DART changes the number of tokens entering P, so
the phase scheduler must consume the post-pruning token count when forming P
candidates and estimating E-to-P critical paths.

### 4.6 Experimental server

The upstream server was split into `api/`, `parsing/`, `media/`, and `runtime/`
packages. It now has a bounded asynchronous request queue, engine-bundle cache,
and cleaner cancellation/streaming boundaries.

This is valuable for Current's HTTP surface and engine preparation, but it is
not a replacement for phase serving. The upstream queue still feeds a runtime
request; it does not expose independently batchable E/P/D queues or choose
serial, overlap, and WAIT actions across independent TensorRT contexts.

### 4.7 Other release changes

The official release also contains dual DGX Spark TP=2, JetSpec, deeper MTP
verification trees, NVFP4/Blackwell kernels, JIT XQA on SM90, QNX support, and
many speculative-decoding fixes. These are important upstream capabilities but
are outside the immediate RTX 3080 Cosmos/Gemma phase-serving comparison.

The Python package moved from setuptools to scikit-build-core, added
architecture-aware wheels, and pinned newer tool dependencies. Forward-porting
must keep the new packaging contract instead of copying the old Current
`pyproject.toml` behavior over it.

## 5. Structural comparison

```text
Official v0.10.1                         Current research additions
------------------------------------    ------------------------------------
LLMInferenceRuntime facade              IndependentPhaseAsyncServer
RuntimeCoordinator                      IndependentPhaseCoordinator
LLMRankRuntime                          PhaseThreeCoordinator
EngineExecutor interface                IndependentEngineExecutorPair
TrtEngineExecutor                       E/P/D CUDA streams and TRT contexts
canonical paged KV pool                 StableKVPageManager
ContextCacheCoordinator                 PhasePrefixReuseCache
EncoderEmbeddingCache                   vision/KV ownership horizon
bounded HTTP admission queue            separate E/P/D queues
single-request runtime execution        continuous admission/cancel
CUDA graph cache                        per-phase graph selection/telemetry
                                       PhaseGlobalScheduler
                                       contextual V0/V1/V2 policies
                                       E/P/D/Copy activity timeline
```

The source trees are not competing implementations of exactly the same layer.
`v0.10.1` is now a stronger execution and cache substrate; Current remains the
phase-actor serving and scheduling layer.

## 6. Recommended forward-port order

1. **Freeze the present branch and results.** Do not modify the retained
   `v0.10.0` engines or use `upstream-v0101` as a development root.
2. **Create a new forward-port branch from `v0.10.1`.** Preserve the current
   branch as the performance/correctness oracle.
3. **Port additive phase types first.** Move `cpp/runtime/phase/`, scheduler
   mechanisms, trace analyzers, and unit tests before changing runtime wiring.
4. **Adapt the executor interface.** Add sibling-context creation,
   profile-scoped workspace assignment, identity checks, and graph telemetry to
   upstream `EngineExecutor`/`TrtEngineExecutor`.
5. **Use upstream paged KV as the physical substrate.** Port only stable lease
   ownership, row rebinding, prefix sharing, and scheduler accounting above it.
6. **Attach phase serving below rank coordination.** Integrate into
   `LLMRankRuntime` or a rank-local sibling instead of rebuilding the old
   `LLMInferenceRuntime` monolith.
7. **Replay multimodal adapters at their new paths.** Combine upstream media
   cache/DART with Current E-to-P queue ownership and post-pruning token counts.
8. **Adopt the new server API surface.** Replace its single-runtime adapter
   with the phase async server without forking parsing, media, or engine-cache
   modules.
9. **Validate in gates.** Unit tests, text export/build/inference, VLM
   export/build/inference, then the same V0/V1/V2 and 12-workload comparisons.

The high-risk gates are exact KV/prefix output identity, per-context workspace
memory, CUDA graph replay, E-to-P embedding lifetime, and cancellation while a
phase is outstanding.

## 7. Reproduction commands

```bash
git fetch origin tag v0.10.1
git worktree add --detach .local/upstream-v0101 v0.10.1
git -C .local/upstream-v0101 submodule update --init --recursive

git diff --shortstat v0.10.0..v0.10.1
git diff --dirstat=files,0 v0.10.0..v0.10.1
git merge-base HEAD v0.10.1
git rev-list --left-right --count HEAD...v0.10.1

git diff --name-only v0.10.0..v0.10.1 | sort > /tmp/upstream-v0101-files.txt
git diff --name-only v0.10.0...HEAD | sort > /tmp/current-v010-files.txt
comm -12 /tmp/upstream-v0101-files.txt /tmp/current-v010-files.txt
```

## 8. Decision

Use `.local/upstream-v0101` as the new clean read-only baseline. Do not yet
replace the working branch or its engines. `v0.10.1` materially reduces the
amount of custom paged-KV substrate Current must own, but the runtime split and
executor interface require an explicit forward-port before any fair performance
comparison can be made.
