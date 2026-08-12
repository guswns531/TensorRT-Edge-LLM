SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

# Plan implementation status

## Completed in this step

### Global CUDA graph free-memory reserve

`EngineExecutor::enableAutomaticCudaGraphCapture()` now accepts an optional
`minimumFreeMemoryBytes` value. When configured, a phase context:

1. checks the reserve before attempting a recurring-shape capture;
2. estimates the next graph charge from the configured minimum or observed average;
3. checks the reserve again after capture before retaining the graph;
4. destroys a graph that would cross the reserve and falls back to normal `enqueueV3()`.

The fallback does not run a logical inference twice. A capture-before-execution rejection executes once through the normal
enqueue path; a post-enqueue capture rejection keeps the already completed inference. New telemetry distinguishes
`global_reserve_bypasses` from `global_reserve_rejections`.

CLI:

```text
--cudaGraphReserveMiB 256
```

The default is zero, preserving the previous behavior. Recommended sweep values are 256, 384, and 512 MiB.

### Fixed-128 prefill token budget

`PhaseQueueSchedulerConfig::maxPrefillBatchTokens` and
`llm_phase_bench --prefillTokenBudget N` limit the total token count of one compatible prefill batch. The scheduler still
requires equal chunk length and equal initial/continuation state, so the option cannot create an invalid TensorRT shape.
Zero disables the limit. This isolates batch-token budgeting from adaptive chunk-size changes.

The first unit test covers four compatible rows with a 256-token budget and confirms that only two 128-token rows are
dispatched in one turn.

## Additional completed work

### Measured-cost decode and phase policy

`build_phase_scheduler_cost_model.py` converts CUDA-event kernel-group CSVs into model-neutral decode/prefill p95 cost
points. `PhaseQueueScheduler` can use those points to select a decode batch that meets the oldest request's TPOT slack.
After a deadline is already missed, the recovery path selects the best batch/token efficiency instead of repeatedly choosing
BS1. Unit tests cover both paths. The CUDA-event/SLO/page-pressure phase selector remains opt-in.

### Whole-request paged-KV backpressure

The first 288-request long-prefill run exhausted the 256-bundle pool even though stable slots were available. The facade now
reserves `ceil((prompt + max output) / 128)` bundles at admission. A request waits in the bounded admission queue when either
a stable slot or its conservative page reservation is unavailable. Reservation is returned on finish/cancel, while physical
pages remain allocated lazily as prefill/decode crosses a page boundary.

This is deliberately conservative: early EOS can reserve pages that are never physically allocated. It prevents an admitted
request from failing later during decode and makes queue growth observable instead of aborting the process.

## Validation state

- TensorRT 26.06 / CUDA 13.3, SM86 release build: passed.
- Relevant C++ tests: 46/46 passed.
- Python compile checks and all selected pre-commit hooks: passed.
- Actual Cosmos FP16 indexed-paged engine run: passed.
- Long-prefill and bimodal 288-request traces: passed after page-aware admission.
- 1,024-request endurance: 1,024 completions; final slots 64/64 free; page pool `allocated=0`, `available=256/256`.
- Current policy variants produced identical request ID, output token count, finish reason, and output text.
- `git diff --check`: passed.

## Remaining work

1. Move the selected policy knobs from the benchmark CLI into production `PhaseAsyncServerConfig`.
2. Add admission reservation headroom/fraction controls instead of reserving the complete requested output unconditionally.
3. Add overlap `Co(P,D)` cost points to the offline model; the current JSON primarily drives decode batch selection.
4. Add CUDA sanitizer and Nsight Systems checks for the final policy.
5. Build an equal-transport HTTP front end before calling the current-vLLM numbers production-E2E fair.
