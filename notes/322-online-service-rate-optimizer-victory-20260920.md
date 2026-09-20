# 322. Online service-rate horizon optimizer: Dual-model recovery and Gemma vision-heavy victory

Date: 2026-09-20. Branch: `codex/v0101-phase-forward-port`.

## 1. Outcome

Following the identification of heuristic phase-scheduling limits in Note 320 and Note 321, this campaign implements
and validates an **online Maximum Service Rate (MSR) horizon optimizer with queue-urgency penalty**.

**Key accomplishments**:
1. **Cosmos `poisson` TTFT dramatic recovery**:
   - Dropped mean TTFT from **300.08 ms** (Note 320) to **223.68 ms** (**-25.5% reduction**), outperforming past v3's 243.1 ms by -8.0%.
   - Throughput increased from 1,915.88 tok/s to **2,003.16 tok/s (+4.56%)**.
2. **Cosmos `balanced` throughput recovery**:
   - Recovered throughput from **4,027.52 tok/s** (Note 320) to **4,145.03 tok/s (+117.51 tok/s gain)**.
   - Mean TPOT improved from 13.88 ms to **13.41 ms**, while TTFT dropped to **69.16 ms**.
   - Peak memory remained strictly bounded at **9,601 MiB** (639 MiB headroom below 10,240 MiB).
3. **Gemma 4 `vision-heavy` breakthrough under `shared_ep`**:
   - Throughput jumped from **501.99 tok/s** (Note 317) to **554.39 tok/s (+10.44% gain)**, virtually closing the vLLM gap from -10.3% down to **-0.97%** (vLLM: 559.83 tok/s).
   - Mean TTFT dropped from **422.87 ms** to **381.56 ms (-9.77%)**.
   - **p95 TTFT dropped from 899.47 ms to 648.55 ms (-27.9% reduction)**.
   - Mean TPOT dropped from **34.58 ms** to **30.13 ms (-12.9% improvement)**.
   - Peak memory decreased from **9,823 MiB** to **9,657 MiB** (saving 166 MiB).
4. **Architectural root cause of `pooled_io` with `auto` mode on multimodal workloads**:
   - On Gemma 4, `auto` workspace mode selected `kIndependent` because `freeBytes >= independentVisionBytes`, holding 313 MiB permanently for vision.
   - Simultaneously, `PipelineIOPool` shared IO tensors between prefill and decode; without explicit scheduler serialization between prefill and decode, IO tensor contention stalled decode phases (decode duration ballooned to 3,911 ms).
   - Under `shared_ep` with unpooled PipelineIO, encoder and prefill share context memory while prefill and decode retain independent IO tensors, unlocking seamless P/D concurrency.

## 2. Mathematical formulation: Online MSR with queue-urgency penalty

Rather than relying on heuristic thresholds or static queue caps, `PhaseTransitionPredictor` formulates decode burst
selection as a discrete optimization over the finite burst horizon $N \in [1, 16]$:

$$\mathcal{J}(N) = \frac{\text{Rate}(N)}{\text{PenaltyFactor}(N)}$$

where:
- **Projected service rate**:
  $$\text{Rate}(N) = \frac{N \cdot B_D}{N \cdot T_D + T_{\text{trans}}}$$
  with $B_D$ as candidate decode tokens, $T_D$ as predicted decode step duration (from telemetry/RLS), and $T_{\text{trans}}$ as predicted P $\to$ D transition delay.
- **Queue-urgency penalty factor**:
  $$\text{PenaltyFactor}(N) = 1.0 + \text{SlackPenalty}(N) + \text{Urgency}(N)$$
  where:
  $$\text{SlackPenalty}(N) = \frac{\max(0, W_P + N \cdot T_D - S_P)}{T_P}$$
  $$\text{Urgency}(N) = \begin{cases} \frac{W_P + N \cdot T_D - T_{\text{grace}}}{T_P + T_{\text{trans}}}, & \text{if } W_P + N \cdot T_D > T_{\text{grace}} \\ 0, & \text{otherwise} \end{cases}$$
  with $W_P$ as head-of-line prefill queue wait time, $S_P$ as explicit TTFT slack, $T_P$ as predicted prefill step duration, and $T_{\text{grace}} = 20\text{ ms}$.

### Dynamic behavior:
- **Balanced load** ($W_P \approx 0$ or no prefill queued): The urgency penalty is 0, so $\mathcal{J}(N)$ increases monotonically with $N$, selecting $N \ge 8$ up to 16 to maximize SM amortization.
- **Poisson / bursty load** ($W_P > 20\text{ ms}$ or tight TTFT slack): The denominator expands rapidly with $N$, causing $\mathcal{J}(N)$ to peak at $N = 1$ to 3, yielding the GPU immediately to pending prefills.

## 3. Scorecard summary

| Workload / Model | Note 317 / 320 Baseline | Online MSR Optimizer | Throughput Δ | TTFT Δ | Peak VRAM |
|---|---:|---:|---:|---:|---:|
| **Cosmos `poisson`** | 1,915.88 tok/s / 300.08 ms | **2,003.16 tok/s** / **223.68 ms** | **+4.56%** | **-25.5%** | 9,601 MiB |
| **Cosmos `balanced`** | 4,027.52 tok/s / 13.88 ms TPOT | **4,145.03 tok/s** / **13.41 ms TPOT** | **+2.92%** | **-3.4%** | 9,601 MiB |
| **Gemma 4 `vision-heavy`** | 501.99 tok/s / 422.87 ms | **554.39 tok/s** / **381.56 ms** | **+10.44%** | **-9.77%** | 9,657 MiB |

## 4. Retained artifacts

- Cosmos campaign: `.local/results/cosmos-online-rate-opt-20260920/summary.json`
- Gemma `shared_ep` campaign: `.local/results/gemma-online-rate-opt-shared-ep-20260920/summary.json`
- Gemma `pooled_io` diagnostic: `.local/results/gemma-online-rate-opt-p192-20260920/summary.json`
- Unit tests: `unittests/cpp/runtime/scheduling/phaseTransitionPredictorTest.cpp`
