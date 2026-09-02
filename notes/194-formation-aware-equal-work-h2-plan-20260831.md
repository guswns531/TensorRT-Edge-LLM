<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Equal-Work Formation-Aware H=2 Scheduler Plan

## 0. 2026-08-31 실행 상태

| 단계 | 상태 | 근거 |
|---|---|---|
| P0 결과/trace freeze | 완료 | 동일 384-request, 6-epoch trace와 hash 고정 |
| P1 observable snapshot + equal-work H=2 | 완료 | deterministic replay와 canonical ownership 단위 테스트 통과 |
| P2 contextual decision cost 통합 | 완료 | P+D/E+P/E+D LCB를 selector 입력으로 변환 |
| P3 단일 action authority | 완료 | production residual augmentation까지 post-selector promote/veto/probe 제거 |
| P4 장기 natural trace H=2/myopic A/B | 완료/미승격 | 최종 소스 3회 exact-token run에서 throughput -0.12%; TTFT tail 개선과 TPOT 회귀가 공존 |
| P5 selected-point Nsight | 완료 | 5개 GPU-counter capture와 2개 CUDA/NVTX-only capture; counter sampling perturbation도 분리 기록 |
| P6 최종 12-workload 3회 gate | 완료/실패 | H=2는 일부 workload를 개선하지만 text-heavy -5.06%, multi-image TPOT p95 -22.01% 등 strict gate 실패 |
| P7 fresh vLLM 장기 trace | 계약 실패 | 두 시도 모두 10GB GPU에서 CUDA OOM, 0/3 complete |
| P8 production promotion | 보류 | myopic을 기본값으로 유지하고 H=2는 `TRT_EDGELLM_ENABLE_GLOBAL_FORMATION_AWARE=1` opt-in |

상세 구현·수치·재개 조건은
`notes/195-formation-aware-h2-implementation-and-validation-20260831.md`에 기록한다.

## 1. 목적

이 문서는 현재 transition-aware E/P/D global scheduler를 다음 단계로 확장하는 구현·검증 계획을 고정한다.

핵심 목표는 workload 이름이나 shape별 수동 규칙 없이, 하나의 online selector가 다음 세 효과를 함께 판단하게 만드는 것이다.

1. 현재 action의 GPU 실행 비용
2. KV/vision/workspace ownership의 생성·유지·회수
3. 현재 action이 바로 다음 executable cohort formation에 미치는 영향

최종 정책 순서는 유지한다.

```text
hard feasibility
    dependency / TensorRT profile / single-inflight / memory ownership
                         ↓
robust SLO safety
    TTFT / TPOT / E2E critical path and uncertainty
                         ↓
equal-work H=2 efficiency
    immediate service + successor formation + lifetime transition
```

이 계획은 외부 cost registry, workload label, TTL, 임의의 미래 도착 예측을 사용하지 않는다. Runtime에서 직접 얻은 CUDA 관측과 현재 snapshot, 이미 outstanding인 완료 이벤트만 사용한다.

## 2. 현재 구현에서 유지할 것

- 하나의 CUDA context와 독립 TensorRT E/P/D execution context
- phase별 CUDA stream과 명시적 action lease
- stable indexed KV/page ownership과 vision payload lease
- Legacy-compatible P/D batch builder
- bounded global candidate frontier
- exact CUDA timing model과 contextual P+D, E+P, E+D action-value head
- feasibility와 SLO를 policy score보다 먼저 적용하는 lexicographic selector
- `predictedHorizonUs`와 `horizonReferenceWorkUs`를 사용하는 equal-work 비교 인터페이스

## 3. 현재 구현에서 교체할 것

기존 E formation preview는 vision inter-arrival EWMA로 아직 도착하지 않은 요청 수를 예상한다. 이는 다음 이유로 기본 정책에 부적합하다.

- 실제로 관측되지 않은 미래 arrival을 사용한다.
- workload phase 변화에 민감하다.
- replay에서 동일 snapshot이 동일 action을 만든다는 결정성 설명이 약하다.
- formation-aware action 변화가 실제 관측 state 때문인지 arrival predictor 때문인지 분리하기 어렵다.

따라서 기본 H=2 정책은 다음 정보만 사용한다.

- 현재 E/P/D ready rows와 canonical row order
- 현재 후보가 소비하는 exact request rows
- 후보 실행 뒤 남는 ready rows
- 이미 enqueue되어 completion event가 존재하는 producer rows
- sampling completion처럼 event ID와 bounded completion horizon이 이미 알려진 rows
- 현재 memory ownership과 후보별 allocate/reclaim transition

임의의 future arrival은 H=2 state transition에 넣지 않는다. EWMA 경로는 제거하거나 명시적인 실험용 shadow telemetry로만 격리한다.

## 4. Equal-work 정의

서로 다른 action은 처리하는 work 양이 다를 수 있으므로 단순한 action latency 최솟값은 oracle이 아니다.

각 decision snapshot에서 비교 frontier `W`를 고정한다.

```text
W = current ready work selected for the bounded comparison
  + rows unlocked by one already-outstanding completion event, when included
```

후보 `a`의 비용은 다음과 같다.

```text
C_equal(a, W)
  = C(a)
  + C(residual work in W after a)
```

모든 후보의 `horizonReferenceWorkUs`는 동일한 `W`의 isolated reference work를 사용한다. 작은 batch가 적은 work만 수행해서 유리해지는 비교를 금지한다.

### 4.1 Immediate equal-work oracle

현재 ready E와 선택된 P/D 후보만 비교한다.

```text
E first      = C(E)   + C(PD)
PD first     = C(PD)  + C(E)
E+PD overlap = C(E+PD)
```

세 대안은 동일한 E rows와 P/D rows를 완료한다.

### 4.2 H=2 oracle

현재 action `a0` 적용 뒤 결정적 successor state `S1 = F(S0, a0)`를 만든다. `S1`에서 최대 하나의 successor action `a1`을 평가한다.

```text
J2(a0) = C_equal(a0) + min C_equal(a1 | F(S0, a0))
```

여기서 `F`는 현재 row 소비, 이미 알려진 event completion, ownership transition만 적용한다. 새로운 외부 arrival을 생성하지 않는다.

## 5. Deterministic snapshot/replay 계약

새 snapshot은 최소한 다음을 포함한다.

```text
snapshot epoch
canonical E/P/D ready request IDs
stable KV slot IDs
phase-local row/token/context shape
phase capacity and TensorRT execution variant
outstanding context set and residual cost
known completion event IDs, horizon, and unlocked request IDs
protected request slack
memory ownership totals and candidate deltas
candidate IDs and direct/covered/unknown cost provenance
```

동일 snapshot과 동일 cost state는 동일 candidate frontier, 동일 transition, 동일 선택을 만들어야 한다.

Replay는 live queue를 mutate하지 않는다. 후보 적용은 immutable snapshot 복사본에만 수행한다.

## 6. Bounded cohort 후보

후보 폭발을 막기 위해 phase별 top-k를 제한한다.

```text
E: current maximum-compatible, oldest-SLO-critical, formation-preserving (max 3)
P: current packed maximum, oldest-TTFT-critical, vision-release (max 3)
D: current ready, nearest known completion refill (max 2)
pair: best E+P, E+D, P+D (max 3)
WAIT: nearest useful concrete event (max 1)
total: <= 12
```

첫 production H=2는 기존 batch builder가 만든 canonical row 집합을 재사용한다. 별도의 scheduler 전용 row reorder를 만들지 않는다.

## 7. Online cost 사용

Execution model과 decision model을 분리한다.

- exact runtime cost: GPU timing, replay/oracle reference, debugging
- contextual RLS model: overlap normalized advantage와 uncertainty
- EMA residual: 동일 node의 짧은 drift 보정

Contextual 모델이 예측하는 값은 absolute runtime이 아니라 normalized overlap advantage다.

```text
advantage = (serial reference - overlap makespan) / serial reference
overlap makespan = serial reference * (1 - predicted advantage)
```

LCB가 음수이면 production selector는 overlap을 이익으로 간주하지 않는다. 단, 충분한 SLO slack이 있는 bounded safe probe는 기존 rate limit 아래 허용한다.

## 8. 구현 단계

### P0 — 결과와 계약 freeze

- 현재 12-workload, controlled E/P 및 E/D 결과를 보존한다.
- serving contract, engine, model, request trace hash를 manifest에 기록한다.
- frozen vLLM 결과는 같은 contract에서는 재사용한다.

### P1 — Snapshot과 deterministic replay

- immutable formation snapshot 타입 추가
- candidate row consumption과 known completion transition 추가
- stable candidate/snapshot identity 검증
- 동일 snapshot 반복 replay identity 테스트

### P2 — Immediate equal-work oracle

- 서로 다른 work 양을 동일 frontier로 정규화
- serial order와 overlap을 동일 reference work로 비교
- current selector 선택과 offline oracle regret 기록

### P3 — Bounded cohort + H=2 oracle

- current ready backlog와 concrete completion event에서 successor 후보 생성
- 최대 두 action horizon 계산
- 후보 수 상한과 host decision time 계측

### P4 — Online H=2 selector

- feasibility와 SLO gate 뒤에만 H=2 efficiency 적용
- predicted horizon/reference를 기존 global selector에 전달
- myopic 선택, H=2 선택, oracle 선택을 동시에 shadow 기록
- active 적용은 promotion gate 이후에만 켠다.

### P5 — 검증

- 단위: transition, row conservation, no duplicate consumption, equal reference work, deterministic replay
- controlled positive: immediate overlap과 H=2가 같은 action을 선택하는 case
- controlled counterexample: immediate overlap은 이득이지만 successor fragmentation까지 보면 serial/cohort-preserving action이 이기는 case
- known-event refill: NOW D와 WAIT(event)+larger D 비교
- SLO: H=2가 deadline-unsafe action을 선택하지 않음
- memory: 동일 free bytes이지만 ownership transition이 다른 후보 구분

### P6 — Long-lived natural trace

- 기존 12 workload를 하나의 장수 process에서 순서를 permute하여 반복
- workload 이름을 scheduler에 전달하지 않음
- 학습 warmup 이후 state distribution 변화에 대한 action/uncertainty 추적
- 기록: lookahead opportunity, action change, oracle regret, SLO goodput, scheduler decision median/p95

### P7 — 성능·논문 ablation

동일 runtime과 candidate mechanism에서 다음만 바꾼다.

| Variant | Immediate cost | H=2 formation | Contextual online cost | Active overlap |
|---|---:|---:|---:|---:|
| Serial | O | X | X | X |
| Always overlap | X | X | X | O |
| Best static per trace | offline | X | X | trace별 |
| Myopic online | O | X | O | O |
| H=2 shadow | O | O | O | myopic |
| H=2 active | O | O | O | O |
| Offline oracle | exact replay | O | direct observation | oracle |

측정값:

- request/s, token/s
- E2E mean/p95
- TTFT mean/p95
- TPOT mean/p95
- SLO goodput와 failure attribution
- E/P/D dispatch count와 batch distribution
- D completion-to-next-D gap
- action-induced batch fragmentation count
- H=2 opportunity/change/oracle-regret
- scheduler host decision median/p95/max
- E/P/D/Copy stream busy-mask와 actual overlap fidelity
- peak VRAM, KV/vision ownership high watermark

## 9. Promotion gate

H=2를 active 기본값으로 승격하려면 모두 만족해야 한다.

1. correctness: exact row/slot ownership, no duplicate/lost request, action fidelity 위반 0
2. determinism: 동일 snapshot replay 선택 100% 동일
3. controlled counterexample: myopic과 다른 action을 선택하고 two-action makespan/regret 개선
4. natural frequency: selection change가 실 workload에서 관측되고, 변화가 없는 경우에도 host overhead가 gate 내
5. performance: 12 workload 각각 legacy/current 최선 대비 throughput과 E2E/TTFT/TPOT p95 회귀 3% 이내
6. positive workloads: 최소 한 개 이상 SLO goodput 또는 tail latency 유의미 개선
7. scheduler overhead: decision p95 < 50 us, 목표 < 20 us

## 10. Stop/De-emphasis gate

다음이면 formation-aware H=2를 주 contribution에서 supporting mechanism으로 낮춘다.

- controlled counterexample 외 자연 workload selection change가 사실상 0
- 변화는 있지만 SLO goodput/tail을 개선하지 못함
- candidate materialization/preview host overhead가 GPU 이득을 상쇄
- 정확한 successor를 만들기 위해 arbitrary future arrival predictor가 필수임

이 경우에도 shape-dependent overlap과 action-induced fragmentation characterization은 유지하고, runtime decision-realization 병목 최적화를 우선한다.

### 10.1 이번 실행의 판정

이번 revision은 stop/de-emphasis 조건 중 다음에 해당한다.

- 자연 workload에서 selection change는 존재하지만 전 workload SLO/tail을 함께 개선하지 못했다.
- 장기 trace의 최종 소스 A/B는 throughput parity이며 TTFT와 TPOT 사이 trade-off가 남았다.
- 12-workload strict 3% gate를 여러 항목에서 넘었다.
- 과거 E1/P8 positive counterexample은 현재 serial path 최적화 뒤 재현되지 않았다.

따라서 H=2 mechanism과 telemetry는 유지하되 production default로 승격하지 않는다. 이것은 workload별 fine-tuning을 추가하는 대신, 검증된 myopic policy를 기본값으로 보존하는 판정이다.

## 11. vLLM 및 Nsight 비교 정책

- 동일 request contract와 동일 frozen trace에서는 vLLM을 매 코드 변경마다 재실행하지 않는다.
- Current serving contract, sampling, model/engine 또는 admission 의미가 변할 때 fresh vLLM을 재실행한다.
- 최종 selected points는 Current와 vLLM 모두 5회 이상 반복한다.
- 이번 P5에서 완료한 Nsight selected-point 범위는 다음과 같다.
  - E1/P8 serial vs overlap
  - E8/P8 serial vs overlap
  - E1/D32 serial vs overlap
- H=2가 myopic과 다르게 선택한 natural action과 saturation boundary Current/vLLM의 paired Nsight는 production promotion 이후의 확장 항목이다. 이번 revision은 H=2가 strict gate를 통과하지 못했으므로 추가 profiler 비용을 쓰지 않고, unprofiled end-to-end gate와 기존 vLLM anchor를 최종 판정에 사용한다.

## 12. 성공 시 최종 아키텍처

```text
Request DAG + stable ownership
              │
              ▼
      Immutable global snapshot
              │
              ▼
 Legacy-compatible bounded candidate builders
              │
              ▼
 Hard feasibility ── dependency/TRT/context/memory
              │
              ▼
 Robust SLO protection ── TTFT/TPOT/E2E slack
              │
              ▼
 Equal-work H=2 transition evaluator
       │                   │
       │                   └─ ready-state/known-event formation
       └─ exact cost + contextual RLS uncertainty
              │
              ▼
 Explicit E/P/D, E+P/E+D/P+D, WAIT action lease
              │
              ▼
 CUDA observation ── exact timing + contextual update
```

핵심 연구 명제는 다음과 같다.

> A phase action changes not only immediate GPU service, but also persistent-state lifetime and the shape of the next executable work. The scheduler therefore compares legal actions over an equal-work bounded transition horizon rather than with workload-specific profiles or myopic action latency.
