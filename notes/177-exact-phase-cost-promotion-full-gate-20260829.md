# Exact Phase-Cost Promotion과 전체 승격 Gate

## 목적과 결론

이 단계는 workload 이름이나 latency/throughput/VLM mode를 입력으로 쓰지 않고, 동일한 Global
phase-action scheduler가 현재 ready state, request slack, stable ownership, 실제 CUDA 비용만 사용하게
만드는 작업의 마지막 local-cost 승격 단계다. 외부 cost registry나 fleet service는 도입하지 않았다.

balanced 기준의 승격 후보는 **exact deployment에 대한 decode batching cost만** 남았다. Prefill
formation과 overlap cost를 읽고 평가하는 구현은 완성했지만, balanced A/B gate를 통과하지 못했으므로
promotion metadata에서 꺼 둔다. 다만 마지막 multi-image matched A/B에서 strict latency gate가 하나
남았으므로 decode bundle도 repository 기본값으로 연결하지 않고 opt-in candidate로 둔다. 알 수 없거나
artifact가 하나라도 달라진 배포에서는 기존 static/local mechanism으로 안전하게 돌아간다.

```text
engine/config/external weight/plugin/GPU/software fingerprint
                           |
                           v
                exact compatibility check
                           |
             +-------------+-------------+
             |             |             |
         D promoted    P not promoted  overlap not promoted
             |             |             |
      portable D prior   static P      local safe overlap
             +-------------+-------------+
                           |
                  node-local CUDA update
                           |
              request-state action selection
```

## 구현

### 1. Artifact-exact fingerprint와 자동 승격

- `config.json`, `llm.engine`, `embedding.safetensors`, Edge-LLM plugin을 runtime과 bundle tool에서
  SHA-256으로 직접 계산한다.
- TensorRT/CUDA version, SM, GPU memory, KV dtype, P/D/E batch limit, chunk, KV capacity까지 같은 경우만
  `exact`다.
- bundle의 `promotion`은 `decode_batching`, `prefill_batching`, `overlap_selection`을 독립적으로 갖는다.
- exact + 해당 promotion bit인 경우에만 batch formation/overlap policy authority를 연다.
- serial P/D record는 promotion되지 않아도 action-cost prior로 쓸 수 있다.
- `phase_cost_bundle.py promote`는 같은 shape contract로 통제 실험을 끝낸 calibration record를 현재
  exact artifact에 재결합한다. 원본 관측은 바꾸지 않는다.

주요 위치:

- `cpp/runtime/phase/cost/phaseCostOracle.h`
- `cpp/runtime/scheduling/phaseCostKnowledge.cpp`
- `examples/llm/llm_phase_context_smoke.cpp`
- `scripts/phase_cost_bundle.py`

### 2. Producer-class-aware P cost

P cost key에 `primary_work_class`를 추가해 text initial, continuation, vision-produced P를 구분한다.
기존 class-agnostic build record는 class-specific record가 없을 때만 fallback한다. portable record가
일부 batch에만 존재할 때 그것만 보고 큰 runnable P batch를 잘게 쪼개지 않도록 최대 shape coverage가
없으면 기존 formation을 보존한다.

### 3. WAIT

production WAIT는 한 completion horizon만 보는 bounded residual-work 비교를 유지한다.

```text
NOW  = D(now) + residual work after D(now)
WAIT = event wait + D(future dense cohort)
```

D1에서 D4로 짧게 기다리는 small-D case, NOW 뒤 residual D가 남는 case, 최대 두 completion horizon,
TPOT slack 보호를 unit test로 고정했다. 이 경로 뒤에 남아 절대로 실행되지 않던 full-drain simulation
약 200줄은 제거했다. scheduler hot path에서 전체 queue 복사/drain을 다시 하지 않는다.

### 4. 정책별 promotion A/B

동일 balanced real-request HTTP trace와 production warmup/arrival contract를 사용했다.

| 후보 | tok/s | TTFT p95 | TPOT p95 | E2E p95 | 판정 |
|---|---:|---:|---:|---:|---|
| exact D, 3회 | 4,426.89 | 163.26 | 14.01 | 1,762.34 | 유지 |
| exact D + P, 1회 | 4,291.19 | 179.19 | 14.58 | 1,798.24 | P 기각 |
| exact D + overlap, 3회 | 4,327.62 | 167.25 | 14.45 | 1,805.85 | overlap 기각 |
| controlled exact D + class fallback | 4,510.36 | 162.78 | 13.76 | 1,725.89 | full-gate 후보 |

P는 처리량과 TTFT가 동시에 회귀했다. overlap은 첫 실행 4,557 tok/s였지만 3회 중앙값에서 재현되지
않았고 TPOT p95가 D-only 대비 3%를 넘었다. 둘 다 코드 경로는 남기되 production promotion은 하지
않는다. 이것은 workload별 튜닝이 아니라 동일 gate를 action family별로 적용한 결과다.

평가용 local artifact는
`.local/phase-cost-full-promotion-20260829/cosmos-controlled-exact-d-v3.json`이며 repository에는
commit하지 않는다.

## Memory-pressure 검증

vision-heavy trace에서 KV pressure page를 255, reserve를 2로 강제해 E capacity contraction과 vision
slab reclaim을 실제로 발생시켰다.

- output: 64/64 requests, 2,464/2,464 tokens, OOM 없음
- effective E batch: E4에서 E1로 축소
- slab: allocation 16, reuse 48, reclaim 12
- reclaimed vision bytes: 157,351,936 bytes, 약 150 MiB
- throughput: 635.21 tok/s, normal 691.67 대비 -8.2%
- TTFT mean/p95: 1,452.50 / 3,372.93 ms
- TPOT mean/p95: 28.05 / 34.75 ms
- E2E mean/p95: 2,513.98 / 3,777.74 ms
- peak: 9,523 MiB

pressure path는 안전하게 admission/encoder shape를 줄이고 소유권이 끝난 slab을 회수한다. 다만 이
watermark는 정상 부하에 쓰기에는 공격적이고 latency 비용이 크므로 pressure-only fallback으로 둔다.
raw 결과는 `.local/phase-cost-full-promotion-20260829/vision-heavy-memory-pressure-r2`다.

## Production 12-workload fresh screening

warmup 64, client max-in-flight 64, P8/D64/E4, fixed P chunk 128, tied vision-prefill engine, vision payload
release를 공통으로 사용했다. workload 이름은 scheduler 입력에 들어가지 않는다. latency 단위는 ms다.

| workload | tok/s | 이전 Current 대비 | cached vLLM 대비 | TTFT mean / p95 | TPOT mean / p95 | E2E mean / p95 | peak MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| short | 2,438.09 | -2.72% | +22.92% | 88.68 / 173.28 | 13.79 / 27.47 | 339.08 / 419.87 | 9,313 |
| balanced | 4,528.90 | -0.40% | +4.51% | 65.09 / 164.00 | 12.11 / 13.32 | 1,100.18 / 1,698.25 | 9,313 |
| decode-heavy | 5,160.49 | -1.24% | +3.93% | 61.70 / 171.71 | 10.80 / 11.43 | 2,852.18 / 4,392.87 | 9,313 |
| long-prefill | 1,208.62 | +0.15% | +6.92% | 2,017.52 / 2,615.66 | 26.29 / 30.84 | 4,282.21 / 6,186.65 | 9,313 |
| bimodal | 1,893.65 | -1.37% | +2.91% | 1,916.16 / 3,952.05 | 18.17 / 29.72 | 4,425.54 / 9,084.71 | 9,313 |
| text-heavy | 1,918.80 | -1.44% | +17.38% | 324.46 / 1,105.67 | 25.76 / 38.77 | 1,654.82 / 1,756.42 | 9,395 |
| mixed | 1,122.11 | -1.61% | +21.77% | 739.16 / 2,218.00 | 31.68 / 42.59 | 2,254.69 / 2,549.90 | 9,431 |
| vision-heavy | 694.09 | +0.35% | +19.84% | 1,368.37 / 3,158.43 | 28.09 / 37.15 | 2,481.40 / 3,486.53 | 9,429 |
| poisson | 1,970.89 | -0.77% | +9.49% | 191.89 / 753.44 | 21.82 / 40.41 | 1,583.75 / 2,035.35 | 9,395 |
| wave/drain | 97.61 | -0.15% | +1.83% | 230.80 / 355.13 | 9.79 / 12.24 | 534.42 / 588.92 | 9,469 |
| multi-image, screening | 298.78 | -3.37% | +22.19% | 240.64 / 323.83 | 9.24 / 11.34 | 526.95 / 534.73 | 9,487 |
| late-vision D24 | 2,503.88 | -1.37% | +6.13% | 115.28 / 458.90 | 9.43 / 9.49 | 1,465.77 / 1,841.13 | 9,351 |

multi-image는 요청이 5개뿐인 작은 trace라 historical 단일 결과 대비 3%를 넘었다. 동일 시점의 3회
matched control을 추가했다.

| multi-image 3회 | tok/s | TTFT mean / p95 | TPOT mean / p95 | E2E mean / p95 | token hash |
|---|---:|---:|---:|---:|---|
| static local cost | 292.42 | 202.07 / 328.41 | 9.27 / 12.70 | 520.61 / 545.61 | 반복 불일치 |
| exact-D promoted | 290.02 | 229.23 / 331.06 | 9.76 / 13.14 | 531.86 / 548.89 | 3/3 동일 |

exact-D 처리량은 matched static 대비 -0.82%, E2E p95는 +0.60%다. 반면 TPOT mean은 +5.27%, p95는
+3.50%이고 TTFT mean도 높다. 5-request trace에서 static TPOT p95 자체가 11.33--13.57ms로 크게
움직이고 token hash도 달라졌으므로 명백한 GPU service 회귀라고 단정할 수는 없지만, 정해 둔 strict
3% gate를 임의로 완화하지 않는다. exact-D는 token trace를 3/3 고정한 장점이 있으나 **전역 자동
promotion은 보류**한다. 구현과 opt-in artifact는 유지하고 기본 runtime은 bundle이 지정되지 않으면
계속 static/local cost를 사용한다.

fresh Current는 cached fresh vLLM 대비 처리량 12/12 우위를 유지한다. E2E mean/p95도 12/12 낮다.
남은 약점은 기존과 같다.

- short TPOT mean/p95는 vLLM보다 +3.2%/+10.4% 높다.
- bimodal TTFT mean/p95는 +22.3%/+49.6% 높다.
- long-prefill TTFT mean은 +5.6% 높지만 p95는 -9.4%, E2E mean/p95는 -7.7%/-6.1%다.

vLLM은 model, trace SHA, arrival/output contract가 바뀌지 않아 Note 169의 검증된 fresh 3회 결과를
재사용했다. clean v0.10은 production async phase server와 independent E/P/D API가 없어 이 HTTP
contract를 그대로 실행할 수 없으므로 수치표에 섞지 않는다. clean v0.10은 계속 source/feature
baseline이며, 성능 대조는 동일 serving contract를 가진 Current와 vLLM으로 제한한다.

12종 raw 결과:
`.local/phase-cost-full-promotion-20260829/all-workloads-production-exact-d-v3-12x1`

multi-image 3회:
`.local/phase-cost-full-promotion-20260829/multi-image-production-exact-d-v3-3x-r3`와
`.local/phase-cost-full-promotion-20260829/multi-image-production-static-3x`

## 잘못된 진단 실행

처음 실행한 `.local/phase-cost-full-promotion-20260829/all-workloads-controlled-exact-d-v3-12x1`은
batched vision-prefill/release production env가 빠져 E batch가 1로 제한되고 vision storage release가
0이었다. 이 결과는 원인 진단용으로만 보존하며 성능 근거에 포함하지 않는다.

## 검증

- phase bundle Python tests: 5/5 pass
- `PhaseCostKnowledge`, `PhaseGlobalCostModel`, `PhaseQueueScheduler`, `PhaseMemoryBroker`: 158/158 pass
- fallback 추가 뒤 `PhaseCostKnowledge` + `GlobalWait`: 23/23 pass
- TensorRT 11/CUDA 13.3에서 `unitTest`, `llm_phase_context_smoke` build pass
- exact SHA-256은 표준 `abc` digest와 artifact에 대해 검증
- production 12종: 모든 request/output token 수 일치, OOM 없음

## 다음 단계

다음 구현은 새 휴리스틱을 추가하는 것이 아니다.

1. short에서 small-D WAIT/host submission 비용을 kernel-group timeline으로 분리한다.
2. bimodal은 workload label 없이 oldest request의 remaining critical-path/slack ordering만 개선한다.
3. multi-image보다 요청 수가 큰 small-D VLM trace로 exact-D latency와 determinism trade-off를 다시
   검증한 뒤에만 decode promotion을 기본 연결한다.
4. overlap은 직접 관측 sample coverage와 action-fidelity가 충분해질 때 동일 gate로 다시 승격한다.
5. pressure policy는 normal watermark를 바꾸지 않고 KV 70--90%와 vision lifetime이 겹치는 별도
   stress suite에서만 memory horizon을 평가한다.
