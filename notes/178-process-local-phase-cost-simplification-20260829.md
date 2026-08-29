# Process-local phase cost simplification

## 결정

Phase cost는 ML 학습이나 외부 knowledge plane이 아니라 현재 프로세스에서 완료된 CUDA action을 요약하는
runtime measurement로 정의한다. 같은 tracker를 E/P/D coordinator가 공유하며, 다음 scheduler decision에서
즉시 사용한다.

제거한 계층은 다음과 같다.

- build/fleet/node cost bundle
- filesystem journal과 node restore
- 모든 TTL
- startup anchor
- drift와 local-only quarantine
- cost bundle promotion bit
- 별도 Python bundle merge/promotion 도구

## 구현 구조

`PhaseRuntimeCostTracker`는 두 종류의 bounded window를 갖는다.

1. E/P/D/E+D/P+D action 전체의 reference work와 makespan
2. 단독 또는 E/P contention 상태별 decode component GPU 시간

두 번째 window를 tracker 안으로 이동해 기존 scheduler의 별도 `mOnlineDecodeGpuMs`와 global action model의
중복 저장을 제거했다.

Action key는 phase, P/D batch, chunk, context bucket, prefill producer class, eager/graph variant를 구분한다.
Decode component key는 batch, context bucket, encoder contention, prefill contention을 구분한다.

## 수집, 활용, 검증 시점

```text
CUDA completion
  -> action ticket/fidelity 검증
  -> valid observation만 bounded window에 추가
  -> DAG/ownership completion 반영
  -> ready snapshot과 bounded candidates 생성
  -> feasibility
  -> deadline protection
  -> measured efficiency
  -> 다음 action dispatch
```

표본이 없으면 `Unknown`, 표본이 있지만 공통 threshold보다 적으면 `Warming`, 충분하면 `Ready`다. Measured
P/D measured formation은 opt-in 평가 경로에서만 `Ready` 비용을 사용한다. production 기본값은 기존의
deterministic candidate mechanism과 configured cost를 유지한다. Unknown overlap은 충분한 slack 아래의
rate-limited safe probe만 허용하며 직접 표본 수와 robust compression gate를 모두 통과해야 정상 후보가 된다.

시간 만료는 없다. 최근 32개 action 표본과 최근 decode component window에서 새 표본이 오래된 표본을
밀어내므로 clock/thermal/contention 변화에 sample-count 기반으로 적응한다. 프로세스가 재시작되면 비우고
대표 E/P/D shape warmup으로 다시 채운다.

## 정책 경계

Cost tracker는 candidate를 만들거나 feature를 승격하지 않는다.

- candidate builder: compatible batch와 canonical row order
- memory broker: KV/vision feasibility와 lifetime
- cost tracker: CUDA timing 통계
- global selector: serial/overlap/WAIT 선택
- executor: action ticket, CUDA event, fidelity

workload별 profile이나 fine-tuning은 사용하지 않는다. 동일한 snapshot, request slack, ownership, recent GPU
cost로 모든 workload를 처리한다. 기능의 기본 활성화는 persistent cost artifact가 아니라 전체 regression
gate를 통과한 코드 릴리스에서 결정한다.

## 최종 아키텍처

```text
HTTP request / continuous admission
              |
              v
       Request DAG + ownership
        E-ready P-ready D-ready
              |
              v
   deterministic candidate builders
   - compatible batch/shape
   - canonical row order
   - fixed P chunk 128
              |
              v
    bounded global candidate set
    E, P, D, E+D, P+D, WAIT
              |
              v
  feasibility -> deadline safety -> efficiency
     |               |                 |
 dependency/TRT   TTFT/TPOT slack   recent CUDA cost
 memory/context                     + uncertainty
              |
              v
 independent TensorRT E/P/D contexts
       in one shared CUDA context
              |
              v
 action ticket + CUDA completion events
              |
              v
     PhaseRuntimeCostTracker
       (process-local windows)
```

핵심 경계는 다음과 같다.

- candidate formation은 runtime cost가 바꾸지 않는다. measured P/D formation은 명시적 opt-in 실험 경로다.
- selector만 최근 action cost를 이용해 serial/overlap/WAIT 중 하나를 고른다.
- action ticket과 실제 outstanding context set이 다르면 그 timing은 정책 입력으로 채택하지 않는다.
- request DAG와 memory broker가 KV/vision lease의 생성, 소비, 회수 가능 시점을 결정한다.
- process-local observation은 correctness나 feature promotion 권한을 갖지 않는다.

## 제거 규모

추적 파일 기준으로 기존 knowledge-plane 구현 약 3.7K line을 제거하고 tracker와 테스트를 추가했다. 주요
삭제 대상은 `PhaseCostOracle`, `phaseCostKnowledge.cpp`, bundle CLI, bundle Python test, knowledge-plane
unit test와 설계 문서다. 최종 diff는 새 파일을 제외한 추적 파일에서 `+242/-3,758`이며, 새 tracker,
테스트, 설계 문서와 이 결과 note 653 line을 포함하면 전체 변경은 약 2.9K line 순감소한다.

## 검증

- TensorRT 11.0 / CUDA 13.3 / SM86 빌드: `unitTest`, `llm_phase_context_smoke` 통과
- focused C++ tests: async server 25 + global cost model 9 + queue scheduler 128 + memory broker 4 + runtime
  tracker 4 = 170/170 통과
- 전체 C++ suite: 1,143개 중 1,098 pass, 42 skip, 3 fail. 실패 3개를 단독 재실행하면 FP8 prefill은
  통과했고, 변경하지 않은 `InitializeYarnRopeCosSin.Accuracy`와 `InitializeMRopeCosSin.Accuracy` 두 개는
  각각 약 0.0012/0.0015의 기존 SM86 수치 tolerance 차이로 재현됐다.
- runtime source-order와 feature-owner boundary validator 통과
- `git diff --check` 통과
- 12개 real-request HTTP trace에서 요청 수와 요청된 output token 수를 모두 완성

## 12-workload 성능 재검증

Cosmos Reason2-2B, P8/D64/E4, fixed P128, max-in-flight 64, 동일 HTTP request/arrival/output contract를
사용했다. balanced, long-prefill, bimodal, text-heavy, mixed, wave/drain, multi-image는 3회 중앙값이고,
나머지는 전체 screening 1회 값이다. latency 단위는 ms다.

| workload | tok/s | TTFT mean | TTFT p95 | TPOT mean | TPOT p95 | E2E mean | E2E p95 | peak MiB | repeats |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| short | 2,514.39 | 84.21 | 166.39 | 13.39 | 22.23 | 327.98 | 407.63 | 9,313 | 1 |
| balanced | 4,583.10 | 66.44 | 164.34 | 12.00 | 13.27 | 1,088.70 | 1,686.62 | 9,313 | 3 |
| decode-heavy | 5,284.12 | 64.58 | 180.28 | 10.54 | 11.08 | 2,787.99 | 4,248.30 | 9,313 | 1 |
| long-prefill | 1,165.23 | 2,088.32 | 2,679.17 | 27.65 | 31.72 | 4,459.54 | 6,294.78 | 9,313 | 3 |
| bimodal | 1,933.24 | 1,875.59 | 4,089.77 | 17.95 | 29.63 | 4,329.45 | 9,221.24 | 9,313 | 3 |
| text-heavy | 1,966.04 | 312.38 | 1,074.18 | 25.20 | 38.99 | 1,613.93 | 1,714.44 | 9,391 | 3 |
| mixed | 1,129.24 | 723.38 | 2,081.72 | 32.16 | 42.06 | 2,244.12 | 2,528.94 | 9,427 | 3 |
| vision-heavy | 693.13 | 1,379.00 | 3,179.52 | 26.85 | 37.03 | 2,452.94 | 3,495.94 | 9,453 | 1 |
| poisson | 1,970.10 | 198.29 | 743.18 | 21.89 | 40.48 | 1,589.89 | 2,023.73 | 9,377 | 1 |
| wave/drain | 97.90 | 204.37 | 303.20 | 9.50 | 11.40 | 498.86 | 510.20 | 9,485 | 3 |
| multi-image | 310.89 | 209.30 | 295.66 | 9.39 | 12.81 | 497.77 | 511.75 | 9,477 | 3 |
| late-vision D24 | 2,549.00 | 113.78 | 454.32 | 9.26 | 9.34 | 1,441.00 | 1,809.96 | 9,351 | 1 |

### 이전 Current 대비 변화

각 열은 현재 값의 변화율이며 latency는 낮을수록 좋다.

| workload | tok/s | TTFT mean | TTFT p95 | TPOT mean | TPOT p95 | E2E mean | E2E p95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| short | +0.6% | -3.5% | -2.3% | -1.4% | -17.8% | -0.8% | -0.6% |
| balanced | +0.6% | +0.7% | -0.5% | -0.8% | -1.0% | -0.7% | -1.1% |
| decode-heavy | +0.4% | -3.4% | +1.0% | -0.2% | -0.5% | -0.4% | -0.5% |
| long-prefill | -2.3% | +2.0% | +1.1% | +3.5% | +1.6% | +2.6% | +3.8% |
| bimodal | -0.8% | +0.2% | +5.2% | +0.9% | -0.2% | +0.3% | +5.3% |
| text-heavy | -0.1% | +1.4% | +5.9% | -0.0% | +0.7% | +0.2% | +0.1% |
| mixed | -0.8% | +2.5% | +0.5% | +0.3% | +1.7% | +0.9% | +1.1% |
| vision-heavy | +0.3% | -0.1% | +1.7% | -4.0% | -0.2% | -1.5% | -0.3% |
| poisson | -0.4% | +1.5% | -1.0% | +0.6% | -1.0% | +0.7% | +0.3% |
| wave/drain | +0.8% | -12.2% | -13.5% | -1.9% | -4.1% | -6.4% | -13.1% |
| multi-image | +3.0% | -4.9% | -4.2% | -1.5% | -1.7% | -1.8% | -2.7% |
| late-vision D24 | +0.2% | -0.9% | -0.7% | -0.1% | -0.0% | -0.2% | -0.2% |

처리량은 12/12에서 ±3% 보존 범위다. strict latency 3% gate는 완전히 통과하지 않았다. long-prefill
E2E p95 `+3.8%`, bimodal TTFT/E2E p95 `+5.2/+5.3%`, text-heavy TTFT p95 `+5.9%`다. 이 세 trace는
기존 측정에서도 asynchronous formation에 따른 tail 변동이 컸고 처리량 및 대부분의 latency는 보존됐지만,
결과를 유리하게 해석하지 않고 tail 관찰 항목으로 남긴다.

### cached fresh vLLM 대비 변화

model, trace SHA, arrival/output contract와 vLLM 설정은 바뀌지 않아 Note 169의 fresh 3회 결과를
재사용했다.

| workload | tok/s | TTFT mean | TTFT p95 | TPOT mean | TPOT p95 | E2E mean | E2E p95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| short | +26.8% | -51.9% | -37.0% | +0.2% | -10.6% | -23.1% | -19.1% |
| balanced | +5.8% | -41.2% | -41.4% | -1.1% | -1.8% | -5.3% | -4.5% |
| decode-heavy | +6.4% | -45.2% | -42.8% | -3.9% | -4.2% | -5.7% | -4.9% |
| long-prefill | +3.1% | +9.3% | -7.2% | -13.9% | -14.4% | -3.9% | -4.5% |
| bimodal | +5.1% | +19.7% | +54.8% | -22.3% | -20.1% | -8.3% | -1.5% |
| text-heavy | +20.3% | -25.9% | -12.8% | -13.7% | -17.7% | -17.0% | -15.9% |
| mixed | +22.5% | -17.3% | -18.1% | -31.5% | -49.9% | -25.4% | -19.5% |
| vision-heavy | +19.7% | -19.4% | -13.9% | -57.9% | -69.0% | -40.5% | -17.3% |
| poisson | +9.4% | -54.7% | -17.7% | -1.3% | -11.4% | -11.7% | -10.7% |
| wave/drain | +2.1% | -19.1% | -27.6% | -23.6% | -33.9% | -21.8% | -21.4% |
| multi-image | +27.1% | -19.4% | -26.6% | -24.4% | -21.5% | -22.7% | -21.7% |
| late-vision D24 | +8.0% | -25.8% | -28.1% | -6.4% | -6.0% | -8.6% | -7.4% |

Current는 cached fresh vLLM보다 처리량 12/12, E2E mean/p95 12/12에서 우위다. 남은 명확한 약점은
long-prefill TTFT mean과 bimodal TTFT mean/p95다. short TPOT mean은 `+0.2%`로 사실상 동률이고 TPOT
p95는 Current가 낮다.

선택한 최종 실행은 모든 workload에서 반복 내 token hash identity를 통과했다. peak VRAM은
`9,313--9,485 MiB`이며 이전 Current high-watermark보다 8 MiB 증가했다.

## 결과 위치

- 전체 1회 screening: `.local/process-local-cost-20260829/mechanism-parity-remaining-9x1`
- balanced/wave/multi 3회: `.local/process-local-cost-20260829/mechanism-parity-sensitive-3x`
- long-prefill/bimodal 3회: `.local/process-local-cost-20260829/long-bimodal-repeat-3x`
- text-heavy/mixed 3회: `.local/process-local-cost-20260829/text-mixed-repeat-3x`
- wave exact 재검증 3회: `.local/process-local-cost-20260829/wave-exact-repeat-3x`
