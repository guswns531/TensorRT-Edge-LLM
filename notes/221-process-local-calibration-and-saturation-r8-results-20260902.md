# R8 Process-local Calibration, Telemetry Fidelity, Saturation 재검증

날짜: 2026-09-02

브랜치: `codex/v010-phase-forward-port`

대상: `nvidia/Cosmos-Reason2-2B`, FP16, RTX 3080 10 GiB, TensorRT 11.0/CUDA 13.3

선행 문서: `notes/220-prepared-directional-execution-r7-results-20260902.md`

## 1. Executive summary

R8은 R7의 prepared directional execution을 다시 바꾸는 단계가 아니라 다음 네 질문을
검증한 단계다.

1. R7 성능이 fresh process 반복에서도 유지되는가?
2. 외부 cost registry 없이 process-local contextual controller의 준비 상태를 어떻게
   관측할 것인가?
3. 48.8 req/s saturation에서 vLLM 우위가 단발 결과가 아닌가?
4. scheduler가 기록한 action과 residual augmentation 뒤 실제 outstanding set이 같은가?

결과는 다음과 같다.

- 48.8 req/s production trace를 5회 재실행했다. Current 평균은 `41.851 req/s`, 95% t-CI는
  `[41.784, 41.918]`이고 frozen vLLM `40.908 req/s`보다 `2.36%` 높다. TTFT, TPOT,
  E2E mean/p95도 모두 Current가 짧다. Token trace hash는 5/5 동일했다.
- balanced는 frozen Current보다 `+2.81%`, bimodal은 `-2.84%`로 3% gate 안이었다.
  vision-heavy는 `-6.30%`이고 process CV도 `4.47%`여서 아직 promotion 실패다.
- multi-image는 median `306.91 token/s`지만 첫 process가 `114.75 token/s`인 cold outlier였다.
  최종 process state를 재사용한 결과만으로 이 workload를 승격하면 안 된다.
- isolated E1/E2/E4와 E+D를 startup에서 강제로 수집하는 실험은 cold outlier를 일부
  제거했지만 5회 중 두 종류의 token hash가 나왔다. 의미 정확도는 유지됐지만 exact greedy
  identity gate를 통과하지 못해 production default로 승격하지 않는다.
- calibration status에 exact-key readiness와 별도로 P→D/D→P, E→P/P→E, E→D/D→E의
  **process-local contextual readiness**를 노출했다. 처음 보는 exact key를 모두 준비해야만
  controller가 준비됐다고 판단하는 잘못을 피한다.
- vision output의 memory path는 `direct_output`, `explicit_d2d`, `not_issued`로 구분한다.
  검증 run은 direct output 43 batches, 논리 payload 1,279,787,008 bytes, explicit D2D 0이었다.
- residual P+D telemetry에서 plan의 authoritative before-set과 다음 poll의 순간 snapshot이
  달라지는 contract bug를 고쳤다. 수정 뒤 48.8 research run은 dispatch/completion/GPU
  interval 1,065개가 모두 대응했고 action-fidelity failure는 0이었다.
- 선택한 P+D와 E+D Nsight capture는 실제 concurrent CUDA-event interval을 보였다.
  다만 E+D 50% 요청은 host preparation/submission delay 때문에 실제 start가 isolated E
  reference의 140% 지점이어서 목표 bucket으로는 실패했다. **overlap 존재**와 **목표 offset
  실현**은 다른 gate로 유지해야 한다.

따라서 R8에서 production saturation 승리는 반복 확인됐지만, vision-heavy와 startup E
calibration은 미완료다. 새 workload별 rule은 추가하지 않는다. 다음 authority promotion은
natural ready state, robust slack, process-local uncertainty가 모두 준비된 action에만 허용한다.

## 2. 최종 control/data plane

```text
HTTP request
    |
    v
Request DAG + stable KV/vision ownership
    |
    v
Global snapshot
  ready E/P/D, oldest slack, in-flight residual, memory ownership
    |
    +---------------------------+
    | deterministic mechanism   |
    | dependency / TRT profile  |
    | one in-flight per context |
    | page/vision feasibility   |
    +-------------+-------------+
                  |
          feasible actions
                  |
    +-------------v-------------+
    | process-local controller  |
    | P+D / E+P / E+D           |
    | continuous feature model  |
    | mean + uncertainty        |
    | exact key is diagnostics  |
    +-------------+-------------+
                  |
          SLO-safe selection
                  |
                  v
 shared CUDA context, independent TensorRT contexts
       E stream       P stream       D stream       Copy stream
          |              |              |               |
          +--------------+--------------+---------------+
                                 |
                       CUDA start/end events
                                 |
              +------------------+------------------+
              |                  |                  |
       exact cost tracker   contextual update   activity mask
       debugging/reference  process-local only  E=1 P=2 D=4 C=8
```

### 2.1 Execution model과 policy model의 분리

Exact cost key는 kernel shape별 실행 계측과 문제 분석에는 유용하다. 그러나 exact key를
policy의 준비 조건으로 사용하면 처음 보는 `(P batch, D batch, context bucket, graph state)`가
모두 cold가 되고 자연 workload에서 evidence가 지나치게 파편화된다.

R8 status schema는 두 readiness를 분리한다.

```text
exact_cost_calibration_converged
    exact execution keys가 minimum samples를 충족했는가?

contextual_policy_calibration_converged
    현재 process에서 실제 사용된 ordered pair direction이
    family minimum observations를 충족했는가?

calibration_converged
    exact readiness OR contextual readiness
```

Contextual readiness는 `P+D`, `E+P`, `E+D` family와 양방향 direction을 각각 센다. Family가
disabled면 요구하지 않고, 해당 process에서 prediction/observation이 전혀 없었던 direction도
가짜 미완료로 세지 않는다. 이것은 영구 저장된 profile이 아니라 현재 server lifecycle의
관측 상태다.

### 2.2 Copy provenance

Copy stream interval이 없다는 사실만으로 zero-copy라고 결론 내리면 안 된다. R8은 vision
memory counters로 다음을 구분한다.

```text
not_issued    : vision payload output이 없었음
direct_output : encoder가 최종 device buffer로 직접 출력
explicit_d2d  : 별도 device-to-device copy가 발생
```

함께 노출하는 값은 direct-output batch/bytes와 D2D operations/bytes다. Nsight 전체 process의
TensorRT input reformat, H2D metadata, initialization copy와 vision payload ownership copy를
혼동하지 않는다.

## 3. 구현 위치

| 변경 | 위치 | 역할 |
|---|---|---|
| contextual calibration JSON | `examples/llm/llm_phase_context_smoke.cpp` | family/direction별 predictions, observations, minimum, required, ready 보고 |
| copy provenance JSON | `examples/llm/llm_phase_context_smoke.cpp` | direct output과 explicit D2D를 PHASE_METRIC에 구분 |
| authoritative transition before-set | `cpp/runtime/scheduling/phaseThreeCoordinator.cpp` | residual plan이 선택한 outstanding set을 telemetry authority로 사용 |
| compact residual lease extension | `cpp/runtime/scheduling/phaseThreeCoordinator.cpp` | newcomer가 관측될 때 retained incumbent의 허용 outstanding set 확장 |
| compact decision validation | `benchmarks/phase_serving/validate_phase_scheduler_events.py` | candidate frontier를 생략한 research record의 action identity 검증 |
| residual activity reconciliation | `benchmarks/phase_serving/validate_phase_scheduler_events.py` | P/D residual dispatch interval도 runtime dispatch 수에 포함 |
| compact decision unit tests | `tests/python-unittests/test_directional_injection.py` | matching identity 허용, mismatched identity 거부 |

## 4. R7 sensitive workload 5-repeat confirmation

각 run은 fresh server process이며 같은 Release binary, engine, materialized request trace를
사용했다. CI는 5개 값에 대한 95% Student-t interval이다. Frozen Current는 R7 이전 canonical
artifact의 median이다.

| Workload | 5-run token/s | Median | Mean / CV | 95% CI of mean | vs frozen Current | TTFT mean/p95 ms | TPOT mean/p95 ms | E2E mean/p95 ms | Peak MiB |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| balanced | 4481.87, 4482.95, 4531.99, 4309.32, 4509.72 | 4482.95 | 4463.17 / 1.98% | [4353.33, 4573.01] | +2.81% | 70.81 / 171.55 | 12.20 / 13.88 | 1108.87 / 1714.72 | 9237 |
| bimodal | 1855.12, 1917.13, 1886.95, 1875.39, 1896.05 | 1886.95 | 1886.13 / 1.23% | [1857.43, 1914.83] | -2.84% | 1927.34 / 4174.18 | 18.24 / 27.92 | 4432.80 / 9253.20 | 9237 |
| vision-heavy | 658.56, 660.84, 701.04, 713.78, 643.50 | 660.84 | 675.54 / 4.47% | [638.08, 713.01] | -6.30% | 1436.52 / 3247.24 | 38.15 / 56.97 | 2954.92 / 3646.22 | 9507 |
| multi-image | 114.75, 251.03, 306.91, 310.12, 309.36 | 306.91 | 258.43 / 32.56% | cold outlier 포함 | run별 변동 큼 | 268.22 / 312.28 | 8.07 / 9.12 | 513.20 / 519.93 | 9483 |

### 4.1 판단

- balanced와 bimodal은 3% throughput gate를 통과한다.
- vision-heavy는 confidence interval이 넓고 frozen Current 대비 6.30% 낮다. 단순히 E batch를
  더 기다리는 rule을 넣지 않고 E queue age, achieved E shape, P/D interference를 같은 run에서
  연결해야 한다.
- multi-image의 첫 process outlier는 engine rebuild가 아니라 server/GPU lifecycle cold effect다.
  median만 보고 개선이라고 주장하지 않는다. production evaluation은 fresh process 반복과
  cold/warm breakdown을 함께 유지한다.

## 5. Startup encoder calibration 실험

동일 server startup에서 real image를 사용해 isolated E1/E2/E4와 E+D를 네 번씩 관측하게 했다.
목적은 persisted registry 없이 E pair posterior를 준비하는 것이었다.

| Run | Generated token/s | Exact token hash |
|---:|---:|---|
| 1 | 314.669 | A |
| 2 | 204.770 | A |
| 3 | 308.292 | B |
| 4 | 283.705 | A |
| 5 | 309.216 | B |

모든 응답의 semantic check는 통과했다. 그러나 giant-panda image request의 greedy token이
미세 수치 차이로 갈려 A/B 두 hash가 형성됐다. 따라서 이 calibration은 다음 이유로 default
경로에 넣지 않는다.

1. startup에 production과 다른 E shape/order를 강제로 만든다.
2. exact identity를 보장하지 못한다.
3. cold process에서 오히려 한 run을 `204.77 token/s`까지 낮췄다.
4. 자연 traffic이 충분한 process에는 불필요한 calibration work다.

현재 원칙은 **natural observation first, uncertain action serial fallback**이다. 미래 safe probe도
already-outstanding completion과 충분한 request slack 안에서만 허용한다.

## 6. Calibration schema와 zero-copy smoke

Artifact의 multi-image smoke 결과는 다음과 같다.

```text
throughput              309.761 token/s
token hash              expected hash와 일치
semantic                pass
required directions     5
ready directions        5
contextual converged    true
vision copy path        direct_output
direct output batches   43
direct output bytes     1,279,787,008
explicit D2D ops/bytes  0 / 0
```

D→E는 해당 process에서 prediction/observation이 없어서 required direction이 아니었다. 이
값은 모든 이론적 direction을 사전에 학습했다는 뜻이 아니라 **이 process가 실제 사용한
policy direction의 minimum evidence가 준비됐다**는 뜻이다.

## 7. 48.8 req/s production 5-repeat

같은 materialized real-request trace와 production telemetry level을 사용했다.

```text
req/s runs = 41.83009, 41.87473, 41.89269, 41.89249, 41.76624
mean       = 41.85125
median     = 41.87473
CV         = 0.129%
95% t-CI   = [41.78427, 41.91823]
token hash = 5/5 identical
```

| System | req/s | token/s | TTFT mean/p95 ms | TPOT mean/p95 ms | E2E mean/p95 ms | Peak MiB |
|---|---:|---:|---:|---:|---:|---:|
| Latest R8 Current | 41.851 mean | 3629.14 median | 35.80 / 60.97 | 12.68 / 15.18 | 1114.65 / 1812.27 | 9237 |
| Frozen Current | 41.798 median | frozen artifact | 35.44 / 61.00 | 12.91 / 15.78 | 1132.66 / 1860.64 | 9237 |
| Frozen vLLM | 40.908 | 3545.40 | 51.44 / 84.27 | 13.53 / 17.66 | 1205.87 / 2121.30 | 9039 |

R8 Current 대 frozen vLLM:

| Metric | Delta |
|---|---:|
| request throughput | +2.36% |
| TTFT mean / p95 | -30.41% / -27.65% |
| TPOT mean / p95 | -6.28% / -14.03% |
| E2E mean / p95 | -7.57% / -14.57% |

R8 Current 대 frozen Current는 throughput `+0.18%`, TPOT mean/p95 `-1.83%/-3.77%`,
E2E mean/p95 `-1.59%/-2.56%`다. TTFT는 mean `+1.01%`, p95 `+0.05%`로 사실상 같은
범위다. 따라서 48.8 승리는 새로운 calibration schema가 만든 큰 gain이 아니라, R7 이후
production mechanism이 안정적으로 유지되는지 확인한 결과다.

## 8. Production과 research telemetry 분리

동일 48.8 trace를 detailed research telemetry로 한 번 실행했다.

| Mode | req/s | TTFT mean/p95 | TPOT mean/p95 | E2E mean/p95 |
|---|---:|---:|---:|---:|
| Production 5-run median/aggregate | 41.875 | 35.80 / 60.97 | 12.68 / 15.18 | 1114.65 / 1812.27 |
| Research post-fix | 41.728 | 36.04 / 64.28 | 13.16 / 16.16 | 1155.42 / 1850.03 |

Research mode는 JSON serialization, detailed interval tracking, validator input을 추가하므로
production 성능 headline으로 쓰지 않는다. 그래도 throughput 차이는 production median 대비
약 `-0.35%`로 작고 token hash는 동일했다.

### 8.1 48.8 activity mask

Post-fix research run의 CUDA event window는 11,163.95 ms다.

| State | Ratio |
|---|---:|
| idle `0000` | 19.88% |
| P only `0010` | 20.42% |
| D only `0100` | 37.74% |
| P+D `0110` | 21.96% |
| P duty | 42.38% |
| D duty | 59.70% |
| E / Copy duty | 0% / 0% |

이 trace는 text-only이므로 E와 vision Copy가 없는 것이 정상이다. 19.88% idle은 offered-load
arrival gap, request completion/admission boundary, CPU sampling/submission 구간을 모두 포함한다.
따라서 idle이 있다는 이유만으로 overlap을 더 강제하면 안 된다. P와 D가 동시에 ready이고
SLO-safe하며 overlap payoff의 lower confidence bound가 양수인 구간만 대상이다.

## 9. Residual action fidelity bug와 수정

수정 전 validator는 한 P incumbent가 이후 D newcomer로 residual-augment된 15개 completion을
outstanding mismatch로 보고했다. 실제 kernel 실행은 잘못되지 않았고 telemetry contract가
다음 두 이유로 잘못됐다.

1. decision 기록 시 `outstanding_before`를 dispatch plan이 아니라 다음 observational poll에서
   다시 읽었다. residual newcomer enqueue 직후의 짧은 시점에는 incumbent snapshot이 비어
   보일 수 있었다.
2. compact telemetry는 detailed `decision.in_flight`를 생략하므로 retained incumbent lease를
   newcomer의 planned outstanding set으로 확장할 증거가 validator에 없었다.

수정된 invariant는 다음이다.

```text
decision.outstanding_before = dispatch_plan.incremental_action.outstanding_before

if retained incumbent phase belongs to
   authoritative before-set AND planned-set:
       extend incumbent allowed outstanding set to planned-set
```

또한 activity validator는 `prefill_residual_dispatch`와 `decode_residual_dispatch`를 정상 dispatch로
센다. Compact decision은 candidate frontier가 비어도 `selected_action_id == action_id`일 때만
opt-in으로 허용한다.

Post-fix 48.8 research validation:

```text
decisions                         857
dispatch/completion executions  1,065 / 1,065
GPU intervals                   1,065
activity dispatches             1,065
action-fidelity failures            0
missing GPU/activity records        0
residual directions             idle 618, P->D 447
dispatch modes                  co_launch 832, residual 62, single 1,236
internal action violations          0
```

Global decision cost는 643 samples, mean `101.27 us`, p95 `200.36 us`, max `320.51 us`였다.

## 10. Selected Nsight P+D / E+D characterization

Nsight Systems 2026.3.1로 CUDA/NVTX를 캡처했다. CUDA-event activity는 benchmark request
window이고 Nsight summary는 process startup, engine/plugin load와 warmup까지 포함한다.
따라서 두 숫자를 직접 나누어 GPU utilization이라고 부르지 않는다.

### 10.1 P→D 50% request

```text
96 requests, 6,656 generated tokens
2,762.49 token/s, 39.84 req/s
peak 9,295 MiB
actual P->D start fraction 0.657 (nearest bucket 0.75)
overlap 8.156 ms
pair serial-equivalent compression +1.61%
```

CUDA-event request window:

| State | Duration ms | Ratio |
|---|---:|---:|
| idle | 45.04 | 1.87% |
| P only | 186.05 | 7.74% |
| D only | 2013.14 | 83.73% |
| P+D | 160.14 | 6.66% |
| total window | 2404.36 | 100% |

Nsight whole-process kernel total은 2,065.29 ms/63,778 instances다. 상위 kernel family는
Ampere GEMM 128x64 `26.7%`, GEMM 64x64 `18.8%`, XMMA GEMM 128x256 `18.5%`,
XMMA fused GEMM `13.6%`, MHA `7.4%`다. GPU memory operation은 313.47 ms이며 H2D가
296.50 ms로 대부분이다. D2D는 70회, 합계 0.121 ms다. H2D max 약 73 ms는 process
초기화/engine path를 포함하므로 steady-state vision copy로 해석하지 않는다.

### 10.2 E→D 50% request

```text
64 requests, 6,176 generated tokens
2,051.82 token/s, 21.26 req/s
peak 9,459 MiB
requested E->D fraction 0.5
actual fraction 1.404, target causally unreachable
measured overlap 16.789 ms
```

Newcomer D는 isolated E reference 22.5 ms 기준으로 너무 늦게 제출됐지만, overlap 동안 E가
48.39 ms로 느려져 실제 GPU interval은 겹쳤다. 이 cell은 `overlap=true`지만 requested offset
bucket의 accepted sample은 아니다.

CUDA-event request window:

| State | Duration ms | Ratio |
|---|---:|---:|
| idle | 137.27 | 4.64% |
| E only | 342.81 | 11.59% |
| P only | 583.42 | 19.72% |
| D only | 1442.43 | 48.75% |
| E+D | 404.54 | 13.67% |
| P+D | 48.37 | 1.63% |
| total window | 2958.84 | 100% |

Nsight whole-process kernel total은 3,682.50 ms/91,317 instances다. 상위 family는 Ampere
GEMM 128x64 `21.7%`, GEMM 64x64 `16.0%`, XMMA GEMM `8.9%`, prefill/vision-related
FMHA `6.5%`, MHA `6.4%`다. GPU memory operation은 337.19 ms이며 H2D 317.66 ms,
D2D 134회/0.442 ms다.

### 10.3 해석과 제한

- low-idle controlled traces에서도 overlap opportunity는 존재하지만 항상 수익은 아니다.
- P+D는 graph replay와 짧은 submit path 덕분에 requested middle offset에 가깝고 작은 positive
  compression을 만들었다.
- E+D는 encoder input/profile 준비와 submit latency가 크고 interference도 커 target fidelity가
  낮았다. E+D policy score보다 newcomer materialization latency를 먼저 줄여야 한다.
- Nsight capture는 각 direction 한 cell뿐이므로 SM active/DRAM/L2의 통계적 결론이나
  direction-wide payoff curve로 사용하지 않는다. R7 30-cell CUDA-event matrix가 수익성
  부호 근거이고, 이번 Nsight는 kernel/API composition 근거다.
- Copy stream duty 0은 vision adapter가 direct output을 사용했기 때문이다. Nsight의 작은 D2D와
  많은 H2D는 TensorRT/runtime metadata를 포함하며 vision payload slab copy와 동일하지 않다.

## 11. Promotion 상태

| Gate | 상태 | 근거 |
|---|---|---|
| R7 prepared P/D/E mechanism | PASS | controlled 30-cell action fidelity |
| 48.8 production stability | PASS | 5-run CV 0.129%, exact hash 5/5 |
| 48.8 vs frozen vLLM | PASS | req/s +2.36%, 모든 latency mean/p95 개선 |
| balanced regression | PASS | frozen Current +2.81% |
| bimodal regression | PASS | -2.84%, 3% 안 |
| vision-heavy regression | FAIL | -6.30%, CV 4.47% |
| multi-image cold stability | FAIL | first process 114.75 token/s |
| forced startup E calibration | FAIL | exact token hash A/B 분기 |
| process-local readiness observability | PASS | 5/5 used directions ready smoke |
| vision copy provenance | PASS | direct output, explicit D2D 0 |
| residual telemetry fidelity | PASS | post-fix 1,065/1,065, violations 0 |
| second GPU/model generality | NOT RUN | 현재 사용 가능한 단일 RTX 3080/모델 |

## 12. 다음 실행 순서

워크로드 이름을 policy feature로 넣지 않고 다음 순서로 진행한다.

1. **Vision-heavy causal decomposition**
   - fresh process 5회에서 E ready rows, actual E batch, oldest vision slack, E prepare/submit,
     P/D overlap placement를 한 timeline으로 연결한다.
   - regression이 E formation, TensorRT E submit, P/D interference 중 어디서 생기는지 분리한다.
2. **Cold lifecycle isolation**
   - multi-image 첫-run outlier를 plugin/module load, image preprocessing, allocator growth,
     CUDA graph warmup으로 분해한다.
   - production 숫자는 cold-start와 steady-state를 별도 보고한다.
3. **Natural E contextual shadow**
   - forced startup probe 없이 natural E+D/E+P opportunity의 mean, uncertainty, robust slack,
     successor formation을 기록한다.
   - selection change는 shadow로만 계산하고 execution authority는 열지 않는다.
4. **E newcomer materialization optimization**
   - image decode/normalize, optimization profile, bindings/workspace prepare를 incumbent dispatch 전
     bounded prepare 단계로 옮긴다.
   - exact greedy identity와 ordinary production ordering을 보존한다.
5. **Safe authority gate**
   - lower-confidence-bound positive, protected TTFT/TPOT slack positive, action-fidelity ready,
     formation risk bounded인 natural action만 active로 승격한다.
6. **Selected repeats**
   - P→D 0.5/0.9, E→D reachable 0.5/0.75, D→P 0.5를 5회 이상 반복한다.
   - target reachability, pair completion, throughput/latency를 함께 보고한다.
7. **Generality**
   - 두 번째 GPU 또는 다른 supported model이 준비되면 같은 continuous feature/controller를
     재학습 없이 process-local observation만으로 적응시키고 payoff surface를 비교한다.

## 13. Artifact 위치

```text
.local/completion-r7-confirmation-4x5-20260902/
  balanced, bimodal, vision-heavy, multi-image fresh-process repeats

.local/completion-r8-encoder-calibration-multi-5x-20260902/
  startup controlled encoder calibration negative result

.local/completion-r8-calibration-schema-smoke-20260902/
  contextual readiness + copy provenance smoke

.local/completion-r8-load48.8-5x-20260902/
  production 48.8 five repeats

.local/completion-r8-load48.8-research-fidelity-1x-20260902/
  post-fix detailed telemetry, activity, validator input

.local/completion-r8-nsys-selected-20260902/
  P+D/E+D CUDA-event artifacts, .nsys-rep, SQLite, CSV summaries
```

## 14. 최종 판단

R8은 두 가지를 확정했다. 첫째, 현재 v0.10 Current의 48.8 req/s saturation 성능은 단발
우연이 아니며 frozen vLLM보다 request throughput과 모든 latency 축에서 반복 우위다. 둘째,
이 결과를 모든 VLM workload에 일반화할 수는 없다. Vision-heavy와 cold multi-image는 아직
불안정하고 forced startup E calibration은 exact identity를 깨뜨렸다.

따라서 다음 연구 가치는 `if vision-heavy`, `if E1/D32` 같은 static rule이 아니다. Legal
action은 deterministic mechanism이 만들고, natural process-local observation을 받은 continuous
controller가 uncertainty와 request slack 안에서만 authority를 얻어야 한다. Copy ownership과
residual outstanding telemetry는 이제 그 판단을 검증할 만큼 구체적으로 관측된다.

현재 가장 큰 남은 기술 문제는 **E action을 더 자주 겹치는 것**이 아니라, vision-heavy에서
E cohort를 깨뜨리지 않으면서 E newcomer를 causally reachable한 시점에 materialize하고 그
선택이 exact output 및 12-workload gate를 동시에 보존하는지 입증하는 것이다.
