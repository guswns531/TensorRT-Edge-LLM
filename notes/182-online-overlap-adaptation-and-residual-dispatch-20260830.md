# Online overlap adaptation과 residual phase dispatch 최종 구현

## 결론

이번 단계에서는 independent TensorRT execution context가 실제로 동시에 실행될 수 있는지와, 왜 production
trace에서는 E/P/D overlap이 거의 보이지 않는지를 분리해서 해결했다.

핵심 결과는 다음과 같다.

- 한 CUDA primary context 안에서 E/P/D가 서로 다른 TensorRT execution context와 non-blocking stream을
  사용한다. 공유 TensorRT context에서는 residual augmentation을 거부한다.
- scheduler action과 실제 CUDA outstanding phase set을 dispatch lease로 일치시킨다.
- 실행 중인 P 또는 D에 반대 phase를 뒤늦게 붙이는 residual `P+D` 경로를 구현했다.
- 미지의 overlap은 bounded safe probe로만 관측하고, 관측된 shape는 process-local robust cost로 판단한다.
- E/P/D/Copy 작업 구간을 CUDA event로 기록하고 `E=0001`, `P=0010`, `D=0100`, `C=1000` mask로
  sweep한다. sampling도 해당 P/D phase 시간에 포함한다.
- 동일 active lease와 동일 missing cohort를 host poll마다 재평가하던 hot loop를 제거했다. multi-image
  진단에서 residual 후보 평가는 `7,306 -> 2`회로 줄었다.
- Cosmos Reason2-2B의 동일 12개 실제 HTTP arrival/output trace를 workload별 설정 변경 없이 3회씩
  재실행했다. cached fresh vLLM 대비 처리량은 12/12 workload에서 높았다.

독립 context가 overlap을 가능하게 만들지만 overlap 자체가 항상 이득은 아니다. RTX 3080에서 P와 큰 D가
동시에 full-SM kernel을 실행하면 resource contention 때문에 makespan이 serial보다 길어질 수 있다. 따라서
현재 scheduler가 대부분 serial을 선택하는 것은 기계적 실패가 아니라 관측된 비용과 request slack에 따른
결정이다.

## 실행 구조

```text
                         one CUDA primary context
                                  |
              +-------------------+-------------------+
              |                   |                   |
       TensorRT E context  TensorRT P context  TensorRT D context
          encoder stream      prefill stream       decode stream
              |                   |                   |
              +-------------------+-------------------+
                                  |
                         copy stream / events

Global scheduler action
  E | P | D | E+D | P+D | WAIT
            |
            v
PhaseGlobalDispatchPlan (stable action/row/slot lease)
            |
            +-- initial dispatch
            |
            +-- live residual augmentation
                  P running + D ready -> P+D
                  D running + P ready -> P+D
```

CUDA context와 TensorRT context의 역할은 다르다.

- CUDA primary context는 device allocation, stream, event가 공유되는 GPU process boundary다.
- TensorRT execution context는 activation/workspace/binding 실행 상태다.
- 동시 실행에는 P/D execution context와 workspace가 서로 독립이어야 한다.
- 같은 TensorRT context를 두 stream에서 동시에 enqueue하는 것은 허용하지 않는다.

`PhaseExecutionSafetyContract`가 engine/context/workspace alias가 없음을 확인하며,
`PhaseDispatchWorker::augmentNext()`는 `kIndependentConcurrent`에서만 live augmentation을 허용한다.

## action fidelity

이전의 중요한 위험은 scheduler가 `E`를 선택했는데 다음 host poll에서 `D`를 enqueue하여 실제로는 `E+D`가
되는 것이었다. 현재 action은 단순히 한 번 enqueue할 phase가 아니라 다음 decision boundary까지 허용되는
outstanding set이다.

```text
planned lease        observed outstanding       result
-------------------------------------------------------
P                    P                          valid
P+D                  P+D                        valid
P                    P+D                        fidelity violation
shared context P     P+D augmentation request   rejected
```

coordinator와 worker가 동일 plan ID, snapshot epoch, ordered request rows, stable KV slot IDs를 전달한다.
launch 직후 observed outstanding mask가 lease와 다르면 즉시 invariant violation으로 처리한다.

## residual P+D dispatch

처음 action을 고를 때 P와 D가 동시에 ready하지 않아도, 한 phase가 실행되는 동안 반대 queue가 ready가 될 수
있다. 기존 worker의 단일 `mBusy` 상태는 이 기회를 막았다. 새 경로는 다음 순서로 동작한다.

1. 현재 live lease가 P 또는 D 한 개만 소유하는지 확인한다.
2. 반대 queue의 legacy-equivalent batch former로 missing candidate를 preview한다.
3. active action에서 이미 지난 시간을 빼 residual work를 만든다.
4. canonical `P+D` key의 process-local 관측을 찾는다.
5. residual key가 아직 sparse하면 동일 complete `P+D` 관측을 보수적 upper bound로 사용한다.
6. dependency, TensorRT shape, memory horizon, TTFT/TPOT slack을 검사한다.
7. known profitable 또는 bounded safe probe일 때만 live lease를 P+D로 upgrade한다.
8. augmentation 시점에 별도 CUDA event를 기록해 원래 phase 시작이 아닌 residual makespan을 학습한다.

triple phase augmentation은 지원하지 않는다. E가 outstanding인 상태에서 P+D를 추가하거나 P+D에 E를
붙이는 action은 거부한다.

## process-local online cost

외부 cost registry나 workload profile은 사용하지 않는다. 실행 중 직접 얻은 CUDA event 표본만 현재 process에
보관한다.

canonical key는 다음 정보를 보수적인 upper bucket으로 정규화한다.

```text
action kind
+ P batch upper-power-of-two bucket
+ D batch upper-power-of-two bucket
+ fixed chunk bucket
+ P/D context upper bucket
+ eager / graph execution variant
+ text / external prefill class
+ complete / residual augmentation
```

표본 상태는 `no samples -> insufficient -> eligible`로 이동한다. TTL은 없다. serving epoch가 바뀌어도 같은
process와 engine에서 유효한 robust window는 유지되며, process가 종료되면 모두 사라진다.

unknown overlap은 기본적으로 serial fallback이다. 단, 다음 조건을 모두 만족할 때만 safe probe를 허용한다.

- safe probe multiplier가 0보다 큼
- 마지막 probe 이후 bounded decision interval 경과
- protected request의 slack이 robust serial cost의 기본 3배 이상
- dependency/context/shape/memory feasibility 통과

safe probe 0과 3을 bimodal, Poisson, multi-image에서 3회 A/B한 결과 방향이 일관되지 않았다. 따라서 safe
probe 자체가 이번 성능 회귀의 원인은 아니며 기본 3을 유지한다.

## host poll hot-loop 수정

초기 residual 구현은 GPU completion event가 아직 끝나지 않은 동안 같은 live lease와 같은 missing cohort를
매 poll마다 다시 평가했다.

```text
same P lease + same D cohort
        |
        +-- preview / cost lookup / selector 7,306 times
        |
        +-- actual GPU decision boundary: 1
```

이는 online adaptation이 아니라 중복 host work다. poll 횟수로 `mGlobalDecisionSequence`까지 증가해 probe
cooldown 의미도 오염했다.

수정 후 `ActiveGlobalPdExecution`이 마지막 missing candidate ID를 보관한다. 같은 candidate는 한 lease에서
한 번만 평가한다. 다음 상황에서는 자동으로 다시 평가한다.

- queue row/order/shape가 바뀌어 candidate ID가 변경됨
- 현재 P/D가 완료되어 새 dispatch lease가 생성됨
- 다음 phase action이 시작됨

multi-image metrics-on 단일 진단 결과:

| 항목 | 수정 전 | 수정 후 | 변화 |
|---|---:|---:|---:|
| residual P+D opportunities | 7,306 | 2 | -99.97% |
| measured overlap selections | 0 | 0 | 동일 |
| throughput | 282.90 tok/s | 302.81 tok/s | +7.04% |
| E2E p95 | 562.54 ms | 525.23 ms | -6.63% |

즉 개선은 overlap 강제가 아니라 host submission gap 제거에서 왔다.

## E/P/D/Copy activity 정의

`PhaseActivityTimelineRecorder`는 각 stream의 실제 작업 enqueue 앞뒤에 CUDA event를 기록한다.

```text
E = 0001
P = 0010
D = 0100
C = 1000

time ---------------------------------------------------------->
E       [==========]                    [=======]
P                 [=============]
D       [===] [===] [===] [===] [===] [===]
Copy          [--]             [---]

mask    0101  1111  0110 ... 0000 ...
```

이 mask는 “stream이 존재함”이나 “action이 예약됨”이 아니라 CUDA event 사이에 실제 GPU work가 outstanding인
구간을 뜻한다. P/D engine 뒤 sampling도 각각 P/D activity로 포함한다. Copy는 vision staging/release 같은
명시적 async memory operation만 포함하며 TensorRT 내부 copy를 별도 C로 추정하지 않는다.

분석기는 다음 두 시간축을 분리한다.

- planned action: scheduler metric의 plan/observed outstanding mask
- actual activity: CUDA event interval sweep의 E/P/D/C mask

warmup과 measured 요청이 request ID를 재사용하고 sampling correlation ID가 dispatch ID와 충돌할 수 있다.
따라서 분석기는 P/D engine interval만 kind-aware dispatch ID로 찾고, immediate sampling tail과 E/C batch를
measured time window로 결합한다. 이 수정 전 multi-image active span은 잘못된 warmup sampling을 포함해
1,235.5ms였고, 수정 후 실제 537.5ms가 되었다.

## 실제 overlap 검증

safe probe multiplier 1의 balanced 진단에서 measured production 구간은 다음과 같다.

- active span: 5,635.01ms
- P activity: 1,577.57ms
- D activity: 3,829.56ms
- internal idle: 230.94ms
- actual P+D overlap: 3.07ms
- planned P+D: 2회
- actual same-dispatch P+D: 2회
- missed: 0회
- unplanned same-dispatch overlap: 0회

즉 independent context와 stream은 실제 overlap한다. 하지만 한 대표 residual P1+D47은 isolated reference
work 9.02ms에 비해 residual makespan 10.47ms로 길었다. 두 full-SM kernel의 contention 때문에 effective
service compression이 1보다 작으므로 이후 동일 shape는 serial이 맞다.

production 기본 multiplier 3에서는 slack guard가 더 보수적으로 작동한다. E+D는 VLM의 E->P first-token
critical path와 D TPOT를 동시에 보호해야 하므로 현재 trace에서 거의 선택되지 않는다. overlap count를
목표로 삼지 않고 `planned action = actual execution`과 latency/goodput을 목표로 삼는다.

## 최종 12-workload 설정

- model: `nvidia/Cosmos-Reason2-2B`
- GPU: RTX 3080 10GiB, SM86
- TensorRT 11.0.0 / CUDA 13.3 container
- P8 / D64 / E4
- fixed packed prefill chunk 128
- max stable slots 80, max in-flight 64
- independent P/D contexts와 별도 E context
- 같은 HTTP arrival trace, prompt, requested output length, ignore-EOS contract
- workload별 scheduler profile/knob 변경 없음
- production JSON phase metric 출력은 비활성, 내부 CUDA cost learning은 활성
- 각 workload 3회, 아래 값은 run 중앙값

## 최종 절대 성능

latency 각 칸은 `mean / median / p95` ms다.

| workload | tok/s | TTFT | TPOT | E2E | peak MiB |
|---|---:|---:|---:|---:|---:|
| short | 2,491.37 | 84.57 / 67.52 / 170.68 | 13.69 / 11.58 / 27.04 | 332.17 / 341.18 / 411.62 | 9,313 |
| balanced | 4,475.67 | 67.61 / 54.60 / 164.95 | 12.31 / 12.68 / 13.76 | 1,117.55 / 1,196.75 / 1,734.90 | 9,313 |
| decode-heavy | 5,230.16 | 65.58 / 54.59 / 180.72 | 10.64 / 10.87 / 11.20 | 2,818.90 / 3,100.63 / 4,297.24 | 9,313 |
| long-prefill | 1,220.59 | 2,016.29 / 2,135.88 / 2,691.41 | 25.79 / 26.91 / 29.44 | 4,243.37 / 4,123.18 / 5,934.00 | 9,313 |
| bimodal | 1,908.77 | 1,888.38 / 2,393.72 / 3,983.77 | 18.12 / 16.72 / 30.09 | 4,410.55 / 4,239.61 / 9,034.88 | 9,313 |
| text-heavy | 1,932.61 | 324.00 / 162.65 / 1,139.06 | 25.58 / 24.91 / 40.06 | 1,651.28 / 1,699.11 / 1,745.31 | 9,389 |
| mixed | 1,121.04 | 713.35 / 256.72 / 2,059.78 | 32.61 / 36.72 / 41.86 | 2,259.17 / 2,520.22 / 2,547.71 | 9,427 |
| vision-heavy | 690.29 | 1,376.59 / 1,157.68 / 3,190.68 | 27.47 / 30.14 / 37.32 | 2,471.48 / 2,398.85 / 3,505.26 | 9,439 |
| Poisson | 1,934.80 | 208.62 / 67.24 / 766.23 | 22.19 / 19.68 / 40.49 | 1,631.55 / 1,609.09 / 2,073.86 | 9,397 |
| wave/drain | 97.71 | 231.53 / 225.02 / 350.11 | 9.64 / 9.44 / 11.90 | 530.84 / 519.74 / 581.43 | 9,459 |
| multi-image | 276.76 | 237.04 / 274.69 / 358.17 | 9.71 / 8.98 / 13.71 | 548.34 / 553.19 / 575.68 | 9,353 |
| late-vision D24 | 2,548.42 | 115.19 / 44.10 / 458.43 | 9.26 / 9.24 / 9.33 | 1,442.99 / 1,806.36 / 1,809.55 | 9,351 |

모든 run이 requested output token을 완성했다. text workload는 각 3회 token trace SHA-256이 exact match했다.
이번 multi-image 3회도 canonical `9f801809...8742`로 exact match했다. 이 hash는 이전 semantic gate를
통과한 출력이다.

## stable Current 및 최근 checkpoint 비교

`Stable Current`는 Note 167의 확정 3회 promotion baseline이고, `Recent`는 Note 178의 process-local cost
simplification 결과다.

| workload | Final tok/s | vs Stable Current | vs Recent | vs cached fresh vLLM |
|---|---:|---:|---:|---:|
| short | 2,491.37 | +0.59% | -0.92% | **+25.60%** |
| balanced | 4,475.67 | -1.91% | -2.34% | **+3.28%** |
| decode-heavy | 5,230.16 | -0.74% | -1.02% | **+5.33%** |
| long-prefill | 1,220.59 | +2.01% | +4.75% | **+7.97%** |
| bimodal | 1,908.77 | -1.87% | -1.27% | **+3.73%** |
| text-heavy | 1,932.61 | -1.29% | -1.70% | **+18.22%** |
| mixed | 1,121.04 | -1.46% | -0.73% | **+21.66%** |
| vision-heavy | 690.29 | -0.44% | -0.41% | **+19.18%** |
| Poisson | 1,934.80 | -1.89% | -1.79% | **+7.48%** |
| wave/drain | 97.71 | +0.96% | -0.20% | **+1.94%** |
| multi-image | 276.76 | -5.47% | -10.98% | **+13.18%** |
| late-vision D24 | 2,548.42 | +0.16% | -0.02% | **+8.02%** |

11/12 workload는 Stable Current 대비 처리량 ±3% gate 안이다. multi-image는 요청이 5개뿐이고 E batch
formation 경계에 민감하다. 같은 최종 binary의 별도 metrics-on run은 302.81 tok/s였으며 이전 여러 stage도
약 270--313 tok/s를 오갔다. 이를 유리하게 골라 쓰지 않고 최종 연속 suite의 3회 중앙값 276.76을 유지한다.
vLLM보다는 여전히 +13.18%다.

## vLLM p95 비교

아래 값은 `(Current / cached fresh vLLM - 1)`이며 latency이므로 음수가 좋다.

| workload | TTFT p95 | TPOT p95 | E2E p95 |
|---|---:|---:|---:|
| short | -35.3% | +8.7% | -18.3% |
| balanced | -41.2% | +1.8% | -1.8% |
| decode-heavy | -42.7% | -3.1% | -3.8% |
| long-prefill | -6.7% | -20.5% | -10.0% |
| bimodal | +50.8% | -18.9% | -3.5% |
| text-heavy | -7.6% | -15.4% | -14.4% |
| mixed | -19.0% | -50.2% | -18.9% |
| vision-heavy | -13.6% | -68.8% | -17.1% |
| Poisson | -15.1% | -11.3% | -8.5% |
| wave/drain | -16.4% | -31.1% | -10.5% |
| multi-image | -11.0% | -16.0% | -12.0% |
| late-vision D24 | -27.4% | -6.1% | -7.4% |

남은 명확한 약점은 bimodal TTFT tail이다. throughput, TPOT, E2E는 vLLM보다 좋지만 short/long request의
KV-page admission 및 service ordering 때문에 일부 long request의 first token이 늦다. 이는 overlap을 더
늘리는 문제가 아니라 request-level critical-path/fair admission 문제다. short TPOT p95와 balanced TPOT
p95는 vLLM보다 각각 8.7%, 1.8% 느리지만 E2E p95와 throughput은 더 좋다.

## 구현 위치

### activity와 분석

- `cpp/runtime/scheduling/phaseActivityTimeline.h`
- `cpp/runtime/scheduling/phaseActivityTimeline.cpp`
- `benchmarks/phase_serving/analyze_phase_activity.py`
- `tests/python-unittests/test_phase_activity_analysis.py`

### worker와 execution lease

- `cpp/runtime/scheduling/phaseDispatchWorker.h`
- `cpp/runtime/scheduling/phaseDispatchWorker.cpp`
- `cpp/runtime/phase/execution/phaseActionPlan.h`

### global online policy와 비용

- `cpp/runtime/scheduling/phaseGlobalScheduler.cpp`
- `cpp/runtime/phase/policy/phaseGlobalCostModel.h`
- `cpp/runtime/scheduling/phaseQueueScheduler.h`
- `cpp/runtime/scheduling/phaseQueueScheduler.cpp`

### three-phase orchestration

- `cpp/runtime/scheduling/phaseThreeCoordinator.h`
- `cpp/runtime/scheduling/phaseThreeCoordinator.cpp`
- `cpp/runtime/scheduling/phaseVisionAdapter.h`
- `cpp/runtime/scheduling/phaseVisionAdapter.cpp`
- `cpp/runtime/scheduling/independentPhaseCoordinator.h`
- `cpp/runtime/scheduling/independentPhaseCoordinator.cpp`
- `cpp/runtime/scheduling/independentPhaseAsyncServer.h`
- `cpp/runtime/scheduling/independentPhaseAsyncServer.cpp`

### telemetry와 검증

- `examples/llm/llm_phase_context_smoke.cpp`
- `unittests/phaseGlobalSchedulerTest.cpp`
- `unittests/phaseShellTest.cpp`

## 검증 결과

- focused C++ scheduler/activity/worker/server suite: 196/196 pass
- 전체 C++ suite: 1,151 tests, 1,107 pass, 42 skip, 0 fail, 3 disabled
- Python activity analyzer: 4/4 pass
- actual P+D action fidelity: planned 2, actual same-dispatch 2, missed 0
- 12 workload x 3 production HTTP runs: 모두 성공
- requested output token completion: 모두 성공
- peak VRAM: 9,313--9,459MiB, 10GiB RTX 3080에서 OOM 없음

## 현재 판단과 다음 우선순위

이번 단계에서 “왜 overlap이 안 되는가”는 세 종류로 분리됐다.

1. **기계적 불가**: shared TensorRT context/workspace인 경우이며 안전하게 거부한다.
2. **비용 unknown**: bounded safe probe로만 직접 관측한다.
3. **관측상 손해 또는 slack 위험**: independent context라도 serial이 올바른 선택이다.

현재 full-SM TensorRT kernels에서 무작정 overlap 횟수를 늘리는 것은 목표가 아니다. 다음 성능 단계는 다음
순서가 합리적이다.

1. bimodal의 request-level KV admission/TTFT ordering을 profile-free slack 기준으로 개선한다.
2. E+D/P+D의 positive region을 더 만들려면 SM/resource partition backend를 연결한다.
3. resource partition별 cost를 같은 canonical online key의 execution variant로 학습한다.
4. Copy activity가 실제 있는 VLM staging path를 별도 async memcpy stream에 더 넓게 연결한다.
5. multi-image처럼 작은 trace는 더 큰 반복 cohort를 만들어 formation variance와 policy 효과를 분리한다.

가장 중요한 invariant는 계속 다음 세 가지다.

```text
batch gain > wait cost
wait/overlap <= protected request slack
planned outstanding action = actual CUDA execution
```
