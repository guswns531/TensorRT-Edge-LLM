# 260. Release 조건의 v0.10.0/v0.10.1 V0/V1/V2 및 KV 비교

> 중단/정정: VLM generic calibration이 삭제된 `.local/upstream-v010` 이미지 경로를 참조함을
> 추가로 발견했다. Client는 warmup 응답 실패를 검사하지 않고 제출 수를 완료 수로 기록했다.
> 따라서 아래 Release VLM 수치도 old와 동일 학습 조건의 비교가 아니다. Text-only 자료와
> KV microbenchmark는 이 이미지 문제와 별개지만, 전체 suite를 유효 완료로 표시하지 않는다.
> 원본은 보존하고 asset-remap 및 warmup response guard를 적용해 재검증한다.

## 상태

Release 대표 HTTP 6회, phase tests 322개와 opt-in KV metadata test는 완료했다.
Full12는 36 cells × 3 fresh-process repeats로 진행 중이다. 이 문서의 상태를 완료로
바꾸기 전에는 전체 suite 성공을 주장하지 않는다. 결과는
`.local/results/v0101-forward-port/release-parity-20260909/`에 보존한다.

## 이번에 수정한 것

1. `CMakeLists.txt`: 명시한 SM86 architecture를 default multi-SM 목록으로 덮어쓰지 않는다.
2. Runtime을 명시적으로 Release로 빌드한다. C++/CUDA actual flags 모두 -O3/-DNDEBUG,
   CUDA는 기존 Release와 같은 --use_fast_math 및 SM86을 사용한다.
3. `replay_retained_policy_commands.py`: Release cache와 실제 C++/CUDA flags를 사전 검증한다.
4. `PhaseThreeCoordinatorConfig::preserveLegacyPairEligibility`: opt-in 비교 설정으로
   external residual P+D authority 및 E/P-exclusive와 E/D eligibility의 관계를 예전 범위로 맞춘다.
   새 default policy로 승격하지 않는다. E/D workspace safety와 unknown-payload E1 bootstrap은 유지한다.
5. `PhaseKVActiveViewTest` 및 `kvPageTableBench.cpp`: stable lease/active view 비용과
   clean/current underlying page-table upload 비용을 분리해 측정한다.
6. `report_legacy_parity.py`: old/current V0/V1/V2와 동일 계약 frozen vLLM의 전체 latency,
   throughput, memory, output fidelity를 저장한다. 누락된 cell을 성공으로 처리하지 않는다.

기존 dispatch-only telemetry 변경은 보존했으며 이번 변경으로 위장하지 않는다.
Runtime 추가 부분은 source.patch로, 바이너리와 engine identity는 manifest.json으로 기록한다.

## 환경 문제와 upstream 변경의 구분

v0.10.0 retained benchmark는 Release였지만 v0.10.1 비교 빌드는 build type이 비어 있었다.
따라서 그 차이는 upstream v0.10.1의 필연적 성능 회귀가 아니라 우리의 build/reproduction
계약 문제다. 실제로 같은 KV active-view 코드가 Release에서 크게 빨라졌다(note259).
이전 수치를 삭제하지 않되 버전/정책의 인과 효과로 해석하지 않는다.

별도로 존재하는 upstream 변화는 pool-shaped KV API, exporter/TRT engine, XQA cubin 경로 등이다.
이들은 Release를 맞춘 후에도 남는다. 소스가 같은 RLS/lease/view와 다른 engine/kernel을 구분한다.
기존과 다른 출력 hash가 있으므로 cross-engine exact greedy identity를 통과했다고 표현하지 않는다.

## 공통 실험 계약

- Cosmos Reason2-2B FP16, KV FP16, 동일 text/vision engine. 재양자화나 KV 축소 없음.
- Slots80 / KV256 / P8 / D64 / E4, text chunk128, vision prefill profile1024.
- Independent E/P/D, client64, ignore-EOS, generic calibration을 매 process 재실행.
- 기존 manifest와 동일하게 `SYNCHRONIZE_DECODE_SAMPLING=1`, 실제 graph replay는 사용하지 않는다.
  Independent context 지원과 async-sampling/graph 활성화는 서로 다른 조건이다.
- V0 Exact → V1 Scalar → V2 Scalar+Transition 순서, cell당 3회. ABBA가 아니다.
- Old 및 vLLM은 retained 수치다. vLLM의 네 text case는 client64 equal-cap 기록을 사용한다.
- Mean latency는 run mean의 평균, p95는 run p95의 중앙값, throughput은 run 중앙값이다.
  합쳐진 request 전체의 pooled p95나 confidence interval로 표시하지 않는다.

## KV 구조에서 유지한 것과 바뀐 것

| 구분 | 우리 v0.10.0 | 우리 v0.10.1 |
|---|---|---|
| Physical layer pool | `[2,pages,128,Hkv,D]` | 동일 |
| Combined tensor API | owner + non-owning slot alias | pool-shaped tensor 직접 반환 |
| Stable page lease | StableKVPageManager | 같은 구현 |
| P/D active metadata | phase-local active length/page table | 같은 구현 |
| Page table host staging | 3-slot pinned ring | 같은 구현 |
| Undercommit | allowUndercommit | allowPoolUndercommit, 같은 256-page 실행 조건 |
| Admission | 실제 free pages/growth에 따라 제한 | 유지 |

Alias는 추가 GPU allocation이 아니다. 순수 KV payload는 양쪽 모두
`2 × 256 × 128 × 8 × 128 × 2 × 28 = 3584 MiB`다.
Pool의 물리 allocation은 init 때 고정이며 request별 page lease가 동적으로 증가/반환된다.
Clean v0.10.1의 최대-profile worst-case 요구량은 이 조건에서1280pages=17920MiB다.
따라서 clean full server를 같은80-slot/KV256 계약으로 그대로 실행하는 비교는 지원되지 않는다.
그 대신 underlying page-table 소스를 직접 비교한다. 이는 full-server 비교의 대체 증거가 아니다.

### 현재 resident memory 차이의 확인 가능한 부분

| 할당/관측 | Old | Current Release | 차이 |
|---|---:|---:|---:|
| P workspace bytes | 301993472 | 301993472 | 0 |
| D workspace bytes | 21548544 | 21548544 | 0 |
| E workspace bytes | 444873728 | 574899200 | +130025472 (~124.002 MiB) |
| KV payload MiB | 3584 | 3584 | 0 |
| Balanced ready/peak sampled MiB | 9237 | 9399 | +162 |

Current gateway startup가 실제 E workspace574899200을 보고한다. Old 값과 과거 engine inspection은
note244/249에 기록되어 있다. New exact-GELU engine은 matched tanh control 대비 약120MiB 크고,
old direct builder 대비 약124MiB 크다. 둘을 혼동하지 않는다.
따라서 +162MiB의 주요 확인 가능 성분은 vision context다. 차감한 약38MiB는 arithmetic residual이며
allocator trace로 특정 모듈에 귀속한 값이 아니다. 정확한 GELU 자체가 이 메모리를 필연적으로
요구한다는 의미도 아니다. Old direct builder도 exact GELU를 더 작은 workspace로 실행했다.

## 독립 KV 측정의 범위

Active view: BS1/8/32/64 × persistent off/on × churn off/on, warmup100+sample1000.
Page table: BS1/8/32/64 × dirty0/1/all, 같은 warmup/sample, clean/current 각5회 교대.
모두 실제 큰 KV payload나 attention kernel은 실행하지 않는다. 후자는16 update마다 stream을
동기화하고 마지막 K/V device page IDs를 검증한다. Throughput 개선을 직접 환산하지 않는다.
실제 HTTP와 isolated GPU 측정을 겹치지 않는다.

## 대표 smoke에서 확인한 것

두 workload 모두 current V0/V1/V2 output hash가 동일했다. Old와의 hash identity는 별도다.
Balanced V2: current4483.20 vs old4454.09 token/s, 각1회.
Multi-image V2: current295.62 vs old324.84 token/s, 각1회.
따라서 Release가 text 회귀를 크게 줄인다는 신호는 있지만 VLM 회귀 전체를 해결했다고
결론 내리지 않는다. 최종 판단은 아래 full12 repeat 결과를 사용한다.

## 남은 해석 조건

- Full12를 마친 뒤에만 전체 정책 순위와 회복 범위를 판단한다.
- KV256을 유지한 채 남는 전체 peak 차이는 engine/context/vision/driver allocation을 분리해야 한다.
- Multi-image의 old V2는 old V0/V1과도 token trajectory가 다르므로 같은 입력/길이라고
  순수 scheduler-only comparison으로 부르지 않는다.
- Legacy eligibility off/on의 단독 인과 효과는 이번 version comparison만으로 확인되지 않는다.
- Export/model 변경은 하지 않았다. 이번 검증은 이미 export→build→inference 검증된 engine의
  runtime/build-configuration 비교이며 새 모델 전체 검증으로 주장하지 않는다.
