# v0.10.0 Current 프로젝트 리팩터링 최종 승격 결과

작성일: 2026-08-28

## 결론

v0.10.0 기반 Current의 phase-serving 리팩터링 R0--R7을 완료했다. 실행 의미나 hot object layout을
바꿔 성능 회귀가 발생한 변경은 승격하지 않았고, binary identity 또는 실제 HTTP trace gate를 통과한
변경만 branch에 남겼다.

최종 선택 결과는 다음과 같다.

- core archive, TensorRT plugin과 server executable은 R2b 이후 R6까지 byte-identical이다.
- 최종 실제 request gate는 12 workload x 3회이며, 경계선 세 workload는 별도 3회 재측정했다.
- 선택한 최종 결과는 12/12 repeat exact token identity를 만족한다.
- Stage 7 golden 대비 처리량 변화는 `-0.28%`에서 `+4.18%` 범위다.
- latency p95 의심 항목은 독립 재측정 후 모두 `+3%` 이내다.
- peak VRAM high-watermark는 `9,485 MiB`로 Stage 7보다 `8 MiB` 높고 64 MiB gate 안이다.
- 동일 trace의 cached fresh vLLM 대비 처리량은 12/12 workload에서 `+1.56%`에서 `+27.19%`다.

## 단계별 판정

| 단계 | 변경 | 판정 | 성능 검증 |
|---|---|---|---|
| R0 | v0.10.0 feature-owner/source-order manifest | 승격 | runtime 미변경 |
| R1 | phase source/object 순서 명시 | 승격 | binary identity, 192 tests, 12 trace |
| R2a | scheduler option을 별도 translation unit으로 이동 | 거절/복구 | text-heavy `-1.12%` |
| R2b | scheduler option statement fragment | 승격 | 12 trace gate 통과 |
| R2c | server/admission 두 번째 fragment | 거절/복구 | long-prefill `-3.44%` 재현 |
| R3 | action plan과 online cost declaration 분리 | 승격 | binary identity, 145 tests |
| R4 | ready/deadline/ownership value boundary | 승격 | binary identity, 145 tests |
| R5 | production/runtime 대 lab contract 자동 검증 | 승격 | build input 미변경 |
| R6 | canonical `runtime/phase` declaration tree | 승격 | binary identity, 200 tests |
| R7 | 전체 build/unit/실제 HTTP gate | 승격 | 아래 최종 결과 |

중요한 원칙은 "리팩터링이므로 성능이 같을 것"이라고 가정하지 않은 것이다. R2a/R2c처럼 함수 내용이
같아도 translation unit이나 executable layout이 바뀌어 host submission timing이 달라진 변경은 실제
인접 A/B 결과로 거절했다.

## 최종 프로젝트 경계

```text
cpp/runtime/phase
├── mechanism
│   └── phaseReadySnapshot.h
├── policy
│   ├── phaseDeadline.h
│   ├── phaseGlobalCostModel.h
│   └── phaseGlobalScheduler.h
├── ownership
│   └── phaseOwnershipHorizon.h
└── execution
    └── phaseActionPlan.h

cpp/runtime/scheduling
├── phaseQueueScheduler.*             # 기존 mechanism과 hot implementation
├── phaseGlobalScheduler.cpp          # R1의 명시적 object 순서 유지
├── independentPhase*.*               # context/async mechanism
├── phaseThreeCoordinator.*           # E -> P -> D DAG transition
└── phase*.h                          # 기존 include용 forwarding header

examples/llm
├── llm_phase_context_smoke.cpp        # composition/server compatibility root
└── phaseSchedulerOptions.inc          # 같은 TU 안의 scheduler option fragment

benchmarks/phase_serving
├── manifests/phase_runtime_contract.json
├── manifests/v010_feature_owners.json
├── manifests/phase_source_order.json
└── validate_runtime_boundaries.py
```

production scheduler/runtime은 `std::getenv`, `TRT_EDGELLM_*`, text-heavy/vision-heavy 같은 workload label을
직접 읽지 않는다. engine capability, capacity, request SLO, ready/completion state, ownership horizon과
online CUDA-event cost가 값으로 주입된다. 환경변수 호환 파싱은 example composition root의 책임이다.
`validate_runtime_boundaries.py`가 이 의존 방향과 v0.10.0 product diff의 feature ownership을 자동 검사한다.

## Binary와 테스트

R6 최종 build hash는 다음과 같다.

| 산출물 | SHA-256 |
|---|---|
| `libedgellmCore.a` | `eaca8c2fa8fe79b6c9cdd8d03ed618c098479972418c24b7b51984e2df822228` |
| `libNvInfer_edgellm_plugin.so.1.0` | `b78c59b1ee8f760062eb22bddd152dc76a022945b43b7bd45f2d8f7d932f7a70` |
| `llm_phase_context_smoke` | `b6d7c6f30cbc9bd4a939608bdbcd5187394ae9765f1b3f4600411146c01feae7` |

phase 집중 suite는 `200/200` 통과했다. 전체 C++ suite는 1,131개 중 1,087개 통과, 42개가 현재 SM86/CuTe
DSL build 조건 때문에 skip됐고, YaRN RoPE와 M-RoPE accuracy 두 개가 실패했다. 두 실패는 phase 변경
source와 무관하며 R0 golden 이후 해당 kernel/test source diff가 없다. Release `-O3` binary에서 오차가
각각 약 `0.00121`, `0.00150`로 tolerance를 넘지만, 동일 CUDA 13.3/TensorRT 11 컨테이너의 기존 non-`-O3`
binary에서는 두 테스트가 모두 통과했다. 따라서 phase 리팩터링 회귀가 아닌 build-mode 수치 gate로
별도 추적하며, 결과를 전체 green으로 과장하지 않는다.

## 최종 12-workload 결과

모든 latency 단위는 ms다. long-prefill, text-heavy와 wave/drain은 전체 묶음 뒤 실행한 독립 3회 결과를
사용했고, 나머지는 전체 12 x 3 결과를 사용했다.

| workload | tok/s | vs Stage 7 | vs vLLM | TTFT mean / p95 | TPOT mean / p95 | E2E mean / p95 | peak MiB | exact |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| short | 2,515.77 | +0.63% | +26.83% | 84.52 / 166.14 | 13.39 / 22.21 | 327.88 / 407.59 | 9,313 | yes |
| balanced | 4,597.96 | +0.96% | +6.10% | 67.70 / 164.00 | 11.93 / 13.21 | 1,084.96 / 1,685.21 | 9,313 | yes |
| decode-heavy | 5,256.11 | -0.25% | +5.85% | 65.54 / 179.54 | 10.60 / 11.15 | 2,805.01 / 4,273.61 | 9,313 | yes |
| long-prefill | 1,199.61 | +0.59% | +6.12% | 2,047.57 / 2,683.42 | 26.58 / 31.38 | 4,333.65 / 6,156.77 | 9,313 | yes |
| bimodal | 1,907.38 | -0.01% | +3.65% | 1,918.94 / 4,025.60 | 18.24 / 29.57 | 4,422.20 / 8,978.60 | 9,313 | yes |
| text-heavy | 1,975.99 | +0.41% | +20.87% | 319.23 / 1,042.49 | 24.92 / 39.00 | 1,604.85 / 1,707.47 | 9,427 | yes |
| mixed | 1,145.15 | +0.94% | +24.27% | 705.40 / 2,113.13 | 31.37 / 41.42 | 2,212.00 / 2,494.12 | 9,431 | yes |
| vision-heavy | 688.82 | -0.28% | +18.93% | 1,388.95 / 3,182.57 | 27.37 / 37.23 | 2,481.65 / 3,512.76 | 9,431 | yes |
| poisson | 1,991.00 | +0.68% | +10.61% | 207.25 / 783.92 | 21.37 / 39.91 | 1,564.95 / 1,999.04 | 9,395 | yes |
| wave/drain | 97.34 | +0.19% | +1.56% | 238.48 / 359.89 | 9.49 / 11.35 | 538.77 / 594.47 | 9,485 | yes |
| multi-image | 310.99 | +4.18% | +27.19% | 206.91 / 299.47 | 9.49 / 12.99 | 501.67 / 511.84 | 9,479 | yes |
| late-vision D24 | 2,543.13 | -0.07% | +7.79% | 114.83 / 457.11 | 9.28 / 9.34 | 1,443.98 / 1,813.40 | 9,351 | yes |

전체 12 x 3 묶음에서 wave/drain의 첫 반복만 다른 greedy hash를 냈고 뒤의 두 반복은 같았다. 즉시 분리한
3회 재측정에서는 세 hash가 모두 동일했으며 semantic pass도 전체 반복에서 100%였다. 이 재측정 결과를
승격 gate에 사용하되, 최초 변동도 삭제하지 않고 아래 raw result에 보존한다.

## vLLM latency 비교

model, trace SHA, arrival/output contract와 vLLM 설정이 바뀌지 않았으므로 Note 169에서 검증한 fresh 3회
vLLM 결과를 재사용했다. 같은 workload를 매 리팩터링 단계마다 다시 실행해 GPU 시간을 소비하지 않는다.

- 처리량: Current가 12/12에서 높다. 최소 이득은 wave/drain `+1.56%`, 최대는 multi-image `+27.19%`다.
- TTFT mean: long-prefill과 bimodal은 vLLM이 낮고, 나머지 10개는 Current가 낮다.
- TTFT p95: bimodal만 Current가 높고, 나머지 11개는 Current가 낮다.
- TPOT p95: Current가 12/12에서 낮다.
- E2E mean과 E2E p95: Current가 12/12에서 낮다.

남은 명확한 성능 약점은 bimodal request ordering이다. Current가 vLLM보다 처리량 `+3.65%`, TPOT p95가
낮고 E2E mean/p95도 낮지만, TTFT mean/p95는 vLLM보다 높다. 이 문제를 해결할 때 workload 이름 기반
튜닝을 다시 넣지 않고 request-level remaining critical-path/slack ordering만 개선해야 한다.

## Raw result 위치

- 전체 12 x 3: `.local/project-refactor-20260828/r7-final-refactor-12x3`
- 의심 항목 독립 3 x 3: `.local/project-refactor-20260828/r7-suspects-3x`
- golden Stage 7: `.local/current-only-cleanup-20260828/stage7-final-current-12x3`
- cached fresh vLLM 기준과 상세 latency: `notes/169-current-only-stage-gates-20260828.md`

## 최종 유지/보류 결정

유지한다.

- deterministic source ordering
- same-TU scheduler option fragment
- immutable action/cost/ready/deadline/ownership value boundaries
- production/runtime boundary validator
- canonical `runtime/phase` declaration tree와 기존 include forwarding headers

보류한다.

- scheduler/server option의 별도 translation unit 이동
- hot `.cpp` source path 이동
- server/admission 두 번째 include fragment
- compatibility forwarding header 제거

보류 항목은 코드 미완료가 아니라 측정된 executable-layout 회귀를 피하기 위한 의도적 경계다. hot loop를
독립 object/ABI로 먼저 안정화하지 않고 단순히 파일 수나 line count만 줄이는 것은 현재 RTX 3080 serving
성능 계약에 맞지 않는다.
