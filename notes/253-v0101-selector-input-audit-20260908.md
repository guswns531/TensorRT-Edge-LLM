# Selector 입력과 preview frontier 구분

## 범위

2026-09-08, `codex/v0101-phase-forward-port`, 기준 커밋 `184a1cb`.
note252의 D-ready 대기 분석에 이어 selector의 입력 계약을 점검했다.
이번 변경은 감사 API와 단위 테스트이며, serving policy 및 엔진은 변경하지 않았다.
새 HTTP 성능 결과는 없다. note251의 비계측 반복 결과를 계속 성능 기준으로 사용한다.

## 확인한 두 가지 차이

`cpp/runtime/scheduling/phaseThreeCoordinator.cpp::dispatchGlobalAction()`:

1. `mServer.previewGlobalAction()`은 P/D 단계에서 선택한 후보 하나를 반환한다.
   `lastGlobalPreviewCandidates()`는 해당 단계의 후보 전체를 반환한다.
   E와 비교하는 최종 `candidates`에는 선택된 `pd` 하나와 E 및 생성 가능한 E-overlap 후보가 들어간다.
   따라서 P/D preview에 존재하는 D 후보가 최종 selector에도 들어간다고 가정할 수 없다.
2. 최종 비교 전에 `pd->protectedCompletions`에 encoder 요청의 E→P 완료 지연을 추가한다.
   반대로 encoder 후보에는 P/D 보호 요청이 E 실행으로 지연되는 비용을 추가한다.
   같은 candidate ID여도 preview와 final의 보호 제약은 같지 않다.

기존 `unifiedCandidateFrontier`는 P/D preview 전체로 시작한 뒤 ID가 없는 최종 후보만 추가한다.
ID가 같은 후보의 final 보호 제약으로 교체하지 않으므로, 기존 snapshot의 보호 값은
실제 최종 selector 입력과 다를 수 있다. 또한 `recordUnifiedDecision()`은 snapshot의
`predictedSloViolationUs`를 채우지 않는다. JSON의 `max_slo_violation_us=0`은 안전성 증거가 아니다.

이것은 우리 phase coordinator/telemetry 계층의 문제다. upstream v0.10.1의 KV cache나
vision activation 문제와 구분해야 한다. 다만 이 관찰만으로 hierarchical 선택이 실제 성능 회귀의
원인이라고 결론 내릴 수는 없다. 최종 입력과 선택 사유를 계측한 뒤 판단해야 한다.

## 구현

`cpp/runtime/phase/policy/phaseGlobalScheduler.h`에 `PhaseGlobalCandidateAudit`를 추가했다.
`PhaseGlobalScheduler::select(candidates, audit)`의 두 번째 인자는 선택적이며 기본값은 null이다.

각 실제 입력에 대해 다음을 반환한다.

- candidate ID와 hard feasibility
- deadline guard와 protected-completion uncertainty를 포함한 계산된 SLO 위반 시간
- exploration/safe/all-late 단계에서 frontier에 포함됐는지
- dominance pruning으로 제거됐는지

선택 이유, 선택 index 및 최종 위반값은 기존 `PhaseGlobalDecision`을 함께 사용한다.
감사 배열은 호출마다 비우고 실제 입력 순서와 동일하게 채운다. 빈 입력에서도 이전 호출의
결과가 남지 않는다. 감사 비활성 경로에서는 배열 할당과 추가 위반값 계산을 하지 않는다.
점수, exploration, all-late recovery 및 우선순위 로직은 변경하지 않았다.

## 검증 대상

`unittests/cpp/runtime/scheduling/phaseGlobalSchedulerTest.cpp`에 세 테스트를 추가했다.

1. 감사 on/off의 선택 결과 일치, hard-infeasible/late 후보 구분, 빈 입력 초기화.
2. protected completion의 uncertainty와 deadline guard가 위반값에 반영되는지.
3. 같은 ID의 preview는 안전하지만 final의 E 보호 제약 추가 후 위반값이 650µs가 되는 사례.

빌드 환경은 기존 TensorRT 26.06 컨테이너, `TRT_PACKAGE_DIR=/opt/tensorrt`와
`LD_LIBRARY_PATH=/opt/tensorrt/lib`이다. 기존 CMake cache가 `/local`에서 생성됐으므로
호스트 `.local`을 컨테이너 `/local`에 마운트해야 한다.

검증 완료: `unitTestRuntime` 빌드 성공, `PhaseGlobalSchedulerTest.*` **38/38 통과**,
변경 파일의 pre-commit 검사 통과. 테스트 실행에는 `--gpus all`로 링크된 `libcuda.so.1`을
제공했다. 이는 HTTP inference 또는 성능 gate를 실행했다는 의미가 아니다.

## 아직 남은 연결 및 실험

후속 구현·실험은 [note254](254-selector-stage-attribution-and-regression-20260908.md)에 기록했다.
아래 항목은 이 문서 작성 시점의 계획이며, HTTP 감사 연결은 후속 커밋 `a7480e6`에서 완료했다.

감사 API는 아직 HTTP telemetry에 연결하지 않았다. 따라서 이번 변경만으로 note252의
실제 요청별 선택 이유가 밝혀진 것은 아니다.

1. 실제 final select 호출에서 감사 결과와 `PhaseGlobalDecision`을 함께 받아 기록한다.
   preview union과 별도 필드로 보존하고, 이후 transition override가 있다면 별도로 표시한다.
2. full snapshot 없이도 선택 입력 감사와 producer timeline을 켤 수 있는 경량 계측을 연결한다.
3. 같은 엔진/요청/calibration으로 Vision-heavy를 다시 실행해 D-ready 시 E/P 선택을
   최종 후보 부재, hard feasibility, SLO protection, exploration, all-late recovery로 분류한다.
4. 원인이 확정되기 전에는 D 우선순위나 workload-specific gate를 추가하지 않는다.
5. 정책을 변경하는 경우 비계측 반복 및 전체 12-workload 회귀 gate로 검증한다.
   계약이 같다면 frozen vLLM을 재사용하고 TTFT/TPOT/E2E mean·p95도 함께 비교한다.
