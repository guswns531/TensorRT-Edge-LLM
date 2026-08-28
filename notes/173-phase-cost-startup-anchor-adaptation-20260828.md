# Phase cost startup anchor adaptation

## 목적

동일한 engine/profile prior라도 GPU clock, power state, driver 및 node-local contention에 따라 절대 시간이
달라진다. workload 이름이나 정적 serving mode를 추가하지 않고, controlled startup warmup과 명시적인
calibration epoch의 action-fidelity CUDA 관측으로 prior를 보정한다.

## 동작

각 E/P/D/overlap 관측에 대해 같은 action key의 fleet prior를 먼저 찾고, 없으면 build prior를 찾는다.

```text
ratio = observed CUDA makespan / prior makespan
```

phase별 최근 ratio를 bounded window에 보관한다. 최소 표본 수에 도달하면 median ratio를 scale로 사용하고,
MAD 기반 dispersion을 prior uncertainty의 하한으로 사용한다.

```text
scaled prior = configured scale × node anchor scale × portable prior
robust cost  = scaled median + max(prior uncertainty, anchor dispersion)
```

- 기본 최소 표본: phase당 2개
- 기본 window: phase당 16개
- 허용 ratio: 0.25~4.0
- 적용 scale clamp: 0.5~2.0
- prior가 없거나 표본이 부족하면 scale 1.0
- 충분한 exact-key local 표본이 쌓이면 기존 Oracle 우선순위에 따라 local estimate가 prior를 대체

## 안전성

- memory feasibility에는 cost scale을 사용하지 않는다.
- unknown overlap을 eligible로 바꾸지 않는다.
- cancellation/shape/action fidelity gate를 통과해 Oracle에 들어온 표본만 anchor로 사용한다.
- production measurement epoch의 관측은 exact-key local window만 갱신하고 phase-wide scale은 바꾸지 않는다.
- implausible ratio는 scale을 오염시키지 않고 폐기한다.
- bundle이 비활성인 기본 경로에서는 anchor 계산도 일어나지 않아 기존 decision을 유지한다.

## 관측

startup warmup 종료 로그와 `PHASE_METRIC`에 phase별 scale/sample count가 기록된다.

```text
TRT_EDGELLM_PHASE_COST_ANCHOR_MIN_SAMPLES
TRT_EDGELLM_DISABLE_PHASE_COST_AUTO_ANCHOR
TRT_EDGELLM_PHASE_WRITE_BUILD_COST_BUNDLE
```

`TRT_EDGELLM_PHASE_WRITE_BUILD_COST_BUNDLE`을 지정하면 server가 controlled shape warmup의 raw CUDA 표본을
runtime action key 그대로 build bundle로 원자적으로 저장한다. 따라서 별도의 production trace parser 없이도
`build → calibration → portable prior` 경로를 실행할 수 있다.

## 검증

- `unitTest`의 phase runtime 집중 회귀: `232/232`
- `llm_phase_context_smoke` TensorRT/CUDA build 통과
- anchor minimum-sample, robust scale, implausible-ratio rejection, reset 및 production-epoch isolation 단위 테스트
- runtime boundary validator 통과

Cosmos Reason2-2B tied engine에서 실제 두-process startup 검증도 수행했다.

1. 첫 process의 controlled warmup에서 P 3개/D 5개 key, 총 30개 CUDA observation을 가진 build bundle 생성
2. 두 번째 process에서 이를 `compatible` prior로 로드
3. 같은 warmup이 `P scale=1.000, samples=11`, `D scale=1.002, samples=16`으로 자동 수렴

E/overlap 표본은 text-only startup이 해당 action을 실행하지 않았기 때문에 0이며 scale 1.0을 유지했다. 이는
unknown action을 추측하지 않는 의도된 동작이다. 검증 산출물은 프로젝트 바깥의 `.local`에만 저장했다.

다음 단계는 timestamp/TTL과 sustained drift를 분리해, 오래되거나 계속 어긋나는 prior를 local-only 상태로
강등하는 것이다.
