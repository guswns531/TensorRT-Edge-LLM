# origin/main 대비 disaggregation 작업 계획

## 목적

이 브랜치는 `origin/main` 대비 Jetson Thor 환경에서 LLM 추론의 `in-GPU disaggregation` 실험을 진행하고 있습니다. 핵심 목표는 단일 GPU 안에서 `encoding/prefill`과 `decode`를 분리 실행해 end-to-end latency와 request throughput을 개선하는 것입니다.

기준 비교:

- base: `origin/main` (`8fe7fe102e`)
- current: `guswns531/main-diff-plan` (`b2b0ef7`)

## diff 요약

`origin/main...HEAD` 기준:

- 10개 커밋 추가
- 18개 파일 변경
- `4339 insertions`, `631 deletions`

주요 변경 파일:

- runtime:
  - `cpp/runtime/llmInferenceDisaggregationRuntime.{h,cpp}` 신규 추가
  - `cpp/runtime/llmEngineRunner.{h,cpp}` 대규모 수정
  - `cpp/runtime/linearKVCache.{h,cpp}` slot-aware API 추가
- example / benchmark:
  - `examples/llm/llm_inference.cpp`
  - `examples/llm/llm_benchmark.cpp`
  - `examples/llm/CMakeLists.txt`
- 문서 / 운영:
  - `README.md`
  - `remote-build-by-sha.sh`

## 지금까지 한 일

### 1. disaggregation runtime 도입

- `LLMInferenceDisaggregationRuntime`를 추가해 `multimodal -> prefill -> decode` 3-stage worker pipeline을 구현
- `llm_inference`, `llm_benchmark`에 `--disaggregation` 경로를 연결
- `--eagle`과의 상호배타 처리 등 실행 조건 정리

### 2. 동시 실행 안전성 보강

- 전역 실행 직렬화 대신 slot 기반 KV cache 할당 구조로 변경
- `LinearKVCache`에 `slotOffset` 기반 API 추가
- `LLMEngineRunner`에서 prefill/decode execution context memory 분리
- prefill/decode KV cache binding 경로 분리
- 요청별 `StageContext`로 tensor/state를 분리해 공유 상태 경합 감소

### 3. 안정화 작업

- 비동기 요청 수명 문제를 피하기 위해 request deep-copy 적용
- shutdown 순서 정리 및 constructor 예외 cleanup 추가
- `finalizeContext()` idempotent 처리
- `LinearKVCache::resetForNewSequences()` 복사 크기 버그 수정
- disaggregation 경로에서 decode CUDA graph를 기본 비활성화해 안정성 우선으로 운용

### 4. Jetson Thor 전용 TPC 분할 실험

- `--tpcCount`를 decode 전용 TPC 개수로 해석하는 경로 추가
- decode stream과 prefill/multimodal stream에 서로 다른 TPC mask 적용
- Jetson Thor의 GPC/TPC 토폴로지를 runtime 내부에서 사용하도록 구현

### 5. 운영 편의와 재현성 보강

- `--quiet`, `--disaggDecodeCudaGraph` 옵션 추가
- `remote-build-by-sha.sh` 추가
  - 현재 HEAD를 원격 worktree에서 같은 SHA로 빌드
  - branch/worktree 재사용 방식으로 remote build flow 복원
- README에 실험 명령, 제약, 결과를 정리

## 현재 진행 상태 판단

### 구현 상태

- 기능 구현: 많이 진행됨
- runtime 안정화: 1차 완료
- benchmark/운영 스크립트: 사용 가능한 상태
- 성능 최적화: 아직 미완료

### 지금 확인되는 결론

README 기준 최신 정리에서는 아래가 분명합니다.

- disaggregation 자체는 구현되어 있고 benchmark까지 연결됨
- 동시성/종료/crash 성격의 주요 버그는 한 차례 정리됨
- 하지만 최근 3회 재측정 결과에서는 baseline 대비 E2E 성능이 오히려 악화됨
  - Baseline request mean: `1194.996 ms`
  - Disagg(graph off): `1439.764 ms`
  - Disagg(graph on): `1536.883 ms`
- 즉, 현재 병목은 "구현 여부"가 아니라 "실제 성능 이득을 만드는가"로 이동한 상태입니다

## 우리가 지금 하고 있는 일

현재 브랜치의 실질적인 작업 주제는 아래와 같습니다.

1. Jetson Thor에서 LLM 추론 disaggregation runtime을 구현하고,
2. request 동시성/수명/종료 안정성을 확보한 뒤,
3. TPC partitioning 및 decode CUDA graph를 포함한 조합을 벤치마크로 검증하고,
4. 실제로 baseline보다 좋은 E2E 성능이 나오는 구성을 찾는 것

즉, "runtime을 만드는 단계"는 거의 지나갔고, "성능 가설을 검증하고 병목을 제거하는 단계"에 들어가 있습니다.

## 앞으로 해야 할 일

### 우선순위 1. front stage 직렬화 병목 확인

README에 따르면 현재 구조는 아래 제약을 가집니다.

- front는 요청 단위로 `encoding -> prefill` 완료 후 다음 요청으로 진행
- back(decode)은 front와 병렬

이 구조면 decode와 front 간 overlap은 생기지만, 여러 요청의 front를 충분히 파이프라이닝하지 못해 request-level throughput이 제한될 수 있습니다.

해야 할 일:

- stage별 queue depth / idle time / wait reason 로깅 추가
- `multimodal`, `prefill`, `decode` 각 worker의 실제 overlap 비율 측정
- request mean 악화의 주원인이 `front serialization`인지 먼저 확정

### 우선순위 2. TPC 분할 정책 재검증

현재 `--tpcCount`는 decode 전용 TPC 개수로 해석됩니다. 그런데 README sweep 결과만 보면 `tpcCount`에 따라 성능 편차가 크고, 최근 재측정에서는 baseline보다 계속 불리합니다.

해야 할 일:

- 동일 입력/동일 count에서 baseline vs disagg를 다시 측정해 문서 수치 재검증
- `tpcCount` sweep를 다시 수행해 최적점이 재현되는지 확인
- decode 쪽에 너무 많은 TPC를 몰아 front를 약화시키는지 분석
- `decode 전용`, `front 전용` 분배 기준을 고정값 대신 heuristic으로 바꿀지 검토

### 우선순위 3. decode CUDA graph 경로 판단 정리

지금 상태는 다음과 같습니다.

- 안정화 과정에서 기본 비활성화
- 옵션으로 다시 켤 수는 있음
- 최근 결과에서는 graph on이 더 느림

해야 할 일:

- graph capture/replay가 실제로 launch overhead를 상쇄하는지 계측
- disaggregation 경로에서 graph 사용 조건을 더 좁히거나 완전히 제거할지 결정
- 유지 가치가 낮으면 실험용 옵션으로만 남기고 기본 경로에서 제외

### 우선순위 4. system prompt KV cache 재사용 정책 재정리

README 초반 설명에는 system prompt KV cache 재사용이 소개되어 있지만, 안정화 커밋 설명에는 disaggregation runtime에서 해당 경로를 비활성화한 내용이 있습니다.

이 부분은 문서와 실제 동작 사이에 해석 차이가 생길 수 있습니다.

해야 할 일:

- 현재 disaggregation runtime에서 reuse/save가 실제로 비활성화인지 코드 기준으로 재확인
- 맞다면 README 설명을 "기본 runtime 기능"과 "disaggregation 경로 현재 상태"로 분리
- 성능 최적화 후 재도입할지, 당분간 비활성 상태로 둘지 결정

### 우선순위 5. 재현 가능한 실험 체계 정리

이미 `remote-build-by-sha.sh`는 들어와 있으므로, 다음은 측정 재현성을 더 높이는 단계입니다.

해야 할 일:

- benchmark preset을 스크립트화해 baseline/disagg/disagg+graph를 일관되게 반복 실행
- 결과 JSON과 요약 표를 자동으로 저장
- README 수치가 언제, 어떤 SHA, 어떤 옵션에서 나온 것인지 연결

## 추천 실행 순서

1. 현재 SHA에서 baseline/disagg/disagg+graph를 동일 조건으로 다시 측정
2. stage별 대기 시간과 overlap 계측 로그 추가
3. `tpcCount` sweep 재실행
4. decode CUDA graph 유지 여부 결정
5. 문서(README)와 실제 runtime 동작 차이 정리
6. 그 다음에만 추가 구조 변경 진행

## 열려 있는 리스크

- Jetson Thor CUDA 13.x 내부 stream mask 오프셋(`0x54c`) 의존성
- disaggregation의 구조적 복잡도가 baseline 대비 디버깅 비용을 키움
- README 일부 설명과 현재 runtime 동작이 완전히 일치하지 않을 가능성
- 성능 개선 없이 코드 경로만 복잡해질 위험

## 한 줄 결론

현재 브랜치는 `disaggregation runtime 구현 + 1차 안정화`까지는 상당히 진척됐지만, 가장 중요한 목표인 `baseline 대비 E2E 성능 개선`은 아직 달성되지 않았습니다. 다음 단계의 핵심은 새로운 기능 추가가 아니라, 병목 계측과 실험 재현을 통해 어떤 구조가 실제로 성능을 해치고 있는지 먼저 확정하는 것입니다.
