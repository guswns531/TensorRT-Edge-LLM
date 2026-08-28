# Phase cost knowledge plane 1차 구현

## 구현 결과

첨부 설계의 1단계와 2단계를 runtime에 연결하고, 3단계 및 6단계에서 사용할 오프라인 도구를 추가했다.

- versioned `PhaseCostBundle`과 deployment fingerprint
- exact / compatible / shape-only / incompatible 판정
- node-local → fleet → build → analytical fallback 순서의 `PhaseCostOracle`
- E/P/D scheduler가 동일 Oracle을 공유
- hot path에서 파일을 쓰지 않는 background journal/atomic snapshot
- raw observation 기반 build-bundle 생성과 fleet merge
- fingerprint가 맞지 않거나 JSON이 손상되면 prior를 버리고 안전한 기존 fallback 사용

기본 환경에서는 bundle과 journal이 모두 비활성이다. 따라서 이번 변경 자체가 기존 workload별 action이나
batch를 바꾸지 않는다.

## 코드 위치

```text
cpp/runtime/phase/cost/phaseCostOracle.h
cpp/runtime/scheduling/phaseCostKnowledge.cpp
cpp/runtime/scheduling/phaseQueueScheduler.*
cpp/runtime/scheduling/phaseThreeCoordinator.*
examples/llm/llm_phase_context_smoke.cpp
scripts/phase_cost_bundle.py
unittests/phaseCostKnowledgeTest.cpp
tests/python-unittests/test_phase_cost_bundle.py
```

## 검증

- `edgellmCore`와 `llm_phase_context_smoke` TensorRT 11/CUDA 13.3 build 통과
- phase runtime 전체 집중 C++ regression: `228/228`
- cost knowledge 단독 C++ test: `6/6`
- bundle Python test: `2/2`
- runtime boundary validator 통과

## 아직 승격하지 않은 것

- 기존 static D table을 build bundle로 완전히 대체
- 자동 startup anchor 실행과 scale 추정
- TTL 및 drift state
- remote fleet registry transport

이 항목은 cost plane의 기본 비활성 decision identity와 기존 12-workload 성능 gate를 먼저 확인한 뒤
순서대로 승격한다.
