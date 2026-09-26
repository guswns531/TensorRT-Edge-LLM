<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# 335. Owner-aware dynamic runtime: workload 규칙 없이 개선하기

## 목표와 평가 기준

같은 모델/엔진/요청 계약에서 두 모델의 12-workload 모두를 개선하는 것이 목표다. 모든 지표에서
vLLM 우위를 미리 보장하지 않는다. 처리량 개선과 TTFT/TPOT/E2E 평균·p95 회귀를 따로 보고하며,
현재 331/334의 exact-output, multi-image 변동, unrestricted sanitizer 제한을 계속 유지한다.
현재 기준 source는 `9db00f7`; 대응 C++ binary는 `e8164e0`이다. 기존 full24 ×3은 `43c680a`이며
최신 binary 결과로 바꾸어 표기하지 않는다.

워크로드 이름, 모델 이름, JPEG 파일명, 부하 구간별 정책 분기를 추가하지 않는다. 다음은 구분한다.

- correctness: owner lifetime, GPU completion, context single-inflight, DAG dependency.
- capability: engine batch/profile/shape, 모델이 선언한 KV donor 관계.
- observation: 실제 요청 크기, ready work, CUDA service time, 사용 중인 memory.
- policy: 어떤 실행을 우선할 것인가. 관측 기반이어도 최적성 보장은 없고 fairness 선택은 남는다.

## 실행 순서

### A. 물리 KV owner와 logical layer 분리

1. `KVCacheManager`는 donor map을 검증하고 canonical physical owner만 할당한다.
2. logical indexing과 page lease를 유지한다. getter, compact, capture/restore, partial snapshot이
   동일 owner map을 사용하며 물리 공간을 두 번 쓰지 않는다.
3. page당 bytes와 전체 allocated bytes는 unique-owner 실제 shape로 계산한다. heterogeneous head
   크기를 단일 `numLayers * headDim`으로 계산하지 않는다.
4. non-sharing 모델은 identity mapping이며 변경이 없다. Gemma 계산상 1008→432MiB, 576MiB 절감은
   구현 전 예상치이고 실제 peak 감소량으로 주장하지 않는다. KV 용량/정밀도는 축소하지 않는다.

### B. 이미지 scratch를 실제 수요와 GPU lifetime에 맞춤

1. 최대 4096 image capability를 유지한다. 실제 raw/resized 크기에서 필요한 byte를 계산한다.
2. 용량이 부족할 때만 성장하고 충분하면 재사용한다. hot path의 동기 allocate/free를 피한다.
3. 명시 stream ordering과 마지막 GPU consumer의 lifetime을 보장한다.
4. allocated/high-water와 가능한 allocator reserve를 구분한다. 기존 이미지 집합의 약72MiB는
   예상치이며 큰 이미지가 들어오면 달라질 수 있다.

### C. 메모리 admission과 resident decode 서비스 보호 분리

333의 M-RoPE-only gate 완화는 vision 처리량을 올렸지만 text E2E mean을 약345% 악화시켰다.
단순히 gate를 제거하거나 slab 수를 늘려서 default로 바꾸지 않는다.

먼저 현 scheduler의 measured service/ready-work 입력으로 resident D 지연을 표현할 수 있는지 확인한다.
새 controller나 magic timer를 추가하기 전에 기존 global selector의 candidate 비교에 공통 정보를
연결한다. 메모리 안전과 service preference는 별개의 판정이어야 한다. 확실한 동일-state 검증과
양 모델 회귀 검증이 없는 정책은 default로 승격하지 않는다.

### D. 검증 및 promotion

- owner mapping range/cycle/shape, deterministic allocation, alias, copy/restore를 단위 검증.
- Gemma d256/d512, P1/P8/D1/D24, graph on/off, cancel/readmission, prefix snapshot 확인.
- Cosmos identity-map 회귀 확인.
- 동일 engine을 재사용한다. 모델/export 변경이 없으므로 기존 export→build→inference provenance를
  이어받되 새 runtime SHA를 별도 기록한다. engine 재생성이 필요하면 순서를 다시 검증한다.
- 대표 mixed/vision-heavy/multi-image와 text를 먼저 screen하고 전체12×2로 확장한다.
- 변경별 before/after를 분리한다. current pointer와 frozen baseline을 자동 변경하지 않는다.
- 동일 workload/output contract면 frozen vLLM을 재사용한다. Cosmos는 corrected raw-derived summary를
  사용하고 이전 flattened summary를 primary reference로 사용하지 않는다.
- incomplete cell, OOM, greedy divergence를 누락하지 않는다. 단일 실행은 screening이며 반복 확증 아님.

## 증거와 범위

- [331 runtime 재검증](331-runtime-contract-memory-and-full24-revalidation-20260926.md)
- [332 메모리 코드 감사](332-gemma-owner-allocation-memory-audit-20260926.md)
- [333 admission 반례](333-encoder-admission-mrope-lifetime-fix-20260926.md)
- [334 workspace 비교](334-workspace-memory-mode-final-screen-20260926.md)

이 문서는 실행 계획이며 구현/성능 완료 보고가 아니다. 결과와 실제 지원 범위는 후속 보고서에 기록한다.
