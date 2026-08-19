SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

# TensorRT Edge-LLM v0.10.0 Cosmos 비교

## 결론

2026-08-12에 공개된 [TensorRT Edge-LLM v0.10.0](https://github.com/NVIDIA/TensorRT-Edge-LLM/releases/tag/v0.10.0)을
현재 브랜치의 기반인 v0.9.1과 분리된 worktree에서 빌드하고, `nvidia/Cosmos-Reason2-2B` FP16으로
`export -> build -> inference`를 완주했다.

- v0.10 표준 ONNX engine의 고정-shape prefill은 v0.9.1보다 `2.19--3.14%`, decode는 `6.52--7.08%`
  빨랐다. RTX 3080 SM86에서도 새 FMHA/runtime 변경의 실효가 있다.
- 같은 BS8 real-request 묶음에서 생성 처리량은 `1177.3 -> 1205.4 token/s`, 총 GPU service time은
  `20141.4 -> 19672.1ms`로 `2.33%` 줄었다.
- peak GPU memory는 `7348 -> 7524MiB`, 즉 `176MiB` 증가했다. v0.10을 현재 P8/D64 independent
  runtime에 바로 합치면 현재의 512MiB headroom gate를 먼저 다시 확인해야 한다.
- v0.10 context reuse는 세 반복-prefix 요청 중 두 요청에서 hit했다. `1024/1802` prefill token을 재사용했고
  prefill GPU time은 `84.53 -> 62.56ms`, `25.99%` 감소했다. 별도 peak memory 증가는 없었다.
- 새 ONNX-less direct builder는 artifact를 `3.8GiB -> 37MiB`로 줄이고 실행 peak를 `670MiB` 줄였지만,
  같은 output-128 trace의 generation GPU time이 표준 engine보다 `17.11%` 늘었다. 아직 기본 선택으로 쓰지 않는다.
- v0.10 public runtime은 여전히 한 batch를 prefill부터 종료까지 처리한다. 우리의 continuous admission,
  P8/D64 비대칭 batching, independent prefill/decode TensorRT contexts와 phase overlap을 대체하지 않는다.
  현재 balanced 처리량 `4413.4 token/s`와 v0.10 public BS8의 `1205.4 token/s`를 방향성 있게 비교하면
  격차는 기존 `3.78x`에서 약 `3.66x`로만 좁아진다.

따라서 다음 기준점은 **v0.10 표준 exporter/builder + v0.10 context reuse + 현재 independent scheduler**다.
direct builder는 decode 회귀와 정확성 gate를 해결한 뒤 별도 opt-in으로 재평가한다.

## 재현 환경

| 항목 | 값 |
| --- | --- |
| current | `v0.9.1-87-g7041733` |
| clean release | `v0.10.0`, commit `71dd1bae032e70771265917ec74d3ff4cad07a10` |
| GPU | RTX 3080 10GB, driver 610.43.02 |
| model | `nvidia/Cosmos-Reason2-2B`, FP16 weights/KV |
| engine shape | max batch 8, max input 1024, max KV 2048, KV page pool 128 |
| runtime | TensorRT 11.0.0.114, CUDA 13.3 |
| compiler | GCC 13.3.0 |
| runtime image | `nvcr.io/nvidia/tensorrt@sha256:7cd94ee9...b96ceac3` |
| export image | `nvcr.io/nvidia/pytorch@sha256:1dc787f5...a8fecff8` |
| worktree | `.local/upstream-v010` |

v0.10 CMake 기본값은 여러 SM용 XQA/FMHAs를 생성해 10GB 실험 호스트에서 빌드 시간과 디스크를 크게 썼다.
실험 binary는 임시로 `CMAKE_CUDA_ARCHITECTURES=86`을 존중하게 한 뒤 SM86만 빌드했고, 소스 수정은 빌드 후
되돌렸다. CuTe DSL 4.6.1로 SM86 FMHA AOT variant 25개를 생성했다.

실행할 때는 반드시 다음처럼 release와 plugin을 한 쌍으로 고정해야 한다.

```text
EDGELLM_PLUGIN_PATH=.local/upstream-v010/.local/build-v010-sm86-only/libNvInfer_edgellm_plugin.so
```

처음에는 이 변수를 누락해 v0.10 binary가 현재 브랜치의 `build/libNvInfer_edgellm_plugin.so`를 읽었고,
context-reuse prefill에서 Q/K/V shape 검증이 실패했다. v0.10 plugin을 명시한 뒤 동일 입력이 정상 완료됐다.
최종 표는 모두 release/plugin 조합을 명시해 다시 측정한 값이다.

## v0.10에서 관련성이 큰 변경

공식 릴리즈의 큰 변화 중 이 프로젝트와 직접 연결되는 것은 다음 네 가지다.

1. ONNX를 만들지 않고 checkpoint를 runtime weight로 바인딩하는 direct builder가 추가됐다.
2. process-local content-addressed context reuse와 multi-turn paged KV 재사용이 추가됐다.
3. 기존 prefill FMHA 구현이 CuTe DSL 기반 FMHA로 교체됐다.
4. Cosmos 3, Qwen3.8, Nemotron 3.5 등 모델 지원과 multimodal server 경로가 확장됐다.

다만 context reuse와 우리의 stable indexed-paged KV가 해결하는 문제는 다르다.

```text
v0.10 context reuse
  동일 prefix hash -> 보존된 page record를 다음 요청이 참조
  목적: 이미 계산한 prefix prefill 생략

current indexed-paged + scheduler
  request stable slot -> active page lease -> prefill/decode independent contexts
  목적: eviction copy 제거, continuous admission, 비대칭 batching, phase overlap
```

둘은 대체 관계가 아니라 계층 관계다. v0.10의 cache record/refcount/LRU를 prefix-sharing 계층으로 사용하고,
현재 scheduler의 active request lease가 재사용 page를 인계받는 방향이 맞다.

## 고정-shape kernel 비교

조건은 input/past-KV 512, BS 1/2/4/8, warmup 20회, 측정 100회다. decode는 두 버전 모두 CUDA graph를
사용했다. 값은 CUDA-event E2E 평균이며 음수는 v0.10 개선이다.

| Phase | BS | v0.9.1 | v0.10.0 | 변화 |
| --- | ---: | ---: | ---: | ---: |
| prefill | 1 | 21.8803ms | 21.1984ms | -3.117% |
| prefill | 2 | 35.5879ms | 34.4692ms | -3.143% |
| prefill | 4 | 69.1995ms | 67.4346ms | -2.550% |
| prefill | 8 | 134.7067ms | 131.7530ms | -2.193% |
| decode | 1 | 6.3757ms | 5.9281ms | -7.020% |
| decode | 2 | 6.3835ms | 5.9523ms | -6.755% |
| decode | 4 | 6.5241ms | 6.0623ms | -7.078% |
| decode | 8 | 6.8514ms | 6.4047ms | -6.520% |

Prefill 개선은 batch가 커질수록 3.1%에서 2.2%로 줄지만 decode 개선은 전 구간에서 약 7%로 안정적이다.
현재 scheduler에서 decode 비중이 큰 workload일수록 v0.10 forward-port의 잠재 이득이 크다. 단, independent
contexts의 concurrent interference가 포함되면 isolated 7%가 그대로 system throughput에 더해지지는 않는다.

## BS8 real-request 묶음

output 길이 32/48/64/96/128의 기존 입력을 그대로 사용했다. 다섯 파일을 합치면 prompt 25,872 token,
generated 23,712 token으로 두 버전이 같다.

| 항목 | v0.9.1 | v0.10.0 | 변화 |
| --- | ---: | ---: | ---: |
| prefill GPU time | 1431.30ms | 1424.38ms | -0.48% |
| generation GPU time | 18710.14ms | 18247.72ms | -2.47% |
| total GPU service | 20141.43ms | 19672.10ms | -2.33% |
| generated token/s | 1177.27 | 1205.36 | +2.39% |
| all token/s | 2461.79 | 2520.52 | +2.39% |
| peak GPU memory | 7348MiB | 7524MiB | +176MiB |

| Output cap | Prefill token/s v0.9/v0.10 | Generation token/s v0.9/v0.10 |
| ---: | ---: | ---: |
| 32 | 10042.0 / 10003.7 | 1412.6 / 1446.3 |
| 48 | 14807.9 / 14875.6 | 1357.5 / 1391.5 |
| 64 | 15726.1 / 15759.0 | 1314.6 / 1347.4 |
| 96 | 21519.3 / 22211.3 | 1189.3 / 1220.4 |
| 128 | 19477.4 / 19182.2 | 1303.8 / 1336.3 |

실제 묶음에서는 profile 전환, sampling, host dispatch도 포함되므로 isolated decode의 약 7%가 generation
service 기준 약 2.5%로 희석된다. 이것이 v0.10을 현 scheduler에 올렸을 때 기대치를 잡는 더 현실적인 수치다.

## Context reuse 검증

v0.10 공식 `llm_context_reuse.json`은 긴 prefix를 가진 세 요청으로 구성된다. 첫 요청은 cache producer,
두 번째와 세 번째는 같은 512-token page-aligned prefix를 재사용한다.

| 항목 | reuse off | reuse on | 변화 |
| --- | ---: | ---: | ---: |
| admitted / hit | 3 / 0 | 3 / 2 | +2 hits |
| reused tokens | 0 | 1024 | +1024 |
| computed tokens | 1802 | 778 | -56.8% |
| prefill GPU time | 84.53ms | 62.56ms | -25.99% |
| generation GPU time | 18.78ms | 18.87ms | +0.5% |
| peak GPU memory | 7524MiB | 7524MiB | 동일 |
| retained base pages | 0 | 4 / 128 | +4 pages |

계산 token 감소율보다 prefill 시간 감소율이 작다. 첫 cold request 비용은 그대로이고, hit에서도 마지막 partial
block과 runtime 준비 비용이 남기 때문이다. 실제 서비스 이득은 prefix hit rate와 prefix page 정렬에 좌우된다.

현재 다섯 workload 비교에서는 Current와 vLLM 모두 prefix cache를 껐다. 따라서 이 기능을 켠 결과를 기존
`4413 token/s`에 단순 합산하면 안 된다. repeated system prompt/RAG/multi-turn 전용 trace를 별도로 만들어
reuse off/on과 vLLM prefix caching을 같은 hit distribution으로 비교해야 한다.

## Standard ONNX와 direct builder

같은 output-128 입력에서 v0.10 자체 plugin으로 재측정했다.

| 항목 | standard ONNX | direct | direct 변화 |
| --- | ---: | ---: | ---: |
| artifact directory | 3.8GiB | 37MiB | 약 -99% |
| peak GPU memory | 7524MiB | 6854MiB | -670MiB (-8.90%) |
| prefill GPU time | 404.13ms | 401.90ms | -0.55% |
| generation GPU time | 6896.70ms | 8076.90ms | +17.11% |
| total GPU service | 7300.82ms | 8478.80ms | +16.13% |
| generation token/s | 1336.3 | 1141.0 | -14.61% |

direct artifact가 작은 이유는 model weight가 engine 안에 없고 실행 때 `.local` checkpoint에서 약 3.44GB arena로
변환·바인딩되기 때문이다. tied embedding/LM-head alias로 약 622MB의 중복을 피한다. 즉 37MiB만 배포하면 되는
독립 artifact가 아니라 원 checkpoint가 함께 필요하다.

메모리 절감은 independent context의 workspace 부담을 줄이는 후보지만, 현재 decode 회귀가 더 크다. 또한
standard/direct/current 사이의 greedy token exact identity가 아직 성립하지 않아 accuracy/logit gate도 남아 있다.
direct builder는 다음 조건을 모두 만족할 때만 기본 후보로 승격한다.

1. standard 대비 teacher-forced logits와 accuracy suite 통과
2. decode BS1--64와 past-KV 128/512/1536에서 3% regression gate 통과
3. 두 independent execution context가 external weight arena를 실제로 공유함을 allocator 계측으로 확인
4. checkpoint transform startup 비용과 server restart 비용 기록

## 현재 구현과의 충돌 면적

v0.9.1부터 v0.10.0까지 upstream은 1,059개 파일을 바꿨다. 현재 브랜치가 v0.9.1 이후 수정한 파일과 겹치는
파일은 79개다. 특히 다음 hot path가 모두 겹친다.

- `cpp/plugins/attentionPlugin/attentionPlugin.{h,cpp}`
- `cpp/kernels/posEncoding/applyRopeWriteKV.*`
- `cpp/kernels/kvCacheUtilKernels/kvCacheUtilsKernels.*`
- `cpp/runtime/kvCacheManager.*`, `hybridCacheManager.*`
- `cpp/runtime/exec/engineExecutor.*`
- `cpp/runtime/state/pipelineIO.*`
- `cpp/runtime/llmInferenceRuntime.*`
- `cpp/builder/llmBuilder.*`
- Python attention custom op/export/model files

따라서 v0.10을 현재 브랜치에 단순 merge한 뒤 conflict를 기계적으로 해결하면 안 된다. upstream의 새 cache/page
ownership과 CuTe FMHA 계약을 먼저 기준으로 삼고, 우리의 기능을 작은 계층 순서로 forward-port해야 한다.

## Forward-port 순서

1. **v0.10 표준 baseline 고정**
   - v0.10 standard ONNX engine을 correctness/kernel oracle로 보존한다.
   - binary와 plugin fingerprint를 manifest에 넣고 mismatch면 시작을 거부한다.
2. **모델 중립 phase shell 이식**
   - queue, request lifecycle, CUDA event metrics, scheduler policy를 먼저 옮긴다.
   - KV/plugin 변경 없이 v0.10 public output과 단일-context identity를 확인한다.
3. **Independent TensorRT contexts 이식**
   - shared CUDA context, phase별 execution context/workspace/I/O, event handoff를 옮긴다.
   - v0.10의 176MiB 증가를 포함해 P8/D32부터 headroom을 측정하고 D64로 늘린다.
4. **Stable indexed ownership 재연결**
   - v0.10 page table/cache lease를 source of truth로 두고 stable slot mapping을 adapter로 붙인다.
   - upstream context-cache record와 active lease의 refcount/COW 경계를 하나로 합친다.
5. **Packed/chunked prefill 이식**
   - 새 CuTe FMHA shape 계약에 variable-length packed prefill을 다시 연결한다.
   - isolated prefill, overlap cost table, TPOT hard guard 순서로 재검증한다.
6. **Context reuse production 연결**
   - repeated-prefix trace에서 off/on, Current/vLLM을 비교한다.
   - cancellation, cache eviction, LoRA key isolation, page pressure를 포함한다.
7. **전체 회귀 gate**
   - short/balanced/decode-heavy/long-prefill/bimodal 다섯 trace를 세 번씩 실행한다.
   - v0.10 clean public, forward-ported Current, vLLM을 같은 날 비교한다.

첫 production 목표는 direct builder가 아니라 standard ONNX 경로다. 이 순서가 kernel 개선과 scheduler 개선을
분리해 회귀 원인을 추적할 수 있고, direct builder의 외부 weight binding이라는 별도 변수를 뒤로 미룬다.

## 디스크와 artifact

측정 종료 시 filesystem 여유는 약 2.3GiB다. 새 대형 engine은 만들지 않았다.

| Artifact | 크기 |
| --- | ---: |
| v0.10 standard ONNX | 4.4GiB |
| v0.10 standard engine | 3.8GiB |
| v0.10 worktree/build | 1.7GiB |
| v0.10 direct text+visual artifact | 37MiB |

forward-port를 시작하기 전에 standard ONNX와 engine 중 하나를 재생성 가능 artifact로 분류해 정리해야 한다.
이번 비교에서는 재현성 보존을 위해 삭제하지 않았다.

## Artifact 위치

- v0.10 source/build: `.local/upstream-v010/`
- standard ONNX: `.local/cosmos-reason2-2b/v010-onnx-fp16-text/`
- standard engine: `.local/cosmos-reason2-2b/v010-onnx-engine-fp16-b8-kv2048-p128/`
- direct engine: `.local/cosmos-reason2-2b/v010-direct-fp16-b8-kv2048/`
- corrected fixed-shape results: `.local/cosmos-reason2-2b/v010-bench-fixed-plugin-correct/`
- corrected BS8 real-request results: `.local/cosmos-reason2-2b/v010-real-trace-bs8-plugin-correct/`
- context reuse off/on: `.local/cosmos-reason2-2b/v010-context-reuse/`
- corrected direct result: `.local/cosmos-reason2-2b/v010-direct-plugin-correct/`
