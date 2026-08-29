<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Build-local decode cost bundle promotion

## 목적과 경계

이번 단계는 workload별 fine-tuning이나 외부 cost registry 없이, Cosmos Reason2-2B engine에서 직접 얻은
CUDA-event decode 비용을 dynamic batching에 연결하는 단계다. 다음 세 층은 계속 분리한다.

```text
build bundle             deployment-local batch/context baseline
node-local observation   현재 process와 E/P contention 보정
request state            queue, slack, ownership에 따른 실제 action 선택
```

bundle이 존재한다는 이유만으로 production decode 동작을 바꾸지 않는다. decode record, conservative context
coverage, sparse-coverage fallback, decision identity와 3% performance gate를 모두 확인한 뒤
`TRT_EDGELLM_ENABLE_PHASE_BUNDLE_DECODE_BATCHING=1`로 승격한다. control이 없으면 bundle은 Global action
oracle에는 사용되지만 decode batching은 기존 compatibility table을 유지한다.

## 구현

### Stable context coverage

decode key는 batch와 maximum row context bucket을 포함한다. 정확한 bucket이 없을 때 짧은 context 측정으로 긴
context를 추측하지 않는다. 요청 bucket 이상인 관측 중 가장 작은 bucket만 사용하고, batch 보간도 같은
context bucket의 두 직접 관측 사이에서만 허용한다.

```text
request D12/context2
  -> context2의 D8, D16이 있으면 batch 보간
  -> 없고 context4의 D8, D16이 있으면 보수적 covering 보간
  -> full runnable batch coverage가 없으면 largest runnable batch 유지
```

physical profile이나 stable-page feasibility는 기존 mechanism이 계속 보장한다. 비용 coverage가 희소하다는
이유로 cohort를 줄이지 않는다.

### Hot-path curve cache

초기 구현은 한 scheduling decision에서 D1부터 D64까지 각각 Oracle record 전체를 다시 순회했다. GPU work는
동일했지만 host poll 누적이 약 350ms 늘어 balanced throughput이 6% 이상 회귀했다. 이제 하나의
`(action, context, execution variant)`에 대해 모든 batch의 conservative covering curve를 한 번 만들고,
observation이 추가될 때만 무효화한다.

```text
이전: 64 batch queries x record scan x scheduler previews
현재: one curve construction + O(1) batch lookup
```

이 cache는 cost sample만 파생한 값이며 request/queue state를 보존하지 않는다. scheduler copy-on-write preview와
정책 의미도 바꾸지 않는다.

### Startup calibration shape

IPC startup warmup에 다음 control을 추가했다.

```text
TRT_EDGELLM_IPC_WARMUP_DECODE_SHAPES=batch:prompt_tokens,...
TRT_EDGELLM_IPC_WARMUP_SHAPE_SAMPLES=N
TRT_EDGELLM_IPC_WARMUP_POLL_GUARD=N
```

긴 prompt calibration은 chunked prefill 때문에 기존 100만 poll guard를 넘을 수 있어 startup 전용 guard를
분리했다. 실제 bundle은 D1/2/4/8/16의 긴 context, D24/32의 중간 context, D40/48/56/64의 page-pool에 맞는
짧은 context를 3회씩 요청해 생성했다. calibration 중 production dynamic policy가 shape를 바꾸지 않도록 정확한
shape bundle 생성 run에서는 dynamic decode를 잠시 끄고, production 재실행에서는 다시 켰다.

## 실제 cost bundle

RTX 3080 10GB, TensorRT 11.0/CUDA 13.3, Cosmos Reason2-2B FP16 tied engine, P8/D64, FP16 KV 256 pages를
사용했다. 대표 직접 관측은 다음과 같다.

| shape | context bucket | samples | median GPU ms |
|---|---:|---:|---:|
| D1 | 4 | 3 | 6.72 |
| D4 | 4 | 3 | 7.43 |
| D8 | 4 | 9 | 8.77 |
| D16 | 2 | 3 | 8.91 |
| D24 | 2 | 3 | 10.31 |
| D32 | 1 | 3 | 9.10 |
| D48 | 1 | 3 | 10.57 |
| D64 | 1 | 2 | 12.17 |

raw bundle은
`.local/phase-cost-decode-promotion-20260829/cosmos-decode-build-exact-v2.json`에 보존했다. build bundle은
repository에 commit하지 않는다.

## 실패에서 확인한 원인

첫 promotion은 balanced 3회에서 static 대비 throughput `-6.47%`, TPOT p95 `+8.65%`, E2E p95
`+7.39%`였다. contention-aware online learner를 다시 켜도 throughput은 약 `-6%`였다. metric run을 비교한
결과 정책의 GPU work 차이는 원인이 아니었다.

| metric | static | uncached bundle |
|---|---:|---:|
| D dispatch | 464 | 463 |
| mean D batch | 53.17 | 53.29 |
| summed D GPU ms | 3,716.4 | 3,709.7 |
| P dispatch | 106 | 107 |
| summed P GPU ms | 1,522.1 | 1,531.5 |
| summed makespan GPU ms | 5,240.9 | 5,243.4 |
| host poll ms | 5,919.4 | 6,269.1 |

즉 accurate cost가 나쁜 batch를 선택한 것이 아니라 cost lookup 구현이 host submission gap을 만들었다.
covering-curve cache 뒤 balanced 3회는 static 대비 throughput `-0.55%`, TTFT p95 `+0.80%`, TPOT p95
`+1.05%`, E2E p95 `+0.50%`로 모두 3% gate 안에 들어왔다.

## 12-workload 결과

모든 latency 단위는 ms다. short/decode-heavy/poisson/wave-drain은 screening 뒤 별도 3회 중앙값을 사용했고,
balanced도 fresh 3회다. 나머지는 12-workload screening 1회이므로 후속 전면 release gate에서는 3회로
확장한다. vLLM은 model, trace SHA, arrival/output contract가 변하지 않아 Note 171의 cached fresh 3회 결과를
재사용했다.

| workload | tok/s | vs static Current | vs vLLM | TTFT mean / p95 | TPOT mean / p95 | E2E mean / p95 | peak MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| short | 2,506.39 | -0.37% | +26.36% | 84.23 / 166.81 | 13.46 / 22.28 | 329.36 / 409.29 | 9,313 |
| balanced | 4,547.29 | -1.10% | +4.93% | 66.93 / 164.63 | 12.10 / 13.46 | 1,098.19 / 1,707.49 | 9,313 |
| decode-heavy | 5,225.53 | -0.58% | +5.23% | 64.39 / 173.15 | 10.67 / 11.22 | 2,820.41 / 4,295.40 | 9,313 |
| long-prefill | 1,206.78 | +2.43% | +6.75% | 2,039.60 / 2,648.40 | 26.37 / 30.22 | 4,305.78 / 6,088.84 | 9,313 |
| bimodal | 1,919.87 | +0.65% | +4.33% | 1,896.86 / 4,014.70 | 17.98 / 29.68 | 4,377.94 / 9,107.84 | 9,313 |
| text-heavy | 1,946.84 | -0.04% | +19.09% | 331.49 / 1,057.95 | 25.00 / 38.13 | 1,629.76 / 1,732.29 | 9,427 |
| mixed | 1,140.48 | -0.41% | +23.76% | 733.58 / 2,130.35 | 31.25 / 41.45 | 2,227.77 / 2,510.02 | 9,453 |
| vision-heavy | 691.67 | +0.41% | +19.42% | 1,392.18 / 3,154.61 | 26.60 / 37.26 | 2,459.75 / 3,497.76 | 9,453 |
| poisson | 1,986.19 | -0.24% | +10.34% | 190.75 / 748.83 | 21.72 / 40.30 | 1,569.96 / 2,011.95 | 9,395 |
| wave-drain | 97.75 | +0.14% | +1.99% | 224.27 / 336.18 | 9.65 / 11.99 | 523.46 / 567.72 | 9,487 |
| multi-image | 309.19 | -0.58% | +26.45% | 211.78 / 289.04 | 9.52 / 11.23 | 506.76 / 514.39 | 9,487 |
| late-vision | 2,538.58 | -0.18% | +7.60% | 113.95 / 448.02 | 9.30 / 9.37 | 1,445.93 / 1,817.14 | 9,351 |

반복 확인한 네 경계 workload의 throughput/TPOT/E2E 회귀는 모두 1.62% 이내였다. decode-heavy, poisson,
wave-drain TTFT는 static보다 개선됐다. output completion과 token trace hash는 모든 반복에서 일치했다. Current는
cached fresh vLLM 대비 12/12 workload에서 처리량 우위를 유지한다.

## 검증과 raw 결과

- focused cost/scheduler tests: `150/150` pass
- `unitTest`, `llm_phase_context_smoke` TensorRT 11/CUDA 13.3 build pass
- balanced static 3회:
  `.local/phase-cost-decode-promotion-20260829/balanced-static-3x`
- uncached 실패 보존:
  `.local/phase-cost-decode-promotion-20260829/balanced-bundle-3x`
- cached balanced 3회:
  `.local/phase-cost-decode-promotion-20260829/balanced-bundle-cached-3x`
- 12-workload screening:
  `.local/phase-cost-decode-promotion-20260829/all-workloads-bundle-12x1`
- 네 경계 workload 3회:
  `.local/phase-cost-decode-promotion-20260829/suspects-bundle-4x3`
- static/bundle detailed metric:
  `.local/phase-cost-decode-promotion-20260829/diag-*-metrics`

## 다음 단계

이번 단계는 build timing을 hot path에 안전하게 연결했다. 다음은 external registry가 아니라 동일 artifact의
exact identity를 완성하는 일이다. engine/plugin/model hash를 deployment fingerprint에 항상 채우고, exact
bundle은 자동 승격하되 compatible bundle은 현재처럼 explicit promotion을 유지한다. 그 뒤 P와 overlap도 같은
bounded curve lookup으로 옮기고, 전체 12 x 3 release gate를 다시 실행한다.
