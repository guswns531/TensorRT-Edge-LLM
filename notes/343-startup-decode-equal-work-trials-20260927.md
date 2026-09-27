<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# 343. Startup autotuning step 2: equal-work decode trials

## Contract

342의 후보 중 Gemma D8 vs D4+D4, Cosmos D64 vs D32+D32 두 점만 실행한다.
Engine/KV/기본 serving policy는 변경하지 않는다. Full12 및 fresh vLLM은 실행하지 않는다.

- `phaseDecodeEqualWorkTrial.inc`: 명시적 opt-in 진단이며 일반 serving 진입 전 별도 실행 후 종료한다.
- 각 row는 고정 token ID `1 + row % 17`을 128회 반복한 synthetic prefix를 실제 prefill로 처리한다.
  Tokenizer/HTTP trace가 아닌 물리적 batch 비용 실험이다.
- 같은 stable slots/page ownership, 동일 prefix KV, 동일 첫 decode 입력을 dense/split에 재사용한다.
  첫 decode는 position128만 덮어쓰고 다음 공통 dense decode는 position129만 덮어쓴다.
  반복마다 logical length를128로 되돌리므로 기존 prefix는 변경되지 않는다.
- 모든 후보를 CUDA graph capture 후 실행하며 graph miss/failure가 발생하면 실패한다.
- 각 모델 한 process에서 5 blocks × (20 warmup pairs + 50 measured pairs).
  각 pair 내 dense/split 순서를 교대한다. 5개 독립 process 반복이나 독립성 가정의 CI는 아니다.
- 각 dispatch의 CUDA event는 engine 구간만, host drain은 staging/binding/engine/argmax/D2H/sync를 포함한다.
  동일 context와 pinned buffer를 안전하게 재사용하기 위해 split 사이 completion을 기다린다.
- 두 번째 tick은 항상 dense이다. 이는 **강제된 공통 successor**이지 동적 scheduler의 미래 cohort 예측 검증이 아니다.
  P/E overlap, ready queue/admission, HTTP TTFT/TPOT/E2E 또는 자연 cohort fragmentation을 검증하지 않는다.
- GPU timer는 stream상의 engine 구간 elapsed이며 kernel active cycle 측정이 아니다.
- 두 tick의 greedy token 비교를 기록하되 이는 synthetic fixture 검사이며 full-model output-quality gate가 아니다.

## Reproduction

```bash
python3 benchmarks/phase_serving/run_decode_equal_work_trials.py \
  --build-root .local/baselines/startup-autotune-step2-v2-20260927/bin \
  --result-root .local/results/startup-autotune-step2-v2-20260927
```

결과와 command/source/binary/engine identity는 위 result root의 manifest 및 raw JSON에 보존한다.
모든 결과는 diagnostic이며 자동 설정 적용이나 기본 runtime 승격은 하지 않는다.

초기 `startup-autotune-step2-20260927` 실행은 진단 입력의 `inputsEmbeds.reshape` 누락으로
Gemma prefill 초기화 단계에서 실패했다. Timed sample이 없으며 성능 비교에서 제외한다.
수정 후 별도 immutable v2 binary/result 경로로 재실행하여 실패 기록을 덮어쓰지 않았다.

## Results

RTX 3080 10GB / driver610.43.02, 기존 Gemma AWQ 및 Cosmos FP16 engine을 사용했다.
소스는 `9926f08` + manifest에 보존된 patch와 `.inc`이다.
최종 바이너리 SHA256:
`62aab3bf0c8ddd3b8ae9e2d169d0d46c81283d3c2e05b723134681042c4d0e79`.

각 모델에서 dense250회, split250회 측정했다. 두 후보 모두 매번 같은 총 row 수를 한 token 진행시키고,
각 후보 뒤에 같은 크기의 dense successor를 한 번 실행했다. 아래 p95는 250개 표본의 nearest-rank p95다.

### Gemma: D8 vs D4+D4

| 시간, ms | Dense mean | Split mean | 변화 | Dense median / p95 | Split median / p95 |
|---|---:|---:|---:|---:|---:|
| 첫 tick engine GPU 합 | 12.191 | 11.844 | -2.84% | 12.189 / 12.215 | 11.839 / 11.884 |
| 첫 tick host 포함 drain | 12.916 | 13.026 | +0.85% | 12.905 / 12.981 | 13.007 / 13.105 |
| 공통 successor engine GPU | 12.190 | 12.192 | +0.01% | 12.188 / 12.217 | 12.189 / 12.219 |
| 두 tick host 포함 완료 | 25.830 | 25.944 | +0.44% | 25.814 / 25.896 | 25.926 / 26.053 |

GPU 절감은 mean0.347ms지만 engine 바깥 residual은 dense0.725ms에서 split1.182ms로 약0.457ms 증가했다.
Residual에는 input staging, KV metadata, embedding, binding, sampling, D2H, host 제출 및 완료 관측 등이
함께 들어가므로 이를 CPU scheduler 비용 하나로 해석하면 안 된다. 결과적으로 drain은 약0.110ms 느려졌다.
이 측정에서는 **GPU-only 이득이 dispatch 전체 이득으로 이어지지 않았다.**

### Cosmos: D64 vs D32+D32

| 시간, ms | Dense mean | Split mean | 변화 | Dense median / p95 | Split median / p95 |
|---|---:|---:|---:|---:|---:|
| 첫 tick engine GPU 합 | 8.199 | 13.915 | +69.71% | 8.200 / 8.231 | 13.916 / 13.933 |
| 첫 tick host 포함 drain | 8.491 | 14.497 | +70.74% | 8.490 / 8.532 | 14.494 / 14.541 |
| 공통 successor engine GPU | 8.227 | 8.238 | +0.14% | 8.225 / 8.274 | 8.236 / 8.286 |
| 두 tick host 포함 완료 | 17.009 | 23.031 | +35.40% | 17.009 / 17.078 | 23.028 / 23.112 |

Cosmos는 split이 GPU service부터 크게 불리했다. 같은64rows를 D32 두 번으로 처리할 이유가 없는 통제점이다.
단, 이것을 앞으로 올 요청을 기다려 무조건 D64를 채우라는 정책으로 확대하지 않는다.

### Block consistency

| Block | Gemma GPU median 변화 | Gemma drain median 변화 | Cosmos GPU median 변화 | Cosmos drain median 변화 |
|---|---:|---:|---:|---:|
| 1 | -2.851% | +0.788% | +69.817% | +70.821% |
| 2 | -2.881% | +0.748% | +69.655% | +70.718% |
| 3 | -2.863% | +0.860% | +69.719% | +70.771% |
| 4 | -2.852% | +0.841% | +69.722% | +70.721% |
| 5 | -2.875% | +0.782% | +69.555% | +70.609% |

부호는5개 블록에서 동일했다. 다만 같은 process/fixture의 반복이므로 다양한 request 분포나 재시작 분산을
대표하지 않는다. 이번 두 점에서만 split 미적용을 판단하기에 충분하며 full12는 추가로 실행하지 않는다.

### Correctness and scope checks

- Gemma 비교4,000 tokens / 불일치0, Cosmos32,000 tokens / 불일치0.
  같은 fixture의 반복 비교 횟수이며 서로 다른4,000/32,000개의 semantic test가 아니다.
- 각 모델 graph hit1,750 / miss0. Count에는350 pairs의 warmup 및 measurement가 모두 포함된다.
- 모든 stable slots 반환: Gemma24, Cosmos80. 종료 후 GPU1MiB, 실행 중 container 없음.
- Python result-contract tests4개 통과. Missing/duplicate pair, graph fallback, invalid order, nonfinite timing 검사.
- C++ 진단 타깃 build 성공. 일반 serving 실행 분기는 opt-in 환경변수 없이는 변경되지 않는다.
- HTTP workload가 아니므로 TTFT/TPOT/E2E와 vLLM 비교는 **N/A**다. `drain_ms`를 HTTP E2E로 표시하지 않는다.
- Vision encoder는 실행/상주하지 않는 D-isolated 실험이다. Full VLM의 메모리 사용량이나 overlap 간섭을 대표하지 않는다.

## Decision and next step

1. **두 split 모두 자동 적용하지 않는다.** 기존 champion binary와 설정을 유지한다.
2. Startup autotuner의 선택 근거는 engine GPU time만이 아니라 **동일 작업량의 host 포함 drain**이어야 한다.
   실제 serving 경로는 더 비동기적으로 동작할 수 있으므로 이 진단 결과를 전체 serving optimum으로 간주하지 않는다.
3. 후보 generator의 guarded proxy는 저렴한 탐색 우선순위로 유지할 수 있지만, candidate promotion은 직접 검증과
   별도 serving gate가 필요하다. 342의 bucketed 비용과 이번 exact prefix fixture의 수치를 혼합하지 않는다.
4. D split 탐색은 여기서 종료한다. 다음 작은 단계는 이 검증 contract를 startup trial 결과 형식에 연결하고,
   이득 없음이면 현 설정 유지/no-op으로 끝나는 선택·거부 처리를 명시하는 것이다.
5. 그다음 P/E 후보를 하나씩 검토한다. 새로운 적용 후보가 없는 상태에서 full12나 vLLM을 반복하지 않는다.
