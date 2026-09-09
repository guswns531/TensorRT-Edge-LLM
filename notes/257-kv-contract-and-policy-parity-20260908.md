# 257. v0.10.0/v0.10.1 KV 계약과 정책 성능 비교

> 2026-09-09 정정: v0.10.1 `v0101-forward-build-make`의 실제 C++ flags에 -O3가 없고
> build type이 비어 있음을 발견했다. old는 Release였다. 아래 수치는 유효한 관측 기록이지만
> 공정한 Release-to-Release 버전 비교가 아니다. 같은 KV 소스라도 컴파일 옵션 때문에 비용이
> 달라질 수 있다. 후속 Release 재검증이 필요하다.

## 목적과 비교축

기존 v0.10.0 실행 조건을 보존하며 forward-port 효과를 검증한다. 모델·KV·candidate
변경을 한 번에 정책 개선으로 해석하지 않는다. 외부 cost registry나 workload별 규칙은 추가하지 않는다.

- 이전 구현: root `f4eb53e`, V0 Exact / V1 Scalar / V2 Scalar+Transition.
- 현재 구현: `3e4bf35`, 같은 세 정책. 기존 dispatch telemetry 미커밋 변경은 보존한다.
- clean tags: `v0.10.0`, `v0.10.1` (`e8b2952`).
- 이번 첫 screening: 현재 binary에서 Balanced/Multi-image × V0/V1/V2, 각 1회.
- 현재 binary SHA256: `b6cc00c88e5485a8df8823e44752e250000a03884489f1fdac8e62910dc9246a`.

## 1. Upstream 변경과 우리 구현을 구분

| 항목 | clean v0.10.0 | clean v0.10.1 | 우리 v0.10.1 |
|---|---|---|---|
| 실제 owning allocation | per-layer paged pool | 동일 pool layout | 동일 pool layout |
| combined tensor API | pool owner + slot-shaped alias | pool-shaped tensor 직접 반환 | upstream pool API 사용 |
| KV page-table upload | dirty rows + pinned staging 1개 | 구현 동일 | staging ring 3개, event query, 필요 시 wait |
| 물리 pool 최소 크기 | 최대 batch×최대 길이의 worst-case 이상 | 동일 제한 | 명시적 undercommit 허용 |
| phase-local P/D metadata | 우리 phase 계층 없음 | 우리 phase 계층 없음 | 독립 active length/page table |
| stable request/page lease | 우리 StableKVPageManager 없음 | 우리 StableKVPageManager 없음 | v0.10.0 구현과 동일 |

두 upstream tag의 `kvPageTable.cpp`는 동일하다. v0.10.1이 처음 paged KV를 도입한 것이 아니다.
Upstream 자체에도 page-table row compaction 및 context reuse 기능이 있다. 따라서 KV payload
무복사를 우리만의 기능이라고 주장하지 않는다. 우리의 추가점은 독립 phase의 재배칭에서 유지되는
stable ownership, phase-local mutable metadata, 작은 pool의 admission/backpressure 결합이다.

우리 old/current의 `stableKVPageManager.{cpp,h}`, `phaseKVActiveView.cpp`, `kvPageTable.cpp`는
동일하다. `allowUndercommit` → `allowPoolUndercommit`은 이름/연결 경로 변경이며 정책 철학 변경이 아니다.
다만 현재 flag=true,numPages=0 조합의 validation은 예전 명시적 양수 검사와 다르므로, flag만 같다는
이유로 모든 입력 계약이 동일하다고 말하지 않는다. 이번 엔진은 명시적으로 256 pages다.

## 2. 물리 메모리와 alias

실제 layer pool은 두 버전 모두 `[2, pages, 128, Hkv, D]`다. 이전의 slot-shaped alias는
추가 GPU allocation이 아니었다. alias 제거만으로 수백 MiB가 감소하거나 증가하지 않는다.

Cosmos 현재 엔진: FP16, 28 attention layers, Hkv=8, D=128, 256 pages.

`KV bytes = 2 × 256 × 128 × 8 × 128 × 2 × 28 = 3,758,096,384 bytes = 3,584 MiB`.

이는 순수 KV payload의 구성상 계산값이며 전체 GPU peak 측정값이 아니다.
80 slots × 2048 tokens를 전부 worst-case 예약하면 1280 pages, KV만 17,920 MiB가 필요하다.
따라서 clean upstream의 해당 worst-case allocation을 RTX 3080에서 그대로 성능 비교하는 것은 불가능하다.
256-page pool에 같은 80-slot profile을 허용하는 우리의 undercommit은 여전히 필요하다.

Physical pool 자체는 시작 시 고정 allocation이다. 동적으로 바뀌는 것은 request의 page lease와
사용 page 수이지, 매 decode마다 cudaMalloc으로 pool 크기를 바꾸는 것이 아니다.

## 3. 보존할 조건과 아직 남아 있는 차이

Cosmos FP16, P8/D64/E4, text chunk128, vision profile1024, slots80, KV256,
client64, ignore-EOS, generic calibration을 유지한다. CUDA graph는 지원 여부가 아니라 실제
capture/hit와 manifest 옵션으로 확인한다. KV를 줄여 workspace 차이를 숨기지 않는다.

이번 screening에서는 새 frontier 옵션과 기각한 encoder horizon을 켜지 않는다. 정책 변경 없이
기존 retained command의 policy 문자열과 output path만 바꾼다. 각 process에서 동일 calibration
요청을 사용하지만 CUDA 관측이 달라지므로 posterior 값까지 동일하다고 주장하지 않는다.

아직 old/current 순수 정책 parity는 아니다:

1. 현재 external prefill residual P+D에도 Scalar 평가가 허용된다.
2. E/P와 E/D exclusive gate가 분리되었다.
3. unknown vision payload의 E1 bootstrap이 추가되었다.
4. JIT XQA, exporter/TRT engine, native exact vision GELU 경로가 다르다.

안전장치를 지워 parity를 만들지 않는다. 위 차이를 개별 실험으로 분리하고, bootstrap을 제외하려면
양쪽 모두 calibration 이후 payload 크기가 알려졌다는 관측이 필요하다.

## 4. 단계별 검증

1. 동일 current binary의 V0/V1/V2 screening으로 최근 측정 공백을 줄인다.
2. 이전 retained 12-workload 결과와 TTFT/TPOT/E2E mean/p95 및 throughput을 비교한다.
   역사적 old 1회는 contemporaneous paired 결과가 아니므로 causal 주장에 사용하지 않는다.
3. old/current 정책 호출 범위 차이를 통제한 후 full12를 교대 3회 측정한다.
4. KV-only 측정은 같은 engine/kernel에서 page-table bind/upload와 lease churn을 분리한다.
   host wait, H2D bytes, row reuse, allocation bytes, pressure를 함께 보고한다.
5. upstream pool API의 alias 제거와 XQA 변경을 별개로 본다. 서로 다른 engine의 E2E 차이를
   KV-only speedup으로 표시하지 않는다. clean full-server는 별도 end-to-end baseline이다.
6. 동일 계약 frozen vLLM은 재사용한다. 새 output/client 계약이면 다시 측정한다.

## 재현

`benchmarks/phase_serving/replay_retained_policy_commands.py`는 신뢰하는 로컬 commands.json만 입력한다.
output 디렉토리가 이미 있으면 실패해 기존 결과를 덮어쓰지 않는다.

```bash
python3 benchmarks/phase_serving/replay_retained_policy_commands.py \
  --commands ../results/v0101-forward-port/host-path-20260908/full12/commands.json \
  --output ../results/v0101-forward-port/new-release-screening \
  --build-cache ../v0101-forward-build-make/CMakeCache.txt \
  --cases balanced multi-image
```

위 경로는 이 worktree 루트 기준이다. commands.json의 실제 absolute output 경로가 실행 기록이다.
위 예제는 강화된 Release preflight를 사용하는 새 실행 예제이며 아래 과거 수치를 재현하는
미최적화 명령이 아니다. 실제 full12 순서는 note258처럼 policy별 3회 순차 실행으로 변경했다.
현재 문서의 계획을 완료된 KV-only 실험으로 읽으면 안 된다.

## 5. 완료된 첫 screening: 6회

현재 세 variant는 각 workload 내 greedy hash가 모두 같다. 하지만 old/current 사이 hash는 다르다.
Old multi-image는 V2와 V0/V1 사이에도 hash가 다르다. 고정 output token 수의 비교이며,
cross-engine exact identity나 policy-only causal comparison을 통과했다는 의미는 아니다.

모두 각 1회. Old는 retained history, current만 이번에 실행했다. 시간 단위 ms, 처리량 token/s.
아래 peak는 benchmark GPU memory sampling 값이며 allocator 정밀 high-water mark가 아니다.
현재 Balanced도 vision engine을 적재하는 기존 full-serving 계약을 유지했다.

| Workload | 버전/정책 | token/s | TTFT mean/p95 | TPOT mean/p95 | E2E mean/p95 | GPU peak MiB |
|---|---|---:|---:|---:|---:|---:|
| Balanced | old V0 | 4562.71 | 66.38/166.95 | 12.07/13.48 | 1095.71/1702.06 | 9237 |
| Balanced | old V1 | 4455.01 | 67.22/170.24 | 12.34/13.87 | 1118.36/1739.66 | 9237 |
| Balanced | old V2 | 4454.09 | 66.62/168.90 | 12.36/14.06 | 1117.87/1745.87 | 9237 |
| Balanced | current V0 | 4119.81 | 65.94/170.05 | 13.53/15.53 | 1220.07/1946.65 | 9399 |
| Balanced | current V1 | 4194.22 | 67.44/176.67 | 13.23/14.86 | 1192.96/1862.58 | 9399 |
| Balanced | current V2 | 4025.80 | 68.54/173.57 | 13.78/15.54 | 1241.47/1969.12 | 9399 |
| Multi-image | old V0 | 310.23 | 269.92/308.67 | 7.74/8.64 | 509.86/515.43 | 9379 |
| Multi-image | old V1 | 312.12 | 267.92/304.71 | 7.71/8.62 | 506.92/512.37 | 9433 |
| Multi-image | old V2 | 324.84 | 258.23/284.15 | 7.37/7.88 | 486.72/492.09 | 9395 |
| Multi-image | current V0 | 291.59 | 248.98/329.61 | 9.48/12.72 | 542.71/548.50 | 9443 |
| Multi-image | current V1 | 288.31 | 255.69/336.02 | 9.46/12.69 | 548.89/554.65 | 9443 |
| Multi-image | current V2 | 290.34 | 251.83/332.08 | 9.46/12.69 | 545.11/550.85 | 9443 |

동일 계약 frozen vLLM(note256, 재실행하지 않음):

| Workload | token/s | TTFT mean/p95 | TPOT mean/p95 | E2E mean/p95 |
|---|---:|---:|---:|---:|
| Balanced | 4318.34 | 110.37/247.20 | 12.23/13.71 | 1154.26/1771.51 |
| Multi-image | 244.52 | 259.81/402.58 | 12.42/16.32 | 644.30/653.90 |

### 해석

- 이번 Balanced에서 current V1은 V0보다 처리량이 약 1.81% 높으나 old V1보다는 약 5.85% 낮다.
  current V2는 old V2보다 약 9.62% 낮다. 단발/고정순서이므로 확정적 정책 순위는 아니다.
- Multi-image는 current 세 정책 차이가 약 1% 수준이다. old V2의 높은 결과에는 서로 다른
  greedy trajectory라는 confound가 있어 그대로 회복 목표의 exact reference로 쓰지 않는다.
- Balanced GPU peak +162 MiB는 양쪽 KV256 물리 payload의 차이로 설명되지 않는다.
  엔진/context/vision buffer 등 나머지 allocation과 런타임 sampling 시점을 분해해야 한다.
- current V1은 frozen vLLM 대비 Balanced 처리량 약 -2.88%, Multi-image 약 +17.91%다.
  Balanced TTFT는 더 낮지만 TPOT/E2E는 더 높다. 모든 latency 승리라는 결론은 아니다.
- KV-only 비용이나 page-table staging 효과는 이번 E2E 결과에서 식별할 수 없다.
  256-page payload를 유지한 상태에서 metadata/lease churn 단위 비교가 다음 작업이다.

실행 스크립트는 Python compile 검증, 비교 도구는 세 정책 token hash 동일성 검사를 통과했다.
`comparison.json/csv`에 현재 정책별 전체 지표를 저장했다. runtime 소스·엔진·KV 설정은 수정하지 않았다.
