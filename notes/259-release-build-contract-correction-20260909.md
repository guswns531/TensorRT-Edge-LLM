# 259. Release build 계약 불일치 발견과 교정

## 확정된 사실

- Old retained V1 commands는 `.local/v010-forward-build/examples/llm/llm_phase_context_smoke`를 사용한다.
- 해당 cache는 `CMAKE_BUILD_TYPE:STRING=Release`, effective CXX는 `-O3 -DNDEBUG`다.
- Current `.local/v0101-forward-build-make`는 build type이 비어 있었고 effective CXX에도 -O3가 없었다.
- CUDA flags도 current는 Release의 `-O3 -DNDEBUG --use_fast_math`가 없었다.
- Cache에는 architectures=86이 있었지만 upstream CMake가 이를 덮어써 actual flags는
  80/86/89/90/100a/120이었다. Cache 값만으로 검증하면 이 차이를 놓친다.

따라서 기존의 code-version/policy/KV attribution에 compiler configuration confound가 있었다.
원시 수치가 조작되거나 inference가 실패한 것은 아니다. 하지만 공정한 old/current 성능
비교라는 해석은 철회하고 Release 조건으로 다시 평가해야 한다.

## 발견 계기

Balanced V1 gateway의 전체-process host counters에서 old 대비 current의 poll/serialization
비용 차이가 컸다. 이 counter는 calibration을 포함하므로 measurement-only decomposition은
아니다. 이를 단서로 실제 `flags.make`를 검사해 mismatch를 확인했다.

소스가 동일했던 KV active view도 미최적화 빌드에서는 다르게 느릴 수 있다.
기존 note258의 49µs 결과를 소스만 근거로 old와 같다고 취급하면 안 된다.

## 보존과 중단

미최적화 HTTP는29 cells/87 runs까지 보존했다. 마지막 실행 중인 cell은 정상 종료시켰다.
큐와 finalizer만 종료하여 미최적화 비교를 더 확대하지 않았다.
`legacy-parity-20260908/unoptimized-binaries/`에 runtime 및 plugin을 보존했다.
원래 binary SHA256은 `ace355156fa6c525822e13d79ca329ffef8644404c6efd936c9e8ac0673e9295`다.
이 결과를 Release 재측정 결과로 덮어쓰지 않는다.

## 수정

1. CMake에서 caller가 지정한 `CMAKE_CUDA_ARCHITECTURES`를 보존한다.
2. 빌드에 `-DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86`을 명시한다.
3. HTTP 비교 runner에 `--build-cache`를 필수로 추가한다. cache의 Release와 actual core CXX
   flags의 -O3를 검사하여 불일치하면 실행을 거부한다.
4. 모델/engine/KV256/phase 정책을 동시에 바꾸지 않는다. 기존 engine으로 먼저 runtime flags의
   효과를 측정한다. 새 engine tactic 재탐색은 이 비교와 분리한다.

기존 사용자 dispatch-only telemetry 수정과 note248 draft는 보존한다. Release 기본값을
upstream 전체에 강제로 바꾸지는 않았으며, 연구 벤치마크의 사전 검사를 강화했다.

## 검증 계획

- Release runtime/phase unit tests, KV metadata benchmark 재실행.
- Balanced/Multi-image smoke에서 같은 요청과 greedy hashes 확인.
- Release V0/V1/V2 full12를 같은 repeat/calibration 조건으로 다시 실행.
- Clean/current page-table 실제 소스도 같은 Release common implementation에 링크해 비교.
- 기존 frozen vLLM은 입력/출력/클라이언트 계약이 같으면 재사용.

현재 이 문서는 발견과 수정 기록이다. Release 성능 향상의 크기나 회귀 해결 완료를 아직
주장하지 않는다. 결과가 나오면 별도 절에 추가한다.

## Release 기초 검증

Runtime SHA256: `6c02172bcc0e14caa71c4f93d84b5b0f77d7d1ffd2f94431a0cfc0f87ead951f`.
Phase/independent tests 322개 통과, opt-in benchmark 1개는 기본 실행에서 skip한 뒤 별도 통과했다.
KV metadata의 같은 100 warmup + 1000 iteration 계약에서 얻은 host median은 다음과 같다.

| BS64 경로 | 미최적화 µs | Release µs |
|---|---:|---:|
| Stable, persistent off | 49.303 | 12.466 |
| Stable, persistent on | 27.539 | 7.667 |
| Churn, persistent off | 54.650 | 15.518 |
| Churn, persistent on | 41.834 | 11.710 |

KV tensor 크기와 algorithm은 바꾸지 않았다. 이 수치는 metadata microbenchmark이며,
전체 inference latency가 동일 비율로 감소한다는 뜻이 아니다. Release HTTP는 별도 확인한다.
결과 루트는 `.local/results/v0101-forward-port/release-parity-20260909/`다.
