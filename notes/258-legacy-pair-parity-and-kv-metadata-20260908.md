# 258. Legacy pair eligibility와 KV metadata 독립 검증

> 2026-09-09 중단/정정: 29개 cell, 87회 완료 후 build-type mismatch를 발견해 큐를 중단했다.
> 실행 실패가 아니라 비교 계약 교정이다. current는 build type이 비어 -O3/-DNDEBUG가 없었고,
> old는 Release였다. 아래 KV microbenchmark도 미최적화 core의 결과다. 같은 소스라는 이유로
> KV host 비용이 동일하다고 추론하면 안 된다. Release/SM86 재빌드 후 별도 결과로 다시 측정한다.

## 실행 계약

Source base `3e4bf35` + 이번 working diff. Binary SHA256:
`ace355156fa6c525822e13d79ca329ffef8644404c6efd936c9e8ac0673e9295`.
기존 dispatch telemetry 미커밋 수정은 보존했다. 성능 측정은 instrumentation off.
Note257의 text/vision engine을 재사용했으며 모델, quantization, KV256, slots80, P8/D64/E4,
P128/vision P1024, generic calibration, client64, ignore-EOS를 변경하지 않았다.

## 구현

`PhaseThreeCoordinatorConfig::preserveLegacyPairEligibility`는 기본 false다.
`TRT_EDGELLM_LEGACY_PAIR_ELIGIBILITY`가 있으면 비교 실행에서만 활성화한다. 값 0이 아니라
환경 변수 제거가 off다. 아래 두 변경만 v0.10.0 범위로 제한한다.

1. External prefill residual P+D는 contextual Scalar 평가에서 제외한다.
2. E/P exclusive candidate 조건이면 E+D 후보도 제외한다.

E/D shared workspace safety는 legacy 옵션과 무관하게 유지한다. 실제 초기 unknown-payload E1
bootstrap은 제거하지 않는다. 따라서 이름은 pair eligibility parity이며 전체 runtime equivalence가 아니다.
Legacy 조건을 기본값으로 승격하지 않았다. 이 실험 자체로 성능 향상이 보장된다고 판단하지 않는다.

코드: `cpp/runtime/scheduling/phaseThreeCoordinator.{h,cpp}`,
CLI 연결: `examples/llm/llm_phase_context_smoke.cpp`.
회귀 테스트는 external/text 허용 범위와 E/D workspace 배타성의 우선 적용을 확인한다.

## 테스트

TensorRT26.06 container, `TRT_PACKAGE_DIR=/opt/tensorrt`, 명시적 LD_LIBRARY_PATH로 빌드.
`unitTestRuntime --gtest_filter='Phase*:*Independent*'`: 323개 중 322 PASS, opt-in benchmark 1 SKIP.
Opt-in benchmark는 별도 실행하여 PASS했다. 새 모델-facing export 변경은 없으며 기존 검증된 엔진을 사용한다.

## KV-only microbenchmark 범위

`TRT_EDGELLM_KV_METADATA_BENCH=1`로
`PhaseKVActiveViewTest.IsolatedMetadataCostBenchmark`를 실행한다.

- BS1/8/32/64 × persistent page binding off/on × churn off/on = 16조건.
- 각 조건 warmup100 + 측정1000, 순차 고정 순서, 1회 sweep.
- 256-page 주소 공간, 80 stable slots, sequence length256. 실제 큰 KV payload allocation이나
  attention kernel은 실행하지 않는다. Metadata와 host lease 관리의 독립 비용이다.
- Churn: 매 step 한 lease를 반환/재할당하고 active row를 회전한다.
- Host 시간은 event record + `prepare()` + event record까지다. `complete()` 비용은 제외한다.
- GPU event interval에는 CPU submission gap이 포함될 수 있다. 순수 DMA/kernel busy time이 아니다.
- 매 iteration 종료 event를 기다려 pinned active-length staging 재사용을 안전하게 한다.
  따라서 concurrent contention이나 staging-ring exhaustion benchmark는 아니다.
- Upload byte counter는 warmup을 포함한 전체1100회 누계다.

### Host prepare median/p95 (µs)

| BS | stable off | stable on | churn off | churn on |
|---|---:|---:|---:|---:|
| 1 | 12.111/12.448 | 6.015/6.267 | 12.480/12.777 | 6.734/7.032 |
| 8 | 15.599/15.878 | 7.947/8.270 | 20.018/20.733 | 13.814/14.143 |
| 32 | 30.084/30.604 | 16.324/16.663 | 35.082/37.846 | 25.276/38.943 |
| 64 | 49.303/52.146 | 27.539/28.076 | 54.650/86.486 | 41.834/42.521 |

열의 off/on은 persistent page binding 설정을 뜻한다. stable/churn은 request ownership 조작의 차이다.
Lease churn median은 약2.0–2.4µs이며 모든 조건에서 host staging wait는 0이었다.
Stable cohort는 off에서도 dirty check가 작동하여 page table H2D는 초기8192 bytes뿐이다.
따라서 on의 시간 단축은 H2D 감소가 아니라 host row binding/validation 생략으로 해석한다.
Churn BS64의 table upload는9,011,200 bytes로 off/on이 같다.

이 기능은 old/current에서 소스가 같으므로 v0.10.1 회귀 원인의 증거는 아니다.
단, 이후 발견된 build-type 차이 때문에 소스 동일성을 성능 동일성으로 확장할 수 없다.
Clean upstream의 single pinned staging과 직접 timing A/B한 실험도 아니다.

## HTTP full12

12 workloads × V0/V1/V2 × fresh process 3 repeats = 108 runs를 시작했다.
매 run에서239(text) 또는319(VLM) generic calibration requests를 다시 실행한다.
Policy 순서는 Exact→Scalar→Scalar+Transition이며 policy 간 ABBA는 아니다.

결과 디렉토리: `.local/results/v0101-forward-port/legacy-parity-20260908/`.
`full12/commands.json`에 전체 command, `report/summary.{json,csv,md}`에 집계를 저장한다.
미완료 cell을 명시하며 old/vLLM은 retained, current만 fresh로 기록한다.
Full12가 완료되기 전에는 이 절을 전체108회 완료의 증거로 인용하지 않는다.

## Remaining gates

- Same-binary legacy eligibility off/on의 선택 차이와 성능을 분리한다.
- Full12의 throughput뿐 아니라 TTFT/TPOT/E2E mean/p95, hash, memory를 확인한다.
- Old/current의 cross-engine greedy 차이가 있는 상태에서 KV-only speedup을 주장하지 않는다.
- Clean upstream metadata implementation과 직접 비교하려면 별도 isolated 실험이 필요하다.

## 자동 후속 실행

`finalize_parity_suite.py`가 모든36개 cell의 repeats=3 aggregate를 확인한 뒤 final report를
생성하고 GPU memory가50 MiB 미만이 될 때까지 확인한다. 이어서 실제 clean/current
`kvPageTable.cpp`를 각각 링크한 `kvPageTableBench.cpp`를 각5회 교대 실행한다.
각 binary는 같은 current core archive의 Tensor 구현을 사용한다. clean/current 사이
`common/tensor.{cpp,h}`의 diff는 없음을 확인했다. clean checkout은 수정하지 않는다.

Standalone 조건: BS1/8/32/64 × dirty rows=0/1/all, warmup100 + sample1000,
16 update마다 명시적 stream sync. setRow+upload host median/p95를 측정하고 마지막 K/V
page ID의 device 값을 검증한다. P/D lease benchmark와 달리 underlying page-table API만 비교한다.
Dirty rows가1인 BS1과all인 BS1은 같은 작업의 중복 측정이다. 실제 KV payload/attention은 없다.

모든 HTTP와10개 isolated 실행이 성공해야 `completion.json`을 쓴다. GPU가 다른 작업에 사용
중이면 isolated 실험은 실패로 멈추며 다른 프로세스를 강제 종료하지 않는다. Build command:

```bash
for variant in upstream-v0101 v0101-forward-port; do
  g++ -O3 -std=c++17 -I/local/$variant/cpp -I/opt/tensorrt/include \
    -I/usr/local/cuda/include \
    /local/v0101-forward-port/benchmarks/phase_serving/kvPageTableBench.cpp \
    /local/$variant/cpp/runtime/state/kvPageTable.cpp \
    /local/v0101-forward-build-make/cpp/libedgellmCore.a \
    -L/usr/local/cuda/lib64 -L/usr/local/cuda/lib64/stubs -L/opt/tensorrt/lib \
    -lcudart -lcuda -lnvinfer \
    -o /local/results/v0101-forward-port/legacy-parity-20260908/kv-table-$variant
done
```

TensorRT26.06 container에 repository `.local`을 `/local`로 mount한다. 이 standalone CPU build가
초기 V0 측정 중 수행되었으므로 전체 suite를 strict host-idle 실험이라고 부르지 않는다.
GPU 측정은 겹치지 않는다. 작은 차이는 이후 교대 검증이 필요하다.
