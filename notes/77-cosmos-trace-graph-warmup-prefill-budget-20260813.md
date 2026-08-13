# Cosmos short trace graph warmup과 prefill batch 확대

## 결론

Cosmos-Reason2-2B FP16 text serving의 short 48-request trace에서 두 병목을 분리해 확인했다.

1. 측정 중 CUDA graph capture를 trace replay로 측정 밖으로 옮기면 token/s가 `+6.92%` 증가한다.
2. `prefillTokenBudget`을 256에서 512로 올려 128-token chunk 네 개가 실제 P4를 형성하게 하면 cold 상태에서도
   token/s가 `+20.76%` 증가한다.
3. 둘을 합치면 기존 Current 대비 token/s `+26.98%`, TTFT p95 `-28.90%`, E2E p95 `-21.93%`다.

최종 short 결과는 1958.0 token/s이며 같은 trace의 vLLM 2021.4 token/s보다 3.14% 낮다. E2E p95 격차도
30.56%에서 2.13%로 줄었다. 다만 이 최종 수치는 동일 trace를 두 번 replay한 graph-warm upper bound다. 정확한 미래
trace를 알 수 없는 production 비교에는 cold budget-512 결과인 1862.2 token/s, vLLM 대비 -7.88%를 보수적인
기준으로 사용한다.

## 구현

`llm_phase_bench`에 `--traceWarmupRepeats N`, Cosmos matrix runner에 `--trace-warmup-repeats N`을 추가했다.
text-only real-request trace를 측정 전에 같은 production `PhaseAsyncServer`로 replay한다.

warmup epoch가 끝나면 다음 상태를 초기화한다.

- stable KV slot과 paged KV bundle은 모두 반환된 상태인지 기존 invariant로 검사한다.
- global slot length를 0으로 되돌린다.
- prefill/decode queue와 admission state가 비었는지 검사한다.
- scheduler EWMA, rolling TPOT p95, hysteresis, overlap debt, cohort history를 초기화한다.
- adaptive page-growth pressure와 lease count를 초기화한다.
- request/dispatch/kernel-group 측정 버퍼를 비운다.

반대로 prefill/decode `EngineExecutor`의 CUDA graph cache는 유지한다. 같은 server를 유지해 request ID는 단조 증가하며,
lifecycle의 request-ID 재사용 보호를 우회하지 않는다. 따라서 warmup이 측정 scheduler 결정에 학습 상태를 남기지 않고
graph cache만 남긴다.

v1은 CUDA graph가 켜진 text-only trace에만 허용한다. VLM encoder replay는 image preprocessing 및 encoder graph의
별도 epoch reset이 필요하므로 명시적으로 거부한다.

## 실험 조건

- GPU: RTX 3080 10GB
- model: `nvidia/Cosmos-Reason2-2B`, FP16 weights / FP16 KV
- engine: indexed-paged, max prefill 16 / max decode 64 / 80 slots / 256 page bundles
- runtime: shared CUDA primary context, independent prefill/decode TensorRT contexts, P4/D64
- chunk: fixed 128, ragged batching
- scheduler: `throughput-balanced`, direct-cost v5
- trace: 48 requests, 4312 prompt tokens, 1040 generated tokens, arrival offsets 고정
- 통계: 각 short 조건을 독립 프로세스로 3회 실행한 median

## short 3회 median

| Current 조건 | token/s | TTFT p95 | TPOT p95 | E2E p95 | 기존 대비 token/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| budget 256, cold graph | 1542.0 | 408.18ms | 17.017ms | 645.51ms | 기준 |
| budget 256, trace warmup 2 | 1648.7 | 395.60ms | 16.381ms | 603.65ms | +6.92% |
| budget 512, cold graph | 1862.2 | 295.69ms | 16.078ms | 531.34ms | +20.76% |
| **budget 512, trace warmup 2** | **1958.0** | **290.21ms** | **15.440ms** | **503.98ms** | **+26.98%** |
| vLLM warm server | 2021.4 | 258.58ms | 25.493ms | 493.46ms | - |

budget 512 cold 대비 warmup 2회의 추가 효과는 token/s `+5.15%`, E2E p95 `-5.15%`다. 최종 Current는
vLLM보다 TPOT p95가 39.43% 낮지만 TTFT p95는 12.23% 높다. 남은 short 격차는 decode가 아니라 admission 직후
prefill 시작 시간과 initial-prefill kernel/layout에 집중되어 있다.

## 실제 batch formation 변화

budget 256에서는 24번의 prefill dispatch 중 P2가 16번이고 P4는 4번뿐이었다. 128-token chunk 두 개가 이미
256-token budget을 소진하기 때문이다.

budget 512에서는 15번의 prefill dispatch 중 P4가 13번이다. prefill dispatch 수가 24에서 15로 줄고, 각 요청의
prompt가 더 적은 wave로 decode queue에 진입한다. 이 변화가 graph warmup보다 큰 개선을 만들었다.

즉 short workload의 우선순위는 단순히 P를 작게 유지하는 것이 아니다. GPU interference가 아직 낮은 initial burst에서는
P4/512를 사용하고, decode pressure가 생긴 뒤에는 기존 direct-cost/hysteresis guard가 overlap을 제한하는 편이 낫다.

## 긴 workload 회귀

trace warmup 2회를 각각 한 번 적용하고 기존 cold 3회 median과 비교했다.

| workload | cold token/s | warm token/s | 변화 | TTFT p95 변화 | TPOT p95 변화 | E2E p95 변화 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| balanced 288 | 4114.0 | 4165.5 | +1.25% | -0.03% | +0.22% | -1.21% |
| decode-heavy 288 | 4646.5 | 4665.8 | +0.42% | -0.11% | +0.21% | -0.46% |

긴 trace에서는 capture 비용이 전체 실행에 희석되므로 효과가 작고 tail 회귀도 없다.

## 메모리와 해석 한계

short budget-512 cold 실행은 종료 시 9606.9MiB, graph replay 2회는 9662.9MiB를 사용했다. graph cache가
prefill/decode 2/11개에서 4/22개로 늘면서 약 56MiB가 추가됐다. 사용자는 이번 단계에서 메모리보다 latency/throughput을
우선했으므로 이 설정을 허용했지만, 10GB 장치의 free headroom은 약 211MiB에 불과하다.

exact-trace replay는 graph cache 최적화의 상한선을 찾는 benchmark 기능이지 미래 shape를 미리 아는 production
기능이 아니다. production에서는 최근 shape histogram이나 정해진 P/D bucket manifest로 대표 shape를 warmup해야 한다.
다음 구현은 이 결과를 이용해 trace 내용 대신 `{phase, batch, chunk/context bucket}` 목록을 입력받는 profile-driven
graph priming API로 일반화한다.

## 다음 단계

1. short preset의 prefill token budget을 512로 분리하고 balanced/decode-heavy는 256을 유지한다.
2. exact replay를 profile-driven P/D shape manifest warmup으로 바꿔 production에서 재현 가능하게 한다.
3. short의 남은 TTFT 12.23% 격차를 initial-prefill prepare/engine/sample kernel-group으로 분해한다.
4. dense ragged `[B,Smax]`를 true packed/varlen layout으로 바꿔 P4/512의 padding과 graph shape 수를 줄인다.

artifact:

- `.local/cosmos-reason2-2b/trace-graph-warmup-20260813/`
- `.local/vllm-cosmos-reason2-2b/latest-comparison-20260813/short/`
