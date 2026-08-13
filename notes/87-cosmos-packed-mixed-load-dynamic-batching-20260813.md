# Cosmos packed mixed-load dynamic batching 분석

## 결론

동일 packed P8/D32 engine에 저부하, burst, 회복 부하가 연속되는 288-request trace를 투입했다. fixed와
dynamic decode/prefill을 각각 분리해 3회 측정한 결과, 현재 production 후보는 **fixed packed prefill + dynamic
decode**다.

dynamic decode-only는 fixed 대비 전체 처리량 `+0.019%`, TTFT p95 `+0.014%`, TPOT p95 `+0.006%`로 사실상
동일했다. 반면 dynamic prefill-only는 TTFT p95 `+2.55%`, TPOT p95 `+3.00%`, E2E p95 `+2.30%` 손실을
만들었다. 두 축을 함께 켜도 손실이 남으므로 dynamic prefill은 기본 정책으로 승격하지 않는다.

## Mixed-load trace

새 `build_mixed_load_trace.py`는 실제 source request의 prompt와 output length를 유지하면서 구간별 Poisson arrival를
결정적인 seed로 생성한다.

| phase | requests | arrival rate | 앞 구간과 gap | 실제 arrival 범위 |
| --- | ---: | ---: | ---: | ---: |
| low | 24 | 10 req/s | 0 | 0~3.004s |
| burst | 192 | 1,000 req/s | 250ms | 3.254~3.448s |
| recovered | 72 | 25 req/s | 6.5s | 9.948~12.723s |

모든 정책은 Cosmos-Reason2-2B FP16, packed prefill, indexed-paged FP16 KV, independent TensorRT contexts,
32 stable slots, 256 page bundles, fixed chunk 128, P8/D32 cap, CUDA graph off 조건이다.

## Cost model 수정

기존 decode cost는 prefill과 겹친 dispatch까지 모두 합쳐 p95를 계산했다. overlap slowdown은 별도 table에도
존재하므로 decode base cost에 간섭이 중복 반영됐다. 특히 128-context D4/D8/D16의 소수 overlap sample이 p95를
부풀려 dynamic decode가 D4를 과도하게 고르는 문제가 있었다.

schema v6 builder는 동일 `(batch, context)`의 decode-only sample을 우선한다. decode-only shape가 전혀 없을 때만
`all_dispatch_fallback`을 사용하고 각 point에 `sample_scope`를 기록한다. 짧은 prompt 32개 통제 trace를 추가해
D32/128 decode-only sample 276개를 확보했다.

또한 dynamic prefill cost lookup은 실제 overlap일 때만 planned decode batch coverage를 요구한다. prefill-only
dispatch는 D0 cost를 사용한다. 이전에는 decode queue가 존재한다는 이유만으로 겹치지 않는 P8 prefill도 D32
coverage를 요구해 P4 이하로 축소될 수 있었다.

late-prefill probe를 추가한 최종 packed cost table은 prefill 93점, decode 11점, direct overlap 85점이다.
P4/D32 initial과 P8/D32 continuation을 포함한다.

## 정책별 3-run median

| policy | generated tok/s | TTFT p95 | TPOT p95 | E2E p95 |
| --- | ---: | ---: | ---: | ---: |
| fixed P/D | 1,839.17 | 4,432.67ms | 10.784ms | 5,270.42ms |
| dynamic decode-only | 1,839.52 | 4,433.30ms | 10.785ms | 5,276.88ms |
| dynamic prefill-only | 1,839.72 | 4,545.57ms | 11.108ms | 5,391.80ms |
| dynamic P+D | 1,839.19 | 4,521.12ms | 11.014ms | 5,371.61ms |

전체 wall duration은 마지막 recovered arrival 시각의 영향을 크게 받으므로 처리량 차이가 압축된다. request tail과
구간별 결과를 함께 봐야 한다.

### Dynamic decode-only 구간별 변화

| phase | TTFT p95 | TPOT p95 | E2E p95 |
| --- | ---: | ---: | ---: |
| low | -1.79% | -0.08% | +1.65% |
| burst | +0.02% | +0.06% | +0.11% |
| recovered | -2.78% | +0.82% | -0.34% |

dynamic decode-only는 부하 전환에도 fixed와 거의 같은 안정성을 보였다. 현 cost curve에서 D32가 대부분 가장 효율적이라
큰 speedup을 만들지는 않지만, sparse/low-load에서 작은 runnable batch를 자연스럽게 사용하면서 burst 회귀가 없다.

dynamic prefill은 비용표가 선택한 작은 P가 개별 kernel latency를 줄이는 대신 burst의 prefill queue를 오래 유지해
TTFT와 decode interference 시간을 증가시켰다. packed prefill은 이미 P8 efficiency가 좋으므로 지금 workload에서는
fixed 최대-compatible batch가 더 적합하다.

원시 artifact는 `.local/cosmos-reason2-2b/packed-mixed-load-20260813/`에 있다.

## 다음 단계

1. production 기본 후보를 fixed packed prefill + dynamic decode로 두고 balanced/decode-heavy 288 trace 회귀를
   확인한다.
2. dynamic prefill은 queue backlog와 predicted drain time을 score에 포함하기 전까지 opt-in 실험 기능으로 유지한다.
3. P8 initial direct-overlap을 강제로 만들 수 있는 dispatch-control microbenchmark를 추가한다.
4. packed D64/80-slot engine을 만들 수 있으면 같은 mixed-load trace에서 D32와 비교한다.
