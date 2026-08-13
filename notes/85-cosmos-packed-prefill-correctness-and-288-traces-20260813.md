# Cosmos packed prefill 정확성 gate와 288-request trace 결과

## 결론

Cosmos-Reason2-2B FP16의 기존 indexed-paged engine과 true packed-prefill engine을 동일한 P8/D32,
`maxBatchSize=32`, 32 stable slots, 256 page bundles, independent TensorRT contexts 조건으로 비교했다.

먼저 baseline이 생성한 16-token trajectory를 packed engine에 teacher-force하고 매 step의 151,936개
full-vocabulary logits를 비교했다. 16/16 step에서 top-1과 sampled token이 모두 일치했고 최저 cosine은
0.99997389로 0.999 gate를 통과했다. 별도 TensorRT build의 tactic 차이로 greedy text가 달라질 수 있다는 이전
불확실성을 수치 tolerance gate로 해소했다.

그 다음 288-request real-arrival trace를 balanced와 decode-heavy 각각 3회 실행했다. 3회 결과의 median에서
packed는 balanced 생성 처리량을 7.14% 높이고 TTFT p95를 7.62%, TPOT p95를 6.94%, E2E p95를 7.27%
낮췄다. decode-heavy에서도 생성 처리량 +2.73%, TTFT p95 -3.18%, TPOT p95 -2.91%, E2E p95
-2.89%로 이득을 유지했다.

## 정확성 검증

검증 입력은 한 개의 text request, greedy sampling, 16 output token이다. baseline dump에서 생성 token을
추출하고 packed 실행의 decode input으로 강제해 두 engine이 완전히 같은 autoregressive trajectory를 따라가게
했다. CUDA graph는 끄고 logits-only dump를 사용했다.

| metric | result | gate |
| --- | ---: | ---: |
| compared decode steps | 16 | 16 |
| full vocabulary per step | 151,936 | 동일 shape |
| worst cosine | 0.99997389 | >= 0.999 |
| largest absolute error | 0.234375 | 관찰값 |
| largest mean absolute error | 0.0305828 | 관찰값 |
| minimum close fraction (`atol=rtol=0.01`) | 0.766046 | 참고값 |
| top-1 matches | 16/16 | 16/16 |
| sampled-token matches | 16/16 | 16/16 |

`min_close_fraction`은 gate로 강제하지 않았다. FP16 tactic/reduction 순서 차이는 많은 작은 element-wise 차이를
만들지만 cosine과 모든 top-1 선택이 안정적이기 때문이다. artifact는
`.local/cosmos-reason2-2b/packed-prefill-correctness-20260813/`에 있다.

정확성 분석 도구 `scripts/cosmos_reason2/compare_indexed_logits.py`는 PyTorch tensor 연산을 NumPy로 교체했다.
따라서 4GB급 PyTorch package 없이 runtime container의 NumPy와 작은 `safetensors` package만으로 logits dump를
분석할 수 있다.

## public text-only Cosmos 경로

`LLMInferenceRuntime` 생성자는 `numDeepstackFeatures > 0`인 model config만 보고 visual engine directory를
무조건 요구했다. 그래서 실제 입력이 text-only여도 Cosmos public `handleRequest()`가 시작 전에 거부됐다.
이 선제 검사를 제거했다.

안전성 검사는 사라진 것이 아니다. request validation은 image/audio/trajectory buffer가 실제로 들어왔을 때 해당
runner가 없으면 계속 실패한다. deepstack tensor는 text-only prefill에서 기존대로 zero-fill된다. 이 변경으로
Cosmos public runtime의 text-only logits dump가 encoder engine 추가 적재 없이 성공했고, 10GB GPU에서 visual
engine을 함께 적재할 때 발생하던 OOM도 피했다.

## 288-request 측정 조건

- GPU: RTX 3080 10GB
- model/KV: Cosmos-Reason2-2B FP16, indexed-paged FP16 KV, 128 tokens/page
- engine: P8/D32, `maxBatchSize=32`, independent prefill/decode TensorRT contexts
- scheduler: fixed-128 chunk, ragged prefill batching, prefill token budget 1,024
- overlap cap: prefill tokens 1,024
- slots/page pool: 32 slots, 256 page bundles, full request reservation
- CUDA graph: off
- EOS: ignored, configured output work를 모두 실행
- arrival: source trace의 `arrival_offset_us` 보존
- 반복: workload별 동일 trace 3회

balanced trace는 output token 합 24,960, request별 최대 128이다. decode-heavy trace는 output token 합 74,880,
request별 최대 384다. 두 trace 모두 288 request이며 약 0.3초 안에 arrival이 몰리는 overload 형태다.

## 3회 median 결과

### Balanced

| metric | indexed-paged baseline | packed prefill | change |
| --- | ---: | ---: | ---: |
| duration | 9112.7 ms | 8505.8 ms | -6.66% |
| request/s | 31.60 | 33.86 | +7.14% |
| generated token/s | 2739.0 | 2934.5 | +7.14% |
| TTFT median | 3715.0 ms | 3425.5 ms | -7.79% |
| TTFT p95 | 7522.5 ms | 6948.9 ms | -7.62% |
| TPOT median | 11.173 ms | 10.392 ms | -6.99% |
| TPOT p95 | 11.501 ms | 10.702 ms | -6.94% |
| E2E median | 4657.2 ms | 4303.0 ms | -7.60% |
| E2E p95 | 8384.3 ms | 7774.9 ms | -7.27% |

### Decode-heavy

| metric | indexed-paged baseline | packed prefill | change |
| --- | ---: | ---: | ---: |
| duration | 23047.2 ms | 22434.1 ms | -2.66% |
| request/s | 12.50 | 12.84 | +2.73% |
| generated token/s | 3249.0 | 3337.8 | +2.73% |
| TTFT median | 9421.0 ms | 9145.1 ms | -2.93% |
| TTFT p95 | 19074.6 ms | 18467.5 ms | -3.18% |
| TPOT median | 9.297 ms | 9.041 ms | -2.75% |
| TPOT p95 | 9.432 ms | 9.158 ms | -2.91% |
| E2E median | 11705.4 ms | 11357.8 ms | -2.97% |
| E2E p95 | 21453.3 ms | 20833.7 ms | -2.89% |

모든 실행에서 실제 P=8과 D=32가 관찰됐다. 세 번 합산 dispatch는 balanced에서 2,622회, decode-heavy에서
7,800회이며 prefill/decode overlap dispatch는 각각 baseline/packed 603/603과 525/531회였다. 즉 이 결과는
P8/D32를 설정만 한 synthetic 표가 아니라 두 최대 batch가 실제 형성된 online queue 실행이다.

## kernel-group 해석

workload별 세 run의 kernel-group median을 다시 median한 값은 다음과 같다.

| workload | engine | prefill engine median | prefill p95 | decode engine median | decode p95 |
| --- | --- | ---: | ---: | ---: | ---: |
| balanced | baseline | 17.893 ms | 22.898 ms | 7.412 ms | 10.443 ms |
| balanced | packed | 13.051 ms | 15.178 ms | 7.442 ms | 11.642 ms |
| decode-heavy | baseline | 18.013 ms | 23.107 ms | 7.788 ms | 10.349 ms |
| decode-heavy | packed | 13.245 ms | 15.573 ms | 7.810 ms | 11.404 ms |

packed의 직접 이득은 prefill median 약 26.5~27.6% 감소다. packed engine의 decode median은 0.3~0.4% 느리고
decode p95도 더 크다. 그런데 E2E TPOT은 좋아졌다. prefill 점유 시간이 줄면서 independent decode stream이 받는
간섭과 대기 시간이 감소했기 때문이다. 따라서 이 결과를 “decode kernel 자체 최적화”로 해석하면 안 된다.

balanced가 decode-heavy보다 크게 개선되는 것도 같은 이유다. 전체 work에서 prefill 비중이 클수록 packed
prefill의 직접 절감이 E2E에 더 크게 반영된다. 출력이 3배인 decode-heavy에서는 decode가 지배하므로 이득이
약 3%로 수렴한다.

## page pressure와 backpressure

baseline과 packed의 KV allocation은 동일하다. balanced의 peak는 68/256 bundles(26.56%), decode-heavy는
120/256 bundles(46.88%)였다. 따라서 이번 trace에서 page pool exhaustion은 없었고 packed가 KV memory를
추가 절감한 것도 아니다.

288개 요청이 약 0.3초 안에 도착하므로 observed pending queue peak는 255였다. pressure model의 256 block
event는 slot 또는 page 조건을 함께 세는 값이다. page peak가 50% 미만이므로 실제 admission backpressure의
주원인은 32 stable slots와 service rate보다 빠른 burst arrival이다. 다음 실험에서 page pool을 더 늘려도 이
queue peak는 크게 줄지 않으며, slots/admission policy나 arrival load를 함께 바꿔야 한다.

## 검증

- Cosmos text-only public `llm_inference`: baseline/packed 각각 16-token 실행 성공
- full-vocabulary teacher-forced logits comparison: PASS
- Python unit test: 2 passed
- C++ rebuild: `llm_inference`, `llm_phase_bench` 성공
- real-request A/B: 2 workloads x 3 repeats x 2 engines, 전부 return code 0
- pre-commit: 수정 파일 전체 PASS

원시 benchmark artifact는 `.local/cosmos-reason2-2b/packed-prefill-ab-20260813/`, tracked run 결과는
`notes/results/cosmos-packed-prefill-288-ab-20260813.csv`에 있다.

## 다음 단계

1. cost model schema에 `prefill_layout=dense|packed` 축을 추가한다.
2. packed P1/P2/P4/P8와 prefix bucket별 직접 prefill cost table을 생성한다.
3. packed overlap P/D 조합의 direct-cost table을 별도로 측정한다. 현재 dense table을 packed scheduler에 그대로
   사용하면 prefill 비용을 과대평가한다.
4. 새 packed cost table을 throughput-balanced dynamic P/D scheduler에 연결하고 fixed P8/D32와 비교한다.
5. decode-heavy에서 관찰된 packed engine의 decode p95 증가를 timing cache/tactic replay engine으로 분리해,
   engine build noise와 true interference를 구분한다.
