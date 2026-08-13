# Cosmos와 vLLM의 남은 성능 차이

## 결론

현재 Cosmos-Reason2-2B FP16 경로는 더 이상 모든 workload에서 vLLM보다 느린 상태가 아니다. 기존 공정 비교에서
balanced generated-token 처리량 차이는 `-0.50%`, decode-heavy는 `+4.75%`였고, 이후 true packed prefill은
indexed-paged baseline 대비 balanced `+7.14%`, decode-heavy `+2.73%`를 추가로 얻었다. CUDA graph priming과
D64 engine도 짧은 trace의 처리량을 더 높였다.

따라서 다음 목표를 단순히 kernel을 더 빠르게 만드는 것으로 잡으면 안 된다. 남은 차이는 주로 다음 네 곳에 있다.

1. heterogeneous continuous batching과 admission
2. 두 TensorRT execution context 및 phase I/O의 메모리 비용
3. shape별 CUDA graph coverage와 cold capture
4. prefix reuse, copy-on-write, production frontend 같은 serving 기능

## 새 long-KV decode cost table

P8/D64, 80 stable slots, 256 page bundles, FP16 indexed-paged KV engine에서 isolated decode를 측정했다. 각 row는
warmup 20회 뒤 100회 CUDA-event E2E sample의 median이다. CUDA graph와 일반 enqueue를 모두 측정했다.

| past KV | feasible decode BS | graph median | graph token/s | enqueue median | graph speedup |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 16 | 6.477ms | 2,470 | 6.836ms | 5.54% |
| 128 | 24 | 6.853ms | 3,502 | 7.191ms | 4.93% |
| 128 | 32 | 7.134ms | 4,486 | 7.482ms | 4.88% |
| 128 | 48 | 7.772ms | 6,176 | 8.117ms | 4.44% |
| 128 | 64 | 8.364ms | 7,652 | 8.717ms | 4.23% |
| 512 | 16 | 7.383ms | 2,167 | 7.730ms | 4.70% |
| 512 | 24 | 8.134ms | 2,951 | 8.494ms | 4.43% |
| 512 | 32 | 8.866ms | 3,609 | 9.216ms | 3.95% |
| 512 | 48 | 10.329ms | 4,647 | 10.695ms | 3.54% |
| 1024 | 16 | 8.681ms | 1,843 | 9.034ms | 4.06% |
| 1024 | 24 | 10.081ms | 2,381 | 10.452ms | 3.68% |
| 1536 | 16 | 10.001ms | 1,600 | 10.348ms | 3.46% |

표의 token/s는 `batch_size / median_seconds`다. 기존 `llm_bench` summary의 decode token/s는 batch size를
곱하지 않은 per-row 표기라 이 분석에는 사용하지 않았다.

## page-pool 제약

page 크기는 128 tokens이고 pool은 256 bundles다. 동일 길이 요청만 넣은 synthetic batch에서 다음 조합은
측정 전에 capacity-limited로 판정된다.

| past KV | 요청당 필요한 bundle | 가능한 이론상 최대 BS | 측정 결과 |
| ---: | ---: | ---: | --- |
| 128 | 2 | 128 | engine D64까지 가능 |
| 512 | 5 | 51 | D48 가능, D64 불가 |
| 1024 | 9 | 28 | D24 가능, D32 이상 불가 |
| 1536 | 13 | 19 | D16 가능, D24 이상 불가 |

이 결과는 `maxDecodeBatchSize=64`가 모든 KV 길이에서 보장되는 batch가 아니라는 뜻이다. 실제 online batch는
KV 길이가 섞여 있으므로 `max(KV)`만으로 cap을 정해서도 안 된다. 예를 들어 KV1024 한 요청과 KV128 31개인
D32는 32개의 KV1024 요청보다 page와 attention work가 훨씬 작다.

현재 scheduler cost key는 `(batch size, max context length)`다. 다음 버전은 최소한
`(batch size, total context tokens, max context length)`로 바꾸고, page reservation 합과 graph shape hit 여부를
함께 봐야 한다. 이것이 vLLM의 block-aware continuous batching과 가장 가까운 다음 개선이다.

## 무엇이 이미 경쟁력 있는가

- decode-heavy workload에서는 기존 비교만으로도 generated token/s와 TPOT이 vLLM보다 좋았다.
- packed prefill은 prefill engine median을 약 26.5~27.6% 줄였다.
- isolated decode에서 D16에서 D64로 갈수록 처리량이 계속 증가한다. D64 profile 자체는 유효하다.
- CUDA graph는 long-KV 전 구간에서 약 3.5~5.5%의 반복 가능한 이득을 준다.
- indexed-paged KV는 eviction KV copy를 없애고 stable ownership과 phase overlap의 기반을 제공한다.

## vLLM보다 부족한 부분

### 1. Heterogeneous continuous batching

현재 phase queue는 prefill/decode를 독립적으로 잘 묶지만 decode batch cost를 row 수와 최대 KV로 근사한다. vLLM은
매 scheduling iteration마다 실행 가능한 sequence와 token/block budget을 함께 보고 batch를 다시 만든다. 우리도
active row별 context length 합, page bundle 합, graph-covered batch shape를 입력으로 삼아야 한다.

### 2. Admission이 지나치게 보수적인 구간

현재 whole-request page reservation은 runtime OOM을 안전하게 막지만, 아직 생성하지 않은 먼 미래 output page까지
잡아 queue를 키울 수 있다. growth lease를 작은 단위로 주고 low watermark에서 다음 bundle을 보충하면 같은 pool로
더 많은 short request를 active 상태에 둘 수 있다. 단, decode 직전에 page가 없어지는 상황은 backpressure로 막아야
한다.

### 3. 메모리 footprint

동일 raw KV 3,584MiB 조건에서 Current는 vLLM보다 약 1.4~1.5GiB를 더 사용했다. 주원인은 두 independent
TensorRT context workspace, phase별 최대-shape I/O/deepstack buffer, CUDA graph cache다. 이는 더 큰 page pool과
graph coverage를 동시에 제한한다. profile별 I/O allocation, deepstack zero-view 공유, context workspace/build tactic
상한 조정이 다음 메모리 작업이다.

### 4. Graph lifecycle

graph 자체는 이번 측정에서 확실히 3.5~5.5% 이득이다. 문제는 online shape가 많을 때 cold capture 비용과 graph
memory가 증가한다는 점이다. 자주 쓰는 D batch bucket만 미리 capture하고 나머지는 enqueue fallback으로 두는
bounded LFU/LRU 정책이 필요하다.

### 5. Prefix reuse와 production 경로

현재 비교는 prefix cache를 양쪽 모두 꺼 공정성을 맞췄다. production에서는 vLLM의 prefix caching과 block
copy-on-write가 반복 system prompt/RAG workload에서 큰 이점이다. indexed-paged allocator 위에 immutable prefix
page의 reference count와 tail-page copy-on-write를 추가해야 한다. 또한 Current는 direct C++ trace injection,
vLLM은 localhost HTTP streaming이므로 동일 frontend가 붙기 전까지 production E2E 우위라고 단정하면 안 된다.

## 성능을 더 끌어올리는 순서

1. decode snapshot에 `totalContextTokens`와 `required/available page bundles`를 추가한다.
2. long-KV cost table을 batch/total-KV bucket cost model로 바꾸고 dynamic decode selection에 연결한다.
3. whole-request reservation과 growth lease를 workload별로 A/B해 active sequence 수와 page-block wait를 줄인다.
4. graph는 D8/16/24/32/48/64 중 실제 빈도가 높은 shape만 priming하고 memory budget 내에서 교체한다.
5. prefill은 128-contract graph engine과 256-contract graph-off engine을 workload router로 선택한다.
6. phase별 I/O/workspace 중복을 줄인 뒤 확보한 메모리를 page pool 또는 graph cache에 재투자한다.
7. immutable prefix page sharing/COW를 구현하고 반복-prefix trace에서 별도 비교한다.
8. 마지막으로 동일 HTTP/streaming frontend와 EOS 정책으로 vLLM을 재측정한다.

원시 결과는 `.local/cosmos-reason2-2b/long-kv-decode-cost-d64-20260813/decode-cost-table.csv`에 있다.
