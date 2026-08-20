# Cosmos v0.10 B16/B32 확장과 production gate

## 결론

이번 단계에서는 오래된 ONNX artifact 때문에 B8에 막혔던 문제를 제거하고, 동일 Cosmos-Reason2-2B FP16
모델로 independent prefill/decode 경로를 B16과 B32까지 확장했다.

- fresh ONNX의 `last_token_ids`는 `[token_batch, num_selected]` 두 축이 동적이다.
- B16/KV2048에서는 P8/D16이 1,300.4 generated token/s를 기록했다.
- B32/KV1024에서는 P8/D32가 1,921.4 generated token/s를 기록했다.
- B32는 B16 P8/D16보다 처리량이 47.8% 높고 TTFT median이 36.5% 낮다.
- 256개의 128-token KV page bundle을 유지하면서 B32 ready/peak 메모리는 8,459/8,465 MiB다.
- 실제 두 image HTTP 요청은 request별 prefill row를 분리한 뒤 2/2 완료했다.
- tied engine은 artifact 계약을 검사한 뒤 실제 semantic inference를 통과하며, 계약이 없는 이전 artifact는 거부한다.

다만 fixed-output 동일 HTTP trace에서 vLLM보다 처리량은 아직 45.6~57.0% 낮다. Current의 긴 요청 TPOT은
vLLM보다 좋은 경우가 많지만 D32 admission wave 때문에 TTFT가 훨씬 길다. 다음 성능 우선순위는 scheduler 미세 조정이
아니라 D48/D64 active-row 확장과 prefill admission 지연 축소다.

## 실행 계약

- GPU: RTX 3080 10GB, SM86
- TensorRT/CUDA: 11.0.0 / 13.3
- model: `nvidia/Cosmos-Reason2-2B`, FP16, language-model-only 비교
- KV: FP16 indexed-paged, 128-token page bundle, pool 256 pages
- stable slots: 80
- packed/chunked prefill: 최대 chunk 128
- 성능 gate: EOS 무시, 요청된 output token 수까지 고정
- 각 primary 결과: 새 process lifecycle 3회 중앙값
- vLLM: 같은 모델과 FP16 KV, KV budget 3,758,096,384 bytes, chunked prefill, prefix cache off

`last_token_ids` 문제는 exporter source 결함이 아니었다. 예전 ONNX는 두 번째 축이 1로 고정되어 있었지만 현재
export를 다시 수행하자 `token_batch`, `num_selected`가 모두 동적으로 생성되었다. fresh shared ONNX SHA-256은 다음과
같다.

| artifact | SHA-256 |
| --- | --- |
| `model.onnx` | `85258a400c2d9b1a8737c9bf911624a25783e1c8d73129a6fd6cc7e755818832` |
| `model.onnx.data` | `9756b1c94bbeaf16b490775001410b4b1af531f11c619de42eba211bc30a0786` |

## 엔진과 메모리

| engine | common max batch | max KV/request | KV pages | selected P/D | ready/peak |
| --- | ---: | ---: | ---: | --- | ---: |
| B16 tied | 16 | 2,048 | 256 | P8/D16 | 9,293/9,299 MiB |
| B32 tied | 32 | 1,024 | 256 | P8/D32 | 8,459/8,465 MiB |
| vLLM | max seq 80 | max model 2,048 | matched bytes | continuous | 7,953/7,959 MiB |

Cosmos의 한 128-token page bundle은 모든 28 layer의 K/V를 포함해 약 14 MiB다.

```text
28 layers * 2(K,V) * 8 KV heads * 128 tokens * 128 head dim * 2 bytes
= 14,680,064 bytes/page bundle
```

따라서 256 pages는 약 3.5 GiB이고 vLLM과 같은 수준의 KV budget이다. B32 Current가 vLLM보다 peak 기준
506 MiB 더 쓰는 주된 이유는 independent TensorRT execution context 두 개의 workspace와 phase별 I/O buffer다.
반대로 B32가 B16보다 834 MiB 적은 것은 KV pool을 줄인 결과가 아니다. 두 엔진 모두 page pool은 256개이며,
B32 build가 더 작은 TensorRT tactic/workspace를 선택했고 request별 최대 KV 계약도 1,024로 줄었다.

메모리 면에서 B32는 약 1.78 GiB headroom을 남겨 이후 B48/D48 또는 graph cache 실험 여지가 있다. 단,
serialized engine build별 tactic 차이가 크므로 엔진 크기만으로 runtime memory를 추정하면 안 된다.

## 비대칭 batch scaling

balanced trace는 run마다 288 requests, prompt 25,872 tokens, output 24,960 tokens로 고정했다.

| configuration | token/s | TTFT med/p95 | TPOT med/p95 | E2E med/p95 |
| --- | ---: | ---: | ---: | ---: |
| B16 P8/D8 | 869.8 | 13,246/25,685 ms | 17.55/18.46 ms | 14,760/27,233 ms |
| B16 P4/D16 | 1,297.8 | 8,802/17,136 ms | 11.78/12.58 ms | 9,874/18,126 ms |
| B16 P8/D16 | **1,300.4** | 8,841/17,092 ms | **11.77/12.57 ms** | 9,861/18,161 ms |
| B32 P8/D32 | **1,921.4** | **5,612/11,112 ms** | 15.66/16.88 ms | **6,842/12,188 ms** |

해석은 다음과 같다.

- B16에서 P4와 P8의 처리량 차이는 0.2%뿐이다. decode batch가 병목인 포화 workload에서는 P를 키우는 것보다
  D16을 채우는 것이 중요하다.
- P4/D16은 TTFT median이 P8/D16보다 39 ms 빠르지만 p95와 E2E는 P8/D16이 조금 낫다. 기본값은 P8/D16이다.
- D32는 한 decode step 자체가 D16보다 느리지만 한 번에 처리하는 row 수가 두 배여서 전체 admission wave 수를 줄인다.
- B32 TPOT가 B16보다 나빠진 것은 larger batch의 개별 step 비용과 1,024-token cost profile 차이 때문이다. 대신
  queue 대기와 TTFT가 크게 줄어 E2E가 좋아진다.

## CUDA-event decode cost table

아래 값은 각각의 profile에서 측정한 kernel-group CUDA-event p95이며 scheduler의 초기 cost table로 연결했다.

| D batch | context contract | p95 ms | D batch | context contract | p95 ms |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2,048 | 6.238 | 17 | 1,024 | 6.927 |
| 2 | 2,048 | 6.294 | 18 | 1,024 | 6.988 |
| 3 | 2,048 | 6.370 | 19 | 1,024 | 7.016 |
| 4 | 2,048 | 6.321 | 20 | 1,024 | 7.129 |
| 5 | 2,048 | 6.402 | 21 | 1,024 | 7.105 |
| 6 | 2,048 | 6.360 | 22 | 1,024 | 7.082 |
| 7 | 2,048 | 6.467 | 23 | 1,024 | 7.106 |
| 8 | 2,048 | 6.467 | 24 | 1,024 | 7.178 |
| 9 | 2,048 | 6.680 | 25 | 1,024 | 7.291 |
| 10 | 2,048 | 6.617 | 26 | 1,024 | 7.278 |
| 11 | 2,048 | 6.767 | 27 | 1,024 | 7.341 |
| 12 | 2,048 | 6.712 | 28 | 1,024 | 7.264 |
| 13 | 2,048 | 6.924 | 29 | 1,024 | 7.374 |
| 14 | 2,048 | 6.790 | 30 | 1,024 | 7.435 |
| 15 | 2,048 | 6.904 | 31 | 1,024 | 7.523 |
| 16 | 2,048 | 6.862 | 32 | 1,024 | 7.484 |

cost는 완전히 단조롭지 않다. TensorRT tactic과 GPU scheduling noise 때문에 인접 batch가 역전될 수 있으므로,
scheduler는 선형 공식 대신 관측 table과 EWMA를 사용한다. 포화 fixed-output trace에서는 decode queue target을
50 ms로 두어 upper bucket이 형성되게 했다. interactive workload에서는 이 값을 그대로 사용하면 안 된다.

## 다섯 real-request workload와 vLLM

같은 materialized trace와 fixed output work를 사용했다. Current는 각 workload에 검증된 P/D 설정을 사용했고
vLLM은 option sweep에서 선택한 auto-tuned/token-budget-1024 설정이다.

| workload | Current token/s | vLLM token/s | gap | Current TTFT med/p95 | vLLM TTFT med/p95 | Current TPOT med/p95 | vLLM TPOT med/p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| short | 859.7 | 1,998.7 | -57.0% | 280/832 ms | 160/259 ms | 28.09/42.90 ms | 12.35/23.20 ms |
| balanced | 1,921.4 | 4,234.4 | -54.6% | 5,612/11,112 ms | 1,859/4,389 ms | 15.66/16.88 ms | 17.06/17.89 ms |
| decode-heavy | 2,554.5 | 4,797.9 | -46.8% | 12,106/24,559 ms | 4,503/11,565 ms | 11.72/12.09 ms | 14.86/15.40 ms |
| long-prefill | 609.0 | 1,119.8 | -45.6% | 19,461/37,504 ms | 10,124/19,773 ms | 24.63/27.71 ms | 35.03/37.38 ms |
| bimodal | 928.2 | 1,809.1 | -48.7% | 21,424/42,200 ms | 9,241/19,938 ms | 16.37/19.13 ms | 22.24/45.94 ms |

Current는 balanced/decode-heavy/long-prefill/bimodal에서 decode가 시작된 뒤 TPOT은 vLLM보다 8~30% 좋다.
그럼에도 throughput과 E2E가 뒤처지는 이유는 다음 요청이 D32 active set에 들어가기까지의 TTFT와 wave drain
시간이다. vLLM은 더 큰 continuous batch, block-level paged attention, 더 넓은 graph/batch shape coverage로 이
구간을 줄인다. short workload는 작은 shape graph와 launch overhead에서도 Current가 아직 뒤진다.

clean v0.10 public runtime에는 동일 continuous HTTP endpoint가 없으므로 fresh HTTP 숫자를 만들어내지 않았다.
기존 fixed-batch/replay oracle은 비교 문맥만 제공하며 fixed-output 공정 표에 섞지 않는다.

## long prefill과 fixed 128

현재 adaptive controller는 long-prefill 부하에서 decode pressure를 보고 chunk를 32로 줄였고 처리량은
322.6 token/s에 머물렀다. fixed 128로 고정하면 609.0 token/s로 88.8% 증가했다. ragged/wavefront batching만
켜고 chunk가 32로 남아 있으면 효과가 없었다.

이는 adaptive 정책을 제거해야 한다는 뜻이 아니다. 현재 cost model이 긴 prefill의 batch progress와 D32 queue
회복을 충분히 표현하지 못한다는 뜻이다. 다음 controller는 `chunk`만 고르는 대신 `(prefill rows, chunk tokens,
decode target)`의 joint cost를 workload class와 queue age에 따라 선택해야 한다.

## VLM multi-image placement 수정

실제 woman/dog 및 red-panda image URL 두 요청을 동시에 넣었을 때 두 문제가 드러났다.

1. 서로 다른 image request가 한 packed prefill batch에 합쳐졌다.
2. decode stage도 prefill용 vision payload를 다시 binding했다.

`PhaseWorkItem::exclusivePrefill`을 추가해 하나의 image request는 128-token chunking을 유지하되 다른 image
request와 같은 row batch에 들어가지 않게 했다. vision embedding/deepstack/M-RoPE는 prefill에서만 binding한다.
수정 후 2/2 requests, prompt 1,017 tokens, output 32 tokens가 완료되었고 70.4 token/s, TTFT median 315.5 ms,
TPOT median 9.23 ms, E2E median 453.9 ms를 기록했다. 이는 정확성 smoke이지 text throughput 비교값은 아니다.

## tied-engine production contract

baseline과 tied runtime이 같은 serialized engine을 쓴다는 사실을 runtime에서도 확인하도록
`tied_engine_contract`를 추가했다.

```text
materializer
  ├─ baseline/tied ONNX SHA-256 exact 확인
  ├─ engine SHA-256을 audit manifest에 기록
  ├─ engine size + 4개 위치의 1 MiB CRC32 기록
  └─ embedding raw allocation bytes 기록

runtime before alias publication
  ├─ local engine filename/path 확인
  ├─ engine size와 sampled CRC32 확인
  ├─ embedding allocation bytes 확인
  └─ 통과 후에만 LM-head non-owning alias 공개
```

전체 multi-GB SHA를 startup마다 다시 읽으면 약 12.8초의 추가 비용이 발생해 runtime 검증은 4 MiB 분산 표본으로
바꿨다. 전체 SHA는 materialization audit manifest에 남는다. 최종 B8 tied semantic smoke는 wall 4.11초에
완료했고, 이전 contract 없는 artifact는 명시적 오류로 거부되었다. controlled BS1 48-request exact identity와
594 MiB 절감, 3% 성능 gate 결과는 이전 문서의 결론을 유지한다.

## 테스트와 artifact

- phase scheduler focused tests: 67/67 pass
- Python materializer/HTTP harness tests: 5/5 pass
- B8 sampled-contract semantic inference: pass
- legacy tied artifact missing-contract negative gate: expected rejection
- VLM two-image real HTTP smoke: 2/2 pass
- B16/B32 fixed-output workload: requested token 수와 generated token 수 exact

주요 artifact:

- ONNX/B8 tied gate: `.local/cosmos-reason2-2b/tied-gate-20260820/`
- B16/B32 engines: `.local/cosmos-reason2-2b/batch-scale-20260820/`
- Current results: `.local/cosmos-reason2-2b/batch-scale-results-20260820/`
- vLLM results: `.local/vllm-cosmos-reason2-2b/batch-scale-fixed-comparison-20260820/`
- VLM trace: `notes/results/cosmos-v010-vlm-http-trace-20260820.json`

## 다음 우선순위

1. B48 또는 B64/KV1024 build feasibility와 실제 D48/D64 형성 여부를 먼저 검증한다.
2. D32 active set을 유지하면서 새 prefill admission을 허용하는 rolling decode cohort를 설계한다.
3. short workload용 P1/P2, D1/D2/D4 graph capture/replay와 graph cache hit를 production trace에서 계측한다.
4. long-prefill의 `(P batch, chunk, D target)` joint controller를 offline cost table에서 학습한다.
5. independent context의 중복 workspace를 profile별 상한과 TensorRT memory strategy로 줄이되 overlap 안전성을
   유지한다.
6. image request의 packed multi-row placement descriptor를 구현한 뒤 `exclusivePrefill` 제한을 완화한다.

