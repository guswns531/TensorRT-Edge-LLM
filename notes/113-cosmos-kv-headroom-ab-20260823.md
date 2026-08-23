# Cosmos KV headroom 확대 A/B

## 결론

GPU headroom을 KV page pool과 stable staging slot에 재투자하는 실험을 진행했다. `KV288/stable88`은
decode-heavy TTFT median을 약 12.6% 줄였지만 TPOT median을 약 12.1% 악화시키고 E2E median도 약 3.3%
악화시켰다. 사전에 정한 TPOT 3% gate를 통과하지 못했으므로 채택하지 않았고 `KV320/stable96`은 빌드하지 않았다.

후보 엔진은 삭제해 디스크를 회수했고 결과 CSV/JSON만 보존했다. Production 선택은 계속
`KV256/stable80/P8/D64`다.

## 메모리 계약

Cosmos FP16 KV page bundle 하나는 약 14MiB다.

```text
28 layers * K/V 2 * 8 KV heads * 128 tokens * 128 head dim * FP16 2 bytes
= 14,680,064 bytes = 14MiB
```

| 구성 | pages | stable slots | D cohort | staging | ready GPU | headroom |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 선택값 | 256 | 80 | 64 | 16 | 8,088MiB | 1,778MiB |
| 후보 | 288 | 88 | 64 | 24 | 8,536MiB | 1,330MiB |

실측 증가량은 예상과 같은 448MiB였다. OOM이나 page exhaustion은 없었다.

## KV288 결과

현재 선택값은 최신 Priority 1~4 결과, KV288은 동일 HTTP/SSE client와 fixed-output trace의 3-run 중앙값이다.

| workload | 구성 | token/s | TTFT med/p95 | TPOT med/p95 | E2E med/p95 |
| --- | --- | ---: | ---: | ---: | ---: |
| short | KV256 | 2,410.4 | 96.6/181.1ms | **11.26**/27.53ms | **354.9/425.5ms** |
| short | KV288 | 2,381.4 | **78.5/179.4ms** | 11.94/27.72ms | 359.9/431.2ms |
| balanced | KV256 representative | 4,607.4 | **1,602/4,052ms** | **15.82/17.21ms** | **3,083/4,922ms** |
| balanced | KV288 | **4,614.2** | 1,671/**3,978ms** | 17.42/19.24ms | 3,310/4,932ms |
| decode-heavy | KV256 representative | 5,257.4 | 4,777/10,709ms | **12.30/12.95ms** | **7,836/13,357ms** |
| decode-heavy | KV288 | **5,283.9** | **4,174/10,043ms** | 13.80/14.30ms | 8,096/**13,275ms** |

KV288의 주효과는 throughput 증가가 아니라 더 많은 request가 prefill을 마치고 첫 token을 일찍 받는 것이다.
Balanced/decode-heavy 처리량은 0.1~0.5% 증가했지만 TPOT가 10% 이상 나빠졌다.

## 왜 TTFT가 줄고 TPOT가 늘었나

Stable slots가 80에서 88로 늘면 D64 cohort 밖의 prefetched request가 16개에서 24개로 늘어난다.

```text
D64 active cohort
+ 24 prefilled staging requests
```

Prefill 최종 logits에서 greedy token을 먼저 sampling하고 HTTP로 전송하므로 staging request의 TTFT 시점은 빨라진다.
하지만 그 request는 cohort row가 비기 전까지 다음 decode token을 만들 수 없다. 이 대기시간이 TTFT가 아니라 TPOT에
포함된다. 즉 실제 완료시간을 같은 비율로 줄인 것이 아니라 latency 구간을 TTFT에서 TPOT로 이동시킨 효과가 크다.

Decode-heavy에서 다음 현상이 이를 확인한다.

- TTFT median: 4,777 -> 4,174ms
- TPOT median: 12.30 -> 13.80ms
- E2E median: 7,836 -> 8,096ms
- throughput: 5,257 -> 5,284 token/s

## 추가 완화 실험

### Stable84/staging20

Stable slot을 84로 줄여도 decode-heavy TPOT median은 13.77ms로 거의 변하지 않았다. 따라서 단순히 staging 4개를
줄이는 것으로는 해결되지 않았다.

### P4 prefill 제한

KV288/stable88에서 runtime prefill batch를 P8에서 P4로 제한했다. 간섭 window는 작아졌지만 enqueue 수와 총
prefill 간섭이 늘었다.

- balanced throughput: 4,520 token/s
- TTFT median: 1,855ms
- TPOT median: 17.74ms

모두 P8보다 나빠 기각했다.

## KV320을 빌드하지 않은 이유

KV288 단계에서 이미 TPOT 3% gate를 크게 초과했다. KV320/stable96은 staging request를 32개로 늘려 같은 현상을
강화할 가능성이 높다. 추가 896MiB를 사용해도 active decode는 D64로 고정되므로 raw decode throughput 자체는
증가하지 않는다. 따라서 디스크와 build 시간을 더 사용하지 않고 중단했다.

## 다음 메모리 활용 방향

추가 메모리는 active admission보다 다음 용도가 더 타당하다.

1. 반복 system prompt용 prefix cache page budget
2. 첫 token을 외부로 내보내지 않는 hidden prefill staging 설계와 별도 E2E gate
3. TensorRT timing cache를 고정한 동일 tactic KV256/KV288 controlled build
4. 실제 long-context page pressure workload에서만 KV pool 확장

## Artifact

- 결과: `.local/cosmos-reason2-2b/kv-headroom-results-20260823/`
- KV288 후보 engine: gate 실패 후 삭제
- 유지 engine: `.local/cosmos-reason2-2b/asymmetric-20260820/engine-b80-p8-d64-kv2048-tied/`
