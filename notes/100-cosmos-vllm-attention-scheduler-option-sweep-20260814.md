SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

# Cosmos vLLM attention/scheduler option sweep

## 결론

Current는 scheduler와 KV cache만 바꾼 구현이 아니다. packed/ragged prefill, compact causal FMHA 입력,
indexed-paged KV write/read와 continuation prefix gather까지 attention 실행 경로를 변경했다. 따라서 기존 vLLM
자동 attention backend 하나만 비교 기준으로 두는 것은 충분하지 않았다.

vLLM 0.27.1의 실제 옵션을 모두 스크리닝한 결과, 이 RTX 3080/Cosmos text-only 조건의 최적 production 설정은
다음과 같다.

```text
--attention-backend TRITON_ATTN
--max-num-batched-tokens 8192
--watermark 0.05
--enable-chunked-prefill
--max-num-seqs 80
--kv-cache-memory 3758096384
```

- Triton attention과 5% KV watermark를 함께 사용하면 기존 vLLM 자동 설정 대비 generated token/s가 다섯
  workload에서 `+1.90%`, `+10.59%`, `+9.37%`, `+12.13%`, `+13.84%` 증가했다.
- tuned vLLM을 기준으로 다시 보면 Current가 모든 workload에서 빠르다는 이전 결론은 유지되지 않는다.
  Current는 short와 long-prefill에서 확실히 빠르고, balanced는 tuned vLLM이 빠르며, decode-heavy와 bimodal은
  처리량이 사실상 동률이고 latency 성격이 다르다.
- Current의 강점은 short/long-prefill과 decode TPOT이고, tuned vLLM의 강점은 balanced 처리량과
  decode/bimodal TTFT다.
- Current의 기존 vLLM 대비 큰 향상을 attention backend 선택 하나로 설명할 수는 없지만, auto FlashAttention만
  사용한 비교가 Current에 유리했던 것은 맞다.

## 조건

| 항목 | 값 |
| --- | --- |
| GPU/driver | RTX 3080 10GB / 610.43.02 |
| model | `nvidia/Cosmos-Reason2-2B`, local FP16 checkpoint |
| vLLM | 0.27.1, image digest `c2f3b1b9...a9f31da2` |
| model mode | `--language-model-only`, text-only |
| KV | FP16, raw 3.5GiB, 32,768-token capacity |
| prefix cache | disabled |
| serving | 동일 localhost OpenAI HTTP/SSE client, stream interval 1 |
| generation | greedy, seed 0, EOS enabled |
| graph | vLLM FULL+PIECEWISE CUDA graph, 0.23GiB |
| screening | balanced/decode-heavy/bimodal, fresh server, 각 1회 |
| final | short/balanced/decode-heavy/long-prefill/bimodal, 설정별 fresh server, 각 3회 |

서버 로그에서 decoder attention은 실제 `AttentionBackendEnum.TRITON_ATTN`으로 선택됐다. text-only 실행이므로
로그에 함께 보이는 ViT `FLASH_ATTN` backend는 이번 workload에 실행되지 않는다.

## 1차 옵션 스크리닝

아래 변화율은 기존 `8192-token/auto attention` vLLM에 대한 세 screening workload의 기하평균이다.
양수는 개선이다.

| 설정 | request/s | token/s | E2E p95 | 판정 |
| --- | ---: | ---: | ---: | --- |
| Triton + async + watermark 0.05 | +10.55% | +10.57% | +11.04% | async 제거 후 최종 확인 |
| Triton + watermark 0.05 | +10.54% | +10.48% | **+11.23%** | 최종 winner |
| Triton + async | +8.31% | +8.27% | +8.82% | async는 중립 |
| Triton attention | +8.20% | +8.16% | +8.65% | attention attribution 기준 |
| Triton + graph cap 80 | +7.35% | +7.35% | +7.69% | 기본 graph보다 낮음 |
| FlashInfer attention | +3.41% | +3.28% | +3.74% | 개선되나 VRAM +540MiB |
| watermark 0.05 | +1.86% | +1.92% | +1.80% | 긴 혼합 workload에서만 큼 |
| auto, 8192 tokens | 0.00% | 0.00% | 0.00% | 기존 기준 |
| performance throughput | -0.20% | -0.14% | -0.31% | 중립 |
| explicit FlashAttention | -0.32% | -0.39% | -0.56% | auto와 동일, run noise |
| block size 32 | -0.38% | -0.36% | -0.43% | 채택하지 않음 |
| async scheduling | -0.44% | -0.46% | -0.49% | 채택하지 않음 |
| graph capture cap 80 | -0.46% | -0.49% | -0.97% | 채택하지 않음 |
| full-ISL reserve off | -0.59% | -0.58% | -0.65% | 보수 admission 유지 |
| performance interactivity | -0.82% | -0.87% | -0.82% | 채택하지 않음 |
| token budget 4096 | -0.54% | -0.42% | -0.65% | 8192 유지 |
| token budget 2048 | -1.37% | -1.35% | -1.59% | 채택하지 않음 |
| token budget 1024 | -1.51% | -1.45% | -1.76% | Current와 강제 대칭화하지 않음 |
| eager, CUDA graph off | -37.71% | -37.63% | -34.76% | diagnostic only |

`--enable-dbo`도 시도했지만 이 모델에서는 시작하지 못했다. vLLM model runner V2가 DBO를 지원하지 않아 V1으로
내려간 뒤, microbatching이 요구하는 distributed all-to-all backend가 단일 GPU의
`allgather_reducescatter` 구성에는 지원되지 않는다는 validation error가 발생했다. 따라서 DBO 결과를 성능
순위에 넣지 않았다.

## 최종 처리량: Current 대 tuned vLLM

값은 세 complete run의 중앙값이다. EOS 위치가 backend마다 조금 다르므로 token/s와 request/s를 함께 본다.

| workload | Current tok/s | tuned vLLM tok/s | Current 변화 | Current req/s | tuned vLLM req/s | request winner |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| short | **2,463.9** | 2,075.6 | **+18.71%** | **113.72** | 95.79 | Current +18.71% |
| balanced | 4,413.4 | **4,531.7** | -2.61% | 53.60 | **55.04** | vLLM +2.68% |
| decode-heavy | **4,848.9** | 4,829.4 | +0.40% | 24.29 | **24.41** | vLLM +0.48% |
| long-prefill | **1,418.4** | 1,224.3 | **+15.86%** | **16.57** | 14.55 | Current +13.94% |
| bimodal-mixed | 1,997.2 | **2,011.5** | -0.71% | 14.57 | **14.91** | vLLM +2.35% |

기존 auto-vLLM과 비교했을 때 Current token/s 우위는
`+20.96/+7.70/+9.82/+29.91/+13.04%`였다. tuned vLLM을 기준으로는
`+18.71/-2.61/+0.40/+15.86/-0.71%`로 바뀐다. balanced와 bimodal의 결론이 역전되고 decode-heavy는
동률이 됐다.

clean upstream fixed-BS8 oracle의 기존 token/s `838.6/1167.4/1000.9/819.9/930.3`와 비교하면 tuned vLLM은
각각 `2.48x/3.88x/4.83x/1.49x/2.16x`다. Current의 upstream 대비 값은 기존
`2.94x/3.78x/4.84x/1.73x/2.15x` 그대로다.

## 최종 latency: Current 대 tuned vLLM

표의 값은 `median/p95`다. 작은 값이 좋다.

| workload | TTFT Current | TTFT tuned vLLM | TPOT Current | TPOT tuned vLLM | E2E Current | E2E tuned vLLM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| short | **148.7/204.7** | 197.6/260.0 | **8.96/14.62** | 9.33/22.26 | **333.5/405.1** | 423.4/485.0 |
| balanced | **1630.0**/3854.2 | 1670.6/**3687.0** | 16.29/19.12 | **15.47/16.86** | 3051.3/4812.6 | **3027.4/4642.9** |
| decode-heavy | 3689.6/8144.1 | **3168.9/7761.7** | **12.42/13.21** | 14.03/15.35 | **5751.4**/10680.0 | 5911.1/**10616.6** |
| long-prefill | **7727.4/15336.2** | 8865.2/17541.4 | **21.98/22.88** | 29.60/33.86 | **9562.6/16595.2** | 11448.2/19021.5 |
| bimodal-mixed | 7278.1/15634.9 | **6774.0/15108.1** | **15.45/18.12** | 18.07/36.12 | **9570.8**/17799.6 | 9570.9/**17649.8** |

- short는 Current가 모든 latency 지표에서 우세하다.
- balanced는 Current TTFT median만 41ms 낮고, tuned vLLM이 throughput과 나머지 tail/latency를 이긴다.
- decode-heavy는 tuned vLLM이 TTFT를 낮추지만 Current가 TPOT median/p95를 각각 약 11.5%/13.9% 낮춘다.
- long-prefill은 Current가 모든 latency와 처리량에서 우세하다.
- bimodal은 tuned vLLM이 TTFT와 request throughput을 이기지만 Current TPOT p95는 `18.12ms` 대
  `36.12ms`로 약 절반이다. E2E는 거의 동률이다.

## Watermark가 개선한 이유

Triton 단독과 Triton+watermark는 attention backend, raw KV bytes, CUDA graph와 token budget이 같다. 차이는
전체 KV block의 5%를 admission headroom으로 남기는 것뿐이다.

| workload | Triton preemption/run | Triton+watermark | Triton tok/s | combined tok/s | 변화 |
| --- | ---: | ---: | ---: | ---: | ---: |
| short | 0/0/0 | 0/0/0 | 2,081.5 | 2,075.6 | -0.28% |
| balanced | 0/0/0 | 0/0/0 | 4,551.8 | 4,531.7 | -0.44% |
| decode-heavy | 0/0/0 | 0/0/0 | 4,827.9 | 4,829.4 | +0.03% |
| long-prefill | 26/25/25 | **0/0/0** | 1,176.0 | **1,224.3** | **+4.10%** |
| bimodal-mixed | 59/71/70 | **0/0/0** | 1,861.1 | **2,011.5** | **+8.08%** |

watermark는 KV 메모리를 줄이거나 늘리지 않는다. 같은 32,768-token pool에서 약 5%를 즉시 admission하지 않고
남겨 preemption과 chunk 재계산을 방지한다. long/bimodal의 개선은 preemption count가 0으로 바뀐 것과 직접
일치한다. 이 결과는 Current의 bounded reservation/growth lease에서도 throughput만 보지 말고 재계산 없는
headroom을 명시적으로 유지해야 한다는 근거다.

## 메모리

| backend | ready | 관측 max | 비고 |
| --- | ---: | ---: | --- |
| vLLM Triton+watermark | 7,991MiB | **8,319MiB** | raw KV 3.5GiB, graph 0.23GiB |
| Current | workload별 9,059--9,185MiB | **9,185MiB** | independent TRT contexts/phase buffers 포함 |
| clean upstream | - | 7,348MiB | fixed BS8, continuous serving 없음 |

Current는 tuned vLLM max보다 866MiB 더 사용한다. raw KV budget 차이는 아니며 independent TensorRT context
workspace, phase별 I/O/plugin workspace와 graph cache가 주된 차이다. 반대로 vLLM FlashInfer backend는 ready
8,533MiB로 auto/Triton보다 약 540MiB 더 사용하면서 Triton보다 느렸으므로 최종 후보에서 제외했다.

## 해석과 기준 변경

1. 앞으로 `vLLM-auto`는 재현용 baseline이고, production competitor는
   `TRITON_ATTN + 8192 tokens + watermark 0.05`로 둔다.
2. Current packed-prefill의 기존 직접 A/B 개선은 balanced `+7.14%`, decode-heavy `+2.73%`였다. 이번 tuned
   vLLM attention 개선은 balanced `+10.59%`, decode-heavy `+9.37%`다. 서로 다른 시스템 내부 A/B이므로 이
   백분율을 더하거나 직접 kernel 우열로 해석하면 안 된다.
3. tuned vLLM이 balanced를 이겼으므로 다음 Current 최적화는 단순 scheduler parameter sweep보다 packed
   prefill FMHA의 layer별 prefix gather 제거와 decoder attention kernel 자체 비교가 우선이다.
4. Current가 short와 long-prefill에서 여전히 크게 앞서는 것은 independent phase overlap, bounded P8/D64,
   packed chunk 처리의 system-level 장점이 남아 있다는 뜻이다.
5. decode-heavy/bimodal에서는 throughput만 보면 동률이지만 Current의 TPOT가 훨씬 낮다. 다음 비교는 하나의
   champion 숫자보다 TTFT-SLO와 TPOT-SLO별 achievable throughput frontier로 해야 한다.

## Artifact와 재현

- runner: `scripts/cosmos_reason2/run_vllm_option_sweep.py`
- analyzer: `scripts/cosmos_reason2/analyze_vllm_option_sweep.py`
- screening raw result: `.local/vllm-cosmos-reason2-2b/option-sweep-20260814/screen/`
- final raw result: `.local/vllm-cosmos-reason2-2b/option-sweep-20260814/final/`
- prior Current/vLLM/upstream result: `.local/cosmos-reason2-2b/all-workload-recheck-20260814/`
