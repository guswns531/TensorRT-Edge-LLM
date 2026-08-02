# Gemma 4 VLM real-request 정확성 및 성능 비교

## Workload

[`gemma4_vlm_real_requests.json`](../tests/test_cases/gemma4_vlm_real_requests.json)은 단순 token-length synthetic
입력이 아니라 실제 chat message와 이미지 경로를 사용한다. 12개 요청은 다음 사용 형태를 섞는다.

- 장면 설명과 접근성 alt text
- ER 다이어그램 분석과 schema review
- red panda/giant panda 정방향 및 역방향 비교
- 사진과 야생동물, 다이어그램과 사진처럼 서로 다른 이미지 비교
- wildlife catalog, factual caption, retrieval tag 생성

기본 batch size는 2다. 총 6 batch, 18 images, 4,648 image tokens, 5,075 computed prefill tokens를 처리하며,
각 batch의 vision token 수는 visual engine의 1,120-token profile 안에 있다. Sampling은
`temperature=0`, `top_k=1`, 최대 generation length는 96이다.

## 측정 조건

- GPU: RTX 3080 10GB
- LLM: Gemma 4 E2B INT4-AWQ backbone, FP16 embedding/PLE/LM head/KV
- vision encoder: FP16 TensorRT engine
- legacy와 indexed를 교차 순서로 각각 3 process 반복
- process마다 첫 batch warmup 2회, engine load와 warmup은 serving wall time에서 제외
- CUDA graph decode 경로 사용
- GPU stage는 `vision_encoder`, `llm_prefill`, `llm_generation` CUDA event profile의 합

[`summarize_vlm_real_requests.py`](../scripts/gemma4_e2b_indexed/summarize_vlm_real_requests.py)가 log의 benchmark
시작/종료 timestamp와 profile JSON을 결합한다. 전체 median 원자료 요약은
[`gemma4-e2b-vlm-real-request-comparison.csv`](gemma4-e2b-vlm-real-request-comparison.csv)에 있다. 전체 3회 기반
p95 CSV와 response/profile/log는 `/tmp/gemma4-e2b/perf/vlm-real`에 보존했다. 3개 sample의 p95는 경향 확인용이며
production tail latency 근거로 사용하면 안 된다.

전체 실행은 다음 명령으로 재현한다.

```bash
WORK_DIR=/tmp/gemma4-e2b REPEATS=3 WARMUP=2 \
    scripts/gemma4_e2b_indexed/run_vlm_real_request_bench.sh
```

## Legacy와 indexed BS2

| metric | legacy median | indexed median | indexed delta |
|---|---:|---:|---:|
| serving wall | 3,587 ms | 3,604 ms | +0.47% |
| request throughput | 3.345 req/s | 3.330 req/s | -0.47% |
| vision GPU | 273.53 ms | 273.02 ms | -0.19% |
| prefill GPU | 464.37 ms | 464.99 ms | +0.13% |
| generation GPU | 2,782.65 ms | 2,796.86 ms | +0.51% |
| measured GPU sum | 3,521.35 ms | 3,538.76 ms | +0.49% |
| prefill throughput | 10,928.8 token/s | 10,914.1 token/s | -0.13% |
| generation throughput | 283.18 token/s | 284.60 token/s | +0.50% |
| peak VRAM | 8,888 MiB | 8,894 MiB | +6 MiB |

Indexed의 median overhead는 wall/GPU time 모두 약 0.5%이고 기존 3% gate 안이다. Vision과 prefill token 수는
항상 같지만 긴 자연어 응답은 process마다 785~818 token으로 달라졌다. 따라서 generation total time만 비교하지
않고 token-normalized throughput을 함께 봐야 한다. Indexed generation은 이 기준에서 0.50% 높았고 차이는 측정
노이즈 수준이다.

## Indexed BS1과 BS2 batching

| metric | BS1 median | BS2 median | BS2 delta |
|---|---:|---:|---:|
| serving wall | 5,610 ms | 3,604 ms | -35.76% |
| request throughput | 2.139 req/s | 3.330 req/s | +55.66% |
| prefill throughput | 9,956.3 token/s | 10,914.1 token/s | +9.62% |
| generation throughput | 165.88 token/s | 284.60 token/s | +71.58% |
| measured GPU sum | 5,541.89 ms | 3,538.76 ms | -36.15% |
| peak VRAM | 8,894 MiB | 8,894 MiB | 0 MiB |

이 workload에서는 indexed lookup 자체보다 실제 batch 형성이 훨씬 큰 효과를 냈다. 특히 두 request의 decode를
묶으면서 generation GPU time/token이 6.03ms에서 3.51ms로 줄었다. 이번 `llm_inference` 측정은 입력 파일을 미리
BS2로 묶은 offline batching이며, encoder/prefill/decode queue가 arrival window에서 동적으로 같은 batch를 만든
결과는 아니다. Online queue scheduler의 real JSON admission은 별도 harness가 필요하다.

## Semantic 결과

Strict reference 기준으로 12개 중 10개가 통과했다. 실패 두 건은 legacy/indexed의 모든 3회 반복에서 같았다.

1. red panda와 giant panda를 표로 비교하라는 요청은 첫 red panda만 설명하고 두 번째 이미지가 없다고 답했다.
   같은 prompt의 BS1 실행에서도 동일해 cross-row placement 문제는 아니다.
2. 여성/개 사진과 red panda를 비교하는 요청은 두 번째 동물을 `wild bear cub`로 잘못 불렀다. 두 이미지의 서로
   다른 장면과 subject라는 관계 자체는 인식했다.

반면 역순 panda 비교, diagram/photo 분류, 두 종 동일성 질문, 두 이미지 retrieval tag는 이미지 순서를 지켜
정상 응답했다. 이는 multi-image embedding 연결은 동작하지만 E2B 모델이 prompt 표현과 세밀한 동물 분류에
취약하다는 의미다.

Greedy 설정이어도 긴 응답 문자열은 process마다 일부 표현이 달랐다. Legacy와 indexed의 차이는
`characteristic`/`typical` 같은 동의 표현과 설명 길이였고 semantic 판정은 같았다. 따라서 실제 request 검증에서는
긴 출력의 byte equality보다 task reference와 순서 보존을 사용해야 한다.
