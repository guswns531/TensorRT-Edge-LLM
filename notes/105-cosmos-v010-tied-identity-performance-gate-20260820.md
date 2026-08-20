# Cosmos v0.10 tied LM-head 동일성·메모리·성능 gate

## 결론

동일 TensorRT engine bytes를 사용하는 baseline/tied runtime variant로 gate를 다시 측정했다.

- controlled BS1 greedy identity: **48/48 requests, 4,160/4,160 token IDs exact match**
- GPU ready/peak memory: **6,489/6,495MiB -> 5,895/5,901MiB**, 각각 **594MiB 절감**
- 288-request fixed-output throughput: **920.289 -> 922.057 token/s**, tied **+0.192%**
- TTFT median/p95: **-0.549% / -0.378%**
- TPOT median/p95: **+0.140% / +0.058%**
- E2E median/p95: **+0.020% / -0.115%**

따라서 exact controlled identity와 3% performance gate를 모두 통과한다.

## 공정성 계약

- source commit: `17e2da16849bf6a626bfae3f6217a063449dd936`
- TensorRT 11.0.0 / CUDA 13.3 / RTX 3080 10GB
- model: `nvidia/Cosmos-Reason2-2B`
- engine: B8, max input 1,024, max KV 2,048, 128 KV-pool pages, packed-prefill chunk 128
- runtime: 80 stable slots, in-flight 16, independent prefill/decode TensorRT contexts
- balanced trace SHA-256:
  `290d34061a173c13440e10247f55bd442a131e525a40c30a52d0b39a549d6538`

Baseline과 tied ONNX는 별도 exporter process에서 생성했지만 model graph/data가 byte-identical이다.

| artifact | shared SHA-256 |
| --- | --- |
| `model.onnx` | `85258a400c2d9b1a8737c9bf911624a25783e1c8d73129a6fd6cc7e755818832` |
| `model.onnx.data` | `9756b1c94bbeaf16b490775001410b4b1af531f11c619de42eba211bc30a0786` |

두 runtime directory의 `llm.engine`은 같은 inode이며 SHA-256은
`56b09d97f038178a0c7182c9305ccbaa07c81c53e0c0da0f8ff94772b313d2e7`이다.
따라서 TensorRT tactic이나 별도 engine build가 A/B를 오염시키지 않는다. 차이는 다음뿐이다.

```text
baseline: engine + embedding [V,H] + external LM head [H,V]
tied:     same engine + one embedding_transposed [H,V] alias
```

전체 622MB tensor 비교에서 tied embedding은 baseline embedding의 transpose 및 baseline LM-head sidecar와
byte-exact하게 일치했다.

## Controlled greedy identity

Online 동시 trace는 같은 engine을 반복 실행해도 CUDA event 완료 시점에 따라 batch evolution과 EOS 시점이 달라질 수
있다. 따라서 cross-engine exact identity는 batching 변수를 제거한 BS1 sequential gate로 분리했다.

- balanced trace의 앞 48 requests
- output length 32/48/64/96/128 조합
- EOS 무시, 요청된 output length까지 고정
- baseline과 tied를 각각 새 process에서 실행

두 variant 모두 4,160 token IDs를 생성했고 48개 요청이 모두 일치했다. token trace SHA-256은 양쪽 모두 다음과 같다.

```text
b14d1c789b567b4dc231444871c6c7a1b8fa3d05e879c4cc8a97a4afdc7d9480
```

## Fixed-output real-request 성능

성능은 288 requests, prompt 25,872 tokens, requested/generated output 24,960 tokens로 고정했다. variant마다 새
process lifecycle 3회이고 아래 값은 세 run의 중앙값이다.

| metric | baseline | tied | tied 변화 |
| --- | ---: | ---: | ---: |
| generated token/s | 920.289 | 922.057 | +0.192% |
| request/s | 10.619 | 10.639 | +0.192% |
| TTFT median | 12,514.12ms | 12,445.36ms | -0.549% |
| TTFT p95 | 24,217.36ms | 24,125.71ms | -0.378% |
| TPOT median | 16.5797ms | 16.6029ms | +0.140% |
| TPOT p95 | 17.1911ms | 17.2010ms | +0.058% |
| E2E median | 13,897.62ms | 13,900.38ms | +0.020% |
| E2E p95 | 25,661.36ms | 25,631.96ms | -0.115% |
| GPU ready | 6,489MiB | 5,895MiB | -594MiB |
| GPU peak | 6,495MiB | 5,901MiB | -594MiB |

모든 latency/throughput 변화가 1%보다 작고 3% gate 안이다.

## EOS 활성 결과를 primary로 쓰지 않는 이유

첫 측정에서 baseline 동일 engine의 세 lifecycle 생성량이 21,938/22,491/23,040으로 달랐다. 원인은 tied가 아니라
online batching 완료 순서가 EOS와 slot release를 바꾸기 때문이다. 고정-output 재측정에서는 모든 run이 정확히
24,960 tokens를 수행한다.

동시 trace의 token hash 역시 같은 engine 안에서 달라질 수 있으므로 exact identity 증거로 사용하지 않는다. 성능은
fixed work, token identity는 controlled BS1이라는 서로 다른 gate를 사용한다.

## TensorRT build 노이즈에서 얻은 교훈

처음 baseline/tied를 동일 옵션이지만 별도 TensorRT build로 만들었을 때 engine weight가 약 3.05GB/4.23GB로 크게
달라졌다. ONNX graph SHA는 같았으므로 이는 별도 tactic selection lifecycle의 교란이었다. 최종 gate는
`materialize_tied_engine_variant.py`가 ONNX SHA 일치를 확인한 뒤 baseline engine inode를 tied runtime과 공유한다.
이는 weight alias의 효과만 측정하는 가장 엄격한 A/B다.

## Artifact

- paired ONNX/engine: `.local/cosmos-reason2-2b/tied-gate-20260820/`
- fixed-output result: `.local/cosmos-reason2-2b/tied-gate-results-20260820/{baseline-fixed,tied-fixed}`
- identity result: `.local/cosmos-reason2-2b/tied-gate-results-20260820/{identity-baseline,identity-tied}`
