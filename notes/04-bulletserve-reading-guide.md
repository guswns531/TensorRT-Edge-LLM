# BulletServe 구현 읽기 가이드

참조 대상:

- 저장소: [zejia-lin/BulletServe](https://github.com/zejia-lin/BulletServe)
- 확인한 commit: [`445afae2`](https://github.com/zejia-lin/BulletServe/commit/445afae2abc15d578d3107d37a9b57ebc49e5e46)
- 논문: [Bullet: Boosting GPU Utilization for LLM Serving via Dynamic Spatial-Temporal Orchestration](https://arxiv.org/abs/2504.19516)

BulletServe는 SGLang 기반 연구 prototype이다. 그대로 복사하기보다 mechanism과 policy를 나누어 읽는다.

## 1. 전체 구조에서 가져올 아이디어

- prefill은 compute-intensive, decode는 memory-bound라는 상보성을 이용
- prefill/decode를 별도 scheduling domain으로 유지
- phase별 전용 CUDA stream
- 실행 전 현재 batch 상태로 resource budget 결정
- stream에 SM/TPC mask 적용
- layer group마다 prefill을 끊어 재결정 기회 확보
- CUDA event와 request timing으로 predictor를 보정
- scheduler policy와 resource control을 분리

## 2. 실제 구현 위치

### stream 생성과 mask 적용

`python/sglang/srt/managers/tp_worker.py`

- decode stream은 높은 priority, prefill은 normal priority로 생성
- forward 직전에 shared manager에서 TPC 수를 결정
- TPC 수가 바뀌면 `SMController.set_stream_mask()` 호출
- decode는 mask 범위를 반대쪽 끝에서 잡아 prefill과 분리

`python/sglang/srt/bullet/sm_controller.py`

- Python `ctypes`로 `libsmctrl.so` 호출
- torch stream의 raw `cuda_stream` handle 전달
- 128-bit mask 지원
- 실제 resource count는 `libsmctrl_get_tpc_info_cuda()`로 query

### 상태 공유와 정책

`python/sglang/srt/bullet/shared_mng.py`

- prefill token 수
- decode batch 수
- decode total context
- 남은 layer 수
- prefill/decode TPC budget
- queue request 수

상태는 shared NumPy memory에 놓인다. fixed mode와 dynamic predictor library mode가 분리돼 있다.

### layer-wise prefill

`python/sglang/srt/bullet/model_monkey_patch.py`

- model의 `start_layer`, `end_layer`를 매 step 변경
- `layers_per_step` 단위로 partial forward
- 중간 hidden state를 다음 step으로 전달

`python/sglang/srt/managers/tp_worker_overlap_thread.py`

- scheduler stream과 forward stream 사이 event
- forward를 별도 host thread에서 launch
- prefill이면 layerwise generator 반복
- 각 layer group 뒤 현재 stream synchronize 후 다음 group 진행
- 결과 D2H 완료 event를 output queue에 전달

이 구조는 eager PyTorch model이라 가능하다. TensorRT의 monolithic engine에는 직접 적용할 수 없다.

### timing

`python/sglang/srt/bullet/observability.py`

- CUDA event ring으로 GPU forward 시간 기록
- request별 TTFT/TPOT 상태
- predictor input/output과 실측값 기록

현재 코드에는 `torch.cuda.synchronize()`가 포함된 timing/debug 경로도 있으므로 “모든 계측이 완전히
non-intrusive”하다고 가정하면 안 된다. 핵심 아이디어만 가져와 Edge-LLM에서는 event query 기반으로 다시
설계한다.

### KV pool 공유

`python/sglang/srt/bullet/memory_pool_rpc_v2.py`

- GPU KV tensor를 공유 가능한 descriptor로 전달
- request-to-token pool과 token-to-KV pool을 여러 worker가 재구성
- shared ring과 host lock으로 allocation/free 조정

BulletServe는 현재 Edge-LLM의 fixed linear batch-slot cache와 달리 token-to-KV pool allocator를 이미 가진
SGLang 구조에서 출발한다. 이 차이 때문에 KV 공유 코드를 직접 이식할 수 없다.

## 3. BulletServe와 이번 목표의 차이

| 항목 | BulletServe | 이번 목표 |
|---|---|---|
| runtime | SGLang/PyTorch | C++/TensorRT |
| worker | prefill/decode 별도 process 중심 | 같은 process 우선 |
| GPU sharing | MPS + libsmctrl | stream/context 분리 후 backend 선택 |
| KV cache | token pool/shared tensors | layer별 fixed linear batch slots |
| scheduling granularity | layer group | 첫 단계 phase, 장기적으로 engine segment |
| phase | prefill/decode | encoder/prefill/decode |
| execution split | model layer loop 변경 가능 | monolithic TRT engine은 불가능 |

## 4. 주의할 점

### libsmctrl은 공식 CUDA API가 아니다

stream 내부 구조의 version-specific offset을 사용한다. BulletServe root README와 `csrc/README.md`의 지원 CUDA
버전 설명도 서로 다르다. Jetson Orin, Thor, DRIVE에서 사용하는 CUDA/driver 조합마다 반드시 validator test를
먼저 통과해야 한다.

### mask 단위

코드의 변수명은 `num_tpcs`다. “SM N개”라고 문서화하기 전에 GPU에서 TPC-to-SM 관계와 mask 범위를 query하고
검증해야 한다.

### TensorRT auxiliary stream

Edge-LLM의 `EngineExecutor`는 TensorRT engine이 요구하는 auxiliary stream을 자동 생성한다. main stream
mask만 설정하면 auxiliary kernel이 partition 밖에서 실행할 수 있다. backend가 aux stream handle을 모두
제어하거나, PoC에서는 aux stream을 끄고 성능/정확성을 비교해야 한다.

### 세 phase를 항상 동시에 돌리지 않는다

encoder와 prefill도 compute-heavy일 수 있다. decode와 무엇을 같이 실행할지는 workload와 SLO에 따라 결정해야
한다. BulletServe의 두 phase 상보성이 encoder까지 자동으로 확장된다고 가정하지 않는다.

## 5. 이 저장소에 적용할 때 가져올 것과 미룰 것

### 먼저 가져올 것

- 전용 stream과 priority
- phase state snapshot
- fixed policy와 dynamic policy interface 분리
- event ring
- resource backend abstraction
- decode SLO guard

### 뒤로 미룰 것

- analytical roofline model 전체
- layer-wise prefill
- multi-process MPS deployment
- shared paged KV allocator
- 매 layer dynamic repartition
