# Pending admission과 backpressure

## 왜 KV slot 앞에 별도 queue가 필요한가

indexed-linear cache의 physical slot 수는 고정이다. 모든 slot이 decode 요청에 lease된 상태에서 새 요청을
바로 `PhaseRequestLifecycle::submit()`에 넣으면 allocator exhaustion 예외가 발생한다. 이 요청은 아직 GPU
상태를 가지지 않으므로 KV slot을 미리 빼앗거나 cache를 이동할 이유가 없다.

`PhaseContextServingFacade::submitOrQueue()`는 admission을 두 상태로 나눈다.

```text
new request
    |
    +-- free slot 있음 --> ADMITTED(slot N) --> prefill queue
    |
    +-- free slot 없음 --> PENDING(no slot) --> bounded FIFO
                              |
                       terminal/cancel로 slot 반환
                              |
                         ADMITTED(slot N)
```

pending 요청은 source `DecodingInferenceContext`의 row를 빌려 등록하지만 GPU KV allocation, global slot
length, phase queue entry를 만들지 않는다. 따라서 pending 수가 늘어도 KV cache GPU 사용량은 변하지 않는다.

## API와 backpressure 규칙

- 기존 `submit()`은 즉시 admission 전용이며 호환성을 유지한다.
- `submitOrQueue()`는 `PhaseAdmissionResult`로 `kAdmitted` 또는 `kPending`을 반환한다.
- `maxPendingAdmissions`는 host FIFO의 상한이다. 기본값 0은 기존 fail-fast 동작을 유지한다.
- slot과 pending FIFO가 모두 차면 새 요청을 명시적으로 거부한다.
- `onAdmission`은 최초 pending 상태와 나중의 실제 slot lease를 모두 알린다.
- `request()`는 pending 요청을 `PhaseRequestStatus::kPending`, `kvSlotId=-1`로 노출한다.

## 반환과 재입장 순서

decode terminal 또는 queued request cancel은 먼저 stable slot lease를 반환한다. terminal callback이 실행되는
동안에는 worker가 아직 in-flight batch를 정리 중일 수 있으므로, facade는 drain 필요 상태만 기록한다. CUDA
event completion과 scheduler completion이 끝난 뒤 `wait()`, `poll()`, 다음 `dispatchNext()` 경계에서 FIFO를
drain한다. 이 순서로 같은 scheduler를 callback 안에서 재진입하지 않는다.

pending 자체를 취소하면 KV slot release나 CUDA 작업은 없다. registration과 FIFO entry만 제거하고
`kCancelled`, `kvSlotId=-1` terminal snapshot을 보낸다.

## 메모리 효율과 제한

- GPU KV cache는 여전히 `[maxSlots, 2, Hkv, capacity, D]` 고정 allocation이다.
- pending queue는 작은 host metadata만 사용하므로 GPU fragmentation을 만들지 않는다.
- admission 후에는 free-list가 반환한 stable slot을 그대로 사용하며 KV tensor compaction은 없다.
- source context는 borrowed object이므로 pending 또는 active 상태가 끝날 때까지 호출자가 수명을 보장해야 한다.
- V1 facade는 단일 host thread에서 호출한다고 가정한다. multi-producer admission에는 외부 synchronization이
  필요하다.

GPU 단위 테스트는 slot 2개를 채운 뒤 세 번째 요청의 pending 전환, queue overflow, terminal 후 FIFO
