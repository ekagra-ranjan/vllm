# Async Scheduling in vLLM — A Deep Dive

## 1. The Problem: GPU Idle Time

In a synchronous engine loop, the GPU sits idle while the CPU runs Python scheduling, input preparation, and output processing:

```
Sync scheduling:

CPU:  [schedule(N)] [prepare_inputs(N)] [                    ] [process_output(N)] [schedule(N+1)] ...
GPU:                                     [===== execute(N) ==]                                     [===== execute(N+1) ==]
                                                               ↑ GPU idle here ↑
```

As batch sizes grow, CPU overhead grows proportionally, widening the idle gap.

## 2. The Solution: Overlap CPU and GPU Work

Async scheduling lets the CPU prepare step N+1 **while the GPU is still executing step N**:

```
Async scheduling:

CPU:  [schedule+prepare(N)] [process_output(N-1) + schedule+prepare(N+1)] [process_output(N) + schedule+prepare(N+2)] ...
GPU:                         [========= execute(N) ======================] [========= execute(N+1) ====================]
```

No GPU idle gaps — the CPU is always one step ahead.

## 3. How It Works in the Code

### 3.1 Engine Core: Two Execution Paths

The engine core (`vllm/v1/engine/core.py`) chooses between two step functions based on whether a batch queue exists:

```python
self.batch_queue_size = self.model_executor.max_concurrent_batches
self.batch_queue = None
if self.batch_queue_size > 1:
    self.batch_queue = deque(maxlen=self.batch_queue_size)

self.step_fn = (
    self.step if self.batch_queue is None else self.step_with_batch_queue
)
```

- **`step()`** — used when async scheduling is **disabled** (`max_concurrent_batches == 1`). Strictly sequential: schedule, execute, wait, process.
- **`step_with_batch_queue()`** — used when async scheduling is **enabled** (`max_concurrent_batches >= 2`). Allows multiple batches to be in flight.

When async scheduling is on, the uniproc executor reports `max_concurrent_batches = 2`:

```python
# vllm/v1/executor/uniproc_executor.py
@cached_property
def max_concurrent_batches(self) -> int:
    return 2 if self.scheduler_config.async_scheduling else 1
```

### 3.2 The Batch Queue: Pipelining Without Blocking

The key to async scheduling is the **early return** in `step_with_batch_queue()`:

```python
def step_with_batch_queue(self):
    # 1. Schedule a new batch and call execute_model
    exec_future = self.model_executor.execute_model(scheduler_output, non_block=True)
    batch_queue.appendleft((future, scheduler_output, exec_future))

    # 2. If queue isn't full and oldest result isn't ready, return early
    if (
        model_executed
        and len(batch_queue) < self.batch_queue_size
        and not batch_queue[-1][0].done()
    ):
        return None, True  # <-- RETURN WITHOUT CALLING future.result()

    # 3. Otherwise, block on the oldest result
    future, scheduler_output, exec_model_fut = batch_queue.pop()
    model_output = future.result()
    ...
```

When the early return fires, the engine calls `step_with_batch_queue()` again, which calls `execute_model()` for the **next** batch — all before the **previous** batch's `future.result()` has been called. This is how CPU and GPU work overlap.

### 3.3 Scheduler: Working with Stale Information

Because the scheduler runs before the previous step's output is known, it cannot know the actual tokens that were generated. The `AsyncScheduler` handles this with **output placeholders**:

```python
# vllm/v1/core/sched/async_scheduler.py
def _update_after_schedule(self, scheduler_output):
    for req_id in scheduler_output.num_scheduled_tokens:
        request = self.requests[req_id]
        if request.is_prefill_chunk:
            continue
        # Assume 1 new token + spec tokens will be generated
        request.num_output_placeholders += 1 + cur_num_spec_tokens
        # Use placeholder token ids (actual ids unknown yet)
        request.spec_token_ids = self._spec_token_placeholders  # [-1, -1, ...]
```

Each request tracks `num_output_placeholders` — tokens the scheduler has "assumed" will be produced but aren't confirmed yet. When actual output arrives, placeholders are reconciled:

```python
def _update_request_with_output(self, request, new_token_ids):
    new_token_ids, stopped = super()._update_request_with_output(request, new_token_ids)
    request.num_output_placeholders -= len(new_token_ids)
    assert request.num_output_placeholders >= 0
    ...
```

### 3.4 Model Runner: The AsyncGPUModelRunnerOutput

When async scheduling is enabled, the model runner does **not** synchronize the GPU before returning. Instead it returns an `AsyncGPUModelRunnerOutput` that wraps a deferred GPU-to-CPU copy:

```python
class AsyncGPUModelRunnerOutput(AsyncModelRunnerOutput):
    def __init__(self, ...):
        # Enqueue D2H copy on a SEPARATE stream (not the default stream)
        with torch.cuda.stream(async_output_copy_stream):
            async_output_copy_stream.wait_stream(default_stream)  # wait for forward pass
            self.sampled_token_ids_cpu = self._sampled_token_ids.to("cpu", non_blocking=True)
            self.async_copy_ready_event.record()

    def get_output(self):
        # Only wait for the D2H copy, not the entire GPU pipeline
        self.async_copy_ready_event.synchronize()
        ...
```

`get_output()` is what `future.result()` ultimately calls. It waits for the **output copy** event, which transitively ensures the default stream's forward pass has completed (because the copy stream did `wait_stream(default_stream)` first).

## 4. The Pinned Buffer Race Condition

### 4.1 The Problem

The model runner reuses **pinned CPU buffers** (`positions`, `input_ids`, `query_start_loc`, `seq_lens`, etc.) across steps. During input preparation, it:

1. **Writes** new values into these pinned buffers on the CPU.
2. **Enqueues** non-blocking H2D (host-to-device) copies onto the CUDA default stream.

Since the copies are non-blocking, the GPU reads from the pinned buffers **asynchronously** via DMA. If the CPU starts writing the next step's data into the same buffers before the GPU has finished reading, the data gets corrupted:

```
Step N:   CPU writes to pinned buf → enqueues non_blocking H2D copy → CPU returns
          GPU starts DMA read from pinned buf (async, on default stream)
Step N+1: CPU writes to SAME pinned buf  ← RACE! GPU may still be DMA-reading
```

### 4.2 Why `future.result()` Is Not Always Sufficient

In the simple `step()` path, `future.result()` blocks until `get_output()` completes, which transitively ensures all default stream work (including the H2D copies) has finished. So the race cannot happen — the GPU is done reading the pinned buffers before the next `execute_model` is called.

**But async scheduling does not use the `step()` path.** It uses `step_with_batch_queue()`, which can **return early** without calling `future.result()`, then immediately call `execute_model()` again. In this path, there is no guarantee that the GPU has finished reading the previous step's pinned buffers.

### 4.3 The Solution: `synchronize_input_prep()`

The model runner uses a CUDA event to protect the pinned buffers:

```python
def __init__(self, ...):
    if self.use_async_scheduling:
        self.prepare_inputs_event = torch.Event()

@contextmanager
def synchronize_input_prep(self):
    if self.prepare_inputs_event is None:
        yield
        return

    # ENTRY: Wait until GPU has finished reading the prior step's pinned buffers
    self.prepare_inputs_event.synchronize()
    try:
        yield   # <-- CPU writes to pinned buffers + enqueues H2D copies here
    finally:
        # EXIT: Record event on default stream after H2D copies are enqueued
        self.prepare_inputs_event.record()
```

The two halves work together:

- **`record()`** (exit) — Places a marker on the CUDA default stream. The event will be "signaled" when the GPU reaches this point, meaning all prior work on the stream (including the H2D DMA reads from pinned buffers) has completed.

- **`synchronize()`** (entry, next step) — Blocks the CPU until the event is signaled. Only then does the CPU proceed to overwrite the pinned buffers with new data.

```
Timeline with event protection:

Default stream: [H2D(N)] [event_record] [forward(N)] [sampling(N)] [H2D(N+1)] [event_record] ...
CPU:                                     |← sync() →|← write bufs + enqueue H2D(N+1) →|
                                          ↑ CPU waits here until H2D(N) DMA is done
```

This is more precisely scoped than waiting for the full forward pass. The CPU only waits for the H2D copies to finish, not for the forward pass. This maximizes overlap: the CPU can prepare the next step's inputs while the GPU is still running the current step's forward pass.

### 4.4 First Step: No Prior Record

A freshly constructed `torch.Event()` has never been recorded. Calling `.synchronize()` on an unrecorded event is a **no-op** — it returns immediately. This is correct because on the very first step, there is no prior step whose GPU work could still be reading from the pinned buffers. The `finally` block then calls `record()`, establishing the synchronization point for step 2.

## 5. Constraints on the Scheduler and Model Runner

### 5.1 Scheduler Constraints

| Constraint | Reason |
|---|---|
| Must use output placeholders instead of real tokens | Real tokens from step N aren't available when scheduling step N+1 |
| Must track `num_output_placeholders` per request | To account for "in-flight" tokens in budgets and termination checks |
| Must handle placeholder reconciliation when actual output arrives | `_update_request_with_output` decrements placeholders |
| Draft token ids use placeholders (`[-1, -1, ...]`) | Actual draft tokens are determined in the worker, not the scheduler |

### 5.2 Model Runner Constraints

| Constraint | Reason |
|---|---|
| No `torch.accelerator.synchronize()` in the execution path | Would stall the GPU pipeline and eliminate overlap |
| No implicit sync (e.g., unpinned `.to("cuda")`) | Unpinned transfers trigger a hidden synchronization |
| Must use pinned memory + `non_blocking=True` for H2D copies | Required for true async DMA |
| Must use `synchronize_input_prep()` to protect pinned buffers | Prevents race between CPU writes and GPU DMA reads |
| Must copy mutable state before returning output | `req_ids`, `req_id_to_index` are copied because the model runner may mutate them for the next step while the engine is still reading the current step's output |
| Must return `AsyncGPUModelRunnerOutput` (not a resolved output) | Lets `execute_model` return to the caller without waiting for GPU completion |
| Draft tokens handled internally (not via engine round-trip) | The scheduler has already moved on; it can't accept draft tokens after the fact |

### 5.3 Not All Configurations Support Async Scheduling

Async scheduling is disabled when:
- Speculative decoding methods other than Eagle or ngram-GPU are used.
- `disable_padded_drafter_batch=True` is set.
- The distributed executor backend doesn't support it (only `mp`, `uni`, and `external_launcher` do).

## 6. Summary: Sync vs Async Scheduling

| Aspect | Sync Scheduling | Async Scheduling |
|---|---|---|
| Step function | `step()` | `step_with_batch_queue()` |
| CPU/GPU overlap | None — GPU waits for CPU | CPU prepares N+1 while GPU runs N |
| Scheduler class | `Scheduler` | `AsyncScheduler` |
| Token knowledge | Knows all prior tokens | Uses placeholders for in-flight tokens |
| Model runner return type | `ModelRunnerOutput` | `AsyncGPUModelRunnerOutput` |
| Pinned buffer protection | Not needed (`future.result()` is sufficient) | `synchronize_input_prep()` with CUDA events |
| Spec decode draft tokens | Engine passes draft tokens to scheduler | Worker handles draft tokens internally |
| State management | Direct mutation is safe | Must copy mutable state to avoid races |
