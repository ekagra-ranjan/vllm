# Padded Drafter Batch in vLLM Speculative Decoding

## 1. Background: The Speculative Decoding Loop

Speculative decoding works in a **draft-then-verify** loop:

1. The **drafter** (e.g. EAGLE head) proposes `K` draft tokens per request.
2. The **target model** verifies all `K` tokens in one forward pass.
3. Some tokens are **accepted**, some are **rejected**. Only the longest accepted prefix is kept.
4. The drafter must now produce the *next* round of proposals using the verification results.

The critical question is: **how does the drafter learn what was accepted and prepare its next inputs?**

Each request may have a different number of accepted tokens. For example, with `K=3` draft tokens across 3 requests:

```
Request A: proposed [t1, t2, t3] → accepted [t1, t2, t3]     (all 3 accepted)
Request B: proposed [t1, t2, t3] → accepted [t1]              (2 rejected)
Request C: proposed [t1, t2, t3] → accepted [t1, t2]          (1 rejected)
```

After verification, the drafter's input batch needs to account for these varying acceptance lengths. There are two strategies for this, and the choice between them is what "padded drafter batch" is about.

## 2. The Two Strategies

### 2.1 Non-Padded (CPU path, `disable_padded_drafter_batch=True`)

**Strip rejected tokens, produce compact variable-length inputs.**

After the target model samples, the system:

1. **Waits for GPU → CPU sync** to get `sampled_token_ids` as a Python `list[list[int]]` — a ragged list where each sublist has a different length depending on how many tokens were accepted.
2. Runs `prepare_next_token_ids_cpu` — a **Python loop** that iterates through each request to find its next token.
3. Runs `prepare_inputs` — **CPU-side NumPy** operations that:
   - Compute `num_rejected_tokens` per request
   - Rebuild `query_start_loc` by subtracting rejected counts
   - Construct a compact `token_indices` array via `np.repeat` + offset math
   - Use fancy indexing (`hidden_states[token_indices]`) to gather only accepted tokens

The non-padded `prepare_inputs` works as follows (from the code comments):

```
Given:
  query_start_loc: [0, q1, q1+q2, q1+q2+q3]
  seq_lens:        [s1, s2, s3]
  num_rejected:    [n1, n2, n3]

Produces:
  query_start_loc: [0, q1-n1, q1+q2-n1-n2, q1+q2+q3-n1-n2-n3]
  seq_lens:        [s1-n1+1, s2-n2+1, s3-n3+1]
  token_indices:   [0..q1-n1-1,  q1..q1+q2-n2-1,  q1+q2..q1+q2+q3-n3-1]
```

The `token_indices` array is used for fancy indexing to extract only the accepted tokens from the hidden states and input IDs.

**Key property:** All this work happens on the CPU, *after* bookkeeping completes. The drafter cannot start until the CPU finishes computing which tokens were rejected.

### 2.2 Padded (GPU path, the default)

**Keep rejected tokens as padding; filter them out after the drafter runs.**

Instead of removing rejected tokens, the drafter receives **all tokens** — accepted and rejected alike. The rejected tokens sit in their original positions as inert padding. After the drafter produces its output, a `token_indices_to_sample` tensor tells the system which output positions correspond to real (non-rejected) tokens.

After the target model samples, the system:

1. Reads `sampled_token_ids` as a **GPU tensor** of shape `(num_reqs, num_spec_tokens+1)`, where rejected tokens have value `-1`.
2. Runs `eagle_prepare_next_token_padded_kernel` — a **Triton kernel on GPU** that, for each request in parallel:
   - Counts valid (non-`-1`) tokens → `valid_sampled_tokens_count`
   - Finds the last accepted token → `next_token_ids`
   - Falls back to a pre-uploaded "backup" token for discarded requests
3. Runs `eagle_prepare_inputs_padded_kernel` — a **Triton kernel on GPU** that, for each request:
   - Computes `num_rejected_tokens = num_draft_tokens + 1 - valid_count`
   - Computes `token_indices_to_sample = last_token_position - num_rejected_tokens`

The drafter then runs its forward pass on the full (padded) input. Only the positions indicated by `token_indices_to_sample` are used from the output.

**Key property:** No CPU synchronization is needed. The entire flow from sampling → drafter input prep → drafter forward pass stays on the GPU as a stream of CUDA/Triton kernel launches.

## 3. Why Padded Drafter Batch Exists

The padded approach exists to **eliminate CPU-GPU synchronization** in the speculative decoding loop. This has two major benefits:

### 3.1 Enabling Async Scheduling

vLLM's async scheduler overlaps CPU work (scheduling step N+1) with GPU work (executing step N). This requires the GPU execution path to be a continuous stream of kernel launches with no CPU sync points.

The non-padded path breaks this because it requires:
- GPU → CPU transfer of sampled token IDs (sync point)
- CPU-side Python/NumPy computation (serial bottleneck)
- CPU → GPU transfer of computed indices

The padded path keeps everything on the GPU, preserving the async overlap. This is enforced at the config level — `disable_padded_drafter_batch=True` is **incompatible** with async scheduling and will raise an error.

### 3.2 Reducing CPU Overhead

Even without async scheduling, the CPU path involves Python loops, NumPy operations, and `torch.from_numpy(...).to(device)` transfers. On fast GPUs, this CPU overhead is significant relative to the drafter's actual GPU compute time. The padded path replaces all of it with lightweight Triton kernels.

### 3.3 Timeline Comparison

```
NON-PADDED (CPU path):
  GPU: [target fwd] [sample] ---- wait ---- [drafter fwd]
  CPU:                        [bookkeep] [prepare_inputs]
                                  ↑ sync point

PADDED (GPU path):
  GPU: [target fwd] [sample] [prepare_padded] [drafter fwd]
  CPU:                       [bookkeep — overlapped, not blocking GPU]
                                  ↑ no sync
```

## 4. Hardware Applicability

The padded drafter batch is **not specific to any GPU generation**. It benefits all NVIDIA hardware (A100, H100, B200, etc.) because it is a software optimization for CPU-GPU overlap.

However, the benefit scales with GPU speed:

| GPU | Relative Benefit | Reason |
|-----|-----------------|--------|
| A100 | Moderate | GPU forward passes are slower, so CPU overhead is a smaller fraction |
| H100 | High | GPU is fast enough that CPU sync becomes noticeable |
| B200 | Very High | GPU is so fast that any CPU sync is a large fraction of iteration time |

The faster the GPU, the more time is wasted waiting for CPU-side processing in the non-padded path. On B200, the drafter model forward pass may take only a fraction of a millisecond, making even small CPU sync overheads relatively expensive.

## 5. Constraints

### 5.1 Async Scheduling Requires It

```python
# From vllm/config/vllm.py
if self.speculative_config.disable_padded_drafter_batch:
    raise ValueError(
        "Async scheduling is not compatible with "
        "disable_padded_drafter_batch=True."
    )
```

### 5.2 Draft Models and Parallel Drafting Require It

Features that use `needs_extra_input_slots` (draft models, parallel drafting) only support the padded path:

```python
# From vllm/v1/spec_decode/eagle.py
def _raise_if_padded_drafter_batch_disabled(self):
    if self.speculative_config.disable_padded_drafter_batch:
        raise NotImplementedError(
            "Speculative Decoding with draft models or parallel drafting only "
            "supports padded drafter batch."
        )
```

### 5.3 Extra GPU Compute (Tradeoff)

Rejected tokens are processed through the drafter model as padding. Their results are discarded afterward. This is wasted FLOPs, but the overhead is typically small because:
- The drafter model (e.g. EAGLE head) is small (single transformer layer)
- The number of rejected tokens per request is typically 0-2
- The time saved by avoiding CPU sync far exceeds the extra compute

### 5.4 Fixed-Shape Tensor Representation

The padded path requires `sampled_token_ids` as a **fixed-shape GPU tensor** `(num_reqs, num_spec_tokens+1)` with `-1` sentinel values for rejected entries. The non-padded path uses a ragged Python `list[list[int]]`. This shapes how downstream code branches:

```python
# From gpu_model_runner.py — the two branches
if spec_config.disable_padded_drafter_batch:
    assert isinstance(sampled_token_ids, list)
    next_token_ids = self.drafter.prepare_next_token_ids_cpu(...)
else:
    assert isinstance(sampled_token_ids, torch.Tensor)
    next_token_ids, valid_count = self.drafter.prepare_next_token_ids_padded(...)
```

## 6. Assumptions

### 6.1 All Drafter Input Prep Can Be Expressed as GPU Kernels

The two Triton kernels (`eagle_prepare_next_token_padded_kernel` and `eagle_prepare_inputs_padded_kernel`) must handle all the logic that the CPU path does:
- Counting valid/rejected tokens
- Finding the last accepted token ID per request
- Computing which output indices to sample from
- Handling edge cases (discarded requests, partial prefills)

### 6.2 Backup Token IDs Are Pre-Uploaded

For requests where the sampler didn't produce a valid next token (e.g. partial prefills, discarded requests), a "backup" token ID is needed. Since the GPU kernel can't call `request.get_token_id()`, these backup values are **pre-computed on CPU and uploaded to GPU** before the kernel runs:

```python
# Pre-compute backup tokens from request state
self.backup_next_token_ids.np[:num_reqs] = np.array([
    requests[gpu_input_batch.req_ids[i]].get_token_id(
        common_attn_metadata.seq_lens_cpu[i].item()
    )
    for i in range(num_reqs)
], dtype=np.int32)
self.backup_next_token_ids.copy_to_gpu(num_reqs)
```

### 6.3 Padding Overhead Is Negligible

The design assumes that the extra FLOPs from processing rejected-token padding through the drafter are negligible compared to the latency saved by avoiding CPU-GPU sync. This holds because:
- Drafter models are small (EAGLE head ≈ 1 transformer layer)
- Rejection rates are typically low (speculative decoding is most useful when acceptance is high)
- The CPU sync avoided can be on the order of hundreds of microseconds to milliseconds

### 6.4 Attention Backend Handles Padding Correctly

Rejected tokens that become padding must not corrupt the KV cache. The slot mapping for these tokens is set to `PADDING_SLOT_ID`, directing writes to a safe throwaway location.

## 7. Key Code Locations

| File | Function/Class | Role |
|------|---------------|------|
| `vllm/config/speculative.py` | `disable_padded_drafter_batch` | Config flag (default `False` = padded enabled) |
| `vllm/config/vllm.py` | `VllmConfig` init | Enforces mutual exclusion with async scheduling |
| `vllm/v1/worker/gpu_model_runner.py` | `sample_tokens` | Branches between padded and non-padded flows |
| `vllm/v1/spec_decode/eagle.py` | `prepare_next_token_ids_padded` | GPU path: find next token + count valid tokens |
| `vllm/v1/spec_decode/eagle.py` | `prepare_inputs_padded` | GPU path: compute sample indices for drafter output |
| `vllm/v1/spec_decode/eagle.py` | `prepare_next_token_ids_cpu` | CPU path: Python loop over requests |
| `vllm/v1/spec_decode/eagle.py` | `prepare_inputs` | CPU path: NumPy compact indexing |
| `vllm/v1/spec_decode/utils.py` | `eagle_prepare_next_token_padded_kernel` | Triton kernel: valid token counting + next token selection |
| `vllm/v1/spec_decode/utils.py` | `eagle_prepare_inputs_padded_kernel` | Triton kernel: sample index + rejected count computation |

## 8. Summary

The padded drafter batch is a performance optimization that trades a small amount of redundant GPU compute (processing rejected tokens as padding) for the elimination of CPU-GPU synchronization in the speculative decoding loop. It is the default behavior, required for async scheduling, and benefits all GPU hardware — with increasing returns on faster GPUs where CPU overhead is the dominant bottleneck.
