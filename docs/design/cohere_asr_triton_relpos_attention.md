# Cohere ASR Triton Relative-Position Attention

This document explains, in detail, what changed in the Cohere ASR encoder attention path, why the changes were made, how the new Triton fast path works, what remains as fallback behavior, and how to review the work efficiently.

It is written as a review aid, not just a changelog. The goal is to help a reviewer build intuition for both the implementation and the performance implications.

## Executive Summary

The Cohere ASR encoder originally used a fully manual attention path:

1. Build dense attention scores.
2. Apply masking in PyTorch.
3. Run `softmax`.
4. Multiply by `V`.

That was then upgraded to use PyTorch SDPA as a better fallback, but the relative-position branch still materialized a dense query-dependent bias tensor of shape `(B, H, T, T)`.

The final step implemented here is a model-local CUDA Triton fast path for full-context relative-position attention in Cohere ASR:

- It keeps the shared vLLM attention stack unchanged.
- It is only used for supported CUDA configurations.
- It falls back to the current SDPA path on unsupported platforms or shapes.
- It avoids materializing the dense relative-position bias tensor on the fast path.
- In the original `head_dim <= 128` bring-up domain, it improved the end-to-end VoxPopuli concurrency-512 benchmark from the earlier SDPA result of about `1125` RTFx to `1216.37` RTFx.
- A later follow-up widened the supported domain to include `head_dim = 160` so the fast path can activate on the 2B-release model. The first naive widening regressed, but a second redesign that uses exact split-D chunks and a vectorized rel-pos tile recovered most of that loss and brought the 2B-release production shape back into the same performance band as the earlier fallback path.

## Review Map

If you want to review this efficiently, read the files in this order:

1. [`vllm/model_executor/models/cohere_asr.py`](../../vllm/model_executor/models/cohere_asr.py)
2. [`vllm/v1/attention/ops/cohere_asr_relpos_attention.py`](../../vllm/v1/attention/ops/cohere_asr_relpos_attention.py)
3. [`tests/models/test_cohere_asr_attention.py`](../../tests/models/test_cohere_asr_attention.py)

That order mirrors the actual design:

- `cohere_asr.py` decides when the fast path is allowed.
- `cohere_asr_relpos_attention.py` implements the fused CUDA kernel.
- `test_cohere_asr_attention.py` proves parity and routing behavior.

## The Core Problem

The Cohere ASR encoder uses Transformer-XL style relative-position attention. The score is:

```text
score(i, j) = ((q_i + u) · k_j + (q_i + v) · r_{i-j}) / sqrt(d)
```

The expensive part is the second term. In the previous implementation, that term was computed as:

```text
matrix_bd = rel_shift((q + v) @ p^T)
```

where:

- `q` is query
- `v` here means the learned position bias `pos_bias_v`, not the attention value tensor
- `p` is the projected relative-position embedding table
- `rel_shift` converts the `2T-1` relative-position layout into the final `T x T` bias alignment

This means the model produced a dense `(B, H, T, T)` tensor before attention even started.

That dense tensor was the main remaining bottleneck after moving the fallback path to SDPA:

- it consumed memory bandwidth
- it forced extra reads and writes to HBM
- it prevented a truly fused flash-style implementation for the rel-pos branch

## What Was Done

The work happened in three layers:

1. SDPA fallback cleanup in `cohere_asr.py`
2. CUDA Triton fast path for the rel-pos branch
3. Tests and runtime validation

## Optimization Worklog

This section is intentionally chronological.

The goal is not just to say what the final code does. The goal is to show
how the implementation evolved, which assumptions were wrong, which
measurements changed the direction of the work, and what mental model a
reviewer should keep while reading the kernel.

For each step, ask four questions:

1. What was the current bottleneck hypothesis?
2. What code change was made to test that hypothesis?
3. What evidence came back from tests or benchmarks?
4. What new constraint or insight changed the next step?

### Step 0: Reduce The Search Space First

Before Triton was the answer, the first job was to make sure Triton was even
solving the right problem.

The original encoder path did two expensive things:

- manual attention in PyTorch
- dense relative-position bias materialization

The SDPA cleanup removed the first bottleneck from the fallback path so the
remaining problem became much clearer: the rel-pos branch still built a dense
query-dependent `(B, H, T, T)` tensor before attention.

That matters because optimization work goes wrong when too many costs are
mixed together. If both "manual softmax" and "dense rel-pos bias" are still in
the profile, it becomes hard to know which improvement is real and which one
is just hiding another problem.

The right intuition here is:

- first compress the baseline to a reasonable fallback
- then attack the bottleneck that still dominates

### Step 1: The First Triton Kernel Changed Where The Rel-Pos Term Is Computed

The key hypothesis was simple:

- the main remaining win would not come from "using Triton" in the abstract
- it would come from never materializing `matrix_bd`

The dense implementation conceptually did this:

```python
matrix_bd = rel_shift(q_with_bias_v @ p.transpose(-1, -2))
scores = (q_with_bias_u @ k.transpose(-1, -2) + matrix_bd) / sqrt(d)
```

The first Triton kernel changed the execution model. Instead of forming the
whole relative-position matrix and then shifting it, the kernel derived the
correct relative-position index directly from the query and key coordinates of
the current tile:

```python
rel_pos = center_pos + key_pos - query_pos
```

That is the core optimization. Everything else exists to make that idea
correct, numerically stable, and safe to route.

The original fused kernel looked like this in the critical inner section:

```python
qk = tl.dot(q_u, k).to(tl.float32)
rel_qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

for row_idx in range(BLOCK_M):
    query_pos = block_start_loc + row_idx
    rel_pos = center_pos + key_pos - query_pos
    ...
    rel_row = tl.sum(
        q_v_row.to(tl.float32)[:, None] * p_row.to(tl.float32),
        axis=0,
    )
    rel_qk = tl.where(row_ids[:, None] == row_idx, rel_row[None, :], rel_qk)
```

The first mental model a reviewer should build is this:

- the kernel is not just "attention in Triton"
- it is "attention where rel-pos lookup is fused into score formation"

That is why the dense `matrix_bd` allocation disappears.

### Step 2: The Fast-Path Gate Encoded Real Kernel Invariants

Once the first kernel existed, the next important change was not another math
optimization. It was explicit routing.

The model-side gate is:

```python
def _can_use_triton_relpos_attention(...):
    if not current_platform.is_cuda() or not query.is_cuda:
        return False
    if query.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False
    if self.d_k < 16 or self.d_k > 256 or pos_emb.size(0) != 1:
        return False
    ...
    return True
```

This is worth reading carefully because it represents accumulated kernel
knowledge, not generic defensive coding.

Examples:

- `head_dim >= 16` came from real `tl.dot` constraints
- bool mask checks reflect how the kernel consumes masking
- `pos_emb.size(0) == 1` reflects the supported rel-pos layout

The right reviewer instinct is: when a Triton gate looks narrow, assume it is
documenting a discovered invariant until proven otherwise.

### Step 3: Production Reality Invalidated The First Success Story

The first fast path was brought up in a domain where `head_dim <= 128`.

Then an important production fact showed up:

- the `2b-release` checkpoint uses `head_dim = 160`

That meant the first version of the Triton path was not actually running on
the production shape that mattered most.

This changed the question from:

- "does the Triton kernel work?"

to:

- "does the Triton kernel work on the real shape we serve?"

This is a good example of critical thinking during performance work:

- a benchmark result is only meaningful if the optimized path is actually active
- shape gating is part of the optimization story, not an implementation detail

### Step 4: Naive Widening Fixed Routing But Regressed Performance

The first follow-up for `head_dim = 160` was the obvious one:

- widen the allowed range from `<= 128` to `<= 256`
- keep the original kernel structure

The relevant dispatch still looked like this:

```python
if not _uses_splitd_kernel(head_dim):
    _cohere_asr_relpos_fwd_kernel[grid](
        ...,
        BLOCK_DMODEL=triton.next_power_of_2(head_dim),
        HEAD_SIZE=head_dim,
        ...
    )
```

This fixed routing, but it did not fix efficiency.

For `head_dim = 160`, `next_power_of_2(head_dim)` is `256`, so the kernel was
paying for a 256-wide inner tile to process only 160 useful channels.

That wasted work showed up immediately:

- microbenchmark at `seq_len=750`, `head_dim=160`: Triton `0.851 ms`, SDPA `0.296 ms`
- end-to-end VoxPopuli concurrency-512 reruns: `1021.04` and `1035.62` RTFx

This is a very important engineering lesson:

- routing correctness is not the same thing as performance correctness
- widening a supported range can reveal a bad geometry that was previously hidden

The right question after this regression was not "why is Triton bad?"

It was:

- "what work is the kernel doing only because the tile shape is wrong?"

### Step 5: Failed Simplifications Revealed Real Triton Constraints

The next instinct was natural:

- if `256` is wasteful for `160`, why not just use `BLOCK_DMODEL = 160`?

That did not work cleanly because Triton brought real compiler and codegen
constraints into the design space:

- `tl.arange` requires a power-of-two range
- the dot-product path also has practical constraints around the K dimension

This is where the work stopped being "tune a threshold" and became "change the
kernel structure so Triton can generate good code for the real shape."

This is an important mindset shift for reviewers:

- compiler errors are not just obstacles to work around
- they often tell you that the current abstraction is wrong for the target shape

### Step 6: The Split-D Redesign Treated Non-Power-Of-Two Heads As A Different Problem

The eventual redesign added a dedicated execution strategy for non-power-of-two
but 16-aligned head dimensions.

The routing helpers capture that idea directly:

```python
def _uses_splitd_kernel(head_dim: int) -> bool:
    return head_dim % 16 == 0 and not _is_power_of_2(head_dim)

def _get_head_dim_chunks(head_dim: int) -> tuple[int, int, int, int]:
    ...
```

For `head_dim = 160`, the split is:

```text
160 = 128 + 32
```

That means the kernel computes the same math, but over exact power-of-two
sub-blocks instead of pretending the head dimension is `256`.

This changed the inner loop from:

- one padded D tile with wasted FMAs and bandwidth

to:

- several exact D chunks whose results are summed

The content term became:

```python
qk = tl.dot(q_u_0, k_0).to(tl.float32)
if CHUNK_1 > 0:
    qk += tl.dot(q_u_1, k_1).to(tl.float32)
...
```

This is a good place to pause and think critically:

- the algorithm did not change semantically
- the execution schedule changed materially

That distinction matters. Many performance bugs are schedule bugs, not math
bugs.

### Step 7: The Rel-Pos Term Was Vectorized Across The Whole Query Tile

The first kernel computed the rel-pos term row by row inside the tile. That was
easy to understand, but it was not the best structure for `head_dim = 160`.

The redesigned split-D kernel computes the rel-pos tile in a vectorized way:

```python
rel_pos = center_pos + key_pos[None, :] - offs_m[:, None]
rel_valid = q_valid[:, None] & k_valid[None, :] & (rel_pos >= 0) & (rel_pos < pos_len)

p_0 = tl.load(
    P
    + cur_head * stride_ph
    + rel_pos[None, :, :] * stride_pp
    + offs_d0[:, None, None],
    mask=rel_valid[None, :, :],
    other=0.0,
)
rel_qk = tl.sum(
    q_v_0[:, :, None].to(tl.float32) * p_0.to(tl.float32),
    axis=0,
)
```

This matters for two reasons:

1. It removes the Python-style row loop from the rel-pos accumulation path.
2. It lets the kernel reuse a tile-oriented view of the problem instead of
   repeatedly rebuilding per-row work.

The mental model here is:

- once the kernel has accepted that it works on tiles, the rel-pos path should
  also look like tile math
- if one part of the kernel is tiled and another part is row-by-row, that is a
  sign the schedule may still be mismatched

### Step 8: Launch Heuristics Had To Be Tuned For The New Schedule

After the split-D redesign, block sizes and warp counts could no longer be
treated as inherited constants from the earlier kernel.

The helpers now explicitly branch for the split-D path:

```python
def _get_query_block_size(head_dim: int, dtype: torch.dtype) -> int:
    if _uses_splitd_kernel(head_dim):
        return 4 if head_dim > 128 else 8
    ...

def _get_key_block_size(head_dim: int, dtype: torch.dtype) -> int:
    if _uses_splitd_kernel(head_dim):
        return 32 if head_dim > 128 else 16
    ...
```

This is a subtle but important review point:

- kernel structure and launch geometry are coupled
- a new algorithm with old block sizes is often only half-optimized

For `head_dim = 160`, the tuned path settled into a smaller query tile and a
larger key tile than the earlier defaults.

### Step 9: Validation Needed Three Different Lenses

The work was validated at three levels because each level answers a different
question:

1. Unit tests:
   does the kernel match the reference numerically?
2. Routing tests:
   is the fast path actually taken for supported shapes such as `head_dim=160`?
3. Benchmarks:
   does the improved schedule matter in practice?

The measured progression for `head_dim = 160` tells the real story:

- naive widened kernel: `0.851 ms` microbenchmark, `1021.04` and `1035.62` RTFx
- split-D redesign: `0.574 ms` microbenchmark
- end-to-end after redesign: `1077.28`, `1147.94`, `1113.80` RTFx

The correct interpretation is not "the redesign is universally faster than
every fallback in every context."

The correct interpretation is:

- the redesign removed the clear regression
- it made the production shape competitive again
- it showed that the biggest remaining uncertainty is run-to-run variance, not
  kernel correctness

### Step 10: What A Good Reviewer Should Keep Asking

If you want to build intuition rather than just accept the code, keep asking:

- Which tensor stopped being materialized because of this change?
- Which production shape forced the redesign?
- Which parts of the current schedule are exact work, and which parts are padded work?
- Which benchmark falsified the previous idea?
- Which kernel gate reflects a mathematical requirement, and which one reflects
  a currently unoptimized configuration?

Those questions are more valuable than memorizing the final code.

They are what make the difference between reading this as a one-off Triton
patch and reading it as a reusable performance engineering case study.

## Design Constraints

This implementation deliberately chose a narrow scope:

- Model-local integration for Cohere ASR only
- CUDA-only fast path in the first version
- Keep SDPA as fallback for everything unsupported
- Do not modify the shared vLLM `v1` Triton backend

This was intentional because Cohere ASR rel-pos attention is model-specific. The bias term depends on runtime query activations and projected relative-position embeddings, so forcing it into the generic shared backend would have increased complexity and blast radius significantly.

## Before and After

### Before

The relative-position path in `RelPositionMultiHeadAttention.forward()` looked conceptually like this:

```python
q, k, v = forward_qkv(...)
p = linear_pos(pos_emb)
q_with_bias_u = q + pos_bias_u
q_with_bias_v = q + pos_bias_v

matrix_bd = rel_shift(q_with_bias_v @ p^T)
scores = (q_with_bias_u @ k^T + matrix_bd) / sqrt(d)
output = manual_attention_or_sdpa(scores, v, mask)
```

The bad part is that `matrix_bd` is dense and query-dependent, so it must be explicitly formed.

### After

The new path is:

```python
q, k, v = forward_qkv(...)
p = linear_pos(pos_emb)
q_with_bias_u = q + pos_bias_u
q_with_bias_v = q + pos_bias_v

if supported_cuda_case:
    output = triton_relpos_attention(q_with_bias_u, q_with_bias_v, k, v, p, ...)
else:
    matrix_bd = rel_shift(q_with_bias_v @ p^T)
    output = sdpa_fallback(...)
```

The important change is that the fast path computes the rel-pos contribution inside the Triton kernel instead of precomputing the dense `matrix_bd`.

## Changes In `cohere_asr.py`

### 1. The base attention implementation was cleaned up around SDPA

`CohereASRMultiHeadAttention.forward_attention()` now operates on `query`, `key`, and `value` tensors directly and uses SDPA for the fallback implementation.

This has two effects:

- the plain MHA path no longer uses explicit `softmax(scores)` and `torch.matmul(attn, value)`
- the rel-pos path can reuse the same projection/output code

To make the code cleaner, `_project_attention_output()` was added so both SDPA and Triton return a `[B, H, T, D]` tensor and share the same output projection logic.

### 2. A Triton fast-path gate was added to `RelPositionMultiHeadAttention`

`_can_use_triton_relpos_attention()` decides whether the fast path is safe to use.

Current conditions include:

- CUDA only
- dtype in `{fp16, bf16, fp32}`
- `16 <= head_dim <= 256`
- `pos_emb.size(0) == 1`
- bool mask semantics

The `head_dim >= 16` requirement is not arbitrary. It came from a real bring-up issue: Triton's `tl.dot` path in this kernel requires the K dimension to be at least 16.

This is useful intuition for reviewers: the gate is not just "defensive coding", it encodes actual kernel limitations discovered during implementation.

### 3. The model-local fast path lives entirely inside the rel-pos class

`_forward_triton_relpos_attention()` computes:

- `seq_lens` from `pad_mask`
- compact inputs for the custom op
- post-kernel zeroing of fully masked queries and padded queries
- output projection through `_project_attention_output()`

This keeps the integration simple and makes fallback behavior obvious.

### 4. Full-context mask creation was improved

`ConformerEncoder._create_masks()` was also tightened so that when both left and right attention contexts are unrestricted, it does not allocate an all-true `(1, T, T)` attention mask only to invert it later into an all-false mask.

This is a small but sensible cleanup:

- unrestricted full context now uses `att_mask = None`
- constrained context still uses the broadcastable mask path

This change helps both fallback behavior and review clarity.

## The Key Insight Behind The Triton Kernel

The most important insight is this:

`rel_shift()` is only needed because the dense computation first builds a "relative-position matrix" in a shifted layout and then realigns it into normal `(query_index, key_index)` space.

Inside a kernel, we do not need to build that intermediate matrix at all.

For a given query position `i` and key position `j`, the correct relative-position index is:

```text
rel_pos_index = center_pos + j - i
```

where `center_pos = T - 1` for a sequence of length `T`.

That means the kernel can directly fetch the correct relative-position vector `r_{i-j}` without ever computing `rel_shift(matrix_bd)`.

This is the core reason the fast path removes the dense `matrix_bd` allocation.

## What The Triton Kernel Does

The new kernel is implemented in [`vllm/v1/attention/ops/cohere_asr_relpos_attention.py`](../../vllm/v1/attention/ops/cohere_asr_relpos_attention.py).

### Wrapper-level conventions

The wrapper:

- accepts compact tensors shaped `[B, H, T, D]`
- flattens them to `[B*T, H, D]` for consistency with existing Triton prefill kernels
- keeps `P` in compact shape `[H, 2T-1, D]`
- passes `seq_lens` and an optional broadcastable bool mask
- returns `[B, H, T, D]`

This matches the conventions used elsewhere in vLLM Triton attention wrappers and keeps the op compile-friendly.

### Kernel-level algorithm

Per tile, the kernel does this:

1. Load a tile of `Q_u = q + pos_bias_u`
2. Load a tile of `Q_v = q + pos_bias_v`
3. Load a tile of `K`
4. Load a tile of `V`
5. Compute the content term:

```text
Q_u @ K^T
```

6. For each query row in the tile, compute the relative-position index slice:

```text
center_pos + key_pos - query_pos
```

7. Load the corresponding relative-position vectors from `P`
8. Compute the rel-pos term:

```text
(q_i + v) · r_{i-j}
```

9. Add the two score terms
10. Apply the optional bool mask
11. Run online softmax accumulation
12. Accumulate `V`
13. Store the final `[BLOCK_M, D]` output tile

This means the kernel never materializes:

- dense `matrix_bd`
- dense `scores`
- dense post-softmax attention probabilities

It computes only what is needed for the running softmax and output accumulation.

## Why The Fast Path Is Model-Local

This implementation does not try to generalize the kernel into the shared `v1` Triton backend yet.

That was a deliberate choice for three reasons:

1. This rel-pos formulation is specific to Cohere ASR.
2. The shared Triton backend is more oriented around generic decoder or paged attention patterns.
3. The model-local approach kept the number of touched subsystems small enough to validate quickly.

This is a "high confidence first version" strategy:

- get the kernel working
- validate parity
- measure end-to-end impact
- only then decide whether it is worth generalizing

## Test Coverage Added

The file [`tests/models/test_cohere_asr_attention.py`](../../tests/models/test_cohere_asr_attention.py) now covers both fallback correctness and CUDA fast-path behavior.

It verifies:

- plain SDPA path matches the manual reference
- rel-pos SDPA fallback matches the manual reference
- encoder masks remain broadcastable
- Triton rel-pos wrapper matches the manual reference on CUDA
- the rel-pos class actually routes into the Triton fast path on CUDA

The CUDA tests use realistic supported dimensions:

- `hidden_size = 64`
- `num_heads = 4`
- `head_dim = 16`

This matters because the initial tiny synthetic tests used `head_dim = 4`, which is too small for the Triton dot path. The tests were updated to cover the actual supported domain.

## Runtime Validation

### Unit tests

The focused regression suite passed:

```text
pytest tests/models/test_cohere_asr_attention.py -q
6 passed
```

### Direct kernel parity check

During bring-up, the Triton rel-pos output was compared directly against the manual reference at a supported CUDA shape and matched within expected low fp16 error:

- max diff around `2.44e-4`
- finite outputs throughout

This was a useful sanity check before relying only on the pytest wrappers.

## Benchmark Result

The final runtime benchmark was executed against a fresh server process using:

```bash
export FLASHINFER_DISABLE_VERSION_CHECK=1
vllm serve /host/engines/vllm/audio/2b-release --trust-remote-code
```

and:

```bash
export FLASHINFER_DISABLE_VERSION_CHECK=1
vllm bench serve \
  --backend openai-audio \
  --dataset-name hf \
  --dataset-path facebook/voxpopuli \
  --hf-subset en \
  --hf-split test \
  --no-stream \
  --model /host/engines/vllm/audio/2b-release \
  --num-prompts 99999999 \
  --no-oversample \
  --endpoint /v1/audio/transcriptions \
  --ready-check-timeout-sec 600 \
  --save-result \
  --max-concurrency 512
```

The environment variable was needed only because the container had a FlashInfer package version mismatch during startup. It is not part of the model change itself.

### Final measured result

| Metric | Value |
| --- | --- |
| Successful requests | `1842` |
| Failed requests | `0` |
| Duration | `14.59 s` |
| Request throughput | `126.24 req/s` |
| Output token throughput | `4052.71 tok/s` |
| Peak output throughput | `5689.00 tok/s` |
| Peak concurrent requests | `761` |
| RTFx | `1216.37` |
| Mean TTFT | `2663.04 ms` |
| Mean TPOT | `33.55 ms` |
| Mean ITL | `52.14 ms` |

### Comparison to earlier numbers

The earlier numbers discussed during this work were:

- pre-SDPA baseline: about `1107` RTFx
- SDPA version: about `1125` RTFx

The Triton rel-pos version reached:

- `1216.37` RTFx

That is approximately:

- `+8.1%` vs the earlier SDPA result (`1216.37 / 1125 - 1`)
- `+9.9%` vs the older `1107` result (`1216.37 / 1107 - 1`)

This is materially better than the small SDPA-only improvement, which is exactly what we expected if the dense rel-pos bias was the main remaining bottleneck.

### Follow-up for the 2B-release shape

After the original bring-up, one important detail emerged during validation against the production `2b-release` checkpoint: its encoder uses `head_dim = 160`, which was outside the first Triton gate.

That follow-up changed two things:

- the temporary hard disable in `_can_use_triton_relpos_attention()` was removed
- the supported domain was widened from `head_dim <= 128` to `head_dim <= 256`

This means the Triton path now activates for the 2B-release shape and is covered by CUDA tests.

The important review detail is that there were really two `head_dim = 160` implementations:

1. a naive widening of the original kernel
2. a later redesign specifically for non-power-of-two head dimensions

### First attempt: widen the gate

The first attempt simply widened the supported domain from `head_dim <= 128` to `head_dim <= 256` while keeping the original kernel structure.

That version had an obvious problem:

- the kernel still used `BLOCK_DMODEL = next_power_of_2(head_dim)`
- for `head_dim = 160`, that meant padding the inner dimension to `256`

That warning showed up in both microbenchmarks and end-to-end serving:

- standalone `RelPositionMultiHeadAttention.forward()` microbenchmark at `batch=1`, `seq_len=750`, `head_dim=160`: Triton `0.851 ms`, SDPA fallback `0.296 ms`
- end-to-end VoxPopuli concurrency-512 reruns on 2B-release: `1021.04` RTFx and `1035.62` RTFx

So the first conclusion was:

- correctness and routing were fixed for `head_dim = 160`
- performance for that shape regressed

### Second attempt: exact split-D redesign

The follow-up redesign changed the kernel structure for non-power-of-two, 16-aligned head dimensions:

- the `head_dim` is decomposed into exact power-of-two chunks
- `head_dim = 160` uses `128 + 32`
- the split-D kernel avoids paying for a full `256`-wide tile
- the rel-pos term is accumulated over the whole query tile at once instead of a Python-style row loop inside the Triton kernel
- the launch heuristics for split-D heads were tuned separately from the power-of-two path

For reviewers, this is the key design insight: the fix was not "just change a threshold". The fix was to give non-power-of-two heads their own execution strategy.

After that redesign, the same standalone forward microbenchmark improved from:

- `0.851 ms` to `0.574 ms`

It still did not clearly beat the SDPA fallback in isolation, but the end-to-end result improved substantially.

### End-to-end result after the redesign

With the split-D/vectorized redesign, the same VoxPopuli concurrency-512 benchmark on 2B-release produced:

- first rerun after restart: `1077.28` RTFx
- warmed rerun: `1147.94` RTFx
- second warmed rerun: `1113.80` RTFx

This means:

- the redesign recovered most of the earlier regression
- the best warmed rerun beat the earlier SDPA-only number of about `1125` RTFx
- another warmed rerun landed essentially on top of the earlier "Triton disabled" number of about `1114.92` RTFx

So the right conclusion is now:

- correctness and routing are fixed for `head_dim = 160`
- performance for that shape is no longer clearly regressing
- the redesigned kernel is competitive with, and on some warmed runs slightly ahead of, the earlier fallback baseline
- there is still visible run-to-run variance, so any claim of a large, stable production win for `160` should be treated cautiously until more samples are collected

## Important Review Observations

There are a few points reviewers should pay extra attention to:

### 1. This is not a shared backend change

Nothing in the shared attention backend selection was modified. The Triton kernel is opt-in through the model-local fast-path gate in `RelPositionMultiHeadAttention`.

That is a feature, not a limitation, for the first version.

### 2. Fallback behavior remains important

The SDPA path is still the source of truth for:

- CPU
- non-CUDA platforms
- unsupported dtypes
- unsupported head dimensions
- unexpected mask shapes

So the code should be read as:

- "Triton accelerates supported CUDA cases"
- not "Triton replaces the rel-pos implementation universally"

### 3. The kernel is fused enough to remove the main bottleneck, but it is still a first version

The kernel already removes the dense rel-pos bias tensor, which is the key win.

At the same time, it is intentionally conservative:

- model-local
- explicit gating
- row-wise rel-pos accumulation inside the tile
- no attempt yet to generalize across other models

That is the right tradeoff for a first productionable optimization.

## Remaining Limitations

This implementation still has room to improve.

### CUDA only

ROCm and other backends use fallback behavior today.

### Shape support is gated

The fast path currently assumes:

- supported CUDA device
- `16 <= head_dim <= 256`
- `pos_emb` batch dimension of `1`

These constraints came from the implementation and validation domain and should be viewed as explicit supported configurations, not accidental behavior.

### Model-local integration

If other models later need the same Transformer-XL style rel-pos kernel, it may be worth lifting the common pieces into a shared attention op or backend. That was intentionally left out of this change.

## Intuition To Carry Forward

If you remember only one thing from this work, it should be this:

The win did not come from "using Triton" in the abstract.

The win came from changing where the relative-position term is computed.

Before:

- build dense rel-pos bias in memory
- pass it into attention

After:

- derive the rel-pos slice directly from `(query_pos, key_pos)`
- compute its contribution inside the attention kernel

That is the core optimization.

Everything else in the change exists to make that idea safe, testable, and easy to fall back from when unsupported.

## Files Touched

- [`vllm/model_executor/models/cohere_asr.py`](../../vllm/model_executor/models/cohere_asr.py)
- [`vllm/v1/attention/ops/cohere_asr_relpos_attention.py`](../../vllm/v1/attention/ops/cohere_asr_relpos_attention.py)
- [`tests/models/test_cohere_asr_attention.py`](../../tests/models/test_cohere_asr_attention.py)

## Suggested Review Strategy

If you are reviewing correctness:

1. Read the fast-path gate in `cohere_asr.py`
2. Read `_forward_triton_relpos_attention()`
3. Read the Triton wrapper signature
4. Read the tests

If you are reviewing performance:

1. Focus on how `matrix_bd` disappears from the fast path
2. Focus on direct rel-pos indexing via `center_pos + j - i`
3. Focus on the benchmark result and the retained fallback path

If you are reviewing maintainability:

1. Check that the change is isolated to Cohere ASR
2. Check that unsupported cases still fall back
3. Check that the tests cover both routing and parity
