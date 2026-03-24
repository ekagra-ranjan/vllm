# Cohere ASR RTFx Plan Execution

## Goal

This note summarizes the optimization pass that followed the earlier
tokenizer/prompt-cache work.

The goal of this pass was not to improve bounded microbenchmarks in isolation.
The goal was to improve the real saturated benchmark:

- workload: `facebook/voxpopuli`, subset `en`, split `test`
- endpoint: `/v1/audio/transcriptions`
- concurrency: `512`
- acceptance gate: warmed `LibriSpeech -> VoxPopuli` run inside Docker with
  `FLASHINFER_DISABLE_VERSION_CHECK=1`

Frozen starting baseline for this pass:

| Metric | Baseline |
| --- | ---: |
| Req/s | `108.88` |
| Tok/s | `3494.94` |
| RTFx | `1049.03` |
| Mean TTFT (ms) | `2218.87` |

## What Changed In The Tree

The final code left in the tree after this pass keeps only two classes of
changes:

1. A production behavior change for frontend preprocessing.
2. Dev-mode-only measurement hooks for frontend and encoder attribution.

Persistent code changes:

- `vllm/model_executor/models/cohere_asr.py`
  - default Cohere ASR preprocess thread count now falls back to `8` when
    `VLLM_COHERE_ASR_PREPROCESS_THREADS` is unset
  - new dev-mode encoder measurements:
    - pad-efficiency stats before and after subsampling
    - attention layout timers
    - layernorm timers
    - residual-add timers
    - encoder input/output layout timers
    - output projection timer
- `vllm/transformers_utils/processors/cohere_asr.py`
  - new frontend measurement counters for actual feature-extractor batch size,
    raw sample volume, max sample length, and same-call pad waste

Changes that were prototyped but reverted after benchmarking:

- cross-request raw-audio microbatching in
  `vllm/entrypoints/openai/speech_to_text/speech_to_text.py`
- filterbank direct power-spectrum fast path in
  `vllm/transformers_utils/processors/cohere_asr.py`
- encoder width trimming before `pre_encode` and after subsampling in
  `vllm/model_executor/models/cohere_asr.py`

## Summary Table

| Change | Motivation | Kept? | Warmed VoxPopuli outcome |
| --- | --- | --- | --- |
| frontend batch-shape measurement | determine whether batching opportunity is real | yes | diagnostic only |
| preprocess thread sweep | frontend filterbank path looked CPU-bound | yes, best setting kept as default | best tested run: `RTFx 1165.92` |
| cross-request audio microbatching | real workload almost never batches in `extract_features()` | no | `RTFx 1102.91` |
| filterbank fast path | reduce CPU passes inside feature extraction | no | `RTFx 1149.53` |
| encoder pad/glue measurement | choose structural encoder work from data | yes | diagnostic only |
| encoder width trimming | attack measured `~45%` encoder pad waste | no | `RTFx 1146.95` |

Final accepted-code rerun:

| Metric | Final accepted code |
| --- | ---: |
| Req/s | `119.94` |
| Tok/s | `3849.53` |
| RTFx | `1155.58` |
| Mean TTFT (ms) | `1799.02` |

That final state is materially better than the frozen baseline even though some
single candidate runs were slightly higher. The accepted code keeps only changes
that consistently cleared the end-to-end acceptance bar.

## 1. Frontend Measurement Pass

### Motivation

The system-level breakdown already said preprocessing was large, but that did
not answer the first practical question:

- does the real `c512` traffic naturally batch audio clips together, or is the
  feature extractor still mostly operating on single-request clips?

That answer matters because it determines whether the cheapest next move is:

- a thread sweep
- cross-request batching
- or intra-call padding reduction

### Implementation

In `vllm/transformers_utils/processors/cohere_asr.py`,
`CohereASRFeatureExtractor.extract_features()` was instrumented to record:

- `feature_batch_size_items`
- `feature_batch_raw_samples`
- `feature_batch_max_samples`
- `feature_batch_pad_waste_ratio`

These were added to the existing `StageTimingRegistry` path so they show up in
the same dev-mode ASR debug endpoint as the timing stats.

### Observation

One dev-mode VoxPopuli `c512` pass showed:

| Signal | Value |
| --- | ---: |
| average feature batch size | `1.003` |
| max feature batch size | `3` |
| average same-call pad waste | `0.10%` |
| average raw samples per call | `154,138` |

### Why It Was Observed

The real request path almost always presents the feature extractor with one clip
at a time. The current workload shape does not create meaningful same-call audio
microbatches, and the clips inside each call are already very tightly packed.

That means:

- intra-call padding reduction is not a meaningful frontend win
- any batching win has to come from cross-request batching
- the cheapest first experiment is the existing preprocess thread knob

## 2. Preprocess Thread Sweep

### Motivation

The feature extractor is CPU-side and the hottest nested frontend bucket is
still the filterbank path. The existing renderer/MM wrapper was effectively
running this path with the default Torch thread policy, which in practice meant
`1` thread unless overridden.

This made thread count the highest-confidence cheap frontend experiment.

### Implementation

In `vllm/model_executor/models/cohere_asr.py`,
`CohereASRMultiModalProcessor.get_torch_num_threads()` now does:

- if `VLLM_COHERE_ASR_PREPROCESS_THREADS` is set and valid, use it
- otherwise, fall back to `min(os.cpu_count(), 8)` with a floor of `1`

So the production default is now `8`, but the environment variable still has
full control for manual sweeps or rollback.

### Observation

Warmed `LibriSpeech -> VoxPopuli` `c512` sweep:

| Threads | Req/s | Tok/s | RTFx | Mean TTFT (ms) |
| --- | ---: | ---: | ---: | ---: |
| `1` | `97.29` | `3122.45` | `937.35` | `3188.48` |
| `2` | `100.75` | `3233.44` | `970.68` | `2958.86` |
| `4` | `110.17` | `3535.97` | `1061.49` | `2471.24` |
| `8` | `121.01` | `3883.93` | `1165.92` | `1938.13` |

### Why It Was Observed

This is the cleanest result from the whole pass.

The frontend feature path is CPU-bound enough that allowing more Torch CPU
threads improved the real saturated benchmark directly. Unlike several earlier
encoder-local wins, this one translated cleanly because it reduced a large
system-level bucket instead of a small nested sub-bucket.

### Final Decision

Kept.

This is the only production behavior change from this pass that remained in the
final code.

## 3. Cross-Request Audio Microbatching

### Motivation

The frontend measurement pass showed that `extract_features()` almost never saw
natural batching, even though the real benchmark was highly concurrent. That
made cross-request microbatching the most plausible next frontend structural
experiment.

### Implementation

Prototype only, later reverted.

The attempt:

- added an async cross-request raw-audio microbatcher in
  `vllm/entrypoints/openai/speech_to_text/speech_to_text.py`
- batched concurrent single-chunk requests before feature extraction
- reused the existing
  `CohereASRForConditionalGeneration.batch_preprocess_audio_chunks()` helper
  in `vllm/model_executor/models/cohere_asr.py`

During debugging, the helper was also updated so each returned item was trimmed
back to its own `length`, because the first version preserved the longest padded
width from the whole microbatch for every request.

### Observation

Warmed result after correctness fix:

| Metric | Result |
| --- | ---: |
| Req/s | `114.47` |
| Tok/s | `3674.39` |
| RTFx | `1102.91` |
| Mean TTFT (ms) | `2288.53` |

The first version also triggered worker OOM because every request inherited the
widest feature tensor in the microbatch.

### Why It Was Observed

Even after fixing the OOM bug, the cross-request batching overhead diluted the
local feature-extractor savings:

- queueing and microbatch coordination added latency
- packing/unpacking work moved into the request path
- the frontend saved some CPU work, but not enough to beat the simpler
  thread-only win

This is a useful negative result because it shows that "batch more audio" is not
automatically a throughput win once request-path overhead is included.

### Final Decision

Reverted.

## 4. Filterbank Fast Path

### Motivation

The nested frontend breakdown still showed `filterbank_forward_secs` dominating
the feature extractor. The specific idea here was to reduce one obvious extra
CPU pass in the common Cohere ASR configuration, which uses power spectrograms.

### Implementation

Prototype only, later reverted.

The attempted change in `vllm/transformers_utils/processors/cohere_asr.py`:

- replaced the usual `abs()` then `square()` style path with a direct
  power-spectrum path using the complex STFT output directly
- the intention was to reduce memory traffic and elementwise work in the hot
  filterbank path

### Observation

Warmed result:

| Metric | Result |
| --- | ---: |
| Req/s | `119.31` |
| Tok/s | `3829.83` |
| RTFx | `1149.53` |
| Mean TTFT (ms) | `1848.95` |

### Why It Was Observed

This optimization attacked a real hotspot, but only one slice of it.
The broader filterbank pipeline still includes:

- STFT
- mel projection
- log
- normalization

So the rewritten power-spectrum step was not large enough to overtake the
thread-default win at full saturation.

### Final Decision

Reverted.

## 5. Encoder Structural Measurement

### Motivation

Earlier encoder-local work on attention and compile islands did not survive the
saturated end-to-end benchmark. Before trying another structural encoder change,
the missing question was:

- is the real opportunity still hidden in untimed glue, or is it in batch
  padding waste?

### Implementation

In `vllm/model_executor/models/cohere_asr.py`, dev-mode-only measurements were
added for:

- encoder pad efficiency
  - `encoder_input_pad_efficiency`
  - `encoder_subsampled_pad_efficiency`
- Conformer glue
  - `encoder_layernorm_secs`
  - `encoder_residual_add_secs`
- attention layout work
  - `encoder_attention_qkv_layout_secs`
  - `encoder_attention_output_layout_secs`
- encoder boundary layout work
  - `encoder_input_layout_secs`
  - `encoder_output_layout_secs`
  - `encoder_output_proj_secs`

These are gated so the expensive scalar observations only run in dev mode.

### Observation

One dev-mode VoxPopuli `c512` pass showed:

| Signal | Value |
| --- | ---: |
| encoder batch time | `142.16 ms` |
| pre-encode time | `11.20 ms` |
| subsampling conv time | `10.39 ms` |
| input pad efficiency | `55.37%` |
| subsampled pad efficiency | `55.49%` |
| layernorm total | `0.886 s` |
| residual-add total | `0.803 s` |
| qkv layout total | `0.104 s` |
| attention output layout total | `0.170 s` |

### Why It Was Observed

The important conclusion is not that glue is zero. It is that glue is smaller
than the structural waste:

- encoder batches are only about `55%` live tokens
- about `45%` of the padded width is waste before and after subsampling
- newly measured layout and norm/residual buckets are real, but they are not as
  large as the padding opportunity

That shifted the next encoder prototype toward padded-width reduction instead of
another narrow kernel change.

## 6. Encoder Width Trimming Prototype

### Motivation

Given the measured `~45%` encoder pad waste, the most direct structural test was
to trim padded widths:

- once before `pre_encode`
- once after subsampling before the Conformer layers

### Implementation

Prototype only, later reverted.

The attempt in `vllm/model_executor/models/cohere_asr.py`:

- sliced frontend feature tensors to the longest real feature length in the
  batch before entering the encoder
- sliced post-subsampling activations to the longest real subsampled length
  before the encoder stack

### Observation

Stable warmed rerun:

| Metric | Result |
| --- | ---: |
| Req/s | `119.04` |
| Tok/s | `3820.72` |
| RTFx | `1146.95` |
| Mean TTFT (ms) | `1876.37` |

### Why It Was Observed

The measured padding waste was real, but reducing padded width alone still did
not beat the frontend thread-default win.

Most likely reasons:

- width trimming helps, but the remaining encoder cost is still dominated by the
  real work in pre-encode, attention, conv, and FFN
- shape trimming introduces some countervailing overhead and does not change the
  higher-level scheduling structure of the workload
- the frontend thread win attacked a large system-level bottleneck directly,
  while encoder width trimming only reduced part of the largest worker bucket

### Final Decision

Reverted.

## Why The Final Result Looks The Way It Does

The accepted result from this pass is mostly explained by one fact:

- the frontend filterbank path was a large CPU-side bottleneck, and letting it
  use more CPU threads produced a direct system-level improvement

The rejected experiments failed for different reasons:

- cross-request microbatching added too much coordination overhead
- the filterbank fast path reduced too small a slice of total filterbank work
- encoder width trimming reduced real waste, but not enough to beat the simpler
  frontend win

That pattern is consistent with the earlier lesson from attention and
`torch.compile` experiments: local wins only matter if they move a large enough
system-level bucket.

## Final Accepted State

Final warmed `LibriSpeech -> VoxPopuli` `c512` result on the accepted code:

| Metric | Final |
| --- | ---: |
| Req/s | `119.94` |
| Tok/s | `3849.53` |
| RTFx | `1155.58` |
| Mean TTFT (ms) | `1799.02` |

Compared with the frozen starting baseline:

| Metric | Baseline | Final | Delta |
| --- | ---: | ---: | ---: |
| Req/s | `108.88` | `119.94` | `+11.06` |
| Tok/s | `3494.94` | `3849.53` | `+354.59` |
| RTFx | `1049.03` | `1155.58` | `+106.55` |
| Mean TTFT (ms) | `2218.87` | `1799.02` | `-419.85` |

## Next Steps

Recommended next steps, in order:

1. Run the small fixed transcript-quality sanity set on the accepted thread
   default, because this pass focused on throughput and did not rerun the full
   quality check.
2. Keep using the new dev-mode frontend and encoder measurements as the first
   screen for future candidates.
3. If frontend work is revisited, prefer larger filterbank changes over more
   request-path batching logic:
   - faster STFT backend
   - fused mel/log/normalization work
   - inference-specific simplifications that remove whole CPU passes
4. If encoder work is revisited, use the new pad-efficiency data to guide a
   bigger structural idea rather than another micro-optimization:
   - tighter length bucketing upstream
   - subsampling specialization
   - a structural conformer-conv simplification that changes real batch work
5. Avoid revisiting:
   - more attention mode experiments
   - submodule-only `torch.compile` islands
   - tiny glue-only optimizations without evidence that they move warmed `RTFx`

## Takeaway

This pass was useful because it separated "interesting local speedups" from
"changes that actually improve the real throughput benchmark."

The best result came from a simple frontend CPU-threading change.
The most useful long-term output, besides that production win, is the new
measurement coverage that now makes future frontend and encoder decisions much
less speculative.
