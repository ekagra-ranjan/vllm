ls "/root/repos/vllm-ekagra/vllm" && python3 - <<'PY'
from pathlib import Path
report = '''# Cohere ASR Bottleneck Profiling Report

## Goal

This report documents exactly what I ran, what instrumentation I added, why I added it, how the bounded `torch` profiler runs were configured, and how the final bottleneck ranking was derived.

The original goal was to answer:

1. Where is time going in the Cohere ASR vLLM serving path under the real high-throughput workload?
2. Is the bottleneck in frontend preprocessing, IPC, scheduler overhead, encoder, decoder, or pad-to-pack conversion?
3. Which component should be optimized first?

The target workload was the user's VoxPopuli EN benchmark with:

- model: `/host/engines/vllm/audio/2b-release`
- benchmark client: `vllm bench serve`
- endpoint: `/v1/audio/transcriptions`
- concurrency target: `512`
- dataset: `facebook/voxpopuli`, subset `en`, split `test`

## Executive Summary

The current bottleneck is the frontend render path wrapped by `renderer_render_secs`, not the HF audio feature extractor, IPC, or scheduler.

The most important findings are:

1. The dominant top-level bucket is `renderer_render_secs` in `vllm/entrypoints/openai/speech_to_text/speech_to_text.py`, with `1863.86s` accumulated over the full saturation run, or `1.011s` per request on average.
2. The actual multimodal/audio preprocessing is small by comparison: `preprocessor_total_secs` is only `6.05s` total, or `3.27ms/request`.
3. IPC/request marshaling is tiny: client encode + client send + engine decode + engine preprocess is only about `0.336s` total across the full run.
4. Scheduler CPU overhead is also tiny: `0.53s` total.
5. On the worker/GPU side, encoder work dominates the measured model time. Within the measured worker buckets, encoder forward is first and pad-to-pack conversion is second.
6. The bounded `torch` profiler runs confirm that, inside the encoder, relative-position attention and GEMM-heavy work are the main CUDA consumers.

The immediate optimization target should be the frontend render/tokenize/process-for-engine path inside `render_cmpl_async()`. After that, the next GPU target should be encoder attention/GEMM work, followed by `pack_encoder_outputs`.

## Important Interpretation Note

The timing totals in this report are **accumulated request-time**, not wall-clock time.

Because many requests overlap at high concurrency, the sum of stage totals is much larger than the benchmark wall-clock duration. For example:

- full saturation benchmark wall clock: `16.45s`
- accumulated `renderer_render_secs`: `1863.86s`

That does **not** mean the server spent `1863s` of real wall time rendering. It means requests collectively spent that much time inside that timed region. The percentages below therefore represent:

- share of measured accumulated stage-time

not:

- share of end-to-end wall-clock runtime

This distinction is especially important for async/frontend wrapper timers, which can include queueing or backpressure while a request is waiting within an async stage.

## What I Did

I followed a staged profiling plan so that the first pass used the lowest-overhead signals possible, and only then added targeted instrumentation for missing buckets.

### Step 1: Reproduce the Throughput Baseline

I first reproduced the saturation workload and a concurrency sweep so I had a throughput baseline before changing anything.

The main saturation benchmark was the same workload shape the user described, run inside the container with:

```bash
docker exec -e FLASHINFER_DISABLE_VERSION_CHECK=1 vllm-ekagra bash -lc '
  cd /host/vllm-ekagra/vllm &&
  time vllm bench serve \
    --backend openai-audio \
    --dataset-name hf \
    --dataset-path facebook/voxpopuli \
    --hf-subset en \
    --hf-split test \
    --no-stream \
    --trust-remote-code \
    --model /host/engines/vllm/audio/2b-release \
    --num-prompts 99999999 \
    --no-oversample \
    --endpoint /v1/audio/transcriptions \
    --ready-check-timeout-sec 600 \
    --save-result \
    --max-concurrency 512
'
```

I also ran a sweep at concurrency `{1, 4, 16, 64, 256, 512}` with `256` prompts to understand where throughput flattened and where TTFT began to rise sharply.

### Step 2: Reuse Existing Timing Hooks Before Editing Code

Before adding new timers, I checked what low-overhead timing hooks already existed in the codebase.

The key existing hook was in `vllm/v1/worker/gpu_model_runner.py`:

- `timed_encoder_operation(...)`
- `get_encoder_timing_stats()`

That hook already measured per-request encoder forward time on the worker side and synchronized around the encoder region. I reused it rather than adding redundant new encoder timing.

I also relied on vLLM's existing multimodal processor stats machinery, but that required making sure `enable_mm_processor_stats` was actually turned on for `vllm serve`.

### Step 3: Add Only the Missing Timers

After checking the existing hooks, I added timers only for buckets that were still missing but were relevant to the user's bottleneck question:

- frontend audio decode / chunking / prompt build / renderer wrapper
- HF feature extraction internals
- client-side request serialization and send
- engine-side request decode and add preprocessing
- scheduler total CPU time
- model-side encoder projection, pad-to-pack conversion, concat, and decoder forward

The goal was to avoid blanketing the code with heavy instrumentation and to keep the full saturation benchmark as realistic as possible.

### Step 4: Add One Aggregated Debug Endpoint

To make collection practical, I added one dev-only endpoint that gathers and resets all timing registries in a single call:

- `POST /debug/asr_bottleneck_stats`

That endpoint collects:

- frontend speech-to-text timers
- frontend HF feature extractor timers
- frontend multimodal processor stats
- core client IPC timers
- engine core timers
- scheduler timers
- worker model timers
- worker encoder per-request timers

The endpoint was deliberately resettable so I could:

1. clear all registries before a run
2. run the benchmark
3. fetch one clean snapshot after the run

### Step 5: Use Bounded `torch` Profiler Runs Instead of Profiling the Full 512-Concurrency Saturation Run

I did **not** try to run `torch` profiler on the entire full-concurrency saturation workload.

That would have had two problems:

1. it would generate very large traces
2. the profiler overhead itself would distort the very queueing behavior I was trying to understand

Instead, I used short, bounded, random-audio runs through `nick-bench.py` to fingerprint the steady-state worker kernel mix with representative clip durations.

This gave me:

- one frontend-inclusive trace
- one worker-only trace around `8s` clips
- one worker-only trace around `20s` clips

## Instrumentation Additions

This section lists what I added, where, and why.

### 1. Shared Timing Registry

File:

- `vllm/utils/stage_timing.py`

What I added:

- `StageTimingStat`
- `StageTimingRegistry`

Why:

- I wanted one small reusable primitive for low-overhead wall-clock timing across the frontend, engine core, and worker code paths.
- The registry is thread-safe and supports `record(...)`, `add(...)`, `snapshot()`, and `snapshot_and_reset()`.

### 2. Frontend Speech-to-Text Path

Files:

- `vllm/entrypoints/openai/speech_to_text/speech_to_text.py`
- `vllm/renderers/base.py`

What I instrumented:

In `speech_to_text.py`:

- `audio_decode_resample_secs`
- `audio_chunking_secs`
- `language_detect_secs`
- `prompt_build_secs`
- `renderer_render_secs`

In `renderers/base.py`:

- `render_cmpl_total_secs`
- `render_prompts_secs`
- `tokenize_prompts_secs`
- `apply_prompt_extras_secs`
- `process_for_engine_secs`

Why:

- The user explicitly asked whether bottlenecks might be in frontend preprocessing, request processing, or audio handling.
- The first pass showed that `renderer_render_secs` dominated the top-level accumulated time, so I split `render_cmpl_async()` to isolate render, tokenize, extras, and process-for-engine time.

What this timer covers:

- `renderer_render_secs` in `speech_to_text.py` still wraps `renderer.render_cmpl_async(...)` as a compatibility total.
- The new `renderers/base.py` timers split that total into:
- `render_prompts_secs`
- `tokenize_prompts_secs`
- `apply_prompt_extras_secs`
- `process_for_engine_secs`

So the report can now distinguish “broad renderer time” from the specific tokenizer sub-bucket that dominates it.

### 3. Cohere ASR Feature Extractor / Filterbank

File:

- `vllm/transformers_utils/processors/cohere_asr.py`

What I instrumented:

- `feature_batch_prep_secs`
- `feature_tensorize_secs`
- `filterbank_forward_secs`
- `feature_to_cpu_secs`
- `feature_return_tensors_secs`

Why:

- This is the actual CPU/GPU-side audio feature extraction path for Cohere ASR.
- The user explicitly called out the feature preprocessor as a possible bottleneck.
- Splitting the feature extractor let me distinguish:
  - padding and batching overhead
  - tensorization overhead
  - filterbank/STFT/mel work
  - CPU transfer / return tensor conversion

### 4. Client-Side IPC / Serialization

File:

- `vllm/v1/engine/core_client.py`

What I instrumented:

- `request_encode_secs`
- `request_send_secs`

Why:

- The user asked whether IPC and serialization/deserialization were currently hurting throughput.
- These timers measure:
  - encoding the request into the msgpack/ZMQ payload
  - sending the multipart message to the engine core

This was important to quantify whether IPC was worth optimizing before touching the model.

### 5. Engine Core Request Decode and Preprocessing

File:

- `vllm/v1/engine/core.py`

What I instrumented:

- `input_socket_decode_add_secs`
- `preprocess_add_request_secs`

What else I added:

- `get_asr_debug_timing_stats()`

Why:

- This captures the engine-side half of the request marshaling path:
  - request decode from the input socket
  - request object creation / preprocessing before scheduling
- `get_asr_debug_timing_stats()` was added so the dev endpoint could fetch engine-core and scheduler timers in one call.

### 6. Scheduler CPU Time

File:

- `vllm/v1/core/sched/scheduler.py`

What I instrumented:

- `schedule_total_secs`

Why:

- The user explicitly asked whether vLLM scheduler overhead might be a bottleneck.
- Measuring the total scheduler call was enough for the first pass. I did not split every internal scheduler subphase because that would increase complexity before we knew whether scheduler time was even material.

### 7. Worker Model Path

File:

- `vllm/model_executor/models/cohere_asr.py`

What I instrumented:

- `encoder_forward_batch_secs`
- `encoder_decoder_proj_secs`
- `pack_encoder_outputs_secs`
- `get_encoder_outputs_total_secs`
- `concat_encoder_outputs_secs`
- `decoder_forward_secs`
- `logits_forward_secs`

Why:

- The user explicitly pointed to the pad-to-pack conversion and asked for encoder vs decoder vs conversion breakdown.
- The existing worker hook in `gpu_model_runner.py` already gave me per-request encoder timing.
- What was missing was the detailed breakdown inside the model wrapper:
- `encoder_decoder_proj_secs`
- `pack_encoder_outputs_secs`
- `concat_encoder_outputs_secs`
- `decoder_forward_secs`
- `logits_forward_secs`

Methodology correction:

- For GPU stages (`encoder_forward_batch_secs`, `encoder_decoder_proj_secs`, `concat_encoder_outputs_secs`, `decoder_forward_secs`, `logits_forward_secs`) I changed the timers to be device-complete in dev mode by synchronizing before and after the region.
- The synchronization is skipped during CUDA graph capture so server startup and graph capture remain legal.
- I intentionally kept `pack_encoder_outputs_secs` as a wall-clock timer because it is a Python-side loop over packed outputs rather than a pure CUDA kernel region.

This let me answer whether pad-to-pack was worth optimizing relative to the rest of the worker model time.

### 8. Worker Exposure of Model Timers

File:

- `vllm/v1/worker/gpu_worker.py`

What I added:

- `get_asr_model_timing_stats()`

Why:

- The debug endpoint needed a way to ask the worker for the model-specific timing registry.
- The worker had to expose a getter that called `get_and_reset_timing_stats()` on the wrapped model.

### 9. Dev-Only Timing Aggregation Endpoint

File:

- `vllm/entrypoints/serve/rpc/api_router.py`

What I added:

- `_aggregate_per_request_stage_stats(...)`
- `_merge_stage_snapshots(...)`
- `_aggregate_encoder_timing_stats(...)`
- `frontend.renderer` aggregation via `get_and_reset_renderer_timing_stats()`
- `POST /debug/asr_bottleneck_stats`

Why:

- The instrumentation lives in several different layers:
  - frontend
  - renderer/mm processor
  - engine core
  - worker
- This endpoint made it possible to collect a consistent snapshot from all of them at once and then reset the registries.

### 10. Dev-Mode Enabling of MM Processor Stats

File:

- `vllm/entrypoints/openai/api_server.py`

What I added:

- when `VLLM_SERVER_DEV_MODE=1`, force `args.enable_mm_processor_stats = True`

Why:

- The multimodal processor stats existed, but `vllm serve` was not accepting the expected CLI flag directly in the way needed here.
- Without forcing this on in dev mode, the renderer-side multimodal timing registry would not have emitted the stats needed for the profiling report.

## How I Collected the Data

### A. Full Saturation Timing Run

I started an instrumented dev server so the debug endpoint would be available and the multimodal processor stats would be enabled:

```bash
docker exec \
  -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
  -e VLLM_SERVER_DEV_MODE=1 \
  vllm-ekagra bash -lc '
    cd /host/vllm-ekagra/vllm &&
    vllm serve /host/engines/vllm/audio/2b-release --trust-remote-code
  '
```

Then I reset the timing registries before benchmarking:

```bash
docker exec vllm-ekagra bash -lc \
  "curl -s -X POST http://localhost:8000/debug/asr_bottleneck_stats >/dev/null"
```

Then I ran the full VoxPopuli saturation benchmark and fetched the timing snapshot afterwards:

```bash
docker exec -e FLASHINFER_DISABLE_VERSION_CHECK=1 vllm-ekagra bash -lc '
  cd /host/vllm-ekagra/vllm &&
  time vllm bench serve \
    --backend openai-audio \
    --dataset-name hf \
    --dataset-path facebook/voxpopuli \
    --hf-subset en \
    --hf-split test \
    --no-stream \
    --trust-remote-code \
    --model /host/engines/vllm/audio/2b-release \
    --num-prompts 99999999 \
    --no-oversample \
    --endpoint /v1/audio/transcriptions \
    --ready-check-timeout-sec 600 \
    --save-result \
    --max-concurrency 512
'
```

```bash
docker exec vllm-ekagra bash -lc \
  "curl -s -X POST http://localhost:8000/debug/asr_bottleneck_stats" \
  > /root/repos/vllm-ekagra/vllm/tmp/asr_bottleneck/mmstats/c512_full_mmstats_v2.json
```

Artifacts used for the final report:

- benchmark metrics: `tmp/asr_bottleneck/mmstats/c512_full_timing_stats_v2.json`
- aggregated timing snapshot: `tmp/asr_bottleneck/mmstats/c512_full_mmstats_v2.json`

### B. Short Concurrency-512 Recovery Run for Batch-Level Worker Timing

One longer fully instrumented rerun hit a transient `CUBLAS_STATUS_EXECUTION_FAILED` during logits, so instead of waiting for another hours-long saturation pass, I restarted the dev server and ran a shorter but shape-representative benchmark:

- concurrency: `512`
- prompts: `256`

This was enough to recover:

- `encoder_forward_batch_secs`
- `get_encoder_outputs_total_secs`
- `pack_encoder_outputs_secs`
- `encoder_decoder_proj_secs`
- `decoder_forward_secs`

Artifact:

- `tmp/asr_bottleneck/mmstats/c512_256_devmode_mmstats.json`

### C. Throughput Sweep

I also collected a concurrency sweep at:

- `1`
- `4`
- `16`
- `64`
- `256`
- `512`

Artifacts:

- `tmp/asr_bottleneck/sweep/c1_n256.json`
- `tmp/asr_bottleneck/sweep/c4_n256.json`
- `tmp/asr_bottleneck/sweep/c16_n256.json`
- `tmp/asr_bottleneck/sweep/c64_n256.json`
- `tmp/asr_bottleneck/sweep/c256_n256.json`
- `tmp/asr_bottleneck/sweep/c512_n256.json`

### D. Bounded `torch` Profiler Runs

I used three bounded profiler runs.

#### 1. Frontend-inclusive profiler run

Why:

- capture AsyncLLM frontend CPU trace and a worker trace with minimal runtime

Server:

```bash
docker exec -e FLASHINFER_DISABLE_VERSION_CHECK=1 vllm-ekagra bash -lc '
  cd /host/vllm-ekagra/vllm &&
  CFG=$(python3 -c '"'"'import json; print(json.dumps({
    "profiler": "torch",
    "torch_profiler_dir": "/host/vllm-ekagra/vllm/tmp/vllm_profiles/front8",
    "wait_iterations": 1,
    "warmup_iterations": 1,
    "active_iterations": 4
  }))'"'"') &&
  vllm serve /host/engines/vllm/audio/2b-release --trust-remote-code --profiler-config "$CFG"
'
```

Driver:

```bash
docker exec -e FLASHINFER_DISABLE_VERSION_CHECK=1 vllm-ekagra bash -lc '
  cd /host/vllm-ekagra/vllm &&
  python3 nick-bench.py \
    --base-url http://localhost:8000/v1 \
    --api-key EMPTY \
    --max-concurrent 1 \
    --target-duration 8.0 \
    --use-random-dataset \
    --max-samples 3 \
    --model /host/engines/vllm/audio/2b-release \
    --profile
'
```

#### 2. Worker-only `8s` profile

Why:

- isolate the worker kernel mix around a median-ish clip duration without frontend profiler overhead

Server config difference:

- `ignore_frontend: true`

Driver:

```bash
docker exec -e FLASHINFER_DISABLE_VERSION_CHECK=1 vllm-ekagra bash -lc '
  cd /host/vllm-ekagra/vllm &&
  python3 nick-bench.py \
    --base-url http://localhost:8000/v1 \
    --api-key EMPTY \
    --max-concurrent 16 \
    --target-duration 8.0 \
    --use-random-dataset \
    --max-samples 32 \
    --model /host/engines/vllm/audio/2b-release \
    --profile
'
```

#### 3. Worker-only `20s` profile

Why:

- see how the kernel mix shifts on longer audio where encoder work should matter more

Driver:

```bash
docker exec -e FLASHINFER_DISABLE_VERSION_CHECK=1 vllm-ekagra bash -lc '
  cd /host/vllm-ekagra/vllm &&
  python3 nick-bench.py \
    --base-url http://localhost:8000/v1 \
    --api-key EMPTY \
    --max-concurrent 8 \
    --target-duration 20.0 \
    --use-random-dataset \
    --max-samples 16 \
    --model /host/engines/vllm/audio/2b-release \
    --profile
'
```

Profiler artifacts:

- frontend-inclusive traces: `tmp/vllm_profiles/front8/`
- worker-only `8s` traces: `tmp/vllm_profiles/worker8/`
- worker-only `20s` traces: `tmp/vllm_profiles/worker20/`
- profiler driver logs:
  - `tmp/asr_bottleneck/front8_nickbench.txt`
  - `tmp/asr_bottleneck/worker8_nickbench.txt`
  - `tmp/asr_bottleneck/worker20_nickbench.txt`

## What the `torch` Profiler Gives

Each profiler run produces two useful outputs.

### 1. Timeline traces

Files like:

- `rank0.*.pt.trace.json.gz`
- `*.async_llm.*.pt.trace.json.gz`

These traces can be opened in TensorBoard's profiler viewer or other Chrome-trace-compatible tools to inspect:

- CPU timeline
- CUDA timeline
- overlap
- iteration boundaries
- operation ordering

### 2. Key-averages summary

File:

- `profiler_out_0.txt`

This is the summary table printed from `torch.profiler.key_averages()` and sorted by `self_cuda_time_total` on the worker side.

This table is useful for ranking kernels and ops such as:

- `vllm::cohere_asr_triton_relpos_attention`
- `aten::addmm`
- `aten::cudnn_convolution`

One nuance:

- rows like `execute_context_*` are iteration/context annotations, not actual kernels to optimize directly

## Benchmark Results

### Concurrency Sweep

| Concurrency | Request throughput (req/s) | RTFx | Mean TTFT (ms) | P99 TTFT (ms) |
| --- | ---: | ---: | ---: | ---: |
| 1 | 11.70 | 107.21 | 54.80 | 66.07 |
| 4 | 22.75 | 208.48 | 88.94 | 119.41 |
| 16 | 59.82 | 548.31 | 142.83 | 216.66 |
| 64 | 99.34 | 910.59 | 485.41 | 615.39 |
| 256 | 105.81 | 969.83 | 1542.42 | 2007.22 |
| 512 | 109.08 | 999.81 | 1499.88 | 1975.91 |

Interpretation:

- Throughput scales well up to around `64`.
- From `64` to `512`, throughput improves only modestly.
- TTFT rises sharply after `64`, which is consistent with frontend queueing/backpressure saturation rather than scheduler CPU or IPC saturation.

### Full Saturation Benchmark

From `tmp/asr_bottleneck/results/devmode_refined/openai-audio-infqps-concurrency512-2b-release-20260318-022212.json`:

| Metric | Value |
| --- | ---: |
| duration | `15.79s` |
| completed requests | `1842` |
| failed requests | `0` |
| request throughput | `116.64 req/s` |
| output throughput | `3742.99 tok/s` |
| total token throughput | `3976.27 tok/s` |
| RTFx | `1123.84` |
| mean TTFT | `2425.32ms` |
| P99 TTFT | `4086.32ms` |

## Top-Level Timing Breakdown

The refined table below uses the aggregated timing snapshot from `tmp/asr_bottleneck/mmstats/c512_full_mmstats_v3.json`.

To avoid double-counting the old broad wrapper, this table replaces `renderer_render_secs` with the renderer sub-buckets from `frontend.renderer.*`.

These percentages are shares of the measured top-level accumulated stage-time after that de-duplication.

| Stage | Total secs | Count | Avg / count | Share of measured top-level time | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| `tokenize_prompts_secs` | `1514.398` | `1843` | `0.821703s` | `98.6848%` | Async tokenizer work inside `render_cmpl_async()` |
| `encoder_forward_secs` | `10.422` | `1849` | `0.005637s` | `0.6792%` | Existing synchronized per-request encoder timing |
| `process_for_engine_secs` | `6.494` | `1843` | `0.003524s` | `0.4232%` | Renderer post-tokenization processing |
| `audio_decode_resample_secs` | `1.482` | `1843` | `0.000804s` | `0.0966%` | Frontend decode/resample |
| `decoder_forward_secs` | `0.662` | `122` | `0.005430s` | `0.0432%` | Device-complete timer for uncaptured decoder batches |
| `schedule_total_secs` | `0.535` | `841` | `0.000637s` | `0.0349%` | Scheduler total CPU time |
| `input_socket_decode_add_secs` | `0.149` | `1849` | `0.000080s` | `0.0097%` | Engine socket decode |
| `logits_forward_secs` | `0.124` | `835` | `0.000148s` | `0.0081%` | Device-complete logits timing outside the graph replay |
| `request_encode_secs` | `0.116` | `1849` | `0.000063s` | `0.0075%` | Client serialization |
| `request_send_secs` | `0.075` | `1849` | `0.000041s` | `0.0049%` | Client ZMQ send |
| `encoder_decoder_proj_secs` | `0.058` | `122` | `0.000473s` | `0.0038%` | Device-complete projection timing |
| `pack_encoder_outputs_secs` | `0.032` | `122` | `0.000263s` | `0.0021%` | Python pad-to-pack loop after GPU timing correction |
| `preprocess_add_request_secs` | `0.013` | `1849` | `0.000007s` | `0.0008%` | Engine request preprocessing |
| `prompt_build_secs` | `0.009` | `1843` | `0.000005s` | `0.0006%` | Prompt assembly in speech-to-text path |
| `concat_encoder_outputs_secs` | `0.005` | `122` | `0.000038s` | `0.0003%` | Device-complete concat timing |
| `render_prompts_secs` | `0.003` | `1843` | `0.000002s` | `0.0002%` | Async prompt rendering |
| `audio_chunking_secs` | `0.002` | `5` | `0.000403s` | `0.0001%` | Speech chunking for long clips |
| `apply_prompt_extras_secs` | `0.001` | `1843` | `0.000001s` | `0.0001%` | Prompt extras application |

### What This Means

The refinement changes the interpretation from “somewhere in the renderer wrapper” to a much narrower conclusion:

- the bottleneck is specifically `tokenize_prompts_secs`
- `tokenize_prompts_secs` is `1514.398s`, which is `99.57%` of `render_cmpl_total_secs`
- worker encoder time is still the largest worker cost, but it is far smaller than tokenizer time in the saturated serve path
- the old large `pack_encoder_outputs_secs` reading was mostly a timing artifact from host-side measurement around GPU work

The practical conclusion is now stronger than in the first pass:

- the first optimization target should be the tokenizer path and the queueing around `AsyncMicrobatchTokenizer`

## Frontend Preprocessing Breakdown

The multimodal/audio preprocessing remains small overall, but the split now shows exactly where it sits:

- tokenization dominates the renderer path
- within `process_for_engine_secs`, the multimodal preprocessor is the main cost

| Frontend sub-bucket | Total secs | Count | Avg / count | Comment |
| --- | ---: | ---: | ---: | --- |
| `preprocessor_total_secs` | `6.117` | `1849` | `0.003308s` | Total multimodal processor path |
| `apply_hf_processor_secs` | `6.018` | `1849` | `0.003255s` | Most of preprocessor total |
| `filterbank_forward_secs` | `5.298` | `1849` | `0.002865s` | Actual filterbank/STFT/mel work |
| `feature_batch_prep_secs` | `0.271` | `1849` | `0.000147s` | Padding and batch shaping |
| `feature_tensorize_secs` | `0.014` | `1849` | `0.000007s` | Tensor creation/move |
| `feature_to_cpu_secs` | `0.017` | `1849` | `0.000009s` | CPU return path |
| `feature_return_tensors_secs` | `0.017` | `1849` | `0.000009s` | Tensor conversion |
| `audio_decode_resample_secs` | `1.482` | `1843` | `0.000804s` | Separate frontend decode/resample step |

Three important ratios from the split run:

- `tokenize_prompts_secs / render_cmpl_total_secs = 99.5712%`
- `preprocessor_total_secs / process_for_engine_secs = 94.1933%`
- `filterbank_forward_secs / process_for_engine_secs = 81.5768%`

So the frontend preprocessor is real work, but it is a distant second-order cost behind tokenization.

## IPC and Scheduler Breakdown

The IPC/request marshaling path is also very small.

Combined totals:

- client encode: `0.116s`
- client send: `0.075s`
- engine decode add: `0.149s`
- engine preprocess add: `0.013s`

Combined IPC/request-add path total:

- `0.353s`

Share of measured top-level time:

- about `0.023%`

Scheduler total:

- `0.535s`

Share of measured top-level time:

- about `0.035%`

Conclusion:

- IPC and scheduler overhead are currently too small to be the first optimization target.

## Worker / Model Breakdown

### Full Run Worker Breakdown

Within the measured worker model buckets from the refined full run:

| Worker bucket | Total secs | Count | Avg / count | Share of measured worker model time |
| --- | ---: | ---: | ---: | ---: |
| `encoder_forward_batch_secs` | `10.299` | `122` | `0.084415s` | `92.12%` |
| `decoder_forward_secs` | `0.662` | `122` | `0.005430s` | `5.93%` |
| `logits_forward_secs` | `0.124` | `835` | `0.000148s` | `1.11%` |
| `encoder_decoder_proj_secs` | `0.058` | `122` | `0.000473s` | `0.52%` |
| `pack_encoder_outputs_secs` | `0.032` | `122` | `0.000263s` | `0.29%` |
| `concat_encoder_outputs_secs` | `0.005` | `122` | `0.000038s` | `0.04%` |

Interpretation:

- On the worker, encoder work is still the main model cost.
- After the GPU-timing correction, `pack_encoder_outputs_secs` is no longer a major worker bottleneck.
- The previous `2.380s` pack result dropped to `0.032s` after the device-complete timers were added, so the first-pass pack result should be treated as a host-side measurement artifact rather than true device work.
- `encoder_forward_secs` from the existing per-request hook (`10.422s`) closely matches the batch-level device-timed encoder total (`10.299s`), which is a good sanity check on the corrected methodology.

## `torch` Profiler Findings

### Why the bounded runs were useful

The stage timers tell us:

- where accumulated request-time is being spent

The `torch` profiler tells us:

- which concrete kernels and PyTorch ops dominate within the worker compute path

This is why I used both.

### Worker `8s` profile highlights

From `tmp/vllm_profiles/worker8/profiler_out_0.txt`:

- `aten::addmm`: `22.95%` self CUDA
- `vllm::cohere_asr_triton_relpos_attention`: `16.80%` self CUDA
- `aten::cudnn_convolution`: `10.70%` self CUDA
- additional BF16 GEMM kernels account for another meaningful chunk

Interpretation:

- even at moderate clip lengths, encoder-side GEMMs and relative-position attention are the main worker CUDA consumers

### Worker `20s` profile highlights

From `tmp/vllm_profiles/worker20/profiler_out_0.txt`:

- `vllm::cohere_asr_triton_relpos_attention`: `29.11%` self CUDA
- `aten::addmm`: `19.78%` self CUDA
- `aten::cudnn_convolution`: `8.35%` self CUDA
- `aten::copy_`: `8.32%` self CUDA

Interpretation:

- as clip length grows, relative-position attention becomes even more dominant
- the worker-side optimization target after the frontend is:
- encoder attention / GEMM efficiency
- then the smaller decoder / logits work if frontend tokenization is already addressed

### Frontend-inclusive profile

The frontend-inclusive run created:

- an AsyncLLM CPU trace
- a worker trace

This is useful for opening the timeline and checking:

- whether the frontend and worker are overlapping as expected
- where request lifetime is spent around profiling boundaries

The stage-timer evidence now shows that `tokenize_prompts_secs` is the dominant top-level bucket, so the frontend-inclusive trace is mainly supporting evidence for how much of the request lifetime is spent waiting around the tokenizer path.

## Final Bottleneck Ranking

Based on the combined benchmark metrics, refined stage timers, and bounded profiler traces, the current bottleneck ranking is:

1. **Frontend async tokenization (`tokenize_prompts_secs`)**  
   Evidence: `1514.398s`, `98.6848%` of de-duplicated measured top-level time, and `99.57%` of `render_cmpl_total_secs`.

2. **Worker encoder forward path**  
   Evidence: `encoder_forward_secs` / `encoder_forward_batch_secs` are the largest worker-side buckets, and the `torch` profiler still points to encoder attention and GEMMs as the dominant CUDA kernels.

3. **Renderer `process_for_engine(...)` / multimodal preprocessing**  
   Evidence: `process_for_engine_secs` is the second renderer sub-bucket, and `preprocessor_total_secs` is `94.19%` of it.

4. **Audio decode / resample**  
   Evidence: measurable, but only `1.482s` total across the full run.

5. **Worker decoder forward and logits**  
   Evidence: both are present after the GPU-timing correction, but they are much smaller than encoder work and vastly smaller than tokenization.

6. **Scheduler CPU / IPC / engine request marshaling**  
   Evidence: all are negligible in the current workload.

## Why I Concluded the Frontend is the First Optimization Target

The key logic was:

1. Throughput stops scaling strongly after about concurrency `64`.
2. TTFT rises dramatically after that point.
3. The split renderer timers now show that the dominant sub-bucket is specifically `tokenize_prompts_secs`, not prompt rendering or prompt-extra handling.
4. `process_for_engine_secs` and the multimodal preprocessor are real, but still tiny compared with tokenization.
5. The corrected worker timings show encoder work is the main model cost, while pad-to-pack is not a major limiter after the measurement fix.

So the next optimization should be:

- optimize the tokenizer path first, especially the queueing and single-thread execution in `AsyncMicrobatchTokenizer`
- check whether a fast tokenizer, a wider tokenizer executor, or a different batching strategy reduces `tokenize_prompts_secs`

After that, move to:

- encoder attention/GEMM work
- then the smaller frontend preprocessing costs
- only then revisit decoder / logits micro-optimizations

## Caveats and Limitations

1. `renderer_render_secs` still exists as a broad compatibility wrapper, but root-cause analysis should use the split `frontend.renderer.*` buckets from the refined run.
2. Some timers are per-request and others are per-batch, so `avg / count` values are not always in the same unit.
3. The device-complete worker timers intentionally skip synchronization during CUDA graph capture, so `decoder_forward_secs` should be read as the uncaptured decoder-stage timing; `logits_forward_secs` remains useful because it is timed outside the graph replay.
4. The bounded `nick-bench.py` profiler runs used random audio because the goal was path/kernel profiling, not transcript quality evaluation.

## Output Files

The main artifacts for this report are:

- `tmp/asr_bottleneck/mmstats/c512_full_timing_stats_v2.json`
- `tmp/asr_bottleneck/mmstats/c512_full_mmstats_v2.json`
- `tmp/asr_bottleneck/mmstats/c512_full_mmstats_v3.json`
- `tmp/asr_bottleneck/results/devmode_refined/openai-audio-infqps-concurrency512-2b-release-20260318-022212.json`
- `tmp/asr_bottleneck/sweep/`
- `tmp/vllm_profiles/front8/`
- `tmp/vllm_profiles/worker8/`
- `tmp/vllm_profiles/worker20/`

This report itself is saved at:

- `tmp/asr_bottleneck/profiling.md`

