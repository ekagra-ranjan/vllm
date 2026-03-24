# Cross-Request Audio Microbatching For Cohere ASR

## Goal

The goal of this work was to improve Cohere ASR end-to-end throughput, measured
primarily by RTFx, by batching HF audio preprocessing across concurrent
requests.

The starting hypothesis was:

1. The HF audio preprocessor is a meaningful CPU-side cost.
2. vLLM's default flow preprocesses audio per request, which loses parallelism.
3. A cross-request async microbatcher should improve throughput by processing
   multiple clips together.

The non-negotiable constraint was correctness: the model must still receive the
right audio features and produce sane transcripts.

## Files Changed

The implementation work touched these files:

- `vllm/entrypoints/openai/speech_to_text/speech_to_text.py`
- `vllm/model_executor/models/cohere_asr.py`
- `vllm/renderers/base.py`

At a high level:

- `speech_to_text.py`
  - Added the `AsyncMicrobatchAudioPreprocessor`.
  - Added `VLLM_COHERE_ASR_CROSS_REQUEST_AUDIO_MICROBATCH` to toggle the async
    microbatch path on and off.
  - Moved thread-setting into the actual preprocess execution path instead of
    the per-request enqueue path.
  - Added opt-in low-overhead instrumentation via
    `VLLM_COHERE_ASR_CROSS_REQUEST_AUDIO_MICROBATCH_STATS=1`.

- `cohere_asr.py`
  - Added a custom preprocessed-audio modality path so already-computed
    `input_features` and `length` can flow through the multimodal machinery
    without re-running the HF processor.
  - Added a persistent worker-local audio preprocessor factory so the async
    worker can reuse the feature extractor rather than reinitializing it for
    every batch.

- `renderers/base.py`
  - Re-enabled the multimodal renderer path so prompts carry audio features all
    the way through to the model.

## Why Each Change Was Needed

### 1. Re-enable multimodal processing in the renderer

Why:

- When the multimodal path in `renderers/base.py` was commented out, the prompt
  no longer carried the audio features through the normal renderer flow.
- The model then effectively decoded without real audio input, which produced
  nonsense outputs such as `"I'm sorry, but I'm sorry."`.

What changed:

- The multimodal path in the renderer was restored.

Outcome:

- Correctness was restored, but this alone was not enough because the normal
  renderer path would still try to re-run HF preprocessing unless we taught it
  how to consume already-preprocessed audio.

### 2. Add a custom preprocessed-audio modality path

Why:

- Passing a raw dict like `{"input_features": ..., "length": ...}` into the
  default multimodal parser did not work because the default path expects raw
  audio-like inputs.
- Reusing `DictEmbeddingItems` was not correct either because it triggered
  multimodal embedding validation that was not appropriate for Cohere ASR
  preprocessed audio features.

What changed:

- Added `CohereASRPreprocessedAudioItems`.
- Added a custom `CohereASRMultiModalDataParser`.

Outcome:

- The model could consume already-preprocessed audio features correctly.
- This avoided re-running the HF processor in the normal multimodal flow.

### 3. Reuse a persistent feature extractor inside the async worker

Why:

- The first attempt at async preprocessing repeatedly paid expensive extractor
  setup costs inside the worker path.
- That meant the async path was doing extra work even before comparing any
  batching benefit.

What changed:

- Added `create_audio_microbatch_preprocessor()` and a persistent
  `_PersistentCohereASRAudioBatchPreprocessor`.

Outcome:

- This fixed the cold-start behavior inside the async worker.
- It removed one obvious source of regression.
- Even after that fix, end-to-end throughput still did not beat the old path.

### 4. Move thread-setting to the execution path

Why:

- Setting `HF_PROCESSOR_NUM_THREADS` around the per-request async path would be
  the wrong place conceptually because the real CPU work is done in the batch
  worker.
- The async path should set thread count where preprocessing actually runs:
  once per real preprocess invocation, not once per request enqueue.

What changed:

- The thread-setting context was placed inside the actual async worker batch
  execution path.
- The direct same-request multi-chunk path still sets threads around its own
  direct batch preprocess call.

Outcome:

- This gave a cleaner comparison of thread-count effects and avoided misleading
  per-request thread management.

## Environment And Setup

All commands below assume an interactive shell inside the Docker container:

```bash
docker exec -it vllm-ekagra bash
export FLASHINFER_DISABLE_VERSION_CHECK=1
cd /host/vllm-ekagra/vllm
MODEL_ID=/host/engines/vllm/audio/2b-release
```

Useful environment variables:

- `FLASHINFER_DISABLE_VERSION_CHECK=1`
- `MODEL_ID=/host/engines/vllm/audio/2b-release`
- `VLLM_MAX_AUDIO_CLIP_FILESIZE_MB=50`
- `HF_PROCESSOR_NUM_THREADS=<n>`
- `CROSS_REQUEST_AUDIO_MICROBATCH_MAX_BATCH_SIZE=<n>`
- `VLLM_COHERE_ASR_CROSS_REQUEST_AUDIO_MICROBATCH=0|1`
- `VLLM_COHERE_ASR_CROSS_REQUEST_AUDIO_MICROBATCH_STATS=0|1`
- `CROSS_REQUEST_AUDIO_MICROBATCH_STATS_LOG_EVERY_BATCHES=<n>`

Note:

- `VLLM_COHERE_ASR_CROSS_REQUEST_AUDIO_MICROBATCH` and
  `VLLM_COHERE_ASR_CROSS_REQUEST_AUDIO_MICROBATCH_STATS` are read directly by
  the Python code. vLLM prints warnings that they are "unknown environment
  variables", but those warnings are expected and harmless.

## Sanity Check: Correctness

This is the simplest correctness check that should confirm the model is not
decoding without audio features.

### Start the server

```bash
export FLASHINFER_DISABLE_VERSION_CHECK=1
cd /host/vllm-ekagra/vllm
MODEL_ID=/host/engines/vllm/audio/2b-release

VLLM_MAX_AUDIO_CLIP_FILESIZE_MB=50 \
VLLM_COHERE_ASR_CROSS_REQUEST_AUDIO_MICROBATCH=1 \
vllm serve "$MODEL_ID" --trust-remote-code
```

### Send `mary_had_lamb.ogg`

```bash
MODEL_ID=/host/engines/vllm/audio/2b-release
AUDIO_PATH=~/.cache/vllm/assets/vllm_public_assets/mary_had_lamb.ogg

curl -s http://localhost:8000/v1/audio/transcriptions \
  -H "Authorization: Bearer ${VLLM_API_KEY:-EMPTY}" \
  -F "file=@${AUDIO_PATH}" \
  -F "model=${MODEL_ID}"
```

Expected behavior:

- The output should be a normal transcript related to "Mary had a little lamb".
- If the output degenerates into text like `"I'm sorry, but I'm sorry."`, the
  audio features are likely not making it into the model correctly.

Why this check was done:

- The very first failure mode was not performance-related; it was correctness.
- Before benchmarking anything, we needed to confirm that preprocessed audio was
  actually reaching the model.

## Experiment Log

## 1. VoxPopuli Sweep: HF Threads x Async Microbatch Size

Why:

- Once correctness was restored, the next question was whether cross-request
  batching improved end-to-end throughput on a realistic dataset.
- VoxPopuli gave a representative heterogeneous real workload.

Script:

- `bench_asr_hf_threads_microbatch_sweep.sh`

What the script does:

- Iterates over `HF_PROCESSOR_NUM_THREADS` and
  `CROSS_REQUEST_AUDIO_MICROBATCH_MAX_BATCH_SIZE`.
- Restarts the server for every combination.
- Runs `vllm bench serve` against `facebook/voxpopuli`.
- Writes per-run JSON and a summary table.

Important note:

- In this sweep, `MB=1` means "async microbatcher enabled with max batch size
  1".
- That is **not** the same as the old "async disabled" baseline.

### Reproduce the sweep

```bash
export FLASHINFER_DISABLE_VERSION_CHECK=1
cd /host/vllm-ekagra/vllm
MODEL_ID=/host/engines/vllm/audio/2b-release

time MAX_CONCURRENCY=512 MODEL_ID="$MODEL_ID" \
  ./bench_asr_hf_threads_microbatch_sweep.sh
```

Optional, to save the whole terminal output:

```bash
time MAX_CONCURRENCY=512 MODEL_ID="$MODEL_ID" \
  ./bench_asr_hf_threads_microbatch_sweep.sh &> thread.txt
```

Artifacts:

- `asr_bench_results/hf_threads_microbatch_sweep/summary.txt`
- `asr_bench_results/hf_threads_microbatch_sweep/*.json`
- `asr_bench_results/hf_threads_microbatch_sweep/*.log`

### Key results from the sweep

Best async configuration in the sweep:

- `HF_PROCESSOR_NUM_THREADS=16`
- `CROSS_REQUEST_AUDIO_MICROBATCH_MAX_BATCH_SIZE=64`
- `RTFx=999.39`
- `Req/s=103.72`
- `TTFT=2631.72 ms`

Representative bad configuration:

- `HF_PROCESSOR_NUM_THREADS=2`
- `CROSS_REQUEST_AUDIO_MICROBATCH_MAX_BATCH_SIZE=2`
- `RTFx=557.62`
- `Req/s=57.87`
- `TTFT=6739.72 ms`

Observed pattern:

- `MB=2` was consistently poor.
- Larger `MB` values sometimes helped relative to smaller async values.
- But the async sweep still did not clearly beat the earlier non-async baseline.

Prior non-async reference points from earlier runs:

- About `RTFx=1053.36` at `HF_PROCESSOR_NUM_THREADS=1`
- About `RTFx=1090.19` at `HF_PROCESSOR_NUM_THREADS=8`

One in-repo non-async baseline artifact with `RTFx=1090.19` exists at:

- `openai-audio-infqps-concurrency512-2b-release-20260319-210228.json`

Conclusion from this experiment:

- Cross-request async microbatching did not obviously improve throughput on the
  heterogeneous VoxPopuli workload.
- At that point there were two main possibilities:
  1. Padding waste from variable-length clips was wiping out the benefit.
  2. The async microbatch architecture itself was not giving enough compute
     amortization.

## 2. Homogeneous Fixed-Duration Benchmark

Why:

- We needed to separate "batching is a bad idea" from "heterogeneous clip
  lengths are a bad workload for batching".
- `nick-bench.py` can generate fixed-duration random audio, which removes clip
  length variability.

Important note:

- Random audio is not for transcript quality; it is for load generation.
- Use `mary_had_lamb.ogg` for correctness checks and `nick-bench.py` for
  controlled throughput measurements.

### Async enabled run

Start the server:

```bash
export FLASHINFER_DISABLE_VERSION_CHECK=1
cd /host/vllm-ekagra/vllm

HF_PROCESSOR_NUM_THREADS=16 \
CROSS_REQUEST_AUDIO_MICROBATCH_MAX_BATCH_SIZE=64 \
VLLM_COHERE_ASR_CROSS_REQUEST_AUDIO_MICROBATCH=1 \
vllm serve /host/engines/vllm/audio/2b-release --trust-remote-code
```

In another shell in the same container:

```bash
export FLASHINFER_DISABLE_VERSION_CHECK=1
cd /host/vllm-ekagra/vllm

python3 nick-bench.py \
  --base-url http://localhost:8000/v1 \
  --api-key EMPTY \
  --max-concurrent 512 \
  --target-duration 8.0 \
  --use-random-dataset \
  --max-samples 4096 \
  --model /host/engines/vllm/audio/2b-release
```

Measured result:

- `Throughput nick: 139.7556 req/s`
- `Average latency: 3.4301 s`
- `P95 latency: 8.9563 s`

Artifact:

- `tmp/asr_bottleneck/results/async_microbatch_stats_nickbench.txt`

### Async disabled control

Start the server:

```bash
export FLASHINFER_DISABLE_VERSION_CHECK=1
cd /host/vllm-ekagra/vllm

HF_PROCESSOR_NUM_THREADS=16 \
VLLM_COHERE_ASR_CROSS_REQUEST_AUDIO_MICROBATCH=0 \
vllm serve /host/engines/vllm/audio/2b-release --trust-remote-code
```

Run the same load:

```bash
export FLASHINFER_DISABLE_VERSION_CHECK=1
cd /host/vllm-ekagra/vllm

python3 nick-bench.py \
  --base-url http://localhost:8000/v1 \
  --api-key EMPTY \
  --max-concurrent 512 \
  --target-duration 8.0 \
  --use-random-dataset \
  --max-samples 4096 \
  --model /host/engines/vllm/audio/2b-release
```

Measured result:

- `Throughput nick: 153.8733 req/s`
- `Average latency: 3.1015 s`
- `P95 latency: 6.2606 s`

Artifact:

- `tmp/asr_bottleneck/results/no_async_nickbench.txt`

Conclusion from this experiment:

- Async microbatching was about `9.2%` slower than the no-async control even on
  a homogeneous fixed-duration workload.
- That means padding variability is **not** the main reason the async path
  underperformed.

## 3. Low-Overhead Instrumentation

Why:

- After the homogeneous benchmark still regressed, the remaining question was:
  are we mostly forming tiny batches, or are even decent-sized batches not
  helping enough?

Implementation notes:

- Instrumentation is opt-in.
- Disabled path is effectively unchanged.
- Enabled path uses fixed-size arrays plus a few running sums and maxes.
- It records:
  - batch-size distribution
  - queue wait time
  - per-batch preprocess time

### Reproduce the instrumented run

Start the server:

```bash
export FLASHINFER_DISABLE_VERSION_CHECK=1
cd /host/vllm-ekagra/vllm

HF_PROCESSOR_NUM_THREADS=16 \
CROSS_REQUEST_AUDIO_MICROBATCH_MAX_BATCH_SIZE=64 \
VLLM_COHERE_ASR_CROSS_REQUEST_AUDIO_MICROBATCH=1 \
VLLM_COHERE_ASR_CROSS_REQUEST_AUDIO_MICROBATCH_STATS=1 \
CROSS_REQUEST_AUDIO_MICROBATCH_STATS_LOG_EVERY_BATCHES=20 \
vllm serve /host/engines/vllm/audio/2b-release --trust-remote-code \
  > tmp/asr_bottleneck/results/async_microbatch_stats_server.log 2>&1
```

Run the same homogeneous load:

```bash
export FLASHINFER_DISABLE_VERSION_CHECK=1
cd /host/vllm-ekagra/vllm

python3 nick-bench.py \
  --base-url http://localhost:8000/v1 \
  --api-key EMPTY \
  --max-concurrent 512 \
  --target-duration 8.0 \
  --use-random-dataset \
  --max-samples 4096 \
  --model /host/engines/vllm/audio/2b-release \
  > tmp/asr_bottleneck/results/async_microbatch_stats_nickbench.txt 2>&1
```

Inspect the stats:

```bash
rg "Async audio microbatch stats|Final async audio microbatch stats" \
  tmp/asr_bottleneck/results/async_microbatch_stats_server.log
```

Artifacts:

- `tmp/asr_bottleneck/results/async_microbatch_stats_server.log`
- `tmp/asr_bottleneck/results/async_microbatch_stats_nickbench.txt`

### Final instrumented summary

Final line from the run:

```text
Final async audio microbatch stats: batches=951 items=4608 avg_batch_size=4.85 avg_qwait_ms=172.88 avg_pre_ms=22.28 max_qwait_ms=1339.00 max_pre_ms=582.31
```

Key distribution facts:

- About `5.9%` of items were processed as singleton batches.
- About `69.1%` of items were processed in batch sizes `4-7`.
- About `19.4%` of items were processed in size `64` batches.

Per-batch preprocess times:

- size `1`: `4.01 ms` per batch, about `4.01 ms/item`
- size `4`: `25.72 ms` per batch, about `6.43 ms/item`
- size `5`: `26.60 ms` per batch, about `5.32 ms/item`
- size `6`: `27.57 ms` per batch, about `4.60 ms/item`
- size `7`: `28.19 ms` per batch, about `4.03 ms/item`
- size `64`: `123.59 ms` per batch, about `1.93 ms/item`

Queueing facts:

- Overall average queue wait was `172.88 ms/item`.
- Size `64` batches had `721.30 ms` average queue wait.
- Common steady-state batches of size `4-7` still had around `19-22 ms`
  average queue wait.

Important caveat:

- The current counters include warmup requests because the stats are not reset
  after warmup.
- The qualitative conclusion still holds, but measured-phase-only stats would be
  even cleaner.

Conclusion from this experiment:

- The problem is **not** simply "the batcher only makes singleton batches".
- The batcher does form batches.
- The problem is that the common batches are only medium-sized, and those
  medium-sized batches do not reduce per-item preprocess cost enough to justify
  the queueing they introduce.

## 4. Multi-Worker CPU Microbatching

Why:

- The single-worker design showed that batching helps preprocessing only when
  batches get large, but one global worker imposed substantial head-of-line
  blocking.
- The next hypothesis was that multiple workers, each with its own persistent
  preprocessor, could reduce queueing enough to improve end-to-end throughput.

Implementation summary:

- The async preprocessor was changed from one global worker to a small worker
  pool with:
  - one shared request queue
  - one single-thread executor per worker
  - one worker-local persistent Cohere ASR preprocessor per worker
- A total Torch-thread budget cap of `64` was added for the async pool.

New runtime knobs:

- `CROSS_REQUEST_AUDIO_MICROBATCH_NUM_WORKERS`
- `CROSS_REQUEST_AUDIO_MICROBATCH_MAX_TOTAL_TORCH_THREADS`

Thread-budget policy:

- `threads_per_worker = min(HF_PROCESSOR_NUM_THREADS, 64 // worker_count)`
- So with `HF_PROCESSOR_NUM_THREADS=16` and `worker_count=8`, the effective
  async preprocessing thread count becomes `8` per worker, not `16`.

### Reproduce the focused CPU worker sweep

All runs below used:

- `HF_PROCESSOR_NUM_THREADS=16`
- `CROSS_REQUEST_AUDIO_MICROBATCH_MAX_BATCH_SIZE=64`
- `CROSS_REQUEST_AUDIO_MICROBATCH_MAX_TOTAL_TORCH_THREADS=64`
- `VLLM_COHERE_ASR_CROSS_REQUEST_AUDIO_MICROBATCH=1`
- `nick-bench.py` with:
  - `--max-concurrent 512`
  - `--target-duration 8.0`
  - `--use-random-dataset`
  - `--max-samples 4096`

Example server launch:

```bash
export FLASHINFER_DISABLE_VERSION_CHECK=1
cd /host/vllm-ekagra/vllm

HF_PROCESSOR_NUM_THREADS=16 \
CROSS_REQUEST_AUDIO_MICROBATCH_MAX_BATCH_SIZE=64 \
CROSS_REQUEST_AUDIO_MICROBATCH_NUM_WORKERS=8 \
CROSS_REQUEST_AUDIO_MICROBATCH_MAX_TOTAL_TORCH_THREADS=64 \
VLLM_COHERE_ASR_CROSS_REQUEST_AUDIO_MICROBATCH=1 \
vllm serve /host/engines/vllm/audio/2b-release --trust-remote-code
```

Example homogeneous benchmark:

```bash
python3 nick-bench.py \
  --base-url http://localhost:8000/v1 \
  --api-key EMPTY \
  --max-concurrent 512 \
  --target-duration 8.0 \
  --use-random-dataset \
  --max-samples 4096 \
  --model /host/engines/vllm/audio/2b-release
```

Uniform-workload RTFx is derived as:

```text
RTFx = (num_samples * clip_duration_s) / total_time_s
```

For these runs:

```text
RTFx = (4096 * 8.0) / total_time_s
```

### Correctness check

Using the multi-worker path with `workers=4`, the `mary_had_lamb.ogg` request
still returned a sensible transcript containing `"Mary had a little lamb..."`.

This confirmed that the worker-pool refactor did not break the preprocessed
audio handoff into the model.

### Homogeneous worker-count results (`MB=64`, `HF_T=16`)

Measured results:

- `workers=1`
  - `Throughput nick: 137.4839 req/s`
  - `RTFx=1099.87`
  - `Average latency: 3.4918 s`
  - `P95 latency: 8.5978 s`
- `workers=2`
  - `Throughput nick: 126.0780 req/s`
  - `RTFx=1008.62`
  - `Average latency: 3.8059 s`
  - `P95 latency: 8.4781 s`
- `workers=4`
  - `Throughput nick: 135.5626 req/s`
  - `RTFx=1084.50`
  - `Average latency: 3.5207 s`
  - `P95 latency: 6.4727 s`
- `workers=8`
  - `Throughput nick: 154.5303 req/s`
  - `RTFx=1236.24`
  - `Average latency: 3.0846 s`
  - `P95 latency: 6.4597 s`

Artifacts:

- `tmp/asr_bottleneck/results/multi_worker_cpu_validation/worker1_nickbench.txt`
- `tmp/asr_bottleneck/results/multi_worker_cpu_validation/worker2_nickbench.txt`
- `tmp/asr_bottleneck/results/multi_worker_cpu_validation/worker4_nickbench.txt`
- `tmp/asr_bottleneck/results/multi_worker_cpu_validation/worker8_nickbench.txt`

Conclusion from this experiment:

- Moving from one worker to multiple workers changed the qualitative result.
- `workers=8` was clearly best on the homogeneous workload.
- The worker-pool design reduced enough queueing that async microbatching
  finally beat the earlier single-worker async CPU path.

### VoxPopuli confirmation (`MB=64`, `HF_T=16`)

To confirm that the homogeneous win was not a synthetic-only artifact, a
smaller real-dataset check was run comparing `workers=1` and `workers=8`.

Results:

- `workers=1`
  - `RTFx=945.08`
  - `Req/s=98.09`
  - `Mean TTFT=2924.95 ms`
- `workers=8`
  - `RTFx=1041.08`
  - `Req/s=108.05`
  - `Mean TTFT=2448.13 ms`

Artifacts:

- `tmp/asr_bottleneck/results/multi_worker_cpu_validation/voxpopuli_worker1.json`
- `tmp/asr_bottleneck/results/multi_worker_cpu_validation/voxpopuli_worker8.json`

Conclusion from this experiment:

- The multi-worker improvement carried over to the real VoxPopuli workload.
- `workers=8` improved both throughput and TTFT over `workers=1`.

## 5. Larger Microbatch Cap On The Best Worker Topology

Why:

- After `workers=8` emerged as the best CPU topology, the next question was
  whether increasing the maximum microbatch cap beyond `64` would help further.

### Uniform `MB=512` follow-up runs

These runs used the same homogeneous workload as above, but with
`CROSS_REQUEST_AUDIO_MICROBATCH_MAX_BATCH_SIZE=512`.

Reproduction example:

```bash
export FLASHINFER_DISABLE_VERSION_CHECK=1
cd /host/vllm-ekagra/vllm

HF_PROCESSOR_NUM_THREADS=16 \
CROSS_REQUEST_AUDIO_MICROBATCH_MAX_BATCH_SIZE=512 \
CROSS_REQUEST_AUDIO_MICROBATCH_NUM_WORKERS=8 \
CROSS_REQUEST_AUDIO_MICROBATCH_MAX_TOTAL_TORCH_THREADS=64 \
VLLM_COHERE_ASR_CROSS_REQUEST_AUDIO_MICROBATCH=1 \
vllm serve /host/engines/vllm/audio/2b-release --trust-remote-code
```

Measured results:

- `worker=1, MB=512, HF_T=16`
  - `Throughput nick: 133.5896 req/s`
  - `RTFx=1068.72`
- `worker=4, MB=512, HF_T=16`
  - `Throughput nick: 140.9461 req/s`
  - `RTFx=1127.57`
- `worker=4, MB=512, HF_T=4`
  - `Throughput nick: 136.7802 req/s`
  - `RTFx=1094.24`
- `worker=8, MB=512, HF_T=16`
  - effective async pool threads: `8 workers x 8 threads`
  - `Throughput nick: 157.3537 req/s`
  - `RTFx=1258.83`
  - `Average latency: 3.0405 s`
  - `P95 latency: 6.3987 s`

Artifacts:

- `tmp/asr_bottleneck/results/uniform_worker_mb512/w1_mb512_t16_nickbench.txt`
- `tmp/asr_bottleneck/results/uniform_worker_mb512/w4_mb512_t16_nickbench.txt`
- `tmp/asr_bottleneck/results/uniform_worker_mb512/w4_mb512_t4_nickbench.txt`
- `tmp/asr_bottleneck/results/uniform_worker_mb512/w8_mb512_t16_nickbench.txt`

### Direct `MB=64` vs `MB=512` comparison

At `workers=1`, larger `MB` did not help:

- `worker=1, HF_T=16`
  - `MB=64`: `RTFx=1099.87`
  - `MB=512`: `RTFx=1068.72`
  - delta: about `-2.8%`

At `workers=4`, larger `MB` helped moderately:

- `worker=4, HF_T=16`
  - `MB=64`: `RTFx=1084.50`
  - `MB=512`: `RTFx=1127.57`
  - delta: about `+4.0%`

At `workers=8`, larger `MB` helped slightly:

- `worker=8, HF_T=16`
  - `MB=64`: `RTFx=1236.24`
  - `MB=512`: `RTFx=1258.83`
  - delta: about `+1.8%`

Conclusion from this experiment:

- Larger `MB` is not universally better.
- It hurts the one-worker topology, helps the four-worker topology, and gives a
  small but real improvement on the best eight-worker topology.
- The main win came from fixing worker topology; increasing `MB` is a
  second-order improvement on top of that.

### Wait-timeout sweep on the best topology

Why:

- Once `workers=8, MB=512, HF_T=16` became the best uniform CPU setup, the next
  question was whether the async queue should wait a little longer to form
  better cross-request batches.
- `_CROSS_REQUEST_AUDIO_MICROBATCH_WAIT_TIMEOUT_S` was made env-configurable and
  swept while keeping everything else fixed.

Reproduction example:

```bash
export FLASHINFER_DISABLE_VERSION_CHECK=1
cd /host/vllm-ekagra/vllm

python3 bench_asr_wait_timeout_sweep.py
```

This script runs the uniform workload with:

- `CROSS_REQUEST_AUDIO_MICROBATCH_NUM_WORKERS=8`
- `CROSS_REQUEST_AUDIO_MICROBATCH_MAX_BATCH_SIZE=512`
- `HF_PROCESSOR_NUM_THREADS=16`
- `CROSS_REQUEST_AUDIO_MICROBATCH_MAX_TOTAL_TORCH_THREADS=64`
- `max-concurrent=512`
- `target-duration=8.0`
- `max-samples=4096`

Measured results:

| wait timeout (s) | RTFx | avg batch size | avg qwait (ms) | avg preprocess / item (ms) | singleton items |
| --- | ---: | ---: | ---: | ---: | ---: |
| `0.0` | `1188.81` | `1.00` | `166.33` | `19.52` | `100.00%` |
| `0.001` | `1384.91` | `1.16` | `32.62` | `14.41` | `85.68%` |
| `0.002` | `1325.78` | `1.16` | `39.46` | `14.16` | `85.31%` |
| `0.005` | `1117.01` | `1.29` | `44.91` | `14.12` | `72.63%` |
| `0.01` | `1081.68` | `1.39` | `67.67` | `13.80` | `61.31%` |
| `0.02` | `1026.32` | `1.58` | `70.34` | `14.13` | `46.20%` |

Interpretation:

- `0.0s` is too aggressive: batching collapses to pure singletons and queue
  wait becomes much worse.
- `0.001s` is the best point from this sweep. It improved homogeneous
  throughput to `RTFx=1384.91`, beating the previous `0.002s` baseline by about
  `+4.5%` and beating `0.0s` by about `+16.5%`.
- Increasing the wait timeout beyond `0.001-0.002s` does create somewhat larger
  batches, but the extra batching does not repay the additional queueing cost.
- On this topology, a very small nonzero wait is beneficial, but longer waits
  are harmful.

Artifacts:

- `bench_asr_wait_timeout_sweep.py`
- `tmp/asr_bottleneck/results/wait_timeout_sweep/summary.tsv`
- `tmp/asr_bottleneck/results/wait_timeout_sweep/summary.json`

### Open-loop bulk benchmark for backlog drain

Why:

- The earlier `nick-bench.py` runs are a closed-loop benchmark: after the
  initial wave, new requests are only injected when older requests complete.
- That is useful for steady-state throughput, but it does not match the user
  goal of "I already have 4096 audios; finish them all as fast as possible."
- To test that workload directly, a new benchmark `bulk-bench.py` was added.
  It keeps the same data generation, warmup, and summary reporting as
  `nick-bench.py`, but in the measured phase it releases the full backlog at
  once and reports end-to-end `RTFx`.

Reproduction example:

```bash
export FLASHINFER_DISABLE_VERSION_CHECK=1
cd /host/vllm-ekagra/vllm

HF_PROCESSOR_NUM_THREADS=16 \
CROSS_REQUEST_AUDIO_MICROBATCH_MAX_BATCH_SIZE=512 \
CROSS_REQUEST_AUDIO_MICROBATCH_NUM_WORKERS=8 \
CROSS_REQUEST_AUDIO_MICROBATCH_MAX_TOTAL_TORCH_THREADS=64 \
CROSS_REQUEST_AUDIO_MICROBATCH_WAIT_TIMEOUT_S=0.002 \
VLLM_COHERE_ASR_CROSS_REQUEST_AUDIO_MICROBATCH=1 \
vllm serve /host/engines/vllm/audio/2b-release --trust-remote-code
```

Then:

```bash
python3 bulk-bench.py \
  --base-url http://localhost:8000/v1 \
  --api-key EMPTY \
  --max-concurrent 512 \
  --target-duration 8.0 \
  --use-random-dataset \
  --max-samples 4096 \
  --model /host/engines/vllm/audio/2b-release
```

What `bulk-bench.py` preserves:

- same random and non-random dataset options as `nick-bench.py`
- same fixed-duration audio generation path
- same warmup phase
- same latency summary metrics
- additional explicit `RTFx` output

Bulk-bench ablation matrix:

| config | async | workers | MB | HF threads | wait timeout | RTFx | total time (s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseline_no_async_default` | `off` | - | - | default | - | `1045.78` | `31.3335` |
| `baseline_no_async_hf8` | `off` | - | - | `8` | - | `826.54` | `39.6448` |
| `async_a_w8_mb512_hf16_wait0001` | `on` | `8` | `512` | `16` | `0.001` | `869.29` | `37.6952` |
| `async_b_w8_mb64_hf16_wait0001` | `on` | `8` | `64` | `16` | `0.001` | `1007.92` | `32.5106` |
| `async_c_w4_mb512_hf16_wait0001` | `on` | `4` | `512` | `16` | `0.001` | `878.36` | `37.3059` |
| `async_d_w8_mb512_hf16_wait0` | `on` | `8` | `512` | `16` | `0.0` | `666.14` | `49.1906` |
| `async_e_w8_mb512_hf16_wait0002` | `on` | `8` | `512` | `16` | `0.002` | `1061.37` | `30.8732` |

Interpretation:

- Under this open-loop bulk-drain workload, the preferred async wait timeout
  changed: `0.002s` was best, while `0.001s` was clearly worse.
- The best async bulk configuration was:
  - `workers=8`
  - `CROSS_REQUEST_AUDIO_MICROBATCH_MAX_BATCH_SIZE=512`
  - `HF_PROCESSOR_NUM_THREADS=16`
  - `CROSS_REQUEST_AUDIO_MICROBATCH_MAX_TOTAL_TORCH_THREADS=64`
  - `CROSS_REQUEST_AUDIO_MICROBATCH_WAIT_TIMEOUT_S=0.002`
  - `RTFx=1061.37`
- That best bulk async result beat the no-async default baseline only slightly:
  about `+1.5%` in `RTFx` (`1061.37` vs `1045.78`).
- The bulk benchmark therefore changes the conclusion from the earlier
  closed-loop runs:
  - for closed-loop `nick-bench.py`, `0.001s` was best
  - for open-loop `bulk-bench.py`, `0.002s` was best
- `HF_PROCESSOR_NUM_THREADS=8` in the synchronous no-async path hurt
  performance rather than helping it.
- `wait=0.0` was especially poor even under true bulk submission, so some
  nonzero wait still matters.

Artifacts:

- `bulk-bench.py`
- `tmp/asr_bottleneck/results/bulk_bench_ablations/summary.tsv`
- `tmp/asr_bottleneck/results/bulk_bench_ablations/summary.json`

### Direct HF preprocessing cost on CPU vs GPU

Why:

- The end-to-end async microbatch results depend on both batching policy and the
  raw scaling behavior of the HF preprocessing path itself.
- To isolate that component, `CohereASRFeatureExtractor.__call__` was benchmarked
  directly on both `cpu` and `cuda` across a wide range of batch sizes using
  uniform random `8.0s` audio.
- This gives a cleaner signal for which device benefits more from batching,
  independent of server queueing and decoder effects.

Reproduction example:

```bash
export FLASHINFER_DISABLE_VERSION_CHECK=1
cd /host/vllm-ekagra/vllm

python3 bench_cohere_asr_preproc_device.py
```

This benchmark measures `CohereASRFeatureExtractor.__call__`, which currently
calls `extract_features(..., to_cpu=True)`, so the GPU numbers include the cost
of bringing the output tensors back to CPU.

Measured median cost per item:

| batch size | CPU ms / item | GPU ms / item |
| --- | ---: | ---: |
| `1` | `0.9698` | `0.6198` |
| `4` | `1.1946` | `0.2564` |
| `16` | `0.6420` | `0.1395` |
| `64` | `0.8843` | `0.2501` |
| `128` | `1.1937` | `0.3443` |
| `256` | `1.2143` | `0.3475` |
| `512` | `1.1844` | `0.3461` |
| `1024` | `1.1558` | `0.3482` |

Interpretation:

- Both devices were best at `BS=16` for this direct preprocessing path.
- GPU showed much larger batching gains than CPU:
  - CPU improved from `0.9698` ms/item at `BS=1` to `0.6420` at `BS=16`
    (about `1.5x` better).
  - GPU improved from `0.6198` ms/item at `BS=1` to `0.1395` at `BS=16`
    (about `4.4x` better).
- After `BS=16`, per-item cost rose again on both devices, so this path does
  not support a simple "bigger batches are always better" conclusion.
- Even with the output copy back to CPU, GPU retained a much larger batching
  advantage than CPU. That suggests the raw HF preprocessing path itself has
  stronger batching upside on GPU, even though earlier end-to-end GPU serving
  experiments had stability issues.

Artifacts:

- `bench_cohere_asr_preproc_device.py`
- `tmp/asr_bottleneck/results/cohere_preproc_device_bench/summary.csv`
- `tmp/asr_bottleneck/results/cohere_preproc_device_bench/summary.json`
- `tmp/asr_bottleneck/results/cohere_preproc_device_bench/cpu_per_item_cost.png`
- `tmp/asr_bottleneck/results/cohere_preproc_device_bench/cuda_per_item_cost.png`

### Gated coordinator: CPU vs GPU preprocessing

Why:

- Even when `bulk-bench.py` releases all `4096` requests at once, the async audio
  microbatch queue still fills in a staggered way because requests only reach that
  queue after request parsing, audio decode, resampling, and request-specific
  setup.
- The gated coordinator fixed "prebuilding batches ahead of free workers", but it
  still cannot make arrivals simultaneous if upstream CPU work dribbles requests
  into the shared queue.
- The hypothesis was that moving HF feature extraction from CPU to GPU would free
  more CPU for that upstream work, making arrivals denser by the time they hit the
  async queue.

Implementation notes:

- Added `COHERE_ASR_MICROBATCH_PREPROCESS_DEVICE={cpu,cuda}` to the server path.
- The feature extractor now runs on that device, but still returns features to CPU
  (`to_cpu=True`) before handing off to the multimodal path, so these results are
  not from leaving preprocessed tensors resident on GPU.
- Extended `bench_asr_coordinator_policy_sweep.py` with `--preproc-device` and
  device-aware result tagging.

Reduced search matrix (`512` samples, `128` concurrency):

Bulk search with `enter=256`, `exit=64`, `wait=0.001`:

| config | device | workers | drain target | RTFx | avg batch size | singleton items |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `cpu_control` | `cpu` | `8` | `48` | `503.69` | `1.37` | `59.57%` |
| `gpu_target16` | `cuda` | `1` | `16` | `523.33` | `1.44` | `66.02%` |
| `gpu_target32` | `cuda` | `1` | `32` | `522.10` | `1.45` | `66.21%` |
| `gpu_target48` | `cuda` | `1` | `48` | `512.94` | `1.57` | `60.55%` |
| `gpu_target64` | `cuda` | `1` | `64` | `527.31` | `1.49` | `63.28%` |

Closed-loop `nick-bench.py` search with the same thresholds:

| config | device | workers | drain target | RTFx | avg batch size | singleton items |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `cpu_control` | `cpu` | `8` | `48` | `558.61` | `1.32` | `63.48%` |
| `gpu_target16` | `cuda` | `1` | `16` | `574.64` | `1.49` | `62.11%` |
| `gpu_target32` | `cuda` | `1` | `32` | `568.22` | `1.43` | `66.80%` |
| `gpu_target48` | `cuda` | `1` | `48` | `566.39` | `1.25` | `76.80%` |
| `gpu_target64` | `cuda` | `1` | `64` | `569.21` | `1.52` | `61.52%` |

The reduced search winners were:

- bulk: `device=cuda`, `workers=1`, `drain_target=64`
- `nick`: `device=cuda`, `workers=1`, `drain_target=16`

Full `4096`-sample validation:

| workload | config | device | workers | drain target | RTFx | avg batch size | singleton items |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `bulk-bench.py` | `cpu_control` | `cpu` | `8` | `48` | `707.32` | `2.84` | `0.02%` |
| `bulk-bench.py` | `gpu_best_reduced` | `cuda` | `1` | `64` | `984.58` | `8.57` | `0.00%` |
| `nick-bench.py` | `cpu_control` | `cpu` | `8` | `48` | `884.19` | `2.07` | `4.56%` |
| `nick-bench.py` | `gpu_best_reduced` | `cuda` | `1` | `16` | `1485.22` | `4.98` | `6.45%` |

Interpretation:

- GPU preprocessing with a single worker materially improved the gated coordinator
  family on both workloads once validated at full scale.
- On the gated coordinator path, GPU preprocessing improved:
  - bulk `RTFx` from `707.32` to `984.58` (`+39%`)
  - closed-loop `nick` `RTFx` from `884.19` to `1485.22` (`+68%`)
- For `nick-bench.py`, that GPU result is also the new best measured uniform
  configuration overall, beating the previous best CPU async result
  (`1485.22` vs `1384.91`, about `+7%`).
- For `bulk-bench.py`, GPU preprocessing significantly improved the gated
  coordinator path, but it still did **not** beat the earlier fixed-policy async
  CPU best (`984.58` vs `1061.37`, about `-7%`).
- The reduced `512`-sample search was only weakly predictive of the full `4096`
  result. In reduced runs, GPU did not obviously make batches denser; at full
  scale, it did.
- This is consistent with the original hypothesis: offloading HF preprocessing to
  GPU can free enough CPU for upstream request parsing / decode / resample work
  that the async queue becomes materially denser at large backlog, even though
  arrivals are still fundamentally staggered before that queue boundary.

Follow-up bulk tuning with the GPU-preproc base (`device=cuda`, `workers=1`):

- Lowering `HF_PROCESSOR_NUM_THREADS` from `16` to `1` improved the full bulk run:
  - `drain_target=64`, `wait=0.001`
  - `RTFx=1036.76` vs `984.58` before (`+5.3%`)
- Changing `CROSS_REQUEST_AUDIO_MICROBATCH_WAIT_TIMEOUT_S` from `0.001` to `0.002`
  regressed bulk throughput:
  - `RTFx=889.91`
- Tightening hysteresis to `enter=64`, `exit=4` also regressed bulk throughput:
  - `RTFx=1007.44`

Full bulk drain-target sweep on the improved GPU base (`HF_PROCESSOR_NUM_THREADS=1`,
`enter=256`, `exit=64`, `wait=0.001`):

| drain target | RTFx | avg batch size | avg qwait (ms) | singleton items |
| --- | ---: | ---: | ---: | ---: |
| `16` | `1134.24` | `7.63` | `250.73` | `0.00%` |
| `32` | `1238.71` | `8.29` | `219.85` | `0.02%` |
| `48` | `1008.12` | `8.73` | `229.10` | `0.00%` |
| `64` | `541.90` | `8.85` | `220.87` | `0.00%` |

Revalidation:

- `drain_target=32` did **not** reproduce cleanly on immediate rerun:
  - first run: `RTFx=1238.71`
  - confirm run: `RTFx=999.89`
- `drain_target=16` was much more stable across two full runs:
  - first run: `RTFx=1134.24`
  - confirm run: `RTFx=1118.79`

Updated interpretation:

- The most reliable improvement for open-loop bulk so far is:
  - `COHERE_ASR_MICROBATCH_PREPROCESS_DEVICE=cuda`
  - `workers=1`
  - `HF_PROCESSOR_NUM_THREADS=1`
  - `CROSS_REQUEST_AUDIO_MICROBATCH_DRAIN_ENTER_THRESHOLD=256`
  - `CROSS_REQUEST_AUDIO_MICROBATCH_DRAIN_EXIT_THRESHOLD=64`
  - `CROSS_REQUEST_AUDIO_MICROBATCH_DRAIN_TARGET_BATCH_SIZE=16`
  - `CROSS_REQUEST_AUDIO_MICROBATCH_WAIT_TIMEOUT_S=0.001`
- That stable full-bulk configuration delivered `RTFx=1118.79` on the confirm run,
  which is about `+5.4%` over the earlier best fixed-policy CPU bulk result
  (`1061.37`).
- `drain_target=32` achieved the highest single observed bulk run, but the rerun
  variance was too large to treat it as the new settled best configuration yet.

Artifacts:

- `bench_asr_coordinator_policy_sweep.py`
- `tmp/asr_bottleneck/results/coordinator_policy_sweep/device_bulk_reduced_cpu`
- `tmp/asr_bottleneck/results/coordinator_policy_sweep/device_bulk_reduced_gpu`
- `tmp/asr_bottleneck/results/coordinator_policy_sweep/device_nick_reduced_cpu`
- `tmp/asr_bottleneck/results/coordinator_policy_sweep/device_nick_reduced_gpu`
- `tmp/asr_bottleneck/results/coordinator_policy_sweep/device_bulk_validation_cpu`
- `tmp/asr_bottleneck/results/coordinator_policy_sweep/device_bulk_validation_gpu`
- `tmp/asr_bottleneck/results/coordinator_policy_sweep/device_bulk_validation_gpu_hf1`
- `tmp/asr_bottleneck/results/coordinator_policy_sweep/device_bulk_validation_gpu_wait0002`
- `tmp/asr_bottleneck/results/coordinator_policy_sweep/device_bulk_validation_gpu_hf1_enter64_exit4`
- `tmp/asr_bottleneck/results/coordinator_policy_sweep/device_bulk_validation_gpu_hf1_target_sweep`
- `tmp/asr_bottleneck/results/coordinator_policy_sweep/device_bulk_validation_gpu_hf1_target16_confirm`
- `tmp/asr_bottleneck/results/coordinator_policy_sweep/device_bulk_validation_gpu_hf1_target32_confirm`
- `tmp/asr_bottleneck/results/coordinator_policy_sweep/device_nick_validation_cpu`
- `tmp/asr_bottleneck/results/coordinator_policy_sweep/device_nick_validation_gpu`

## Overall Analysis

The original single-worker cross-request async microbatch design did **not**
deliver the hoped-for speedup, but the later multi-worker CPU design did, and the
latest gated coordinator plus GPU preprocessing improved the closed-loop path
further.

Why the original single-worker design did not work:

1. Correctness had to be fixed first.
   Without restoring the multimodal renderer path and teaching Cohere ASR how
   to consume already-preprocessed audio, the model was decoding without the
   right audio signal.

2. Reusing a persistent extractor fixed one cold-start problem but did not fix
   the end-to-end regression.
   This means the remaining issue is structural, not just one-time setup cost.

3. On realistic VoxPopuli, single-worker async microbatching did not beat the earlier
   baseline.

4. On homogeneous fixed-duration data, single-worker async microbatching still lost.
   That rules out padding variability as the main explanation.

5. Instrumentation showed that:
   - singleton batches are not dominant
   - medium-sized batches are dominant
   - medium-sized batches do not amortize enough CPU cost
   - large batches help per-item preprocess cost, but they come with heavy queue
     wait

In short:

- The original design serialized all cross-request preprocessing through one
  global worker.
- The resulting queueing cost is real.
- The batches commonly formed are not large enough to offset that cost.

What changed with the multi-worker redesign:

1. A shared queue feeding multiple workers reduced head-of-line blocking.
2. Each worker kept its own persistent preprocessor, so extractor reuse was
   preserved without sharing one instance across threads.
3. The `64`-thread cap kept CPU preprocessing bounded while still allowing
   multiple workers to make progress.
4. Once the worker topology was fixed, a larger `MB` (`512`) gave a small
   additional gain on the best topology (`workers=8`), but topology mattered
   much more than `MB` alone.
5. After that, tuning `_CROSS_REQUEST_AUDIO_MICROBATCH_WAIT_TIMEOUT_S` showed
   that a tiny nonzero wait (`0.001s`) was better than both `0.0s` and the
   earlier `0.002s` default on the best topology.
6. Finally, switching to an open-loop bulk benchmark changed the preferred
   timeout again: for one-shot backlog drain, `0.002s` slightly beat the
   no-async baseline, while `0.001s` no longer looked best.
7. A direct device-level benchmark of the HF preprocessing path showed that GPU
   has much stronger batching upside than CPU, but both devices peaked around
   moderate batch sizes (`BS=16`) rather than the largest tested sizes.
8. Extending the gated coordinator to run HF preprocessing on GPU improved the
   coordinator family substantially at full scale, especially for closed-loop
   `nick-bench.py`, where `device=cuda`, `workers=1`, and a moderate drain target
   became the new best measured uniform setup.
9. Reduced `512`-sample coordinator searches were not sufficient to judge this
   device tradeoff; the stronger effect only became clear on full `4096`-sample
   validation, which suggests queue-density behavior changes materially with
   backlog scale.

Current best measured uniform configurations:

- Closed-loop `nick-bench.py` best:
  - `COHERE_ASR_MICROBATCH_PREPROCESS_DEVICE=cuda`
  - `workers=1`
  - `CROSS_REQUEST_AUDIO_MICROBATCH_MAX_BATCH_SIZE=512`
  - `CROSS_REQUEST_AUDIO_MICROBATCH_DRAIN_ENTER_THRESHOLD=256`
  - `CROSS_REQUEST_AUDIO_MICROBATCH_DRAIN_EXIT_THRESHOLD=64`
  - `CROSS_REQUEST_AUDIO_MICROBATCH_DRAIN_TARGET_BATCH_SIZE=16`
  - `CROSS_REQUEST_AUDIO_MICROBATCH_WAIT_TIMEOUT_S=0.001`
  - `HF_PROCESSOR_NUM_THREADS=16`
  - `CROSS_REQUEST_AUDIO_MICROBATCH_MAX_TOTAL_TORCH_THREADS=64`
  - homogeneous `RTFx=1485.22`
- Open-loop `bulk-bench.py` best:
  - `COHERE_ASR_MICROBATCH_PREPROCESS_DEVICE=cuda`
  - `workers=1`
  - `HF_PROCESSOR_NUM_THREADS=1`
  - `CROSS_REQUEST_AUDIO_MICROBATCH_MAX_BATCH_SIZE=512`
  - `CROSS_REQUEST_AUDIO_MICROBATCH_DRAIN_ENTER_THRESHOLD=256`
  - `CROSS_REQUEST_AUDIO_MICROBATCH_DRAIN_EXIT_THRESHOLD=64`
  - `CROSS_REQUEST_AUDIO_MICROBATCH_DRAIN_TARGET_BATCH_SIZE=16`
  - `CROSS_REQUEST_AUDIO_MICROBATCH_WAIT_TIMEOUT_S=0.001`
  - `CROSS_REQUEST_AUDIO_MICROBATCH_MAX_TOTAL_TORCH_THREADS=64`
  - homogeneous `RTFx=1118.79` on confirm run
- Open-loop `bulk-bench.py` best gated-coordinator result:
  - `COHERE_ASR_MICROBATCH_PREPROCESS_DEVICE=cuda`
  - `workers=1`
  - `CROSS_REQUEST_AUDIO_MICROBATCH_MAX_BATCH_SIZE=512`
  - `CROSS_REQUEST_AUDIO_MICROBATCH_DRAIN_ENTER_THRESHOLD=256`
  - `CROSS_REQUEST_AUDIO_MICROBATCH_DRAIN_EXIT_THRESHOLD=64`
  - `CROSS_REQUEST_AUDIO_MICROBATCH_DRAIN_TARGET_BATCH_SIZE=16`
  - `CROSS_REQUEST_AUDIO_MICROBATCH_WAIT_TIMEOUT_S=0.001`
  - `HF_PROCESSOR_NUM_THREADS=1`
  - `CROSS_REQUEST_AUDIO_MICROBATCH_MAX_TOTAL_TORCH_THREADS=64`
  - homogeneous `RTFx=1118.79` on confirm run

Fresh `nick-bench.py` comparison rerun on current code (`4096` samples):

| config | key settings | RTFx | delta vs sync | avg latency (s) | p95 latency (s) | avg batch size | avg qwait (ms) | avg prep qwait (ms) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `sync_preproc_cpu_baseline` | async off | `1259.65` | `0.0%` | `3.0329` | `6.3554` | `-` | `-` | `-` |
| `older_gpu_optimization` | `device=cuda`, `workers=1`, `HF_THREADS=16`, `enter=256`, `exit=64`, `target=16`, `wait=0.001`, `batch_coordinator=1` | `1378.27` | `+9.4%` | `2.7753` | `8.5502` | `5.01` | `34.04` | `-` |
| `older_gpu_optimization_hf1` | `device=cuda`, `workers=1`, `HF_THREADS=1`, `enter=256`, `exit=64`, `target=16`, `wait=0.001`, `batch_coordinator=1` | `1466.37` | `+16.4%` | `2.6139` | `8.3602` | `4.70` | `33.35` | `-` |
| `multistage_gpu` | `device=cuda`, `workers=1`, `prep_workers=24`, `HF_THREADS=1`, `enter=256`, `exit=64`, `target=64`, `wait=0.001`, `multi_stage=1` | `1279.01` | `+1.5%` | `2.9999` | `8.6306` | `5.58` | `25.42` | `40.64` |
| `multistage_backlog_gpu` | `device=cuda`, `workers=1`, `prep_workers=24`, `HF_THREADS=1`, `enter=256`, `exit=64`, `target=64`, `wait=0.001`, `multi_stage=1`, `ready_backlog_gating=1`, `rb_target=64`, `rb_wait=0.002` | `1296.52` | `+2.9%` | `2.9903` | `8.1838` | `5.97` | `27.25` | `48.08` |

Interpretation of the fresh `nick-bench.py` rerun:

- On current code, the best result in this fresh closed-loop comparison was the
  older single-stage GPU optimization with `HF_THREADS=1`.
- Relative to the fresh sync baseline, that `HF_THREADS=1` variant improved
  `RTFx` by about `+16.4%` (`1466.37` vs `1259.65`).
- Within the older single-stage GPU path, reducing `HF_PROCESSOR_NUM_THREADS`
  from `16` to `1` improved `RTFx` from `1378.27` to `1466.37` (about `+6.4%`)
  while also reducing average latency (`2.7753s` to `2.6139s`).
- The multistage GPU variants did beat sync, but only modestly:
  `+1.5%` ungated and `+2.9%` with backlog gating.
- Backlog gating helped the multistage path a little (`1296.52` vs `1279.01`),
  but it still trailed the older GPU optimization with `HF_THREADS=1` by about
  `-11.6%`.
- The multistage variants realized slightly larger average batches than the
  older GPU optimization variants (`5.58`/`5.97` vs `5.01`/`4.70`), but that
  benefit was not
  enough to overcome the added prep-stage queueing (`avg_prep_qwait_ms`
  `40.64`/`48.08`).
- As with the fresh bulk rerun below, this section is an apples-to-apples
  comparison on current code; it does **not** replace the earlier best-record
  closed-loop confirm run of `RTFx=1485.22`, though this `HF_THREADS=1` rerun
  came close.
- So, for closed-loop `nick-bench.py`, the recent bulk-oriented multistage and
  backlog-gating changes do **not** currently beat the older GPU-tuned
  coordinator path.

Artifacts:

- `tmp/asr_bottleneck/results/nick_compare_20260323/nick_summary.json`
- `tmp/asr_bottleneck/results/nick_compare_20260323/`

Fresh isolated `vllm bench serve` VoxPopuli rerun on `facebook/voxpopuli`
(`hf_subset=en`, `hf_split=test`):

- The earlier partial numbers were invalid because both benchmark containers were
  running with Docker `host` networking, so `127.0.0.1:8000` collided with
  another active server. The rows below are from a clean rerun on isolated
  ports (`8013` for the main sweep, `8015` for a targeted retry).

| config | max concurrency | key settings | status | RTFx | completed | failed | req/s |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: |
| `sync_preproc_cpu_baseline` | `512` | async off | full run | `1043.96` | `1842` | `0` | `108.35` |
| `older_gpu_optimization_hf1` | `384` | `device=cuda`, `workers=1`, `HF_THREADS=1`, `target=16`, `multi_stage=0` | partial, OOM | `889.12` | `1725` | `117` | `92.27` |
| `older_gpu_optimization_hf1` | `256` | `device=cuda`, `workers=1`, `HF_THREADS=1`, `target=16`, `multi_stage=0` | partial, OOM | `623.35` | `520` | `1322` | `63.81` |
| `older_gpu_optimization_hf1` | `128` | `device=cuda`, `workers=1`, `HF_THREADS=1`, `target=16`, `multi_stage=0` | targeted retry, partial, engine died | `882.14` | `1201` | `641` | `91.66` |
| `multistage_gpu` | `384` | `device=cuda`, `workers=1`, `prep_workers=24`, `HF_THREADS=1`, `target=64`, `multi_stage=1` | partial, OOM | `777.66` | `1200` | `642` | `80.76` |
| `multistage_gpu` | `256` | `device=cuda`, `workers=1`, `prep_workers=24`, `HF_THREADS=1`, `target=64`, `multi_stage=1` | partial, OOM | `706.88` | `530` | `1312` | `72.60` |
| `multistage_gpu` | `128` | `device=cuda`, `workers=1`, `prep_workers=24`, `HF_THREADS=1`, `target=64`, `multi_stage=1` | full run | `994.86` | `1842` | `0` | `103.25` |
| `multistage_backlog_gpu` | `384` | `device=cuda`, `workers=1`, `prep_workers=24`, `HF_THREADS=1`, `target=64`, `multi_stage=1`, `rb_target=64`, `rb_wait=0.002` | partial, OOM | `833.86` | `1726` | `116` | `86.56` |
| `multistage_backlog_gpu` | `256` | `device=cuda`, `workers=1`, `prep_workers=24`, `HF_THREADS=1`, `target=64`, `multi_stage=1`, `rb_target=64`, `rb_wait=0.002` | partial, OOM | `619.88` | `530` | `1312` | `63.66` |
| `multistage_backlog_gpu` | `128` | `device=cuda`, `workers=1`, `prep_workers=24`, `HF_THREADS=1`, `target=64`, `multi_stage=1`, `rb_target=64`, `rb_wait=0.002` | full run | `1005.99` | `1842` | `0` | `104.41` |

Interpretation of the isolated rerun:

- On this heterogeneous real-dataset benchmark, the best stable ranking was
  `sync_preproc_cpu_baseline_c512` > `multistage_backlog_gpu_c128` >
  `multistage_gpu_c128`.
- Relative to the sync baseline, the two stable GPU candidates were still lower
  on end-to-end throughput: about `-3.6%` for `multistage_backlog_gpu_c128` and
  `-4.7%` for `multistage_gpu_c128`.
- Backlog gating helped the multistage path slightly at the first stable point
  (`1005.99` vs `994.86` at `c128`), but not enough to beat the sync CPU
  baseline.
- The older single-stage GPU path never produced a full-dataset stable run in
  this isolated rerun. Even the dedicated `c128` retry only completed `1201`
  requests before the engine died.
- Higher GPU concurrencies (`256` and `384`) were still unstable across all
  async GPU variants on VoxPopuli, with repeated partial completions and
  `CUDA OOM` / engine-death behavior.

Artifacts:

- `tmp/asr_bottleneck/results/voxpopuli_concurrency_sweep_20260323_clean/voxpopuli_concurrency_summary.json`
- `tmp/asr_bottleneck/results/voxpopuli_concurrency_sweep_20260323_clean/`
- `tmp/asr_bottleneck/results/voxpopuli_older_gpu_c128_retry_20260323/older_gpu_optimization_hf1_c128.json`
- `tmp/asr_bottleneck/results/voxpopuli_older_gpu_c128_retry_20260323/`

Fresh `bulk-bench.py` comparison rerun on current code (`4096` samples):

| config | key settings | RTFx | avg latency (s) | avg batch size | avg qwait (ms) |
| --- | --- | ---: | ---: | ---: | ---: |
| `sync_preproc_baseline` | async off | `1006.91` | `18.7930` | `-` | `-` |
| `async_preproc_cpu_best_fixed_policy` | `device=cpu`, `workers=8`, `max_batch=512`, `HF_THREADS=16`, `wait=0.002`, `adaptive_drain=0`, `batch_coordinator=0` | `890.01` | `23.1228` | `1.43` | `120.75` |
| `async_preproc_gpu_best` | `device=cuda`, `workers=1`, `max_batch=512`, `HF_THREADS=1`, `enter=256`, `exit=64`, `target=16`, `wait=0.001`, `batch_coordinator=1` | `1055.03` | `19.9779` | `7.57` | `255.82` |

Interpretation of the fresh rerun:

- The fresh rerun preserved the same ordering as the earlier bulk conclusions:
  GPU-best async was the strongest of the three, sync remained competitive, and
  the old fixed-policy CPU async path was weakest.
- In this rerun, GPU-best async beat sync by about `+4.8%` in `RTFx`
  (`1055.03` vs `1006.91`).
- The old fixed-policy CPU async path under current code came in about `-11.6%`
  below sync (`890.01` vs `1006.91`) and remained heavily singleton-dominated
  (`avg_batch_size=1.43`).
- These reruns are useful for apples-to-apples comparison on current code, but
  they do **not** replace the earlier best-record confirm run. The best measured
  open-loop bulk result remains the GPU configuration above with
  `RTFx=1118.79`.

Artifacts:

- `tmp/asr_bottleneck/results/fresh_bulk_compare_20260323/summary.json`
- `tmp/asr_bottleneck/results/fresh_bulk_compare_20260323/`

Fresh `bulk-bench.py` VoxPopuli sorted single-chunk comparison
(`facebook/voxpopuli` `en/test`, requests sorted by raw duration, only clips
shorter than `30s`, `1830` submitted requests):

| config | key settings | status | RTFx | successful | failed | total time (s) |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `sync_preproc` | async off | full run | `1023.41` | `1830` | `0` | `16.9076` |
| `older_gpu_optimization_hf1` | `device=cuda`, `workers=1`, `HF_THREADS=1`, `enter=256`, `exit=64`, `target=16`, `wait=0.001`, `batch_coordinator=1` | full run | `1190.70` | `1830` | `0` | `14.5322` |
| `multistage_backlog_gpu` | `device=cuda`, `workers=1`, `prep_workers=24`, `HF_THREADS=1`, `enter=256`, `exit=64`, `target=64`, `wait=0.001`, `ready_backlog_target=64`, `ready_backlog_wait=0.002` | partial with `500`s | `358.82` | `1045` | `785` | `16.3422` |

Interpretation of the sorted single-chunk Vox bulk run:

- On this open-loop heterogeneous workload, `older_gpu_optimization_hf1` was the
  strongest configuration and beat `sync_preproc` by about `+16.3%` in `RTFx`
  (`1190.70` vs `1023.41`).
- `sync_preproc` remained stable and fully completed all `1830` filtered
  requests, but it no longer led once the workload was both sorted by duration
  and restricted to single-chunk clips.
- `multistage_backlog_gpu` was not stable in this mode. The server emitted many
  `500 Internal Server Error` responses and the engine log shows a
  `torch.OutOfMemoryError` during model execution after the multistage
  preprocessor had already built large ready batches.
- Operationally, these runs needed isolated ports plus container cleanup/restart
  between some configurations because GPU allocations remained sticky after
  server shutdown; the numbers above are from the successful measured runs after
  that cleanup.

Artifacts:

- `tmp/asr_bottleneck/results/bulk_bench_vox_sorted_singlechunk_gpu1_20260323/`
- `tmp/asr_bottleneck/results/bulk_bench_vox_sorted_singlechunk_restart_20260323/`

Follow-up multistage backlog policy matrix on the same sorted single-chunk Vox
bulk workload (same `1830` requests, now varying only
`drain_target_batch_size`, `ready_backlog_target`, and `max_concurrent`):

| max concurrent | drain target | ready backlog target | status | RTFx | total time (s) |
| ---: | ---: | ---: | --- | ---: | ---: |
| `64` | `16` | `16` | full run | `1462.95` | `11.8278` |
| `64` | `16` | `32` | full run | `1500.99` | `11.5280` |
| `64` | `32` | `16` | full run | `1503.23` | `11.5108` |
| `64` | `32` | `32` | full run | `858.43` | `20.1571` |
| `96` | `16` | `16` | full run | `1296.32` | `13.3481` |
| `96` | `16` | `32` | full run | `1297.74` | `13.3335` |
| `96` | `32` | `16` | full run | `1306.66` | `13.2425` |
| `96` | `32` | `32` | full run | `1301.30` | `13.2970` |
| `128` | `16` | `16` | full run | `1209.75` | `14.3034` |
| `128` | `16` | `32` | full run | `941.41` | `18.3804` |
| `128` | `32` | `16` | full run | `916.76` | `18.8745` |
| `128` | `32` | `32` | full run | `1182.59` | `14.6319` |

Interpretation of the policy matrix:

- The earlier `multistage_backlog_gpu` failure was not caused by the
  multi-stage/backlog design in isolation. It was specifically the aggressive
  `target=64`, `ready_backlog_target=64`, `max_concurrent=128` setting that was
  too dense for this workload.
- Reducing the policy targets made the path stable across the entire matrix.
  All `12/12` configurations above completed `1830/1830` requests with no
  engine OOM or HTTP `500` failures.
- The best configuration in this matrix was `max_concurrent=64`,
  `drain_target_batch_size=32`, `ready_backlog_target=16`, which reached
  `RTFx=1503.23`.
- That best tuned multistage-backlog setting beat the same-workload
  `older_gpu_optimization_hf1` result (`1190.70`) by about `+26.3%`, and beat
  the best tuned `sync_preproc` result from the fresh sweep below
  (`1073.20` at `max_concurrent=64`) by about `+40.1%`.
- `max_concurrent=96` was also consistently strong (`RTFx` about `1296-1307`)
  and may be the safer higher-throughput operating point if we want less
  sensitivity to exact target settings.
- One outlier remained: `max_concurrent=64`, `drain_target=32`,
  `ready_backlog_target=32` stayed fully stable but collapsed to
  `RTFx=858.43`, suggesting that over-buffering can still hurt end-to-end bulk
  latency badly even without causing an OOM.

Exact values used for the best `RTFx=1503.23` multistage-backlog run:

```bash
HF_PROCESSOR_NUM_THREADS=1
COHERE_ASR_MICROBATCH_PREPROCESS_DEVICE=cuda
VLLM_COHERE_ASR_CROSS_REQUEST_AUDIO_MICROBATCH=1
VLLM_COHERE_ASR_CROSS_REQUEST_AUDIO_MICROBATCH_BATCH_COORDINATOR=1
VLLM_COHERE_ASR_CROSS_REQUEST_AUDIO_MICROBATCH_MULTI_STAGE_PIPELINE=1
VLLM_COHERE_ASR_CROSS_REQUEST_AUDIO_MICROBATCH_READY_BACKLOG_GATING=1
VLLM_COHERE_ASR_CROSS_REQUEST_AUDIO_MICROBATCH_ADAPTIVE_DRAIN=1
VLLM_COHERE_ASR_CROSS_REQUEST_AUDIO_MICROBATCH_STATS=1
CROSS_REQUEST_AUDIO_MICROBATCH_MAX_BATCH_SIZE=512
CROSS_REQUEST_AUDIO_MICROBATCH_NUM_WORKERS=1
CROSS_REQUEST_AUDIO_MICROBATCH_PREP_NUM_WORKERS=24
CROSS_REQUEST_AUDIO_MICROBATCH_MAX_TOTAL_TORCH_THREADS=64
CROSS_REQUEST_AUDIO_MICROBATCH_WAIT_TIMEOUT_S=0.001
CROSS_REQUEST_AUDIO_MICROBATCH_DRAIN_ENTER_THRESHOLD=256
CROSS_REQUEST_AUDIO_MICROBATCH_DRAIN_EXIT_THRESHOLD=64
CROSS_REQUEST_AUDIO_MICROBATCH_DRAIN_TARGET_BATCH_SIZE=32
CROSS_REQUEST_AUDIO_MICROBATCH_READY_BACKLOG_ENTER_THRESHOLD=256
CROSS_REQUEST_AUDIO_MICROBATCH_READY_BACKLOG_EXIT_THRESHOLD=64
CROSS_REQUEST_AUDIO_MICROBATCH_READY_BACKLOG_TARGET=16
CROSS_REQUEST_AUDIO_MICROBATCH_READY_BACKLOG_MAX_WAIT_S=0.002
CROSS_REQUEST_AUDIO_MICROBATCH_STATS_LOG_EVERY_BATCHES=200
```

and the benchmark CLI was:

```bash
python3 bulk-bench.py \
  --dataset voxpopuli \
  --hf-subset en \
  --hf-split test \
  --sorted \
  --single-chunk-only \
  --max-concurrent 64
```

Artifacts:

- `tmp/asr_bottleneck/results/vox_bulk_multistage_backlog_matrix_20260323/summary.json`
- `tmp/asr_bottleneck/results/vox_bulk_multistage_backlog_matrix_20260323/`

Fresh `sync_preproc` concurrency sweep on the same sorted single-chunk Vox bulk
workload:

| max concurrent | status | RTFx | total time (s) | avg latency (s) | p95 latency (s) |
| ---: | --- | ---: | ---: | ---: | ---: |
| `64` | full run | `1073.20` | `16.1233` | `8.7302` | `13.8670` |
| `128` | full run | `1005.51` | `17.2087` | `9.4047` | `13.7739` |
| `256` | full run | `1023.91` | `16.8994` | `9.4076` | `13.9002` |
| `512` | full run | `1025.57` | `16.8720` | `9.2902` | `13.8067` |

Interpretation of the sync sweep:

- `sync_preproc` did have a better operating point on this workload than the
  earlier single `128`-concurrency row suggested. The best fresh result was
  `max_concurrent=64` with `RTFx=1073.20`.
- That `64` setting beat the fresh `512` run by about `+4.6%` in `RTFx`
  (`1073.20` vs `1025.57`).
- `max_concurrent=128` was actually the weakest point in the sweep, while
  `256` and `512` were close to one another and clearly below `64`.
- Even after tuning sync concurrency, the best multistage-backlog policy above
  still remained far ahead on this workload (`1503.23` vs `1073.20`).

Artifacts:

- `tmp/asr_bottleneck/results/vox_bulk_sync_concurrency_sweep_20260323/summary.json`
- `tmp/asr_bottleneck/results/vox_bulk_sync_concurrency_sweep_20260323/`

Fresh `bulk-bench.py` Vox feature-contribution ladder on the same sorted
single-chunk workload (same `1830` requests, fixed `max_concurrent=64`; for the
GPU rows below, hold the drain policy at `enter=256`, `exit=64`, `target=16`
and only add the named feature at each step):

| row | feature set | key settings | RTFx | delta vs sync | total time (s) | avg latency (s) | p95 latency (s) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `0` | `sync baseline` | async off | `1190.12` | `+0.0%` | `14.5392` | `7.5349` | `12.5900` |
| `1` | `sync + gpu` | async off, `device=cuda`, `HF_THREADS=1` | `1201.51` | `+1.0%` | `14.4014` | `7.5324` | `12.2853` |
| `2` | `async MB + cpu` | `device=cpu`, `workers=8`, `HF_THREADS=16`, `wait=0.002` | `1176.84` | `-1.1%` | `14.7033` | `8.2915` | `12.7497` |
| `3` | `async MB + gpu` | `device=cuda`, `workers=1`, `HF_THREADS=1`, `wait=0.001` | `1371.43` | `+15.2%` | `12.6171` | `7.0877` | `10.6740` |
| `4` | `async MB + gpu + batch coordinator` | row `3` + `batch_coordinator=1` | `1498.76` | `+25.9%` | `11.5451` | `6.7144` | `9.5616` |
| `5` | `async MB + gpu + batch coordinator + adaptive` | row `4` + `adaptive_drain=1` | `1463.37` | `+23.0%` | `11.8244` | `6.8554` | `9.8298` |
| `6` | `async MB + gpu + batch coordinator + adaptive + multistage` | row `5` + `multi_stage=1`, `prep_workers=24` | `1472.39` | `+23.7%` | `11.7519` | `7.0700` | `9.7863` |
| `7` | `async MB + gpu + batch coordinator + adaptive + multistage + backlog` | row `6` + `ready_backlog_gating=1`, `rb_target=16`, `rb_wait=0.002` | `1474.88` | `+23.9%` | `11.7321` | `6.8952` | `9.8096` |

Interpretation of the Vox feature ladder:

- Adding `COHERE_ASR_MICROBATCH_PREPROCESS_DEVICE=cuda` to the synchronous path
  by itself barely changed throughput. The `sync + gpu` row reached `1201.51`,
  only about `+1.0%` above the plain sync baseline (`1190.12`).
- The CPU async-microbatch row remained slightly below sync on this workload
  (`1176.84`, about `-1.1%` vs baseline), so just turning on async batching is
  not enough.
- The first major gain came from combining async microbatching with GPU
  preprocessing. That lifted `RTFx` from `1201.51` in the `sync + gpu` control
  to `1371.43` in the plain GPU-async row, about `+14.1%` relative to the
  synchronous GPU control and `+15.2%` relative to the plain sync baseline.
- The single biggest incremental improvement still came from adding the batch
  coordinator on top of GPU preprocessing. That raised `RTFx` from `1371.43` to
  `1498.76`, which is about `+9.3%` versus the plain GPU-async row and
  `+25.9%` versus sync.
- In this fixed-policy ladder, `adaptive_drain` did **not** help. Turning it on
  reduced `RTFx` from `1498.76` to `1463.37` (about `-2.4%` relative to the
  fixed-coordinator row).
- Adding the multi-stage prep pipeline and then backlog-gated dispatch only
  recovered small amounts on top of that adaptive row at these held-constant
  targets: `1472.39` and `1474.88`, respectively.
- So for component attribution on this real Vox bulk workload, the big gains
  came from `async MB + gpu` and then `batch_coordinator=1`, not from putting
  the unchanged synchronous path on GPU. The later adaptive / multistage /
  backlog features were secondary unless their policy knobs were tuned further.
- The separately tuned backlog policy above still produced the best same-workload
  result, `RTFx=1503.23` with `drain_target_batch_size=32` and
  `ready_backlog_target=16`. That is only about `+1.9%` above the fixed-policy
  backlog row here (`1474.88`) and about `+0.3%` above the simpler
  fixed-coordinator GPU row (`1498.76`), which reinforces that the bulk of the
  improvement had already arrived before the final multistage/backlog features.

Artifacts:

- `tmp/asr_bottleneck/results/vox_bulk_feature_ladder_20260323/summary.json`
- `tmp/asr_bottleneck/results/vox_bulk_feature_ladder_20260323/`
- `tmp/asr_bottleneck/results/vox_bulk_sync_gpu_20260323/summary.json`
- `tmp/asr_bottleneck/results/vox_bulk_sync_gpu_20260323/`

## Librispeech repro

Fresh `bulk-bench.py` Librispeech feature-contribution ladder on sorted
single-chunk `openslr/librispeech_asr:test.clean` (same `2611` requests, fixed
`max_concurrent=64`; for the GPU rows below, hold the drain policy at
`enter=256`, `exit=64`, `target=16` and only add the named feature at each
step):

| row | feature set | key settings | RTFx | delta vs sync | total time (s) | avg latency (s) | p95 latency (s) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `0` | `sync baseline` | async off | `968.01` | `+0.0%` | `19.7965` | `9.6876` | `16.8262` |
| `1` | `sync + gpu` | async off, `device=cuda`, `HF_THREADS=1` | `997.82` | `+3.1%` | `19.2050` | `9.5768` | `16.1862` |
| `2` | `async MB + cpu` | `device=cpu`, `workers=8`, `HF_THREADS=16`, `wait=0.002` | `1022.36` | `+5.6%` | `18.7442` | `9.5995` | `15.7211` |
| `3` | `async MB + gpu` | `device=cuda`, `workers=1`, `HF_THREADS=1`, `wait=0.001` | `1250.32` | `+29.2%` | `15.3266` | `8.3114` | `12.3486` |
| `4` | `async MB + gpu + batch coordinator` | row `3` + `batch_coordinator=1` | `1239.33` | `+28.0%` | `15.4626` | `8.2488` | `12.3897` |
| `5` | `async MB + gpu + batch coordinator + adaptive` | row `4` + `adaptive_drain=1` | `1239.51` | `+28.0%` | `15.4603` | `8.5502` | `12.5414` |
| `6` | `async MB + gpu + batch coordinator + adaptive + multistage` | row `5` + `multi_stage=1`, `prep_workers=24` | `1193.60` | `+23.3%` | `16.0547` | `8.6942` | `12.7697` |
| `7` | `async MB + gpu + batch coordinator + adaptive + multistage + backlog` | row `6` + `ready_backlog_gating=1`, `rb_target=16`, `rb_wait=0.002` | `1214.57` | `+25.5%` | `15.7778` | `8.5112` | `12.7425` |

Interpretation of the Librispeech feature ladder:

- On this cleaner, more homogeneous workload, moving the synchronous path to GPU
  helped slightly more than it did on Vox, but it was still only a modest gain:
  `997.82`, about `+3.1%` over the plain sync baseline (`968.01`).
- Unlike Vox, the CPU async-microbatch row was already modestly beneficial here.
  `async MB + cpu` reached `1022.36`, about `+5.6%` over sync.
- The biggest gain again came from combining async microbatching with GPU
  preprocessing. That lifted `RTFx` from `1022.36` in the CPU-async row to
  `1250.32` in the plain GPU-async row, about `+22.3%` relative to the CPU-async
  control and `+29.2%` relative to sync.
- The fixed `batch_coordinator=1` setting that helped strongly on Vox did **not**
  transfer here. Turning it on actually reduced `RTFx` slightly from `1250.32`
  to `1239.33` (about `-0.9%` relative to the plain GPU-async row).
- `adaptive_drain=1` was effectively flat on top of that coordinator row
  (`1239.51`), while the fixed `multi_stage=1` configuration regressed more
  noticeably to `1193.60`.
- Backlog gating recovered part of the multistage loss, bringing the final row
  to `1214.57`, but that still remained below the simpler plain GPU-async row
  (`1250.32`).
- So for this Librispeech sorted single-chunk repro, most of the value came from
  `async MB + gpu`. The later coordinator / adaptive / multistage / backlog
  additions were not additive under the held-constant policy that worked well on
  Vox, which suggests those knobs are more workload-sensitive than the basic
  async+GPU win.

Artifacts:

- `tmp/asr_bottleneck/results/librispeech_bulk_feature_ladder_20260323/summary.json`
- `tmp/asr_bottleneck/results/librispeech_bulk_feature_ladder_20260323/`

## Exact Ladder Repro Commands

The Vox and Librispeech ladders above used the same per-row `vllm serve`
configuration and only changed the `bulk-bench.py` dataset arguments. To rerun
either ladder manually, first pick one row command from the list below, launch
that server, then run one of the two benchmark commands in a separate shell.

Common setup:

```shell
export MODEL_ID=/host/engines/vllm/audio/2b-release
export PORT=8037
cd /host/vllm-ekagra/vllm
```

Run the Vox ladder benchmark:

```shell
python3 bulk-bench.py \
  --base-url http://127.0.0.1:${PORT}/v1 \
  --api-key EMPTY \
  --model ${MODEL_ID} \
  --dataset voxpopuli \
  --hf-subset en \
  --hf-split test \
  --sorted \
  --single-chunk-only \
  --max-concurrent 64
```

Run the Librispeech ladder benchmark:

```shell
python3 bulk-bench.py \
  --base-url http://127.0.0.1:${PORT}/v1 \
  --api-key EMPTY \
  --model ${MODEL_ID} \
  --dataset librispeech_asr \
  --hf-split test.clean \
  --sorted \
  --single-chunk-only \
  --max-concurrent 64
```

Exact `vllm serve` command for each ladder row:

Row `0`: `sync baseline`

```shell
FLASHINFER_DISABLE_VERSION_CHECK=1 \
CUDA_VISIBLE_DEVICES=1 \
VLLM_COHERE_ASR_CROSS_REQUEST_AUDIO_MICROBATCH=0 \
vllm serve ${MODEL_ID} --trust-remote-code --port ${PORT}
```

Row `1`: `sync + gpu`

```shell
FLASHINFER_DISABLE_VERSION_CHECK=1 \
CUDA_VISIBLE_DEVICES=1 \
VLLM_COHERE_ASR_CROSS_REQUEST_AUDIO_MICROBATCH=0 \
COHERE_ASR_MICROBATCH_PREPROCESS_DEVICE=cuda \
HF_PROCESSOR_NUM_THREADS=1 \
vllm serve ${MODEL_ID} --trust-remote-code --port ${PORT}
```

Row `2`: `async MB + cpu`

```shell
FLASHINFER_DISABLE_VERSION_CHECK=1 \
CUDA_VISIBLE_DEVICES=1 \
VLLM_COHERE_ASR_CROSS_REQUEST_AUDIO_MICROBATCH=1 \
COHERE_ASR_MICROBATCH_PREPROCESS_DEVICE=cpu \
HF_PROCESSOR_NUM_THREADS=16 \
CROSS_REQUEST_AUDIO_MICROBATCH_NUM_WORKERS=8 \
CROSS_REQUEST_AUDIO_MICROBATCH_MAX_BATCH_SIZE=512 \
CROSS_REQUEST_AUDIO_MICROBATCH_MAX_TOTAL_TORCH_THREADS=64 \
CROSS_REQUEST_AUDIO_MICROBATCH_WAIT_TIMEOUT_S=0.002 \
VLLM_COHERE_ASR_CROSS_REQUEST_AUDIO_MICROBATCH_ADAPTIVE_DRAIN=0 \
VLLM_COHERE_ASR_CROSS_REQUEST_AUDIO_MICROBATCH_BATCH_COORDINATOR=0 \
VLLM_COHERE_ASR_CROSS_REQUEST_AUDIO_MICROBATCH_STATS=1 \
CROSS_REQUEST_AUDIO_MICROBATCH_STATS_LOG_EVERY_BATCHES=200 \
vllm serve ${MODEL_ID} --trust-remote-code --port ${PORT}
```

Row `3`: `async MB + gpu`

```shell
FLASHINFER_DISABLE_VERSION_CHECK=1 \
CUDA_VISIBLE_DEVICES=1 \
VLLM_COHERE_ASR_CROSS_REQUEST_AUDIO_MICROBATCH=1 \
COHERE_ASR_MICROBATCH_PREPROCESS_DEVICE=cuda \
HF_PROCESSOR_NUM_THREADS=1 \
CROSS_REQUEST_AUDIO_MICROBATCH_NUM_WORKERS=1 \
CROSS_REQUEST_AUDIO_MICROBATCH_MAX_BATCH_SIZE=512 \
CROSS_REQUEST_AUDIO_MICROBATCH_MAX_TOTAL_TORCH_THREADS=64 \
CROSS_REQUEST_AUDIO_MICROBATCH_WAIT_TIMEOUT_S=0.001 \
CROSS_REQUEST_AUDIO_MICROBATCH_DRAIN_ENTER_THRESHOLD=256 \
CROSS_REQUEST_AUDIO_MICROBATCH_DRAIN_EXIT_THRESHOLD=64 \
CROSS_REQUEST_AUDIO_MICROBATCH_DRAIN_TARGET_BATCH_SIZE=16 \
VLLM_COHERE_ASR_CROSS_REQUEST_AUDIO_MICROBATCH_BATCH_COORDINATOR=0 \
VLLM_COHERE_ASR_CROSS_REQUEST_AUDIO_MICROBATCH_ADAPTIVE_DRAIN=0 \
VLLM_COHERE_ASR_CROSS_REQUEST_AUDIO_MICROBATCH_MULTI_STAGE_PIPELINE=0 \
VLLM_COHERE_ASR_CROSS_REQUEST_AUDIO_MICROBATCH_STATS=1 \
CROSS_REQUEST_AUDIO_MICROBATCH_STATS_LOG_EVERY_BATCHES=200 \
vllm serve ${MODEL_ID} --trust-remote-code --port ${PORT}
```

Row `4`: `async MB + gpu + batch coordinator`

```shell
FLASHINFER_DISABLE_VERSION_CHECK=1 \
CUDA_VISIBLE_DEVICES=1 \
VLLM_COHERE_ASR_CROSS_REQUEST_AUDIO_MICROBATCH=1 \
COHERE_ASR_MICROBATCH_PREPROCESS_DEVICE=cuda \
HF_PROCESSOR_NUM_THREADS=1 \
CROSS_REQUEST_AUDIO_MICROBATCH_NUM_WORKERS=1 \
CROSS_REQUEST_AUDIO_MICROBATCH_MAX_BATCH_SIZE=512 \
CROSS_REQUEST_AUDIO_MICROBATCH_MAX_TOTAL_TORCH_THREADS=64 \
CROSS_REQUEST_AUDIO_MICROBATCH_WAIT_TIMEOUT_S=0.001 \
CROSS_REQUEST_AUDIO_MICROBATCH_DRAIN_ENTER_THRESHOLD=256 \
CROSS_REQUEST_AUDIO_MICROBATCH_DRAIN_EXIT_THRESHOLD=64 \
CROSS_REQUEST_AUDIO_MICROBATCH_DRAIN_TARGET_BATCH_SIZE=16 \
VLLM_COHERE_ASR_CROSS_REQUEST_AUDIO_MICROBATCH_BATCH_COORDINATOR=1 \
VLLM_COHERE_ASR_CROSS_REQUEST_AUDIO_MICROBATCH_ADAPTIVE_DRAIN=0 \
VLLM_COHERE_ASR_CROSS_REQUEST_AUDIO_MICROBATCH_MULTI_STAGE_PIPELINE=0 \
VLLM_COHERE_ASR_CROSS_REQUEST_AUDIO_MICROBATCH_STATS=1 \
CROSS_REQUEST_AUDIO_MICROBATCH_STATS_LOG_EVERY_BATCHES=200 \
vllm serve ${MODEL_ID} --trust-remote-code --port ${PORT}
```

Row `5`: `async MB + gpu + batch coordinator + adaptive`

```shell
FLASHINFER_DISABLE_VERSION_CHECK=1 \
CUDA_VISIBLE_DEVICES=1 \
VLLM_COHERE_ASR_CROSS_REQUEST_AUDIO_MICROBATCH=1 \
COHERE_ASR_MICROBATCH_PREPROCESS_DEVICE=cuda \
HF_PROCESSOR_NUM_THREADS=1 \
CROSS_REQUEST_AUDIO_MICROBATCH_NUM_WORKERS=1 \
CROSS_REQUEST_AUDIO_MICROBATCH_MAX_BATCH_SIZE=512 \
CROSS_REQUEST_AUDIO_MICROBATCH_MAX_TOTAL_TORCH_THREADS=64 \
CROSS_REQUEST_AUDIO_MICROBATCH_WAIT_TIMEOUT_S=0.001 \
CROSS_REQUEST_AUDIO_MICROBATCH_DRAIN_ENTER_THRESHOLD=256 \
CROSS_REQUEST_AUDIO_MICROBATCH_DRAIN_EXIT_THRESHOLD=64 \
CROSS_REQUEST_AUDIO_MICROBATCH_DRAIN_TARGET_BATCH_SIZE=16 \
VLLM_COHERE_ASR_CROSS_REQUEST_AUDIO_MICROBATCH_BATCH_COORDINATOR=1 \
VLLM_COHERE_ASR_CROSS_REQUEST_AUDIO_MICROBATCH_ADAPTIVE_DRAIN=1 \
VLLM_COHERE_ASR_CROSS_REQUEST_AUDIO_MICROBATCH_MULTI_STAGE_PIPELINE=0 \
VLLM_COHERE_ASR_CROSS_REQUEST_AUDIO_MICROBATCH_STATS=1 \
CROSS_REQUEST_AUDIO_MICROBATCH_STATS_LOG_EVERY_BATCHES=200 \
vllm serve ${MODEL_ID} --trust-remote-code --port ${PORT}
```

Row `6`: `async MB + gpu + batch coordinator + adaptive + multistage`

```shell
FLASHINFER_DISABLE_VERSION_CHECK=1 \
CUDA_VISIBLE_DEVICES=1 \
VLLM_COHERE_ASR_CROSS_REQUEST_AUDIO_MICROBATCH=1 \
COHERE_ASR_MICROBATCH_PREPROCESS_DEVICE=cuda \
HF_PROCESSOR_NUM_THREADS=1 \
CROSS_REQUEST_AUDIO_MICROBATCH_NUM_WORKERS=1 \
CROSS_REQUEST_AUDIO_MICROBATCH_MAX_BATCH_SIZE=512 \
CROSS_REQUEST_AUDIO_MICROBATCH_MAX_TOTAL_TORCH_THREADS=64 \
CROSS_REQUEST_AUDIO_MICROBATCH_WAIT_TIMEOUT_S=0.001 \
CROSS_REQUEST_AUDIO_MICROBATCH_DRAIN_ENTER_THRESHOLD=256 \
CROSS_REQUEST_AUDIO_MICROBATCH_DRAIN_EXIT_THRESHOLD=64 \
CROSS_REQUEST_AUDIO_MICROBATCH_DRAIN_TARGET_BATCH_SIZE=16 \
VLLM_COHERE_ASR_CROSS_REQUEST_AUDIO_MICROBATCH_BATCH_COORDINATOR=1 \
VLLM_COHERE_ASR_CROSS_REQUEST_AUDIO_MICROBATCH_ADAPTIVE_DRAIN=1 \
VLLM_COHERE_ASR_CROSS_REQUEST_AUDIO_MICROBATCH_MULTI_STAGE_PIPELINE=1 \
CROSS_REQUEST_AUDIO_MICROBATCH_PREP_NUM_WORKERS=24 \
VLLM_COHERE_ASR_CROSS_REQUEST_AUDIO_MICROBATCH_STATS=1 \
CROSS_REQUEST_AUDIO_MICROBATCH_STATS_LOG_EVERY_BATCHES=200 \
vllm serve ${MODEL_ID} --trust-remote-code --port ${PORT}
```

Row `7`: `async MB + gpu + batch coordinator + adaptive + multistage + backlog`

```shell
FLASHINFER_DISABLE_VERSION_CHECK=1 \
CUDA_VISIBLE_DEVICES=1 \
VLLM_COHERE_ASR_CROSS_REQUEST_AUDIO_MICROBATCH=1 \
COHERE_ASR_MICROBATCH_PREPROCESS_DEVICE=cuda \
HF_PROCESSOR_NUM_THREADS=1 \
CROSS_REQUEST_AUDIO_MICROBATCH_NUM_WORKERS=1 \
CROSS_REQUEST_AUDIO_MICROBATCH_MAX_BATCH_SIZE=512 \
CROSS_REQUEST_AUDIO_MICROBATCH_MAX_TOTAL_TORCH_THREADS=64 \
CROSS_REQUEST_AUDIO_MICROBATCH_WAIT_TIMEOUT_S=0.001 \
CROSS_REQUEST_AUDIO_MICROBATCH_DRAIN_ENTER_THRESHOLD=256 \
CROSS_REQUEST_AUDIO_MICROBATCH_DRAIN_EXIT_THRESHOLD=64 \
CROSS_REQUEST_AUDIO_MICROBATCH_DRAIN_TARGET_BATCH_SIZE=16 \
VLLM_COHERE_ASR_CROSS_REQUEST_AUDIO_MICROBATCH_BATCH_COORDINATOR=1 \
VLLM_COHERE_ASR_CROSS_REQUEST_AUDIO_MICROBATCH_ADAPTIVE_DRAIN=1 \
VLLM_COHERE_ASR_CROSS_REQUEST_AUDIO_MICROBATCH_MULTI_STAGE_PIPELINE=1 \
CROSS_REQUEST_AUDIO_MICROBATCH_PREP_NUM_WORKERS=24 \
VLLM_COHERE_ASR_CROSS_REQUEST_AUDIO_MICROBATCH_READY_BACKLOG_GATING=1 \
CROSS_REQUEST_AUDIO_MICROBATCH_READY_BACKLOG_ENTER_THRESHOLD=256 \
CROSS_REQUEST_AUDIO_MICROBATCH_READY_BACKLOG_EXIT_THRESHOLD=64 \
CROSS_REQUEST_AUDIO_MICROBATCH_READY_BACKLOG_TARGET=16 \
CROSS_REQUEST_AUDIO_MICROBATCH_READY_BACKLOG_MAX_WAIT_S=0.002 \
VLLM_COHERE_ASR_CROSS_REQUEST_AUDIO_MICROBATCH_STATS=1 \
CROSS_REQUEST_AUDIO_MICROBATCH_STATS_LOG_EVERY_BATCHES=200 \
vllm serve ${MODEL_ID} --trust-remote-code --port ${PORT}
```

Multi-stage prep-worker sub-stage sweep on current code (`4096` samples, compared to a fresh `sync_preproc_baseline` from the same artifact set):

| config | key settings | RTFx | delta vs sync | avg batch size | avg prep qwait (ms) | avg prepare (ms) | avg decode (ms) | avg resample (ms) | avg chunk (ms) | avg ready depth |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `sync_preproc_baseline` | async off | `771.59` | `0.0%` | `-` | `-` | `-` | `-` | `-` | `-` | `-` |
| `gpu_multistage_prep8` | `device=cuda`, `workers=1`, `prep_workers=8`, `enter=256`, `exit=64`, `target=16`, `wait=0.001` | `947.68` | `+22.8%` | `8.48` | `423.52` | `23.72` | `8.26` | `0.00` | `0.01` | `3.95` |
| `gpu_multistage_prep12` | `device=cuda`, `workers=1`, `prep_workers=12`, `enter=256`, `exit=64`, `target=16`, `wait=0.001` | `770.28` | `-0.2%` | `9.33` | `327.15` | `27.16` | `9.04` | `0.00` | `0.01` | `4.99` |
| `gpu_multistage_prep16` | `device=cuda`, `workers=1`, `prep_workers=16`, `enter=256`, `exit=64`, `target=16`, `wait=0.001` | `583.62` | `-24.4%` | `9.80` | `303.56` | `29.52` | `9.39` | `0.00` | `0.01` | `5.77` |
| `gpu_multistage_prep24` | `device=cuda`, `workers=1`, `prep_workers=24`, `enter=256`, `exit=64`, `target=16`, `wait=0.001` | `1032.49` | `+33.8%` | `10.32` | `247.65` | `33.28` | `11.51` | `0.00` | `0.01` | `7.87` |

Interpretation of the sub-stage sweep:

- Every request in this homogeneous `8s` workload had `resampled_requests=0`,
  `split_requests=0`, and `pyav_fallback_requests=0`.
- On this workload, the prep stage is therefore effectively dominated by audio
  decode plus executor/queue overhead; resampling and chunking are not the
  current bottlenecks.
- As `prep_workers` increased, `avg_decode_ms` rose from `8.26` to `11.51`,
  and total `avg_prepare_ms` rose from `23.72` to `33.28`. That is consistent
  with growing contention in the decode-heavy prep stage.
- At the same time, more prep parallelism increased `avg_ready_depth` from
  `3.95` to `7.87`, which means it did help feed the downstream GPU preprocessor
  more densely.
- The end-to-end `RTFx` ordering in this rerun was noisy, so this sweep is more
  useful for localizing the prep-stage cost than for declaring a final best
  `prep_workers` default. The stable signal is that, for this uniform `8s`
  dataset, upstream prep time is not being spent in resampling or chunking.

Artifacts:

- `tmp/asr_bottleneck/results/prep_worker_substage_sweep_20260323/sync_preproc_baseline.json`
- `tmp/asr_bottleneck/results/prep_worker_substage_sweep_20260323/bulk_summary.json`
- `tmp/asr_bottleneck/results/prep_worker_substage_sweep_20260323/`

What backlog-gated dispatch means on this code path:

- The multistage coordinator normally dispatches to the GPU preprocessor as
  soon as a worker slot is free and there is any ready work available.
- Backlog-gated dispatch changes that behavior when the overall pipeline backlog
  is already high: instead of dispatching immediately, the coordinator can wait
  briefly for the ready side to accumulate more items so the next realized batch
  is denser.
- Internally, the gate is driven by two related quantities:
  - `pipeline_backlog = prep_queue + ready_queue + pending_items`
  - `ready_backlog = ready_queue + pending_items`
- The gate turns on when `pipeline_backlog >= ready_backlog_enter_threshold`,
  turns off when `pipeline_backlog <= ready_backlog_exit_threshold`, and while
  active waits up to `ready_backlog_max_wait_s` for
  `ready_backlog >= ready_backlog_target`.
- This logic is only active when all three are enabled:
  `batch_coordinator`, `multi_stage_pipeline`, and
  `ready_backlog_gating`.

New env knobs added for backlog-gated dispatch:

- `VLLM_COHERE_ASR_CROSS_REQUEST_AUDIO_MICROBATCH_READY_BACKLOG_GATING`
  enables or disables the gate.
- `CROSS_REQUEST_AUDIO_MICROBATCH_READY_BACKLOG_ENTER_THRESHOLD` sets the
  `pipeline_backlog` level that enters gate mode.
- `CROSS_REQUEST_AUDIO_MICROBATCH_READY_BACKLOG_EXIT_THRESHOLD` sets the
  `pipeline_backlog` level that exits gate mode.
- `CROSS_REQUEST_AUDIO_MICROBATCH_READY_BACKLOG_TARGET` sets the desired
  `ready_backlog` before dispatch proceeds.
- `CROSS_REQUEST_AUDIO_MICROBATCH_READY_BACKLOG_MAX_WAIT_S` sets the maximum
  extra wait time used to build that `ready_backlog`.

Values used for the repeat-validated gated candidate:

- `VLLM_COHERE_ASR_CROSS_REQUEST_AUDIO_MICROBATCH_READY_BACKLOG_GATING=1`
- `CROSS_REQUEST_AUDIO_MICROBATCH_READY_BACKLOG_ENTER_THRESHOLD=256`
- `CROSS_REQUEST_AUDIO_MICROBATCH_READY_BACKLOG_EXIT_THRESHOLD=64`
- `CROSS_REQUEST_AUDIO_MICROBATCH_READY_BACKLOG_TARGET=64`
- `CROSS_REQUEST_AUDIO_MICROBATCH_READY_BACKLOG_MAX_WAIT_S=0.002`
- Base coordinator settings held constant around that experiment:
  `device=cuda`, `workers=1`, `prep_workers=24`, `enter=256`, `exit=64`,
  `target=64`, `wait=0.001`, `HF_PROCESSOR_NUM_THREADS=1`

Backlog-gated dispatch repeat check on current code (`4096` samples, reusing the fresh `sync_preproc_baseline=771.59` from the prep-worker sub-stage sweep):

| config | key settings | RTFx repeats | mean RTFx | stdev | delta vs sync |
| --- | --- | --- | ---: | ---: | ---: |
| `ungated_control` | `device=cuda`, `workers=1`, `prep_workers=24`, `enter=256`, `exit=64`, `target=64`, `wait=0.001`, `ready_backlog_gating=off` | `1168.05`, `1274.32`, `1259.89` | `1234.08` | `57.64` | `+59.9%` |
| `gated_rb64_wait0.002` | `device=cuda`, `workers=1`, `prep_workers=24`, `enter=256`, `exit=64`, `target=64`, `wait=0.001`, `ready_backlog_gating=on`, `rb_target=64`, `rb_wait=0.002` | `985.85`, `1314.79`, `1106.32` | `1135.66` | `166.42` | `+47.2%` |

Interpretation of the repeat check:

- A prior single run had made `rb_target=64`, `rb_wait=0.002` look like a new
  best candidate, but the repeat check did **not** confirm that result.
- The gated variant did build a deeper ready backlog and occasionally larger
  realized batches, but it was much noisier and slower on average than the
  ungated control.
- On mean `RTFx`, the gated repeat set came in about `-8.0%` below the ungated
  control (`1135.66` vs `1234.08`).
- The best stable candidate from this follow-up is therefore still the
  ungated `target=64` multistage configuration, not the backlog-gated variant.

Artifacts:

- `tmp/asr_bottleneck/results/backlog_gating_repeats_20260323/`

## What We Learned

- Cross-request batching is not automatically beneficial.
- The one-global-worker topology was the wrong shape for this model.
- Medium-sized CPU-side preprocessing batches for this model are not efficient
  enough to recover queueing overhead.
- A worker pool with a bounded total Torch-thread budget is materially better
  than one global worker for this workload.
- The best results so far came from device-aware topology choices:
  - multiple CPU workers for CPU preprocessing
  - one GPU preprocessor worker for GPU preprocessing
  - persistent preprocessors
  - a global thread cap
  - only then tuning the coordinator thresholds and drain target
- The benchmark arrival pattern matters materially:
  - in closed-loop `nick-bench.py`, `0.001s` was the best wait timeout
  - in open-loop `bulk-bench.py`, the best timeout depended on topology; for the
    tuned GPU-preproc coordinator, `0.001s` was better than `0.002s`
- The raw HF preprocessing path batches much more efficiently on GPU than CPU,
  but its direct per-item cost still bottoms out at moderate batch sizes rather
  than continuously improving up to `1024`.
- Moving preprocessing to GPU helped the gated coordinator far more at full
  `4096`-sample scale than the reduced `512`-sample search suggested, which
  implies that queue ingress density is strongly load-dependent.
- For the tuned GPU-preproc bulk path, reducing `HF_PROCESSOR_NUM_THREADS` from
  `16` to `1` improved throughput, which supports the idea that the GPU worker
  does not benefit from high CPU Torch thread counts and that keeping those CPU
  resources free helps the upstream request pipeline.
- On the tuned GPU-preproc bulk path, smaller drain targets (`16`) were more
  reliable than larger ones (`48`/`64`), while `32` showed the highest single
  run but also large rerun variance.
- In the multistage pipeline, backlog-gated dispatch (`rb_target=64`,
  `rb_wait=0.002`) increased ready backlog depth and occasionally produced
  larger batches, but repeat runs showed it was noisier and slower on average
  than the ungated `target=64` control, so it is not the current default
  candidate.
- On the real heterogeneous `facebook/voxpopuli` benchmark driven by
  `vllm bench serve`, the clean isolated rerun still flipped the synthetic
  `nick-bench.py` ordering: the best stable ranking was
  `sync_preproc_cpu_baseline_c512` > `multistage_backlog_gpu_c128` >
  `multistage_gpu_c128`, while the older single-stage GPU path never completed
  the full dataset even after a dedicated `c128` retry.
- Duration-aware bucketing may still help further on heterogeneous datasets, but
  it is no longer the first thing to try.

## Reproduction Checklist

If you want to rerun everything from scratch:

1. Enter the container and set up:

```bash
docker exec -it vllm-ekagra bash
export FLASHINFER_DISABLE_VERSION_CHECK=1
cd /host/vllm-ekagra/vllm
MODEL_ID=/host/engines/vllm/audio/2b-release
```

2. Run the correctness check:

```bash
VLLM_MAX_AUDIO_CLIP_FILESIZE_MB=50 \
VLLM_COHERE_ASR_CROSS_REQUEST_AUDIO_MICROBATCH=1 \
vllm serve "$MODEL_ID" --trust-remote-code
```

Then, in another shell:

```bash
AUDIO_PATH=~/.cache/vllm/assets/vllm_public_assets/mary_had_lamb.ogg
curl -s http://localhost:8000/v1/audio/transcriptions \
  -H "Authorization: Bearer ${VLLM_API_KEY:-EMPTY}" \
  -F "file=@${AUDIO_PATH}" \
  -F "model=${MODEL_ID}"
```

3. Run the isolated VoxPopuli sweep:

```bash
python3 tmp/asr_bottleneck/voxpopuli_concurrency_sweep.py \
  --port 8013 \
  --result-dir tmp/asr_bottleneck/results/voxpopuli_concurrency_sweep_20260323_clean
```

If another local server is already using that port, pick a different unused
port instead of `8013`.

4. Run the homogeneous async benchmark:

```bash
HF_PROCESSOR_NUM_THREADS=16 \
CROSS_REQUEST_AUDIO_MICROBATCH_MAX_BATCH_SIZE=64 \
VLLM_COHERE_ASR_CROSS_REQUEST_AUDIO_MICROBATCH=1 \
vllm serve "$MODEL_ID" --trust-remote-code
```

Then:

```bash
python3 nick-bench.py \
  --base-url http://localhost:8000/v1 \
  --api-key EMPTY \
  --max-concurrent 512 \
  --target-duration 8.0 \
  --use-random-dataset \
  --max-samples 4096 \
  --model "$MODEL_ID"
```

5. Run the homogeneous no-async control:

```bash
HF_PROCESSOR_NUM_THREADS=16 \
VLLM_COHERE_ASR_CROSS_REQUEST_AUDIO_MICROBATCH=0 \
vllm serve "$MODEL_ID" --trust-remote-code
```

Then run the same `nick-bench.py` command again.

6. Run the instrumented async benchmark:

```bash
HF_PROCESSOR_NUM_THREADS=16 \
CROSS_REQUEST_AUDIO_MICROBATCH_MAX_BATCH_SIZE=64 \
VLLM_COHERE_ASR_CROSS_REQUEST_AUDIO_MICROBATCH=1 \
VLLM_COHERE_ASR_CROSS_REQUEST_AUDIO_MICROBATCH_STATS=1 \
CROSS_REQUEST_AUDIO_MICROBATCH_STATS_LOG_EVERY_BATCHES=20 \
vllm serve "$MODEL_ID" --trust-remote-code \
  > tmp/asr_bottleneck/results/async_microbatch_stats_server.log 2>&1
```

Then:

```bash
python3 nick-bench.py \
  --base-url http://localhost:8000/v1 \
  --api-key EMPTY \
  --max-concurrent 512 \
  --target-duration 8.0 \
  --use-random-dataset \
  --max-samples 4096 \
  --model "$MODEL_ID" \
  > tmp/asr_bottleneck/results/async_microbatch_stats_nickbench.txt 2>&1
```

And inspect:

```bash
rg "Async audio microbatch stats|Final async audio microbatch stats" \
  tmp/asr_bottleneck/results/async_microbatch_stats_server.log
```

7. Run the focused multi-worker CPU benchmark:

```bash
HF_PROCESSOR_NUM_THREADS=16 \
CROSS_REQUEST_AUDIO_MICROBATCH_MAX_BATCH_SIZE=64 \
CROSS_REQUEST_AUDIO_MICROBATCH_NUM_WORKERS=8 \
CROSS_REQUEST_AUDIO_MICROBATCH_MAX_TOTAL_TORCH_THREADS=64 \
VLLM_COHERE_ASR_CROSS_REQUEST_AUDIO_MICROBATCH=1 \
vllm serve "$MODEL_ID" --trust-remote-code
```

Then:

```bash
python3 nick-bench.py \
  --base-url http://localhost:8000/v1 \
  --api-key EMPTY \
  --max-concurrent 512 \
  --target-duration 8.0 \
  --use-random-dataset \
  --max-samples 4096 \
  --model "$MODEL_ID"
```

8. Run the best current closed-loop uniform configuration:

```bash
COHERE_ASR_MICROBATCH_PREPROCESS_DEVICE=cuda \
HF_PROCESSOR_NUM_THREADS=16 \
CROSS_REQUEST_AUDIO_MICROBATCH_MAX_BATCH_SIZE=512 \
CROSS_REQUEST_AUDIO_MICROBATCH_NUM_WORKERS=1 \
CROSS_REQUEST_AUDIO_MICROBATCH_MAX_TOTAL_TORCH_THREADS=64 \
CROSS_REQUEST_AUDIO_MICROBATCH_DRAIN_ENTER_THRESHOLD=256 \
CROSS_REQUEST_AUDIO_MICROBATCH_DRAIN_EXIT_THRESHOLD=64 \
CROSS_REQUEST_AUDIO_MICROBATCH_DRAIN_TARGET_BATCH_SIZE=16 \
CROSS_REQUEST_AUDIO_MICROBATCH_WAIT_TIMEOUT_S=0.001 \
VLLM_COHERE_ASR_CROSS_REQUEST_AUDIO_MICROBATCH=1 \
VLLM_COHERE_ASR_CROSS_REQUEST_AUDIO_MICROBATCH_ADAPTIVE_DRAIN=1 \
vllm serve "$MODEL_ID" --trust-remote-code
```

Then run the same `nick-bench.py` command again.

9. Run the CPU wait-timeout sweep on the best CPU topology:

```bash
python3 bench_asr_wait_timeout_sweep.py
```

This writes results to:

- `tmp/asr_bottleneck/results/wait_timeout_sweep/summary.tsv`
- `tmp/asr_bottleneck/results/wait_timeout_sweep/summary.json`

10. Run the open-loop bulk benchmark on the best measured bulk configuration:

```bash
COHERE_ASR_MICROBATCH_PREPROCESS_DEVICE=cuda \
HF_PROCESSOR_NUM_THREADS=1 \
CROSS_REQUEST_AUDIO_MICROBATCH_MAX_BATCH_SIZE=512 \
CROSS_REQUEST_AUDIO_MICROBATCH_NUM_WORKERS=1 \
CROSS_REQUEST_AUDIO_MICROBATCH_MAX_TOTAL_TORCH_THREADS=64 \
CROSS_REQUEST_AUDIO_MICROBATCH_DRAIN_ENTER_THRESHOLD=256 \
CROSS_REQUEST_AUDIO_MICROBATCH_DRAIN_EXIT_THRESHOLD=64 \
CROSS_REQUEST_AUDIO_MICROBATCH_DRAIN_TARGET_BATCH_SIZE=16 \
CROSS_REQUEST_AUDIO_MICROBATCH_WAIT_TIMEOUT_S=0.001 \
VLLM_COHERE_ASR_CROSS_REQUEST_AUDIO_MICROBATCH=1 \
VLLM_COHERE_ASR_CROSS_REQUEST_AUDIO_MICROBATCH_ADAPTIVE_DRAIN=1 \
vllm serve "$MODEL_ID" --trust-remote-code
```

Then:

```bash
python3 bulk-bench.py \
  --base-url http://localhost:8000/v1 \
  --api-key EMPTY \
  --max-concurrent 512 \
  --target-duration 8.0 \
  --use-random-dataset \
  --max-samples 4096 \
  --model "$MODEL_ID"
```

Bulk ablation summaries live in:

- `tmp/asr_bottleneck/results/bulk_bench_ablations/summary.tsv`
- `tmp/asr_bottleneck/results/bulk_bench_ablations/summary.json`

11. Run the direct HF preprocessing device benchmark:

```bash
python3 bench_cohere_asr_preproc_device.py
```

This writes results to:

- `tmp/asr_bottleneck/results/cohere_preproc_device_bench/summary.csv`
- `tmp/asr_bottleneck/results/cohere_preproc_device_bench/summary.json`
- `tmp/asr_bottleneck/results/cohere_preproc_device_bench/cpu_per_item_cost.png`
- `tmp/asr_bottleneck/results/cohere_preproc_device_bench/cuda_per_item_cost.png`

12. Run the coordinator CPU-vs-GPU preprocessing sweep:

```bash
python3 bench_asr_coordinator_policy_sweep.py \
  --benchmark bulk \
  --preproc-device cuda \
  --workers 1 \
  --hf-threads 16 \
  --max-total-torch-threads 64 \
  --enter-thresholds 256 \
  --exit-thresholds 64 \
  --drain-targets 16 32 48 64 \
  --timeouts 0.001 \
  --max-concurrent 128 \
  --max-samples 512 \
  --result-dir tmp/asr_bottleneck/results/coordinator_policy_sweep/device_bulk_reduced_gpu
```

Then rerun with:

- `--benchmark nick`
- `--preproc-device cpu --workers 8 --drain-targets 48` for the CPU control
- `--max-concurrent 512 --max-samples 4096` for full validation

Coordinator device-sweep artifacts live in:

- `tmp/asr_bottleneck/results/coordinator_policy_sweep/device_bulk_reduced_cpu`
- `tmp/asr_bottleneck/results/coordinator_policy_sweep/device_bulk_reduced_gpu`
- `tmp/asr_bottleneck/results/coordinator_policy_sweep/device_nick_reduced_cpu`
- `tmp/asr_bottleneck/results/coordinator_policy_sweep/device_nick_reduced_gpu`
- `tmp/asr_bottleneck/results/coordinator_policy_sweep/device_bulk_validation_cpu`
- `tmp/asr_bottleneck/results/coordinator_policy_sweep/device_bulk_validation_gpu`
- `tmp/asr_bottleneck/results/coordinator_policy_sweep/device_nick_validation_cpu`
- `tmp/asr_bottleneck/results/coordinator_policy_sweep/device_nick_validation_gpu`
