# Cohere ASR Attention Mode Selection

## Summary

This note records a direct comparison of the three Cohere ASR encoder attention modes now available behind `VLLM_COHERE_ASR_ENCODER_ATTENTION_MODE`:

- `manual`
- `sdpa`
- `triton`

The conclusion is straightforward: for the current `2b-release` production workload, `sdpa` is the best default. It wins on both the bounded worker-local profile and the full VoxPopuli EN concurrency-512 serve benchmark, so the model-local default has been changed from `triton` to `sdpa`.

## Why This Comparison Was Run

The encoder had already gone through three generations of optimization:

1. manual relative-position attention
2. SDPA fallback
3. Triton relative-position kernel

Earlier work suggested that some encoder wins were real in isolation but did not always improve end-to-end `RTFx`. This comparison was run to answer two questions with current artifacts instead of historical assumptions:

1. Which attention mode is actually best for the encoder on the current model shape?
2. Does the worker-local win also improve the real serve workload?

## Methodology

Two measurements were collected for each mode.

### 1. Bounded worker-only profile

- server in dev mode with `torch` profiler
- worker-focused profile with `ignore_frontend: true`
- `nick-bench.py` random-audio workload
- `target-duration 20.0`
- `max-concurrent 8`

Artifacts:

- `tmp/vllm_profiles/attention_modes/<mode>_worker20/`
- `tmp/asr_bottleneck/attention_modes/<mode>_worker20_stats.json`

### 2. Full serve workload

- `vllm serve /host/engines/vllm/audio/2b-release --trust-remote-code`
- VoxPopuli EN
- `vllm bench serve --backend openai-audio ... --max-concurrency 512`

Artifacts:

- `tmp/asr_bottleneck/results/attention_modes/<mode>_c512_devmode.json`
- `tmp/asr_bottleneck/mmstats/attention_modes/<mode>_c512_devmode.json`

## Exact Reproduction Commands

All commands below were run inside the provided `vllm-ekagra` container, with `FLASHINFER_DISABLE_VERSION_CHECK=1`, from `/host/vllm-ekagra/vllm`.

### Directory setup

```bash
docker exec vllm-ekagra bash -lc '
  export FLASHINFER_DISABLE_VERSION_CHECK=1 &&
  cd /host/vllm-ekagra/vllm &&
  mkdir -p \
    tmp/vllm_profiles/attention_modes \
    tmp/asr_bottleneck/attention_modes \
    tmp/asr_bottleneck/results/attention_modes \
    tmp/asr_bottleneck/mmstats/attention_modes
'
```

### 1. Bounded worker-only profile

Repeat the following sequence for each mode in `manual`, `sdpa`, and `triton`.

Start the server:

```bash
docker exec vllm-ekagra bash -lc '
  export FLASHINFER_DISABLE_VERSION_CHECK=1 \
    VLLM_SERVER_DEV_MODE=1 \
    VLLM_COHERE_ASR_ENCODER_ATTENTION_MODE=<mode> \
    MODEL_ID=/host/engines/vllm/audio/2b-release &&
  cd /host/vllm-ekagra/vllm &&
  CFG=$(python3 -c "import json; print(json.dumps({
    \"profiler\": \"torch\",
    \"torch_profiler_dir\": \"/host/vllm-ekagra/vllm/tmp/vllm_profiles/attention_modes/<mode>_worker20\",
    \"ignore_frontend\": True,
    \"wait_iterations\": 1,
    \"warmup_iterations\": 1,
    \"active_iterations\": 4
  }))") &&
  vllm serve "$MODEL_ID" --trust-remote-code
'
```

Wait for health:

```bash
docker exec vllm-ekagra bash -lc '
  export FLASHINFER_DISABLE_VERSION_CHECK=1 &&
  for i in $(seq 1 120); do
    curl -sf http://localhost:8000/health >/dev/null && exit 0
    sleep 2
  done
  exit 1
'
```

Run the bounded profiling load and save timing stats:

```bash
docker exec vllm-ekagra bash -lc '
  export FLASHINFER_DISABLE_VERSION_CHECK=1 \
    MODEL_ID=/host/engines/vllm/audio/2b-release &&
  cd /host/vllm-ekagra/vllm &&
  python3 nick-bench.py \
    --base-url http://localhost:8000/v1 \
    --api-key EMPTY \
    --max-concurrent 8 \
    --target-duration 20.0 \
    --use-random-dataset \
    --max-samples 16 \
    --model "$MODEL_ID" \
    --profile \
    > tmp/asr_bottleneck/attention_modes/<mode>_worker20_nickbench.txt 2>&1 &&
  curl -s -X POST http://localhost:8000/debug/asr_bottleneck_stats \
    > tmp/asr_bottleneck/attention_modes/<mode>_worker20_stats.json
'
```

Stop the server:

```bash
docker exec vllm-ekagra bash -lc '
  export FLASHINFER_DISABLE_VERSION_CHECK=1 &&
  pkill -f "vllm serve /host/engines/vllm/audio/2b-release" || true
'
```

### 2. Full VoxPopuli EN concurrency-512 benchmark

Repeat the following sequence for each mode in `manual`, `sdpa`, and `triton`.

Start the server:

```bash
docker exec vllm-ekagra bash -lc '
  export FLASHINFER_DISABLE_VERSION_CHECK=1 \
    VLLM_SERVER_DEV_MODE=1 \
    VLLM_MAX_AUDIO_CLIP_FILESIZE_MB=50 \
    VLLM_COHERE_ASR_ENCODER_ATTENTION_MODE=<mode> \
    MODEL_ID=/host/engines/vllm/audio/2b-release &&
  cd /host/vllm-ekagra/vllm &&
  vllm serve "$MODEL_ID" --trust-remote-code
'
```

Wait for health:

```bash
docker exec vllm-ekagra bash -lc '
  export FLASHINFER_DISABLE_VERSION_CHECK=1 &&
  for i in $(seq 1 300); do
    curl -sf http://localhost:8000/health >/dev/null && exit 0
    sleep 2
  done
  exit 1
'
```

Run the full benchmark and save timing stats:

```bash
docker exec vllm-ekagra bash -lc '
  export FLASHINFER_DISABLE_VERSION_CHECK=1 \
    MODEL_ID=/host/engines/vllm/audio/2b-release \
    MAX_CONCURRENCY=512 &&
  cd /host/vllm-ekagra/vllm &&
  set -o pipefail &&
  time vllm bench serve \
    --backend openai-audio \
    --dataset-name hf \
    --dataset-path facebook/voxpopuli \
    --hf-subset en \
    --hf-split test \
    --no-stream \
    --trust-remote-code \
    --model "$MODEL_ID" \
    --num-prompts 99999999 \
    --no-oversample \
    --endpoint /v1/audio/transcriptions \
    --ready-check-timeout-sec 600 \
    --save-result \
    --result-dir tmp/asr_bottleneck/results/attention_modes \
    --result-filename <mode>_c512_devmode.json \
    --max-concurrency "$MAX_CONCURRENCY" \
    2>&1 | tee tmp/asr_bottleneck/results/attention_modes/<mode>_c512_devmode.txt &&
  curl -s -X POST http://localhost:8000/debug/asr_bottleneck_stats \
    > tmp/asr_bottleneck/mmstats/attention_modes/<mode>_c512_devmode.json
'
```

Stop the server:

```bash
docker exec vllm-ekagra bash -lc '
  export FLASHINFER_DISABLE_VERSION_CHECK=1 &&
  pkill -f "vllm serve /host/engines/vllm/audio/2b-release" || true
'
```

## Results

### Worker-Only Encoder Timing

| Mode | `encoder_forward_secs` avg/request | `encoder_forward_batch_secs` total | `encoder_forward_batch_secs` avg/batch |
| --- | ---: | ---: | ---: |
| `manual` | `19.62 ms` | `0.914 s` | `91.40 ms` |
| `sdpa` | `16.45 ms` | `0.810 s` | `90.02 ms` |
| `triton` | `23.62 ms` | `1.379 s` | `125.37 ms` |

Takeaway:

- `sdpa` is the best worker-local mode.
- `manual` is slower than `sdpa`.
- `triton` is the slowest of the three on the current production-relevant shape.

### Worker CUDA Kernel Mix

This bounded profile also helps explain why `triton` loses here.

| Mode | Self CUDA total | Attention self CUDA share | `aten::addmm` self CUDA share | `aten::cudnn_convolution` self CUDA share |
| --- | ---: | ---: | ---: | ---: |
| `manual` | `40.402 ms` | `8.67%` | `22.08%` | `9.33%` |
| `sdpa` | `38.035 ms` | `4.21%` | `23.38%` | `9.80%` |
| `triton` | `45.079 ms` | `29.15%` | `19.90%` | `8.39%` |

Takeaway:

- In this shape regime, `sdpa` keeps attention relatively cheap.
- The Triton relpos kernel is still a large CUDA consumer, but it is not cheaper than the SDPA path overall.
- The relpos kernel itself is not enough to produce a net win on this workload.

### Full VoxPopuli EN Concurrency-512 Benchmark

| Mode | Req/s | Tok/s | `RTFx` | Mean `TTFT` |
| --- | ---: | ---: | ---: | ---: |
| `manual` | `92.81` | `2978.79` | `894.18` | `3365.43 ms` |
| `sdpa` | `95.18` | `3055.17` | `917.06` | `3057.00 ms` |
| `triton` | `93.83` | `3011.51` | `904.06` | `3071.49 ms` |

Takeaway:

- `sdpa` is also the best end-to-end mode.
- Relative to `manual`, `sdpa` improves:
  - req/s by about `2.6%`
  - tok/s by about `2.6%`
  - `RTFx` by about `2.6%`
  - mean `TTFT` by about `9.2%`
- `triton` lands between `manual` and `sdpa` on this run, but still loses to `sdpa`.

## Why The E2E Win Is Smaller Than The Worker Win

The encoder did improve with `sdpa`, but the full system did not become encoder-attention-only.

Full-run timing snapshots:

| Mode | `process_for_engine_secs` total | `encoder_forward_batch_secs` total | `decoder_forward_secs` total | `audio_decode_resample_secs` total |
| --- | ---: | ---: | ---: | ---: |
| `manual` | `8.83 s` | `13.40 s` | `2.62 s` | `1.44 s` |
| `sdpa` | `8.28 s` | `12.39 s` | `2.62 s` | `1.42 s` |
| `triton` | `8.54 s` | `13.93 s` | `2.28 s` | `1.49 s` |

The main implication is:

- `sdpa` reduces encoder cost meaningfully
- frontend preprocessing remains large
- decoder and other worker kernels still matter
- at concurrency `512`, the workload is saturated enough that a worker-only win is diluted unless it is large enough to move the dominant system bottleneck

So encoder attention still matters, but it is no longer the only active bottleneck. That is why the worker-local win is larger than the final `RTFx` gain.

## Recommendation

Use `sdpa` as the Cohere ASR default attention mode for the current production model and workload.

Rationale:

1. It is the fastest mode in the bounded worker profile.
2. It is also the fastest mode on the real concurrency-512 serve benchmark.
3. It improves both throughput and `TTFT`, so this is not a worker-only microbenchmark win.
4. The Triton relpos kernel is not competitive enough for the current `head_dim=160` production case.

Operational note:

- Keep `VLLM_COHERE_ASR_ENCODER_ATTENTION_MODE` as an override for future experiments.
- If Triton relpos is revisited later, it should only become the default again after it beats `sdpa` on both worker-local timing and the full serve benchmark.
