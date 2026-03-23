# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import io   
import os
import math
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from collections.abc import AsyncGenerator, Callable
from functools import cached_property, partial
from typing import Any, Final, Literal, TypeAlias, TypeVar, cast
from vllm.utils.torch_utils import set_default_torch_num_threads

import numpy as np
from fastapi import Request
from transformers import PreTrainedTokenizerBase

import vllm.envs as envs
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.logger import RequestLogger
from vllm.entrypoints.openai.engine.protocol import (
    DeltaMessage,
    ErrorResponse,
    RequestResponseMetadata,
    UsageInfo,
)
from vllm.entrypoints.openai.engine.serving import OpenAIServing, SpeechToTextRequest
from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.entrypoints.openai.speech_to_text.protocol import (
    TranscriptionResponse,
    TranscriptionResponseStreamChoice,
    TranscriptionResponseVerbose,
    TranscriptionSegment,
    TranscriptionStreamResponse,
    TranslationResponse,
    TranslationResponseStreamChoice,
    TranslationResponseVerbose,
    TranslationSegment,
    TranslationStreamResponse,
)
from vllm.entrypoints.utils import get_max_tokens
from vllm.exceptions import VLLMValidationError
from vllm.inputs import EncoderDecoderInputs, ProcessorInputs
from vllm.logger import init_logger
from vllm.logprobs import FlatLogprobs, Logprob
from vllm.model_executor.models import SupportsTranscription
from vllm.multimodal.audio import split_audio
from vllm.multimodal.media.audio import extract_audio_from_video_bytes
from vllm.outputs import RequestOutput
from vllm.renderers.inputs import DictPrompt, EncoderDecoderDictPrompt
from vllm.renderers.inputs.preprocess import parse_enc_dec_prompt, parse_model_prompt
from vllm.sampling_params import BeamSearchParams, SamplingParams
from vllm.tokenizers import get_tokenizer
from vllm.utils.import_utils import PlaceholderModule

try:
    import librosa
except ImportError:
    librosa = PlaceholderModule("librosa")  # type: ignore[assignment]

try:
    import soundfile as sf
except ImportError:
    sf = PlaceholderModule("soundfile")  # type: ignore[assignment]

# Public libsndfile error codes exposed via `soundfile.LibsndfileError.code`, soundfile
# being librosa's main backend. Used to validate if an audio loading error is due to a
# server error vs a client error (invalid audio file).
# 1 = unrecognised format      (file is not a supported audio container)
# 3 = malformed file           (corrupt or structurally invalid audio)
# 4 = unsupported encoding     (codec not supported by this libsndfile build)
_BAD_SF_CODES = {1, 3, 4}

SpeechToTextResponse: TypeAlias = TranscriptionResponse | TranslationResponse
SpeechToTextResponseVerbose: TypeAlias = (
    TranscriptionResponseVerbose | TranslationResponseVerbose
)
SpeechToTextSegment: TypeAlias = TranscriptionSegment | TranslationSegment
T = TypeVar("T", bound=SpeechToTextResponse)
V = TypeVar("V", bound=SpeechToTextResponseVerbose)
S = TypeVar("S", bound=SpeechToTextSegment)

ResponseType: TypeAlias = (
    TranscriptionResponse
    | TranslationResponse
    | TranscriptionResponseVerbose
    | TranslationResponseVerbose
)

logger = init_logger(__name__)


def _env_flag(name: str, default: bool) -> bool:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default

    return raw_value.strip().lower() not in {"0", "false", "no", "off", ""}


HF_PROCESSOR_NUM_THREADS = int(os.environ.get("HF_PROCESSOR_NUM_THREADS", 1))        
_CROSS_REQUEST_AUDIO_MICROBATCH_MAX_BATCH_SIZE = int(os.environ.get("CROSS_REQUEST_AUDIO_MICROBATCH_MAX_BATCH_SIZE", 16))
_CROSS_REQUEST_AUDIO_MICROBATCH_WAIT_TIMEOUT_S = float(
    os.environ.get("CROSS_REQUEST_AUDIO_MICROBATCH_WAIT_TIMEOUT_S", 0.002)
)
_CROSS_REQUEST_AUDIO_MICROBATCH_NUM_WORKERS = int(
    os.environ.get("CROSS_REQUEST_AUDIO_MICROBATCH_NUM_WORKERS", 4)
)
_CROSS_REQUEST_AUDIO_MICROBATCH_MAX_TOTAL_TORCH_THREADS = int(
    os.environ.get("CROSS_REQUEST_AUDIO_MICROBATCH_MAX_TOTAL_TORCH_THREADS", 64)
)
_ENABLE_ADAPTIVE_CROSS_REQUEST_AUDIO_MICROBATCH_DRAIN = _env_flag(
    "VLLM_COHERE_ASR_CROSS_REQUEST_AUDIO_MICROBATCH_ADAPTIVE_DRAIN",
    False,
)
_ENABLE_CROSS_REQUEST_AUDIO_MICROBATCH_BATCH_COORDINATOR = _env_flag(
    "VLLM_COHERE_ASR_CROSS_REQUEST_AUDIO_MICROBATCH_BATCH_COORDINATOR",
    True,
)
_CROSS_REQUEST_AUDIO_MICROBATCH_DRAIN_ENTER_THRESHOLD = int(
    os.environ.get("CROSS_REQUEST_AUDIO_MICROBATCH_DRAIN_ENTER_THRESHOLD", 0)
)
_CROSS_REQUEST_AUDIO_MICROBATCH_DRAIN_EXIT_THRESHOLD = int(
    os.environ.get("CROSS_REQUEST_AUDIO_MICROBATCH_DRAIN_EXIT_THRESHOLD", 0)
)
_CROSS_REQUEST_AUDIO_MICROBATCH_DRAIN_TARGET_BATCH_SIZE = int(
    os.environ.get("CROSS_REQUEST_AUDIO_MICROBATCH_DRAIN_TARGET_BATCH_SIZE", 0)
)


def _resolve_async_audio_microbatch_topology(
    *,
    requested_num_workers: int,
    requested_threads_per_worker: int,
    max_total_torch_threads: int,
) -> tuple[int, int]:
    effective_max_total_torch_threads = max(1, max_total_torch_threads)
    effective_num_workers = max(
        1,
        min(requested_num_workers, effective_max_total_torch_threads),
    )
    threads_per_worker = max(
        1,
        min(
            requested_threads_per_worker,
            effective_max_total_torch_threads // effective_num_workers,
        ),
    )
    return effective_num_workers, threads_per_worker


_ENABLE_CROSS_REQUEST_AUDIO_MICROBATCH = _env_flag(
    "VLLM_COHERE_ASR_CROSS_REQUEST_AUDIO_MICROBATCH",
    True,
)
_ENABLE_CROSS_REQUEST_AUDIO_MICROBATCH_STATS = _env_flag(
    "VLLM_COHERE_ASR_CROSS_REQUEST_AUDIO_MICROBATCH_STATS",
    False,
)
_CROSS_REQUEST_AUDIO_MICROBATCH_STATS_LOG_EVERY_BATCHES = int(
    os.environ.get("CROSS_REQUEST_AUDIO_MICROBATCH_STATS_LOG_EVERY_BATCHES", 100)
)


class _AsyncMicrobatchStats:
    def __init__(
        self,
        *,
        max_batch_size: int,
        log_every_batches: int,
    ) -> None:
        self.max_batch_size = max_batch_size
        self.log_every_batches = max(1, log_every_batches)

        array_size = max_batch_size + 1
        self.batch_counts_by_size = [0] * array_size
        self.item_counts_by_size = [0] * array_size
        self.queue_wait_secs_by_size = [0.0] * array_size
        self.preprocess_secs_by_size = [0.0] * array_size
        self.max_queue_wait_secs_by_size = [0.0] * array_size
        self.max_preprocess_secs_by_size = [0.0] * array_size

        self.total_batches = 0
        self.total_items = 0
        self.total_queue_wait_secs = 0.0
        self.total_preprocess_secs = 0.0
        self.max_queue_wait_secs = 0.0
        self.max_preprocess_secs = 0.0

    def record(
        self,
        *,
        batch_size: int,
        total_queue_wait_secs: float,
        max_queue_wait_secs: float,
        preprocess_secs: float,
    ) -> None:
        self.total_batches += 1
        self.total_items += batch_size
        self.total_queue_wait_secs += total_queue_wait_secs
        self.total_preprocess_secs += preprocess_secs
        self.max_queue_wait_secs = max(self.max_queue_wait_secs, max_queue_wait_secs)
        self.max_preprocess_secs = max(self.max_preprocess_secs, preprocess_secs)

        self.batch_counts_by_size[batch_size] += 1
        self.item_counts_by_size[batch_size] += batch_size
        self.queue_wait_secs_by_size[batch_size] += total_queue_wait_secs
        self.preprocess_secs_by_size[batch_size] += preprocess_secs
        self.max_queue_wait_secs_by_size[batch_size] = max(
            self.max_queue_wait_secs_by_size[batch_size],
            max_queue_wait_secs,
        )
        self.max_preprocess_secs_by_size[batch_size] = max(
            self.max_preprocess_secs_by_size[batch_size],
            preprocess_secs,
        )

    def _format_bucket_summary(self) -> str:
        parts = list[str]()
        for batch_size in range(1, self.max_batch_size + 1):
            batch_count = self.batch_counts_by_size[batch_size]
            if batch_count == 0:
                continue

            item_count = self.item_counts_by_size[batch_size]
            avg_queue_wait_ms = (
                self.queue_wait_secs_by_size[batch_size] * 1000 / item_count
            )
            avg_preprocess_ms = (
                self.preprocess_secs_by_size[batch_size] * 1000 / batch_count
            )
            max_queue_wait_ms = self.max_queue_wait_secs_by_size[batch_size] * 1000
            max_preprocess_ms = self.max_preprocess_secs_by_size[batch_size] * 1000
            parts.append(
                "size=%d batches=%d avg_pre_ms=%.2f avg_qwait_ms=%.2f "
                "max_pre_ms=%.2f max_qwait_ms=%.2f"
                % (
                    batch_size,
                    batch_count,
                    avg_preprocess_ms,
                    avg_queue_wait_ms,
                    max_preprocess_ms,
                    max_queue_wait_ms,
                )
            )

        return "; ".join(parts)

    def summary(self) -> str:
        avg_batch_size = (
            self.total_items / self.total_batches if self.total_batches > 0 else 0.0
        )
        avg_queue_wait_ms = (
            self.total_queue_wait_secs * 1000 / self.total_items
            if self.total_items > 0
            else 0.0
        )
        avg_preprocess_ms = (
            self.total_preprocess_secs * 1000 / self.total_batches
            if self.total_batches > 0
            else 0.0
        )

        return (
            "batches=%d items=%d avg_batch_size=%.2f avg_qwait_ms=%.2f "
            "avg_pre_ms=%.2f max_qwait_ms=%.2f max_pre_ms=%.2f | %s"
            % (
                self.total_batches,
                self.total_items,
                avg_batch_size,
                avg_queue_wait_ms,
                avg_preprocess_ms,
                self.max_queue_wait_secs * 1000,
                self.max_preprocess_secs * 1000,
                self._format_bucket_summary(),
            )
        )

    def maybe_log(self, *, prefix: str = "Async audio microbatch stats") -> None:
        if self.total_batches % self.log_every_batches == 0:
            logger.info("%s: %s", prefix, self.summary())

    def reset(self) -> str | None:
        previous_summary = self.summary() if self.total_batches > 0 else None

        array_size = self.max_batch_size + 1
        self.batch_counts_by_size = [0] * array_size
        self.item_counts_by_size = [0] * array_size
        self.queue_wait_secs_by_size = [0.0] * array_size
        self.preprocess_secs_by_size = [0.0] * array_size
        self.max_queue_wait_secs_by_size = [0.0] * array_size
        self.max_preprocess_secs_by_size = [0.0] * array_size

        self.total_batches = 0
        self.total_items = 0
        self.total_queue_wait_secs = 0.0
        self.total_preprocess_secs = 0.0
        self.max_queue_wait_secs = 0.0
        self.max_preprocess_secs = 0.0

        return previous_summary

class AsyncMicrobatchAudioPreprocessor:
    """Microbatch single-clip requests across concurrent calls."""

    def __init__(
        self,
        preprocess_fn: Callable[[list[np.ndarray], object], list[dict[str, Any]]],
        model_config: object,
        *,
        preprocess_factory: (
            Callable[[object], Callable[[list[np.ndarray]], list[dict[str, Any]]]] | None
        ) = None,
        num_workers: int = _CROSS_REQUEST_AUDIO_MICROBATCH_NUM_WORKERS,
        num_threads: int = HF_PROCESSOR_NUM_THREADS,
        max_batch_size: int = _CROSS_REQUEST_AUDIO_MICROBATCH_MAX_BATCH_SIZE,
        batch_wait_timeout_s: float = _CROSS_REQUEST_AUDIO_MICROBATCH_WAIT_TIMEOUT_S,
        max_total_torch_threads: int = (
            _CROSS_REQUEST_AUDIO_MICROBATCH_MAX_TOTAL_TORCH_THREADS
        ),
    ) -> None:
        self.preprocess_fn = preprocess_fn
        self.model_config = model_config
        self.preprocess_factory = preprocess_factory
        self.requested_num_workers = max(1, num_workers)
        self.requested_num_threads = max(1, num_threads)
        self.max_total_torch_threads = max(1, max_total_torch_threads)
        self.num_workers, self.num_threads = _resolve_async_audio_microbatch_topology(
            requested_num_workers=self.requested_num_workers,
            requested_threads_per_worker=self.requested_num_threads,
            max_total_torch_threads=self.max_total_torch_threads,
        )
        self.max_batch_size = max_batch_size
        self.batch_wait_timeout_s = batch_wait_timeout_s
        self.adaptive_drain_enabled = (
            _ENABLE_ADAPTIVE_CROSS_REQUEST_AUDIO_MICROBATCH_DRAIN
        )
        self.batch_coordinator_enabled = (
            _ENABLE_CROSS_REQUEST_AUDIO_MICROBATCH_BATCH_COORDINATOR
        )
        default_drain_target_batch_size = max(
            1,
            (self.max_batch_size + self.num_workers - 1) // self.num_workers,
        )
        self.drain_target_batch_size = max(
            1,
            _CROSS_REQUEST_AUDIO_MICROBATCH_DRAIN_TARGET_BATCH_SIZE
            or default_drain_target_batch_size,
        )
        self.drain_exit_threshold = max(
            1,
            _CROSS_REQUEST_AUDIO_MICROBATCH_DRAIN_EXIT_THRESHOLD
            or self.drain_target_batch_size,
        )
        self.drain_enter_threshold = max(
            self.drain_exit_threshold + 1,
            _CROSS_REQUEST_AUDIO_MICROBATCH_DRAIN_ENTER_THRESHOLD
            or self.max_batch_size,
        )
        self._drain_mode = False
        self._worker_preprocess_fns: list[
            Callable[[list[np.ndarray]], list[dict[str, Any]]] | None
        ] = [None] * self.num_workers
        self._stats = (
            _AsyncMicrobatchStats(
                max_batch_size=max_batch_size,
                log_every_batches=_CROSS_REQUEST_AUDIO_MICROBATCH_STATS_LOG_EVERY_BATCHES,
            )
            if _ENABLE_CROSS_REQUEST_AUDIO_MICROBATCH_STATS
            else None
        )
        self._stats_log_prefix = (
            "Async audio microbatch stats "
            f"(workers={self.num_workers}, "
            f"torch_threads_per_worker={self.num_threads}, "
            f"max_total_torch_threads={self.max_total_torch_threads})"
        )
        self._loop = asyncio.get_running_loop()
        self._item_queue: asyncio.Queue[
            tuple[np.ndarray, asyncio.Future[dict[str, Any]], float | None]
        ] = asyncio.Queue()
        self._batch_queue: asyncio.Queue[
            list[tuple[np.ndarray, asyncio.Future[dict[str, Any]], float | None]]
        ] | None = (
            asyncio.Queue(maxsize=self.num_workers)
            if self.batch_coordinator_enabled
            else None
        )
        self._idle_worker_slots: asyncio.Semaphore | None = (
            asyncio.Semaphore(self.num_workers)
            if self.batch_coordinator_enabled
            else None
        )
        self._executors = [
            ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix=f"async-audio-microbatch-{worker_idx}",
            )
            for worker_idx in range(self.num_workers)
        ]
        self._coordinator_task = (
            self._loop.create_task(self._coordinator_loop())
            if self.batch_coordinator_enabled
            else None
        )
        self._batcher_tasks = [
            self._loop.create_task(self._batch_loop(worker_idx))
            for worker_idx in range(self.num_workers)
        ]
        logger.info(
            "Initialized async audio microbatch pool with %d workers, "
            "%d torch threads per worker (requested workers=%d, "
            "requested threads=%d, total cap=%d, adaptive_drain=%s, "
            "batch_coordinator=%s, "
            "drain_target_batch_size=%d, drain_enter_threshold=%d, "
            "drain_exit_threshold=%d).",
            self.num_workers,
            self.num_threads,
            self.requested_num_workers,
            self.requested_num_threads,
            self.max_total_torch_threads,
            self.adaptive_drain_enabled,
            self.batch_coordinator_enabled,
            self.drain_target_batch_size,
            self.drain_enter_threshold,
            self.drain_exit_threshold,
        )

    async def preprocess(self, chunk: np.ndarray) -> dict[str, Any]:
        result_future: asyncio.Future[dict[str, Any]] = self._loop.create_future()
        enqueued_at = self._loop.time() if self._stats is not None else None
        await self._item_queue.put((chunk, result_future, enqueued_at))
        return await result_future

    def reset_stats(self) -> dict[str, Any]:
        self._drain_mode = False
        if self._stats is None:
            return {
                "enabled": False,
                "reset": False,
                "summary_before_reset": None,
            }

        summary_before_reset = self._stats.reset()
        return {
            "enabled": True,
            "reset": True,
            "summary_before_reset": summary_before_reset,
        }

    def _resolve_batch_policy(self, queue_depth: int) -> tuple[int, float]:
        if not self.adaptive_drain_enabled:
            return self.max_batch_size, self.batch_wait_timeout_s

        if not self._drain_mode and queue_depth >= self.drain_enter_threshold:
            self._drain_mode = True
        elif self._drain_mode and queue_depth <= self.drain_exit_threshold:
            self._drain_mode = False

        if self._drain_mode:
            return self.drain_target_batch_size, 0.0

        return self.max_batch_size, self.batch_wait_timeout_s

    async def _fill_pending_items(
        self,
        pending_items: list[
            tuple[np.ndarray, asyncio.Future[dict[str, Any]], float | None]
        ],
        *,
        target_batch_size: int,
        wait_timeout_s: float,
    ) -> None:
        if wait_timeout_s <= 0:
            while len(pending_items) < target_batch_size:
                try:
                    pending_items.append(self._item_queue.get_nowait())
                except asyncio.QueueEmpty:
                    break
            return

        deadline = self._loop.time() + wait_timeout_s
        while len(pending_items) < target_batch_size:
            timeout = deadline - self._loop.time()
            if timeout <= 0:
                break
            try:
                pending_items.append(
                    await asyncio.wait_for(self._item_queue.get(), timeout)
                )
            except asyncio.TimeoutError:
                break

    async def _coordinator_loop(self) -> None:
        assert self._batch_queue is not None
        assert self._idle_worker_slots is not None
        pending_items: list[
            tuple[np.ndarray, asyncio.Future[dict[str, Any]], float | None]
        ] = []

        while True:
            slot_reserved = False
            try:
                if not pending_items:
                    pending_items.append(await self._item_queue.get())

                queue_depth = len(pending_items) + self._item_queue.qsize()
                target_batch_size, effective_wait_timeout_s = (
                    self._resolve_batch_policy(queue_depth)
                )
                await self._fill_pending_items(
                    pending_items,
                    target_batch_size=target_batch_size,
                    wait_timeout_s=effective_wait_timeout_s,
                )

                # Do not remove items from the shared backlog until a worker
                # slot is genuinely available; otherwise queued-but-not-running
                # batches hide work from the coordinator's queue-depth policy.
                await self._idle_worker_slots.acquire()
                slot_reserved = True

                queue_depth = len(pending_items) + self._item_queue.qsize()
                target_batch_size, _ = self._resolve_batch_policy(queue_depth)
                await self._fill_pending_items(
                    pending_items,
                    target_batch_size=target_batch_size,
                    wait_timeout_s=0.0,
                )

                batch_size = min(len(pending_items), target_batch_size)
                batch_items = pending_items[:batch_size]
                pending_items = pending_items[batch_size:]
                await self._batch_queue.put(batch_items)
                slot_reserved = False
            except asyncio.CancelledError as exc:
                if slot_reserved:
                    self._idle_worker_slots.release()
                for _, future, _ in pending_items:
                    if not future.done():
                        future.set_exception(exc)
                raise
            except Exception as exc:
                if slot_reserved:
                    self._idle_worker_slots.release()
                for _, future, _ in pending_items:
                    if not future.done():
                        future.set_exception(exc)
                raise

    async def _collect_batch_direct(
        self,
    ) -> tuple[
        list[np.ndarray],
        list[asyncio.Future[dict[str, Any]]],
        list[float | None],
    ]:
        chunk, result_future, enqueued_at = await self._item_queue.get()
        batch_items = [(chunk, result_future, enqueued_at)]

        queue_depth = len(batch_items) + self._item_queue.qsize()
        target_batch_size, effective_wait_timeout_s = self._resolve_batch_policy(
            queue_depth
        )
        await self._fill_pending_items(
            batch_items,
            target_batch_size=target_batch_size,
            wait_timeout_s=effective_wait_timeout_s,
        )

        chunks = [item_chunk for item_chunk, _, _ in batch_items]
        result_futures = [future for _, future, _ in batch_items]
        enqueued_ats = [item_enqueued_at for _, _, item_enqueued_at in batch_items]
        return chunks, result_futures, enqueued_ats

    def _ensure_worker_preprocess_fn(
        self,
        worker_idx: int,
    ) -> Callable[[list[np.ndarray]], list[dict[str, Any]]]:
        worker_preprocess_fn = self._worker_preprocess_fns[worker_idx]
        if worker_preprocess_fn is None:
            if self.preprocess_factory is None:
                worker_preprocess_fn = partial(
                    self.preprocess_fn,
                    model_config=self.model_config,
                )
            else:
                init_start = time.perf_counter()
                worker_preprocess_fn = self.preprocess_factory(self.model_config)
                init_ms = (time.perf_counter() - init_start) * 1000
                logger.info(
                    "Initialized async audio microbatch worker %d in %.2f ms.",
                    worker_idx,
                    init_ms,
                )
            self._worker_preprocess_fns[worker_idx] = worker_preprocess_fn

        return worker_preprocess_fn

    def _run_preprocess_batch(
        self,
        worker_idx: int,
        chunks: list[np.ndarray],
    ) -> list[dict[str, Any]]:
        with set_default_torch_num_threads(self.num_threads):
            worker_preprocess_fn = self._ensure_worker_preprocess_fn(worker_idx)
            return worker_preprocess_fn(chunks)

    async def _batch_loop(self, worker_idx: int) -> None:
        executor = self._executors[worker_idx]
        while True:
            result_futures: list[asyncio.Future[dict[str, Any]]] = []
            slot_reserved = False
            try:
                if self.batch_coordinator_enabled:
                    assert self._batch_queue is not None
                    batch_items = await self._batch_queue.get()
                    slot_reserved = True
                    chunks = [chunk for chunk, _, _ in batch_items]
                    result_futures = [future for _, future, _ in batch_items]
                    enqueued_ats = [enqueued_at for _, _, enqueued_at in batch_items]
                else:
                    chunks, result_futures, enqueued_ats = (
                        await self._collect_batch_direct()
                    )

                batch_start = self._loop.time() if self._stats is not None else None
                preprocess_start = (
                    time.perf_counter() if self._stats is not None else None
                )
                preprocess_fn = partial(self._run_preprocess_batch, worker_idx, chunks)
                results = await self._loop.run_in_executor(executor, preprocess_fn)
                if (
                    self._stats is not None
                    and batch_start is not None
                    and preprocess_start is not None
                ):
                    total_queue_wait_secs = 0.0
                    max_queue_wait_secs = 0.0
                    for item_enqueued_at in enqueued_ats:
                        if item_enqueued_at is None:
                            continue
                        wait_secs = batch_start - item_enqueued_at
                        total_queue_wait_secs += wait_secs
                        if wait_secs > max_queue_wait_secs:
                            max_queue_wait_secs = wait_secs

                    self._stats.record(
                        batch_size=len(chunks),
                        total_queue_wait_secs=total_queue_wait_secs,
                        max_queue_wait_secs=max_queue_wait_secs,
                        preprocess_secs=time.perf_counter() - preprocess_start,
                    )
                    self._stats.maybe_log(prefix=self._stats_log_prefix)

                for future, result in zip(result_futures, results):
                    if not future.done():
                        future.set_result(result)
            except asyncio.CancelledError as exc:
                for future in result_futures:
                    if not future.done():
                        future.set_exception(exc)
                raise
            except Exception as exc:
                for future in result_futures:
                    if not future.done():
                        future.set_exception(exc)
            finally:
                if slot_reserved:
                    assert self._idle_worker_slots is not None
                    self._idle_worker_slots.release()

    def __del__(self) -> None:
        stats = getattr(self, "_stats", None)
        if stats is not None and stats.total_batches > 0:
            logger.info(
                "Final %s: %s",
                self._stats_log_prefix.lower(),
                stats.summary(),
            )
        executors = getattr(self, "_executors", None)
        if executors is not None:
            for executor in executors:
                executor.shutdown(wait=False, cancel_futures=True)
        coordinator_task = getattr(self, "_coordinator_task", None)
        batcher_tasks = getattr(self, "_batcher_tasks", None)
        loop = getattr(self, "_loop", None)
        if coordinator_task is not None and loop is not None and not loop.is_closed():
            loop.call_soon_threadsafe(coordinator_task.cancel)
        if batcher_tasks is not None and loop is not None and not loop.is_closed():
            for batcher_task in batcher_tasks:
                loop.call_soon_threadsafe(batcher_task.cancel)


class OpenAISpeechToText(OpenAIServing):
    """Base class for speech-to-text operations like transcription and
    translation."""

    def __init__(
        self,
        engine_client: EngineClient,
        models: OpenAIServingModels,
        *,
        request_logger: RequestLogger | None,
        return_tokens_as_token_ids: bool = False,
        task_type: Literal["transcribe", "translate"] = "transcribe",
        enable_force_include_usage: bool = False,
    ):
        super().__init__(
            engine_client=engine_client,
            models=models,
            request_logger=request_logger,
            return_tokens_as_token_ids=return_tokens_as_token_ids,
        )

        self.default_sampling_params = self.model_config.get_diff_sampling_param()
        self.task_type: Final = task_type

        self.asr_config = self.model_cls.get_speech_to_text_config(
            self.model_config, task_type
        )

        self.enable_force_include_usage = enable_force_include_usage

        self.max_audio_filesize_mb = envs.VLLM_MAX_AUDIO_CLIP_FILESIZE_MB
        if self.model_cls.supports_segment_timestamp:
            self.tokenizer = cast(
                PreTrainedTokenizerBase,
                get_tokenizer(
                    tokenizer_name=self.model_config.tokenizer,
                    tokenizer_mode=self.model_config.tokenizer_mode,
                ),
            )

        if self.default_sampling_params:
            logger.info(
                "Overwriting default completion sampling param with: %s",
                self.default_sampling_params,
            )

    @cached_property
    def model_cls(self) -> type[SupportsTranscription]:
        from vllm.model_executor.model_loader import get_model_cls

        model_cls = get_model_cls(self.model_config)
        return cast(type[SupportsTranscription], model_cls)

    @cached_property
    def _async_audio_microbatch_topology(self) -> tuple[int, int]:
        return _resolve_async_audio_microbatch_topology(
            requested_num_workers=_CROSS_REQUEST_AUDIO_MICROBATCH_NUM_WORKERS,
            requested_threads_per_worker=HF_PROCESSOR_NUM_THREADS,
            max_total_torch_threads=(
                _CROSS_REQUEST_AUDIO_MICROBATCH_MAX_TOTAL_TORCH_THREADS
            ),
        )

    @cached_property
    def _async_audio_preprocessor(self) -> AsyncMicrobatchAudioPreprocessor | None:
        if not _ENABLE_CROSS_REQUEST_AUDIO_MICROBATCH:
            return None

        batch_preprocess_audio = getattr(
            self.model_cls, "batch_preprocess_audio_chunks", None
        )
        if not callable(batch_preprocess_audio):
            return None

        preprocess_factory = getattr(
            self.model_cls, "create_audio_microbatch_preprocessor", None
        )
        num_workers, num_threads = self._async_audio_microbatch_topology

        return AsyncMicrobatchAudioPreprocessor(
            batch_preprocess_audio,
            self.model_config,
            preprocess_factory=(
                preprocess_factory if callable(preprocess_factory) else None
            ),
            num_workers=num_workers,
            num_threads=num_threads,
        )

    def reset_async_audio_microbatch_stats(self) -> dict[str, Any]:
        if not _ENABLE_CROSS_REQUEST_AUDIO_MICROBATCH:
            return {
                "available": False,
                "reset": False,
                "reason": "async_audio_microbatch_disabled",
            }

        preprocessor = self.__dict__.get("_async_audio_preprocessor")
        if preprocessor is None:
            return {
                "available": True,
                "reset": False,
                "reason": "async_audio_microbatch_not_initialized",
            }

        assert isinstance(preprocessor, AsyncMicrobatchAudioPreprocessor)
        reset_result = preprocessor.reset_stats()
        return {
            "available": True,
            "reset": reset_result["reset"],
            "stats_enabled": reset_result["enabled"],
            "summary_before_reset": reset_result["summary_before_reset"],
        }

    async def _detect_language(
        self,
        audio_chunk: np.ndarray,
        request_id: str,
    ) -> str:
        """Auto-detect the spoken language from an audio chunk.

        Delegates prompt construction and output parsing to the model class
        via ``get_language_detection_prompt`` and
        ``parse_language_detection_output``.
        """
        prompt = self.model_cls.get_language_detection_prompt(
            audio_chunk,
            self.asr_config,
        )
        allowed_token_ids = self.model_cls.get_language_token_ids(
            self.tokenizer,
        )
        sampling_params = SamplingParams(
            max_tokens=1,
            temperature=0.0,
            allowed_token_ids=allowed_token_ids,
        )

        result_generator = self.engine_client.generate(
            prompt,
            sampling_params,
            request_id,
        )

        final_output: RequestOutput
        async for final_output in result_generator:
            if final_output.finished:
                break

        token_ids = list(final_output.outputs[0].token_ids)
        lang = self.model_cls.parse_language_detection_output(
            token_ids,
            self.tokenizer,
        )

        logger.info("Auto-detected language: '%s'", lang)
        return lang

    async def _preprocess_speech_to_text(
        self,
        request: SpeechToTextRequest,
        audio_data: bytes,
        request_id: str,
    ) -> tuple[list[ProcessorInputs], float]:
        # Validate request
        language = self.model_cls.validate_language(request.language)
        # Skip to_language validation to avoid extra logging for Whisper.
        to_language = (
            self.model_cls.validate_language(request.to_language)
            if request.to_language
            else None
        )

        if len(audio_data) / 1024**2 > self.max_audio_filesize_mb:
            raise VLLMValidationError(
                "Maximum file size exceeded",
                parameter="audio_filesize_mb",
                value=len(audio_data) / 1024**2,
            )

        # Decode audio bytes.  For container formats (MP4, M4A, WebM) that
        # soundfile cannot detect from a BytesIO stream, _load_audio_bytes
        # transparently falls back to ffmpeg via an in-memory fd.
        # NOTE resample to model SR here for efficiency. This is also a
        # pre-requisite for chunking, as it assumes Whisper SR.
        try:
            with io.BytesIO(audio_data) as buf:
                y, sr = librosa.load(buf, sr=self.asr_config.sample_rate)  # type: ignore[return-value]
        except sf.LibsndfileError as exc:
            # Only fall back for known format-detection failures.
            # Re-raise anything else (e.g. corrupt but recognised format).
            if exc.code not in _BAD_SF_CODES:
                raise
            logger.debug(
                "librosa/soundfile could not decode audio from BytesIO "
                "(code=%s: %s); falling back to pyav in-process decode",
                exc.code,
                exc,
            )
            try:
                native_y, native_sr = extract_audio_from_video_bytes(audio_data)
                sr = self.asr_config.sample_rate
                y = librosa.resample(native_y, orig_sr=native_sr, target_sr=sr)
            except Exception as pyav_exc:
                logger.debug(
                    "pyAV fallback also failed: %s",
                    pyav_exc,
                )
                raise ValueError("Invalid or unsupported audio file.") from pyav_exc

        duration = librosa.get_duration(y=y, sr=sr)
        do_split_audio = (
            self.asr_config.allow_audio_chunking
            and duration > self.asr_config.max_audio_clip_s
        )

        if not do_split_audio:
            chunks = [y]
        else:
            assert self.asr_config.max_audio_clip_s is not None
            assert self.asr_config.min_energy_split_window_size is not None
            chunks = split_audio(
                audio_data=y,
                sample_rate=int(sr),
                max_clip_duration_s=self.asr_config.max_audio_clip_s,
                overlap_duration_s=self.asr_config.overlap_chunk_second,
                min_energy_window_size=self.asr_config.min_energy_split_window_size,
            )

        if language is None and getattr(
            self.model_cls, "supports_explicit_language_detection", False
        ):
            # Auto-detect language from the first chunk.
            language = await self._detect_language(
                chunks[0], f"{request_id}-lang_detect"
            )
            request.language = language

        preprocessed_chunks = None
        batch_preprocess_audio = getattr(
            self.model_cls, "batch_preprocess_audio_chunks", None
        )
        if callable(batch_preprocess_audio):
            _, microbatch_num_threads = self._async_audio_microbatch_topology
            if len(chunks) > 1:
                with set_default_torch_num_threads(microbatch_num_threads):
                    preprocessed_chunks = batch_preprocess_audio(
                        chunks,
                        self.model_config,
                    )
            elif len(chunks) == 1 and self._async_audio_preprocessor is not None:
                preprocessed_chunks = [
                    await self._async_audio_preprocessor.preprocess(chunks[0])
                ]

        parsed_prompts: list[DictPrompt] = []
        for i, chunk in enumerate(chunks):
            # The model has control over the construction, as long as it
            # returns a valid PromptType.
            prompt = self.model_cls.get_generation_prompt(
                audio=chunk,
                stt_config=self.asr_config,
                model_config=self.model_config,
                language=language,
                task_type=self.task_type,
                request_prompt=request.prompt,
                to_language=to_language,
            )
            if preprocessed_chunks is not None:
                prompt["multi_modal_data"] = {
                    "audio": preprocessed_chunks[i],
                }

            parsed_prompt: DictPrompt
            if request.response_format == "verbose_json":
                parsed_prompt = parse_enc_dec_prompt(prompt)
                parsed_prompt = self._preprocess_verbose_prompt(parsed_prompt)
            else:
                parsed_prompt = parse_model_prompt(self.model_config, prompt)

            parsed_prompts.append(parsed_prompt)

        engine_prompts = await self.renderer.render_cmpl_async(parsed_prompts)

        return engine_prompts, duration

    def _preprocess_verbose_prompt(self, prompt: EncoderDecoderDictPrompt):
        dec_prompt = prompt["decoder_prompt"]

        if not (isinstance(dec_prompt, dict) and "prompt" in dec_prompt):
            raise VLLMValidationError(
                "Expected decoder_prompt to contain text",
                parameter="decoder_prompt",
                value=type(dec_prompt).__name__,
            )

        dec_prompt["prompt"] = dec_prompt["prompt"].replace(
            "<|notimestamps|>", "<|0.00|>"
        )

        return prompt

    @staticmethod
    def _get_decoder_prompt_len(engine_prompts: list[ProcessorInputs]) -> int:
        """Get the length of the decoder prompt. Currently we need to offset
        by the decoder prompt length when running beam search because the mm
        encoder is not currently cached and runs on decode calls; because of
        this, we need to make sure the redundant encoder calls won't exceed
        the context :(

        FIXME (Alex) - this will be removed in the very near future once the
        encoder/decoder caching is implemented.
        """
        input_len = 0
        assert len(engine_prompts) > 0
        first_eng_prompt = engine_prompts[0]

        if first_eng_prompt.get("type") == "enc_dec":
            first_eng_prompt = cast(EncoderDecoderInputs, first_eng_prompt)
            input_len = len(first_eng_prompt["decoder_prompt"]["prompt_token_ids"])
        return input_len

    def _get_verbose_segments(
        self,
        tokens: tuple,
        log_probs: FlatLogprobs | list[dict[int, Logprob]],
        request: SpeechToTextRequest,
        segment_class: type[SpeechToTextSegment],
        start_time: float = 0,
    ) -> list[SpeechToTextSegment]:
        """
        Convert tokens to verbose segments.

        This method expects the model to produce
        timestamps as tokens (similar to Whisper).
        If the tokens do not include timestamp information,
        the segments may not be generated correctly.

        Note: No_speech_prob field is not supported
        in this implementation and will be None. See docs for details.
        """
        BASE_OFFSET = 0.02
        init_token = self.tokenizer.encode("<|0.00|>", add_special_tokens=False)[0]
        if tokens[-1] == self.tokenizer.eos_token_id:
            tokens = tokens[:-1]

        tokens_with_start = (init_token,) + tokens
        segments: list[SpeechToTextSegment] = []
        last_timestamp_start = 0

        if tokens_with_start[-2] < init_token and tokens_with_start[-1] >= init_token:
            tokens_with_start = tokens_with_start + (tokens_with_start[-1],)
        avg_logprob = 0.0
        for idx in range(1, len(tokens_with_start)):
            # Timestamp tokens (e.g., <|0.00|>) are assumed to be sorted.
            # If the ordering is violated, this slicing may produce incorrect results.
            token = tokens_with_start[idx]
            if token >= init_token and tokens_with_start[idx - 1] >= init_token:
                sliced_timestamp_tokens = tokens_with_start[last_timestamp_start:idx]
                start_timestamp = sliced_timestamp_tokens[0] - init_token
                end_timestamp = sliced_timestamp_tokens[-1] - init_token
                text = self.tokenizer.decode(sliced_timestamp_tokens[1:-1])
                text_bytes = text.encode("utf-8")

                casting_segment = cast(
                    SpeechToTextSegment,
                    segment_class(
                        id=len(segments),
                        seek=start_time,
                        start=start_time + BASE_OFFSET * start_timestamp,
                        end=start_time + BASE_OFFSET * end_timestamp,
                        temperature=request.temperature,
                        text=text,
                        # The compression ratio measures
                        # how compressible the generated text is.
                        # A higher ratio indicates more repetitive content,
                        # which is a strong sign of hallucination in outputs.
                        compression_ratio=len(text_bytes)
                        / len(zlib.compress(text_bytes)),
                        tokens=sliced_timestamp_tokens[1:-1],
                        avg_logprob=avg_logprob / (idx - last_timestamp_start),
                    ),
                )
                segments.append(casting_segment)
                last_timestamp_start = idx
                avg_logprob = 0
            else:
                avg_logprob += log_probs[idx - 1][token].logprob
        return segments

    async def _create_speech_to_text(
        self,
        audio_data: bytes,
        request: SpeechToTextRequest,
        raw_request: Request,
        response_class: type[ResponseType],
        stream_generator_method: Callable[..., AsyncGenerator[str, None]],
    ) -> T | V | AsyncGenerator[str, None] | ErrorResponse:
        """Base method for speech-to-text operations like transcription and
        translation."""
        if request.stream and request.use_beam_search:
            return self.create_error_response(
                "Streaming is not currently supported with beam search"
            )

        error_check_ret = await self._check_model(request)
        if error_check_ret is not None:
            return error_check_ret

        # If the engine is dead, raise the engine's DEAD_ERROR.
        # This is required for the streaming case, where we return a
        # success status before we actually start generating text :).
        if self.engine_client.errored:
            raise self.engine_client.dead_error

        if request.response_format not in ["text", "json", "verbose_json"]:
            return self.create_error_response(
                "Currently only support response_format: "
                "`text`, `json` or `verbose_json`"
            )

        if (
            request.response_format == "verbose_json"
            and not self.model_cls.supports_segment_timestamp
        ):
            return self.create_error_response(
                f"Currently do not support verbose_json for {request.model}"
            )

        if request.response_format == "verbose_json" and request.stream:
            return self.create_error_response(
                "verbose_json format doesn't support streaming case"
            )
        request_id = f"{self.task_type}-{self._base_request_id(raw_request)}"

        request_metadata = RequestResponseMetadata(request_id=request_id)
        if raw_request:
            raw_request.state.request_metadata = request_metadata

        lora_request = self._maybe_get_adapters(request)

        engine_prompts, duration_s = await self._preprocess_speech_to_text(
            request=request,
            audio_data=audio_data,
            request_id=request_id,
        )

        # Schedule the request and get the result generator.
        max_model_len = self.model_config.max_model_len
        list_result_generator: list[AsyncGenerator[RequestOutput, None]] | None = None

        input_len = (
            OpenAISpeechToText._get_decoder_prompt_len(engine_prompts)
            if request.use_beam_search
            else 0
        )

        # Unlike most decoder-only models, whisper generation length is not
        # constrained by the size of the input audio, which is mapped to a
        # fixed-size log-mel-spectogram. Still, allow for fewer tokens to be
        # generated by respecting the extra completion tokens arg.
        max_tokens = get_max_tokens(
            max_model_len,
            request.max_completion_tokens,
            input_len,
            self.default_sampling_params,
        )

        if request.use_beam_search:
            sampling_params = request.to_beam_search_params(
                max_tokens, self.default_sampling_params
            )
        else:
            sampling_params = request.to_sampling_params(
                max_tokens,
                self.default_sampling_params,
            )

        if request.response_format == "verbose_json":
            sampling_params.logprobs = 1

        list_result_generator = []
        for i, engine_prompt in enumerate(engine_prompts):
            request_id_item = f"{request_id}_{i}"

            self._log_inputs(
                request_id_item,
                engine_prompt,
                params=sampling_params,
                lora_request=lora_request,
            )

            trace_headers = (
                None
                if raw_request is None
                else await self._get_trace_headers(raw_request.headers)
            )

            if isinstance(sampling_params, BeamSearchParams):
                generator = self.beam_search(
                    prompt=engine_prompt,
                    params=sampling_params,
                    request_id=request_id_item,
                    lora_request=lora_request,
                    trace_headers=trace_headers,
                )
            else:
                generator = self.engine_client.generate(
                    engine_prompt,
                    sampling_params,
                    request_id_item,
                    lora_request=lora_request,
                    trace_headers=trace_headers,
                )

            list_result_generator.append(generator)

        if request.stream:
            return stream_generator_method(
                request, list_result_generator, request_id, request_metadata, duration_s
            )
        # Non-streaming response.
        total_segments = []
        text_parts = []
        try:
            assert list_result_generator is not None
            segments_types: dict[str, type[SpeechToTextSegment]] = {
                "transcribe": TranscriptionSegment,
                "translate": TranslationSegment,
            }
            segment_class: type[SpeechToTextSegment] = segments_types[self.task_type]
            text = ""
            chunk_size_in_s = self.asr_config.max_audio_clip_s
            if chunk_size_in_s is None:
                assert len(list_result_generator) == 1, (
                    "`max_audio_clip_s` is set to None, audio cannot be chunked"
                )
            for idx, result_generator in enumerate(list_result_generator):
                start_time = (
                    float(idx * chunk_size_in_s) if chunk_size_in_s is not None else 0.0
                )
                async for op in result_generator:
                    if request.response_format == "verbose_json":
                        assert op.outputs[0].logprobs
                        segments: list[SpeechToTextSegment] = (
                            self._get_verbose_segments(
                                tokens=tuple(op.outputs[0].token_ids),
                                segment_class=segment_class,
                                request=request,
                                start_time=start_time,
                                log_probs=op.outputs[0].logprobs,
                            )
                        )

                        total_segments.extend(segments)
                        text_parts.extend([seg.text for seg in segments])
                    else:
                        raw_text = op.outputs[0].text
                        text_parts.append(self.model_cls.post_process_output(raw_text))
            text = "".join(text_parts)
            if self.task_type == "transcribe":
                final_response: ResponseType
                # add usage in TranscriptionResponse.
                usage = {
                    "type": "duration",
                    # rounded up as per openAI specs
                    "seconds": int(math.ceil(duration_s)),
                }
                if request.response_format != "verbose_json":
                    final_response = cast(
                        T, TranscriptionResponse(text=text, usage=usage)
                    )
                else:
                    final_response = cast(
                        V,
                        TranscriptionResponseVerbose(
                            text=text,
                            language=request.language,
                            duration=str(duration_s),
                            segments=total_segments,
                        ),
                    )
            else:
                # no usage in response for translation task
                if request.response_format != "verbose_json":
                    final_response = cast(T, TranslationResponse(text=text))
                else:
                    final_response = cast(
                        V,
                        TranslationResponseVerbose(
                            text=text,
                            language=request.language,
                            duration=str(duration_s),
                            segments=total_segments,
                        ),
                    )
            return final_response
        except asyncio.CancelledError:
            return self.create_error_response("Client disconnected")

    async def _speech_to_text_stream_generator(
        self,
        request: SpeechToTextRequest,
        list_result_generator: list[AsyncGenerator[RequestOutput, None]],
        request_id: str,
        request_metadata: RequestResponseMetadata,
        audio_duration_s: float,
        chunk_object_type: Literal["translation.chunk", "transcription.chunk"],
        response_stream_choice_class: type[TranscriptionResponseStreamChoice]
        | type[TranslationResponseStreamChoice],
        stream_response_class: type[TranscriptionStreamResponse]
        | type[TranslationStreamResponse],
    ) -> AsyncGenerator[str, None]:
        created_time = int(time.time())
        model_name = request.model

        completion_tokens = 0
        num_prompt_tokens = 0

        include_usage = self.enable_force_include_usage or request.stream_include_usage
        include_continuous_usage = (
            request.stream_continuous_usage_stats
            if include_usage and request.stream_continuous_usage_stats
            else False
        )

        try:
            for result_generator in list_result_generator:
                async for res in result_generator:
                    # On first result.
                    if res.prompt_token_ids is not None:
                        num_prompt_tokens = len(res.prompt_token_ids)
                        if audio_tokens := self.model_cls.get_num_audio_tokens(
                            audio_duration_s, self.asr_config, self.model_config
                        ):
                            num_prompt_tokens += audio_tokens

                    # We need to do it here, because if there are exceptions in
                    # the result_generator, it needs to be sent as the FIRST
                    # response (by the try...catch).

                    # Just one output (n=1) supported.
                    assert len(res.outputs) == 1
                    output = res.outputs[0]

                    # TODO: For models that output structured formats (e.g.,
                    # Qwen3-ASR with "language X<asr_text>" prefix), streaming
                    # would need buffering to strip the prefix properly since
                    # deltas may split the tag across chunks.
                    delta_message = DeltaMessage(content=output.text)
                    completion_tokens += len(output.token_ids)

                    if output.finish_reason is None:
                        # Still generating, send delta update.
                        choice_data = response_stream_choice_class(delta=delta_message)
                    else:
                        # Model is finished generating.
                        choice_data = response_stream_choice_class(
                            delta=delta_message,
                            finish_reason=output.finish_reason,
                            stop_reason=output.stop_reason,
                        )

                    chunk = stream_response_class(
                        id=request_id,
                        object=chunk_object_type,
                        created=created_time,
                        choices=[choice_data],
                        model=model_name,
                    )

                    # handle usage stats if requested & if continuous
                    if include_continuous_usage:
                        chunk.usage = UsageInfo(
                            prompt_tokens=num_prompt_tokens,
                            completion_tokens=completion_tokens,
                            total_tokens=num_prompt_tokens + completion_tokens,
                        )

                    data = chunk.model_dump_json(exclude_unset=True)
                    yield f"data: {data}\n\n"

            # Once the final token is handled, if stream_options.include_usage
            # is sent, send the usage.
            if include_usage:
                final_usage = UsageInfo(
                    prompt_tokens=num_prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=num_prompt_tokens + completion_tokens,
                )

                final_usage_chunk = stream_response_class(
                    id=request_id,
                    object=chunk_object_type,
                    created=created_time,
                    choices=[],
                    model=model_name,
                    usage=final_usage,
                )
                final_usage_data = final_usage_chunk.model_dump_json(
                    exclude_unset=True, exclude_none=True
                )
                yield f"data: {final_usage_data}\n\n"

            # report to FastAPI middleware aggregate usage across all choices
            request_metadata.final_usage_info = UsageInfo(
                prompt_tokens=num_prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=num_prompt_tokens + completion_tokens,
            )

        except Exception as e:
            logger.exception("Error in %s stream generator.", self.task_type)
            data = self.create_streaming_error_response(e)
            yield f"data: {data}\n\n"
        # Send the final done message after all response.n are finished
        yield "data: [DONE]\n\n"
