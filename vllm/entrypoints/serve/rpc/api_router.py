# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from collections import defaultdict
from http import HTTPStatus
from typing import Any

from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response

import vllm.envs as envs
from vllm.engine.protocol import EngineClient
from vllm.logger import init_logger

logger = init_logger(__name__)

router = APIRouter()


def engine_client(request: Request) -> EngineClient:
    return request.app.state.engine_client


def _aggregate_per_request_stage_stats(
    per_request_stats: dict[str, dict[str, float | int]],
) -> dict[str, dict[str, float | int]]:
    stage_totals: dict[str, dict[str, float]] = defaultdict(
        lambda: {"total_secs": 0.0, "count": 0.0, "max_secs": 0.0}
    )
    for request_stats in per_request_stats.values():
        for stage, value in request_stats.items():
            if not isinstance(value, int | float):
                continue
            numeric_value = float(value)
            stage_total = stage_totals[stage]
            stage_total["total_secs"] += numeric_value
            stage_total["count"] += 1.0
            stage_total["max_secs"] = max(stage_total["max_secs"], numeric_value)

    return {
        stage: {
            "total_secs": metrics["total_secs"],
            "count": int(metrics["count"]),
            "avg_secs": (
                metrics["total_secs"] / metrics["count"] if metrics["count"] else 0.0
            ),
            "max_secs": metrics["max_secs"],
        }
        for stage, metrics in stage_totals.items()
    }


def _merge_stage_snapshots(
    snapshots: list[dict[str, dict[str, float | int]]],
) -> dict[str, dict[str, float | int]]:
    merged: dict[str, dict[str, float]] = defaultdict(
        lambda: {"total_secs": 0.0, "count": 0.0, "max_secs": 0.0}
    )
    for snapshot in snapshots:
        for stage, stats in snapshot.items():
            total_secs = float(stats.get("total_secs", 0.0))
            count = float(stats.get("count", 0))
            max_secs = float(stats.get("max_secs", 0.0))
            merged_stats = merged[stage]
            merged_stats["total_secs"] += total_secs
            merged_stats["count"] += count
            merged_stats["max_secs"] = max(merged_stats["max_secs"], max_secs)

    return {
        stage: {
            "total_secs": stats["total_secs"],
            "count": int(stats["count"]),
            "avg_secs": stats["total_secs"] / stats["count"] if stats["count"] else 0.0,
            "max_secs": stats["max_secs"],
        }
        for stage, stats in merged.items()
    }


def _aggregate_encoder_timing_stats(
    worker_stats: list[dict[str, dict[str, float | int]]],
) -> dict[str, Any]:
    total_encoder_forward_secs = 0.0
    total_encoder_calls = 0
    num_requests = 0
    max_encoder_forward_secs = 0.0
    for stats_by_request in worker_stats:
        for stats in stats_by_request.values():
            encoder_forward_secs = float(stats.get("encoder_forward_secs", 0.0))
            total_encoder_forward_secs += encoder_forward_secs
            total_encoder_calls += int(stats.get("num_encoder_calls", 0))
            max_encoder_forward_secs = max(max_encoder_forward_secs, encoder_forward_secs)
            num_requests += 1

    return {
        "encoder_forward_secs": {
            "total_secs": total_encoder_forward_secs,
            "count": num_requests,
            "avg_secs": (
                total_encoder_forward_secs / num_requests if num_requests else 0.0
            ),
            "max_secs": max_encoder_forward_secs,
        },
        "total_encoder_calls": total_encoder_calls,
        "avg_encoder_calls_per_request": (
            total_encoder_calls / num_requests if num_requests else 0.0
        ),
        "num_requests": num_requests,
    }


@router.post("/collective_rpc")
async def collective_rpc(raw_request: Request):
    try:
        body = await raw_request.json()
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail=f"JSON decode error: {e}",
        ) from e
    method = body.get("method")
    if method is None:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail="Missing 'method' in request body",
        )
    # For security reason, only serialized string args/kwargs are passed.
    # User-defined `method` is responsible for deserialization if needed.
    args: list[str] = body.get("args", [])
    kwargs: dict[str, str] = body.get("kwargs", {})
    timeout: float | None = body.get("timeout")
    results = await engine_client(raw_request).collective_rpc(
        method=method, timeout=timeout, args=tuple(args), kwargs=kwargs
    )
    if results is None:
        return Response(status_code=200)
    response: list[Any] = []
    for result in results:
        if result is None or isinstance(result, dict | list):
            response.append(result)
        else:
            response.append(str(result))
    return JSONResponse(content={"results": response})


@router.post("/debug/asr_bottleneck_stats")
async def asr_bottleneck_stats(raw_request: Request):
    from vllm.entrypoints.openai.speech_to_text.speech_to_text import (
        get_and_reset_speech_to_text_timing_stats,
    )
    from vllm.renderers.base import get_and_reset_renderer_timing_stats
    from vllm.transformers_utils.processors.cohere_asr import (
        get_and_reset_cohere_asr_processor_timing_stats,
    )
    from vllm.v1.engine.core_client import get_and_reset_core_client_timing_stats

    client = engine_client(raw_request)
    renderer = client.renderer
    mm_timing_registry = getattr(renderer, "_mm_timing_registry", None)
    mm_processor_stats = (
        _aggregate_per_request_stage_stats(mm_timing_registry.stat())
        if mm_timing_registry is not None
        else {}
    )

    engine_core_stats: dict[str, Any] = {}
    engine_core = getattr(client, "engine_core", None)
    call_utility_async = getattr(engine_core, "call_utility_async", None)
    if call_utility_async is not None:
        engine_core_stats = await call_utility_async("get_asr_debug_timing_stats")

    worker_model_stats = await client.collective_rpc("get_asr_model_timing_stats")
    worker_encoder_stats = await client.collective_rpc("get_encoder_timing_stats")

    response = {
        "frontend": {
            "speech_to_text": get_and_reset_speech_to_text_timing_stats(),
            "renderer": get_and_reset_renderer_timing_stats(),
            "feature_extractor": get_and_reset_cohere_asr_processor_timing_stats(),
            "mm_processor": mm_processor_stats,
            "core_client": get_and_reset_core_client_timing_stats(),
        },
        "engine_core": engine_core_stats,
        "worker": {
            "model": _merge_stage_snapshots(worker_model_stats),
            "encoder": _aggregate_encoder_timing_stats(worker_encoder_stats),
        },
    }
    return JSONResponse(content=response)


def attach_router(app: FastAPI):
    if not envs.VLLM_SERVER_DEV_MODE:
        return
    app.include_router(router)
