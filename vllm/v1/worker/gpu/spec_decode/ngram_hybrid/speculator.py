# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hybrid n-gram + model speculators ("ngram-eagle" and friends).

Both drafters run every step. Per request the n-gram draft wins when it found
a match, otherwise the model draft is used, so each request carries a variable
number of drafts (`K_ngram` on a hit, `K_model` on a miss). The merged count is
published through `num_valid_drafts_for_trim`, which lets `VariableDraftTrimmer`
rebuild the verification layout on device with no CPU synchronization.
"""

from functools import lru_cache
from typing import Any

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.dp_utils import DPSyncState
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.spec_decode.ngram.speculator import NgramGPUSpeculator
from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator
from vllm.v1.worker.gpu.states import RequestState

logger = init_logger(__name__)


@triton.jit
def _merge_kernel(
    ngram_drafts_ptr,  # *int64  [B, K_NGRAM]        batch indexed
    model_drafts_ptr,  # *int64  [B, K_MODEL]        batch indexed
    ngram_valid_ptr,  # *int32  [max_num_reqs]      req-slot indexed
    idx_mapping_ptr,  # *int64  [B]  batch_idx -> req_state_idx
    last_sampled_ptr,  # *int64  [max_num_reqs]      req-slot indexed
    out_drafts_ptr,  # *int64  [B, K_TOTAL]        (output, batch indexed)
    out_valid_ptr,  # *int32  [max_num_reqs]      (output, req-slot indexed)
    K_NGRAM: tl.constexpr,
    K_MODEL: tl.constexpr,
    K_TOTAL: tl.constexpr,
    K_TOTAL_PO2: tl.constexpr,
):
    b = tl.program_id(0).to(tl.int64)
    req_state_idx = tl.load(idx_mapping_ptr + b).to(tl.int64)

    # The n-gram reports 0 valid drafts when it found no match.
    ngram_valid = tl.load(ngram_valid_ptr + req_state_idx).to(tl.int64)
    use_ngram = ngram_valid > 0

    k = tl.arange(0, K_TOTAL_PO2).to(tl.int64)
    num_src = tl.where(use_ngram, K_NGRAM, K_MODEL)
    in_src = k < num_src

    ngram_tok = tl.load(
        ngram_drafts_ptr + b * K_NGRAM + k, mask=in_src & (k < K_NGRAM), other=0
    )
    model_tok = tl.load(
        model_drafts_ptr + b * K_MODEL + k, mask=in_src & (k < K_MODEL), other=0
    )
    # Slots past the chosen drafter's width are trimmed from the verification
    # batch on GPU, or verified as ordinary (rejectable) drafts when trimming
    # is unavailable. It must still be a real token id so pad with last_sampled
    pad = tl.load(last_sampled_ptr + req_state_idx)
    out = tl.where(in_src, tl.where(use_ngram, ngram_tok, model_tok), pad)
    tl.store(out_drafts_ptr + b * K_TOTAL + k, out, mask=k < K_TOTAL)

    tl.store(out_valid_ptr + req_state_idx, tl.where(use_ngram, ngram_valid, K_MODEL))


class NgramHybridMixin:
    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        req_states: RequestState,
    ):
        spec = vllm_config.speculative_config
        assert spec is not None and spec.ngram_hybrid
        if spec.enable_adaptive_verification:
            raise ValueError(
                f"ngram-{spec.method} does not support adaptive verification."
            )
        if spec.draft_sample_method == "probabilistic":
            raise ValueError(
                f"ngram-{spec.method} does not support probabilistic draft sampling."
            )

        super().__init__(vllm_config, device)

        self.num_ngram_steps = spec.num_ngram_speculative_tokens()
        self.num_total_steps = spec.num_speculative_tokens
        self.ngram = NgramGPUSpeculator(
            vllm_config, device, req_states, self.num_ngram_steps
        )
        # Merged drafts, in the total width the model runner scatters into
        # RequestState.draft_tokens.
        self.combined_drafts = torch.zeros(
            self.max_num_reqs, self.num_total_steps, dtype=torch.int64, device=device
        )
        # Post-merge per-request counts. Distinct from the n-gram's own buffer,
        # which only knows about its half.
        self.num_valid_drafts_for_trim = torch.zeros(
            self.max_num_reqs, dtype=torch.int32, device=device
        )
        logger.info(
            "Hybrid speculative decoding: ngram(k=%d) with %s(k=%d) fallback",
            self.num_ngram_steps,
            spec.method,
            self.num_speculative_steps,
        )

    def draft_num_speculative_steps(self) -> int:
        return self.speculative_config.num_model_speculative_tokens()

    def load_model(self, target_model: nn.Module) -> None:
        super().load_model(target_model)
        if self.supports_mm_inputs:
            # The n-gram half is text-only, so the drafters would disagree on
            # what they are conditioning on. Fall back to text-only drafting,
            # as the pure n-gram speculator does.
            logger.warning_once(
                "Hybrid n-gram speculative decoding does not pass multimodal "
                "embeddings to the drafter; using text-only draft inputs."
            )
            self.supports_mm_inputs = False

    @torch.inference_mode()
    def propose(
        self,
        input_batch: InputBatch,
        attn_metadata: dict[str, Any],
        slot_mappings: dict[str, torch.Tensor],
        last_hidden_states: torch.Tensor,
        aux_hidden_states: list[torch.Tensor] | None,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        last_sampled: torch.Tensor,
        next_prefill_tokens: torch.Tensor,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
        dp_sync: DPSyncState | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        is_profile: bool = False,
    ) -> torch.Tensor:
        num_reqs = input_batch.num_reqs
        model_drafts = super().propose(
            input_batch,
            attn_metadata,
            slot_mappings,
            last_hidden_states,
            aux_hidden_states,
            num_sampled,
            num_rejected,
            last_sampled,
            next_prefill_tokens,
            temperature,
            seeds,
            dp_sync=dp_sync,
            dummy_run=dummy_run,
            skip_attn_for_dummy_run=skip_attn_for_dummy_run,
            mm_inputs=mm_inputs,
            is_profile=is_profile,
        )
        if dummy_run:
            # The model half still runs, to keep DP/EP ranks in sync, but the
            # n-gram half reads persistent request state that dummy runs may
            # not touch.
            return self.combined_drafts[:num_reqs]

        ngram_drafts = self.ngram.propose(
            input_batch,
            attn_metadata,
            slot_mappings,
            last_hidden_states,
            aux_hidden_states,
            num_sampled,
            num_rejected,
            last_sampled,
            next_prefill_tokens,
            temperature,
            seeds,
        )
        _merge_kernel[(num_reqs,)](
            ngram_drafts,
            model_drafts,
            self.ngram.num_valid_drafts_for_trim,
            input_batch.idx_mapping,
            last_sampled.view(-1),
            self.combined_drafts,
            self.num_valid_drafts_for_trim,
            self.num_ngram_steps, # ngram spec steps
            self.num_speculative_steps, # model spec steps
            self.num_total_steps, # total spec steps
            triton.next_power_of_2(self.num_total_steps),
            num_warps=1,
        )
        return self.combined_drafts[:num_reqs]


@lru_cache
def make_hybrid_speculator_cls(base_cls: type) -> type:
    """Build the `ngram-<x>` speculator class for a model speculator class."""
    if not issubclass(base_cls, DraftModelSpeculator):
        raise NotImplementedError(
            f"Hybrid n-gram speculative decoding needs a model-based drafter, "
            f"but {base_cls.__name__} is not a DraftModelSpeculator."
        )

    return type(f"Ngram{base_cls.__name__}", (NgramHybridMixin, base_cls), {})
