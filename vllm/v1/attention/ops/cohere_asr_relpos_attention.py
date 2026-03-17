# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Triton fast path for Cohere ASR relative-position encoder attention."""

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import RCP_LN2
from vllm.utils.torch_utils import direct_register_custom_op


@triton.jit
def _cohere_asr_relpos_fwd_kernel(
    Q_U,
    Q_V,
    K,
    V,
    P,
    MASK,
    sm_scale,
    center_pos,
    pos_len,
    B_Start_Loc,
    B_Seqlen,
    Out,
    stride_qbs,
    stride_qh,
    stride_kbs,
    stride_kh,
    stride_vbs,
    stride_vh,
    stride_ph,
    stride_pp,
    stride_mbatch,
    stride_mq,
    stride_mk,
    stride_obs,
    stride_oh,
    HAS_MASK: tl.constexpr,
    MASK_BATCH_BROADCAST: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    start_m = tl.program_id(2)

    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_start = tl.load(B_Start_Loc + cur_batch)
    block_start_loc = BLOCK_M * start_m

    offs_m = block_start_loc + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    row_ids = tl.arange(0, BLOCK_M)

    q_valid = offs_m < cur_batch_seq_len
    d_valid = offs_d < HEAD_SIZE

    q_offsets = (
        (cur_batch_start + offs_m[:, None]) * stride_qbs
        + cur_head * stride_qh
        + offs_d[None, :]
    )
    q_u = tl.load(
        Q_U + q_offsets,
        mask=q_valid[:, None] & d_valid[None, :],
        other=0.0,
    )
    q_v = tl.load(
        Q_V + q_offsets,
        mask=q_valid[:, None] & d_valid[None, :],
        other=0.0,
    )

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    end_n = cur_batch_seq_len
    for start_n in range(0, end_n, BLOCK_N):
        key_pos = start_n + offs_n
        k_valid = key_pos < cur_batch_seq_len

        k_offsets = (
            (cur_batch_start + key_pos[None, :]) * stride_kbs
            + cur_head * stride_kh
            + offs_d[:, None]
        )
        v_offsets = (
            (cur_batch_start + key_pos[:, None]) * stride_vbs
            + cur_head * stride_vh
            + offs_d[None, :]
        )

        k = tl.load(
            K + k_offsets,
            mask=d_valid[:, None] & k_valid[None, :],
            other=0.0,
        )
        v = tl.load(
            V + v_offsets,
            mask=k_valid[:, None] & d_valid[None, :],
            other=0.0,
        )

        qk = tl.dot(q_u, k).to(tl.float32)
        rel_qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

        # Each query row needs a shifted contiguous slice of the relative
        # position table: center_pos + key_pos - query_pos.
        for row_idx in range(BLOCK_M):
            query_pos = block_start_loc + row_idx
            q_valid_row = query_pos < cur_batch_seq_len
            rel_pos = center_pos + key_pos - query_pos
            rel_valid = q_valid_row & k_valid & (rel_pos >= 0) & (rel_pos < pos_len)

            q_v_row_offsets = (
                (cur_batch_start + query_pos) * stride_qbs
                + cur_head * stride_qh
                + offs_d
            )
            q_v_row = tl.load(
                Q_V + q_v_row_offsets,
                mask=d_valid & q_valid_row,
                other=0.0,
            )

            p_offsets = (
                cur_head * stride_ph
                + rel_pos[None, :] * stride_pp
                + offs_d[:, None]
            )
            p_row = tl.load(
                P + p_offsets,
                mask=d_valid[:, None] & rel_valid[None, :],
                other=0.0,
            )
            rel_row = tl.sum(
                q_v_row.to(tl.float32)[:, None] * p_row.to(tl.float32),
                axis=0,
            )
            rel_qk = tl.where(row_ids[:, None] == row_idx, rel_row[None, :], rel_qk)

        score_mask = q_valid[:, None] & k_valid[None, :]
        if HAS_MASK:
            mask_batch = 0 if MASK_BATCH_BROADCAST else cur_batch
            mask_offsets = (
                mask_batch * stride_mbatch
                + offs_m[:, None] * stride_mq
                + key_pos[None, :] * stride_mk
            )
            blocked = tl.load(
                MASK + mask_offsets,
                mask=score_mask,
                other=True,
            )
            score_mask &= ~blocked

        qk = (qk + rel_qk) * sm_scale
        masked_qk = tl.where(score_mask, qk, -1.0e8)
        m_ij = tl.maximum(m_i, tl.max(masked_qk, axis=1))
        p = tl.where(score_mask, tl.math.exp2(masked_qk - m_ij[:, None]), 0.0)
        l_ij = tl.sum(p, axis=1)

        alpha = tl.math.exp2(m_i - m_ij)
        acc = acc * alpha[:, None]
        l_i = l_i * alpha + l_ij
        acc = tl.dot(p.to(v.dtype), v, acc)
        m_i = m_ij

    acc = tl.where(l_i[:, None] > 0, acc / l_i[:, None], 0.0)

    out_offsets = (
        (cur_batch_start + offs_m[:, None]) * stride_obs
        + cur_head * stride_oh
        + offs_d[None, :]
    )
    tl.store(
        Out + out_offsets,
        acc,
        mask=q_valid[:, None] & d_valid[None, :],
    )


def _get_query_block_size(dtype: torch.dtype) -> int:
    return 8 if dtype == torch.float32 else 16


def _get_key_block_size(dtype: torch.dtype) -> int:
    if dtype == torch.float32:
        return 32
    if current_platform.is_cuda_alike() and current_platform.has_device_capability(80):
        return 64
    return 32


def cohere_asr_triton_relpos_attention(
    q_u: torch.Tensor,
    q_v: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    p: torch.Tensor,
    seq_lens: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Args:
        q_u, q_v, k, v: [batch, head, seq_len, head_dim]
        p: [head, 2 * seq_len - 1, head_dim]
        seq_lens: [batch] valid sequence lengths
        attn_mask: optional bool tensor [1|batch, seq_len, seq_len]
    Returns:
        [batch, head, seq_len, head_dim]
    """
    batch_size, num_heads, q_len, head_dim = q_u.shape
    assert q_v.shape == q_u.shape
    assert k.shape == q_u.shape
    assert v.shape == q_u.shape
    assert p.shape == (num_heads, 2 * q_len - 1, head_dim)
    if head_dim < 16 or head_dim > 128:
        raise ValueError(
            "cohere_asr_triton_relpos_attention supports head_dim in [16, 128]."
        )

    q_u_flat = q_u.transpose(1, 2).contiguous().view(batch_size * q_len, num_heads, head_dim)
    q_v_flat = q_v.transpose(1, 2).contiguous().view(batch_size * q_len, num_heads, head_dim)
    k_flat = k.transpose(1, 2).contiguous().view(batch_size * q_len, num_heads, head_dim)
    v_flat = v.transpose(1, 2).contiguous().view(batch_size * q_len, num_heads, head_dim)
    p = p.contiguous()
    seq_lens = seq_lens.to(dtype=torch.int32, device=q_u.device).contiguous()
    attn_mask = attn_mask.contiguous() if attn_mask is not None else None

    out_flat = torch.zeros_like(q_u_flat)
    start_loc = torch.arange(
        0,
        batch_size * q_len,
        step=q_len,
        dtype=torch.int32,
        device=q_u.device,
    )

    block_m = _get_query_block_size(q_u.dtype)
    block_n = _get_key_block_size(q_u.dtype)
    num_warps = 4 if head_dim <= 64 else 8
    sm_scale = (1.0 / (head_dim**0.5)) * RCP_LN2

    grid = (batch_size, num_heads, triton.cdiv(q_len, block_m))
    _cohere_asr_relpos_fwd_kernel[grid](
        q_u_flat,
        q_v_flat,
        k_flat,
        v_flat,
        p,
        attn_mask,
        sm_scale,
        q_len - 1,
        p.size(1),
        start_loc,
        seq_lens,
        out_flat,
        q_u_flat.stride(0),
        q_u_flat.stride(1),
        k_flat.stride(0),
        k_flat.stride(1),
        v_flat.stride(0),
        v_flat.stride(1),
        p.stride(0),
        p.stride(1),
        0 if attn_mask is None else attn_mask.stride(0),
        0 if attn_mask is None else attn_mask.stride(1),
        0 if attn_mask is None else attn_mask.stride(2),
        out_flat.stride(0),
        out_flat.stride(1),
        HAS_MASK=attn_mask is not None,
        MASK_BATCH_BROADCAST=attn_mask is None or attn_mask.size(0) == 1,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_DMODEL=triton.next_power_of_2(head_dim),
        HEAD_SIZE=head_dim,
        num_warps=num_warps,
        num_stages=1,
    )

    return out_flat.view(batch_size, q_len, num_heads, head_dim).transpose(1, 2)


def cohere_asr_triton_relpos_attention_fake(
    q_u: torch.Tensor,
    q_v: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    p: torch.Tensor,
    seq_lens: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    return torch.empty_like(q_u)


direct_register_custom_op(
    op_name="cohere_asr_triton_relpos_attention",
    op_func=cohere_asr_triton_relpos_attention,
    fake_impl=cohere_asr_triton_relpos_attention_fake,
)


def cohere_asr_triton_relpos_attention_wrapper(
    q_u: torch.Tensor,
    q_v: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    p: torch.Tensor,
    seq_lens: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    return torch.ops.vllm.cohere_asr_triton_relpos_attention(
        q_u,
        q_v,
        k,
        v,
        p,
        seq_lens,
        attn_mask,
    )
