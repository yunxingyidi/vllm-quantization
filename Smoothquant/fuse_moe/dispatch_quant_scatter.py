"""Opt-in quantize-once dispatch for local BF16 AllGather MoE inputs.

The native router handles ordering and counts using a one-column carrier. The
custom kernel quantizes each original token once and scatters data and scales
with that exact mapping. Unsupported communication/shape contracts fall back.
"""
from __future__ import annotations
import os
from pathlib import Path
import torch

_ENABLED = False


def initialize() -> None:
    global _ENABLED
    if os.getenv('GROUP_QUANT_DISPATCH_SCATTER', '0') != '1':
        return
    path = Path(__file__).resolve().parents[2] / 'custom_op/dispatch_quant_scatter/build/libdispatch_quant_scatter_ops.so'
    if not path.is_file():
        raise RuntimeError(f'Build dispatch_quant_scatter before enabling it: {path}')
    torch.ops.load_library(str(path))
    _ENABLED = True


def maybe_dispatch(dispatcher, *, hidden_states, topk_weights, topk_ids,
                   expert_map, log2phy, apply_router_weight_on_input,
                   dynamic_eplb, pertoken_scale):
    if (not _ENABLED
            or type(dispatcher).__name__ != 'TokenDispatcherWithAllGather'
            or hidden_states.dtype != torch.bfloat16 or hidden_states.ndim != 2
            or not hidden_states.is_contiguous()
            or hidden_states.shape[0] < 2048
            or hidden_states.shape[1] < 64 or hidden_states.shape[1] > 8192
            or hidden_states.shape[1] % 64
            or topk_ids.ndim != 2 or topk_ids.shape[0] != hidden_states.shape[0]
            or not 0 < topk_ids.shape[1] <= 64
            or expert_map is not None or log2phy is not None
            or apply_router_weight_on_input or dynamic_eplb or pertoken_scale is not None):
        return None
    # Preserve upstream metadata and dtype conventions. The first column is
    # only an opaque routing carrier, never an activation for the experts.
    output = dispatcher.token_dispatch(
        hidden_states=hidden_states[:, :1].contiguous(),
        topk_weights=topk_weights, topk_ids=topk_ids, with_quant=False,
    )
    mapping = output['context_metadata']['expanded_row_idx']
    q, scale = torch.ops.group_quant.dispatch_quant_scatter(
        hidden_states, mapping, topk_ids.shape[1])
    dispatcher.original_shape = hidden_states.shape
    dispatcher.with_quant = True
    output['hidden_states'] = q
    output['dynamic_scale'] = scale
    return output
