# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.


import os
import sysconfig
from collections import OrderedDict
from pathlib import Path

import torch
import torch_npu
from torch.nn.functional import pad
from vllm.forward_context import get_forward_context
from vllm.triton_utils import HAS_TRITON
from typing import Optional

from Smoothquant.custom_linear_quant import (
    _get_env_int,
    _group_weight_scale,
    _pack_grouped_activation_q,
    _pack_grouped_weight_to_nz,
)
from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.utils import (
    dispose_tensor,
    enable_custom_op,
    get_ascend_device_type,
    get_weight_prefetch_method,
)

ASCEND_DEVICE_TYPE = get_ascend_device_type()

_EXPERT_GMM2_PACK_CACHE: OrderedDict[
    tuple, tuple[torch.Tensor, torch.Tensor]
] = OrderedDict()
_EXPERT_GMM2_STORAGE_TO_PACK_KEY: dict[tuple, tuple] = {}
_EXPERT_GMM2_OP_LOADED = False
_GMM1_VECIN_FULLK_OP_LOADED = False
_EXPERT_GMM2_SHAPE_LOG_CALLS = 0


def _custom_gmm_swiglu_enabled(fusion, dynamic_eplb):
    return fusion and dynamic_eplb and enable_custom_op()


def cumsum_group_list(
    group_list: torch.Tensor, src_list_type: int, dst_list_type: int, active_num: int = 0, expert_num: int = 0
) -> torch.Tensor:
    if src_list_type not in [0, 1, 2]:
        raise ValueError(f"group_list_type should be in [0, 1, 2], but received {src_list_type}")

    if src_list_type == dst_list_type:
        return group_list
    if src_list_type == 1 and dst_list_type == 0:
        return group_list.cumsum(dim=0)
    if src_list_type == 0 and dst_list_type == 1:
        group_diff = torch.diff(group_list)
        new_group = torch.cat([group_list[0].unsqueeze(0), group_diff], dim=0)
        return new_group
    if src_list_type == 2 and dst_list_type == 0:
        experts = pad(group_list[:, 0], (1, 0))
        tokens = pad(group_list[:, 1].cumsum(dim=0), (1, 0))
        cumsum_group_list = torch.full(
            size=(expert_num,), fill_value=active_num, dtype=group_list.dtype, device=group_list.device
        )

        for i, (start, end) in enumerate(zip(experts[:-1], experts[1:])):
            if end > start:
                cumsum_group_list[start:end] = tokens[i]

        return cumsum_group_list
    raise NotImplementedError(
        f"Conversion from src_list_type={src_list_type} to dst_list_type={dst_list_type} is not implemented yet. "
        "This feature is under development."
    )


def _require_single_tensor_for_swiglu_quant(
    tensor_or_list: list[torch.Tensor] | torch.Tensor, *, name: str
) -> torch.Tensor:
    if isinstance(tensor_or_list, list):
        if len(tensor_or_list) != 1:
            raise ValueError(f"{name} must be a tensor or a single-element list, but got {len(tensor_or_list)}.")
        return tensor_or_list[0]
    return tensor_or_list


def _normalize_pertoken_scale(scale: torch.Tensor) -> torch.Tensor:
    if scale.dim() == 2 and scale.size(-1) == 1:
        scale = scale.squeeze(-1)
    return scale.contiguous().to(torch.float32)


def _infer_group_size_from_quant_state(
    weight_q: list[torch.Tensor] | torch.Tensor,
    weight_e: list[torch.Tensor] | torch.Tensor,
) -> int:
    """Recover the effective per-projection group size from packed state."""
    if isinstance(weight_e, list):
        if not weight_e:
            raise ValueError("GroupQuant exponent list must not be empty.")
        exponent = weight_e[0]
    else:
        exponent = weight_e[0] if weight_e.dim() >= 3 else weight_e
    if exponent.dim() < 2:
        raise ValueError(
            f"GroupQuant exponent tensor must have [N, G], got {exponent.shape}."
        )
    output_channels = int(exponent.shape[-2])
    group_count = int(exponent.shape[-1])

    if isinstance(weight_q, list):
        if not weight_q:
            raise ValueError("GroupQuant weight list must not be empty.")
        elements_per_expert = int(weight_q[0].numel())
    else:
        expert_count = int(weight_e.shape[0]) if weight_e.dim() >= 3 else 1
        elements_per_expert = int(weight_q.numel()) // expert_count
    denominator = output_channels * group_count
    if denominator <= 0 or elements_per_expert % denominator != 0:
        raise ValueError(
            "Cannot infer GroupQuant group size from weight/exponent shapes: "
            f"weight_elements={elements_per_expert}, exponent_shape={tuple(exponent.shape)}."
        )
    return elements_per_expert // denominator


def _dynamic_quant_per_token_k_grouped(
    x: torch.Tensor,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token dynamic quant followed by K-group-major activation packing."""
    x_q, token_scale = torch_npu.npu_dynamic_quant(input=x.contiguous())
    return (
        _pack_grouped_activation_q(x_q.contiguous(), group_size),
        _normalize_pertoken_scale(token_scale).reshape(-1),
    )


def _expand_per_token_scale_for_grouped_fallback(
    token_scale: torch.Tensor,
    *,
    num_tokens: int,
    num_groups: int,
) -> torch.Tensor:
    """Adapt native [M] activation scales to the legacy [G, M] fallback ABI."""
    token_scale = _normalize_pertoken_scale(token_scale).reshape(-1)
    if token_scale.numel() != num_tokens:
        raise ValueError(
            f"Per-token activation scale must have {num_tokens} elements, "
            f"but got {token_scale.numel()}."
        )
    return (
        token_scale.reshape(1, num_tokens)
        .expand(num_groups, num_tokens)
        .contiguous()
        .reshape(-1)
    )


def _convert_group_quant_to_weight_scale(
    delta_base: torch.Tensor | list[torch.Tensor],
    e: torch.Tensor | list[torch.Tensor],
    residual_bits: int,
) -> list[torch.Tensor]:
    """Convert GroupQuant parameters to [G, N] weight_scale format for fused kernels."""
    if isinstance(delta_base, list) and isinstance(e, list):
        weight_scales = []
        for db, e_val in zip(delta_base, e):
            # Apply GroupQuant weight scale formula
            weight_scale = _group_weight_scale(db, e_val, residual_bits, True)
            weight_scales.append(weight_scale)
        return weight_scales
    else:
        # Single tensor case
        weight_scale = _group_weight_scale(delta_base, e, residual_bits, True)
        if weight_scale.dim() == 3:
            return [
                weight_scale[expert_idx]
                for expert_idx in range(weight_scale.shape[0])
            ]
        return [weight_scale]


def _prepare_swiglu_group_quant_weight_scale_list(
    weight_scale: torch.Tensor | list[torch.Tensor],
) -> list[torch.Tensor]:
    """Normalize GroupQuant scales to the fused SwiGLU kernel contract.

    The fused tensor-list op expects one scale tensor per expert. For GroupQuant,
    each expert scale must be laid out as [G, N], where G is the K-axis group
    count and N is the logical output width before the SwiGLU halve.
    """
    if isinstance(weight_scale, list):
        scale_list = weight_scale
    elif weight_scale.dim() == 3:
        scale_list = [weight_scale[i] for i in range(weight_scale.shape[0])]
    else:
        scale_list = [weight_scale]

    normalized_scales: list[torch.Tensor] = []
    for scale in scale_list:
        if scale.dim() == 2:
            scale = scale.transpose(0, 1)
        scale = scale.contiguous().to(torch.float32)
        if scale.dim() == 2:
            scale = scale.reshape(-1).contiguous()
        normalized_scales.append(scale)
    return normalized_scales


def _select_expert_tensor(
    tensor_or_list: torch.Tensor | list[torch.Tensor] | None,
    expert_idx: int,
    *,
    name: str,
    expert_first_dim: bool = False,
) -> torch.Tensor:
    if tensor_or_list is None:
        raise ValueError(f"{name} is required for GroupQuant MoE.")
    if isinstance(tensor_or_list, list):
        if not tensor_or_list:
            raise ValueError(f"{name} must not be an empty list.")
        if len(tensor_or_list) == 1:
            tensor = tensor_or_list[0]
            if tensor.dim() <= 2:
                if expert_first_dim and tensor.dim() >= 2:
                    return tensor[expert_idx]
                return tensor
            return tensor[expert_idx]
        return tensor_or_list[expert_idx]
    if tensor_or_list.dim() == 1:
        return tensor_or_list
    if tensor_or_list.dim() == 2:
        if expert_first_dim:
            return tensor_or_list[expert_idx]
        return tensor_or_list
    return tensor_or_list[expert_idx]


def _group_quant_matmul(
    x: torch.Tensor,
    *,
    input_scale: torch.Tensor | None,
    weight_q: torch.Tensor,
    delta_base: torch.Tensor,
    e: torch.Tensor,
    residual_bits: int,
    group_size: int,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    from custom_op.quant_group_matmul.python_api import quant_matmul_k_grouped

    if x.numel() == 0:
        return torch.empty((0, weight_q.shape[0]), device=x.device, dtype=output_dtype)

    input_size = weight_q.shape[-1]
    if input_size % group_size != 0:
        raise ValueError(f"Input size {input_size} must be divisible by group_size {group_size}.")

    num_groups = input_size // group_size
    weight_nz = _pack_grouped_weight_to_nz(weight_q, group_size)
    weight_scale_grouped = _group_weight_scale(
        delta_base=delta_base,
        e=e,
        residual_bits=residual_bits,
        use_moe=True
    )
    weight_scale_grouped = weight_scale_grouped.transpose(0, 1).contiguous().to(torch.float32)
    block_dim = _get_env_int("GROUP_QUANT_BLOCK_DIM", 8)
    parall_num = _get_env_int("GROUP_QUANT_PARALL_NUM", 4)

    if input_scale is not None and x.dtype == torch.int8:
        x_q_packed = _pack_grouped_activation_q(x.contiguous(), group_size)
        pertoken_scale = _expand_per_token_scale_for_grouped_fallback(
            input_scale,
            num_tokens=x.shape[0],
            num_groups=num_groups,
        )
    else:
        x_q_packed, token_scale = _dynamic_quant_per_token_k_grouped(
            x, group_size
        )
        pertoken_scale = _expand_per_token_scale_for_grouped_fallback(
            token_scale,
            num_tokens=x.shape[0],
            num_groups=num_groups,
        )

    out = quant_matmul_k_grouped(
        x_int8=x_q_packed,
        weight_nz_int8=weight_nz,
        scale_f32=weight_scale_grouped,
        pertoken_scale_f32=pertoken_scale,
        k_group_size=group_size,
        k_group_num=num_groups,
        block_dim=block_dim,
        parall_num=parall_num,
    )

    return out.to(output_dtype)


def _expert_count(tensor_or_list: torch.Tensor | list[torch.Tensor]) -> int:
    if isinstance(tensor_or_list, list):
        return len(tensor_or_list)
    if tensor_or_list.dim() <= 2:
        return 1
    return tensor_or_list.shape[0]


def _expert_count_from_group_quant_metadata(
    delta_base: torch.Tensor | list[torch.Tensor],
    e: torch.Tensor | list[torch.Tensor],
    *,
    name: str,
) -> int:
    """Read E from MoE GroupQuant metadata without losing a 2-D expert axis.

    Tensor-backed metadata is laid out as delta_base=[E,N], e=[E,N,G].
    EPLB uses one [N]/[N,G] tensor per expert in a list.  The generic
    ``_expert_count`` helper cannot distinguish [E,N] metadata from a
    single-expert [N,K] weight and therefore intentionally treats a 2-D
    tensor as one expert; using it here made the dummy profile see one GMM2
    weight alongside the router's 128 expert counts.
    """
    if isinstance(delta_base, list) or isinstance(e, list):
        if not isinstance(delta_base, list) or not isinstance(e, list):
            raise ValueError(
                f"{name} delta_base/e must both be tensors or both be lists."
            )
        if not delta_base or len(delta_base) != len(e):
            raise ValueError(
                f"{name} delta_base/e lists must have the same non-zero length."
            )
        return len(delta_base)

    if delta_base.dim() != 2 or e.dim() != 3:
        raise ValueError(
            f"{name} tensor metadata must be delta_base=[E,N] and e=[E,N,G], "
            f"but got {tuple(delta_base.shape)} and {tuple(e.shape)}."
        )
    if delta_base.shape[:2] != e.shape[:2]:
        raise ValueError(
            f"{name} delta_base/e expert and output dimensions must match, "
            f"but got {tuple(delta_base.shape)} and {tuple(e.shape)}."
        )
    return int(delta_base.shape[0])


def _group_quant_expert_matmul(
    x: torch.Tensor,
    *,
    weight_q: torch.Tensor | list[torch.Tensor],
    delta_base: torch.Tensor | list[torch.Tensor],
    e: torch.Tensor | list[torch.Tensor],
    residual_bits: int,
    group_size: int,
    group_list: torch.Tensor,
    group_list_type: int,
    output_dtype: torch.dtype,
    input_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    cumsum_groups = cumsum_group_list(
        group_list,
        group_list_type,
        0,
        active_num=x.shape[0],
        expert_num=_expert_count(weight_q),
    )
    expert_num = cumsum_groups.numel()
    first_weight = _select_expert_tensor(weight_q, 0, name="weight_q")
    out = torch.empty(
        (x.shape[0], first_weight.shape[0]),
        device=x.device,
        dtype=output_dtype,
    )

    start = 0
    normalized_input_scale = (
        _normalize_pertoken_scale(input_scale) if input_scale is not None else None
    )
    for expert_idx in range(expert_num):
        end = int(cumsum_groups[expert_idx].item())
        if end <= start:
            start = end
            continue

        token_slice = slice(start, end)
        expert_x = x[token_slice]
        expert_input_scale = (
            normalized_input_scale[token_slice]
            if normalized_input_scale is not None
            else None
        )
        expert_out = _group_quant_matmul(
            expert_x,
            input_scale=expert_input_scale,
            weight_q=_select_expert_tensor(weight_q, expert_idx, name="weight_q"),
            delta_base=_select_expert_tensor(
                delta_base,
                expert_idx,
                name="delta_base",
                expert_first_dim=True,
            ),
            e=_select_expert_tensor(
                e,
                expert_idx,
                name="e",
                expert_first_dim=True,
            ),
            residual_bits=residual_bits,
            group_size=group_size,
            output_dtype=output_dtype,
        )
        out[token_slice].copy_(expert_out)
        start = end
    return out


def _load_expert_gmm2_op() -> None:
    """Load the optional torch adapter without changing the default path."""
    global _EXPERT_GMM2_OP_LOADED
    if _EXPERT_GMM2_OP_LOADED:
        return
    configured = os.getenv("GROUP_QUANT_EXPERT_GMM2_LIBRARY")
    if configured:
        candidates = [Path(configured)]
    else:
        op_dir = Path(__file__).resolve().parents[2] / "custom_op" / "quant_matmul_expert_k_grouped"
        all_candidates = sorted(
            (op_dir / "build").glob(
                "lib.*/quant_matmul_expert_k_grouped_torch*.so"
            )
        )
        extension_suffix = sysconfig.get_config_var("EXT_SUFFIX")
        candidates = [
            candidate
            for candidate in all_candidates
            if extension_suffix and candidate.name.endswith(extension_suffix)
        ]
    if not candidates or not candidates[-1].is_file():
        raise RuntimeError(
            "QuantMatmulExpertKGrouped torch adapter for the current Python "
            "ABI is unavailable. Build it "
            "with: cd custom_op/quant_matmul_expert_k_grouped && "
            "python setup.py build_ext"
        )
    torch.ops.load_library(str(candidates[-1]))
    if not hasattr(torch.ops._C_ascend, "quant_matmul_expert_k_grouped"):
        raise RuntimeError(
            "QuantMatmulExpertKGrouped adapter loaded, but "
            "torch.ops._C_ascend.quant_matmul_expert_k_grouped was not "
            "registered. Rebuild the custom GMM2 adapter."
        )
    _EXPERT_GMM2_OP_LOADED = True


def _load_gmm1_vecin_fullk_op() -> None:
    """Load the opt-in shape-specialized GMM1 torch adapter."""
    global _GMM1_VECIN_FULLK_OP_LOADED
    if _GMM1_VECIN_FULLK_OP_LOADED:
        return
    configured = os.getenv("GROUP_QUANT_GMM1_VECIN_FULLK_LIBRARY")
    if configured:
        candidates = [Path(configured)]
    else:
        op_dir = Path(__file__).resolve().parents[2] / "custom_op" / "gmm1_vecin_fullk"
        all_candidates = sorted(
            (op_dir / "build").glob("lib.*/gmm1_vecin_fullk_torch*.so")
        )
        extension_suffix = sysconfig.get_config_var("EXT_SUFFIX")
        candidates = [
            candidate
            for candidate in all_candidates
            if extension_suffix and candidate.name.endswith(extension_suffix)
        ]
    if not candidates or not candidates[-1].is_file():
        raise RuntimeError(
            "Gmm1VecinFullK torch adapter for the current Python ABI is "
            "unavailable. Build it with: cd custom_op/gmm1_vecin_fullk && "
            "python setup.py build_ext"
        )
    torch.ops.load_library(str(candidates[-1]))
    if not hasattr(torch.ops._C_ascend, "gmm1_vecin_fullk"):
        raise RuntimeError(
            "Gmm1VecinFullK adapter loaded, but "
            "torch.ops._C_ascend.gmm1_vecin_fullk was not registered."
        )
    _GMM1_VECIN_FULLK_OP_LOADED = True


def _expert_gmm2_pack_key(
    weight_q: torch.Tensor | list[torch.Tensor],
    delta_base: torch.Tensor | list[torch.Tensor],
    e: torch.Tensor | list[torch.Tensor],
    residual_bits: int,
    group_size: int,
) -> tuple:
    def identity(value: torch.Tensor | list[torch.Tensor]) -> tuple:
        tensors = value if isinstance(value, list) else [value]
        return tuple(
            (tensor.device, tensor.data_ptr(), tensor._version)
            for tensor in tensors
        )

    return (
        identity(weight_q),
        identity(delta_base),
        identity(e),
        residual_bits,
        group_size,
    )


def _expert_gmm2_storage_key(
    weight_q: torch.Tensor | list[torch.Tensor],
    delta_base: torch.Tensor | list[torch.Tensor],
    e: torch.Tensor | list[torch.Tensor],
    residual_bits: int,
    group_size: int,
) -> tuple:
    def identity(value: torch.Tensor | list[torch.Tensor]) -> tuple:
        tensors = value if isinstance(value, list) else [value]
        return tuple((tensor.device, tensor.data_ptr()) for tensor in tensors)

    return (
        identity(weight_q),
        identity(delta_base),
        identity(e),
        residual_bits,
        group_size,
    )


def _prepare_expert_gmm2_weight(
    weight_q: torch.Tensor | list[torch.Tensor],
    delta_base: torch.Tensor | list[torch.Tensor],
    e: torch.Tensor | list[torch.Tensor],
    residual_bits: int,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    key = _expert_gmm2_pack_key(
        weight_q, delta_base, e, residual_bits, group_size
    )
    storage_key = _expert_gmm2_storage_key(
        weight_q, delta_base, e, residual_bits, group_size
    )
    stale_key = _EXPERT_GMM2_STORAGE_TO_PACK_KEY.get(storage_key)
    if stale_key is not None and stale_key != key:
        # Dynamic EPLB updates expert tensors in place.  Drop the previous
        # model-sized packed copy immediately instead of retaining one copy per
        # rebalance in the global LRU.
        _EXPERT_GMM2_PACK_CACHE.pop(stale_key, None)
    cached = _EXPERT_GMM2_PACK_CACHE.get(key)
    if cached is not None:
        _EXPERT_GMM2_PACK_CACHE.move_to_end(key)
        return cached

    if isinstance(delta_base, list) or isinstance(e, list):
        if not isinstance(delta_base, list) or not isinstance(e, list):
            raise ValueError("GMM2 delta_base and e must both use tensor-list form.")
        if len(delta_base) != len(e) or not delta_base:
            raise ValueError("GMM2 tensor lists must be non-empty and have equal lengths.")
        normalized_delta_base = torch.stack(delta_base, dim=0)
        normalized_e = torch.stack(e, dim=0)
    else:
        normalized_delta_base = delta_base
        normalized_e = e

    if isinstance(weight_q, list):
        if len(weight_q) != normalized_delta_base.shape[0]:
            raise ValueError("GMM2 weight list length must match expert metadata.")
        # Dynamic EPLB stores already packed per-expert weights.  Concatenation
        # happens only on a cache miss (initial load or an in-place rebalance).
        normalized_weight_q = torch.cat(weight_q, dim=0).contiguous()
    else:
        normalized_weight_q = weight_q

    if (normalized_weight_q.dim() not in (1, 3) or
            normalized_delta_base.dim() != 2 or normalized_e.dim() != 3):
        raise ValueError(
            "QuantMatmulExpertKGrouped requires weight_q [E,N,K] or "
            "prepacked 1-D storage, "
            "delta_base [E,N], and e [E,N,G]."
        )
    if normalized_weight_q.dim() == 1:
        expected_elements = (
            normalized_delta_base.shape[0]
            * normalized_delta_base.shape[1]
            * normalized_e.shape[-1]
            * group_size
        )
        if normalized_weight_q.numel() != expected_elements:
            raise ValueError(
                "Prepacked GMM2 weight element count mismatch: expected "
                f"{expected_elements}, got {normalized_weight_q.numel()}."
            )
        packed_weight = normalized_weight_q.contiguous()
    else:
        packed_weight = torch.cat(
            [
                _pack_grouped_weight_to_nz(
                    normalized_weight_q[expert_idx], group_size
                )
                for expert_idx in range(normalized_weight_q.shape[0])
            ]
        ).contiguous()
    scales = []
    for expert_idx in range(normalized_delta_base.shape[0]):
        scale = _group_weight_scale(
            normalized_delta_base[expert_idx],
            normalized_e[expert_idx],
            residual_bits,
            use_moe=True,
        )
        scales.append(scale.transpose(0, 1))
    packed_scale = torch.cat(scales, dim=0).contiguous().to(torch.float32)
    result = (packed_weight, packed_scale)
    _EXPERT_GMM2_PACK_CACHE[key] = result
    _EXPERT_GMM2_STORAGE_TO_PACK_KEY[storage_key] = key
    cache_limit = max(1, _get_env_int("GROUP_QUANT_EXPERT_GMM2_CACHE_SIZE", 128))
    while len(_EXPERT_GMM2_PACK_CACHE) > cache_limit:
        evicted_key, _ = _EXPERT_GMM2_PACK_CACHE.popitem(last=False)
        stale_storage_keys = [
            storage for storage, pack_key
            in _EXPERT_GMM2_STORAGE_TO_PACK_KEY.items()
            if pack_key == evicted_key
        ]
        for storage in stale_storage_keys:
            del _EXPERT_GMM2_STORAGE_TO_PACK_KEY[storage]
    return result


def _group_quant_expert_matmul_custom(
    x: torch.Tensor,
    *,
    weight_q: torch.Tensor | list[torch.Tensor],
    delta_base: torch.Tensor | list[torch.Tensor],
    e: torch.Tensor | list[torch.Tensor],
    residual_bits: int,
    group_size: int,
    group_list: torch.Tensor,
    group_list_type: int,
    output_dtype: torch.dtype,
    input_scale: torch.Tensor | None = None,
    expert_counts: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run GMM2 as one expert-aware kernel launch."""
    global _EXPERT_GMM2_SHAPE_LOG_CALLS
    first_delta = _select_expert_tensor(
        delta_base, 0, name="delta_base", expert_first_dim=True
    )
    first_e = _select_expert_tensor(
        e, 0, name="e", expert_first_dim=True
    )
    expert_num = _expert_count_from_group_quant_metadata(
        delta_base, e, name="GMM2"
    )
    if first_delta.dim() != 1 or first_e.dim() != 2:
        raise ValueError("GMM2 per-expert delta_base/e must be [N] and [N,G].")
    output_size = first_delta.shape[0]
    input_size = first_e.shape[-1] * group_size
    if x.shape[0] == 0:
        return torch.empty(
            (0, output_size), device=x.device, dtype=output_dtype
        )
    if input_size % group_size != 0:
        raise ValueError(
            f"Input size {input_size} must be divisible by group_size {group_size}."
        )
    _load_expert_gmm2_op()
    if input_scale is None or x.dtype != torch.int8:
        x_q_packed, x_scale = _dynamic_quant_per_token_k_grouped(x, group_size)
    else:
        # Rank 3 identifies native row-major input without copying group slices.
        # Keep the old rank-2 contract for comparisons and the local-int32 trial.
        use_nd = (
            input_size > group_size
            and os.getenv("GROUP_QUANT_GMM2_ND_INPUT", "1") != "0"
            and os.getenv("GROUP_QUANT_GMM2_LOCAL_INT32", "0") == "0"
        )
        if use_nd:
            x_q_packed = x.contiguous().view(x.shape[0], -1, group_size)
        else:
            x_q_packed = _pack_grouped_activation_q(x.contiguous(), group_size)
        x_scale = _normalize_pertoken_scale(input_scale).reshape(-1)
        if x_scale.numel() != x.shape[0]:
            raise ValueError(
                f"GMM2 per-token activation scale must have {x.shape[0]} "
                f"elements, but got {x_scale.numel()}."
            )
    packed_weight, packed_scale = _prepare_expert_gmm2_weight(
        weight_q, delta_base, e, residual_bits, group_size
    )
    if expert_counts is None:
        cumulative_groups = cumsum_group_list(
            group_list,
            group_list_type,
            0,
            active_num=x.shape[0],
            expert_num=expert_num,
        )
        expert_counts = cumsum_group_list(cumulative_groups, 0, 1)
    # The fused GMM1 path already materializes this exact routing form for its
    # TensorList ABI.  Reusing it avoids a second device-side cumsum/cast and
    # gives the future single-launch GMM1->GMM2 operator one stable routing
    # contract: cumulative counts for GMM1, per-expert counts for GMM2.
    expert_counts = expert_counts.contiguous().to(torch.int64)
    if expert_counts.numel() != expert_num:
        raise ValueError(
            f"GMM2 expert_counts must have {expert_num} elements, got "
            f"{expert_counts.numel()}."
        )
    shape_log_limit = int(os.getenv("GROUP_QUANT_GMM2_SHAPE_LOG", "0"))
    if _EXPERT_GMM2_SHAPE_LOG_CALLS < shape_log_limit:
        counts_cpu = expert_counts.cpu()
        nonzero = counts_cpu[counts_cpu > 0]
        print(
            "GroupQuant GMM2 shape: "
            f"M={x.shape[0]}, K={input_size}, N={output_size}, E={expert_num}, "
            f"active_experts={nonzero.numel()}, "
            f"expert_m_min={int(nonzero.min()) if nonzero.numel() else 0}, "
            f"expert_m_max={int(nonzero.max()) if nonzero.numel() else 0}, "
            f"expert_m_mean={float(nonzero.float().mean()) if nonzero.numel() else 0.0:.2f}",
            flush=True,
        )
        _EXPERT_GMM2_SHAPE_LOG_CALLS += 1
    out = torch.ops._C_ascend.quant_matmul_expert_k_grouped(
        x_q_packed.contiguous(),
        packed_weight,
        expert_counts,
        packed_scale,
        x_scale.contiguous().to(torch.float32),
    )
    # GMM2 writes BF16 directly.  Keep the common BF16 inference path free of
    # the former FP16 -> BF16 cast; only non-BF16 callers need a compatibility
    # conversion after the custom operator.
    return out if output_dtype == torch.bfloat16 else out.to(output_dtype)


def _group_quant_swiglu(
    gate_up_out: torch.Tensor,
    *,
    swiglu_limit: float,
) -> torch.Tensor:
    if swiglu_limit and swiglu_limit > 1e-8:
        gate, up = gate_up_out.chunk(2, dim=-1)
        gate = torch.clamp(gate, max=swiglu_limit)
        up = torch.clamp(up, min=-swiglu_limit, max=swiglu_limit)
        return torch.nn.functional.silu(gate) * up
    return torch_npu.npu_swiglu(gate_up_out)

def group_quant_apply_mlp(
    hidden_states: torch.Tensor,
    w1: list[torch.Tensor] | torch.Tensor,
    w2: list[torch.Tensor] | torch.Tensor,
    group_list: torch.Tensor,
    w1_fixpipe_scale: torch.Tensor | list[torch.Tensor] | None = None,
    w1_weight_delta_base: torch.Tensor | list[torch.Tensor] | None = None,
    w1_weight_e: torch.Tensor | list[torch.Tensor] | None = None,
    w2_weight_delta_base: torch.Tensor | list[torch.Tensor] | None = None,
    w2_weight_e: torch.Tensor | list[torch.Tensor] | None = None,
    group_quant_residual_bits: int | None = None,
    group_quant_group_size: int | None = None,
    group_list_type: int = 1,
    dynamic_scale: torch.Tensor | None = None,
    topk_scales: torch.Tensor | None = None,
    activation: str | None = None,
    w1_scale_bias: torch.Tensor | None = None,
    w2_scale_bias: torch.Tensor | None = None,
    w1_offset: torch.Tensor | None = None,
    w2_offset: torch.Tensor | None = None,
    fusion: bool = False,
    dynamic_eplb: bool = False,
    use_mxfp_quant: bool = False,
    act_quant_type: torch.dtype = torch.float8_e4m3fn,
    weight_quant_type: torch.dtype | None = None,
    scale_type: torch.dtype | None = None,
    per_token_scale_type: torch.dtype | None = None,
    use_bf16: bool = True,
    swiglu_limit: float = 0.0,
) -> tuple[torch.Tensor, torch.npu.Event | None]:
    """Apply MoE MLP with GroupQuant weights and fused kernels when available.

    This function supports both the GroupQuant-enhanced fused kernels and
    fallback expert-loop implementations for GroupQuant quantization.
    """

    if use_mxfp_quant:
        raise NotImplementedError("GroupQuant MoE path does not support MXFP activation quantization yet.")
    if w1_offset is not None or w2_offset is not None:
        raise NotImplementedError("GroupQuant MoE path does not support antiquant offset weights.")
    if w1_scale_bias is not None or w2_scale_bias is not None:
        raise NotImplementedError("GroupQuant MoE path does not support scale-bias assisted weights.")
    if group_quant_residual_bits is None or group_quant_group_size is None:
        raise ValueError("GroupQuant MoE requires residual_bits and group_size runtime metadata.")

    w1_group_size = _infer_group_size_from_quant_state(w1, w1_weight_e)
    w2_group_size = _infer_group_size_from_quant_state(w2, w2_weight_e)

    act_name = getattr(activation, "value", activation)
    matmul_output_dtype = torch.bfloat16 if use_bf16 else torch.float16
    final_output_dtype = hidden_states.dtype if hidden_states.dtype in (torch.float16, torch.bfloat16, torch.float32) else matmul_output_dtype

    weight_prefetch_method = get_weight_prefetch_method()
    if weight_prefetch_method:
        weight_prefetch_method.maybe_prefetch_moe_weight_postprocess(hidden_states)

    if act_name not in (None, "silu"):
        raise NotImplementedError(f"GroupQuant MoE path only supports silu/SwiGLU, got {act_name}.")

    # This is deliberately kept outside the fused-GMM1 branch.  The upcoming
    # single-launch GMM1->GMM2 kernel consumes the same two routing views; a
    # non-fused caller must continue to use the existing conversion path.
    gmm2_expert_counts: torch.Tensor | None = None
    # The loading contract identifies native GMM1 without inspecting device
    # values: a complete [E,K,N] NZ tensor plus FP32 [E,N] channel scales.
    # Custom/grouped/EPLB weights retain their TensorList + packed-scale ABI.
    native_gmm1 = (
        isinstance(w1, torch.Tensor) and w1.ndim == 3
        and isinstance(w1_fixpipe_scale, torch.Tensor)
        and w1_fixpipe_scale.dtype == torch.float32
        and w1_fixpipe_scale.ndim == 2
        and w1.dtype == torch.int8
        and w1.shape[1] == 2048 == w1_group_size
        and 0 < w1.shape[2] <= 1536 and w1.shape[2] % 64 == 0
        and w1.shape[0] == w1_fixpipe_scale.shape[0]
        and w1.shape[2] == w1_fixpipe_scale.shape[1]
        and not dynamic_eplb
    )
    if native_gmm1:
        if not fusion:
            raise ValueError("Native NZ GMM1 weights require the fused MLP path")
        if swiglu_limit and swiglu_limit > 1e-8:
            raise NotImplementedError("Native GMM1 does not accept swiglu_limit")
        if dynamic_scale is None:
            unquantized_hidden_states = hidden_states
            hidden_states, pertoken_scale = torch_npu.npu_dynamic_quant(input=hidden_states)
            dispose_tensor(unquantized_hidden_states)
        else:
            pertoken_scale = dynamic_scale
        pertoken_scale = _normalize_pertoken_scale(pertoken_scale)
        cumulative_groups = cumsum_group_list(
            group_list, group_list_type, 0,
            active_num=hidden_states.shape[0], expert_num=w1.shape[0],
        )
        gmm2_expert_counts = cumsum_group_list(cumulative_groups, 0, 1)
        hidden_states, w2_input_scale, _ = torch_npu.npu_grouped_matmul_swiglu_quant(
            hidden_states, w1, cumulative_groups, w1_fixpipe_scale, pertoken_scale,
        )
    elif (
        fusion
        and enable_custom_op()
        and (
            os.getenv("GROUP_QUANT_USE_FUSED_SWIGLU_OP", "0") == "1"
            or os.getenv("GROUP_QUANT_USE_GMM1_VECIN_FULLK", "0") == "1"
        )
    ):
        if dynamic_scale is None:
            unquantized_hidden_states = hidden_states
            hidden_states, pertoken_scale = torch_npu.npu_dynamic_quant(
                input=hidden_states,
            )
            dispose_tensor(unquantized_hidden_states)
        else:
            pertoken_scale = dynamic_scale
        if pertoken_scale is not None:
            pertoken_scale = _normalize_pertoken_scale(pertoken_scale)

        if isinstance(w1, list):
            w1_list = w1
        else:
            # Compatibility path for callers that did not pre-convert the
            # model weights during process_weights_after_loading().
            w1_list = [
                torch_npu.npu_format_cast(
                    w1[expert_idx].transpose(0, 1).contiguous(), 29
                )
                for expert_idx in range(w1.shape[0])
            ]
        if swiglu_limit and swiglu_limit > 1e-8:
            raise NotImplementedError(
                "The installed fused GMM1 ABI does not accept swiglu_limit."
            )
        cumulative_groups = cumsum_group_list(
            group_list, group_list_type, 0,
            active_num=hidden_states.shape[0],
            expert_num=_expert_count_from_group_quant_metadata(
                w1_weight_delta_base, w1_weight_e, name="GMM1"
            ),
        )
        gmm2_expert_counts = cumsum_group_list(cumulative_groups, 0, 1)
        # Derive TP-local N from the weight, not from scale.numel(): grouped
        # quantization scales may carry an additional K-group dimension and
        # must not be mistaken for a wider output tensor.
        local_full_n = (
            w1_list[0].numel() // 2048
            if w1_list and w1_list[0].numel() % 2048 == 0 else 0
        )
        packed_scale_items = (
            w1_fixpipe_scale
            if isinstance(w1_fixpipe_scale, list)
            else (
                [w1_fixpipe_scale[0]]
                if w1_fixpipe_scale is not None and w1_fixpipe_scale.ndim > 0
                else []
            )
        )
        packed_scale_expert_num = (
            len(packed_scale_items)
            if isinstance(w1_fixpipe_scale, list)
            else (
                int(w1_fixpipe_scale.shape[0])
                if w1_fixpipe_scale is not None and w1_fixpipe_scale.ndim > 0
                else 0
            )
        )
        packed_scale_per_expert = (
            packed_scale_items[0].numel()
            if packed_scale_items and packed_scale_items[0] is not None
            else 0
        )
        k_group_num = (
            packed_scale_per_expert // local_full_n
            if local_full_n > 0
            and packed_scale_per_expert % local_full_n == 0
            else 0
        )
        # Compatibility callers may not have model-load-time FixPipe scales.
        # Build FP32 scales only for that path; production inference supplies
        # packed INT64 scales and must not recreate them on every forward.
        w1_weight_scale_list = None
        if w1_fixpipe_scale is None:
            w1_weight_scale = _convert_group_quant_to_weight_scale(
                w1_weight_delta_base,
                w1_weight_e,
                group_quant_residual_bits,
            )
            w1_weight_scale_list = _prepare_swiglu_group_quant_weight_scale_list(
                w1_weight_scale
            )
            k_group_num = (
                w1_weight_scale_list[0].numel() // local_full_n
                if local_full_n > 0 and w1_weight_scale_list and
                w1_weight_scale_list[0].numel() % local_full_n == 0 else 0
            )
        elif not (
            isinstance(w1_fixpipe_scale, torch.Tensor)
            and w1_fixpipe_scale.dtype == torch.int64
        ):
            # Standard fused GMM1 consumes FP32 [G, N]-packed scales.  The
            # GroupQuant loader materializes this TensorList once, so avoid
            # recreating it from delta/e on every forward.
            w1_weight_scale_list = (
                [scale.contiguous().to(torch.float32)
                 for scale in w1_fixpipe_scale]
                if isinstance(w1_fixpipe_scale, list)
                else [w1_fixpipe_scale.contiguous().to(torch.float32)]
            )
        use_gmm1_vecin_fullk = (
            os.getenv("GROUP_QUANT_USE_GMM1_VECIN_FULLK", "0") == "1"
            and hidden_states.ndim == 2
            and hidden_states.shape[1] == 2048
            and 0 < local_full_n <= 1536
            and local_full_n % 64 == 0
            and k_group_num > 0
            and 2048 % k_group_num == 0
            and (2048 // k_group_num) % 16 == 0
            and all(weight.numel() == 2048 * local_full_n for weight in w1_list)
            and (
                (
                    w1_fixpipe_scale is not None
                    and packed_scale_expert_num == len(w1_list)
                    and all(
                        scale is not None
                        and scale.dtype == torch.int64
                        and scale.numel() == k_group_num * local_full_n
                        for scale in packed_scale_items
                    )
                )
                or (
                    w1_weight_scale_list is not None
                    and all(
                        scale.numel() == k_group_num * local_full_n
                        for scale in w1_weight_scale_list
                    )
                )
            )
        )
        if use_gmm1_vecin_fullk:
            _load_gmm1_vecin_fullk_op()
            if w1_fixpipe_scale is None:
                # Compatibility for direct callers. Model loading precomputes
                # this tensor so production inference does not launch
                # npu_trans_quant_param on every layer/forward.
                packed_fp32_scale = torch.stack(
                    [scale.reshape(k_group_num, local_full_n)
                     for scale in w1_weight_scale_list],
                    dim=0,
                ).contiguous().to(torch.float32)
                packed_w1_weight_scale = torch_npu.npu_trans_quant_param(
                    packed_fp32_scale
                ).contiguous()
            elif isinstance(w1_fixpipe_scale, list):
                packed_w1_weight_scale = torch.stack(
                    [scale.reshape(k_group_num, local_full_n)
                     for scale in w1_fixpipe_scale],
                    dim=0,
                ).contiguous()
            else:
                packed_w1_weight_scale = w1_fixpipe_scale.reshape(
                    len(w1_list), k_group_num, local_full_n
                ).contiguous()
            if packed_w1_weight_scale.dtype != torch.int64:
                raise ValueError(
                    "GMM1 FixPipe scale must be packed INT64 from "
                    "torch_npu.npu_trans_quant_param."
                )
            hidden_states, w2_input_scale = torch.ops._C_ascend.gmm1_vecin_fullk(
                hidden_states,
                w1_list,
                packed_w1_weight_scale,
                pertoken_scale,
                cumulative_groups,
            )
        else:
            if w1_weight_scale_list is None:
                w1_weight_scale = _convert_group_quant_to_weight_scale(
                    w1_weight_delta_base,
                    w1_weight_e,
                    group_quant_residual_bits,
                )
                w1_weight_scale_list = _prepare_swiglu_group_quant_weight_scale_list(
                    w1_weight_scale
                )
            hidden_states, w2_input_scale, _ = torch.ops._C_ascend.grouped_matmul_swiglu_quant_weight_nz_tensor_list(
                x=hidden_states,
                weight=w1_list,
                weight_scale=w1_weight_scale_list,
                x_scale=pertoken_scale,
                group_list=cumulative_groups,
            )
    else:
        hidden_states = _group_quant_expert_matmul(
            hidden_states,
            weight_q=w1,
            delta_base=w1_weight_delta_base,
            e=w1_weight_e,
            residual_bits=group_quant_residual_bits,
            group_size=w1_group_size,
            group_list=group_list,
            group_list_type=group_list_type,
            input_scale=dynamic_scale,
            output_dtype=matmul_output_dtype,
        )
        hidden_states = _group_quant_swiglu(
            hidden_states,
            swiglu_limit=swiglu_limit,
        )
        w2_input_scale = None

    if topk_scales is not None:
        if w2_input_scale is not None:
            raise NotImplementedError(
                "The fused GroupQuant GMM1 path does not support topk_scales yet."
            )
        topk_scales = topk_scales if topk_scales.dim() > 1 else topk_scales.reshape(-1, 1)
        hidden_states = hidden_states * topk_scales.to(hidden_states.dtype)

    # The GroupQuant integration consumes only the tensor result. Creating an
    # event on every layer adds stream bookkeeping and the event was discarded
    # by unified_apply_mlp, so do not allocate one here.
    before_gmm2_evt = None
    use_expert_gmm2_op = (
        os.getenv("GROUP_QUANT_USE_EXPERT_GMM2_OP", "0") == "1"
    )
    if os.getenv("GROUP_QUANT_BENCH_NATIVE_GMM2", "0") == "1":
        if (not isinstance(w2, torch.Tensor) or w2.ndim != 3
                or w2_input_scale is None or w2_weight_e.shape[-1] != 1):
            raise ValueError("Native benchmark GMM2 requires single-group NZ weights and quantized input")
        hidden_states = torch_npu.npu_grouped_matmul(
            x=[hidden_states], weight=[w2], scale=[w2_weight_delta_base],
            per_token_scale=[w2_input_scale], group_list=group_list,
            group_list_type=group_list_type, split_item=2, group_type=0,
            output_dtype=matmul_output_dtype,
        )[0].to(final_output_dtype)
    elif use_expert_gmm2_op:
        hidden_states = _group_quant_expert_matmul_custom(
            hidden_states,
            weight_q=w2,
            delta_base=w2_weight_delta_base,
            e=w2_weight_e,
            residual_bits=group_quant_residual_bits,
            group_size=w2_group_size,
            group_list=group_list,
            group_list_type=group_list_type,
            input_scale=w2_input_scale,
            output_dtype=matmul_output_dtype,
            # Only the custom fused GMM1 path has produced the compatible
            # route table above.  The fallback path preserves its original
            # independent routing conversion.
            expert_counts=(gmm2_expert_counts
                           if w2_input_scale is not None else None),
        )
        if hidden_states.dtype != final_output_dtype:
            hidden_states = hidden_states.to(final_output_dtype)
    else:
        hidden_states = _group_quant_expert_matmul(
            hidden_states,
            weight_q=w2,
            delta_base=w2_weight_delta_base,
            e=w2_weight_e,
            residual_bits=group_quant_residual_bits,
            group_size=w2_group_size,
            group_list=group_list,
            group_list_type=group_list_type,
            input_scale=w2_input_scale,
            output_dtype=matmul_output_dtype,
        ).to(final_output_dtype)

    return hidden_states, before_gmm2_evt

def quant_apply_mlp(hidden_states: torch.Tensor,
                    w1: list[torch.Tensor],
                    w1_scale: list[torch.Tensor],
                    w2: list[torch.Tensor],
                    w2_scale: list[torch.Tensor],
                    group_list: torch.Tensor,
                    group_list_type: int = 1,
                    dynamic_scale: torch.Tensor = None,
                    w1_scale_bias: torch.Tensor = None,
                    w2_scale_bias: torch.Tensor = None,
                    w1_offset: Optional[torch.Tensor] = None,
                    w2_offset: Optional[torch.Tensor] = None,
                    fusion: bool = False,
                    dynamic_eplb: bool = False) -> torch.Tensor:
    if w1_offset is not None:
        unquantized_hidden_states = hidden_states
        quantized_hidden_states = None
    elif dynamic_scale is None:
        unquantized_hidden_states = hidden_states
        hidden_states, pertoken_scale = torch_npu.npu_dynamic_quant(
            hidden_states)
        # Dispose the original unquantized hidden states
        # to save npu memory because they're no longer used.
        dispose_tensor(unquantized_hidden_states)
        quantized_hidden_states = None
    else:
        unquantized_hidden_states = None
        pertoken_scale = dynamic_scale
        quantized_hidden_states = hidden_states

    bias1, bias2 = None, None
    _output_dtype = w2_scale[0].dtype

    weight_prefetch_method = get_weight_prefetch_method()
    if weight_prefetch_method:
        weight_prefetch_method.maybe_prefetch_moe_weight_postprocess(
            hidden_states)
    is_mc2 = get_forward_context().moe_comm_type == MoECommType.MC2
    if w1_scale_bias is None and w1_offset is None and is_mc2:
        if _custom_gmm_swiglu_enabled(fusion, dynamic_eplb):
            # gmm1: gate_up_proj & act_fn: swiglu
            hidden_states, swiglu_out_scale, _ = (
                torch.ops._C_ascend.
                grouped_matmul_swiglu_quant_weight_nz_tensor_list(
                    x=hidden_states,
                    weight=w1,
                    weight_scale=w1_scale,
                    x_scale=pertoken_scale,
                    group_list=cumsum_group_list(group_list, group_list_type,
                                                 0),
                ))
        elif fusion and not dynamic_eplb:
            # gmm1: gate_up_proj & act_fn: swiglu
            hidden_states, swiglu_out_scale, _ = torch_npu.npu_grouped_matmul_swiglu_quant(
                x=hidden_states,
                weight=w1[0],
                group_list=cumsum_group_list(group_list, group_list_type, 0),
                weight_scale=w1_scale[0],
                x_scale=pertoken_scale)
            if quantized_hidden_states is not None:
                dispose_tensor(quantized_hidden_states)
        else:
            if w1_scale[0].dtype != torch.float32:
                w1_scale[0] = w1_scale[0].to(torch.float32)
            # gmm1: gate_up_proj
            hidden_states = torch_npu.npu_grouped_matmul(
                x=[hidden_states],
                weight=w1,
                split_item=3,
                group_list_type=group_list_type,
                group_type=0,
                group_list=group_list,
                output_dtype=torch.int32)[0]
            if quantized_hidden_states is not None:
                dispose_tensor(quantized_hidden_states)
            # act_fn: swiglu
            hidden_states, swiglu_out_scale = torch_npu.npu_dequant_swiglu_quant(
                x=hidden_states,
                weight_scale=w1_scale[0],
                activation_scale=pertoken_scale,
                bias=None,
                quant_scale=None,
                quant_offset=None,
                group_index=cumsum_group_list(group_list, group_list_type, 1),
                activate_left=True,
                quant_mode=1,
            )
        # gmm2: down_proj
        hidden_states = torch_npu.npu_grouped_matmul(
            x=[hidden_states],
            weight=w2,
            scale=w2_scale,
            per_token_scale=[swiglu_out_scale],
            split_item=2,
            group_list_type=group_list_type,
            group_type=0,
            group_list=group_list,
            output_dtype=w2_scale[0].dtype)[0]
    elif w1_offset is not None:
        # gmm1: gate_up_proj
        hidden_states = torch_npu.npu_grouped_matmul(
            x=[unquantized_hidden_states],
            weight=[w1],
            antiquant_scale=[w1_scale],
            antiquant_offset=[w1_offset],
            split_item=2,
            group_list_type=group_list_type,
            group_type=0,
            group_list=group_list,
            output_dtype=_output_dtype)[0]
        dispose_tensor(unquantized_hidden_states)
        # act_fn: swiglu
        hidden_states = torch_npu.npu_swiglu(hidden_states)
        # gmm2: down_proj
        hidden_states = torch_npu.npu_grouped_matmul(
            x=[hidden_states],
            weight=[w2],
            antiquant_scale=[w2_scale],
            antiquant_offset=[w2_offset],
            split_item=2,
            group_list_type=group_list_type,
            group_type=0,
            group_list=group_list,
            output_dtype=_output_dtype)[0]
    else:
        if w1_scale_bias is not None:
            if group_list_type == 0:
                group_list = torch.cat(
                    [group_list[:1],
                     torch.diff(group_list, dim=0)])
                group_list_type = 1
            bias1 = [w1_scale_bias] if not fusion else w1_scale_bias
            bias2 = [w2_scale_bias]
            # TODO w4a8 scene: dynamic acquisition of dtype in the future
            _output_dtype = torch.bfloat16

        if _custom_gmm_swiglu_enabled(fusion, dynamic_eplb):
            # gmm1: gate_up_proj & act_fn: swiglu
            hidden_states, swiglu_out_scale, _ = (
                torch.ops._C_ascend.
                grouped_matmul_swiglu_quant_weight_nz_tensor_list(
                    x=hidden_states,
                    weight=w1,
                    weight_scale=w1_scale,
                    x_scale=pertoken_scale,
                    group_list=cumsum_group_list(group_list, group_list_type,
                                                 0),
                    bias=bias1,
                ))
        elif fusion and not dynamic_eplb:
            # gmm1: gate_up_proj & act_fn: swiglu
            hidden_states, swiglu_out_scale, _ = torch_npu.npu_grouped_matmul_swiglu_quant(
                x=hidden_states,
                weight=w1[0],
                bias=bias1,
                group_list=cumsum_group_list(group_list, group_list_type, 0),
                weight_scale=w1_scale[0],
                x_scale=pertoken_scale)
            if quantized_hidden_states is not None:
                dispose_tensor(quantized_hidden_states)
        else:
            w1_scale[0] = w1_scale[0].to(w2_scale[0].dtype)
            # gmm1: gate_up_proj
            hidden_states = torch_npu.npu_grouped_matmul(
                x=[hidden_states],
                weight=w1,
                scale=w1_scale,
                bias=bias1,
                per_token_scale=[pertoken_scale],
                split_item=2,
                group_list_type=group_list_type,
                group_type=0,
                group_list=group_list,
                output_dtype=_output_dtype)[0]
            if quantized_hidden_states is not None:
                dispose_tensor(quantized_hidden_states)
            # act_fn: swiglu
            if HAS_TRITON:
                from vllm_ascend.ops.triton.activation.swiglu_quant import \
                    swiglu_quant
                hidden_states, swiglu_out_scale = swiglu_quant(
                    hidden_states,
                    group_list=group_list,
                    group_list_type=group_list_type)
            else:
                hidden_states = torch_npu.npu_swiglu(hidden_states)
                hidden_states, swiglu_out_scale = torch_npu.npu_dynamic_quant(
                    hidden_states)
        # gmm2: down_proj
        hidden_states = torch_npu.npu_grouped_matmul(
            x=[hidden_states],
            weight=w2,
            scale=w2_scale,
            bias=bias2,
            per_token_scale=[swiglu_out_scale],
            split_item=2,
            group_list_type=group_list_type,
            group_type=0,
            group_list=group_list,
            output_dtype=_output_dtype)[0]
    return hidden_states


def unquant_apply_mlp(hidden_states: torch.Tensor,
                      w1: torch.Tensor,
                      w2: torch.Tensor,
                      group_list: torch.Tensor,
                      group_list_type: int = 1,
                      topk_scales: Optional[torch.Tensor] = None,
                      need_trans: bool = True) -> torch.Tensor:

    if need_trans:
        w1 = w1.transpose(1, 2)
        w2 = w2.transpose(1, 2)

    gate_up_out = torch_npu.npu_grouped_matmul(
        x=[hidden_states],
        weight=[w1],
        split_item=2,
        group_list_type=group_list_type,
        group_type=0,
        group_list=group_list,
    )[0]
    if get_ascend_device_type() == AscendDeviceType._310P:
        gate_up_out = torch_npu.npu_swiglu(gate_up_out.to(torch.float32)).to(
            torch.float16)
    else:
        gate_up_out = torch_npu.npu_swiglu(gate_up_out)

    if topk_scales is not None:
        gate_up_out *= topk_scales

    hidden_states = torch_npu.npu_grouped_matmul(
        x=[gate_up_out],
        weight=[w2],
        split_item=2,
        group_list_type=group_list_type,
        group_type=0,
        group_list=group_list,
    )[0]
    return hidden_states


def unified_apply_mlp(
    hidden_states: torch.Tensor,
    w1: torch.Tensor | list[torch.Tensor],
    w2: torch.Tensor | list[torch.Tensor],
    group_list: torch.Tensor,
    w1_scale: list[torch.Tensor] | torch.Tensor | None = None,
    w2_scale: list[torch.Tensor] | torch.Tensor | None = None,
    dynamic_scale: torch.Tensor | None = None,
    group_list_type: int = 1,
    w1_scale_bias: torch.Tensor | None = None,
    w2_scale_bias: torch.Tensor | None = None,
    w1_offset: torch.Tensor | None = None,
    w2_offset: torch.Tensor | None = None,
    topk_scales: torch.Tensor | None = None,
    with_quant: bool = False,
    fusion: bool = False,
    need_trans: bool = True,
    dynamic_eplb: bool = False,
    activation: str | None = "silu",
    group_quant: bool = False,
    w1_bias: torch.Tensor | None = None,
    w2_bias: torch.Tensor | None = None,
    w1_weight_delta_base: torch.Tensor | list[torch.Tensor] | None = None,
    w1_weight_e: torch.Tensor | list[torch.Tensor] | None = None,
    w2_weight_delta_base: torch.Tensor | list[torch.Tensor] | None = None,
    w2_weight_e: torch.Tensor | list[torch.Tensor] | None = None,
    group_quant_residual_bits: int | None = None,
    group_quant_group_size: int | None = None,
    use_mxfp_quant: bool = False,
    act_quant_type: torch.dtype = torch.float8_e4m3fn,
    weight_quant_type: torch.dtype | None = None,
    scale_type: torch.dtype | None = None,
    per_token_scale_type: torch.dtype | None = None,
    use_bf16: bool | None = None,
    swiglu_limit: float = 0.0,
    use_w4a8_per_channel_gmm_swiglu: bool = False,
) -> torch.Tensor:
    use_bf16 = hidden_states.dtype == torch.bfloat16 if use_bf16 is None else use_bf16

    if group_quant:
        hidden_states, _ = group_quant_apply_mlp(
            hidden_states=hidden_states,
            w1=w1,
            w1_fixpipe_scale=w1_scale,
            w1_weight_delta_base=w1_weight_delta_base,
            w1_weight_e=w1_weight_e,
            w2=w2,
            w2_weight_delta_base=w2_weight_delta_base,
            w2_weight_e=w2_weight_e,
            group_quant_residual_bits=group_quant_residual_bits,
            group_quant_group_size=group_quant_group_size,
            group_list=group_list,
            group_list_type=group_list_type,
            dynamic_scale=dynamic_scale,
            topk_scales=topk_scales,
            activation=activation,
            w1_scale_bias=w1_scale_bias,
            w2_scale_bias=w2_scale_bias,
            w1_offset=w1_offset,
            w2_offset=w2_offset,
            fusion=fusion,
            dynamic_eplb=dynamic_eplb,
            use_mxfp_quant=use_mxfp_quant,
            act_quant_type=act_quant_type,
            weight_quant_type=weight_quant_type,
            scale_type=scale_type,
            per_token_scale_type=per_token_scale_type,
            use_bf16=use_bf16,
            swiglu_limit=swiglu_limit,
        )
        return hidden_states

    if with_quant:
        assert w1_scale is not None and w2_scale is not None
        hidden_states, _ = quant_apply_mlp(
            hidden_states=hidden_states,
            w1=w1,
            w1_scale=w1_scale,
            w2=w2,
            w2_scale=w2_scale,
            group_list=group_list,
            dynamic_scale=dynamic_scale,
            group_list_type=group_list_type,
            w1_scale_bias=w1_scale_bias,
            w2_scale_bias=w2_scale_bias,
            w1_offset=w1_offset,
            w2_offset=w2_offset,
            fusion=fusion,
            dynamic_eplb=dynamic_eplb,
            use_mxfp_quant=use_mxfp_quant,
            act_quant_type=act_quant_type,
            weight_quant_type=weight_quant_type,
            scale_type=scale_type,
            per_token_scale_type=per_token_scale_type,
            use_bf16=use_bf16,
            swiglu_limit=swiglu_limit,
            use_w4a8_per_channel_gmm_swiglu=use_w4a8_per_channel_gmm_swiglu,
        )
        return hidden_states

    hidden_states, _ = unquant_apply_mlp(
        hidden_states=hidden_states,
        w1=w1,
        w2=w2,
        group_list=group_list,
        w1_bias=w1_bias,
        w2_bias=w2_bias,
        activation=activation,
        group_list_type=group_list_type,
        topk_scales=topk_scales,
        need_trans=need_trans,
    )
    return hidden_states
