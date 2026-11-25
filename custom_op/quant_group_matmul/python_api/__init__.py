import os
import warnings
from pathlib import Path

import torch

from . import quant_group_matmul_ext

_WARNED_ASCENDC_DYNAMIC_QUANT = False


def load_ops_library() -> str:
    """Return the loaded extension path that registers torch.ops.vllm_smooth."""
    return str(Path(quant_group_matmul_ext.__file__).resolve())


def dynamic_quant_k_grouped(
    x: torch.Tensor,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    K-axis grouped activation dynamic quant prototype.

    Expected:
      x: [M, K] fp16/bf16 on NPU
      group_size: K-group width
    Returns:
      x_int8: [k_group_num * M, group_size] int8 on NPU
      x_scale: [k_group_num, M] float32 on NPU

    This entry defaults to the stage-1 "C++ pack + torch_npu.npu_dynamic_quant"
    reference path. Set GROUP_QUANT_DYNAMIC_QUANT_BACKEND=ascendc to use the
    scalar AscendC correctness prototype.
    """
    import torch_npu

    if x.dim() != 2:
        raise ValueError("x must be 2D [M, K].")
    if x.size(1) % group_size != 0:
        raise ValueError(f"K={x.size(1)} must be divisible by group_size={group_size}.")
    if os.getenv("GROUP_QUANT_DYNAMIC_QUANT_BACKEND", "torch_npu") == "ascendc":
        global _WARNED_ASCENDC_DYNAMIC_QUANT
        if not _WARNED_ASCENDC_DYNAMIC_QUANT:
            warnings.warn(
                "GROUP_QUANT_DYNAMIC_QUANT_BACKEND=ascendc selects the scalar "
                "AscendC dynamic-quant prototype. It is for correctness/debug "
                "only and is much slower than the default torch_npu backend.",
                RuntimeWarning,
                stacklevel=2,
            )
            _WARNED_ASCENDC_DYNAMIC_QUANT = True
        return torch.ops.vllm_smooth.dynamic_quant_k_grouped_ascendc(
            x,
            group_size,
            int(os.getenv("GROUP_QUANT_DYNAMIC_QUANT_BLOCK_DIM", "24")),
        )

    group_num = x.size(1) // group_size
    x_grouped = quant_group_matmul_ext.pack_activation_k_grouped(x, group_size)
    x_q, x_scale = torch_npu.npu_dynamic_quant(x_grouped)
    return x_q, x_scale.view(group_num, x.size(0)).contiguous()


def quant_matmul_grouped(
    x_int8: torch.Tensor,
    weight_nz_int8: torch.Tensor,
    scale_f32: torch.Tensor,
    pertoken_scale_f32: torch.Tensor,
    group_list: torch.Tensor,
    group_axis: int = 0,
    k_group_size: int = 0,
    k_group_num: int = 0,
    block_dim: int = 8,
    parall_num: int = 4,
) -> torch.Tensor:
    """
    Grouped quant matmul wrapper.
    Expected:
      x_int8: [M, K] int8 on NPU
      weight_nz_int8: grouped NZ-packed int8 weight buffer on NPU
      scale_f32:
        - [group_num, N] for current M-axis grouping
        - [k_group_num, N] for pure K-axis grouping
        - [expert_num * k_group_num, N] for expert x K-axis grouping
      pertoken_scale_f32: [M] float32 on NPU
      group_list:
        - [group_num] int64 on NPU, sum(group_list) == M for M-axis grouping
        - [1] int64 on NPU with value M for pure K-axis grouping
        - [expert_num] int64 on NPU, sum(group_list) == M for expert x K-axis grouping
      group_axis:
        - 0 for current M-axis grouping
        - 1 for pure K-axis grouping
        - 2 for expert x K-axis grouping
      k_group_size: K-axis group size
      k_group_num: K-axis group count
    Returns:
      y: [M, N] float16 on NPU

    Notes:
      - `group_axis=0` uses the current device kernel directly.
      - `group_axis=1` accumulates K-groups for one shared weight bank.
      - `group_axis=2` accumulates K-groups independently inside each expert
        segment described by `group_list`.
    """
    return torch.ops.vllm_smooth.quant_matmul_grouped(
        x_int8,
        weight_nz_int8,
        scale_f32,
        pertoken_scale_f32,
        group_list,
        group_axis,
        k_group_size,
        k_group_num,
        block_dim,
        parall_num,
    )


def quant_matmul_k_grouped(
    x_int8: torch.Tensor,
    weight_nz_int8: torch.Tensor,
    scale_f32: torch.Tensor,
    pertoken_scale_f32: torch.Tensor,
    k_group_size: int,
    k_group_num: int,
    block_dim: int = 8,
    parall_num: int = 4,
) -> torch.Tensor:
    """
    K-axis grouped quant matmul wrapper.

    This avoids constructing `group_list` in Python. The extension infers the
    logical M from `x_int8.shape[0] // k_group_num` and reuses a cached NPU
    group-list tensor for launch.
    """
    return torch.ops.vllm_smooth.quant_matmul_k_grouped(
        x_int8,
        weight_nz_int8,
        scale_f32,
        pertoken_scale_f32,
        k_group_size,
        k_group_num,
        block_dim,
        parall_num,
    )


def quant_matmul_expert_k_grouped(
    x_int8: torch.Tensor,
    weight_nz_int8: torch.Tensor,
    scale_f32: torch.Tensor,
    pertoken_scale_f32: torch.Tensor,
    group_list: torch.Tensor,
    k_group_size: int,
    k_group_num: int,
    block_dim: int = 8,
    parall_num: int = 4,
) -> torch.Tensor:
    """Expert-aware K-axis grouped quant matmul wrapper."""
    return quant_group_matmul_ext.forward_grouped(
        x_int8,
        weight_nz_int8,
        scale_f32,
        pertoken_scale_f32,
        group_list,
        2,
        k_group_size,
        k_group_num,
        block_dim,
        parall_num,
    )


def quant_matmul_k_grouped_dynamic(
    x: torch.Tensor,
    weight_nz_int8: torch.Tensor,
    scale_f32: torch.Tensor,
    group_size: int,
    block_dim: int = 8,
    parall_num: int = 4,
) -> torch.Tensor:
    """
    Prototype fused entry for K-axis grouped dynamic activation quant + matmul.

    Today this is an interface-level fusion: it still calls dynamic quant and
    quant matmul as two device ops, but hides the packed activation layout from
    vLLM integration code. The next optimization can replace this body with a
    single AscendC fused kernel without changing callers.
    """
    if os.getenv("GROUP_QUANT_USE_TILE_DYNAMIC_MATMUL", "0") == "1":
        return torch.ops.vllm_smooth.quant_matmul_k_grouped_dynamic(
            x,
            weight_nz_int8,
            scale_f32,
            group_size,
            block_dim,
            parall_num,
        )
    if os.getenv("GROUP_QUANT_USE_EXTERNAL_DYNAMIC_MATMUL", "0") == "1":
        return quant_group_matmul_ext.forward_k_grouped_dynamic_external_fused(
            x,
            weight_nz_int8,
            scale_f32,
            group_size,
            block_dim,
            parall_num,
        )
    if os.getenv("GROUP_QUANT_USE_EXTERNAL_DYNAMIC_2KERNEL", "0") == "1":
        return quant_group_matmul_ext.forward_k_grouped_dynamic_external_2kernel(
            x,
            weight_nz_int8,
            scale_f32,
            group_size,
            block_dim,
            parall_num,
        )
    if os.getenv("GROUP_QUANT_USE_SINGLE_KERNEL_DYNAMIC_MATMUL", "0") == "1":
        return quant_group_matmul_ext.forward_k_grouped_dynamic_fused(
            x,
            weight_nz_int8,
            scale_f32,
            group_size,
            block_dim,
            parall_num,
        )
    if os.getenv("GROUP_QUANT_USE_CPP_DYNAMIC_MATMUL", "0") == "1":
        try:
            return quant_group_matmul_ext.forward_k_grouped_dynamic(
                x,
                weight_nz_int8,
                scale_f32,
                group_size,
                block_dim,
                parall_num,
            )
        except Exception as exc:
            warnings.warn(
                "C++ GroupQuant dynamic matmul wrapper failed; falling back to "
                f"Python two-step wrapper. Error: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
    x_int8, pertoken_scale = dynamic_quant_k_grouped(x, group_size)
    k_group_num = x.size(1) // group_size
    return quant_matmul_k_grouped(
        x_int8=x_int8,
        weight_nz_int8=weight_nz_int8,
        scale_f32=scale_f32,
        pertoken_scale_f32=pertoken_scale.reshape(-1).contiguous().to(torch.float32),
        k_group_size=group_size,
        k_group_num=k_group_num,
        block_dim=block_dim,
        parall_num=parall_num,
    )


def quant_matmul_expert_k_grouped_dynamic(
    x_q: torch.Tensor,
    weight_nz_int8: torch.Tensor,
    scale_f32: torch.Tensor,
    pertoken_scale_f32: torch.Tensor,
    group_list: torch.Tensor,
    k_group_size: int,
    k_group_num: int,
    block_dim: int = 8,
    parall_num: int = 4,
) -> torch.Tensor:
    """Expert-aware grouped-weight matmul using packed int8 activations."""
    return quant_matmul_expert_k_grouped(
        x_int8=x_q,
        weight_nz_int8=weight_nz_int8,
        scale_f32=scale_f32,
        pertoken_scale_f32=pertoken_scale_f32,
        group_list=group_list,
        k_group_size=k_group_size,
        k_group_num=k_group_num,
        block_dim=block_dim,
        parall_num=parall_num,
    )


def debug_quant_matmul_k_grouped_passthrough_fused(
    x_int8: torch.Tensor,
    weight_nz_int8: torch.Tensor,
    scale_f32: torch.Tensor,
    pertoken_scale_f32: torch.Tensor,
    k_group_size: int,
    k_group_num: int,
    block_dim: int = 8,
    parall_num: int = 4,
) -> torch.Tensor:
    """Debug path: reuse fused-kernel workspace layout without dynamic quant."""
    return quant_group_matmul_ext.debug_forward_k_grouped_passthrough_fused(
        x_int8,
        weight_nz_int8,
        scale_f32,
        pertoken_scale_f32,
        k_group_size,
        k_group_num,
        block_dim,
        parall_num,
    )


def debug_quant_matmul_k_grouped_workspace_passthrough_fused(
    x_int8: torch.Tensor,
    weight_nz_int8: torch.Tensor,
    scale_f32: torch.Tensor,
    pertoken_scale_f32: torch.Tensor,
    k_group_size: int,
    k_group_num: int,
    block_dim: int = 8,
    parall_num: int = 4,
) -> torch.Tensor:
    """Debug path: copy quantized inputs into fused workspace before matmul."""
    return quant_group_matmul_ext.debug_forward_k_grouped_workspace_passthrough_fused(
        x_int8,
        weight_nz_int8,
        scale_f32,
        pertoken_scale_f32,
        k_group_size,
        k_group_num,
        block_dim,
        parall_num,
    )


def debug_quant_matmul_k_grouped_dynamic_external_fused(
    x: torch.Tensor,
    weight_nz_int8: torch.Tensor,
    scale_f32: torch.Tensor,
    k_group_size: int,
    block_dim: int = 8,
    parall_num: int = 4,
) -> torch.Tensor:
    """Debug path: dynamic quant writes external xQ/per-token-scale tensors."""
    return quant_group_matmul_ext.forward_k_grouped_dynamic_external_fused(
        x,
        weight_nz_int8,
        scale_f32,
        k_group_size,
        block_dim,
        parall_num,
    )


def debug_quant_matmul_k_grouped_dynamic_external_2kernel(
    x: torch.Tensor,
    weight_nz_int8: torch.Tensor,
    scale_f32: torch.Tensor,
    k_group_size: int,
    block_dim: int = 8,
    parall_num: int = 4,
) -> torch.Tensor:
    """Debug path: dynamic quant writes external tensors, next kernel matmul reads them."""
    return quant_group_matmul_ext.forward_k_grouped_dynamic_external_2kernel(
        x,
        weight_nz_int8,
        scale_f32,
        k_group_size,
        block_dim,
        parall_num,
    )


def debug_tile_dynamic_pack(
    x: torch.Tensor,
    k_group_size: int,
    block_dim: int = 8,
    parall_num: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Debug tile dynamic vector/MTE pack output as external xQ and scale tensors."""
    return quant_group_matmul_ext.debug_tile_dynamic_pack(
        x,
        k_group_size,
        block_dim,
        parall_num,
    )


def quant_matmul(
    x_int8: torch.Tensor,
    weight_nz_int8: torch.Tensor,
    scale_f32: torch.Tensor,
    pertoken_scale_f32: torch.Tensor,
    block_dim: int = 8,
    parall_num: int = 4,
) -> torch.Tensor:
    """
    Single-group quant matmul wrapper built from groupmatmul kernel.
    Expected:
      x_int8: [M, K] int8 on NPU
      weight_nz_int8: NZ-packed int8 weight buffer on NPU
      scale_f32: [N] float32 on NPU
      pertoken_scale_f32: [M] float32 on NPU
    Returns:
      y: [M, N] float16 on NPU
    """
    group_list = torch.full(
        (1,),
        x_int8.size(0),
        dtype=torch.int64,
        device=x_int8.device,
    )
    if scale_f32.dim() == 1:
        scale_f32 = scale_f32.unsqueeze(0)
    return quant_matmul_grouped(
        x_int8,
        weight_nz_int8,
        scale_f32,
        pertoken_scale_f32,
        group_list,
        0,
        0,
        0,
        block_dim,
        parall_num,
    )


def quant_matmul_int8(
    x_int8: torch.Tensor,
    weight_nz_int8: torch.Tensor,
    scale_f32: torch.Tensor,
    pertoken_scale_f32: torch.Tensor,
    bias_int8: torch.Tensor | None = None,
    block_dim: int = 8,
    parall_num: int = 4,
) -> torch.Tensor:
    """
    INT8 output variant.
    Expected:
      x_int8: [M, K] int8 on NPU
      weight_nz_int8: NZ-packed int8 weight buffer on NPU
      scale_f32: [N] float32 on NPU
      pertoken_scale_f32: [M] float32 on NPU
      bias_int8: [N] int8 on NPU (optional)
    Returns:
      y_int8: [M, N] int8 on NPU
    """
    group_list = torch.full(
        (1,),
        x_int8.size(0),
        dtype=torch.int64,
        device=x_int8.device,
    )
    if scale_f32.dim() == 1:
        scale_f32 = scale_f32.unsqueeze(0)
    return quant_group_matmul_ext.forward_int8_grouped(
        x_int8,
        weight_nz_int8,
        scale_f32,
        pertoken_scale_f32,
        group_list,
        0,
        0,
        0,
        bias_int8,
        block_dim,
        parall_num,
    )
