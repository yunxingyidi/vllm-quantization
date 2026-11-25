"""
GroupQuant-enhanced grouped_matmul_swiglu_quant_weight_nz_tensor_list operator

This module provides the GroupQuant-enhanced version of the fused MoE operator
that supports both per-token and group-based quantization.
"""

try:
    # Import the custom algorithm if available
    from .grouped_matmul_swiglu_quant_torch_adpt import (
        grouped_matmul_swiglu_quant_weight_nz_tensor_list
    )

    __all__ = ['grouped_matmul_swiglu_quant_weight_nz_tensor_list']

except ImportError as e:
    import warnings
    warnings.warn(
        f"GroupQuant-enhanced algorithm not available: {e}. "
        "Falling back to system default.",
        RuntimeWarning
    )
    __all__ = []