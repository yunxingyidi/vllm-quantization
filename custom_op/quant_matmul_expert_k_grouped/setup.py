import os
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension
from torch_npu.utils.cpp_extension import NpuExtension


HERE = Path(__file__).resolve().parent
VLLM_ASCEND_CSRC = Path(
    os.environ.get(
        "VLLM_ASCEND_CSRC",
        str(HERE.parents[2] / "vllm-ascend" / "csrc"),
    )
).resolve()
if not (VLLM_ASCEND_CSRC / "aclnn_torch_adapter").is_dir():
    raise RuntimeError(
        "Cannot find vllm-ascend/csrc. Set VLLM_ASCEND_CSRC to the csrc "
        f"directory; resolved path was {VLLM_ASCEND_CSRC}."
    )

setup(
    name="quant_matmul_expert_k_grouped_torch",
    ext_modules=[
        NpuExtension(
            name="quant_matmul_expert_k_grouped_torch",
            sources=[
                str(HERE / "torch_adapter.cpp"),
                str(VLLM_ASCEND_CSRC / "aclnn_torch_adapter/NPUBridge.cpp"),
                str(VLLM_ASCEND_CSRC / "aclnn_torch_adapter/NPUStorageImpl.cpp"),
            ],
            include_dirs=[str(VLLM_ASCEND_CSRC)],
            extra_compile_args=["-O2", "-std=c++17"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
