import os
from pathlib import Path
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension
import torch_npu


ROOT = Path(__file__).resolve().parent
KERNEL_ROOT = ROOT.parent
OUT_DIR = KERNEL_ROOT / "out"
TORCH_NPU_ROOT = Path(torch_npu.__file__).resolve().parent
TORCH_NPU_LIB = TORCH_NPU_ROOT / "lib"

ASCEND_HOME = Path(
    os.environ.get(
        "ASCEND_HOME_PATH",
        os.environ.get("ASCEND_CANN_PACKAGE_PATH", "/usr/local/Ascend/ascend-toolkit/latest"),
    )
).resolve()

ext = CppExtension(
    name="quant_group_matmul_ext",
    sources=[
        str(ROOT / "quant_group_matmul_ext.cpp"),
        str(KERNEL_ROOT / "quant_group_matmul_custom_tiling.cpp"),
    ],
    include_dirs=[
        str(KERNEL_ROOT),
        str(OUT_DIR / "include"),
        str(OUT_DIR / "include" / "ascendc_kernels_npu"),
        str(ASCEND_HOME / "include"),
        str(ASCEND_HOME / "include" / "aclnn"),
        str(TORCH_NPU_ROOT / "include"),
        str(TORCH_NPU_ROOT / "include" / "third_party" / "acl" / "inc"),
    ],
    library_dirs=[
        str(OUT_DIR / "lib64"),
        str(ASCEND_HOME / "lib64"),
        str(TORCH_NPU_LIB),
    ],
    libraries=[
        "torch_npu",
        "ascendc_kernels_npu",
        "ascendcl",
        "nnopbase",
        "tiling_api",
        "register",
        "platform",
        "ascendalog",
    ],
    extra_compile_args=["-O2", "-std=c++17", "-fPIC"],
    extra_link_args=[
        "-Wl,-rpath," + str(OUT_DIR / "lib64"),
        "-Wl,-rpath," + str(ASCEND_HOME / "lib64"),
    ],
)


setup(
    name="quant_group_matmul_ext",
    version="0.1.0",
    ext_modules=[ext],
    cmdclass={"build_ext": BuildExtension},
)
