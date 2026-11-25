# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.16)
find_package(ASC REQUIRED)
project(dispatch_quant_scatter LANGUAGES ASC CXX)
set(CMAKE_CXX_STANDARD 17)
find_package(Python3 COMPONENTS Interpreter REQUIRED)
execute_process(COMMAND ${Python3_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE TORCH_PREFIX)
find_package(Torch REQUIRED HINTS ${TORCH_PREFIX})
execute_process(COMMAND ${Python3_EXECUTABLE} -c "import os, torch_npu; print(os.path.dirname(torch_npu.__file__))"
    OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE TORCH_NPU_PATH)
add_library(dispatch_quant_scatter_ops SHARED
    op_kernel/dispatch_quant_scatter_kernel.asc
    op_extension/dispatch_quant_scatter_torch.cpp)
target_include_directories(dispatch_quant_scatter_ops PRIVATE
    ${TORCH_INCLUDE_DIRS} ${TORCH_NPU_PATH}/include
    $ENV{ASCEND_HOME_PATH}/aarch64-linux/include)
target_link_directories(dispatch_quant_scatter_ops PRIVATE
    ${TORCH_NPU_PATH}/lib $ENV{ASCEND_HOME_PATH}/lib64
    $ENV{ASCEND_HOME_PATH}/aarch64-linux/lib64)
target_link_libraries(dispatch_quant_scatter_ops PRIVATE
    torch_npu ascendcl ascendc_runtime dl)
target_compile_options(dispatch_quant_scatter_ops PRIVATE
    ${TORCH_CXX_FLAGS} $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-2201>)
