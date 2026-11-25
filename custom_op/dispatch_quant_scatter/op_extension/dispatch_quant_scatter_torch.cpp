/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */


#include "acl/acl.h"
#include <ATen/ATen.h>
#include <torch/library.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/core/npu/NPUGuard.h"

void dispatch_quant_scatter_kernel(uint32_t, void*, aclrtStream,
    uint8_t*, uint8_t*, uint8_t*, uint8_t*, uint32_t, uint32_t, uint32_t);

using Result = std::tuple<at::Tensor, at::Tensor>;
Result dispatch_quant_scatter(const at::Tensor& x, const at::Tensor& mapping, int64_t topk) {
    TORCH_CHECK(x.is_privateuseone() && mapping.device() == x.device(), "same NPU required");
    TORCH_CHECK(x.scalar_type() == at::kBFloat16 && x.dim() == 2 && x.is_contiguous(), "contiguous BF16 [M,H] required");
    TORCH_CHECK(mapping.scalar_type() == at::kInt && mapping.is_contiguous() && mapping.dim() == 1, "contiguous INT32 mapping required");
    TORCH_CHECK(topk > 0 && topk <= 64 && mapping.numel() == x.size(0) * topk, "invalid top-k or mapping size");
    TORCH_CHECK(x.size(1) >= 64 && x.size(1) <= 8192 && x.size(1) % 64 == 0, "H must be 64-aligned and <=8192");
    TORCH_CHECK(x.size(0) * topk < INT32_MAX, "INT32 row-index capacity exceeded");
    const c10_npu::NPUGuard deviceGuard(x.device());
    auto y = at::empty({x.size(0) * topk, x.size(1)}, x.options().dtype(at::kChar));
    auto scales = at::empty({x.size(0) * topk}, x.options().dtype(at::kFloat));
    if (!x.size(0)) return {y, scales};
    int32_t device = -1;
    TORCH_CHECK(aclrtGetDevice(&device) == ACL_SUCCESS, "get device failed");
    int64_t cores = 0;
    TORCH_CHECK(aclrtGetDeviceInfo(device, ACL_DEV_ATTR_VECTOR_CORE_NUM, &cores) == ACL_SUCCESS && cores > 0, "query vector cores failed");
    const uint32_t blocks = static_cast<uint32_t>(std::min(cores, x.size(0)));
    auto stream = c10_npu::getCurrentNPUStream().stream(true);
    // stream(true) drains prior host tasks before this direct ACL launch.
    // No tensor-owning handler is retained beyond the launch.
    dispatch_quant_scatter_kernel(blocks, nullptr, stream,
        (uint8_t*)x.data_ptr(), (uint8_t*)mapping.data_ptr(),
        (uint8_t*)y.data_ptr(), (uint8_t*)scales.data_ptr(),
        static_cast<uint32_t>(x.size(0)), static_cast<uint32_t>(x.size(1)), static_cast<uint32_t>(topk));
    return {y, scales};
}
Result dispatch_quant_scatter_meta(const at::Tensor& x, const at::Tensor&, int64_t topk) {
    return {at::empty({x.size(0)*topk,x.size(1)},x.options().dtype(at::kChar)),
            at::empty({x.size(0)*topk},x.options().dtype(at::kFloat))};
}
TORCH_LIBRARY_FRAGMENT(group_quant, m) {
    m.def("dispatch_quant_scatter(Tensor x, Tensor mapping, int topk) -> (Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(group_quant, PrivateUse1, m) {
    m.impl("dispatch_quant_scatter", TORCH_FN(dispatch_quant_scatter));
}
TORCH_LIBRARY_IMPL(group_quant, Meta, m) {
    m.impl("dispatch_quant_scatter", TORCH_FN(dispatch_quant_scatter_meta));
}
