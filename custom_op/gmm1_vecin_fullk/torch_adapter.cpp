#include <torch/extension.h>
#include <torch/library.h>

#include <cstdlib>
#include <cstring>
#include <dlfcn.h>

#include "aclnn_torch_adapter/op_api_common.h"

namespace {
void *GetVllmSmoothCustOpApiFuncAddr(const char *apiName)
{
    static void *legacyHandler = []() -> void * {
        const char *path = std::getenv("GROUP_QUANT_CUSTOM_OPAPI_PATH");
        if (path == nullptr || path[0] == '\0') return nullptr;
        return dlopen(path, RTLD_LAZY | RTLD_LOCAL);
    }();
    return legacyHandler == nullptr ? nullptr : dlsym(legacyHandler, apiName);
}
} // namespace

#define EXEC_VLLM_SMOOTH_NPU_CMD(aclnn_api, ...)                              \
    do {                                                                       \
        static const auto getWorkspaceSizeFuncAddr =                           \
            GetVllmSmoothCustOpApiFuncAddr(#aclnn_api "GetWorkspaceSize");    \
        static const auto opApiFuncAddr =                                      \
            GetVllmSmoothCustOpApiFuncAddr(#aclnn_api);                        \
        static const auto initMemAddr =                                        \
            GetOpApiFuncAddr("InitHugeMemThreadLocal");                       \
        static const auto unInitMemAddr =                                      \
            GetOpApiFuncAddr("UnInitHugeMemThreadLocal");                     \
        static const auto releaseMemAddr = GetOpApiFuncAddr("ReleaseHugeMem"); \
        TORCH_CHECK(                                                           \
            getWorkspaceSizeFuncAddr != nullptr && opApiFuncAddr != nullptr,   \
            #aclnn_api, " is not exported by GROUP_QUANT_CUSTOM_OPAPI_PATH=", \
            std::getenv("GROUP_QUANT_CUSTOM_OPAPI_PATH") == nullptr            \
                ? "<unset>" : std::getenv("GROUP_QUANT_CUSTOM_OPAPI_PATH"));  \
        auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);         \
        uint64_t workspace_size = 0;                                           \
        uint64_t *workspace_size_addr = &workspace_size;                       \
        aclOpExecutor *executor = nullptr;                                     \
        aclOpExecutor **executor_addr = &executor;                             \
        InitHugeMemThreadLocal initMemFunc =                                   \
            reinterpret_cast<InitHugeMemThreadLocal>(initMemAddr);             \
        UnInitHugeMemThreadLocal unInitMemFunc =                               \
            reinterpret_cast<UnInitHugeMemThreadLocal>(unInitMemAddr);         \
        if (initMemFunc) initMemFunc(nullptr, false);                           \
        auto converted_params =                                                \
            ConvertTypes(__VA_ARGS__, workspace_size_addr, executor_addr);     \
        static auto getWorkspaceSizeFunc =                                     \
            ConvertToOpApiFunc(converted_params, getWorkspaceSizeFuncAddr);    \
        auto workspace_status = call(getWorkspaceSizeFunc, converted_params);  \
        TORCH_CHECK(workspace_status == 0,                                     \
                    "call " #aclnn_api " failed, detail:", aclGetRecentErrMsg()); \
        void *workspace_addr = nullptr;                                        \
        at::Tensor workspace_tensor;                                           \
        if (workspace_size != 0) {                                             \
            at::TensorOptions options =                                        \
                at::TensorOptions(torch_npu::utils::get_npu_device_type());    \
            workspace_tensor = at::empty(                                      \
                {static_cast<int64_t>(workspace_size)}, options.dtype(kByte)); \
            workspace_addr = const_cast<void *>(workspace_tensor.storage().data()); \
        }                                                                      \
        auto acl_call = [converted_params, workspace_addr, workspace_size,     \
                         acl_stream, executor]() -> int {                      \
            typedef int (*OpApiFunc)(void *, uint64_t, aclOpExecutor *,        \
                                     const aclrtStream);                       \
            OpApiFunc opApiFunc = reinterpret_cast<OpApiFunc>(opApiFuncAddr);  \
            auto api_ret =                                                     \
                opApiFunc(workspace_addr, workspace_size, executor, acl_stream); \
            TORCH_CHECK(api_ret == 0, "call " #aclnn_api " failed, detail:", \
                        aclGetRecentErrMsg());                                 \
            ReleaseConvertTypes(converted_params);                            \
            ReleaseHugeMem releaseMemFunc =                                   \
                reinterpret_cast<ReleaseHugeMem>(releaseMemAddr);              \
            if (releaseMemFunc) releaseMemFunc(nullptr, false);                \
            return api_ret;                                                    \
        };                                                                     \
        at_npu::native::OpCommand cmd;                                         \
        cmd.Name(#aclnn_api);                                                  \
        cmd.SetCustomHandler(acl_call);                                        \
        cmd.Run();                                                             \
        if (unInitMemFunc) unInitMemFunc(nullptr, false);                       \
    } while (false)

namespace vllm_smooth {
std::tuple<at::Tensor, at::Tensor> gmm1_vecin_fullk(
    const at::Tensor &x,
    const at::TensorList &weight,
    const at::Tensor &weight_scale,
    const at::Tensor &x_scale,
    const at::Tensor &group_list)
{
    TORCH_CHECK(x.dim() == 2 && x.size(1) == 2048 && x.scalar_type() == at::kChar,
                "x must be INT8 [M, 2048].");
    TORCH_CHECK(!weight.empty(), "weight must be a non-empty list.");
    TORCH_CHECK(group_list.dim() == 1 &&
                    static_cast<size_t>(group_list.numel()) == weight.size() &&
                    group_list.scalar_type() == at::kLong,
                "group_list must be INT64 [expert_num].");
    TORCH_CHECK(x_scale.dim() == 1 && x_scale.numel() == x.size(0) &&
                    x_scale.scalar_type() == at::kFloat,
                "x_scale must be FP32 [M].");
    TORCH_CHECK(weight[0].numel() % 2048 == 0,
                "weight items must contain a whole number of local N channels.");
    const int64_t full_n = weight[0].numel() / 2048;
    TORCH_CHECK(full_n > 0 && full_n <= 1536 && full_n % 64 == 0,
                "local weight N must be a multiple of 64 in (0, 1536].");
    const int64_t scale_denominator = static_cast<int64_t>(weight.size()) * full_n;
    TORCH_CHECK(weight_scale.numel() % scale_denominator == 0,
                "weight_scale must contain expert_count*G*localN values.");
    const int64_t k_group_num = weight_scale.numel() / scale_denominator;
    TORCH_CHECK(k_group_num > 0 && 2048 % k_group_num == 0 &&
                    (2048 / k_group_num) % 16 == 0,
                "K-group count must divide 2048 and K-group size must be 16-aligned.");
    for (size_t expert = 0; expert < weight.size(); ++expert) {
        const auto &w = weight[expert];
        TORCH_CHECK(w.numel() == 2048 * full_n && w.scalar_type() == at::kChar,
                    "weight items must contain 2048*localN INT8 values; ACLNN "
                    "validates FRACTAL_NZ storage [localN/32,128,16,32].");
        TORCH_CHECK(w.device() == x.device(), "all inputs must be on the same NPU device.");
    }
    TORCH_CHECK(weight_scale.scalar_type() == at::kLong && weight_scale.is_contiguous(),
                "weight_scale must be packed INT64 from npu_trans_quant_param "
                "[expert,G,localN].");
    TORCH_CHECK(weight_scale.device() == x.device() && x_scale.device() == x.device() &&
                    group_list.device() == x.device(),
                "all inputs must be on the same NPU device.");
    auto output = at::empty({x.size(0), full_n / 2}, x.options().dtype(at::kChar));
    auto output_scale = at::empty({x.size(0)}, x.options().dtype(at::kFloat));
    EXEC_VLLM_SMOOTH_NPU_CMD(
        aclnnGmm1VecinFullk,
        x,
        weight,
        weight_scale,
        x_scale,
        group_list,
        output,
        output_scale);
    return {output, output_scale};
}

} // namespace vllm_smooth

TORCH_LIBRARY_FRAGMENT(_C_ascend, ops)
{
    ops.def("gmm1_vecin_fullk(Tensor x, Tensor[] weight, Tensor weight_scale, "
            "Tensor x_scale, Tensor group_list) -> (Tensor, Tensor)");
    ops.impl("gmm1_vecin_fullk", torch::kPrivateUse1,
             &vllm_smooth::gmm1_vecin_fullk);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module)
{
}
