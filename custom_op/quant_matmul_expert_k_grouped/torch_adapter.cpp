#include <torch/extension.h>
#include <torch/library.h>

#include <cstdlib>
#include <dlfcn.h>

#include "aclnn_torch_adapter/op_api_common.h"

namespace {
void *GetVllmSmoothCustOpApiFuncAddr(const char *apiName)
{
    static void *handler = []() -> void * {
        const char *path = std::getenv("GROUP_QUANT_CUSTOM_OPAPI_PATH");
        if (path == nullptr || path[0] == '\0') {
            return nullptr;
        }
        return dlopen(path, RTLD_LAZY | RTLD_LOCAL);
    }();
    return handler == nullptr ? nullptr : dlsym(handler, apiName);
}
} // namespace

// vLLM-Ascend can preload its own libcust_opapi.so before the GroupQuant
// extension is imported.  Looking up a bare SONAME would then return that
// already-loaded library, which does not contain this project-only GMM2 API.
// Resolve the two GMM2 entry points from our absolute library path while
// retaining the standard adapter's tensor conversion and stream handling.
#define EXEC_VLLM_SMOOTH_NPU_CMD(aclnn_api, ...)                              \
    do {                                                                       \
        static const auto getWorkspaceSizeFuncAddr =                           \
            GetVllmSmoothCustOpApiFuncAddr(#aclnn_api "GetWorkspaceSize");    \
        static const auto opApiFuncAddr =                                      \
            GetVllmSmoothCustOpApiFuncAddr(#aclnn_api);                        \
        static const auto initMemAddr = GetOpApiFuncAddr("InitHugeMemThreadLocal"); \
        static const auto unInitMemAddr = GetOpApiFuncAddr("UnInitHugeMemThreadLocal"); \
        static const auto releaseMemAddr = GetOpApiFuncAddr("ReleaseHugeMem"); \
        TORCH_CHECK(                                                           \
            getWorkspaceSizeFuncAddr != nullptr && opApiFuncAddr != nullptr,   \
            #aclnn_api, " is not exported by GROUP_QUANT_CUSTOM_OPAPI_PATH=", \
            std::getenv("GROUP_QUANT_CUSTOM_OPAPI_PATH") == nullptr            \
                ? "<unset>"                                                    \
                : std::getenv("GROUP_QUANT_CUSTOM_OPAPI_PATH"));               \
        auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);         \
        uint64_t workspace_size = 0;                                           \
        uint64_t *workspace_size_addr = &workspace_size;                       \
        aclOpExecutor *executor = nullptr;                                     \
        aclOpExecutor **executor_addr = &executor;                             \
        InitHugeMemThreadLocal initMemFunc =                                   \
            reinterpret_cast<InitHugeMemThreadLocal>(initMemAddr);             \
        UnInitHugeMemThreadLocal unInitMemFunc =                               \
            reinterpret_cast<UnInitHugeMemThreadLocal>(unInitMemAddr);         \
        if (initMemFunc) {                                                     \
            initMemFunc(nullptr, false);                                       \
        }                                                                      \
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
            workspace_tensor = at::empty({workspace_size}, options.dtype(kByte)); \
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
            if (releaseMemFunc) {                                              \
                releaseMemFunc(nullptr, false);                               \
            }                                                                  \
            return api_ret;                                                    \
        };                                                                     \
        at_npu::native::OpCommand cmd;                                         \
        cmd.Name(#aclnn_api);                                                  \
        cmd.SetCustomHandler(acl_call);                                        \
        cmd.Run();                                                             \
        if (unInitMemFunc) {                                                   \
            unInitMemFunc(nullptr, false);                                     \
        }                                                                      \
    } while (false)

namespace vllm_smooth {
at::Tensor quant_matmul_expert_k_grouped(
    const at::Tensor &x,
    const at::Tensor &weight,
    const at::Tensor &group_list,
    const at::Tensor &weight_scale,
    const at::Tensor &x_scale,
    bool cumulative_group_list)
{
    TORCH_CHECK(x.dim() == 2 || x.dim() == 3,
                "x must be packed [G*M,S] or ND [M,G,S].");
    TORCH_CHECK(weight.dim() == 1, "weight must be packed 1-D NZ storage.");
    TORCH_CHECK(group_list.dim() == 1 && group_list.numel() > 0,
                "group_list must be non-empty [E].");
    TORCH_CHECK(weight_scale.dim() == 2 && weight_scale.size(0) > 0 &&
                    weight_scale.size(1) > 0,
                "weight_scale must be non-empty [E * G, N].");
    TORCH_CHECK(x.scalar_type() == at::kChar, "x must be int8.");
    TORCH_CHECK(weight.scalar_type() == at::kChar, "weight must be int8.");
    TORCH_CHECK(group_list.scalar_type() == at::kLong, "group_list must be int64.");
    TORCH_CHECK(weight_scale.scalar_type() == at::kFloat || weight_scale.scalar_type() == at::kBFloat16,
                "weight_scale must be float32 or bfloat16.");
    TORCH_CHECK(x_scale.scalar_type() == at::kFloat, "x_scale must be float32.");
    TORCH_CHECK(x.device() == weight.device() && x.device() == group_list.device() &&
                    x.device() == weight_scale.device() && x.device() == x_scale.device(),
                "all inputs must be on the same NPU device.");
    const int64_t expert_num = group_list.numel();
    TORCH_CHECK(weight_scale.size(0) % expert_num == 0,
                "weight_scale rows must be divisible by expert count.");
    const int64_t k_group_num = weight_scale.size(0) / expert_num;
    TORCH_CHECK(k_group_num > 0, "K-group count must be positive.");
    TORCH_CHECK(x.size(0) > 0 && x.size(1) > 0,
                "x must contain at least one token and one K element.");
    const bool nd_input = x.dim() == 3;
    TORCH_CHECK(nd_input ? x.size(1) == k_group_num : x.size(0) % k_group_num == 0,
                "x group dimension must match weight_scale group count.");
    const int64_t group_size = x.size(nd_input ? 2 : 1);
    TORCH_CHECK(group_size > 0, "group_size must be positive.");
    const int64_t m = nd_input ? x.size(0) : x.size(0) / k_group_num;
    const int64_t n = weight_scale.size(1);
    TORCH_CHECK(group_size % 16 == 0, "group_size must be divisible by 16.");
    TORCH_CHECK(n % 32 == 0, "N must be divisible by 32.");
    TORCH_CHECK(weight.numel() == expert_num * k_group_num * n * group_size,
                "packed weight element count mismatch.");
    TORCH_CHECK(x_scale.dim() == 1 && x_scale.numel() == m,
                "x_scale must be native per-token [M].");
    auto output = at::empty({m, n}, x.options().dtype(at::kBFloat16));
    // A view distinguishes the formats in CANN's tiling cache without
    // copying routing data or launching an NPU operator.
    auto routing = cumulative_group_list ? group_list.unsqueeze(-1) : group_list;
    EXEC_VLLM_SMOOTH_NPU_CMD(
        aclnnQuantMatmulExpertKGrouped,
        x,
        weight,
        routing,
        weight_scale,
        x_scale,
        output);
    return output;
}
} // namespace vllm_smooth

TORCH_LIBRARY_FRAGMENT(_C_ascend, ops)
{
    ops.def("quant_matmul_expert_k_grouped(Tensor x, Tensor weight, Tensor group_list, "
            "Tensor weight_scale, Tensor x_scale, bool cumulative_group_list=False) -> Tensor");
    ops.impl("quant_matmul_expert_k_grouped", torch::kPrivateUse1,
             &vllm_smooth::quant_matmul_expert_k_grouped);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module)
{
}
