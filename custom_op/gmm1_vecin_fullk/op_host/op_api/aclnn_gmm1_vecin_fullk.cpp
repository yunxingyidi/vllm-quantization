#include "aclnn_gmm1_vecin_fullk.h"

#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "../gmm1_vecin_fullk.h"

using namespace op;

namespace {
constexpr int64_t K = 2048;
constexpr int64_t MAX_N = 1536;

int64_t ScaleGroupCount(const aclTensor *scale, int64_t experts, int64_t n)
{
    const auto shape = scale->GetViewShape();
    if (shape.GetDimNum() == 0) return 0;
    int64_t elements = 1;
    for (size_t dim = 0; dim < shape.GetDimNum(); ++dim) {
        if (shape.GetDim(dim) <= 0) return 0;
        elements *= shape.GetDim(dim);
    }
    const int64_t denominator = experts * n;
    return denominator > 0 && elements % denominator == 0
        ? elements / denominator : 0;
}

bool CheckParams(const aclTensor *x, const aclTensorList *weight,
                 const aclTensor *weightScale, const aclTensor *xScale,
                 const aclTensor *groupList, const aclTensor *y,
                 const aclTensor *yScale)
{
    OP_CHECK_NULL(x, return false);
    OP_CHECK_NULL(weight, return false);
    OP_CHECK_NULL(weightScale, return false);
    OP_CHECK_NULL(groupList, return false);
    OP_CHECK_NULL(y, return false);
    OP_CHECK_NULL(yScale, return false);
    const size_t experts = weight->Size();
    OP_CHECK(experts > 0,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "weight list must be non-zero."),
             return false);
    const aclTensor *firstWeight = (*weight)[0];
    OP_CHECK(firstWeight != nullptr, OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "null weight[0]."), return false);
    const auto firstStorage = firstWeight->GetStorageShape();
    OP_CHECK(firstStorage.GetDimNum() == 4 && firstStorage.GetDim(0) > 0 &&
                 firstStorage.GetDim(1) == 128 && firstStorage.GetDim(2) == 16 &&
                 firstStorage.GetDim(3) == 32,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "weight[0] must be INT8 NZ [N/32,128,16,32]."), return false);
    const int64_t n = firstStorage.GetDim(0) * 32;
    const int64_t outN = n / 2;
    OP_CHECK(n <= MAX_N && n % 64 == 0,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "local weight N must be a multiple of 64 in (0,1536]."),
             return false);
    const int64_t kGroupNum = ScaleGroupCount(weightScale, experts, n);
    OP_CHECK(kGroupNum > 0 && K % kGroupNum == 0 && (K / kGroupNum) % 16 == 0,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "weightScale must contain packed [expert,G,localN] values; "
                     "G must divide 2048 and K-group size must be 16-aligned."), return false);
    OP_CHECK(groupList->GetViewShape().GetDimNum() == 1 &&
                 static_cast<size_t>(groupList->GetViewShape().GetDim(0)) == experts,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "groupList length must equal expert count."), return false);
    OP_CHECK(x->GetViewShape().GetDimNum() == 2 && x->GetViewShape().GetDim(1) == K,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x must be [M, 2048]."), return false);
    const int64_t m = x->GetViewShape().GetDim(0);
    OP_CHECK(x->GetDataType() == DataType::DT_INT8,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "x must be INT8 [M, 2048]."), return false);
    OP_CHECK(xScale != nullptr && xScale->GetViewShape() == op::Shape({m}) &&
                 xScale->GetDataType() == DataType::DT_FLOAT,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "xScale must be FP32 [M]."), return false);
    OP_CHECK(y->GetViewShape() == op::Shape({m, outN}) && yScale->GetViewShape() == op::Shape({m}),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "outputs must be [M, localN/2] and [M]."), return false);
    OP_CHECK(groupList->GetDataType() == DataType::DT_INT64 && y->GetDataType() == DataType::DT_INT8 &&
                 yScale->GetDataType() == DataType::DT_FLOAT,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "input/output dtype mismatch."), return false);
    for (size_t i = 0; i < experts; ++i) {
        const aclTensor *w = (*weight)[i];
        OP_CHECK(w != nullptr, OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "null weight item."), return false);
        OP_CHECK(w->GetStorageShape() == op::Shape({n / 32, 128, 16, 32}) &&
                     w->GetDataType() == DataType::DT_INT8,
                 OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                         "weight[%zu] must be INT8 NZ [localN/32,128,16,32].", i), return false);
    }
    OP_CHECK(weightScale->GetDataType() == DataType::DT_INT64,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "weightScale must be contiguous packed INT64 "
                     "from npu_trans_quant_param [expert,G,localN]."),
             return false);
    return true;
}
}  // namespace

extern "C" aclnnStatus aclnnGmm1VecinFullkGetWorkspaceSize(
    const aclTensor *x, const aclTensorList *weight,
    const aclTensor *weightScale, const aclTensor *xScale,
    const aclTensor *groupList, aclTensor *y, aclTensor *yScale,
    uint64_t *workspaceSize, aclOpExecutor **executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);
    L2_DFX_PHASE_1(aclnnGmm1VecinFullk, DFX_IN(x, weight, weightScale, xScale, groupList), DFX_OUT(y, yScale));
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    CHECK_RET(CheckParams(x, weight, weightScale, xScale, groupList, y, yScale), ACLNN_ERR_PARAM_INVALID);
    if (y->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }
    x = l0op::Contiguous(x, uniqueExecutor.get());
    weightScale = l0op::Contiguous(weightScale, uniqueExecutor.get());
    if (xScale != nullptr) xScale = l0op::Contiguous(xScale, uniqueExecutor.get());
    groupList = l0op::Contiguous(groupList, uniqueExecutor.get());
    CHECK_RET(x != nullptr && weightScale != nullptr && xScale != nullptr &&
                  groupList != nullptr, ACLNN_ERR_INNER_NULLPTR);
    for (size_t i = 0; i < weight->Size(); ++i) {
        auto *w = const_cast<aclTensor *>((*weight)[i]);
        w->SetStorageFormat(Format::FORMAT_FRACTAL_NZ);
        w->SetViewFormat(w->GetViewShape().GetDimNum() == 4
                             ? Format::FORMAT_FRACTAL_NZ
                             : Format::FORMAT_ND);
        w->SetOriginalShape(w->GetViewShape());
    }
    auto result = l0op::Gmm1VecinFullk(
        x, weight, weightScale, xScale, groupList, uniqueExecutor.get());
    CHECK_RET(result != std::tuple(nullptr, nullptr), ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(l0op::ViewCopy(std::get<0>(result), y, uniqueExecutor.get()) != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(l0op::ViewCopy(std::get<1>(result), yScale, uniqueExecutor.get()) != nullptr, ACLNN_ERR_INNER_NULLPTR);
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

extern "C" aclnnStatus aclnnGmm1VecinFullk(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
    aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnGmm1VecinFullk);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
