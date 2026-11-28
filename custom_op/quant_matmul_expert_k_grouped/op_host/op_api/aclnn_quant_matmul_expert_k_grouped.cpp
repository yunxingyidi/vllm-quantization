#include "aclnn_quant_matmul_expert_k_grouped.h"

#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/data_type_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "../quant_matmul_expert_k_grouped.h"

using namespace op;

namespace {
aclnnStatus CheckParams(const aclTensor *x, const aclTensor *weight, const aclTensor *groupList,
    const aclTensor *weightScale, const aclTensor *xScale, const aclTensor *out)
{
    CHECK_COND(x != nullptr && weight != nullptr && groupList != nullptr &&
                   weightScale != nullptr && xScale != nullptr && out != nullptr,
               ACLNN_ERR_PARAM_NULLPTR, "QuantMatmulExpertKGrouped inputs and output must not be null.");
    CHECK_COND(x->GetDataType() == DataType::DT_INT8, ACLNN_ERR_PARAM_INVALID, "x must be int8.");
    CHECK_COND(weight->GetDataType() == DataType::DT_INT8, ACLNN_ERR_PARAM_INVALID, "weight must be int8.");
    CHECK_COND(groupList->GetDataType() == DataType::DT_INT64, ACLNN_ERR_PARAM_INVALID,
               "groupList must be int64.");
    const auto weightScaleType = weightScale->GetDataType();
    CHECK_COND(weightScaleType == DataType::DT_FLOAT || weightScaleType == DataType::DT_BF16,
               ACLNN_ERR_PARAM_INVALID, "weightScale must be float32 or bfloat16.");
    CHECK_COND(xScale->GetDataType() == DataType::DT_FLOAT, ACLNN_ERR_PARAM_INVALID,
               "xScale must be float32.");
    CHECK_COND(out->GetDataType() == DataType::DT_BF16, ACLNN_ERR_PARAM_INVALID, "out must be bfloat16.");

    const auto xShape = x->GetViewShape();
    const auto weightShape = weight->GetViewShape();
    const auto groupShape = groupList->GetViewShape();
    const auto weightScaleShape = weightScale->GetViewShape();
    const auto xScaleShape = xScale->GetViewShape();
    const auto outShape = out->GetViewShape();
    CHECK_COND((xShape.GetDimNum() == 2 || xShape.GetDimNum() == 3) && weightShape.GetDimNum() == 1 &&
                   weightScaleShape.GetDimNum() == 2 &&
                   xScaleShape.GetDimNum() == 1 && outShape.GetDimNum() == 2,
               ACLNN_ERR_PARAM_INVALID, "invalid tensor rank.");
    CHECK_COND(groupShape.GetDimNum() == 1 ||
                   (groupShape.GetDimNum() == 2 && groupShape.GetDim(1) == 1),
               ACLNN_ERR_PARAM_INVALID, "groupList must be counts [E] or cumulative ends [E,1].");
    const int64_t expertNum = groupShape.GetDim(0);
    const int64_t scaleGroups = weightScaleShape.GetDim(0);
    CHECK_COND(expertNum > 0 && scaleGroups > 0 && scaleGroups % expertNum == 0,
               ACLNN_ERR_PARAM_INVALID, "weightScale rows must be expertNum * kGroupNum.");
    const int64_t kGroupNum = scaleGroups / expertNum;
    const int64_t packedRows = xShape.GetDim(0);
    const bool ndInput = xShape.GetDimNum() == 3;
    const int64_t groupSize = xShape.GetDim(ndInput ? 2 : 1);
    const int64_t n = weightScaleShape.GetDim(1);
    CHECK_COND(groupSize > 0 && groupSize % 16 == 0 && n > 0 && n % 32 == 0,
               ACLNN_ERR_PARAM_INVALID, "groupSize must align to 16 and N must align to 32.");
    CHECK_COND(packedRows > 0 &&
                   (ndInput ? xShape.GetDim(1) == kGroupNum : packedRows % kGroupNum == 0),
               ACLNN_ERR_PARAM_INVALID, "x must be [G*M,S] packed or [M,G,S] ND with the scale group count.");
    const int64_t m = ndInput ? packedRows : packedRows / kGroupNum;
    CHECK_COND(xScaleShape.GetDim(0) == m, ACLNN_ERR_PARAM_INVALID,
               "xScale must have M per-token elements.");
    CHECK_COND(weightShape.GetDim(0) == expertNum * kGroupNum * n * groupSize,
               ACLNN_ERR_PARAM_INVALID, "packed weight element count mismatch.");
    CHECK_COND(outShape.GetDim(0) == m && outShape.GetDim(1) == n,
               ACLNN_ERR_PARAM_INVALID, "out shape must be [M, N].");
    return ACLNN_SUCCESS;
}
} // namespace

extern "C" aclnnStatus aclnnQuantMatmulExpertKGroupedGetWorkspaceSize(
    const aclTensor *x, const aclTensor *weight, const aclTensor *groupList,
    const aclTensor *weightScale, const aclTensor *xScale, aclTensor *out,
    uint64_t *workspaceSize, aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnQuantMatmulExpertKGrouped,
                   DFX_IN(x, weight, groupList, weightScale, xScale), DFX_OUT(out));
    CHECK_COND(workspaceSize != nullptr && executor != nullptr, ACLNN_ERR_PARAM_NULLPTR,
               "workspaceSize and executor must not be null.");
    CHECK_COND(CheckParams(x, weight, groupList, weightScale, xScale, out) == ACLNN_SUCCESS,
               ACLNN_ERR_PARAM_INVALID, "QuantMatmulExpertKGrouped parameter check failed.");

    auto uniqueExecutor = CREATE_EXECUTOR();
    auto *executorPtr = uniqueExecutor.get();
    CHECK_RET(executorPtr != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    const auto *xContiguous = l0op::Contiguous(x, executorPtr);
    const auto *weightContiguous = l0op::Contiguous(weight, executorPtr);
    const auto *groupListContiguous = l0op::Contiguous(groupList, executorPtr);
    const auto *weightScaleContiguous = l0op::Contiguous(weightScale, executorPtr);
    const auto *xScaleContiguous = l0op::Contiguous(xScale, executorPtr);
    CHECK_RET(xContiguous != nullptr && weightContiguous != nullptr && groupListContiguous != nullptr &&
                  weightScaleContiguous != nullptr && xScaleContiguous != nullptr,
              ACLNN_ERR_INNER_NULLPTR);

    auto *result = l0op::QuantMatmulExpertKGrouped(xContiguous, weightContiguous,
        groupListContiguous, weightScaleContiguous, xScaleContiguous, executorPtr);
    CHECK_RET(result != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto *copyResult = l0op::ViewCopy(result, out, executorPtr);
    CHECK_RET(copyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

extern "C" aclnnStatus aclnnQuantMatmulExpertKGrouped(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnQuantMatmulExpertKGrouped);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
