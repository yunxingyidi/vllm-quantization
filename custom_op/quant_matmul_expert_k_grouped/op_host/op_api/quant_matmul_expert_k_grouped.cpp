#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "../quant_matmul_expert_k_grouped.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(QuantMatmulExpertKGrouped);

aclTensor *QuantMatmulExpertKGrouped(const aclTensor *x, const aclTensor *weight,
    const aclTensor *groupList, const aclTensor *weightScale, const aclTensor *xScale,
    aclOpExecutor *executor)
{
    L0_DFX(QuantMatmulExpertKGrouped, x, weight, groupList, weightScale, xScale);
    const int64_t expertNum = groupList->GetViewShape().GetDim(0);
    const int64_t scaleGroups = weightScale->GetViewShape().GetDim(0);
    const int64_t kGroupNum = scaleGroups / expertNum;
    const auto &xShape = x->GetViewShape();
    const int64_t m = xShape.GetDimNum() == 3 ? xShape.GetDim(0) : xShape.GetDim(0) / kGroupNum;
    const int64_t n = weightScale->GetViewShape().GetDim(1);
    auto *out = executor->AllocTensor({m, n}, DataType::DT_BF16, ge::FORMAT_ND);
    const auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        QuantMatmulExpertKGrouped,
        OP_INPUT(x, weight, groupList, weightScale, xScale),
        OP_OUTPUT(out));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "QuantMatmulExpertKGrouped launcher registration failed.");
        return nullptr;
    }
    return out;
}
} // namespace l0op
