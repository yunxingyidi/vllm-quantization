#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "opdev/make_op_executor.h"
#include "../gmm1_vecin_fullk.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(Gmm1VecinFullk);

const std::tuple<aclTensor *, aclTensor *> Gmm1VecinFullk(
    const aclTensor *x, const aclTensorList *weight,
    const aclTensor *weightScale, const aclTensor *xScale,
    const aclTensor *groupList, aclOpExecutor *executor)
{
    L0_DFX(Gmm1VecinFullk, x, weight, weightScale, xScale, groupList);
    const int64_t m = x->GetViewShape().GetDim(0);
    const int64_t outN = (*weight)[0]->GetStorageShape().GetDim(0) * 16;
    auto y = executor->AllocTensor(gert::Shape({m, outN}), DataType::DT_INT8, ge::FORMAT_ND);
    auto yScale = executor->AllocTensor(gert::Shape({m}), DataType::DT_FLOAT, ge::FORMAT_ND);
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        Gmm1VecinFullk, OP_INPUT(x, weight, weightScale, xScale, groupList),
        OP_OUTPUT(y, yScale));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER, "Gmm1VecinFullk launcher registration failed.");
        return std::tuple(nullptr, nullptr);
    }
    return std::tie(y, yScale);
}
}  // namespace l0op
