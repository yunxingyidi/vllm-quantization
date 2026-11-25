#ifndef OP_API_INC_LEVEL0_GMM1_VECIN_FULLK_H_
#define OP_API_INC_LEVEL0_GMM1_VECIN_FULLK_H_

#include <tuple>
#include "opdev/op_executor.h"

namespace l0op {
const std::tuple<aclTensor *, aclTensor *> Gmm1VecinFullk(
    const aclTensor *x, const aclTensorList *weight,
    const aclTensor *weightScale, const aclTensor *xScale,
    const aclTensor *groupList, aclOpExecutor *executor);
}  // namespace l0op

#endif
