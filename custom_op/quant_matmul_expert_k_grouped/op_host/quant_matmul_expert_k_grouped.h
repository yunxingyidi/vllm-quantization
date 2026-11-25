#ifndef OP_API_QUANT_MATMUL_EXPERT_K_GROUPED_H
#define OP_API_QUANT_MATMUL_EXPERT_K_GROUPED_H

#include "opdev/op_executor.h"

namespace l0op {
aclTensor *QuantMatmulExpertKGrouped(const aclTensor *x, const aclTensor *weight,
    const aclTensor *groupList, const aclTensor *weightScale, const aclTensor *xScale,
    aclOpExecutor *executor);
}

#endif
