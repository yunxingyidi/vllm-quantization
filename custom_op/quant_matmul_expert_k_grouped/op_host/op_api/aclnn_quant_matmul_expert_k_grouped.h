#ifndef ACLNN_QUANT_MATMUL_EXPERT_K_GROUPED_H
#define ACLNN_QUANT_MATMUL_EXPERT_K_GROUPED_H

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

__attribute__((visibility("default"))) aclnnStatus aclnnQuantMatmulExpertKGroupedGetWorkspaceSize(
    const aclTensor *x, const aclTensor *weight, const aclTensor *groupList,
    const aclTensor *weightScale, const aclTensor *xScale, aclTensor *out,
    uint64_t *workspaceSize, aclOpExecutor **executor);

__attribute__((visibility("default"))) aclnnStatus aclnnQuantMatmulExpertKGrouped(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
