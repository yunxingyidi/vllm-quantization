#ifndef ACLNN_GMM1_VECIN_FULLK_H_
#define ACLNN_GMM1_VECIN_FULLK_H_

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

__attribute__((visibility("default"))) aclnnStatus aclnnGmm1VecinFullkGetWorkspaceSize(
    const aclTensor *x, const aclTensorList *weight,
    const aclTensor *weightScale, const aclTensor *xScale,
    const aclTensor *groupList, aclTensor *y, aclTensor *yScale,
    uint64_t *workspaceSize, aclOpExecutor **executor);

__attribute__((visibility("default"))) aclnnStatus aclnnGmm1VecinFullk(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif
#endif
