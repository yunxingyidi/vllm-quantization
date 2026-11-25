#ifndef QUANT_MATMUL_EXPERT_K_GROUPED_TILING_H
#define QUANT_MATMUL_EXPERT_K_GROUPED_TILING_H

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(QuantMatmulExpertKGroupedTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, coreNum);
    TILING_DATA_FIELD_DEF(uint32_t, groupNum);
    TILING_DATA_FIELD_DEF(uint32_t, totalInGroup);
    TILING_DATA_FIELD_DEF(uint32_t, k);
    TILING_DATA_FIELD_DEF(uint32_t, n);
    TILING_DATA_FIELD_DEF(uint32_t, groupAxis);
    TILING_DATA_FIELD_DEF(uint32_t, kGroupSize);
    TILING_DATA_FIELD_DEF(uint32_t, kGroupNum);
    TILING_DATA_FIELD_DEF(uint32_t, ubCalSize);
    TILING_DATA_FIELD_DEF(uint32_t, ubRestBytes);
    TILING_DATA_FIELD_DEF(uint32_t, parallNum);
    TILING_DATA_FIELD_DEF(uint32_t, enableTilePipeline);
    TILING_DATA_FIELD_DEF(uint32_t, tileDynamicSchedule);
    TILING_DATA_FIELD_DEF(uint32_t, reserved0);
    TILING_DATA_FIELD_DEF(uint32_t, reserved1);
    TILING_DATA_FIELD_DEF(uint32_t, reserved2);
    TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, mmTilingData);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(QuantMatmulExpertKGrouped, QuantMatmulExpertKGroupedTilingData)
} // namespace optiling

#endif
