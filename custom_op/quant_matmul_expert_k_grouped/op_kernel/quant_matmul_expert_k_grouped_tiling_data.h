/**
 * @file quant_group_matmul_custom_tiling.h
 *
 * Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#ifndef QUANT_MATMUL_EXPERT_K_GROUPED_TILING_DATA_H
#define QUANT_MATMUL_EXPERT_K_GROUPED_TILING_DATA_H

#include "kernel_tiling/kernel_tiling.h"


struct QuantGroupMatmulCustomTilingData
{
    uint32_t coreNum;
    uint32_t groupNum;
    uint32_t totalInGroup;
    uint32_t k;
    uint32_t n;
    // 0: M-axis expert grouping, 1: K-axis grouping, 2: expert x K-group hybrid.
    uint32_t groupAxis;
    // Used by groupAxis 1/2.
    uint32_t kGroupSize;
    uint32_t kGroupNum;
    uint32_t ubCalSize;
    uint32_t ubRestBytes;
    uint32_t parallNum;
    uint32_t enableTilePipeline;
    // 0: existing schedule, 1: experimental shared M-tile producer / N-tile consumers.
    uint32_t tileDynamicSchedule;
    uint32_t reserved0;
    // Expert tile schedule: 0 per-expert, 1 M-outer, 2 streamed M/N, 3 batched N.
    uint32_t reserved1;
    // bit0: dual AIV; bit1: rank-3 [M,G,S] ND activation; bit2: BF16 weight scale;
    // bit3: group list entries give cumulative end rows.
    uint32_t reserved2;
    TCubeTiling mmTilingData;
};

#endif // QUANT_MATMUL_EXPERT_K_GROUPED_TILING_DATA_H
