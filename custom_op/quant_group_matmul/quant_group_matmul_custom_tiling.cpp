/**
 * @file quant_group_matmul_custom_tiling.cpp
 *
 * Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include <cassert>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <string>

#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"
#include "quant_group_matmul_custom_tiling.h"

using matmul_tiling::TPosition;
using matmul_tiling::CubeFormat;
using matmul_tiling::DataType;

constexpr uint32_t BEST_BASE_M = 128;
constexpr uint32_t BEST_BASE_K = 128;
constexpr uint32_t BEST_BASE_N = 256;
constexpr uint32_t BEST_TILE_DYNAMIC_BASE_M = 32;

uint32_t ReadEnvUint(const char *name, uint32_t defaultValue)
{
    const char *value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return defaultValue;
    }
    char *end = nullptr;
    unsigned long parsed = std::strtoul(value, &end, 10);
    if (end == value || *end != '\0' || parsed == 0) {
        return defaultValue;
    }
    return static_cast<uint32_t>(parsed);
}

bool ReadEnvEnabled(const char *name)
{
    const char *value = std::getenv(name);
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

bool GenerateTiling(QuantGroupMatmulCustomTilingData &gmmTiling)
{
    optiling::TCubeTiling tilingData;
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    matmul_tiling::MultiCoreMatmulTiling tilingApi(*ascendcPlatform);
    const uint32_t shapeM = gmmTiling.totalInGroup;
    const uint32_t shapeN = gmmTiling.n;
    const uint32_t shapeK = gmmTiling.k;

    if (gmmTiling.groupAxis != 0 && gmmTiling.groupAxis != 1 && gmmTiling.groupAxis != 2) {
        return false;
    }
    const uint32_t defaultBaseM =
        ReadEnvEnabled("GROUP_QUANT_USE_TILE_DYNAMIC_MATMUL")
            ? BEST_TILE_DYNAMIC_BASE_M
            : BEST_BASE_M;
    const uint32_t baseM = ReadEnvUint("GROUP_QUANT_SPLIT_M", defaultBaseM);
    const uint32_t baseN = ReadEnvUint("GROUP_QUANT_SPLIT_N", BEST_BASE_N);
    const uint32_t baseK = ReadEnvUint("GROUP_QUANT_SPLIT_K", BEST_BASE_K);
    const uint32_t splitM = std::min(baseM, shapeM);
    const uint32_t splitN = std::min(baseN, shapeN);
    const uint32_t splitK = std::min(baseK, shapeK);

    tilingApi.SetDim(1);
    tilingApi.SetAType(TPosition::GM, CubeFormat::ND, DataType::DT_INT8, false);
    tilingApi.SetBType(TPosition::GM, CubeFormat::NZ, DataType::DT_INT8, false);
    tilingApi.SetCType(TPosition::GM, CubeFormat::ND, DataType::DT_INT32);
    tilingApi.SetBias(false);

    tilingApi.SetOrgShape(shapeM, shapeN, shapeK);
    tilingApi.SetShape(shapeM, shapeN, shapeK);
    tilingApi.SetFixSplit(splitM, splitN, splitK);

    int64_t res = tilingApi.GetTiling(tilingData);
    if (res == -1) {
        std::cout << "gen tiling failed" << std::endl;
        return false;
    }
    tilingData.set_dbL0C(1);
    tilingData.set_stepKa(4);  // 4: L1中左矩阵单次搬运基于baseK的4倍数据
    tilingData.set_stepKb(4);  // 4: L1中右矩阵单次搬运基于baseK的4倍数据
    tilingData.set_depthA1(8);  // 8: stepKa的两倍，开启double buffer
    tilingData.set_depthB1(8);  // 8: stepKb的两倍，开启double buffer
    tilingData.set_stepM(1);
    tilingData.set_stepN(1);

    // Enable tile pipeline by default for K-axis grouped paths if not explicitly set.
    if (gmmTiling.groupAxis != 0 && gmmTiling.enableTilePipeline == 0) {
        const char *pipeEnv = std::getenv("GROUP_QUANT_TILE_PIPELINE");
        if (pipeEnv == nullptr || pipeEnv[0] != '0') {
            gmmTiling.enableTilePipeline = 1;
        }
    }
    
    uint32_t tilingSize = tilingData.GetDataSize();
    tilingData.SaveToBuffer(&gmmTiling.mmTilingData, tilingSize);
    return true;
}
