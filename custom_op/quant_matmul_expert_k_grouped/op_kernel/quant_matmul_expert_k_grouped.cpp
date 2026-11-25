/**
 * @file quant_group_matmul_custom.cpp
 *
 * Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "quant_matmul_expert_k_grouped_tiling_data.h"

using AscendC::LocalTensor;
using AscendC::GlobalTensor;
using AscendC::MatmulType;
using AscendC::TPosition;

using aT = MatmulType<TPosition::GM, CubeFormat::ND, int8_t>;
using bT = MatmulType<TPosition::GM, CubeFormat::NZ, int8_t>;
using BiasT = MatmulType<TPosition::GM, CubeFormat::ND, int32_t>;
using cT = MatmulType<TPosition::GM, CubeFormat::ND, int32_t>;

using MT = matmul::MatmulImpl<aT, bT, cT, BiasT, CFG_MDL>;

using localCT = MatmulType<TPosition::VECIN, CubeFormat::ND, int32_t>;
using LocalMM = matmul::Matmul<aT, bT, localCT, BiasT, CFG_MDL>;

#ifndef GROUP_QUANT_DYNAMIC_PACK_FN
#define GROUP_QUANT_DYNAMIC_PACK_FN DynamicQuantizeKGroupedToWorkspaceDataCopy
#endif

constexpr uint32_t BROADCAST_DIM = 2;
constexpr uint64_t SYNC_AIV_TO_AIC = 3;
constexpr uint64_t SYNC_AIC_TO_AIV = 5;
constexpr uint64_t SYNC_DYNAMIC_QUANT_DONE = 7;
constexpr uint64_t DYNAMIC_WORKSPACE_GUARD_BYTES = 512;
constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t SCALAR_TO_MTE3_EVENT_ID = 0;
constexpr uint32_t LOCAL_GMM2_EPILOGUE_ROWS = 16;

__aicore__ inline uint64_t AlignUp64(uint64_t value, uint64_t align)
{
    return (value + align - 1) / align * align;
}

__aicore__ inline void CleanGmCacheRange(GM_ADDR addr, uint64_t length);
__aicore__ inline int8_t QuantizeBf16Value(float value, float scale);

struct MNConfig {
    uint32_t m = 0;
    uint32_t baseM = 0;
    uint32_t baseN = 0;
    uint32_t expertIdx = 0;
    uint32_t mIdx = 0;
    uint32_t nIdx = 0;
    uint32_t blockDimM = 0;
    uint32_t blockDimN = 0;
    uint32_t singleM = 0;
    uint32_t singleN = 0;
    uint32_t offsetM = 0;
    uint32_t pipeIdx = 0;
    uint64_t workSpaceOffset = 0;
};


template <typename T>
__aicore__ inline void DataCopyPad2D(const LocalTensor<T> dst, const GlobalTensor<T> src, uint32_t dim1, uint32_t dim0,
                                     uint32_t srcDim0) {
    AscendC::DataCopyExtParams params;
    params.blockCount = dim1;
    params.blockLen = dim0 * sizeof(T);
    params.srcStride = (srcDim0 - dim0) * sizeof(T);
    // 32: int32 -> float16, 为防止跨行数据进入同一32B block，提前每行按偶数block对齐
    params.dstStride = AscendC::Ceil(dim0 * sizeof(T), 32) % 2;

    AscendC::DataCopyPadExtParams<T> padParams{true, 0, 0, 0};
    DataCopyPad(dst, src, params, padParams);
}

template <typename T>
__aicore__ inline void DataCopyPad2D(const GlobalTensor<T> dst, const LocalTensor<T> src, uint32_t dim1, uint32_t dim0,
                                     uint32_t srcDim0, uint32_t dstDim0) {
    AscendC::DataCopyExtParams params;
    params.blockCount = dim1;
    params.blockLen = dim0 * sizeof(T);
    // 32: ub访问粒度为32B
    params.srcStride = (srcDim0 - dim0) * sizeof(T) / 32;
    params.dstStride = (dstDim0 - dim0) * sizeof(T);
    DataCopyPad(dst, src, params);
}

class QuantGroupMatmul {
public:
    __aicore__ inline QuantGroupMatmul(MT &matmul) : mm(matmul) {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR bias,
        GM_ADDR groupList, GM_ADDR scale, GM_ADDR perTokenScale, GM_ADDR y, GM_ADDR workspace,
        QuantGroupMatmulCustomTilingData *tilingData, AscendC::TPipe *tPipeIn);
    __aicore__ inline void Process();
private:
    __aicore__ inline void ProcessMGrouped();
    __aicore__ inline void ProcessKGroupedCompat();
    __aicore__ inline void ProcessExpertKGroupedCompat();
    __aicore__ inline void ProcessExpertKGroupedMOuter();
    __aicore__ inline void ProcessKGroupedMOuter();
    __aicore__ inline void ProcessKGroupedSharedMOuter();
    __aicore__ inline void InitMNConfigBase(MNConfig& mnConfig) const;
    __aicore__ inline void RunLogicalGroup(uint32_t logicalGroupIdx, uint32_t logicalM, MNConfig& mnConfig);
    __aicore__ inline uint32_t GetLogicalK() const;
    __aicore__ inline bool UseTileLocalDynamicInputs() const;
    __aicore__ inline bool UseSharedTileDynamicInputs() const;
    __aicore__ inline uint64_t GetXOffset(uint32_t groupIdx, const MNConfig& mnConfig) const;
    __aicore__ inline uint64_t GetWeightOffset(uint32_t groupIdx, const MNConfig& mnConfig) const;
    __aicore__ inline void InitUbBuffer();
    __aicore__ inline void MMCompute(uint32_t groupIdx, MNConfig& mnConfig, bool syncVector = true);
    __aicore__ inline void PrepareTileDynamicInputs(MNConfig& mnConfig);
    __aicore__ inline void PrepareSharedTileDynamicInputs(MNConfig& mnConfig);
    __aicore__ inline void VectorCompute(uint32_t groupIdx, MNConfig& mnConfig);
    __aicore__ inline void VectorComputeKGrouped(MNConfig& mnConfig);
    __aicore__ inline void VectorComputeExpertKGrouped(MNConfig& mnConfig);
    __aicore__ inline void VectorComputeKGroupedShared(MNConfig& mnConfig);
    __aicore__ inline void PostCompute();
    __aicore__ inline void ComputeDequantAndActivate(MNConfig& mnConfig, uint32_t curVecBaseM, uint32_t alignBaseN,
                                                     uint32_t curVecBaseN, uint32_t offsetM,
                                                     uint64_t outOffset, uint32_t offsetN,
                                                     uint32_t groupIdx, uint64_t partialOffset);
    __aicore__ inline uint64_t GetScaleOffset(uint32_t groupIdx, const MNConfig& mnConfig) const;
    __aicore__ inline void EnqueueScale(uint32_t curBaseN, uint64_t scaleOffset);
    __aicore__ inline void DequeueScale(uint32_t alignBaseN);
    __aicore__ inline uint64_t GetPerTokenScaleOffset(uint32_t groupIdx, MNConfig& mnConfig, uint32_t offsetM) const;
    __aicore__ inline void EnqueuePerTokenScale(
        uint32_t groupIdx, MNConfig& mnConfig, uint32_t curBaseM, uint32_t offsetM);
    __aicore__ inline void BroadcastPerTokenScale(uint32_t curBaseM, uint32_t alignBaseN);

private:
    MT& mm;
    GlobalTensor<int8_t> xGm;
    GlobalTensor<int8_t> weightGm;
    GlobalTensor<int32_t> biasGm;
    GlobalTensor<int32_t> mmOutGm;
    GlobalTensor<float> accGm;
    GlobalTensor<float> scaleGm;
    GlobalTensor<float> perTokenScaleGm;
    GlobalTensor<bfloat16_t> rawXGm;
    GlobalTensor<int8_t> tileXGm;
    GlobalTensor<float> tilePerTokenScaleGm;
    GlobalTensor<int64_t> groupListGm;
    GlobalTensor<bfloat16_t> yGm;
    // define the que
    AscendC::TQue<TPosition::VECIN, 1> vecInQueue;
    AscendC::TQue<TPosition::VECOUT, 1> vecOutQueue;
    AscendC::TQue<TPosition::VECIN, 1> scaleInQueue;
    AscendC::TQue<TPosition::VECIN, 1> perTokenScaleInQueue;
    AscendC::TBuf<TPosition::VECCALC> tmpBuff;
    AscendC::TBuf<TPosition::VECCALC> tileQuantRowBuff;
    AscendC::TBuf<TPosition::VECCALC> tileRawFloatBuff;
    AscendC::TBuf<TPosition::VECCALC> tileScaleBuff;
    LocalTensor<float> scaleInUb;
    LocalTensor<float> perTokenScaleInUb;
    LocalTensor<float> dequantMiddleResult;
    LocalTensor<uint8_t> sharedTmpLocal;
    LocalTensor<float> mulsResultLocal;
    LocalTensor<float> pertokenBrcbLocal;
    LocalTensor<float> actResultLocal;
    uint32_t subBlockIdx;
    uint32_t coreIdx;
    uint32_t cubeCount = 0;
    GM_ADDR tileXWorkspaceAddr = nullptr;
    GM_ADDR tileScaleWorkspaceAddr = nullptr;
    AscendC::TPipe *pipe;
    QuantGroupMatmulCustomTilingData *tiling;
};

__aicore__ inline void QuantGroupMatmul::Init(GM_ADDR x, GM_ADDR weight, GM_ADDR bias,
        GM_ADDR groupList, GM_ADDR scale, GM_ADDR perTokenScale, GM_ADDR y, GM_ADDR workspace,
        QuantGroupMatmulCustomTilingData *tilingData, AscendC::TPipe *tPipeIn)
{
    xGm.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(x));
    weightGm.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(weight));
    biasGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(bias));  // unused
    mmOutGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(workspace));
    uint64_t mmWorkspaceGroups =
        ((tilingData->groupAxis == 1 || tilingData->groupAxis == 2) && tilingData->kGroupNum > 0)
            ? static_cast<uint64_t>(tilingData->kGroupNum)
            : 1ULL;
    uint64_t userWorkspaceElems =
        static_cast<uint64_t>(tilingData->parallNum) *
        static_cast<uint64_t>(tilingData->mmTilingData.baseN) *
        static_cast<uint64_t>(tilingData->mmTilingData.baseM) *
        static_cast<uint64_t>(tilingData->coreNum) * mmWorkspaceGroups;
    accGm.SetGlobalBuffer(
        reinterpret_cast<__gm__ float *>(workspace) + userWorkspaceElems);
    scaleGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(scale));
    perTokenScaleGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(perTokenScale));
#if defined(GROUP_QUANT_TILE_DYNAMIC_EXPERIMENT) && defined(GROUP_QUANT_TILE_DYNAMIC_LOCAL_INPUTS)
    rawXGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(x));
    const uint64_t accWorkspaceBytes =
        static_cast<uint64_t>(tilingData->totalInGroup) *
        static_cast<uint64_t>(tilingData->n) * sizeof(float);
    const uint64_t tileXWorkspaceOffset =
        AlignUp64(userWorkspaceElems * sizeof(int32_t) + accWorkspaceBytes, 512);
    const uint64_t tileXWorkspaceBytes =
        static_cast<uint64_t>(tilingData->coreNum) *
        static_cast<uint64_t>(tilingData->kGroupNum) *
        static_cast<uint64_t>(tilingData->mmTilingData.baseM) *
        static_cast<uint64_t>(tilingData->kGroupSize);
    const uint64_t tileScaleWorkspaceOffset =
        AlignUp64(tileXWorkspaceOffset + tileXWorkspaceBytes, 512);
    tileXWorkspaceAddr = workspace + tileXWorkspaceOffset;
    tileScaleWorkspaceAddr = workspace + tileScaleWorkspaceOffset;
    tileXGm.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(tileXWorkspaceAddr));
    tilePerTokenScaleGm.SetGlobalBuffer(
        reinterpret_cast<__gm__ float *>(tileScaleWorkspaceAddr));
#endif
    groupListGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(groupList));
    yGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(y));
    tiling = tilingData;
    subBlockIdx = AscendC::GetSubBlockIdx();
    coreIdx = AscendC::GetBlockIdx();
    if ASCEND_IS_AIV {
        coreIdx /= AscendC::GetTaskRation();
    }
    pipe = tPipeIn;
    InitUbBuffer();
}

__aicore__ inline void QuantGroupMatmul::InitUbBuffer()
{
    if ASCEND_IS_AIC {
        return;
    }
    pipe->InitBuffer(scaleInQueue, BUFFER_NUM, tiling->mmTilingData.baseN * sizeof(float));
    pipe->InitBuffer(perTokenScaleInQueue, BUFFER_NUM, tiling->mmTilingData.baseM * sizeof(float));
    pipe->InitBuffer(vecInQueue, BUFFER_NUM, tiling->ubCalSize * sizeof(cT::T));
    pipe->InitBuffer(vecOutQueue, BUFFER_NUM, tiling->ubCalSize * sizeof(bfloat16_t));
    pipe->InitBuffer(tmpBuff, tiling->ubRestBytes);
    pipe->InitBuffer(tileQuantRowBuff, AlignUp64(tiling->kGroupSize * sizeof(bfloat16_t), 32));
    pipe->InitBuffer(tileRawFloatBuff, AlignUp64(tiling->kGroupSize * sizeof(float), 32));
    pipe->InitBuffer(tileScaleBuff, AlignUp64(tiling->mmTilingData.baseM * sizeof(float), 32));
    uint32_t ubCalSizeFloat = tiling->ubCalSize * sizeof(float);
    // ub分配，依次划分中间结果
    dequantMiddleResult = tmpBuff.GetWithOffset<float>(tiling->ubCalSize, 0);
    pertokenBrcbLocal = tmpBuff.GetWithOffset<float>(tiling->ubCalSize, ubCalSizeFloat);
    // 2: 偏移两份，前面为反量化输出和pertoken scale
    mulsResultLocal = tmpBuff.GetWithOffset<float>(tiling->ubCalSize, 2 * ubCalSizeFloat);
    // 3: 再上面的基础上再偏移一份
    actResultLocal = tmpBuff.GetWithOffset<float>(tiling->ubCalSize, 3 * ubCalSizeFloat);
    // AscendDequant needs two extra chunks of temporary space. Keep them after
    // actResultLocal so K-group accumulation can persist in UB across groups.
    sharedTmpLocal = tmpBuff.GetWithOffset<uint8_t>(2 * ubCalSizeFloat, 4 * ubCalSizeFloat);
}

__aicore__ inline void QuantGroupMatmul::Process()
{
    if (tiling->groupAxis == 0) {
        ProcessMGrouped();
        return;
    }
    if (tiling->groupAxis == 2) {
        if (tiling->reserved1 != 0) {
            ProcessExpertKGroupedMOuter();
            return;
        }
        ProcessExpertKGroupedCompat();
        return;
    }
    if (UseTileLocalDynamicInputs()) {
        if (UseSharedTileDynamicInputs()) {
            ProcessKGroupedSharedMOuter();
            return;
        }
        ProcessKGroupedMOuter();
        return;
    }
    ProcessKGroupedCompat();
}

__aicore__ inline void QuantGroupMatmul::ProcessMGrouped()
{
    MNConfig mnConfig;
    InitMNConfigBase(mnConfig);
    mnConfig.offsetM = 0;
    for (uint32_t groupIdx = 0; groupIdx < tiling->groupNum; ++groupIdx) {
        uint32_t m = static_cast<int32_t>(groupListGm.GetValue(groupIdx));
        if (m <= 0) {
            continue;
        }
        RunLogicalGroup(groupIdx, static_cast<uint32_t>(m), mnConfig);
        mnConfig.offsetM += static_cast<uint32_t>(m);
    }
    PostCompute();
}

__aicore__ inline void QuantGroupMatmul::ProcessKGroupedCompat()
{
    MNConfig mnConfig;
    InitMNConfigBase(mnConfig);
    mnConfig.offsetM = 0;
    mnConfig.m = tiling->totalInGroup;
    mnConfig.blockDimM = AscendC::Ceil(mnConfig.m, mnConfig.singleM);
    const uint32_t curCount = mnConfig.blockDimN * mnConfig.blockDimM;
    uint32_t curBlock = coreIdx;

    while (curBlock < curCount) {
        mnConfig.mIdx = curBlock / mnConfig.blockDimN;
        mnConfig.nIdx = curBlock % mnConfig.blockDimN;
        mnConfig.pipeIdx = cubeCount % tiling->parallNum;
        if ASCEND_IS_AIC {
            if (tiling->enableTilePipeline != 0) {
                if (cubeCount >= tiling->parallNum) {
                    AscendC::CrossCoreWaitFlag(SYNC_AIV_TO_AIC);
                }
            } else if (cubeCount > 0) {
                AscendC::CrossCoreWaitFlag(SYNC_AIV_TO_AIC);
            }
        }
        PrepareTileDynamicInputs(mnConfig);
        for (uint32_t groupIdx = 0; groupIdx < tiling->kGroupNum; ++groupIdx) {
            MMCompute(groupIdx, mnConfig, false);
        }
        if ASCEND_IS_AIC {
            AscendC::CrossCoreSetFlag<2, PIPE_FIX>(SYNC_AIC_TO_AIV);
            cubeCount++;
        }
        if ASCEND_IS_AIV {
            VectorComputeKGrouped(mnConfig);
            cubeCount++;
        }
        curBlock += tiling->coreNum;
    }
    PostCompute();
}

__aicore__ inline void QuantGroupMatmul::ProcessExpertKGroupedCompat()
{
    MNConfig mnConfig;
    InitMNConfigBase(mnConfig);
    mnConfig.offsetM = 0;
    for (uint32_t expertIdx = 0; expertIdx < tiling->groupNum; ++expertIdx) {
        const uint32_t expertM = static_cast<uint32_t>(groupListGm.GetValue(expertIdx));
        if (expertM == 0) {
            continue;
        }
        mnConfig.expertIdx = expertIdx;
        mnConfig.m = expertM;
        mnConfig.blockDimM = AscendC::Ceil(mnConfig.m, mnConfig.singleM);
        const uint32_t curCount = mnConfig.blockDimN * mnConfig.blockDimM;
        uint32_t curBlock = coreIdx;

        while (curBlock < curCount) {
            mnConfig.mIdx = curBlock / mnConfig.blockDimN;
            mnConfig.nIdx = curBlock % mnConfig.blockDimN;
            mnConfig.pipeIdx = cubeCount % tiling->parallNum;
            if ASCEND_IS_AIC {
                if (tiling->enableTilePipeline != 0) {
                    if (cubeCount >= tiling->parallNum) {
                        AscendC::CrossCoreWaitFlag(SYNC_AIV_TO_AIC);
                    }
                } else if (cubeCount > 0) {
                    AscendC::CrossCoreWaitFlag(SYNC_AIV_TO_AIC);
                }
            }
            for (uint32_t groupIdx = 0; groupIdx < tiling->kGroupNum; ++groupIdx) {
                MMCompute(groupIdx, mnConfig, false);
            }
            if ASCEND_IS_AIC {
                AscendC::CrossCoreSetFlag<2, PIPE_FIX>(SYNC_AIC_TO_AIV);
                cubeCount++;
            }
            if ASCEND_IS_AIV {
                VectorComputeExpertKGrouped(mnConfig);
                cubeCount++;
            }
            curBlock += tiling->coreNum;
        }
        mnConfig.offsetM += expertM;
    }
    PostCompute();
}

__aicore__ inline void QuantGroupMatmul::ProcessExpertKGroupedMOuter()
{
    // Keep every N tile of one expert/M tile on the same core. Besides better
    // activation and weight locality, this establishes the execution order
    // needed to reuse one per-token-scale load across all N tiles.
    MNConfig mnConfig;
    InitMNConfigBase(mnConfig);
    uint32_t totalMTiles = 0;
    for (uint32_t expertIdx = 0; expertIdx < tiling->groupNum; ++expertIdx) {
        totalMTiles += AscendC::Ceil(
            static_cast<uint32_t>(groupListGm.GetValue(expertIdx)), mnConfig.singleM);
    }
    // Interleave expert M tiles globally. Serially assigning an expert to all
    // cores leaves cores idle for small per-expert M; this keeps the M-outer
    // locality while retaining the original global parallelism.
    for (uint32_t task = coreIdx; task < totalMTiles; task += tiling->coreNum) {
        uint32_t remaining = task;
        uint32_t expertIdx = 0;
        uint32_t offsetM = 0;
        uint32_t expertM = 0;
        for (; expertIdx < tiling->groupNum; ++expertIdx) {
            expertM = static_cast<uint32_t>(groupListGm.GetValue(expertIdx));
            const uint32_t expertTiles = AscendC::Ceil(expertM, mnConfig.singleM);
            if (remaining < expertTiles) {
                break;
            }
            remaining -= expertTiles;
            offsetM += expertM;
        }
        if (expertIdx == tiling->groupNum || expertM == 0) {
            continue;
        }
        mnConfig.expertIdx = expertIdx;
        mnConfig.offsetM = offsetM;
        mnConfig.m = expertM;
        mnConfig.blockDimM = AscendC::Ceil(mnConfig.m, mnConfig.singleM);
        mnConfig.mIdx = remaining;
        for (uint32_t nIdx = 0; nIdx < mnConfig.blockDimN; ++nIdx) {
                mnConfig.nIdx = nIdx;
                mnConfig.pipeIdx = cubeCount % tiling->parallNum;
                if ASCEND_IS_AIC {
                    if (tiling->enableTilePipeline != 0) {
                        if (cubeCount >= tiling->parallNum) {
                            AscendC::CrossCoreWaitFlag(SYNC_AIV_TO_AIC);
                        }
                    } else if (cubeCount > 0) {
                        AscendC::CrossCoreWaitFlag(SYNC_AIV_TO_AIC);
                    }
                }
                for (uint32_t groupIdx = 0; groupIdx < tiling->kGroupNum; ++groupIdx) {
                    MMCompute(groupIdx, mnConfig, false);
                }
                if ASCEND_IS_AIC {
                    AscendC::CrossCoreSetFlag<2, PIPE_FIX>(SYNC_AIC_TO_AIV);
                    cubeCount++;
                }
                if ASCEND_IS_AIV {
                    VectorComputeExpertKGrouped(mnConfig);
                    cubeCount++;
                }
        }
    }
    PostCompute();
}

__aicore__ inline void QuantGroupMatmul::ProcessKGroupedMOuter()
{
    MNConfig mnConfig;
    InitMNConfigBase(mnConfig);
    mnConfig.offsetM = 0;
    mnConfig.m = tiling->totalInGroup;
    mnConfig.blockDimM = AscendC::Ceil(mnConfig.m, mnConfig.singleM);

    for (uint32_t mBlock = coreIdx; mBlock < mnConfig.blockDimM; mBlock += tiling->coreNum) {
        mnConfig.mIdx = mBlock;
        mnConfig.nIdx = 0;
        mnConfig.pipeIdx = cubeCount % tiling->parallNum;
        PrepareTileDynamicInputs(mnConfig);

        for (uint32_t nIdx = 0; nIdx < mnConfig.blockDimN; ++nIdx) {
            mnConfig.nIdx = nIdx;
            mnConfig.pipeIdx = cubeCount % tiling->parallNum;
            if ASCEND_IS_AIC {
                if (tiling->enableTilePipeline != 0) {
                    if (cubeCount >= tiling->parallNum) {
                        AscendC::CrossCoreWaitFlag(SYNC_AIV_TO_AIC);
                    }
                } else if (cubeCount > 0) {
                    AscendC::CrossCoreWaitFlag(SYNC_AIV_TO_AIC);
                }
            }
            for (uint32_t groupIdx = 0; groupIdx < tiling->kGroupNum; ++groupIdx) {
                MMCompute(groupIdx, mnConfig, false);
            }
            if ASCEND_IS_AIC {
                AscendC::CrossCoreSetFlag<2, PIPE_FIX>(SYNC_AIC_TO_AIV);
                cubeCount++;
            }
            if ASCEND_IS_AIV {
                VectorComputeKGrouped(mnConfig);
                cubeCount++;
            }
        }
    }
    PostCompute();
}

__aicore__ inline void QuantGroupMatmul::ProcessKGroupedSharedMOuter()
{
    MNConfig mnConfig;
    InitMNConfigBase(mnConfig);
    mnConfig.offsetM = 0;
    mnConfig.m = tiling->totalInGroup;
    mnConfig.blockDimM = AscendC::Ceil(mnConfig.m, mnConfig.singleM);

    for (uint32_t mBlock = 0; mBlock < mnConfig.blockDimM; ++mBlock) {
        mnConfig.mIdx = mBlock;
        mnConfig.nIdx = 0;
        PrepareSharedTileDynamicInputs(mnConfig);
        // Shared tile workspace is consumed by all N-tile workers.
        AscendC::SyncAll<false>();

        for (uint32_t nIdx = coreIdx; nIdx < mnConfig.blockDimN; nIdx += tiling->coreNum) {
            mnConfig.nIdx = nIdx;
            mnConfig.pipeIdx = cubeCount % tiling->parallNum;
            if ASCEND_IS_AIC {
                if (tiling->enableTilePipeline != 0) {
                    if (cubeCount >= tiling->parallNum) {
                        AscendC::CrossCoreWaitFlag(SYNC_AIV_TO_AIC);
                    }
                } else if (cubeCount > 0) {
                    AscendC::CrossCoreWaitFlag(SYNC_AIV_TO_AIC);
                }
            }
            for (uint32_t groupIdx = 0; groupIdx < tiling->kGroupNum; ++groupIdx) {
                MMCompute(groupIdx, mnConfig, false);
            }
            if ASCEND_IS_AIC {
                AscendC::CrossCoreSetFlag<2, PIPE_FIX>(SYNC_AIC_TO_AIV);
                cubeCount++;
            }
            VectorComputeKGrouped(mnConfig);
        }
        // Do not overwrite the shared M-tile activation workspace until all N
        // consumers and their vector post-processing have completed.
        PostCompute();
        cubeCount = 0;
        AscendC::SyncAll<false>();
    }
}

__aicore__ inline void QuantGroupMatmul::InitMNConfigBase(MNConfig& mnConfig) const
{
    mnConfig.baseM = tiling->mmTilingData.baseM;
    mnConfig.baseN = tiling->mmTilingData.baseN;
    mnConfig.singleM = tiling->reserved0 == 0 ? mnConfig.baseM : tiling->reserved0;
    mnConfig.singleN = mnConfig.baseN;
    mnConfig.blockDimN = AscendC::Ceil(tiling->n, mnConfig.singleN);
}

__aicore__ inline void QuantGroupMatmul::RunLogicalGroup(
    uint32_t logicalGroupIdx,
    uint32_t logicalM,
    MNConfig& mnConfig)
{
    if (logicalM == 0) {
        return;
    }
    mnConfig.m = logicalM;
    mnConfig.blockDimM = AscendC::Ceil(mnConfig.m, mnConfig.singleM);
    const uint32_t curCount = mnConfig.blockDimN * mnConfig.blockDimM;
    uint32_t curBlock = coreIdx;

    while (curBlock < curCount) {
        mnConfig.mIdx = curBlock / mnConfig.blockDimN;
        mnConfig.nIdx = curBlock % mnConfig.blockDimN;
        MMCompute(logicalGroupIdx, mnConfig);
        VectorCompute(logicalGroupIdx, mnConfig);
        curBlock += tiling->coreNum;
    }
}

__aicore__ inline uint32_t QuantGroupMatmul::GetLogicalK() const
{
    if ((tiling->groupAxis == 1 || tiling->groupAxis == 2) && tiling->kGroupSize > 0) {
        return tiling->kGroupSize;
    }
    return tiling->k;
}

__aicore__ inline bool QuantGroupMatmul::UseTileLocalDynamicInputs() const
{
#if defined(GROUP_QUANT_TILE_DYNAMIC_EXPERIMENT) && defined(GROUP_QUANT_TILE_DYNAMIC_LOCAL_INPUTS)
    // Tile-local input mode will make each logical M tile consume a compact
    // [group, tileM, kGroupSize] xQ buffer and [group, tileM] scale buffer.
    return tiling->groupAxis == 1 && tiling->kGroupSize > 0;
#else
    return false;
#endif
}

__aicore__ inline bool QuantGroupMatmul::UseSharedTileDynamicInputs() const
{
#if defined(GROUP_QUANT_TILE_DYNAMIC_EXPERIMENT) && defined(GROUP_QUANT_TILE_DYNAMIC_LOCAL_INPUTS)
    return UseTileLocalDynamicInputs() && tiling->tileDynamicSchedule == 1;
#else
    return false;
#endif
}

__aicore__ inline uint64_t QuantGroupMatmul::GetXOffset(
    uint32_t groupIdx,
    const MNConfig& mnConfig) const
{
    const uint64_t tokenOffset =
        static_cast<uint64_t>(mnConfig.offsetM + mnConfig.mIdx * mnConfig.singleM);
    if (tiling->groupAxis == 1) {
        if (UseTileLocalDynamicInputs()) {
            // Dormant path for the future fused kernel: xGm will point at a
            // tile-local workspace instead of the full host-packed xQ tensor.
            if (UseSharedTileDynamicInputs()) {
                return static_cast<uint64_t>(groupIdx) *
                           mnConfig.singleM * tiling->kGroupSize;
            }
            return (
                static_cast<uint64_t>(coreIdx) * tiling->kGroupNum +
                static_cast<uint64_t>(groupIdx)) *
                mnConfig.singleM * tiling->kGroupSize;
        }
        // K-axis grouped path consumes host-packed [group, M, kGroupSize].
        return static_cast<uint64_t>(groupIdx) * tiling->totalInGroup * tiling->kGroupSize +
               tokenOffset * tiling->kGroupSize;
    }
    if (tiling->groupAxis == 2) {
        if ((tiling->reserved2 & 2U) != 0) {
            return (tokenOffset * tiling->kGroupNum + groupIdx) * tiling->kGroupSize;
        }
        return static_cast<uint64_t>(groupIdx) * tiling->totalInGroup * tiling->kGroupSize +
               tokenOffset * tiling->kGroupSize;
    }
    return tokenOffset * tiling->k;
}

__aicore__ inline uint64_t QuantGroupMatmul::GetWeightOffset(
    uint32_t groupIdx,
    const MNConfig& mnConfig) const
{
    const uint32_t logicalK = GetLogicalK();
    const uint64_t tailN = static_cast<uint64_t>(mnConfig.nIdx) * mnConfig.singleN;
    if (tiling->groupAxis == 2) {
        const uint64_t flatGroupIdx =
            static_cast<uint64_t>(mnConfig.expertIdx) * tiling->kGroupNum + groupIdx;
        return flatGroupIdx * tiling->n * logicalK + tailN * logicalK;
    }
    return static_cast<uint64_t>(groupIdx) * tiling->n * logicalK + tailN * logicalK;
}

__aicore__ inline void QuantGroupMatmul::MMCompute(uint32_t groupIdx, MNConfig& mnConfig, bool syncVector)
{
    uint32_t tailN = mnConfig.nIdx * mnConfig.singleN;
    uint32_t curSingleN = mnConfig.singleN;
    if (mnConfig.nIdx == mnConfig.blockDimN - 1) {
        curSingleN = tiling->n - tailN;
    }
    uint32_t curSingleM = mnConfig.singleM;
    if (mnConfig.mIdx == mnConfig.blockDimM - 1) {
        curSingleM = mnConfig.m - mnConfig.mIdx * mnConfig.singleM;
    }
    const uint32_t logicalK = GetLogicalK();
    uint64_t xOffset = GetXOffset(groupIdx, mnConfig);
    uint64_t weightOffset = GetWeightOffset(groupIdx, mnConfig);  // for no transpose nz weight
    if (tiling->groupAxis == 1 || tiling->groupAxis == 2) {
        // K-axis grouped accumulation processes all groups of one logical tile
        // consecutively. Each group gets its own per-core slot so AIC cannot
        // overwrite a previous group's mmOut while another AIV sub-block is
        // still consuming it.
        mnConfig.workSpaceOffset =
            mnConfig.singleN * mnConfig.singleM *
            (((tiling->enableTilePipeline != 0
                   ? static_cast<uint64_t>(mnConfig.pipeIdx) * tiling->kGroupNum
                   : 0ULL) +
              groupIdx) *
                 tiling->coreNum +
             coreIdx);
    } else {
        mnConfig.workSpaceOffset = mnConfig.singleN * mnConfig.singleM * \
                                       (coreIdx + (cubeCount % tiling->parallNum) * tiling->coreNum);
    }
    if ASCEND_IS_AIC {
        if (!syncVector) {
            // Synchronization is handled once per logical K-grouped tile.
        } else if (tiling->groupAxis == 1 || tiling->groupAxis == 2) {
            if (cubeCount > 0) {
                AscendC::CrossCoreWaitFlag(SYNC_AIV_TO_AIC);
            }
        } else if (cubeCount >= tiling->parallNum) {
            AscendC::CrossCoreWaitFlag(SYNC_AIV_TO_AIC);
        }
        if (UseTileLocalDynamicInputs()) {
            mm.SetOrgShape(curSingleM, curSingleN, logicalK);
        } else {
            const uint32_t strideKa = (tiling->reserved2 & 2U) != 0
                ? logicalK * tiling->kGroupNum : logicalK;
            mm.SetOrgShape(mnConfig.m, curSingleN, strideKa, logicalK, curSingleN);
        }
        mm.SetSingleShape(curSingleM, curSingleN, logicalK);
        if (UseTileLocalDynamicInputs()) {
            mm.SetTensorA(tileXGm[xOffset]);
        } else {
            mm.SetTensorA(xGm[xOffset]);
        }
        auto weightSlice = weightGm[weightOffset];
        if (mnConfig.blockDimM == 1) {
            weightSlice.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        }
        mm.SetTensorB(weightSlice);
        uint64_t worskspaceOffset = mnConfig.workSpaceOffset;
        while (mm.Iterate()) {
            mm.GetTensorC(mmOutGm[worskspaceOffset], 0);
            if (syncVector) {
                AscendC::CrossCoreSetFlag<2, PIPE_FIX>(SYNC_AIC_TO_AIV);  // 2: mode为2, group内同步
            }
            worskspaceOffset += (mnConfig.baseM * mnConfig.baseN);
        }
    }
    if (syncVector) {
        cubeCount++;
    }
}

__aicore__ inline void QuantGroupMatmul::PrepareTileDynamicInputs(MNConfig& mnConfig)
{
    if (!UseTileLocalDynamicInputs()) {
        return;
    }
#if defined(GROUP_QUANT_TILE_DYNAMIC_EXPERIMENT) && defined(GROUP_QUANT_TILE_DYNAMIC_LOCAL_INPUTS)
    uint32_t curSingleM = mnConfig.singleM;
    if (mnConfig.mIdx == mnConfig.blockDimM - 1) {
        curSingleM = tiling->totalInGroup - mnConfig.mIdx * mnConfig.singleM;
    }
    if ASCEND_IS_AIV {
        if (subBlockIdx == 0) {
            LocalTensor<int8_t> qRowLocal = tileQuantRowBuff.Get<int8_t>();
            LocalTensor<bfloat16_t> bf16StageLocal = tileQuantRowBuff.Get<bfloat16_t>();
            LocalTensor<float> rawFloatLocal = tileRawFloatBuff.Get<float>();
            LocalTensor<float> scaleStoreLocal = tileScaleBuff.Get<float>();
            const uint32_t alignedK = AlignUp64(tiling->kGroupSize, 8);
            for (uint32_t groupIdx = 0; groupIdx < tiling->kGroupNum; ++groupIdx) {
                for (uint32_t localM = 0; localM < curSingleM; ++localM) {
                    const uint64_t globalM =
                        static_cast<uint64_t>(mnConfig.offsetM) +
                        static_cast<uint64_t>(mnConfig.mIdx) * mnConfig.singleM +
                        localM;
                    const uint64_t rawXOffset =
                        globalM * (static_cast<uint64_t>(tiling->kGroupNum) * tiling->kGroupSize) +
                        static_cast<uint64_t>(groupIdx) * tiling->kGroupSize;
                    const uint64_t dstXOffset =
                        (static_cast<uint64_t>(coreIdx) * tiling->kGroupNum +
                         static_cast<uint64_t>(groupIdx)) *
                            mnConfig.singleM * tiling->kGroupSize +
                        static_cast<uint64_t>(localM) * tiling->kGroupSize;

                // Vector load bf16 -> cast to float
                AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(tiling->kGroupSize * sizeof(bfloat16_t)), 0, 0, 0};
                AscendC::DataCopyPadExtParams<bfloat16_t> padParams{true, 0, 0, 0};
                DataCopyPad(bf16StageLocal, rawXGm[rawXOffset], copyParams, padParams);
                AscendC::PipeBarrier<PIPE_MTE2>();
                Cast(rawFloatLocal, bf16StageLocal, AscendC::RoundMode::CAST_NONE, alignedK);
                AscendC::PipeBarrier<PIPE_V>();

                // Vector abs + reduce max
                Abs(rawFloatLocal, rawFloatLocal, alignedK);
                AscendC::PipeBarrier<PIPE_V>();
                float maxAbs = 0.0f;
                LocalTensor<float> reduceTmpLocal = sharedTmpLocal.ReinterpretCast<float>();
                ReduceMax(rawFloatLocal, rawFloatLocal, reduceTmpLocal, static_cast<int32_t>(alignedK), false);
                AscendC::PipeBarrier<PIPE_V>();
                maxAbs = rawFloatLocal.GetValue(0);

                const float tokenScale = maxAbs / 127.0f;

                // Re-load raw data for quantization
                DataCopyPad(bf16StageLocal, rawXGm[rawXOffset], copyParams, padParams);
                AscendC::PipeBarrier<PIPE_MTE2>();
                Cast(rawFloatLocal, bf16StageLocal, AscendC::RoundMode::CAST_NONE, alignedK);
                AscendC::PipeBarrier<PIPE_V>();

                // Vector quantize: x / scale -> round -> clamp -> int8
                if (tokenScale > 0.0f) {
                    Muls(rawFloatLocal, rawFloatLocal, 1.0f / tokenScale, alignedK);
                    AscendC::PipeBarrier<PIPE_V>();
                }
                Maxs(rawFloatLocal, rawFloatLocal, -127.0f, alignedK);
                AscendC::PipeBarrier<PIPE_V>();
                Mins(rawFloatLocal, rawFloatLocal, 127.0f, alignedK);
                AscendC::PipeBarrier<PIPE_V>();
                // float -> int16 -> int8
                LocalTensor<int16_t> i16Tmp = rawFloatLocal.ReinterpretCast<int16_t>();
                Cast(i16Tmp, rawFloatLocal, AscendC::RoundMode::CAST_RINT, alignedK);
                AscendC::PipeBarrier<PIPE_V>();
                Cast(qRowLocal, i16Tmp, AscendC::RoundMode::CAST_RINT, alignedK);
                AscendC::PipeBarrier<PIPE_V>();

                    DataCopy(tileXGm[dstXOffset], qRowLocal, tiling->kGroupSize);
                    scaleStoreLocal.SetValue(localM, tokenScale);
                    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(SCALAR_TO_MTE3_EVENT_ID);
                    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(SCALAR_TO_MTE3_EVENT_ID);
                    AscendC::PipeBarrier<PIPE_MTE3>();
                }
                const uint64_t dstScaleOffset =
                    (static_cast<uint64_t>(coreIdx) * tiling->kGroupNum +
                     static_cast<uint64_t>(groupIdx)) *
                        mnConfig.singleM;
                AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(SCALAR_TO_MTE3_EVENT_ID);
                AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(SCALAR_TO_MTE3_EVENT_ID);
                DataCopyPad2D(tilePerTokenScaleGm[dstScaleOffset], scaleStoreLocal,
                              1, curSingleM, curSingleM, curSingleM);
            }
            AscendC::PipeBarrier<PIPE_ALL>();
            CleanGmCacheRange(
                tileXWorkspaceAddr +
                    static_cast<uint64_t>(coreIdx) * tiling->kGroupNum * mnConfig.singleM * tiling->kGroupSize,
                static_cast<uint64_t>(tiling->kGroupNum) * mnConfig.singleM * tiling->kGroupSize);
            CleanGmCacheRange(
                tileScaleWorkspaceAddr +
                    static_cast<uint64_t>(coreIdx) * tiling->kGroupNum * mnConfig.singleM * sizeof(float),
                static_cast<uint64_t>(tiling->kGroupNum) * mnConfig.singleM * sizeof(float));
        }
        AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(SYNC_DYNAMIC_QUANT_DONE);
    }
    if ASCEND_IS_AIC {
        AscendC::CrossCoreWaitFlag(SYNC_DYNAMIC_QUANT_DONE);
    }
#endif
}

__aicore__ inline void QuantGroupMatmul::PrepareSharedTileDynamicInputs(MNConfig& mnConfig)
{
    if (!UseSharedTileDynamicInputs()) {
        return;
    }
#if defined(GROUP_QUANT_TILE_DYNAMIC_EXPERIMENT) && defined(GROUP_QUANT_TILE_DYNAMIC_LOCAL_INPUTS)
    uint32_t curSingleM = mnConfig.singleM;
    if (mnConfig.mIdx == mnConfig.blockDimM - 1) {
        curSingleM = tiling->totalInGroup - mnConfig.mIdx * mnConfig.singleM;
    }
    if ASCEND_IS_AIV {
        if (subBlockIdx != 0) {
            return;
        }
        LocalTensor<int8_t> qRowLocal = tileQuantRowBuff.Get<int8_t>();
        LocalTensor<bfloat16_t> bf16StageLocal = tileQuantRowBuff.Get<bfloat16_t>();
        LocalTensor<float> rawFloatLocal = tileRawFloatBuff.Get<float>();
        const uint32_t alignedK = AlignUp64(tiling->kGroupSize, 8);
        const uint32_t totalRows = curSingleM * tiling->kGroupNum;
        for (uint32_t packedRow = coreIdx; packedRow < totalRows; packedRow += tiling->coreNum) {
            const uint32_t groupIdx = packedRow / curSingleM;
            const uint32_t localM = packedRow - groupIdx * curSingleM;
            const uint64_t globalM =
                static_cast<uint64_t>(mnConfig.offsetM) +
                static_cast<uint64_t>(mnConfig.mIdx) * mnConfig.singleM +
                localM;
            const uint64_t rawXOffset =
                globalM * (static_cast<uint64_t>(tiling->kGroupNum) * tiling->kGroupSize) +
                static_cast<uint64_t>(groupIdx) * tiling->kGroupSize;
            const uint64_t dstXOffset =
                static_cast<uint64_t>(groupIdx) * mnConfig.singleM * tiling->kGroupSize +
                static_cast<uint64_t>(localM) * tiling->kGroupSize;

            // Vector load bf16 -> cast to float
            AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(tiling->kGroupSize * sizeof(bfloat16_t)), 0, 0, 0};
            AscendC::DataCopyPadExtParams<bfloat16_t> padParams{true, 0, 0, 0};
            DataCopyPad(bf16StageLocal, rawXGm[rawXOffset], copyParams, padParams);
            AscendC::PipeBarrier<PIPE_MTE2>();
            Cast(rawFloatLocal, bf16StageLocal, AscendC::RoundMode::CAST_NONE, alignedK);
            AscendC::PipeBarrier<PIPE_V>();

            // Vector abs + reduce max
            Abs(rawFloatLocal, rawFloatLocal, alignedK);
            AscendC::PipeBarrier<PIPE_V>();
            LocalTensor<float> reduceTmpLocal = sharedTmpLocal.ReinterpretCast<float>();
            ReduceMax(rawFloatLocal, rawFloatLocal, reduceTmpLocal, static_cast<int32_t>(alignedK), false);
            AscendC::PipeBarrier<PIPE_V>();
            float maxAbs = rawFloatLocal.GetValue(0);

            const float tokenScale = maxAbs / 127.0f;

            // Re-load for quantization
            DataCopyPad(bf16StageLocal, rawXGm[rawXOffset], copyParams, padParams);
            AscendC::PipeBarrier<PIPE_MTE2>();
            Cast(rawFloatLocal, bf16StageLocal, AscendC::RoundMode::CAST_NONE, alignedK);
            AscendC::PipeBarrier<PIPE_V>();

            // Vector quantize
            if (tokenScale > 0.0f) {
                Muls(rawFloatLocal, rawFloatLocal, 1.0f / tokenScale, alignedK);
                AscendC::PipeBarrier<PIPE_V>();
            }
            Maxs(rawFloatLocal, rawFloatLocal, -127.0f, alignedK);
            AscendC::PipeBarrier<PIPE_V>();
            Mins(rawFloatLocal, rawFloatLocal, 127.0f, alignedK);
            AscendC::PipeBarrier<PIPE_V>();
            LocalTensor<int16_t> i16Tmp = rawFloatLocal.ReinterpretCast<int16_t>();
            Cast(i16Tmp, rawFloatLocal, AscendC::RoundMode::CAST_RINT, alignedK);
            AscendC::PipeBarrier<PIPE_V>();
            Cast(qRowLocal, i16Tmp, AscendC::RoundMode::CAST_RINT, alignedK);
            AscendC::PipeBarrier<PIPE_V>();

            DataCopy(tileXGm[dstXOffset], qRowLocal, tiling->kGroupSize);
            AscendC::PipeBarrier<PIPE_MTE3>();

            const uint64_t dstScaleOffset =
                static_cast<uint64_t>(groupIdx) * mnConfig.singleM + localM;
            LocalTensor<float> scaleStoreLocal = sharedTmpLocal.ReinterpretCast<float>();
            scaleStoreLocal.SetValue(0, tokenScale);
            AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(SCALAR_TO_MTE3_EVENT_ID);
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(SCALAR_TO_MTE3_EVENT_ID);
            DataCopyPad2D(tilePerTokenScaleGm[dstScaleOffset], scaleStoreLocal, 1, 1, 1, 1);
            AscendC::PipeBarrier<PIPE_MTE3>();
            CleanGmCacheRange(tileXWorkspaceAddr + dstXOffset, tiling->kGroupSize);
            CleanGmCacheRange(tileScaleWorkspaceAddr + dstScaleOffset * sizeof(float), sizeof(float));
        }
    }
#endif
}

__aicore__ inline void QuantGroupMatmul::VectorCompute(uint32_t groupIdx, MNConfig& mnConfig)
{
    if ASCEND_IS_AIC {
        return;
    }
    uint32_t curCubeSingleN = mnConfig.singleN;
    if (mnConfig.nIdx == mnConfig.blockDimN - 1) {
        curCubeSingleN = tiling->n - mnConfig.nIdx * mnConfig.singleN;
    }
    uint32_t curCubeSingleM = mnConfig.singleM;
    if (mnConfig.mIdx == mnConfig.blockDimM - 1) {
        curCubeSingleM = mnConfig.m - mnConfig.mIdx * mnConfig.singleM;
    }
    uint32_t vecBaseM = tiling->ubCalSize / (AscendC::Ceil(mnConfig.baseN, uint32_t(8)) * 8);  //  8: num int32_t in 32B ub block
    vecBaseM = vecBaseM < curCubeSingleM ? vecBaseM : curCubeSingleM;
    uint32_t curVecBaseN = mnConfig.baseN;
    uint64_t scaleOffset = GetScaleOffset(groupIdx, mnConfig);
    uint64_t outOffset = (mnConfig.offsetM + mnConfig.mIdx * mnConfig.singleM) * tiling->n + \
                            mnConfig.nIdx * mnConfig.singleN;
    if (tiling->groupAxis == 1 && subBlockIdx != 0) {
        AscendC::CrossCoreWaitFlag(SYNC_AIC_TO_AIV);
        AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(SYNC_AIV_TO_AIC);
        return;
    }
    uint32_t taskRation = tiling->groupAxis == 1 ? 0 : AscendC::GetTaskRation();
    uint32_t offsetN = 0;
    if (offsetN < curCubeSingleN) {
        curVecBaseN = mnConfig.baseN;
        if (unlikely(offsetN + mnConfig.baseN >= curCubeSingleN)) {
            curVecBaseN = curCubeSingleN - offsetN;
        }
        EnqueueScale(curVecBaseN, scaleOffset + offsetN);
    }
    for (uint32_t vecCount = 0; offsetN < curCubeSingleN; offsetN += mnConfig.baseN) {
        if (unlikely(offsetN + mnConfig.baseN >= curCubeSingleN)) {
            curVecBaseN = curCubeSingleN - offsetN;
        } else {
            curVecBaseN = mnConfig.baseN;
        }
        uint32_t alignBaseN = AscendC::Ceil(curVecBaseN, uint32_t(8)) * 8;  //  8: num int32_t in 32B ub block
        DequeueScale(alignBaseN);
        uint32_t nextOffsetN = offsetN + mnConfig.baseN;
        if (nextOffsetN < curCubeSingleN) {
            uint32_t nextVecBaseN = mnConfig.baseN;
            if (unlikely(nextOffsetN + mnConfig.baseN >= curCubeSingleN)) {
                nextVecBaseN = curCubeSingleN - nextOffsetN;
            }
            EnqueueScale(nextVecBaseN, scaleOffset + nextOffsetN);
        }
        uint32_t curVecBaseM = vecBaseM;
        uint64_t mmOutOffset = mnConfig.workSpaceOffset + offsetN * mnConfig.baseM;
        AscendC::CrossCoreWaitFlag(SYNC_AIC_TO_AIV);
        for (uint32_t offsetM = 0; offsetM < curCubeSingleM; offsetM += vecBaseM) {
            vecCount++;
            if (taskRation != 0 && vecCount % taskRation != subBlockIdx) {
                continue;
            }
            if (unlikely(offsetM + vecBaseM >= curCubeSingleM)) {
                curVecBaseM = curCubeSingleM - offsetM;
            }
            EnqueuePerTokenScale(groupIdx, mnConfig, curVecBaseM, offsetM);
            BroadcastPerTokenScale(curVecBaseM, alignBaseN);
            // 使用AscendDequant接口做perchannel反量化
            LocalTensor<cT::T> mmOutLocal = vecInQueue.AllocTensor<cT::T>();
            DataCopyPad2D(mmOutLocal, mmOutGm[mmOutOffset + offsetM * curVecBaseN],
                          curVecBaseM, curVecBaseN, curVecBaseN);
            vecInQueue.EnQue(mmOutLocal);
            ComputeDequantAndActivate(
                mnConfig,
                curVecBaseM,
                alignBaseN,
                curVecBaseN,
                offsetM,
                outOffset,
                offsetN,
                groupIdx,
                outOffset + offsetM * tiling->n + offsetN);
            if (tiling->groupAxis != 1 || groupIdx + 1 == tiling->kGroupNum) {
                LocalTensor<bfloat16_t> yLocal = vecOutQueue.DeQue<bfloat16_t>();
                DataCopyPad2D(yGm[outOffset + offsetM * tiling->n + offsetN], yLocal,
                              curVecBaseM, curVecBaseN, alignBaseN, tiling->n);
                vecOutQueue.FreeTensor(yLocal);
            }
        }
        scaleInQueue.FreeTensor(scaleInUb);
    }
    AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(SYNC_AIV_TO_AIC);  // 2: mode为2, group内同步
}

__aicore__ inline void QuantGroupMatmul::VectorComputeKGrouped(MNConfig& mnConfig)
{
    if ASCEND_IS_AIC {
        return;
    }
    uint32_t curCubeSingleN = mnConfig.singleN;
    if (mnConfig.nIdx == mnConfig.blockDimN - 1) {
        curCubeSingleN = tiling->n - mnConfig.nIdx * mnConfig.singleN;
    }
    uint32_t curCubeSingleM = mnConfig.singleM;
    if (mnConfig.mIdx == mnConfig.blockDimM - 1) {
        curCubeSingleM = mnConfig.m - mnConfig.mIdx * mnConfig.singleM;
    }
    uint32_t vecBaseM = tiling->ubCalSize / (AscendC::Ceil(mnConfig.baseN, uint32_t(8)) * 8);
    vecBaseM = vecBaseM < curCubeSingleM ? vecBaseM : curCubeSingleM;
    uint64_t outOffset = (mnConfig.offsetM + mnConfig.mIdx * mnConfig.singleM) * tiling->n +
                         mnConfig.nIdx * mnConfig.singleN;

    AscendC::CrossCoreWaitFlag(SYNC_AIC_TO_AIV);
    if (subBlockIdx != 0) {
        AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(SYNC_AIV_TO_AIC);
        return;
    }
    for (uint32_t offsetN = 0; offsetN < curCubeSingleN; offsetN += mnConfig.baseN) {
        uint32_t curVecBaseN = mnConfig.baseN;
        if (unlikely(offsetN + mnConfig.baseN >= curCubeSingleN)) {
            curVecBaseN = curCubeSingleN - offsetN;
        }
        uint32_t alignBaseN = AscendC::Ceil(curVecBaseN, uint32_t(8)) * 8;
        const bool reuseSingleGroupScale = tiling->kGroupNum == 1;
        if (reuseSingleGroupScale) {
            // Channel scale is invariant across the eight M-slices of this
            // N tile. Keep the dequeued tensor alive until every slice has
            // consumed it; FreeTensor retains the original TQue dependency.
            const uint64_t scaleOffset = GetScaleOffset(0, mnConfig);
            EnqueueScale(curVecBaseN, scaleOffset + offsetN);
            DequeueScale(alignBaseN);
        }

        for (uint32_t offsetM = 0; offsetM < curCubeSingleM; offsetM += vecBaseM) {
            uint32_t curVecBaseM = vecBaseM;
            if (unlikely(offsetM + vecBaseM >= curCubeSingleM)) {
                curVecBaseM = curCubeSingleM - offsetM;
            }

            // The registry GMM2 ABI provides one native activation scale per
            // routed token. All K groups reuse it, so retain the broadcast
            // value in UB across the complete K-group accumulation.
            EnqueuePerTokenScale(0, mnConfig, curVecBaseM, offsetM);
            BroadcastPerTokenScale(curVecBaseM, alignBaseN);

            for (uint32_t groupIdx = 0; groupIdx < tiling->kGroupNum; ++groupIdx) {
                const uint64_t scaleOffset = GetScaleOffset(groupIdx, mnConfig);
                if (!reuseSingleGroupScale) {
                    EnqueueScale(curVecBaseN, scaleOffset + offsetN);
                    DequeueScale(alignBaseN);
                }
                uint64_t mmOutOffset =
                    mnConfig.singleN * mnConfig.singleM *
                    (((tiling->enableTilePipeline != 0
                           ? static_cast<uint64_t>(mnConfig.pipeIdx) * tiling->kGroupNum
                           : 0ULL) +
                      groupIdx) *
                         tiling->coreNum +
                     coreIdx) +
                    offsetN * mnConfig.baseM;
                LocalTensor<cT::T> mmOutLocal = vecInQueue.AllocTensor<cT::T>();
                DataCopyPad2D(mmOutLocal, mmOutGm[mmOutOffset + offsetM * curVecBaseN],
                              curVecBaseM, curVecBaseN, curVecBaseN);
                vecInQueue.EnQue(mmOutLocal);
                ComputeDequantAndActivate(
                    mnConfig,
                    curVecBaseM,
                    alignBaseN,
                    curVecBaseN,
                    offsetM,
                    outOffset,
                    offsetN,
                    groupIdx,
                    outOffset + offsetM * tiling->n + offsetN);
                if (!reuseSingleGroupScale) {
                    scaleInQueue.FreeTensor(scaleInUb);
                }
            }

            LocalTensor<bfloat16_t> yLocal = vecOutQueue.DeQue<bfloat16_t>();
            DataCopyPad2D(yGm[outOffset + offsetM * tiling->n + offsetN], yLocal,
                          curVecBaseM, curVecBaseN, alignBaseN, tiling->n);
            vecOutQueue.FreeTensor(yLocal);
        }
        if (reuseSingleGroupScale) {
            scaleInQueue.FreeTensor(scaleInUb);
        }
    }
    AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(SYNC_AIV_TO_AIC);
}

__aicore__ inline void QuantGroupMatmul::VectorComputeExpertKGrouped(MNConfig& mnConfig)
{
    if ASCEND_IS_AIC {
        return;
    }
    uint32_t curCubeSingleN = mnConfig.singleN;
    if (mnConfig.nIdx == mnConfig.blockDimN - 1) {
        curCubeSingleN = tiling->n - mnConfig.nIdx * mnConfig.singleN;
    }
    uint32_t curCubeSingleM = mnConfig.singleM;
    if (mnConfig.mIdx == mnConfig.blockDimM - 1) {
        curCubeSingleM = mnConfig.m - mnConfig.mIdx * mnConfig.singleM;
    }
    uint32_t vecBaseM = tiling->ubCalSize / (AscendC::Ceil(mnConfig.baseN, uint32_t(8)) * 8);
    vecBaseM = vecBaseM < curCubeSingleM ? vecBaseM : curCubeSingleM;
    uint64_t outOffset = (mnConfig.offsetM + mnConfig.mIdx * mnConfig.singleM) * tiling->n +
                         mnConfig.nIdx * mnConfig.singleN;

    AscendC::CrossCoreWaitFlag(SYNC_AIC_TO_AIV);
    const bool dualAiv = (tiling->reserved2 & 1U) != 0;
    // Both sub-blocks already participate in the group ready/free protocol.
    // Give each a disjoint row interval without changing flag multiplicity.
    const uint32_t firstRows = (curCubeSingleM + 1) / 2;
    const uint32_t rowBegin = dualAiv && subBlockIdx != 0 ? firstRows : 0;
    const uint32_t rowEnd = dualAiv && subBlockIdx == 0 ? firstRows : curCubeSingleM;
    if ((!dualAiv && subBlockIdx != 0) || rowBegin == rowEnd) {
        AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(SYNC_AIV_TO_AIC);
        return;
    }
    for (uint32_t offsetN = 0; offsetN < curCubeSingleN; offsetN += mnConfig.baseN) {
        uint32_t curVecBaseN = mnConfig.baseN;
        if (unlikely(offsetN + mnConfig.baseN >= curCubeSingleN)) {
            curVecBaseN = curCubeSingleN - offsetN;
        }
        uint32_t alignBaseN = AscendC::Ceil(curVecBaseN, uint32_t(8)) * 8;
        const bool reuseSingleGroupScale = tiling->kGroupNum == 1;
        if (reuseSingleGroupScale) {
            // One channel-scale load serves all Vector M-slices for this
            // expert/N tile while TQue preserves the MTE2/V lifecycle.
            const uint64_t scaleOffset = GetScaleOffset(0, mnConfig);
            EnqueueScale(curVecBaseN, scaleOffset + offsetN);
            DequeueScale(alignBaseN);
        }
        for (uint32_t offsetM = rowBegin; offsetM < rowEnd; offsetM += vecBaseM) {
            uint32_t curVecBaseM = vecBaseM;
            if (unlikely(offsetM + vecBaseM >= rowEnd)) {
                curVecBaseM = rowEnd - offsetM;
            }

            // The expert registry ABI carries one activation scale per routed
            // token, not one scale per K group.  Load and broadcast it once
            // for the whole K-group accumulation instead of repeating the
            // same GM copy and vector broadcast for every group.
            EnqueuePerTokenScale(0, mnConfig, curVecBaseM, offsetM);
            BroadcastPerTokenScale(curVecBaseM, alignBaseN);

            for (uint32_t groupIdx = 0; groupIdx < tiling->kGroupNum; ++groupIdx) {
                const uint64_t scaleOffset = GetScaleOffset(groupIdx, mnConfig);
                if (!reuseSingleGroupScale) {
                    EnqueueScale(curVecBaseN, scaleOffset + offsetN);
                    DequeueScale(alignBaseN);
                }
                uint64_t mmOutOffset =
                    mnConfig.singleN * mnConfig.singleM *
                    (((tiling->enableTilePipeline != 0
                           ? static_cast<uint64_t>(mnConfig.pipeIdx) * tiling->kGroupNum
                           : 0ULL) +
                      groupIdx) *
                         tiling->coreNum +
                     coreIdx) +
                    offsetN * mnConfig.baseM;
                LocalTensor<cT::T> mmOutLocal = vecInQueue.AllocTensor<cT::T>();
                DataCopyPad2D(mmOutLocal, mmOutGm[mmOutOffset + offsetM * curVecBaseN],
                              curVecBaseM, curVecBaseN, curVecBaseN);
                vecInQueue.EnQue(mmOutLocal);
                ComputeDequantAndActivate(
                    mnConfig,
                    curVecBaseM,
                    alignBaseN,
                    curVecBaseN,
                    offsetM,
                    outOffset,
                    offsetN,
                    groupIdx,
                    outOffset + offsetM * tiling->n + offsetN);
                if (!reuseSingleGroupScale) {
                    scaleInQueue.FreeTensor(scaleInUb);
                }
            }

            LocalTensor<bfloat16_t> yLocal = vecOutQueue.DeQue<bfloat16_t>();
            DataCopyPad2D(yGm[outOffset + offsetM * tiling->n + offsetN], yLocal,
                          curVecBaseM, curVecBaseN, alignBaseN, tiling->n);
            vecOutQueue.FreeTensor(yLocal);
        }
        if (reuseSingleGroupScale) {
            scaleInQueue.FreeTensor(scaleInUb);
        }
    }
    AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(SYNC_AIV_TO_AIC);
}

__aicore__ inline void QuantGroupMatmul::VectorComputeKGroupedShared(MNConfig& mnConfig)
{
    if ASCEND_IS_AIC {
        return;
    }
    if (subBlockIdx != 0) {
        return;
    }
    uint32_t curCubeSingleN = mnConfig.singleN;
    if (mnConfig.nIdx == mnConfig.blockDimN - 1) {
        curCubeSingleN = tiling->n - mnConfig.nIdx * mnConfig.singleN;
    }
    uint32_t curCubeSingleM = mnConfig.singleM;
    if (mnConfig.mIdx == mnConfig.blockDimM - 1) {
        curCubeSingleM = mnConfig.m - mnConfig.mIdx * mnConfig.singleM;
    }
    uint32_t vecBaseM = tiling->ubCalSize / (AscendC::Ceil(mnConfig.baseN, uint32_t(8)) * 8);
    vecBaseM = vecBaseM < curCubeSingleM ? vecBaseM : curCubeSingleM;
    uint64_t outOffset = (mnConfig.offsetM + mnConfig.mIdx * mnConfig.singleM) * tiling->n +
                         mnConfig.nIdx * mnConfig.singleN;

    for (uint32_t offsetN = 0; offsetN < curCubeSingleN; offsetN += mnConfig.baseN) {
        uint32_t curVecBaseN = mnConfig.baseN;
        if (unlikely(offsetN + mnConfig.baseN >= curCubeSingleN)) {
            curVecBaseN = curCubeSingleN - offsetN;
        }
        uint32_t alignBaseN = AscendC::Ceil(curVecBaseN, uint32_t(8)) * 8;

        for (uint32_t offsetM = 0; offsetM < curCubeSingleM; offsetM += vecBaseM) {
            uint32_t curVecBaseM = vecBaseM;
            if (unlikely(offsetM + vecBaseM >= curCubeSingleM)) {
                curVecBaseM = curCubeSingleM - offsetM;
            }

            for (uint32_t groupIdx = 0; groupIdx < tiling->kGroupNum; ++groupIdx) {
                uint64_t scaleOffset = GetScaleOffset(groupIdx, mnConfig);
                EnqueueScale(curVecBaseN, scaleOffset + offsetN);
                DequeueScale(alignBaseN);
                EnqueuePerTokenScale(groupIdx, mnConfig, curVecBaseM, offsetM);
                BroadcastPerTokenScale(curVecBaseM, alignBaseN);

                uint64_t mmOutOffset =
                    mnConfig.singleN * mnConfig.singleM *
                    (static_cast<uint64_t>(groupIdx) * tiling->coreNum + coreIdx) +
                    offsetN * mnConfig.baseM;
                LocalTensor<cT::T> mmOutLocal = vecInQueue.AllocTensor<cT::T>();
                DataCopyPad2D(mmOutLocal, mmOutGm[mmOutOffset + offsetM * curVecBaseN],
                              curVecBaseM, curVecBaseN, curVecBaseN);
                vecInQueue.EnQue(mmOutLocal);
                ComputeDequantAndActivate(
                    mnConfig,
                    curVecBaseM,
                    alignBaseN,
                    curVecBaseN,
                    offsetM,
                    outOffset,
                    offsetN,
                    groupIdx,
                    outOffset + offsetM * tiling->n + offsetN);
                scaleInQueue.FreeTensor(scaleInUb);
            }

            LocalTensor<bfloat16_t> yLocal = vecOutQueue.DeQue<bfloat16_t>();
            DataCopyPad2D(yGm[outOffset + offsetM * tiling->n + offsetN], yLocal,
                          curVecBaseM, curVecBaseN, alignBaseN, tiling->n);
            vecOutQueue.FreeTensor(yLocal);
        }
    }
}

__aicore__ inline void QuantGroupMatmul::PostCompute() {
    if ASCEND_IS_AIC {
        if (tiling->groupAxis == 1 || tiling->groupAxis == 2) {
            if (likely(cubeCount > 0)) {
                uint32_t loop = tiling->enableTilePipeline != 0
                                    ? (cubeCount < tiling->parallNum ? cubeCount : tiling->parallNum)
                                    : 1;
                for (int32_t idx = 0; idx < loop; ++idx) {
                    AscendC::CrossCoreWaitFlag(SYNC_AIV_TO_AIC);
                }
            }
            return;
        }
        if (likely(cubeCount > 0)) {  // exits last cycle wait
            uint32_t loop = cubeCount < tiling->parallNum ? cubeCount : tiling->parallNum;
            for (int32_t idx = 0; idx < loop; ++idx) {
                AscendC::CrossCoreWaitFlag(SYNC_AIV_TO_AIC);
            }
        }
    }
}

__aicore__ inline void QuantGroupMatmul::ComputeDequantAndActivate(MNConfig& mnConfig, 
    uint32_t curVecBaseM, uint32_t alignBaseN, uint32_t curVecBaseN, uint32_t offsetM,
    uint64_t outOffset, uint32_t offsetN, uint32_t groupIdx, uint64_t partialOffset)
{
    LocalTensor<int32_t> mmOutInUb = vecInQueue.DeQue<cT::T>();
    AscendDequant(dequantMiddleResult, mmOutInUb, scaleInUb, sharedTmpLocal,
                  {curVecBaseM, alignBaseN, curVecBaseN});
    AscendC::PipeBarrier<PIPE_V>();
    vecInQueue.FreeTensor(mmOutInUb);
    // pertoken反量化
    uint32_t computeSize = curVecBaseM * alignBaseN;
    uint32_t validComputeSize = curVecBaseM * curVecBaseN;
    Mul(mulsResultLocal, dequantMiddleResult, pertokenBrcbLocal, computeSize);
    AscendC::PipeBarrier<PIPE_V>();
    const bool groupedOutput = tiling->groupAxis == 1 || tiling->groupAxis == 2;
    const bool directSingleGroup = groupedOutput && tiling->kGroupNum == 1;
    if (groupedOutput && !directSingleGroup) {
        if (groupIdx == 0) {
            Adds(actResultLocal, mulsResultLocal, static_cast<float>(0.0), computeSize);
        } else {
            Add(actResultLocal, actResultLocal, mulsResultLocal, computeSize);
        }
        AscendC::PipeBarrier<PIPE_V>();
        if (groupIdx + 1 < tiling->kGroupNum) {
            return;
        }
    }
    LocalTensor<bfloat16_t> yLocalInUb = vecOutQueue.AllocTensor<bfloat16_t>();
    // Near torch_npu.npu_quant_matmul semantics: no activation here.
    if (groupedOutput && !directSingleGroup) {
        Cast(yLocalInUb, actResultLocal, AscendC::RoundMode::CAST_RINT, computeSize);
    } else {
        Cast(yLocalInUb, mulsResultLocal, AscendC::RoundMode::CAST_RINT, computeSize);
    }
    AscendC::PipeBarrier<PIPE_V>();
    vecOutQueue.EnQue(yLocalInUb);
}

__aicore__ inline uint64_t QuantGroupMatmul::GetScaleOffset(
    uint32_t groupIdx,
    const MNConfig& mnConfig) const
{
    if (tiling->groupAxis == 2) {
        const uint64_t flatGroupIdx =
            static_cast<uint64_t>(mnConfig.expertIdx) * tiling->kGroupNum + groupIdx;
        return flatGroupIdx * tiling->n +
               mnConfig.nIdx * mnConfig.singleN;
    }
    return static_cast<uint64_t>(groupIdx) * tiling->n +
           mnConfig.nIdx * mnConfig.singleN;
}

__aicore__ inline void QuantGroupMatmul::EnqueueScale(uint32_t curBaseN, uint64_t scaleOffset)
{
    // GM拷贝scale
    AscendC::DataCopyPadExtParams<float> padParams;
    AscendC::DataCopyExtParams scaleParams{1, static_cast<uint32_t>(curBaseN * sizeof(float)), 1, 1, 0};
    LocalTensor<float> scaleLocal = scaleInQueue.AllocTensor<float>();
    DataCopyPad(scaleLocal, scaleGm[scaleOffset], scaleParams, padParams);
    scaleInQueue.EnQue(scaleLocal);
}

__aicore__ inline void QuantGroupMatmul::DequeueScale(uint32_t alignBaseN)
{
    scaleInUb = scaleInQueue.DeQue<float>();
    scaleInUb.SetSize(alignBaseN);
}

__aicore__ inline uint64_t QuantGroupMatmul::GetPerTokenScaleOffset(
        uint32_t groupIdx, MNConfig& mnConfig, uint32_t offsetM) const
{
    if (UseTileLocalDynamicInputs() && tiling->groupAxis == 1) {
        if (UseSharedTileDynamicInputs()) {
            return static_cast<uint64_t>(groupIdx) * mnConfig.singleM + offsetM;
        }
        return (static_cast<uint64_t>(coreIdx) * tiling->kGroupNum +
                static_cast<uint64_t>(groupIdx)) *
                   mnConfig.singleM +
               offsetM;
    }
    uint64_t perTokenScaleOffset = mnConfig.offsetM + mnConfig.mIdx * mnConfig.singleM + offsetM;
    // Expert K-grouped registry op (groupAxis == 2) consumes one native
    // activation scale per logical token.  Every K group reuses that value.
    // Keep the legacy group-major scale layout only for the direct-invoke
    // groupAxis == 1 experimental kernels in this translation unit.
    if (tiling->groupAxis == 1) {
        perTokenScaleOffset +=
            static_cast<uint64_t>(groupIdx) * tiling->totalInGroup;
    }
    return perTokenScaleOffset;
}

__aicore__ inline void QuantGroupMatmul::EnqueuePerTokenScale(
        uint32_t groupIdx, MNConfig& mnConfig, uint32_t curBaseM, uint32_t offsetM)
{
    uint64_t perTokenScaleOffset = GetPerTokenScaleOffset(groupIdx, mnConfig, offsetM);
    // GM拷贝per token scale
    AscendC::DataCopyPadExtParams<float> padParams;
    AscendC::DataCopyExtParams perTokenScaleParams{1, static_cast<uint32_t>(curBaseM * sizeof(float)), 0, 0, 0};
    LocalTensor<float> perTokenScaleLocal = perTokenScaleInQueue.AllocTensor<float>();
    if (UseTileLocalDynamicInputs()) {
        DataCopyPad(perTokenScaleLocal, tilePerTokenScaleGm[perTokenScaleOffset], perTokenScaleParams, padParams);
    } else {
        DataCopyPad(perTokenScaleLocal, perTokenScaleGm[perTokenScaleOffset], perTokenScaleParams, padParams);
    }
    perTokenScaleInQueue.EnQue(perTokenScaleLocal);
}

__aicore__ inline void QuantGroupMatmul::BroadcastPerTokenScale(uint32_t curBaseM, uint32_t alignBaseN)
{
    perTokenScaleInUb = perTokenScaleInQueue.DeQue<float>();
    const uint32_t broadCastDst[BROADCAST_DIM] = {curBaseM, alignBaseN};
    const uint32_t broadCastSrc[BROADCAST_DIM] = {curBaseM, 1};
    AscendC::BroadCast<float, BROADCAST_DIM, 1>(pertokenBrcbLocal, perTokenScaleInUb, broadCastDst, broadCastSrc,
                                                sharedTmpLocal);
    perTokenScaleInQueue.FreeTensor(perTokenScaleInUb);
}

__aicore__ inline void CopyTiling(QuantGroupMatmulCustomTilingData *tiling, GM_ADDR tilingGM)
{
    uint32_t *ptr = reinterpret_cast<uint32_t *>(tiling);
    auto tiling32 = reinterpret_cast<__gm__ uint32_t *>(tilingGM);

    for (uint32_t i = 0; i < sizeof(QuantGroupMatmulCustomTilingData) / sizeof(uint32_t); i++, ptr++) {
        *ptr = *(tiling32 + i);
    }
    return;
}

__aicore__ inline void CleanGmCacheRange(GM_ADDR addr, uint64_t length)
{
    __gm__ uint8_t *start = reinterpret_cast<__gm__ uint8_t *>(
        reinterpret_cast<uint64_t>(addr) / AscendC::CACHE_LINE_SIZE * AscendC::CACHE_LINE_SIZE);
    __gm__ uint8_t *end = reinterpret_cast<__gm__ uint8_t *>(
        (reinterpret_cast<uint64_t>(addr) + length + AscendC::CACHE_LINE_SIZE - 1) /
        AscendC::CACHE_LINE_SIZE * AscendC::CACHE_LINE_SIZE);
    GlobalTensor<uint8_t> cacheLineGm;
    cacheLineGm.SetGlobalBuffer(start);
    const uint64_t lineCount =
        (reinterpret_cast<uint64_t>(end) - reinterpret_cast<uint64_t>(start)) /
        AscendC::CACHE_LINE_SIZE;
    for (uint64_t i = 0; i < lineCount; ++i) {
        AscendC::DataCacheCleanAndInvalid<
            uint8_t,
            AscendC::CacheLine::SINGLE_CACHE_LINE,
            AscendC::DcciDst::CACHELINE_OUT>(cacheLineGm[i * AscendC::CACHE_LINE_SIZE]);
    }
}

extern "C" __global__ __aicore__ void quant_matmul_expert_k_grouped(GM_ADDR x, GM_ADDR weight,
        GM_ADDR groupList, GM_ADDR scale, GM_ADDR perTokenScale, GM_ADDR y, GM_ADDR workspace, GM_ADDR tilingGm)
{
    AscendC::TPipe tPipe;
    QuantGroupMatmulCustomTilingData tilingData;
    CopyTiling(&tilingData, tilingGm);
    if (TILING_KEY_IS(1)) {
        KERNEL_TASK_TYPE(1, KERNEL_TYPE_MIX_AIC_1_1);
        class LocalInt32Gmm2 {
        public:
            AscendC::TPipe *pipe = nullptr;
            LocalMM mm;
            GlobalTensor<int8_t> xGm, weightGm;
            GlobalTensor<int32_t> asyncWorkspaceGm;
            GlobalTensor<float> weightScaleGm, tokenScaleGm;
            GlobalTensor<int64_t> groupsGm;
            GlobalTensor<bfloat16_t> yGm;
            AscendC::TBuf<TPosition::VECIN> cubeBuf;
            AscendC::TBuf<TPosition::VECCALC> scaleBuf, rowScaleBuf;
            AscendC::TBuf<TPosition::VECCALC> tokenScaleBuf;
            QuantGroupMatmulCustomTilingData data;

            struct TileDesc {
                uint32_t expert;
                uint32_t rowStart;
                uint32_t tileM;
                uint32_t nOffset;
                uint32_t tileN;
                uint32_t slot;
            };

            __aicore__ inline void Init(GM_ADDR xAddr, GM_ADDR weightAddr,
                GM_ADDR groupsAddr, GM_ADDR scaleAddr, GM_ADDR tokenScaleAddr,
                GM_ADDR yAddr, GM_ADDR workspaceAddr, AscendC::TPipe *p)
            {
                pipe = p;
                xGm.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(xAddr));
                weightGm.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(weightAddr));
                asyncWorkspaceGm.SetGlobalBuffer(
                    reinterpret_cast<__gm__ int32_t *>(workspaceAddr));
                groupsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(groupsAddr));
                weightScaleGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(scaleAddr));
                tokenScaleGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(tokenScaleAddr));
                yGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(yAddr));
                if ASCEND_IS_AIV {
                    const uint32_t elems = data.mmTilingData.baseM * data.mmTilingData.baseN;
                    // Cube produces tile i into one VECIN slot while Vector
                    // consumes tile i-1 from the other slot.
                    pipe->InitBuffer(cubeBuf, 2 * elems * sizeof(int32_t));
                    pipe->InitBuffer(scaleBuf,
                                     data.mmTilingData.baseN * sizeof(float));
                    // Batch the token/channel outer product so one vector
                    // instruction covers several rows. With baseM=32 and
                    // baseN=512 the full key-1 UB budget remains below 192 KiB.
                    pipe->InitBuffer(rowScaleBuf,
                                     LOCAL_GMM2_EPILOGUE_ROWS *
                                         data.mmTilingData.baseN * sizeof(float));
                    pipe->InitBuffer(tokenScaleBuf, data.mmTilingData.baseM * sizeof(float));
                }
                mm.Init(&data.mmTilingData, pipe);
            }

            __aicore__ inline LocalTensor<int32_t> GetCubeSlot(uint32_t slot)
            {
                const uint32_t elems = data.mmTilingData.baseM * data.mmTilingData.baseN;
                LocalTensor<int32_t> dots = cubeBuf.Get<int32_t>()[slot * elems];
                dots.SetSize(elems);
                return dots;
            }

            __aicore__ inline void SubmitTile(const TileDesc &tile)
            {
                const uint32_t k = data.kGroupSize;
                const uint32_t n = data.n;
                mm.SetOrgShape(tile.tileM, tile.tileN, k, k, tile.tileN);
                mm.SetSingleShape(tile.tileM, tile.tileN, k);
                mm.SetTensorA(xGm[static_cast<uint64_t>(tile.rowStart) * k]);
                mm.SetTensorB(weightGm[
                    (static_cast<uint64_t>(tile.expert) * n + tile.nOffset) * k]);
                const uint32_t elems = data.mmTilingData.baseM * data.mmTilingData.baseN;
                const uint64_t workspaceOffset =
                    (static_cast<uint64_t>(AscendC::GetBlockIdx()) * 2 + tile.slot) * elems;
                mm.SetWorkspace(
                    asyncWorkspaceGm[workspaceOffset].GetPhyAddr(),
                    elems * sizeof(int32_t));
                // Ascend 910B supports asynchronous Iterate only through a
                // bounded GM cache. GetTensorC<false> transfers the completed
                // tile into the selected VECIN ping-pong slot.
                mm.template Iterate<false>();
            }

            __aicore__ inline void CompleteTile(const TileDesc &tile)
            {
                LocalTensor<int32_t> dots = GetCubeSlot(tile.slot);
                mm.template GetTensorC<false>(dots);
            }

            __aicore__ inline void ConsumeTile(const TileDesc &tile)
            {
                LocalTensor<int32_t> dots = GetCubeSlot(tile.slot);
                // INT32 and FP32 have equal width, so the conversion can stay
                // in the completed VECIN slot. This makes two Cube slots fit
                // in UB without separate full-tile fp32/output buffers.
                LocalTensor<float> values = dots.ReinterpretCast<float>();
                LocalTensor<float> scales = scaleBuf.Get<float>();
                LocalTensor<float> rowScales = rowScaleBuf.Get<float>();
                LocalTensor<float> tokenScales = tokenScaleBuf.Get<float>();
                const uint32_t n = data.n;
                AscendC::DataCopyPadExtParams<float> pad{true, 0, 0, 0};
                AscendC::DataCopyExtParams scaleCopy{1,
                    static_cast<uint32_t>(tile.tileN * sizeof(float)), 0, 0, 0};
                DataCopyPad(scales,
                    weightScaleGm[static_cast<uint64_t>(tile.expert) * n + tile.nOffset],
                    scaleCopy, pad);
                AscendC::DataCopyExtParams tokenScaleCopy{1,
                    static_cast<uint32_t>(tile.tileM * sizeof(float)), 0, 0, 0};
                DataCopyPad(tokenScales, tokenScaleGm[tile.rowStart], tokenScaleCopy, pad);
                const event_t mte2ToV = static_cast<event_t>(pipe->FetchEventID(AscendC::HardEvent::MTE2_V));
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(mte2ToV);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(mte2ToV);

                Cast(values, dots, AscendC::RoundMode::CAST_NONE,
                     tile.tileM * tile.tileN);
                AscendC::PipeBarrier<PIPE_V>();
                if ((tile.tileN & 7U) == 0) {
                    const uint32_t rowStrideBlocks = tile.tileN / 8U;
                    const AscendC::BinaryRepeatParams channelBroadcast{
                        1, 1, 1, static_cast<uint8_t>(rowStrideBlocks),
                        static_cast<uint8_t>(rowStrideBlocks), 0};
                    const AscendC::BinaryRepeatParams elementwiseRows{
                        1, 1, 1, static_cast<uint8_t>(rowStrideBlocks),
                        static_cast<uint8_t>(rowStrideBlocks),
                        static_cast<uint8_t>(rowStrideBlocks)};
                    for (uint32_t row = 0; row < tile.tileM;
                         row += LOCAL_GMM2_EPILOGUE_ROWS) {
                        const uint32_t batchRows =
                            row + LOCAL_GMM2_EPILOGUE_ROWS < tile.tileM
                                ? LOCAL_GMM2_EPILOGUE_ROWS
                                : tile.tileM - row;
                        const uint32_t broadcastDst[2] = {batchRows, tile.tileN};
                        const uint32_t broadcastSrc[2] = {batchRows, 1};
                        AscendC::BroadCast<float, 2, 1>(
                            rowScales, tokenScales[row], broadcastDst,
                            broadcastSrc);
                        AscendC::PipeBarrier<PIPE_V>();
                        for (uint32_t col = 0; col < tile.tileN; col += 64U) {
                            const uint64_t mask = col + 64U < tile.tileN
                                ? 64U : tile.tileN - col;
                            Mul(rowScales[col], rowScales[col], scales[col], mask,
                                static_cast<uint8_t>(batchRows),
                                channelBroadcast);
                        }
                        AscendC::PipeBarrier<PIPE_V>();
                        const uint32_t valueOffset = row * tile.tileN;
                        for (uint32_t col = 0; col < tile.tileN; col += 64U) {
                            const uint64_t mask = col + 64U < tile.tileN
                                ? 64U : tile.tileN - col;
                            Mul(values[valueOffset + col],
                                values[valueOffset + col], rowScales[col], mask,
                                static_cast<uint8_t>(batchRows), elementwiseRows);
                        }
                        AscendC::PipeBarrier<PIPE_V>();
                    }
                } else {
                    // Generic unaligned-N tail. The production Qwen shapes
                    // use N aligned to 512 and stay on the batched path.
                    const uint32_t broadcastDst[2] = {1, tile.tileN};
                    const uint32_t broadcastSrc[2] = {1, 1};
                    for (uint32_t row = 0; row < tile.tileM; ++row) {
                        AscendC::BroadCast<float, 2, 1>(
                            rowScales, tokenScales[row], broadcastDst,
                            broadcastSrc);
                        AscendC::PipeBarrier<PIPE_V>();
                        Mul(rowScales, rowScales, scales, tile.tileN);
                        AscendC::PipeBarrier<PIPE_V>();
                        Mul(values[row * tile.tileN],
                            values[row * tile.tileN], rowScales, tile.tileN);
                        AscendC::PipeBarrier<PIPE_V>();
                    }
                }
                LocalTensor<bfloat16_t> out = values.ReinterpretCast<bfloat16_t>();
                Cast(out, values, AscendC::RoundMode::CAST_RINT,
                     tile.tileM * tile.tileN);
                const event_t vToMte3 = static_cast<event_t>(pipe->FetchEventID(AscendC::HardEvent::V_MTE3));
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(vToMte3);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(vToMte3);
                AscendC::DataCopyExtParams outCopy{static_cast<uint16_t>(tile.tileM),
                    static_cast<uint32_t>(tile.tileN * sizeof(bfloat16_t)), 0,
                    static_cast<uint32_t>((n - tile.tileN) * sizeof(bfloat16_t)), 0};
                DataCopyPad(yGm[static_cast<uint64_t>(tile.rowStart) * n + tile.nOffset], out,
                            outCopy);
                const event_t mte3ToV = static_cast<event_t>(pipe->FetchEventID(AscendC::HardEvent::MTE3_V));
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(mte3ToV);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(mte3ToV);
            }

            __aicore__ inline void Process()
            {
                const uint32_t core = AscendC::GetBlockIdx();
                TileDesc pendingTile{};
                bool hasPendingTile = false;
                uint32_t tileSequence = 0;
                uint32_t totalMTiles = 0;
                for (uint32_t expert = 0; expert < data.groupNum; ++expert) {
                    const uint32_t count = static_cast<uint32_t>(groupsGm.GetValue(expert));
                    totalMTiles += (count + data.mmTilingData.baseM - 1) /
                                   data.mmTilingData.baseM;
                }
                for (uint32_t task = core; task < totalMTiles; task += data.coreNum) {
                    uint32_t remaining = task;
                    uint32_t rowStart = 0;
                    uint32_t expert = 0;
                    uint32_t count = 0;
                    for (; expert < data.groupNum; ++expert) {
                        count = static_cast<uint32_t>(groupsGm.GetValue(expert));
                        const uint32_t tiles = (count + data.mmTilingData.baseM - 1) /
                                               data.mmTilingData.baseM;
                        if (remaining < tiles) break;
                        remaining -= tiles;
                        rowStart += count;
                    }
                    if (expert == data.groupNum || count == 0) continue;
                    const uint32_t row = rowStart + remaining * data.mmTilingData.baseM;
                    const uint32_t expertEnd = rowStart + count;
                    const uint32_t tileM = row + data.mmTilingData.baseM < expertEnd
                        ? data.mmTilingData.baseM : expertEnd - row;
                    for (uint32_t col = 0; col < data.n;
                         col += data.mmTilingData.baseN) {
                        const uint32_t tileN = col + data.mmTilingData.baseN < data.n
                            ? data.mmTilingData.baseN : data.n - col;
                        TileDesc current{expert, row, tileM, col, tileN,
                                         tileSequence & 1U};
                        if (hasPendingTile) {
                            CompleteTile(pendingTile);
                        }
                        SubmitTile(current);
                        if (hasPendingTile) {
                            ConsumeTile(pendingTile);
                        }
                        pendingTile = current;
                        hasPendingTile = true;
                        ++tileSequence;
                    }
                }
                if (hasPendingTile) {
                    CompleteTile(pendingTile);
                    ConsumeTile(pendingTile);
                }
                mm.End();
            }
        } op;
        op.data = tilingData;
        REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.mm, &tilingData.mmTilingData);
        op.Init(x, weight, groupList, scale, perTokenScale, y,
                AscendC::GetUserWorkspace(workspace), &tPipe);
        op.Process();
        return;
    }
    if (TILING_KEY_IS(0)) {
        KERNEL_TASK_TYPE(0, KERNEL_TYPE_MIX_AIC_1_2);
        MT mm;
        if ASCEND_IS_AIC {
            mm.SetSubBlockIdx(0);
            mm.Init(&tilingData.mmTilingData, &tPipe);
        }
        QuantGroupMatmul op(mm);
        op.Init(x, weight, nullptr, groupList, scale, perTokenScale, y,
                AscendC::GetUserWorkspace(workspace), &tilingData, &tPipe);
        op.Process();
    }
}

__aicore__ inline int8_t QuantizeBf16Value(float value, float scale)
{
    if (scale == 0.0f) {
        return static_cast<int8_t>(0);
    }
    float scaled = value / scale;
    int32_t rounded = static_cast<int32_t>(scaled >= 0.0f ? scaled + 0.5f : scaled - 0.5f);
    if (rounded > 127) {
        rounded = 127;
    } else if (rounded < -127) {
        rounded = -127;
    }
    return static_cast<int8_t>(rounded);
}

__aicore__ inline void DynamicQuantizeKGroupedToWorkspace(
        GM_ADDR x,
        GM_ADDR xQWorkspace,
        GM_ADDR perTokenScaleWorkspace,
        const QuantGroupMatmulCustomTilingData *tiling)
{
    if ASCEND_IS_AIC {
        return;
    }

    AscendC::TPipe localPipe;
    const uint32_t alignedK = AlignUp64(tiling->kGroupSize, 8);
    const uint32_t qRowBytes = AlignUp64(tiling->kGroupSize, 32);
    const uint32_t floatBytes = AlignUp64(alignedK * sizeof(float), 32);

    AscendC::TBuf<TPosition::VECCALC> qRowBuffer;
    AscendC::TBuf<TPosition::VECCALC> bf16StageBuffer;
    AscendC::TBuf<TPosition::VECCALC> floatWorkBuffer;
    AscendC::TBuf<TPosition::VECCALC> absWorkBuffer;
    localPipe.InitBuffer(qRowBuffer, qRowBytes);
    localPipe.InitBuffer(bf16StageBuffer, floatBytes);
    localPipe.InitBuffer(floatWorkBuffer, floatBytes);
    localPipe.InitBuffer(absWorkBuffer, floatBytes);
    LocalTensor<int8_t> qRowLocal = qRowBuffer.Get<int8_t>();
    LocalTensor<bfloat16_t> bf16StageLocal = bf16StageBuffer.Get<bfloat16_t>();
    LocalTensor<float> floatWorkLocal = floatWorkBuffer.Get<float>();
    LocalTensor<float> absWorkLocal = absWorkBuffer.Get<float>();

    GlobalTensor<bfloat16_t> xGm;
    GlobalTensor<int8_t> xQGm;
    GlobalTensor<float> perTokenScaleGm;
    xGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(x));
    xQGm.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(xQWorkspace));
    perTokenScaleGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(perTokenScaleWorkspace));

    uint32_t coreIdx = AscendC::GetBlockIdx();
    if ASCEND_IS_AIV {
        coreIdx /= AscendC::GetTaskRation();
    }
    const uint32_t coreNum = tiling->coreNum;
    const uint32_t totalRows = tiling->totalInGroup * tiling->kGroupNum;
    const uint32_t logicalK = tiling->kGroupSize * tiling->kGroupNum;

    AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(tiling->kGroupSize * sizeof(bfloat16_t)), 0, 0, 0};
    AscendC::DataCopyPadExtParams<bfloat16_t> padParams{true, 0, 0, 0};

    for (uint32_t packedRow = coreIdx; packedRow < totalRows; packedRow += coreNum) {
        const uint32_t groupIdx = packedRow / tiling->totalInGroup;
        const uint32_t tokenIdx = packedRow - groupIdx * tiling->totalInGroup;
        const uint64_t inputOffset =
            static_cast<uint64_t>(tokenIdx) * logicalK +
            static_cast<uint64_t>(groupIdx) * tiling->kGroupSize;
        const uint64_t outputOffset =
            static_cast<uint64_t>(packedRow) * tiling->kGroupSize;

        DataCopyPad(bf16StageLocal, xGm[inputOffset], copyParams, padParams);
        AscendC::PipeBarrier<PIPE_MTE2>();
        Cast(floatWorkLocal, bf16StageLocal, AscendC::RoundMode::CAST_NONE, alignedK);
        AscendC::PipeBarrier<PIPE_V>();

        Abs(absWorkLocal, floatWorkLocal, alignedK);
        AscendC::PipeBarrier<PIPE_V>();
        LocalTensor<float> reduceTmp = bf16StageLocal.ReinterpretCast<float>();
        ReduceMax(absWorkLocal, absWorkLocal, reduceTmp, static_cast<int32_t>(alignedK), false);
        AscendC::PipeBarrier<PIPE_V>();
        float maxAbs = absWorkLocal.GetValue(0);

        const float tokenScale = maxAbs / 127.0f;
        perTokenScaleGm.SetValue(packedRow, tokenScale);

        if (tokenScale > 0.0f) {
            Muls(floatWorkLocal, floatWorkLocal, 1.0f / tokenScale, alignedK);
            AscendC::PipeBarrier<PIPE_V>();
        }
        Maxs(floatWorkLocal, floatWorkLocal, -127.0f, alignedK);
        AscendC::PipeBarrier<PIPE_V>();
        Mins(floatWorkLocal, floatWorkLocal, 127.0f, alignedK);
        AscendC::PipeBarrier<PIPE_V>();
        LocalTensor<int16_t> i16Tmp = floatWorkLocal.ReinterpretCast<int16_t>();
        Cast(i16Tmp, floatWorkLocal, AscendC::RoundMode::CAST_RINT, alignedK);
        AscendC::PipeBarrier<PIPE_V>();
        Cast(qRowLocal, i16Tmp, AscendC::RoundMode::CAST_RINT, alignedK);
        AscendC::PipeBarrier<PIPE_V>();

        DataCopy(xQGm[outputOffset], qRowLocal, tiling->kGroupSize);
        AscendC::PipeBarrier<PIPE_MTE3>();
    }
}

__aicore__ inline void DynamicQuantizeKGroupedToWorkspaceAicSerial(
        GM_ADDR x,
        GM_ADDR xQWorkspace,
        GM_ADDR perTokenScaleWorkspace,
        const QuantGroupMatmulCustomTilingData *tiling)
{
    if ASCEND_IS_AIV {
        return;
    }

    GlobalTensor<bfloat16_t> xGm;
    GlobalTensor<int8_t> xQGm;
    GlobalTensor<float> perTokenScaleGm;
    xGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(x));
    xQGm.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(xQWorkspace));
    perTokenScaleGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(perTokenScaleWorkspace));

    const uint32_t totalRows = tiling->totalInGroup * tiling->kGroupNum;
    const uint32_t logicalK = tiling->kGroupSize * tiling->kGroupNum;

    for (uint32_t packedRow = 0; packedRow < totalRows; ++packedRow) {
        const uint32_t groupIdx = packedRow / tiling->totalInGroup;
        const uint32_t tokenIdx = packedRow - groupIdx * tiling->totalInGroup;
        const uint64_t inputOffset =
            static_cast<uint64_t>(tokenIdx) * logicalK +
            static_cast<uint64_t>(groupIdx) * tiling->kGroupSize;
        const uint64_t outputOffset =
            static_cast<uint64_t>(packedRow) * tiling->kGroupSize;

        float maxAbs = 0.0f;
        for (uint32_t i = 0; i < tiling->kGroupSize; ++i) {
            float value = AscendC::ToFloat(xGm.GetValue(inputOffset + i));
            float absValue = value >= 0.0f ? value : -value;
            if (absValue > maxAbs) {
                maxAbs = absValue;
            }
        }

        const float tokenScale = maxAbs / 127.0f;
        perTokenScaleGm.SetValue(packedRow, tokenScale);
        for (uint32_t i = 0; i < tiling->kGroupSize; ++i) {
            float value = AscendC::ToFloat(xGm.GetValue(inputOffset + i));
            xQGm.SetValue(outputOffset + i, QuantizeBf16Value(value, tokenScale));
        }
    }
}

__aicore__ inline uint32_t PackInt8ToU32(int8_t v0, int8_t v1, int8_t v2, int8_t v3)
{
    return (static_cast<uint32_t>(static_cast<uint8_t>(v0))) |
           (static_cast<uint32_t>(static_cast<uint8_t>(v1)) << 8) |
           (static_cast<uint32_t>(static_cast<uint8_t>(v2)) << 16) |
           (static_cast<uint32_t>(static_cast<uint8_t>(v3)) << 24);
}

__aicore__ inline void DynamicQuantizeKGroupedToWorkspacePacked32(
        GM_ADDR x,
        GM_ADDR xQWorkspace,
        GM_ADDR perTokenScaleWorkspace,
        const QuantGroupMatmulCustomTilingData *tiling)
{
    if ASCEND_IS_AIC {
        return;
    }

    GlobalTensor<bfloat16_t> xGm;
    GlobalTensor<int32_t> xQWordGm;
    GlobalTensor<int8_t> xQByteGm;
    GlobalTensor<float> perTokenScaleGm;
    xGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(x));
    xQWordGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(xQWorkspace));
    xQByteGm.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(xQWorkspace));
    perTokenScaleGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(perTokenScaleWorkspace));

    uint32_t coreIdx = AscendC::GetBlockIdx();
    if ASCEND_IS_AIV {
        coreIdx /= AscendC::GetTaskRation();
    }
    const uint32_t coreNum = tiling->coreNum;
    const uint32_t totalRows = tiling->totalInGroup * tiling->kGroupNum;
    const uint32_t logicalK = tiling->kGroupSize * tiling->kGroupNum;
    const uint32_t rowsPerCore = AscendC::Ceil(totalRows, coreNum);
    const uint32_t rowStart = coreIdx * rowsPerCore;
    const uint32_t rowEnd =
        rowStart + rowsPerCore < totalRows ? rowStart + rowsPerCore : totalRows;

#ifdef GROUP_QUANT_TILE_DYNAMIC_EXPERIMENT
    const bool useTileDynamicBucket = tiling->kGroupSize == 32;
    if (useTileDynamicBucket) {
        // Experimental insertion point: replace the baseline full-xQ
        // materialization below with tile-level dynamic quant that feeds matmul
        // directly. For now, intentionally fall through to the verified
        // baseline pack so the new kernel entry remains bitwise comparable.
    }
#endif

    constexpr uint32_t SCALE_BATCH = 8;
    uint32_t scaleBatchStart = rowStart;
    uint32_t scaleBatchCount = 0;

    for (uint32_t packedRow = rowStart; packedRow < rowEnd; ++packedRow) {
        const uint32_t groupIdx = packedRow / tiling->totalInGroup;
        const uint32_t tokenIdx = packedRow - groupIdx * tiling->totalInGroup;
        const uint64_t inputOffset =
            static_cast<uint64_t>(tokenIdx) * logicalK +
            static_cast<uint64_t>(groupIdx) * tiling->kGroupSize;
        const uint64_t outputOffset =
            static_cast<uint64_t>(packedRow) * tiling->kGroupSize;

        float maxAbs = 0.0f;
        for (uint32_t i = 0; i < tiling->kGroupSize; ++i) {
            float value = AscendC::ToFloat(xGm.GetValue(inputOffset + i));
            float absValue = value >= 0.0f ? value : -value;
            if (absValue > maxAbs) {
                maxAbs = absValue;
            }
        }

        const float tokenScale = maxAbs / 127.0f;
        perTokenScaleGm.SetValue(packedRow, tokenScale);
        uint32_t i = 0;
        for (; i + 3 < tiling->kGroupSize; i += 4) {
            const int8_t q0 = QuantizeBf16Value(AscendC::ToFloat(xGm.GetValue(inputOffset + i)), tokenScale);
            const int8_t q1 = QuantizeBf16Value(AscendC::ToFloat(xGm.GetValue(inputOffset + i + 1)), tokenScale);
            const int8_t q2 = QuantizeBf16Value(AscendC::ToFloat(xGm.GetValue(inputOffset + i + 2)), tokenScale);
            const int8_t q3 = QuantizeBf16Value(AscendC::ToFloat(xGm.GetValue(inputOffset + i + 3)), tokenScale);
            xQWordGm.SetValue((outputOffset + i) / 4, static_cast<int32_t>(PackInt8ToU32(q0, q1, q2, q3)));
        }
        for (; i < tiling->kGroupSize; ++i) {
            float value = AscendC::ToFloat(xGm.GetValue(inputOffset + i));
            xQByteGm.SetValue(outputOffset + i, QuantizeBf16Value(value, tokenScale));
        }
    }
}

__aicore__ inline void GROUP_QUANT_DYNAMIC_PACK_FN(
        GM_ADDR x,
        GM_ADDR xQWorkspace,
        GM_ADDR perTokenScaleWorkspace,
        const QuantGroupMatmulCustomTilingData *tiling,
        AscendC::TPipe *pipe)
{
    if ASCEND_IS_AIC {
        return;
    }

    const uint32_t alignedK = AlignUp64(tiling->kGroupSize, 8);
    const uint32_t qRowBytes = AlignUp64(tiling->kGroupSize, 32);
    const uint32_t floatBytes = AlignUp64(alignedK * sizeof(float), 32);
    constexpr uint32_t SCALE_BATCH = 32;

    AscendC::TBuf<TPosition::VECCALC> qRowBuffer;
    AscendC::TBuf<TPosition::VECCALC> scaleBuffer;
    AscendC::TBuf<TPosition::VECCALC> bf16StageBuffer;
    AscendC::TBuf<TPosition::VECCALC> floatWorkBuffer;
    AscendC::TBuf<TPosition::VECCALC> absWorkBuffer;
    pipe->InitBuffer(qRowBuffer, qRowBytes);
    pipe->InitBuffer(scaleBuffer, SCALE_BATCH * sizeof(float));
    pipe->InitBuffer(bf16StageBuffer, floatBytes);
    pipe->InitBuffer(floatWorkBuffer, floatBytes);
    pipe->InitBuffer(absWorkBuffer, floatBytes);
    LocalTensor<int8_t> qRowLocal = qRowBuffer.Get<int8_t>();
    LocalTensor<float> scaleLocal = scaleBuffer.Get<float>();
    LocalTensor<bfloat16_t> bf16StageLocal = bf16StageBuffer.Get<bfloat16_t>();
    LocalTensor<float> floatWorkLocal = floatWorkBuffer.Get<float>();
    LocalTensor<float> absWorkLocal = absWorkBuffer.Get<float>();

    GlobalTensor<bfloat16_t> xGm;
    GlobalTensor<int8_t> xQGm;
    GlobalTensor<float> perTokenScaleGm;
    xGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(x));
    xQGm.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(xQWorkspace));
    perTokenScaleGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(perTokenScaleWorkspace));

    uint32_t coreIdx = AscendC::GetBlockIdx();
    if ASCEND_IS_AIV {
        coreIdx /= AscendC::GetTaskRation();
    }
    const uint32_t coreNum = tiling->coreNum;
    const uint32_t totalRows = tiling->totalInGroup * tiling->kGroupNum;
    const uint32_t logicalK = tiling->kGroupSize * tiling->kGroupNum;
    const uint32_t rowsPerCore = AscendC::Ceil(totalRows, coreNum);
    const uint32_t rowStart = coreIdx * rowsPerCore;
    const uint32_t rowEnd =
        rowStart + rowsPerCore < totalRows ? rowStart + rowsPerCore : totalRows;
    uint32_t scaleBatchStart = rowStart;
    uint32_t scaleBatchCount = 0;

    AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(tiling->kGroupSize * sizeof(bfloat16_t)), 0, 0, 0};
    AscendC::DataCopyPadExtParams<bfloat16_t> padParams{true, 0, 0, 0};

    for (uint32_t packedRow = rowStart; packedRow < rowEnd; ++packedRow) {
        const uint32_t groupIdx = packedRow / tiling->totalInGroup;
        const uint32_t tokenIdx = packedRow - groupIdx * tiling->totalInGroup;
        const uint64_t inputOffset =
            static_cast<uint64_t>(tokenIdx) * logicalK +
            static_cast<uint64_t>(groupIdx) * tiling->kGroupSize;
        const uint64_t outputOffset =
            static_cast<uint64_t>(packedRow) * tiling->kGroupSize;

        DataCopyPad(bf16StageLocal, xGm[inputOffset], copyParams, padParams);
        AscendC::PipeBarrier<PIPE_MTE2>();
        Cast(floatWorkLocal, bf16StageLocal, AscendC::RoundMode::CAST_NONE, alignedK);
        AscendC::PipeBarrier<PIPE_V>();

        Abs(absWorkLocal, floatWorkLocal, alignedK);
        AscendC::PipeBarrier<PIPE_V>();
        LocalTensor<float> reduceTmp = bf16StageLocal.ReinterpretCast<float>();
        ReduceMax(absWorkLocal, absWorkLocal, reduceTmp, static_cast<int32_t>(alignedK), false);
        AscendC::PipeBarrier<PIPE_V>();
        float maxAbs = absWorkLocal.GetValue(0);

        const float tokenScale = maxAbs / 127.0f;
        if (scaleBatchCount == 0) {
            scaleBatchStart = packedRow;
        }
        scaleLocal.SetValue(scaleBatchCount, tokenScale);
        ++scaleBatchCount;

        if (tokenScale > 0.0f) {
            Muls(floatWorkLocal, floatWorkLocal, 1.0f / tokenScale, alignedK);
            AscendC::PipeBarrier<PIPE_V>();
        }
        Maxs(floatWorkLocal, floatWorkLocal, -127.0f, alignedK);
        Mins(floatWorkLocal, floatWorkLocal, 127.0f, alignedK);
        AscendC::PipeBarrier<PIPE_V>();
        LocalTensor<int16_t> i16Tmp = floatWorkLocal.ReinterpretCast<int16_t>();
        Cast(i16Tmp, floatWorkLocal, AscendC::RoundMode::CAST_RINT, alignedK);
        AscendC::PipeBarrier<PIPE_V>();
        Cast(qRowLocal, i16Tmp, AscendC::RoundMode::CAST_RINT, alignedK);
        AscendC::PipeBarrier<PIPE_V>();

        DataCopy(xQGm[outputOffset], qRowLocal, tiling->kGroupSize);
        if (scaleBatchCount == SCALE_BATCH) {
            DataCopy(perTokenScaleGm[scaleBatchStart], scaleLocal, SCALE_BATCH);
            scaleBatchCount = 0;
        }
        AscendC::PipeBarrier<PIPE_MTE3>();
        CleanGmCacheRange(xQWorkspace + outputOffset, tiling->kGroupSize);
    }
    if (scaleBatchCount > 0) {
        AscendC::PipeBarrier<PIPE_V>();
        DataCopyPad2D(perTokenScaleGm[scaleBatchStart], scaleLocal, 1, scaleBatchCount, scaleBatchCount, scaleBatchCount);
        AscendC::PipeBarrier<PIPE_MTE3>();
    }
}

__aicore__ inline void CopyQuantInputsToWorkspaceAicSerial(
        GM_ADDR xQ,
        GM_ADDR perTokenScale,
        GM_ADDR xQWorkspace,
        GM_ADDR perTokenScaleWorkspace,
        const QuantGroupMatmulCustomTilingData *tiling)
{
    if ASCEND_IS_AIV {
        return;
    }

    GlobalTensor<int8_t> xQSrcGm;
    GlobalTensor<int8_t> xQDstGm;
    GlobalTensor<float> perTokenScaleSrcGm;
    GlobalTensor<float> perTokenScaleDstGm;
    xQSrcGm.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(xQ));
    xQDstGm.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(xQWorkspace));
    perTokenScaleSrcGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(perTokenScale));
    perTokenScaleDstGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(perTokenScaleWorkspace));

    const uint32_t totalXQElems =
        tiling->totalInGroup * tiling->kGroupNum * tiling->kGroupSize;
    const uint32_t totalScaleElems = tiling->totalInGroup * tiling->kGroupNum;
    for (uint32_t idx = 0; idx < totalXQElems; ++idx) {
        xQDstGm.SetValue(idx, xQSrcGm.GetValue(idx));
    }
    for (uint32_t idx = 0; idx < totalScaleElems; ++idx) {
        perTokenScaleDstGm.SetValue(idx, perTokenScaleSrcGm.GetValue(idx));
    }
}

extern "C" __global__ __aicore__ void quant_group_matmul_dynamic_custom(GM_ADDR x, GM_ADDR weight, GM_ADDR bias,
        GM_ADDR groupList, GM_ADDR scale, GM_ADDR y, GM_ADDR workspace, GM_ADDR tilingGm)
{
    QuantGroupMatmulCustomTilingData tilingData;
    CopyTiling(&tilingData, tilingGm);

    const uint64_t xQBytes =
        static_cast<uint64_t>(tilingData.totalInGroup) *
        static_cast<uint64_t>(tilingData.kGroupNum) *
        static_cast<uint64_t>(tilingData.kGroupSize);
    const uint64_t perTokenScaleBytes =
        static_cast<uint64_t>(tilingData.totalInGroup) *
        static_cast<uint64_t>(tilingData.kGroupNum) *
        sizeof(float);
    const uint64_t xQAlignedBytes = AlignUp64(xQBytes, 32);
    const uint64_t perTokenScaleAlignedBytes = AlignUp64(perTokenScaleBytes, 32);
    const uint64_t xQWorkspaceOffset = AlignUp64(DYNAMIC_WORKSPACE_GUARD_BYTES, 512);
    const uint64_t perTokenScaleWorkspaceOffset = AlignUp64(xQWorkspaceOffset + xQAlignedBytes, 512);
    const uint64_t matmulWorkspaceOffset =
        AlignUp64(perTokenScaleWorkspaceOffset + perTokenScaleAlignedBytes, 512);
    GM_ADDR xQWorkspace = workspace + xQWorkspaceOffset;
    GM_ADDR perTokenScaleWorkspace = workspace + perTokenScaleWorkspaceOffset;
    GM_ADDR matmulWorkspace = workspace + matmulWorkspaceOffset;

    AscendC::TPipe tPipe;
    GROUP_QUANT_DYNAMIC_PACK_FN(x, xQWorkspace, perTokenScaleWorkspace, &tilingData, &tPipe);
    AscendC::PipeBarrier<PIPE_ALL>();
    if ASCEND_IS_AIV {
        AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(SYNC_DYNAMIC_QUANT_DONE);
    }
    if ASCEND_IS_AIC {
        AscendC::CrossCoreWaitFlag(SYNC_DYNAMIC_QUANT_DONE);
    }

    MT mm;
    if ASCEND_IS_AIC {
        mm.SetSubBlockIdx(0);
        mm.Init(&tilingData.mmTilingData, &tPipe);
    }
    QuantGroupMatmul op(mm);
    op.Init(
        xQWorkspace,
        weight,
        bias,
        groupList,
        scale,
        perTokenScaleWorkspace,
        y,
        matmulWorkspace,
        &tilingData,
        &tPipe);
    op.Process();
}

extern "C" __global__ __aicore__ void quant_group_matmul_workspace_passthrough_dynamic_custom(
        GM_ADDR xQ, GM_ADDR weight, GM_ADDR bias, GM_ADDR groupList, GM_ADDR scale,
        GM_ADDR perTokenScale, GM_ADDR y, GM_ADDR workspace, GM_ADDR tilingGm)
{
    QuantGroupMatmulCustomTilingData tilingData;
    CopyTiling(&tilingData, tilingGm);

    const uint64_t xQBytes =
        static_cast<uint64_t>(tilingData.totalInGroup) *
        static_cast<uint64_t>(tilingData.kGroupNum) *
        static_cast<uint64_t>(tilingData.kGroupSize);
    const uint64_t perTokenScaleBytes =
        static_cast<uint64_t>(tilingData.totalInGroup) *
        static_cast<uint64_t>(tilingData.kGroupNum) *
        sizeof(float);
    const uint64_t xQAlignedBytes = AlignUp64(xQBytes, 32);
    const uint64_t perTokenScaleAlignedBytes = AlignUp64(perTokenScaleBytes, 32);
    const uint64_t xQWorkspaceOffset = AlignUp64(DYNAMIC_WORKSPACE_GUARD_BYTES, 512);
    const uint64_t perTokenScaleWorkspaceOffset = AlignUp64(xQWorkspaceOffset + xQAlignedBytes, 512);
    const uint64_t matmulWorkspaceOffset =
        AlignUp64(perTokenScaleWorkspaceOffset + perTokenScaleAlignedBytes, 512);
    GM_ADDR xQWorkspace = workspace + xQWorkspaceOffset;
    GM_ADDR perTokenScaleWorkspace = workspace + perTokenScaleWorkspaceOffset;
    GM_ADDR matmulWorkspace = workspace + matmulWorkspaceOffset;

    CopyQuantInputsToWorkspaceAicSerial(
        xQ, perTokenScale, xQWorkspace, perTokenScaleWorkspace, &tilingData);
    AscendC::PipeBarrier<PIPE_ALL>();
    if ASCEND_IS_AIC {
        AscendC::CrossCoreSetFlag<2, PIPE_FIX>(SYNC_DYNAMIC_QUANT_DONE);
    }
    if ASCEND_IS_AIV {
        AscendC::CrossCoreWaitFlag(SYNC_DYNAMIC_QUANT_DONE);
    }

    MT mm;
    AscendC::TPipe tPipe;
    if ASCEND_IS_AIC {
        mm.SetSubBlockIdx(0);
        mm.Init(&tilingData.mmTilingData, &tPipe);
    }
    QuantGroupMatmul op(mm);
    op.Init(
        xQWorkspace,
        weight,
        bias,
        groupList,
        scale,
        perTokenScaleWorkspace,
        y,
        matmulWorkspace,
        &tilingData,
        &tPipe);
    op.Process();
}

extern "C" __global__ __aicore__ void quant_group_matmul_passthrough_dynamic_custom(
        GM_ADDR xQ, GM_ADDR weight, GM_ADDR bias, GM_ADDR groupList, GM_ADDR scale,
        GM_ADDR perTokenScale, GM_ADDR y, GM_ADDR workspace, GM_ADDR tilingGm)
{
    QuantGroupMatmulCustomTilingData tilingData;
    CopyTiling(&tilingData, tilingGm);

    const uint64_t xQBytes =
        static_cast<uint64_t>(tilingData.totalInGroup) *
        static_cast<uint64_t>(tilingData.kGroupNum) *
        static_cast<uint64_t>(tilingData.kGroupSize);
    const uint64_t perTokenScaleBytes =
        static_cast<uint64_t>(tilingData.totalInGroup) *
        static_cast<uint64_t>(tilingData.kGroupNum) *
        sizeof(float);
    const uint64_t xQAlignedBytes = AlignUp64(xQBytes, 32);
    const uint64_t perTokenScaleAlignedBytes = AlignUp64(perTokenScaleBytes, 32);
    const uint64_t xQWorkspaceOffset = AlignUp64(DYNAMIC_WORKSPACE_GUARD_BYTES, 512);
    const uint64_t perTokenScaleWorkspaceOffset = AlignUp64(xQWorkspaceOffset + xQAlignedBytes, 512);
    const uint64_t matmulWorkspaceOffset =
        AlignUp64(perTokenScaleWorkspaceOffset + perTokenScaleAlignedBytes, 512);
    GM_ADDR matmulWorkspace = workspace + matmulWorkspaceOffset;

    MT mm;
    AscendC::TPipe tPipe;
    if ASCEND_IS_AIC {
        mm.SetSubBlockIdx(0);
        mm.Init(&tilingData.mmTilingData, &tPipe);
    }
    QuantGroupMatmul op(mm);
    op.Init(
        xQ,
        weight,
        bias,
        groupList,
        scale,
        perTokenScale,
        y,
        matmulWorkspace,
        &tilingData,
        &tPipe);
    op.Process();
}

extern "C" __global__ __aicore__ void quant_group_matmul_external_dynamic_custom(
        GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR groupList, GM_ADDR scale,
        GM_ADDR xQ, GM_ADDR perTokenScale, GM_ADDR y, GM_ADDR workspace, GM_ADDR tilingGm)
{
    QuantGroupMatmulCustomTilingData tilingData;
    CopyTiling(&tilingData, tilingGm);

    AscendC::TPipe tPipe;
#if defined(GROUP_QUANT_TILE_DYNAMIC_EXPERIMENT) && defined(GROUP_QUANT_TILE_DYNAMIC_LOCAL_INPUTS)
    (void)xQ;
    (void)perTokenScale;
#else
    GROUP_QUANT_DYNAMIC_PACK_FN(x, xQ, perTokenScale, &tilingData, &tPipe);
    AscendC::PipeBarrier<PIPE_ALL>();
    if ASCEND_IS_AIV {
        AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(SYNC_DYNAMIC_QUANT_DONE);
    }
    if ASCEND_IS_AIC {
        AscendC::CrossCoreWaitFlag(SYNC_DYNAMIC_QUANT_DONE);
    }
#endif

    MT mm;
    if ASCEND_IS_AIC {
        mm.SetSubBlockIdx(0);
        mm.Init(&tilingData.mmTilingData, &tPipe);
    }
    QuantGroupMatmul op(mm);
    op.Init(
#if defined(GROUP_QUANT_TILE_DYNAMIC_EXPERIMENT) && defined(GROUP_QUANT_TILE_DYNAMIC_LOCAL_INPUTS)
        x,
#else
        xQ,
#endif
        weight,
        bias,
        groupList,
        scale,
        perTokenScale,
        y,
        workspace,
        &tilingData,
        &tPipe);
    op.Process();
}

extern "C" __global__ __aicore__ void quant_group_dynamic_pack_custom(
        GM_ADDR x,
        GM_ADDR xQ,
        GM_ADDR perTokenScale,
        GM_ADDR workspace,
        GM_ADDR tilingGm)
{
    QuantGroupMatmulCustomTilingData tilingData;
    CopyTiling(&tilingData, tilingGm);
    AscendC::TPipe tPipe;
    GROUP_QUANT_DYNAMIC_PACK_FN(x, xQ, perTokenScale, &tilingData, &tPipe);
}

extern "C" __global__ __aicore__ void quant_group_dynamic_pack_aic_read_custom(
        GM_ADDR x,
        GM_ADDR xQ,
        GM_ADDR perTokenScale,
        GM_ADDR workspace,
        GM_ADDR tilingGm)
{
    QuantGroupMatmulCustomTilingData tilingData;
    CopyTiling(&tilingData, tilingGm);

    const uint64_t xQBytes =
        static_cast<uint64_t>(tilingData.totalInGroup) *
        static_cast<uint64_t>(tilingData.kGroupNum) *
        static_cast<uint64_t>(tilingData.kGroupSize);
    const uint64_t perTokenScaleBytes =
        static_cast<uint64_t>(tilingData.totalInGroup) *
        static_cast<uint64_t>(tilingData.kGroupNum) *
        sizeof(float);
    const uint64_t xQAlignedBytes = AlignUp64(xQBytes, 32);
    const uint64_t xQWorkspaceOffset = AlignUp64(DYNAMIC_WORKSPACE_GUARD_BYTES, 512);
    const uint64_t perTokenScaleWorkspaceOffset = AlignUp64(xQWorkspaceOffset + xQAlignedBytes, 512);
    GM_ADDR xQWorkspace = workspace + xQWorkspaceOffset;
    GM_ADDR perTokenScaleWorkspace = workspace + perTokenScaleWorkspaceOffset;

    AscendC::TPipe tPipe;
    GROUP_QUANT_DYNAMIC_PACK_FN(x, xQWorkspace, perTokenScaleWorkspace, &tilingData, &tPipe);
    AscendC::PipeBarrier<PIPE_ALL>();
    if ASCEND_IS_AIV {
        AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(SYNC_DYNAMIC_QUANT_DONE);
    }
    if ASCEND_IS_AIC {
        AscendC::CrossCoreWaitFlag(SYNC_DYNAMIC_QUANT_DONE);

        GlobalTensor<int8_t> xQWorkspaceGm;
        GlobalTensor<int8_t> xQOutGm;
        GlobalTensor<float> perTokenScaleWorkspaceGm;
        GlobalTensor<float> perTokenScaleOutGm;
        xQWorkspaceGm.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(xQWorkspace));
        xQOutGm.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(xQ));
        perTokenScaleWorkspaceGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(perTokenScaleWorkspace));
        perTokenScaleOutGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(perTokenScale));

        const uint32_t coreIdx = AscendC::GetBlockIdx();
        const uint32_t coreNum = tilingData.coreNum;
        const uint32_t totalXQElems =
            tilingData.totalInGroup * tilingData.kGroupNum * tilingData.kGroupSize;
        const uint32_t totalScaleElems = tilingData.totalInGroup * tilingData.kGroupNum;
        for (uint32_t idx = coreIdx; idx < totalXQElems; idx += coreNum) {
            xQOutGm.SetValue(idx, xQWorkspaceGm.GetValue(idx));
        }
        for (uint32_t idx = coreIdx; idx < totalScaleElems; idx += coreNum) {
            perTokenScaleOutGm.SetValue(idx, perTokenScaleWorkspaceGm.GetValue(idx));
        }
    }
}

extern "C" __global__ __aicore__ void quant_group_tile_workspace_roundtrip_custom(
        GM_ADDR xQ,
        GM_ADDR perTokenScale,
        GM_ADDR xQOut,
        GM_ADDR perTokenScaleOut,
        GM_ADDR workspace,
        GM_ADDR tilingGm)
{
    if ASCEND_IS_AIV {
        return;
    }

    QuantGroupMatmulCustomTilingData tilingData;
    CopyTiling(&tilingData, tilingGm);

    GlobalTensor<int8_t> xQSrcGm;
    GlobalTensor<float> perTokenScaleSrcGm;
    GlobalTensor<int8_t> xQTileGm;
    GlobalTensor<float> perTokenScaleTileGm;
    GlobalTensor<int8_t> xQOutGm;
    GlobalTensor<float> perTokenScaleOutGm;

    const uint64_t tileXBytes =
        static_cast<uint64_t>(tilingData.kGroupNum) *
        static_cast<uint64_t>(tilingData.mmTilingData.baseM) *
        static_cast<uint64_t>(tilingData.kGroupSize);
    const uint64_t tileScaleOffset = AlignUp64(tileXBytes, 512);

    xQSrcGm.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(xQ));
    perTokenScaleSrcGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(perTokenScale));
    xQTileGm.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(workspace));
    perTokenScaleTileGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace + tileScaleOffset));
    xQOutGm.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(xQOut));
    perTokenScaleOutGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(perTokenScaleOut));

    const uint32_t baseM = tilingData.mmTilingData.baseM;
    for (uint32_t tileMStart = 0; tileMStart < tilingData.totalInGroup; tileMStart += baseM) {
        uint32_t curM = baseM;
        if (tileMStart + curM > tilingData.totalInGroup) {
            curM = tilingData.totalInGroup - tileMStart;
        }

        for (uint32_t groupIdx = 0; groupIdx < tilingData.kGroupNum; ++groupIdx) {
            for (uint32_t localM = 0; localM < curM; ++localM) {
                const uint64_t fullM = static_cast<uint64_t>(tileMStart) + localM;
                const uint64_t fullXOffset =
                    static_cast<uint64_t>(groupIdx) * tilingData.totalInGroup * tilingData.kGroupSize +
                    fullM * tilingData.kGroupSize;
                const uint64_t tileXOffset =
                    static_cast<uint64_t>(groupIdx) * baseM * tilingData.kGroupSize +
                    static_cast<uint64_t>(localM) * tilingData.kGroupSize;
                for (uint32_t kIdx = 0; kIdx < tilingData.kGroupSize; ++kIdx) {
                    xQTileGm.SetValue(tileXOffset + kIdx, xQSrcGm.GetValue(fullXOffset + kIdx));
                }

                const uint64_t fullScaleOffset =
                    static_cast<uint64_t>(groupIdx) * tilingData.totalInGroup + fullM;
                const uint64_t tileScaleIdx =
                    static_cast<uint64_t>(groupIdx) * baseM + localM;
                perTokenScaleTileGm.SetValue(tileScaleIdx, perTokenScaleSrcGm.GetValue(fullScaleOffset));
            }
        }

        AscendC::PipeBarrier<PIPE_ALL>();

        for (uint32_t groupIdx = 0; groupIdx < tilingData.kGroupNum; ++groupIdx) {
            for (uint32_t localM = 0; localM < curM; ++localM) {
                const uint64_t fullM = static_cast<uint64_t>(tileMStart) + localM;
                const uint64_t fullXOffset =
                    static_cast<uint64_t>(groupIdx) * tilingData.totalInGroup * tilingData.kGroupSize +
                    fullM * tilingData.kGroupSize;
                const uint64_t tileXOffset =
                    static_cast<uint64_t>(groupIdx) * baseM * tilingData.kGroupSize +
                    static_cast<uint64_t>(localM) * tilingData.kGroupSize;
                for (uint32_t kIdx = 0; kIdx < tilingData.kGroupSize; ++kIdx) {
                    xQOutGm.SetValue(fullXOffset + kIdx, xQTileGm.GetValue(tileXOffset + kIdx));
                }

                const uint64_t fullScaleOffset =
                    static_cast<uint64_t>(groupIdx) * tilingData.totalInGroup + fullM;
                const uint64_t tileScaleIdx =
                    static_cast<uint64_t>(groupIdx) * baseM + localM;
                perTokenScaleOutGm.SetValue(fullScaleOffset, perTokenScaleTileGm.GetValue(tileScaleIdx));
            }
        }
    }
}

extern "C" __global__ __aicore__ void quant_group_tile_matmul_compare_custom(
        GM_ADDR xQ,
        GM_ADDR weight,
        GM_ADDR fullOut,
        GM_ADDR tileOut,
        GM_ADDR workspace,
        GM_ADDR tilingGm)
{
    if ASCEND_IS_AIV {
        return;
    }

    QuantGroupMatmulCustomTilingData tilingData;
    CopyTiling(&tilingData, tilingGm);

    GlobalTensor<int8_t> xQGm;
    GlobalTensor<int8_t> tileXGm;
    GlobalTensor<int8_t> weightGm;
    GlobalTensor<int32_t> fullOutGm;
    GlobalTensor<int32_t> tileOutGm;
    xQGm.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(xQ));
    tileXGm.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(workspace));
    weightGm.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(weight));
    fullOutGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(fullOut));
    tileOutGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(tileOut));

    const uint32_t curM = tilingData.totalInGroup < tilingData.mmTilingData.baseM
        ? tilingData.totalInGroup
        : tilingData.mmTilingData.baseM;
    const uint32_t curN = tilingData.n < tilingData.mmTilingData.baseN
        ? tilingData.n
        : tilingData.mmTilingData.baseN;
    const uint32_t logicalK = tilingData.kGroupSize;

    for (uint32_t localM = 0; localM < curM; ++localM) {
        const uint64_t srcXOffset = static_cast<uint64_t>(localM) * logicalK;
        const uint64_t dstXOffset = static_cast<uint64_t>(localM) * logicalK;
        for (uint32_t kIdx = 0; kIdx < logicalK; ++kIdx) {
            tileXGm.SetValue(dstXOffset + kIdx, xQGm.GetValue(srcXOffset + kIdx));
        }
    }
    AscendC::PipeBarrier<PIPE_ALL>();
    CleanGmCacheRange(workspace, static_cast<uint64_t>(curM) * logicalK);

    AscendC::TPipe tPipe;
    MT mmFull;
    mmFull.SetSubBlockIdx(0);
    mmFull.Init(&tilingData.mmTilingData, &tPipe);
    mmFull.SetOrgShape(tilingData.totalInGroup, curN, logicalK);
    mmFull.SetSingleShape(curM, curN, logicalK);
    mmFull.SetTensorA(xQGm[0]);
    mmFull.SetTensorB(weightGm[0]);
    uint64_t fullOffset = 0;
    while (mmFull.Iterate()) {
        mmFull.GetTensorC(fullOutGm[fullOffset], 0);
        fullOffset += static_cast<uint64_t>(tilingData.mmTilingData.baseM) * tilingData.mmTilingData.baseN;
    }

    MT mmTile;
    mmTile.SetSubBlockIdx(0);
    mmTile.Init(&tilingData.mmTilingData, &tPipe);
    mmTile.SetOrgShape(curM, curN, logicalK);
    mmTile.SetSingleShape(curM, curN, logicalK);
    mmTile.SetTensorA(tileXGm[0]);
    mmTile.SetTensorB(weightGm[0]);
    uint64_t tileOffset = 0;
    while (mmTile.Iterate()) {
        mmTile.GetTensorC(tileOutGm[tileOffset], 0);
        tileOffset += static_cast<uint64_t>(tilingData.mmTilingData.baseM) * tilingData.mmTilingData.baseN;
    }
}

extern "C" __global__ __aicore__ void quant_group_shared_tile_pack_debug_custom(
        GM_ADDR x,
        GM_ADDR xQOut,
        GM_ADDR perTokenScaleOut,
        GM_ADDR workspace,
        GM_ADDR tilingGm)
{
    QuantGroupMatmulCustomTilingData tilingData;
    CopyTiling(&tilingData, tilingGm);

    GlobalTensor<bfloat16_t> rawXGm;
    GlobalTensor<int8_t> tileXGm;
    GlobalTensor<float> tileScaleGm;
    GlobalTensor<int8_t> xQOutGm;
    GlobalTensor<float> scaleOutGm;

    const uint64_t tileXBytes =
        static_cast<uint64_t>(tilingData.kGroupNum) *
        static_cast<uint64_t>(tilingData.mmTilingData.baseM) *
        static_cast<uint64_t>(tilingData.kGroupSize);
    const uint64_t tileScaleOffset = AlignUp64(tileXBytes, 512);

    rawXGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(x));
    tileXGm.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(workspace));
    tileScaleGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace + tileScaleOffset));
    xQOutGm.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(xQOut));
    scaleOutGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(perTokenScaleOut));

    uint32_t coreIdx = AscendC::GetBlockIdx();
    if ASCEND_IS_AIV {
        coreIdx /= AscendC::GetTaskRation();
    }
    const uint32_t blockDimM = AscendC::Ceil(tilingData.totalInGroup, tilingData.mmTilingData.baseM);
    const uint32_t logicalK = tilingData.kGroupNum * tilingData.kGroupSize;

    for (uint32_t mBlock = 0; mBlock < blockDimM; ++mBlock) {
        uint32_t curM = tilingData.mmTilingData.baseM;
        if (mBlock == blockDimM - 1) {
            curM = tilingData.totalInGroup - mBlock * tilingData.mmTilingData.baseM;
        }

        if ASCEND_IS_AIV {
            const uint32_t totalRows = curM * tilingData.kGroupNum;
            const uint32_t startRow = coreIdx == 0 ? 0 : totalRows;
            for (uint32_t packedRow = startRow; packedRow < totalRows; ++packedRow) {
                const uint32_t groupIdx = packedRow / curM;
                const uint32_t localM = packedRow - groupIdx * curM;
                const uint64_t globalM =
                    static_cast<uint64_t>(mBlock) * tilingData.mmTilingData.baseM + localM;
                const uint64_t rawXOffset =
                    globalM * logicalK + static_cast<uint64_t>(groupIdx) * tilingData.kGroupSize;
                const uint64_t tileXOffset =
                    static_cast<uint64_t>(groupIdx) * tilingData.mmTilingData.baseM * tilingData.kGroupSize +
                    static_cast<uint64_t>(localM) * tilingData.kGroupSize;
                const uint64_t tileScaleElement =
                    static_cast<uint64_t>(groupIdx) * tilingData.mmTilingData.baseM + localM;
                const uint64_t outRow =
                    static_cast<uint64_t>(groupIdx) * tilingData.totalInGroup + globalM;

                float maxAbs = 0.0f;
                for (uint32_t kIdx = 0; kIdx < tilingData.kGroupSize; ++kIdx) {
                    const float value = AscendC::ToFloat(rawXGm.GetValue(rawXOffset + kIdx));
                    const float absValue = value >= 0.0f ? value : -value;
                    if (absValue > maxAbs) {
                        maxAbs = absValue;
                    }
                }
                const float tokenScale = maxAbs / 127.0f;
                for (uint32_t kIdx = 0; kIdx < tilingData.kGroupSize; ++kIdx) {
                    const float value = AscendC::ToFloat(rawXGm.GetValue(rawXOffset + kIdx));
                    const int8_t qValue = QuantizeBf16Value(value, tokenScale);
                    tileXGm.SetValue(tileXOffset + kIdx, qValue);
                    xQOutGm.SetValue(outRow * tilingData.kGroupSize + kIdx, qValue);
                }
                tileScaleGm.SetValue(tileScaleElement, tokenScale);
                scaleOutGm.SetValue(outRow, tokenScale);
                CleanGmCacheRange(workspace + tileXOffset, tilingData.kGroupSize);
                CleanGmCacheRange(workspace + tileScaleOffset + tileScaleElement * sizeof(float), sizeof(float));
            }
        }
        AscendC::SyncAll<false>();
        if ASCEND_IS_AIV {
            if (coreIdx == 0) {
                CleanGmCacheRange(
                    xQOut,
                    static_cast<uint64_t>(tilingData.totalInGroup) *
                        tilingData.kGroupNum * tilingData.kGroupSize);
                CleanGmCacheRange(
                    perTokenScaleOut,
                    static_cast<uint64_t>(tilingData.totalInGroup) *
                        tilingData.kGroupNum * sizeof(float));
            }
        }
        AscendC::SyncAll<false>();
    }
}

extern "C" __global__ __aicore__ void quant_group_tile_dynamic_pack_debug_custom(
        GM_ADDR x,
        GM_ADDR xQOut,
        GM_ADDR perTokenScaleOut,
        GM_ADDR workspace,
        GM_ADDR tilingGm)
{
    if ASCEND_IS_AIC {
        return;
    }

    QuantGroupMatmulCustomTilingData tilingData;
    CopyTiling(&tilingData, tilingGm);

    uint32_t coreIdx = AscendC::GetBlockIdx();
    if ASCEND_IS_AIV {
        coreIdx /= AscendC::GetTaskRation();
    }
    if (AscendC::GetSubBlockIdx() != 0) {
        return;
    }

    AscendC::TPipe localPipe;
    const uint32_t alignedK = AlignUp64(tilingData.kGroupSize, 8);
    const uint32_t qRowBytes = AlignUp64(tilingData.kGroupSize, 32);
    const uint32_t bf16Bytes = AlignUp64(tilingData.kGroupSize * sizeof(bfloat16_t), 32);
    const uint32_t floatBytes = AlignUp64(alignedK * sizeof(float), 32);
    const uint32_t halfBytes = AlignUp64(alignedK * sizeof(half), 32);
    const uint32_t scaleBytes = AlignUp64(tilingData.mmTilingData.baseM * sizeof(float), 32);

    AscendC::TBuf<TPosition::VECCALC> qRowBuffer;
    AscendC::TBuf<TPosition::VECCALC> bf16StageBuffer;
    AscendC::TBuf<TPosition::VECCALC> floatWorkBuffer;
    AscendC::TBuf<TPosition::VECCALC> absWorkBuffer;
    AscendC::TBuf<TPosition::VECCALC> halfBuffer;
    AscendC::TBuf<TPosition::VECCALC> scaleBuffer;
    localPipe.InitBuffer(qRowBuffer, qRowBytes);
    localPipe.InitBuffer(bf16StageBuffer, floatBytes);
    localPipe.InitBuffer(floatWorkBuffer, floatBytes);
    localPipe.InitBuffer(absWorkBuffer, floatBytes);
    localPipe.InitBuffer(halfBuffer, halfBytes);
    localPipe.InitBuffer(scaleBuffer, scaleBytes);

    LocalTensor<int8_t> qRowLocal = qRowBuffer.Get<int8_t>();
    LocalTensor<bfloat16_t> bf16StageLocal = bf16StageBuffer.Get<bfloat16_t>();
    LocalTensor<float> floatWorkLocal = floatWorkBuffer.Get<float>();
    LocalTensor<float> absWorkLocal = absWorkBuffer.Get<float>();
    LocalTensor<half> halfTmpLocal = halfBuffer.Get<half>();
    LocalTensor<float> scaleLocal = scaleBuffer.Get<float>();

    GlobalTensor<bfloat16_t> rawXGm;
    GlobalTensor<int8_t> xQOutGm;
    GlobalTensor<float> scaleOutGm;
    rawXGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(x));
    xQOutGm.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(xQOut));
    scaleOutGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(perTokenScaleOut));

    const uint32_t logicalK = tilingData.kGroupNum * tilingData.kGroupSize;
    const uint32_t totalRows = tilingData.totalInGroup * tilingData.kGroupNum;
    const uint32_t rowsPerCore = AscendC::Ceil(totalRows, tilingData.coreNum);
    const uint32_t rowStart = coreIdx * rowsPerCore;
    const uint32_t rowEnd =
        rowStart + rowsPerCore < totalRows ? rowStart + rowsPerCore : totalRows;
    AscendC::DataCopyExtParams copyParams{
        1, static_cast<uint32_t>(tilingData.kGroupSize * sizeof(bfloat16_t)), 0, 0, 0};
    AscendC::DataCopyPadExtParams<bfloat16_t> padParams{true, 0, 0, 0};
    const auto eventIdVToS = localPipe.FetchEventID(AscendC::HardEvent::V_S);
    const auto eventIdSToV = localPipe.FetchEventID(AscendC::HardEvent::S_V);

    for (uint32_t packedRow = rowStart; packedRow < rowEnd; ++packedRow) {
        const uint32_t groupIdx = packedRow / tilingData.totalInGroup;
        const uint32_t tokenIdx = packedRow - groupIdx * tilingData.totalInGroup;
        const uint64_t rawXOffset =
            static_cast<uint64_t>(tokenIdx) * logicalK +
            static_cast<uint64_t>(groupIdx) * tilingData.kGroupSize;
        const uint64_t outXOffset = static_cast<uint64_t>(packedRow) * tilingData.kGroupSize;

        DataCopyPad(bf16StageLocal, rawXGm[rawXOffset], copyParams, padParams);
        AscendC::PipeBarrier<PIPE_MTE2>();
        Cast(floatWorkLocal, bf16StageLocal, AscendC::RoundMode::CAST_NONE, alignedK);
        AscendC::PipeBarrier<PIPE_V>();

        Abs(absWorkLocal, floatWorkLocal, alignedK);
        AscendC::PipeBarrier<PIPE_V>();
        LocalTensor<float> reduceTmpLocal = bf16StageLocal.ReinterpretCast<float>();
        ReduceMax(absWorkLocal, absWorkLocal, reduceTmpLocal, static_cast<int32_t>(alignedK), false);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetFlag<AscendC::HardEvent::V_S>(eventIdVToS);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(eventIdVToS);
        const float tokenScale = absWorkLocal.GetValue(0) / 127.0f;
        scaleLocal.SetValue(0, tokenScale);
        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(SCALAR_TO_MTE3_EVENT_ID);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(SCALAR_TO_MTE3_EVENT_ID);

        if (tokenScale > 0.0f) {
            AscendC::SetFlag<AscendC::HardEvent::S_V>(eventIdSToV);
            AscendC::WaitFlag<AscendC::HardEvent::S_V>(eventIdSToV);
            Muls(floatWorkLocal, floatWorkLocal, 1.0f / tokenScale, alignedK);
            AscendC::PipeBarrier<PIPE_V>();
        }
        Maxs(floatWorkLocal, floatWorkLocal, -127.0f, alignedK);
        AscendC::PipeBarrier<PIPE_V>();
        Mins(floatWorkLocal, floatWorkLocal, 127.0f, alignedK);
        AscendC::PipeBarrier<PIPE_V>();

        Cast(halfTmpLocal, floatWorkLocal, AscendC::RoundMode::CAST_RINT, alignedK);
        AscendC::PipeBarrier<PIPE_V>();
        Cast(qRowLocal, halfTmpLocal, AscendC::RoundMode::CAST_RINT, alignedK);
        AscendC::PipeBarrier<PIPE_V>();
        DataCopy(xQOutGm[outXOffset], qRowLocal, tilingData.kGroupSize);
        DataCopyPad2D(scaleOutGm[packedRow], scaleLocal, 1, 1, 1, 1);
        AscendC::PipeBarrier<PIPE_MTE3>();
    }
}
