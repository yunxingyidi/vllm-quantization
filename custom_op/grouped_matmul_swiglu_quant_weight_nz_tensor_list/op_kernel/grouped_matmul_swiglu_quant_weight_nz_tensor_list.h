/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file grouped_matmul_swiglu_quant_weight_nz_tensor_list.h
 * \brief
 */
#ifndef ASCENDC_GROUPED_MATMUL_SWIGLU_QUANT_WEIGHT_NZ_TENSOR_LIST_H
#define ASCENDC_GROUPED_MATMUL_SWIGLU_QUANT_WEIGHT_NZ_TENSOR_LIST_H

#include "grouped_matmul_swiglu_quant_weight_nz_tensor_list_utils.h"
namespace GROUPED_MATMUL_SWIGLU_QUANT_WEIGHT_NZ_TENSOR_LIST {
#ifndef GROUP_QUANT_STAGE
#define GROUP_QUANT_STAGE 8
#endif
constexpr uint32_t GROUP_QUANT_DEBUG_STAGE = GROUP_QUANT_STAGE;

template <typename SCALE_DTYPE>
__aicore__ inline float GetGroupQuantScaleValue(GlobalTensor<SCALE_DTYPE>& scaleGM, uint64_t offset)
{
    return static_cast<float>(scaleGM.GetValue(offset));
}

/** @brief internal computation class
*/
template <class mmType, bool sync = false, typename CHANNELDTYPE = float>
class GMMSwigluCompute{
 public:
    using AT = typename mmType::AT::T;
    using BT = typename mmType::BT::T;
    using B = typename mmType::BT;
    using CT = typename mmType::CT::T;
    using BiasT = typename mmType::BiasT::T;
    using WT = int8_t;
    constexpr static bool transposeX = mmType::AT::isTrans;
    constexpr static bool transposeW = mmType::BT::isTrans;

    /** @brief constructor */
    __aicore__ inline GMMSwigluCompute(typename mmType::MT& mm_): mm(mm_)  {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR perChannelScale, GM_ADDR perTokenScale,
                                GM_ADDR groupList, GM_ADDR quantOutput, GM_ADDR quantScaleOutput,
                                GM_ADDR workspace,
                                const GMMSwigluTensorListBaseParams* __restrict gmmBaseParamsIN,
                                const TCubeTiling* __restrict mmTilingDataIN,
                                const GMMSwigluTensorList* __restrict gmmSwigluIN, TPipe* tPipeIN);
    __aicore__ inline void Process();
 private:
    __aicore__ inline void ProcessGroupQuant();
    __aicore__ inline void ProcessGroupQuantProduction();
    __aicore__ inline void ProcessGroupQuantProductionGlobal();
    __aicore__ inline void ProcessGroupQuantLegacyDebug();
    __aicore__ inline bool UseDirectFullKGroupX() const;
    __aicore__ inline bool UseDirectGroupQuantScale() const;
    __aicore__ inline void PackGroupQuantX();
    __aicore__ inline void PackGroupQuantXFullK(uint32_t kGroupIdx);
    __aicore__ inline void PackGroupQuantWeight(uint32_t expertIdx);
    __aicore__ inline void PackGroupQuantScale();
    __aicore__ inline void ProcessGroupQuantTile(uint32_t expertIdx, uint32_t rowStart, uint32_t tileM,
                                                 const MNConfig& mnConfig, uint32_t bufferTileM = 0);
    __aicore__ inline void QuantGroupQuantPostRow(LocalTensor<float> postLocal,
                                                  LocalTensor<float> absLocal,
                                                  LocalTensor<float> reduceWorkLocal,
                                                  LocalTensor<float> reduceResLocal,
                                                  LocalTensor<float> reduceTmpLocal,
                                                  LocalTensor<int8_t> quantLocal,
                                                  LocalTensor<float> quantScaleLocal,
                                                  LocalTensor<int8_t> castSpace,
                                                  uint32_t rowLocal, uint32_t halfTokenLen);

    __aicore__ inline void MMCompute(uint32_t groupIdx, MNConfig& mnConfig, uint32_t coreIdx);
    __aicore__ inline void MMComputeGroupQuant(uint32_t expertIdx, uint32_t kGroupIdx, MNConfig& mnConfig, uint32_t coreIdx);
    __aicore__ inline void MMComputeGroupQuantOriginalWeight(uint32_t expertIdx, uint32_t kGroupIdx, MNConfig& mnConfig, uint32_t coreIdx);
    __aicore__ inline void MMComputeGroupQuantFullK(uint32_t expertIdx, MNConfig& mnConfig, uint32_t coreIdx);

    __aicore__ inline void UpdateMnConfig(MNConfig &mnConfig);

    __aicore__ inline void SetMNConfig(const int32_t splitValue, const uint32_t groupIdx, MNConfig &mnConfig);

    __aicore__ inline void SetMKN(const int32_t splitValue, const uint32_t groupIdx, MNConfig &mnConfig);

    __aicore__ inline uint64_t GetWOffset(uint32_t tailN, uint32_t k);

    __aicore__ inline void MNBlockIdxCompute(MNConfig &mnConfig, const uint32_t curBlock,
    const uint32_t count, const uint32_t thresholdM_dimN);
    template <typename DTYPE_CS>
    __aicore__ inline void UpdateChannelScale(uint32_t loopidx);
    __aicore__ inline void VectorCompute(uint32_t loopidx);
    template <typename DTYPE_CS>
    __aicore__ inline void PreLoadTokenAndChannel(LocalTensor<float>& channelScaleLocal);
    __aicore__ inline void UpdateVecConfig(uint32_t blockIdx, VecConfig& vecConfig);
    __aicore__ inline void customDataCopyIn(uint32_t outLoopIdx);
    template <typename DTYPE_CS>
    __aicore__ inline void customDataCopyInGroupQuant(uint32_t outLoopIdx);
    __aicore__ inline void customDataCopyOut();
    __aicore__ inline void Dequant(uint32_t loopidx);
    __aicore__ inline void Quant(uint32_t loopidx, uint32_t scaleGmRow = 0xFFFFFFFFU);
    __aicore__ inline void Swiglu(uint32_t loopidx);
 private:
    typename mmType::MT& mm;
    const GMMSwigluTensorListBaseParams* __restrict gmmBaseParams;
    const GMMSwigluTensorList* __restrict gmmSwigluTensorList;
    const TCubeTiling* __restrict mmTilingData;
    uint32_t blockIdx;
    VecConfig vecConfig;
    TPipe* pipe;
    GlobalTensor<int8_t> xGM, weightGM;
    GlobalTensor<CHANNELDTYPE> perChannelScaleGM;
    GlobalTensor<float> perTokenScaleGM;
    GlobalTensor<int64_t> groupListGM;
    GlobalTensor<int8_t> quantOutputGM;
    GlobalTensor<float> quantScaleOutputGM;
    GlobalTensor<int32_t> mmOutGM;
    // define the que
    TQue<QuePosition::VECIN, 1> mmOutQueue;
    TQue<QuePosition::VECIN, 1> perChannelScaleInQueue;
    TQue<QuePosition::VECOUT, 1> quantOutQueue;
    TQue<QuePosition::VECOUT, 1> quantScaleOutQueue;
    TQue<QuePosition::VECOUT, 1> groupQuantAccumOutQueue;
    TBuf<TPosition::VECCALC> reduceWorkspace;
    TBuf<TPosition::VECCALC> castWorkspace;
    TBuf<TPosition::VECCALC> groupQuantAccumWorkspace;
    TBuf<TPosition::VECCALC> groupQuantScaleWorkspace;
    TBuf<TPosition::VECCALC> groupQuantPackWorkspace;
    TBuf<TPosition::VECCALC> groupQuantWeightPackWorkspace;
    bool sequentialWrite = true;
    uint32_t cubeNum;  // Matmul completions on the kernel
    uint32_t groupNum; // Matmul completions on the kernel
    int32_t preOffset;
    int64_t aicCoreNum;
    int64_t aivCoreNum;
    GM_ADDR xTensorPtr;
    GM_ADDR weightTensorPtr;
    GM_ADDR perChannelScalePtr;
    GM_ADDR quantOutputPtr;
    GM_ADDR groupQuantXPackedPtr;
    GM_ADDR groupQuantWeightPackedPtr;
    GM_ADDR groupQuantScalePackedPtr;
    GM_ADDR groupQuantAccumPtr;
};

template <typename mmType, bool sync, typename CHANNELDTYPE>
__aicore__ inline void GMMSwigluCompute<mmType, sync, CHANNELDTYPE>::Init(GM_ADDR x, GM_ADDR weight, GM_ADDR perChannelScale, GM_ADDR perTokenScale,
                                                            GM_ADDR groupList, GM_ADDR quantOutput, GM_ADDR quantScaleOutput,
                                                            GM_ADDR workspace,
                                                            const GMMSwigluTensorListBaseParams* __restrict gmmSwigluTensorListBaseParamsIn,
                                                            const TCubeTiling* __restrict mmTilingDataIN,
                                                            const GMMSwigluTensorList* __restrict gmmSwigluIN, TPipe* tPipeIN)
{
    aicCoreNum = GetBlockNum();
    aivCoreNum = aicCoreNum * 2;
    blockIdx = GetBlockIdx();
    mmTilingData = mmTilingDataIN;
    gmmBaseParams = gmmSwigluTensorListBaseParamsIn;
    gmmSwigluTensorList = gmmSwigluIN;
    pipe = tPipeIN;
    xTensorPtr = x;
    weightTensorPtr = weight;
    perChannelScalePtr = perChannelScale;
    quantOutputPtr = quantOutput;
    uint64_t groupQuantPartialElements =
        static_cast<uint64_t>(gmmSwigluTensorList->groupQuantGroupNum) *
        gmmBaseParams->M * gmmSwigluTensorList->tokenLen;
    const uint64_t groupQuantOutputElements =
        static_cast<uint64_t>(gmmBaseParams->M) * gmmSwigluTensorList->tokenLen;
    if (groupQuantPartialElements < groupQuantOutputElements) {
        groupQuantPartialElements = groupQuantOutputElements;
    }
    uint64_t groupQuantPartialBytes = groupQuantPartialElements * sizeof(int32_t);
    groupQuantAccumPtr = workspace + groupQuantPartialBytes;
    groupQuantXPackedPtr = groupQuantAccumPtr +
                            static_cast<uint64_t>(gmmBaseParams->M) *
                            gmmSwigluTensorList->tokenLen * sizeof(float);
    groupQuantWeightPackedPtr = groupQuantXPackedPtr +
                                static_cast<uint64_t>(gmmBaseParams->M) * gmmBaseParams->K * sizeof(int8_t);
    groupQuantScalePackedPtr = groupQuantWeightPackedPtr +
                               static_cast<uint64_t>(gmmBaseParams->K) * gmmBaseParams->N * sizeof(int8_t);
    groupNum = gmmSwigluTensorList->groupListLen;
    if ASCEND_IS_AIC {
        groupListGM.SetGlobalBuffer((__gm__ int64_t *)groupList, gmmSwigluTensorList->groupListLen);
        quantScaleOutputGM.SetGlobalBuffer((__gm__ float *)quantScaleOutput, gmmBaseParams->M);
        uint64_t mmOutSize = static_cast<uint64_t>(gmmBaseParams->M) * gmmSwigluTensorList->tokenLen;
        if (gmmSwigluTensorList->enableGroupQuant) {
            mmOutSize = groupQuantPartialElements;
        }
        mmOutGM.SetGlobalBuffer((__gm__ int32_t *)workspace, mmOutSize);
    }
    if ASCEND_IS_AIV {
        uint64_t mmOutSize = static_cast<uint64_t>(gmmBaseParams->M) * gmmSwigluTensorList->tokenLen;
        if (gmmSwigluTensorList->enableGroupQuant) {
            mmOutSize = groupQuantPartialElements;
        }
        mmOutGM.SetGlobalBuffer((__gm__ int32_t *)workspace, mmOutSize);
        perChannelScaleGM.SetGlobalBuffer((__gm__ CHANNELDTYPE *)perChannelScale, gmmSwigluTensorList->groupListLen * gmmSwigluTensorList->tokenLen);
        perTokenScaleGM.SetGlobalBuffer((__gm__ float *)perTokenScale, gmmBaseParams->M);
        groupListGM.SetGlobalBuffer((__gm__ int64_t *)groupList, gmmSwigluTensorList->groupListLen);
        quantOutputGM.SetGlobalBuffer((__gm__ int8_t *)quantOutput, gmmBaseParams->M * gmmSwigluTensorList->tokenLen / 2);
        quantScaleOutputGM.SetGlobalBuffer((__gm__ float *)quantScaleOutput, gmmBaseParams->M);
    }
}

template <typename mmType, bool sync, typename CHANNELDTYPE>
__aicore__ inline bool GMMSwigluCompute<mmType, sync, CHANNELDTYPE>::UseDirectFullKGroupX() const
{
    // One full-K group has exactly the same row-major layout as the input.
    // Repacking it into groupQuantXPackedPtr would only add a GM -> UB -> GM
    // round trip before the Cube reads the exact same bytes again.
    return gmmSwigluTensorList->groupQuantGroupNum == 1 &&
           gmmSwigluTensorList->groupQuantGroupSize == gmmBaseParams->K;
}

template <typename mmType, bool sync, typename CHANNELDTYPE>
__aicore__ inline bool GMMSwigluCompute<mmType, sync, CHANNELDTYPE>::UseDirectGroupQuantScale() const
{
    // The production loader supplies one contiguous FP32 [G, N] scale tensor
    // for every expert.  It can be consumed directly from the TensorList
    // payload; staging all experts into another GM buffer is redundant.
    return IsSameType<CHANNELDTYPE, float>::value;
}

template <typename mmType, bool sync, typename CHANNELDTYPE>
__aicore__ inline void GMMSwigluCompute<mmType, sync, CHANNELDTYPE>::Process() {
    if (gmmSwigluTensorList->enableGroupQuant) {
        ProcessGroupQuant();
        return;
    }

    MNConfig mnConfig;
    if ASCEND_IS_AIC {
        preOffset = 0;
        int32_t prevSplitValue = 0;
        for (uint32_t groupIdx = 0, count = 0; groupIdx < gmmSwigluTensorList->groupListLen; ++groupIdx) {
            UpdateMnConfig(mnConfig);
            int32_t currSplitValue = static_cast<int32_t>(groupListGM.GetValue(groupIdx));
            int32_t splitValue = currSplitValue - prevSplitValue;
            prevSplitValue = currSplitValue;
            SetMNConfig(splitValue, groupIdx, mnConfig);
            if (mnConfig.m <= 0 || mnConfig.k <= 0 || mnConfig.n <= 0) {
                continue;
            }
            mnConfig.blockDimM = Ceil(mnConfig.m, mnConfig.singleM);
            mnConfig.blockDimN = Ceil(mnConfig.n, mnConfig.singleN);

            uint32_t curCount = count + mnConfig.blockDimM * mnConfig.blockDimN;
            uint32_t curBlock = blockIdx >= count ? blockIdx : blockIdx + gmmBaseParams->coreNum;
            uint32_t thresholdM_dimN = THRESHOLD_BLOCK_NUM * mnConfig.blockDimN;

            while (curBlock < curCount) {
                MNBlockIdxCompute(mnConfig, curBlock, count, thresholdM_dimN);
                MMCompute(groupIdx, mnConfig, blockIdx);
                curBlock += aicCoreNum;
            }
            count = curCount % gmmBaseParams->coreNum;
        }
        SyncAll<false>();
    }

    if ASCEND_IS_AIV {
        UpdateVecConfig(blockIdx, vecConfig);
        if (blockIdx < vecConfig.usedCoreNum) {
            LocalTensor<float> channelScaleLocal = perChannelScaleInQueue.AllocTensor<float>();
            LocalTensor<int32_t> mmLocal = mmOutQueue.AllocTensor<int32_t>();
            LocalTensor<int8_t> quantLocal = quantOutQueue.AllocTensor<int8_t>();
            LocalTensor<float> quantScaleLocal = quantScaleOutQueue.AllocTensor<float>();
            mmOutQueue.EnQue(mmLocal);
            quantScaleOutQueue.EnQue(quantScaleLocal);
            quantOutQueue.EnQue(quantLocal);
            PreLoadTokenAndChannel<CHANNELDTYPE>(channelScaleLocal);
        }
        SyncAll<false>();
        if (blockIdx < vecConfig.usedCoreNum) {
            LocalTensor<int8_t> quantLocal;
            LocalTensor<float> quantScaleLocal;
            for (uint32_t outLoopIdx = 0; outLoopIdx < vecConfig.outLoopNum; outLoopIdx++) {
                vecConfig.innerLoopNum = outLoopIdx == (vecConfig.outLoopNum - 1)
                                            ? vecConfig.tailLoopNum
                                            : gmmSwigluTensorList->maxProcessRowNum;
                customDataCopyIn(outLoopIdx);
                for (uint32_t innerLoopIdx = 0; innerLoopIdx < vecConfig.innerLoopNum; innerLoopIdx++) {
                    UpdateChannelScale<CHANNELDTYPE>(innerLoopIdx);
                    VectorCompute(innerLoopIdx);
                }
                customDataCopyOut();
            }

            LocalTensor<float> channelScaleLocal = perChannelScaleInQueue.DeQue<float>();
            LocalTensor<int32_t> mmLocal = mmOutQueue.DeQue<int32_t>();
            quantLocal = quantOutQueue.DeQue<int8_t>();
            quantScaleLocal = quantScaleOutQueue.DeQue<float>();
            perChannelScaleInQueue.FreeTensor(channelScaleLocal);
            mmOutQueue.FreeTensor(mmLocal);
            quantScaleOutQueue.FreeTensor(quantScaleLocal);
            quantOutQueue.FreeTensor(quantLocal);
        } else {
            return;
        }
    }
}

template <typename mmType, bool sync, typename CHANNELDTYPE>
__aicore__ inline void GMMSwigluCompute<mmType, sync, CHANNELDTYPE>::ProcessGroupQuant()
{
    if (gmmSwigluTensorList->groupQuantGlobalSchedule != 0) {
        ProcessGroupQuantProductionGlobal();
        return;
    }
    ProcessGroupQuantProduction();
}

template <typename mmType, bool sync, typename CHANNELDTYPE>
__aicore__ inline void GMMSwigluCompute<mmType, sync, CHANNELDTYPE>::ProcessGroupQuantProduction()
{
    const uint32_t tokenLen = gmmSwigluTensorList->tokenLen;
    const uint32_t kGroupNum = gmmSwigluTensorList->groupQuantGroupNum;
    if ASCEND_IS_AIV {
        if (!UseDirectGroupQuantScale()) {
            PackGroupQuantScale();
        }
        // A full-K single group is already in the layout consumed by Matmul.
        if (!UseDirectFullKGroupX()) {
            PackGroupQuantX();
        }
    }
    SyncAll<false>();
    uint32_t prevExpertEnd = 0;
    for (uint32_t expertIdx = 0; expertIdx < gmmSwigluTensorList->groupListLen; ++expertIdx) {
        const uint32_t expertEnd = static_cast<uint32_t>(groupListGM.GetValue(expertIdx));
        const uint32_t expertM = expertEnd - prevExpertEnd;
        if (expertM > 0) {
            MNConfig mnConfig;
            mnConfig.m = expertM;
            mnConfig.k = gmmSwigluTensorList->groupQuantGroupSize;
            mnConfig.n = gmmBaseParams->N;
            mnConfig.baseM = BASIC_M;
            mnConfig.baseN = BASIC_N;
            mnConfig.singleM = SINGLE_CORE_M;
            mnConfig.singleN = SINGLE_CORE_N;
            mnConfig.xBaseOffset = static_cast<int64_t>(prevExpertEnd) * gmmBaseParams->K;
            mnConfig.yBaseOffset = static_cast<int64_t>(prevExpertEnd) * mnConfig.n;
            mnConfig.blockDimM = Ceil(mnConfig.m, mnConfig.singleM);
            mnConfig.blockDimN = 1;
            if ASCEND_IS_AIC {
                for (uint32_t kGroupIdx = 0; kGroupIdx < kGroupNum; ++kGroupIdx) {
                    const uint32_t mnBlockCount = mnConfig.blockDimM * mnConfig.blockDimN;
                    const uint32_t thresholdMdimN = THRESHOLD_BLOCK_NUM * mnConfig.blockDimN;
                    for (uint32_t mnBlock = blockIdx; mnBlock < mnBlockCount; mnBlock += aicCoreNum) {
                        MNBlockIdxCompute(mnConfig, mnBlock, 0, thresholdMdimN);
                        MMComputeGroupQuantOriginalWeight(expertIdx, kGroupIdx, mnConfig, blockIdx);
                    }
                }
            }
            SyncAll<false>();
            if ASCEND_IS_AIV {
                const uint32_t halfTokenLen = tokenLen / BISECT;
                constexpr uint32_t ubBudgetBytes = 160 * 1024;
                const uint32_t fixedBytes = tokenLen * sizeof(float) +
                                            (AlignUp<8>(halfTokenLen) * 4 + FLOAT_UB_BLOCK_UNIT_SIZE * 2) *
                                                sizeof(float) +
                                            UB_BLOCK_UNIT_SIZE;
                const uint32_t bytesPerRow = tokenLen * (sizeof(int32_t) + sizeof(float) * 2) +
                                             halfTokenLen * sizeof(int8_t) + sizeof(float);
                uint32_t tileRows = fixedBytes < ubBudgetBytes
                                        ? (ubBudgetBytes - fixedBytes) / bytesPerRow
                                        : 1;
                tileRows = tileRows == 0 ? 1 : tileRows;
                tileRows = tileRows > gmmSwigluTensorList->maxProcessRowNum
                               ? gmmSwigluTensorList->maxProcessRowNum
                               : tileRows;
                // Keep each partial CopyIn at roughly 16 KiB while exposing enough tiles to AIV cores.
                constexpr uint32_t targetPartialBytes = 16 * 1024;
                uint32_t parallelTileRows = targetPartialBytes / (tokenLen * sizeof(int32_t));
                parallelTileRows = parallelTileRows == 0 ? 1 : parallelTileRows;
                tileRows = tileRows > parallelTileRows ? parallelTileRows : tileRows;
                const uint32_t maxTileElements = tileRows * tokenLen;
                const uint32_t alignedTileElements = AlignUp<8>(maxTileElements);
                pipe->Reset();
                pipe->InitBuffer(groupQuantPackWorkspace,
                                 alignedTileElements * sizeof(int32_t) +
                                     alignedTileElements * sizeof(float));
                pipe->InitBuffer(groupQuantAccumOutQueue, 1, alignedTileElements * sizeof(float));
                pipe->InitBuffer(groupQuantScaleWorkspace, AlignUp<8>(tokenLen) * sizeof(float));
                pipe->InitBuffer(quantOutQueue, 1,
                                 AlignUp<int32_t>(tileRows * halfTokenLen, UB_BLOCK_UNIT_SIZE));
                pipe->InitBuffer(quantScaleOutQueue, 1, AlignUp<8>(tileRows) * sizeof(float));
                const uint32_t reduceWorkElements =
                    AlignUp<8>(halfTokenLen) * 4 + FLOAT_UB_BLOCK_UNIT_SIZE * 2;
                pipe->InitBuffer(reduceWorkspace, reduceWorkElements * sizeof(float));
                pipe->InitBuffer(castWorkspace, UB_BLOCK_UNIT_SIZE);
                const uint32_t tileCount = Ceil(expertM, tileRows);
                for (uint32_t tileIdx = blockIdx; tileIdx < tileCount; tileIdx += aivCoreNum) {
                    const uint32_t tileOffset = tileIdx * tileRows;
                    const uint32_t tileM = tileOffset + tileRows < expertM ? tileRows : expertM - tileOffset;
                    ProcessGroupQuantTile(expertIdx, prevExpertEnd + tileOffset, tileM, mnConfig, tileRows);
                }
            }
            SyncAll<false>();
        }
        prevExpertEnd = expertEnd;
    }
}

template <typename mmType, bool sync, typename CHANNELDTYPE>
__aicore__ inline void GMMSwigluCompute<mmType, sync, CHANNELDTYPE>::ProcessGroupQuantProductionGlobal()
{
    const uint32_t tokenLen = gmmSwigluTensorList->tokenLen;
    const uint32_t kGroupNum = gmmSwigluTensorList->groupQuantGroupNum;
    if ASCEND_IS_AIV {
        if (!UseDirectGroupQuantScale()) {
            PackGroupQuantScale();
        }
        if (!UseDirectFullKGroupX()) {
            PackGroupQuantX();
        }
    }
    SyncAll<false>();

    // Flatten expert/K-group/M/N work into one global task stream.  The original
    // implementation restarted scheduling for every expert and K group, which
    // left almost every AIC idle for the small per-expert M common in MoE.
    if ASCEND_IS_AIC {
        uint64_t globalTaskBase = 0;
        uint32_t prevExpertEnd = 0;
        for (uint32_t expertIdx = 0; expertIdx < gmmSwigluTensorList->groupListLen; ++expertIdx) {
            const uint32_t expertEnd = static_cast<uint32_t>(groupListGM.GetValue(expertIdx));
            const uint32_t expertM = expertEnd - prevExpertEnd;
            if (expertM > 0) {
                MNConfig mnConfig;
                mnConfig.m = expertM;
                mnConfig.k = gmmSwigluTensorList->groupQuantGroupSize;
                mnConfig.n = gmmBaseParams->N;
                mnConfig.baseM = BASIC_M;
                mnConfig.baseN = BASIC_N;
                mnConfig.singleM = SINGLE_CORE_M;
                mnConfig.singleN = gmmSwigluTensorList->groupQuantSingleN;
                mnConfig.xBaseOffset = static_cast<int64_t>(prevExpertEnd) * gmmBaseParams->K;
                mnConfig.yBaseOffset = static_cast<int64_t>(prevExpertEnd) * mnConfig.n;
                mnConfig.blockDimM = Ceil(mnConfig.m, mnConfig.singleM);
                mnConfig.blockDimN = Ceil(mnConfig.n, mnConfig.singleN);
                const uint32_t mnBlockCount = mnConfig.blockDimM * mnConfig.blockDimN;
                const uint32_t expertTaskCount = kGroupNum * mnBlockCount;
                const uint32_t taskBaseRemainder = static_cast<uint32_t>(globalTaskBase % aicCoreNum);
                const uint32_t firstLocalTask =
                    (blockIdx + static_cast<uint32_t>(aicCoreNum) - taskBaseRemainder) %
                    static_cast<uint32_t>(aicCoreNum);
                for (uint32_t localTask = firstLocalTask; localTask < expertTaskCount;
                     localTask += static_cast<uint32_t>(aicCoreNum)) {
                    const uint32_t kGroupIdx = localTask / mnBlockCount;
                    const uint32_t mnBlock = localTask - kGroupIdx * mnBlockCount;
                    MNBlockIdxCompute(mnConfig, mnBlock, 0, THRESHOLD_BLOCK_NUM * mnConfig.blockDimN);
                    MMComputeGroupQuantOriginalWeight(expertIdx, kGroupIdx, mnConfig, blockIdx);
                }
                globalTaskBase += expertTaskCount;
            }
            prevExpertEnd = expertEnd;
        }
    }
    SyncAll<false>();

    if ASCEND_IS_AIV {
        const uint32_t halfTokenLen = tokenLen / BISECT;
        const uint32_t ubBudgetBytes = gmmSwigluTensorList->groupQuantUbBudgetBytes;
        const uint32_t fixedBytes = tokenLen * sizeof(float) +
                                    (AlignUp<8>(halfTokenLen) * 4 + FLOAT_UB_BLOCK_UNIT_SIZE * 2) *
                                        sizeof(float) +
                                    UB_BLOCK_UNIT_SIZE;
        const uint32_t bytesPerRow = tokenLen * (sizeof(int32_t) + sizeof(float) * 2) +
                                     halfTokenLen * sizeof(int8_t) + sizeof(float);
        uint32_t tileRows = fixedBytes < ubBudgetBytes ? (ubBudgetBytes - fixedBytes) / bytesPerRow : 1;
        tileRows = tileRows == 0 ? 1 : tileRows;
        tileRows = tileRows > gmmSwigluTensorList->maxProcessRowNum
                       ? gmmSwigluTensorList->maxProcessRowNum
                       : tileRows;
        const uint32_t targetPartialBytes = gmmSwigluTensorList->groupQuantPartialTileBytes;
        uint32_t parallelTileRows = targetPartialBytes / (tokenLen * sizeof(int32_t));
        parallelTileRows = parallelTileRows == 0 ? 1 : parallelTileRows;
        tileRows = tileRows > parallelTileRows ? parallelTileRows : tileRows;

        const uint32_t alignedTileElements = AlignUp<8>(tileRows * tokenLen);
        pipe->Reset();
        pipe->InitBuffer(groupQuantPackWorkspace,
                         alignedTileElements * sizeof(int32_t) + alignedTileElements * sizeof(float));
        pipe->InitBuffer(groupQuantAccumOutQueue, 1, alignedTileElements * sizeof(float));
        pipe->InitBuffer(groupQuantScaleWorkspace, AlignUp<8>(tokenLen) * sizeof(float));
        pipe->InitBuffer(quantOutQueue, 1,
                         AlignUp<int32_t>(tileRows * halfTokenLen, UB_BLOCK_UNIT_SIZE));
        pipe->InitBuffer(quantScaleOutQueue, 1, AlignUp<8>(tileRows) * sizeof(float));
        const uint32_t reduceWorkElements =
            AlignUp<8>(halfTokenLen) * 4 + FLOAT_UB_BLOCK_UNIT_SIZE * 2;
        pipe->InitBuffer(reduceWorkspace, reduceWorkElements * sizeof(float));
        pipe->InitBuffer(castWorkspace, UB_BLOCK_UNIT_SIZE);

        uint64_t globalTileBase = 0;
        uint32_t prevExpertEnd = 0;
        for (uint32_t expertIdx = 0; expertIdx < gmmSwigluTensorList->groupListLen; ++expertIdx) {
            const uint32_t expertEnd = static_cast<uint32_t>(groupListGM.GetValue(expertIdx));
            const uint32_t expertM = expertEnd - prevExpertEnd;
            if (expertM > 0) {
                MNConfig mnConfig;
                mnConfig.m = expertM;
                mnConfig.n = gmmBaseParams->N;
                const uint32_t tileCount = Ceil(expertM, tileRows);
                const uint32_t tileBaseRemainder = static_cast<uint32_t>(globalTileBase % aivCoreNum);
                const uint32_t firstLocalTile =
                    (blockIdx + static_cast<uint32_t>(aivCoreNum) - tileBaseRemainder) %
                    static_cast<uint32_t>(aivCoreNum);
                for (uint32_t localTile = firstLocalTile; localTile < tileCount;
                     localTile += static_cast<uint32_t>(aivCoreNum)) {
                    const uint32_t tileOffset = localTile * tileRows;
                    const uint32_t tileM = tileOffset + tileRows < expertM ? tileRows : expertM - tileOffset;
                    ProcessGroupQuantTile(expertIdx, prevExpertEnd + tileOffset, tileM, mnConfig, tileRows);
                }
                globalTileBase += tileCount;
            }
            prevExpertEnd = expertEnd;
        }
    }
    SyncAll<false>();
}

template <typename mmType, bool sync, typename CHANNELDTYPE>
__aicore__ inline void GMMSwigluCompute<mmType, sync, CHANNELDTYPE>::ProcessGroupQuantLegacyDebug() {
    if constexpr (GROUP_QUANT_DEBUG_STAGE <= 2) {
        if ASCEND_IS_AIV {
            if constexpr (GROUP_QUANT_DEBUG_STAGE >= 1) {
                PackGroupQuantX();
            }
            if constexpr (GROUP_QUANT_DEBUG_STAGE >= 2) {
                for (uint32_t expertIdx = 0; expertIdx < gmmSwigluTensorList->groupListLen; ++expertIdx) {
                    PackGroupQuantWeight(expertIdx);
                }
            }
            UpdateVecConfig(blockIdx, vecConfig);
            if (blockIdx < vecConfig.usedCoreNum) {
                LocalTensor<int8_t> quantLocal = quantOutQueue.AllocTensor<int8_t>();
                LocalTensor<float> quantScaleLocal = quantScaleOutQueue.AllocTensor<float>();
                quantOutQueue.EnQue(quantLocal);
                quantScaleOutQueue.EnQue(quantScaleLocal);
                for (uint32_t outLoopIdx = 0; outLoopIdx < vecConfig.outLoopNum; outLoopIdx++) {
                    vecConfig.innerLoopNum = outLoopIdx == (vecConfig.outLoopNum - 1)
                                                ? vecConfig.tailLoopNum
                                                : gmmSwigluTensorList->maxProcessRowNum;
                    quantLocal = quantOutQueue.DeQue<int8_t>();
                    quantScaleLocal = quantScaleOutQueue.DeQue<float>();
                    LocalTensor<int32_t> quantLocalI32 = quantLocal.template ReinterpretCast<int32_t>();
                    Duplicate(quantLocalI32, static_cast<int32_t>(0),
                              vecConfig.innerLoopNum * gmmSwigluTensorList->tokenLen / 2 / sizeof(int32_t));
                    Duplicate(quantScaleLocal, static_cast<float>(1.0), vecConfig.innerLoopNum);
                    PipeBarrier<PIPE_V>();
                    quantOutQueue.EnQue(quantLocal);
                    quantScaleOutQueue.EnQue(quantScaleLocal);
                    customDataCopyOut();
                }
                quantLocal = quantOutQueue.DeQue<int8_t>();
                quantScaleLocal = quantScaleOutQueue.DeQue<float>();
                quantOutQueue.FreeTensor(quantLocal);
                quantScaleOutQueue.FreeTensor(quantScaleLocal);
            }
        }
        return;
    }

    if constexpr (GROUP_QUANT_DEBUG_STAGE == 5) {
        if ASCEND_IS_AIV {
            const uint64_t totalMmOut = static_cast<uint64_t>(gmmBaseParams->M) * gmmSwigluTensorList->tokenLen;
            for (uint64_t idx = blockIdx; idx < totalMmOut; idx += aivCoreNum) {
                mmOutGM.SetValue(idx, static_cast<int32_t>(1));
            }
            UpdateVecConfig(blockIdx, vecConfig);
            if (blockIdx < vecConfig.usedCoreNum) {
                LocalTensor<int32_t> mmLocal = mmOutQueue.AllocTensor<int32_t>();
                LocalTensor<int8_t> quantLocal = quantOutQueue.AllocTensor<int8_t>();
                LocalTensor<float> quantScaleLocal = quantScaleOutQueue.AllocTensor<float>();
                mmOutQueue.EnQue(mmLocal);
                quantScaleOutQueue.EnQue(quantScaleLocal);
                quantOutQueue.EnQue(quantLocal);
                for (uint32_t outLoopIdx = 0; outLoopIdx < vecConfig.outLoopNum; outLoopIdx++) {
                    vecConfig.innerLoopNum = outLoopIdx == (vecConfig.outLoopNum - 1)
                                                ? vecConfig.tailLoopNum
                                                : gmmSwigluTensorList->maxProcessRowNum;
                    customDataCopyInGroupQuant<CHANNELDTYPE>(outLoopIdx);
                    quantLocal = quantOutQueue.DeQue<int8_t>();
                    quantScaleLocal = quantScaleOutQueue.DeQue<float>();
                    LocalTensor<int32_t> quantLocalI32 = quantLocal.template ReinterpretCast<int32_t>();
                    Duplicate(quantLocalI32, static_cast<int32_t>(0),
                              vecConfig.innerLoopNum * gmmSwigluTensorList->tokenLen / 2 / sizeof(int32_t));
                    Duplicate(quantScaleLocal, static_cast<float>(1.0), vecConfig.innerLoopNum);
                    PipeBarrier<PIPE_V>();
                    quantOutQueue.EnQue(quantLocal);
                    quantScaleOutQueue.EnQue(quantScaleLocal);
                    customDataCopyOut();
                }
                LocalTensor<float> mmLocalOut = mmOutQueue.DeQue<float>();
                quantLocal = quantOutQueue.DeQue<int8_t>();
                quantScaleLocal = quantScaleOutQueue.DeQue<float>();
                mmOutQueue.FreeTensor(mmLocalOut);
                quantScaleOutQueue.FreeTensor(quantScaleLocal);
                quantOutQueue.FreeTensor(quantLocal);
            }
        }
        return;
    }

    if constexpr (GROUP_QUANT_DEBUG_STAGE == 6) {
        const uint64_t totalMmOut = static_cast<uint64_t>(gmmBaseParams->M) * gmmSwigluTensorList->tokenLen;
        if ASCEND_IS_AIC {
            for (uint64_t idx = blockIdx; idx < totalMmOut; idx += aicCoreNum) {
                mmOutGM.SetValue(idx, static_cast<int32_t>(1));
            }
        }
        SyncAll<false>();
        if ASCEND_IS_AIV {
            UpdateVecConfig(blockIdx, vecConfig);
            if (blockIdx < vecConfig.usedCoreNum) {
                LocalTensor<int32_t> mmLocal = mmOutQueue.AllocTensor<int32_t>();
                LocalTensor<int8_t> quantLocal = quantOutQueue.AllocTensor<int8_t>();
                LocalTensor<float> quantScaleLocal = quantScaleOutQueue.AllocTensor<float>();
                mmOutQueue.EnQue(mmLocal);
                quantScaleOutQueue.EnQue(quantScaleLocal);
                quantOutQueue.EnQue(quantLocal);
                for (uint32_t outLoopIdx = 0; outLoopIdx < vecConfig.outLoopNum; outLoopIdx++) {
                    vecConfig.innerLoopNum = outLoopIdx == (vecConfig.outLoopNum - 1)
                                                ? vecConfig.tailLoopNum
                                                : gmmSwigluTensorList->maxProcessRowNum;
                    customDataCopyInGroupQuant<CHANNELDTYPE>(outLoopIdx);
                    quantLocal = quantOutQueue.DeQue<int8_t>();
                    quantScaleLocal = quantScaleOutQueue.DeQue<float>();
                    LocalTensor<int32_t> quantLocalI32 = quantLocal.template ReinterpretCast<int32_t>();
                    Duplicate(quantLocalI32, static_cast<int32_t>(0),
                              vecConfig.innerLoopNum * gmmSwigluTensorList->tokenLen / 2 / sizeof(int32_t));
                    Duplicate(quantScaleLocal, static_cast<float>(1.0), vecConfig.innerLoopNum);
                    PipeBarrier<PIPE_V>();
                    quantOutQueue.EnQue(quantLocal);
                    quantScaleOutQueue.EnQue(quantScaleLocal);
                    customDataCopyOut();
                }
                LocalTensor<float> mmLocalOut = mmOutQueue.DeQue<float>();
                quantLocal = quantOutQueue.DeQue<int8_t>();
                quantScaleLocal = quantScaleOutQueue.DeQue<float>();
                mmOutQueue.FreeTensor(mmLocalOut);
                quantScaleOutQueue.FreeTensor(quantScaleLocal);
                quantOutQueue.FreeTensor(quantLocal);
            }
        }
        return;
    }

    if constexpr (GROUP_QUANT_DEBUG_STAGE == 54) {
        if ASCEND_IS_AIV {
            UpdateVecConfig(blockIdx, vecConfig);
            if (blockIdx < vecConfig.usedCoreNum) {
                LocalTensor<int32_t> mmLocal = mmOutQueue.AllocTensor<int32_t>();
                LocalTensor<int8_t> quantLocal = quantOutQueue.AllocTensor<int8_t>();
                LocalTensor<float> quantScaleLocal = quantScaleOutQueue.AllocTensor<float>();
                mmOutQueue.EnQue(mmLocal);
                quantScaleOutQueue.EnQue(quantScaleLocal);
                quantOutQueue.EnQue(quantLocal);
                for (uint32_t outLoopIdx = 0; outLoopIdx < vecConfig.outLoopNum; outLoopIdx++) {
                    vecConfig.innerLoopNum = outLoopIdx == (vecConfig.outLoopNum - 1)
                                                ? vecConfig.tailLoopNum
                                                : gmmSwigluTensorList->maxProcessRowNum;
                    customDataCopyInGroupQuant<CHANNELDTYPE>(outLoopIdx);
                    for (uint32_t innerLoopIdx = 0; innerLoopIdx < vecConfig.innerLoopNum; innerLoopIdx++) {
                        Swiglu(innerLoopIdx);
                        Quant(innerLoopIdx);
                    }
                    customDataCopyOut();
                }

                LocalTensor<float> mmLocalOut = mmOutQueue.DeQue<float>();
                quantLocal = quantOutQueue.DeQue<int8_t>();
                quantScaleLocal = quantScaleOutQueue.DeQue<float>();
                mmOutQueue.FreeTensor(mmLocalOut);
                quantScaleOutQueue.FreeTensor(quantScaleLocal);
                quantOutQueue.FreeTensor(quantLocal);
            }
            quantScaleOutputGM.SetValue(0, 12345.0F);
        }
        return;
    }

    const uint32_t kGroupNum = GROUP_QUANT_DEBUG_STAGE >= 9 ? gmmSwigluTensorList->groupQuantGroupNum : 1;
    if (kGroupNum == 0) {
        return;
    }
    if ASCEND_IS_AIV {
        if constexpr ((GROUP_QUANT_DEBUG_STAGE >= 1 && GROUP_QUANT_DEBUG_STAGE < 8) ||
                      GROUP_QUANT_DEBUG_STAGE >= 9) {
        PackGroupQuantX();
        }
        if constexpr (GROUP_QUANT_DEBUG_STAGE >= 2) {
        PackGroupQuantScale();
        }
        GlobalTensor<float> accumGM;
        accumGM.SetGlobalBuffer((__gm__ float *)groupQuantAccumPtr,
                                static_cast<uint64_t>(gmmBaseParams->M) * gmmSwigluTensorList->tokenLen);
        for (uint64_t idx = blockIdx; idx < static_cast<uint64_t>(gmmBaseParams->M) * gmmSwigluTensorList->tokenLen;
             idx += aivCoreNum) {
            accumGM.SetValue(idx, static_cast<float>(0.0));
        }
    }
    SyncAll<false>();

    if constexpr (GROUP_QUANT_DEBUG_STAGE == 45) {
        if ASCEND_IS_AIV {
            UpdateVecConfig(blockIdx, vecConfig);
            if (blockIdx < vecConfig.usedCoreNum && vecConfig.taskNum > 0) {
                for (uint32_t row = 0; row < vecConfig.taskNum; ++row) {
                    const float encoded = static_cast<float>(blockIdx * 100000U +
                                                             vecConfig.startIdx * 1000U +
                                                             (vecConfig.curIdx + row) * 10U +
                                                             vecConfig.taskNum);
                    quantScaleOutputGM.SetValue(vecConfig.startIdx + row, encoded);
                }
            }
        }
        SyncAll<false>();
        return;
    }

    if constexpr (GROUP_QUANT_DEBUG_STAGE == 12 || GROUP_QUANT_DEBUG_STAGE == 13 ||
                  GROUP_QUANT_DEBUG_STAGE == 14 || GROUP_QUANT_DEBUG_STAGE == 15 ||
                  GROUP_QUANT_DEBUG_STAGE == 16) {
        const uint32_t probeM = static_cast<uint32_t>(groupListGM.GetValue(0));
        if ASCEND_IS_AIV {
            PackGroupQuantWeight(0);
        }
        SyncAll<false>();
        if (probeM > 0) {
            if ASCEND_IS_AIC {
                MNConfig mnConfig;
                mnConfig.m = probeM;
                mnConfig.k = gmmSwigluTensorList->groupQuantGroupSize;
                if constexpr (GROUP_QUANT_DEBUG_STAGE == 14 || GROUP_QUANT_DEBUG_STAGE == 15 ||
                              GROUP_QUANT_DEBUG_STAGE == 16) {
                    mnConfig.n = gmmBaseParams->N;
                } else {
                    mnConfig.n = gmmBaseParams->N < 32 ? gmmBaseParams->N : 32;
                }
                mnConfig.baseM = BASIC_M;
                mnConfig.baseN = BASIC_N;
                mnConfig.singleM = SINGLE_CORE_M;
                if constexpr (GROUP_QUANT_DEBUG_STAGE == 14 || GROUP_QUANT_DEBUG_STAGE == 15 ||
                              GROUP_QUANT_DEBUG_STAGE == 16) {
                    mnConfig.singleN = SINGLE_CORE_N;
                } else {
                    mnConfig.singleN = 32;
                }
                mnConfig.xBaseOffset = 0;
                mnConfig.yBaseOffset = 0;
                mnConfig.blockDimM = Ceil(mnConfig.m, mnConfig.singleM);
                mnConfig.blockDimN = 1;
                MNBlockIdxCompute(mnConfig, blockIdx, 0, THRESHOLD_BLOCK_NUM);
                if (blockIdx == 0) {
                    const uint32_t probeGroupNum = GROUP_QUANT_DEBUG_STAGE == 16
                                                       ? kGroupNum
                                                       : (GROUP_QUANT_DEBUG_STAGE == 13 && kGroupNum > 1 ? 2 : 1);
                    for (uint32_t probeGroupIdx = 0; probeGroupIdx < probeGroupNum; ++probeGroupIdx) {
                        MMComputeGroupQuant(0, probeGroupIdx, mnConfig, blockIdx);
                    }
                }
            }
        }
        SyncAll<false>();
        if constexpr (GROUP_QUANT_DEBUG_STAGE == 15) {
            if (probeM > 0) {
                if ASCEND_IS_AIV {
                    GlobalTensor<float> accumGM;
                    GlobalTensor<float> scalePackedGM;
                    accumGM.SetGlobalBuffer((__gm__ float *)groupQuantAccumPtr,
                                            static_cast<uint64_t>(gmmBaseParams->M) * gmmSwigluTensorList->tokenLen);
                    scalePackedGM.SetGlobalBuffer((__gm__ float *)groupQuantScalePackedPtr,
                                                  static_cast<uint64_t>(gmmSwigluTensorList->groupListLen) *
                                                  kGroupNum * gmmSwigluTensorList->tokenLen);
                    const uint32_t tokenLen = gmmSwigluTensorList->tokenLen;
                    const uint32_t probeN = gmmBaseParams->N;
                    for (uint64_t linearIdx = blockIdx;
                         linearIdx < static_cast<uint64_t>(probeM) * probeN;
                         linearIdx += aivCoreNum) {
                        const uint32_t rowIdx = static_cast<uint32_t>(linearIdx / probeN);
                        const uint32_t channelIdx = static_cast<uint32_t>(linearIdx - static_cast<uint64_t>(rowIdx) * probeN);
                        const uint64_t outOffset = static_cast<uint64_t>(rowIdx) * tokenLen + channelIdx;
                        const float partial = static_cast<float>(mmOutGM.GetValue(outOffset));
                        const float scale = scalePackedGM.GetValue(channelIdx);
                        const float tokenScale = perTokenScaleGM.GetValue(rowIdx);
                        accumGM.SetValue(outOffset, partial * scale * tokenScale);
                    }
                }
                SyncAll<false>();
            }
        }
        return;
    }

    if constexpr (GROUP_QUANT_DEBUG_STAGE == 17) {
        uint32_t prevProbeSplit = 0;
        for (uint32_t expertIdx = 0; expertIdx < gmmSwigluTensorList->groupListLen; ++expertIdx) {
            const uint32_t currProbeSplit = static_cast<uint32_t>(groupListGM.GetValue(expertIdx));
            const uint32_t probeM = currProbeSplit - prevProbeSplit;
            if ASCEND_IS_AIV {
                PackGroupQuantWeight(expertIdx);
            }
            SyncAll<false>();
            if (probeM > 0) {
                if ASCEND_IS_AIC {
                    MNConfig mnConfig;
                    mnConfig.m = probeM;
                    mnConfig.k = gmmSwigluTensorList->groupQuantGroupSize;
                    mnConfig.n = gmmBaseParams->N;
                    mnConfig.baseM = BASIC_M;
                    mnConfig.baseN = BASIC_N;
                    mnConfig.singleM = SINGLE_CORE_M;
                    mnConfig.singleN = SINGLE_CORE_N;
                    mnConfig.xBaseOffset = static_cast<int64_t>(prevProbeSplit) * gmmBaseParams->K;
                    mnConfig.yBaseOffset = static_cast<int64_t>(prevProbeSplit) * mnConfig.n;
                    mnConfig.blockDimM = Ceil(mnConfig.m, mnConfig.singleM);
                    mnConfig.blockDimN = 1;
                    MNBlockIdxCompute(mnConfig, blockIdx, 0, THRESHOLD_BLOCK_NUM);
                    if (blockIdx == 0) {
                        MMComputeGroupQuant(expertIdx, 0, mnConfig, blockIdx);
                    }
                }
            }
            SyncAll<false>();
            prevProbeSplit = currProbeSplit;
        }
        return;
    }

    if constexpr (GROUP_QUANT_DEBUG_STAGE == 46 || GROUP_QUANT_DEBUG_STAGE == 48) {
        const uint32_t logicalK = gmmSwigluTensorList->groupQuantGroupSize;
        const uint32_t n = gmmBaseParams->N;
        const uint32_t alignedN = AlignUp<32>(n);
        const uint32_t alignedK = AlignUp<16>(logicalK);
        if ASCEND_IS_AIV {
            GlobalTensor<int8_t> xPackedGM;
            GlobalTensor<int8_t> weightPackedGM;
            xPackedGM.SetGlobalBuffer((__gm__ int8_t *)groupQuantXPackedPtr,
                                      static_cast<uint64_t>(gmmBaseParams->M) * logicalK);
            weightPackedGM.SetGlobalBuffer((__gm__ int8_t *)groupQuantWeightPackedPtr,
                                           static_cast<uint64_t>(alignedN) * alignedK);
            if constexpr (GROUP_QUANT_DEBUG_STAGE == 48) {
                constexpr uint32_t fillElems = 1024;
                pipe->Reset();
                pipe->InitBuffer(groupQuantPackWorkspace, fillElems * sizeof(int8_t));
                LocalTensor<int8_t> fillLocal = groupQuantPackWorkspace.Get<int8_t>();
                Duplicate(fillLocal, static_cast<int8_t>(1), fillElems);
                PipeBarrier<PIPE_V>();
                const uint64_t xTotal = static_cast<uint64_t>(gmmBaseParams->M) * logicalK;
                if (blockIdx == 0) {
                    DataCopyParams xCopyParams{1, static_cast<uint16_t>(xTotal * sizeof(int8_t) / UB_BLOCK_UNIT_SIZE),
                                               0, 0};
                    DataCopy(xPackedGM[0], fillLocal, xCopyParams);
                }
                DataCopyParams copyParams{1, static_cast<uint16_t>(fillElems * sizeof(int8_t) / UB_BLOCK_UNIT_SIZE),
                                          0, 0};
                const uint64_t wTotal = static_cast<uint64_t>(alignedN) * alignedK;
                for (uint64_t offset = static_cast<uint64_t>(blockIdx) * fillElems;
                     offset < wTotal; offset += static_cast<uint64_t>(aivCoreNum) * fillElems) {
                    DataCopy(weightPackedGM[offset], fillLocal, copyParams);
                }
            } else {
                for (uint64_t idx = blockIdx; idx < static_cast<uint64_t>(gmmBaseParams->M) * logicalK;
                     idx += aivCoreNum) {
                    xPackedGM.SetValue(idx, static_cast<int8_t>(1));
                }
                for (uint64_t idx = blockIdx; idx < static_cast<uint64_t>(alignedN) * alignedK;
                     idx += aivCoreNum) {
                    weightPackedGM.SetValue(idx, static_cast<int8_t>(1));
                }
            }
        }
        SyncAll<false>();
        if ASCEND_IS_AIC {
            MNConfig mnConfig;
            mnConfig.m = 1;
            mnConfig.k = logicalK;
            mnConfig.n = n;
            mnConfig.baseM = BASIC_M;
            mnConfig.baseN = BASIC_N;
            mnConfig.singleM = SINGLE_CORE_M;
            mnConfig.singleN = SINGLE_CORE_N;
            mnConfig.xBaseOffset = 0;
            mnConfig.yBaseOffset = 0;
            mnConfig.blockDimM = 1;
            mnConfig.blockDimN = 1;
            MNBlockIdxCompute(mnConfig, 0, 0, THRESHOLD_BLOCK_NUM);
            MMComputeGroupQuant(0, 0, mnConfig, blockIdx);
        }
        SyncAll<false>();
        if ASCEND_IS_AIV {
            pipe->Reset();
            pipe->InitBuffer(groupQuantPackWorkspace, n * sizeof(int32_t));
            LocalTensor<int32_t> partialLocal = groupQuantPackWorkspace.Get<int32_t>();
            DataCopyParams copyParams{1, static_cast<uint16_t>(n * sizeof(int32_t) / UB_BLOCK_UNIT_SIZE), 0, 0};
            if (blockIdx == 0) {
                DataCopy(partialLocal, mmOutGM[0], copyParams);
                PipeBarrier<PIPE_ALL>();
                quantScaleOutputGM.SetValue(0, static_cast<float>(partialLocal.GetValue(0)) / QUANT_SCALE_INT8);
            }
        }
        return;
    }

    if constexpr (GROUP_QUANT_DEBUG_STAGE == 47) {
        if ASCEND_IS_AIC {
            quantScaleOutputGM.SetValue(0, 1.0F);
        }
        SyncAll<false>();
        return;
    }

    if constexpr (GROUP_QUANT_DEBUG_STAGE == 49) {
        const uint32_t logicalK = gmmSwigluTensorList->groupQuantGroupSize;
        const uint32_t n = gmmBaseParams->N;
        if ASCEND_IS_AIV {
            PackGroupQuantX();
            PackGroupQuantWeight(0);
        }
        SyncAll<false>();
        if ASCEND_IS_AIC {
            MNConfig mnConfig;
            mnConfig.m = 1;
            mnConfig.k = logicalK;
            mnConfig.n = n;
            mnConfig.baseM = BASIC_M;
            mnConfig.baseN = BASIC_N;
            mnConfig.singleM = SINGLE_CORE_M;
            mnConfig.singleN = SINGLE_CORE_N;
            mnConfig.xBaseOffset = 0;
            mnConfig.yBaseOffset = 0;
            mnConfig.blockDimM = 1;
            mnConfig.blockDimN = 1;
            MNBlockIdxCompute(mnConfig, 0, 0, THRESHOLD_BLOCK_NUM);
            MMComputeGroupQuant(0, 0, mnConfig, blockIdx);
        }
        SyncAll<false>();
        if ASCEND_IS_AIV {
            pipe->Reset();
            pipe->InitBuffer(groupQuantPackWorkspace, n * sizeof(int32_t));
            LocalTensor<int32_t> partialLocal = groupQuantPackWorkspace.Get<int32_t>();
            DataCopyParams copyParams{1, static_cast<uint16_t>(n * sizeof(int32_t) / UB_BLOCK_UNIT_SIZE), 0, 0};
            if (blockIdx == 0) {
                DataCopy(partialLocal, mmOutGM[0], copyParams);
                PipeBarrier<PIPE_ALL>();
                quantScaleOutputGM.SetValue(0, static_cast<float>(partialLocal.GetValue(0)) / QUANT_SCALE_INT8);
            }
        }
        return;
    }

    if constexpr (GROUP_QUANT_DEBUG_STAGE == 57) {
        // Minimal matmul-only probe for M>1: PackX -> PackWeight -> MMComputeGroupQuant(m=M, kGroup0)
        // then DataCopy mmOutGM rows back to verify partial results are non-zero.
        const uint32_t logicalK = gmmSwigluTensorList->groupQuantGroupSize;
        const uint32_t n = gmmBaseParams->N;
        const uint32_t probeMM = gmmBaseParams->M;
        const uint32_t kGroupNum57 = gmmSwigluTensorList->groupQuantGroupNum;
        if ASCEND_IS_AIV {
            PackGroupQuantScale();
        }
        SyncAll<false>();
        // TEST: wrap everything in expert loop like stage 56
        uint32_t prevSplit57 = 0;
        for (uint32_t expertIdx = 0; expertIdx < gmmSwigluTensorList->groupListLen; ++expertIdx) {
            const uint32_t currSplit57 = static_cast<uint32_t>(groupListGM.GetValue(expertIdx));
            const uint32_t probeM57 = currSplit57 - prevSplit57;
            if (probeM57 > 0) {
                MNConfig mnConfig;
                mnConfig.m = probeM57;
                mnConfig.k = logicalK;
                mnConfig.n = n;
                mnConfig.baseM = BASIC_M;
                mnConfig.baseN = BASIC_N;
                mnConfig.singleM = SINGLE_CORE_M;
                mnConfig.singleN = SINGLE_CORE_N;
                mnConfig.xBaseOffset = static_cast<int64_t>(prevSplit57) * gmmBaseParams->K;
                mnConfig.yBaseOffset = static_cast<int64_t>(prevSplit57) * n;
                mnConfig.blockDimM = Ceil(mnConfig.m, mnConfig.singleM);
                mnConfig.blockDimN = 1;
                if ASCEND_IS_AIV {
                    PackGroupQuantX();
                    PackGroupQuantWeight(expertIdx);
                }
                SyncAll<false>();
                if ASCEND_IS_AIC {
                    for (uint32_t kGroupIdx = 0; kGroupIdx < kGroupNum57; ++kGroupIdx) {
                        MNBlockIdxCompute(mnConfig, 0, 0, THRESHOLD_BLOCK_NUM);
                        MMComputeGroupQuant(expertIdx, kGroupIdx, mnConfig, blockIdx);
                    }
                }
                SyncAll<false>();
                if ASCEND_IS_AIV {
                    if (blockIdx == 0) {
                        pipe->Reset();
                        pipe->InitBuffer(groupQuantPackWorkspace, probeMM * n * sizeof(int32_t));
                        LocalTensor<int32_t> partialLocal = groupQuantPackWorkspace.Get<int32_t>();
                        DataCopyParams copyParams{
                            1, static_cast<uint16_t>(probeM57 * n * sizeof(int32_t) / UB_BLOCK_UNIT_SIZE), 0, 0};
                        DataCopy(partialLocal, mmOutGM[static_cast<uint64_t>(prevSplit57) * n], copyParams);
                        PipeBarrier<PIPE_ALL>();
                        quantScaleOutputGM.SetValue(0, static_cast<float>(partialLocal.GetValue(0)));
                        if (probeM57 > 1) {
                            quantScaleOutputGM.SetValue(1, static_cast<float>(partialLocal.GetValue(n)));
                        }
                    }
                }
                SyncAll<false>();
            }
            prevSplit57 = currSplit57;
        }
        return;
    }

    if constexpr (GROUP_QUANT_DEBUG_STAGE == 50) {
        const uint32_t tokenLen = gmmSwigluTensorList->tokenLen;
        if ASCEND_IS_AIV {
            PackGroupQuantScale();
        }
        SyncAll<false>();
        if ASCEND_IS_AIV {
            GlobalTensor<float> scalePackedGM;
            scalePackedGM.SetGlobalBuffer((__gm__ float *)groupQuantScalePackedPtr,
                                          static_cast<uint64_t>(gmmSwigluTensorList->groupListLen) *
                                          gmmSwigluTensorList->groupQuantGroupNum * tokenLen);
            pipe->Reset();
            pipe->InitBuffer(groupQuantScaleWorkspace, tokenLen * sizeof(float));
            LocalTensor<float> scaleLocal = groupQuantScaleWorkspace.Get<float>();
            DataCopyParams copyParams{1, static_cast<uint16_t>(tokenLen * sizeof(float) / UB_BLOCK_UNIT_SIZE),
                                      0, 0};
            if (blockIdx == 0) {
                DataCopy(scaleLocal, scalePackedGM[0], copyParams);
                PipeBarrier<PIPE_ALL>();
                quantScaleOutputGM.SetValue(0, scaleLocal.GetValue(0));
            }
        }
        return;
    }

    if constexpr (GROUP_QUANT_DEBUG_STAGE == 55) {
        const uint32_t tokenLen = gmmSwigluTensorList->tokenLen;
        const uint32_t kGroupNum = gmmSwigluTensorList->groupQuantGroupNum;
        if ASCEND_IS_AIV {
            if (blockIdx == 0) {
                for (uint32_t idx = 0; idx < 16 && idx < gmmBaseParams->M * tokenLen / 2; ++idx) {
                    quantOutputGM.SetValue(idx, static_cast<int8_t>(0));
                }
                GlobalTensor<float> accumGM;
                accumGM.SetGlobalBuffer((__gm__ float *)groupQuantAccumPtr,
                                        static_cast<uint64_t>(gmmBaseParams->M) * tokenLen);
                for (uint32_t rowIdx = 0; rowIdx < gmmBaseParams->M; ++rowIdx) {
                    accumGM.SetValue(static_cast<uint64_t>(rowIdx) * tokenLen, 0.0F);
                }
            }
        }
        SyncAll<false>();
        uint32_t prevProbeSplit = 0;
        for (uint32_t expertIdx = 0; expertIdx < gmmSwigluTensorList->groupListLen; ++expertIdx) {
            const uint32_t currProbeSplit = static_cast<uint32_t>(groupListGM.GetValue(expertIdx));
            const uint32_t probeM = currProbeSplit - prevProbeSplit;
            if ASCEND_IS_AIV {
                PackGroupQuantWeight(expertIdx);
            }
            SyncAll<false>();
            if (probeM > 0) {
                MNConfig mnConfig;
                mnConfig.m = probeM;
                mnConfig.k = gmmSwigluTensorList->groupQuantGroupSize;
                mnConfig.n = gmmBaseParams->N;
                mnConfig.baseM = BASIC_M;
                mnConfig.baseN = BASIC_N;
                mnConfig.singleM = SINGLE_CORE_M;
                mnConfig.singleN = mnConfig.n;
                mnConfig.xBaseOffset = static_cast<int64_t>(prevProbeSplit) * gmmBaseParams->K;
                mnConfig.yBaseOffset = static_cast<int64_t>(prevProbeSplit) * mnConfig.n;
                mnConfig.blockDimM = Ceil(mnConfig.m, mnConfig.singleM);
                mnConfig.blockDimN = 1;
                for (uint32_t kGroupIdx = 0; kGroupIdx < kGroupNum; ++kGroupIdx) {
                    if ASCEND_IS_AIC {
                        MNBlockIdxCompute(mnConfig, 0, 0, THRESHOLD_BLOCK_NUM);
                        if (blockIdx == 0) {
                            MMComputeGroupQuant(expertIdx, kGroupIdx, mnConfig, blockIdx);
                        }
                    }
                    SyncAll<false>();
                    if ASCEND_IS_AIV {
                        if (blockIdx == 0) {
                            GlobalTensor<float> accumGM;
                            GlobalTensor<float> scalePackedGM;
                            accumGM.SetGlobalBuffer((__gm__ float *)groupQuantAccumPtr,
                                                    static_cast<uint64_t>(gmmBaseParams->M) * tokenLen);
                            scalePackedGM.SetGlobalBuffer((__gm__ float *)groupQuantScalePackedPtr,
                                                          static_cast<uint64_t>(gmmSwigluTensorList->groupListLen) *
                                                          kGroupNum * tokenLen);
                            const uint32_t globalRow = prevProbeSplit;
                            const uint64_t partialBase =
                                static_cast<uint64_t>(kGroupIdx) * gmmBaseParams->M * mnConfig.n;
                            const uint64_t partialOffset =
                                partialBase + static_cast<uint64_t>(globalRow) * mnConfig.n;
                            const int32_t partial = mmOutGM.GetValue(partialOffset);
                            const uint64_t scaleOffset =
                                (static_cast<uint64_t>(expertIdx) * kGroupNum + kGroupIdx) * tokenLen;
                            const float scale = scalePackedGM.GetValue(scaleOffset);
                            const float tokenScale = perTokenScaleGM.GetValue(globalRow);
                            const uint64_t accumOffset = static_cast<uint64_t>(globalRow) * tokenLen;
                            const float oldAccum = accumGM.GetValue(accumOffset);
                            const float newAccum = oldAccum + static_cast<float>(partial) * scale * tokenScale;
                            accumGM.SetValue(accumOffset, newAccum);
                        }
                    }
                    SyncAll<false>();
                }
                if ASCEND_IS_AIV {
                    if (blockIdx == 0) {
                        GlobalTensor<float> accumGM;
                        accumGM.SetGlobalBuffer((__gm__ float *)groupQuantAccumPtr,
                                                static_cast<uint64_t>(gmmBaseParams->M) * tokenLen);
                        const uint32_t globalRow = prevProbeSplit;
                        const float accum = accumGM.GetValue(static_cast<uint64_t>(globalRow) * tokenLen);
                        quantScaleOutputGM.SetValue(globalRow, accum);
                        int32_t quantValue = static_cast<int32_t>(accum >= 0.0F ? accum + 0.5F : accum - 0.5F);
                        if (quantValue > 127) {
                            quantValue = 127;
                        }
                        if (quantValue < -128) {
                            quantValue = -128;
                        }
                        quantOutputGM.SetValue(expertIdx, static_cast<int8_t>(quantValue));
                    }
                }
                SyncAll<false>();
            }
            prevProbeSplit = currProbeSplit;
        }
        return;
    }

    if constexpr (GROUP_QUANT_DEBUG_STAGE == 8) {
        const uint32_t tokenLen = gmmSwigluTensorList->tokenLen;
        const uint32_t kGroupNum = gmmSwigluTensorList->groupQuantGroupNum;
        if ASCEND_IS_AIV {
            PackGroupQuantScale();  // pack per-channel scale into [expert*kGroup][tokenLen] layout
        }
        SyncAll<false>();
        uint32_t prevExpertEnd = 0;
        for (uint32_t expertIdx = 0; expertIdx < gmmSwigluTensorList->groupListLen; ++expertIdx) {
            const uint32_t expertEnd = static_cast<uint32_t>(groupListGM.GetValue(expertIdx));
            const uint32_t expertM = expertEnd - prevExpertEnd;
            if (expertM > 0) {
                MNConfig mnConfig;
                mnConfig.m = expertM;
                mnConfig.k = gmmSwigluTensorList->groupQuantGroupSize;
                mnConfig.n = gmmBaseParams->N;
                mnConfig.baseM = BASIC_M;
                mnConfig.baseN = BASIC_N;
                mnConfig.singleM = SINGLE_CORE_M;
                mnConfig.singleN = SINGLE_CORE_N;  // match stage 57
                mnConfig.xBaseOffset = static_cast<int64_t>(prevExpertEnd) * gmmBaseParams->K;
                mnConfig.yBaseOffset = static_cast<int64_t>(prevExpertEnd) * mnConfig.n;
                mnConfig.blockDimM = Ceil(mnConfig.m, mnConfig.singleM);
                mnConfig.blockDimN = 1;
                // Pack X and weight together in one AIV block (mirror stage 57 ordering)
                if ASCEND_IS_AIV {
                    PackGroupQuantX();
                    PackGroupQuantWeight(expertIdx);
                }
                SyncAll<false>();
                // Phase 1: run ALL kGroup matmuls first (each writes a distinct mmOutGM region).
                // Do NOT interleave AIV reads/pipe->Reset here - that corrupts the shared matmul pipe.
                if ASCEND_IS_AIC {
                    for (uint32_t kGroupIdx = 0; kGroupIdx < kGroupNum; ++kGroupIdx) {
                        const uint32_t mnBlockCount = mnConfig.blockDimM * mnConfig.blockDimN;
                        const uint32_t thresholdMdimN = THRESHOLD_BLOCK_NUM * mnConfig.blockDimN;
                        for (uint32_t mnBlock = blockIdx; mnBlock < mnBlockCount; mnBlock += aicCoreNum) {
                            MNBlockIdxCompute(mnConfig, mnBlock, 0, thresholdMdimN);
                            MMComputeGroupQuant(expertIdx, kGroupIdx, mnConfig, blockIdx);
                        }
                    }
                }
                SyncAll<false>();
                if ASCEND_IS_AIV {
                    if (blockIdx == 0) {
                        const uint32_t halfTokenLen = tokenLen / BISECT;
                        constexpr uint32_t ubBudgetBytes = 160 * 1024;
                        const uint32_t fixedBytes = tokenLen * sizeof(float) +
                                                    (AlignUp<8>(halfTokenLen) +
                                                     FLOAT_UB_BLOCK_UNIT_SIZE * 2) * sizeof(float) +
                                                    UB_BLOCK_UNIT_SIZE;
                        const uint32_t bytesPerRow = tokenLen * (sizeof(int32_t) + sizeof(float) * 2) +
                                                     halfTokenLen * sizeof(int8_t) + sizeof(float);
                        uint32_t tileRows = fixedBytes < ubBudgetBytes
                                                ? (ubBudgetBytes - fixedBytes) / bytesPerRow
                                                : 1;
                        tileRows = tileRows == 0 ? 1 : tileRows;
                        tileRows = tileRows > gmmSwigluTensorList->maxProcessRowNum
                                       ? gmmSwigluTensorList->maxProcessRowNum
                                       : tileRows;
                        for (uint32_t tileOffset = 0; tileOffset < expertM; tileOffset += tileRows) {
                            const uint32_t tileM = tileOffset + tileRows < expertM ? tileRows : expertM - tileOffset;
                            ProcessGroupQuantTile(expertIdx, prevExpertEnd + tileOffset, tileM, mnConfig);
                        }
                    }
                }
                SyncAll<false>();
            }
            prevExpertEnd = expertEnd;
        }
        SyncAll<false>();
        return;
    }

    if constexpr (GROUP_QUANT_DEBUG_STAGE == 18) {
        uint32_t prevProbeSplit = 0;
        for (uint32_t expertIdx = 0; expertIdx < gmmSwigluTensorList->groupListLen; ++expertIdx) {
            const uint32_t currProbeSplit = static_cast<uint32_t>(groupListGM.GetValue(expertIdx));
            const uint32_t probeM = currProbeSplit - prevProbeSplit;
            if ASCEND_IS_AIV {
                PackGroupQuantWeight(expertIdx);
            }
            SyncAll<false>();
            if (probeM > 0) {
                if ASCEND_IS_AIC {
                    MNConfig mnConfig;
                    mnConfig.m = probeM;
                    mnConfig.k = gmmSwigluTensorList->groupQuantGroupSize;
                    mnConfig.n = gmmBaseParams->N;
                    mnConfig.baseM = BASIC_M;
                    mnConfig.baseN = BASIC_N;
                    mnConfig.singleM = SINGLE_CORE_M;
                    mnConfig.singleN = SINGLE_CORE_N;
                    mnConfig.xBaseOffset = static_cast<int64_t>(prevProbeSplit) * gmmBaseParams->K;
                    mnConfig.yBaseOffset = static_cast<int64_t>(prevProbeSplit) * mnConfig.n;
                    mnConfig.blockDimM = Ceil(mnConfig.m, mnConfig.singleM);
                    mnConfig.blockDimN = 1;
                    MNBlockIdxCompute(mnConfig, blockIdx, 0, THRESHOLD_BLOCK_NUM);
                    if (blockIdx == 0) {
                        for (uint32_t kGroupIdx = 0; kGroupIdx < kGroupNum; ++kGroupIdx) {
                            MMComputeGroupQuant(expertIdx, kGroupIdx, mnConfig, blockIdx);
                        }
                    }
                }
            }
            SyncAll<false>();
            prevProbeSplit = currProbeSplit;
        }
        return;
    }

    if constexpr (GROUP_QUANT_DEBUG_STAGE == 19 || GROUP_QUANT_DEBUG_STAGE == 20 ||
                  GROUP_QUANT_DEBUG_STAGE == 21 || GROUP_QUANT_DEBUG_STAGE == 22 ||
                  GROUP_QUANT_DEBUG_STAGE == 23 || GROUP_QUANT_DEBUG_STAGE == 25 ||
                  GROUP_QUANT_DEBUG_STAGE == 26 || GROUP_QUANT_DEBUG_STAGE == 43 ||
                  GROUP_QUANT_DEBUG_STAGE == 44 || GROUP_QUANT_DEBUG_STAGE == 45 ||
                  GROUP_QUANT_DEBUG_STAGE == 54 || GROUP_QUANT_DEBUG_STAGE == 56) {
        uint32_t prevProbeSplit = 0;
        for (uint32_t expertIdx = 0; expertIdx < gmmSwigluTensorList->groupListLen; ++expertIdx) {
            const uint32_t currProbeSplit = static_cast<uint32_t>(groupListGM.GetValue(expertIdx));
            const uint32_t probeM = currProbeSplit - prevProbeSplit;
            if ASCEND_IS_AIV {
                PackGroupQuantWeight(expertIdx);
            }
            SyncAll<false>();
            if (probeM > 0) {
                MNConfig mnConfig;
                mnConfig.m = probeM;
                mnConfig.k = gmmSwigluTensorList->groupQuantGroupSize;
                mnConfig.n = gmmBaseParams->N;
                mnConfig.baseM = BASIC_M;
                mnConfig.baseN = BASIC_N;
                mnConfig.singleM = SINGLE_CORE_M;
                if constexpr (GROUP_QUANT_DEBUG_STAGE == 21 || GROUP_QUANT_DEBUG_STAGE == 22 ||
                              GROUP_QUANT_DEBUG_STAGE == 43 || GROUP_QUANT_DEBUG_STAGE == 54 ||
                              GROUP_QUANT_DEBUG_STAGE == 56) {
                    mnConfig.singleN = mnConfig.n;
                } else {
                    mnConfig.singleN = SINGLE_CORE_N;
                }
                mnConfig.xBaseOffset = static_cast<int64_t>(prevProbeSplit) * gmmBaseParams->K;
                mnConfig.yBaseOffset = static_cast<int64_t>(prevProbeSplit) * mnConfig.n;
                mnConfig.blockDimM = Ceil(mnConfig.m, mnConfig.singleM);
                mnConfig.blockDimN = Ceil(mnConfig.n, mnConfig.singleN);
                MNBlockIdxCompute(mnConfig, blockIdx, 0, THRESHOLD_BLOCK_NUM);
                if ASCEND_IS_AIV {
                    GlobalTensor<float> accumGM;
                    accumGM.SetGlobalBuffer((__gm__ float *)groupQuantAccumPtr,
                                            static_cast<uint64_t>(gmmBaseParams->M) * gmmSwigluTensorList->tokenLen);
                    const uint32_t tokenLen = gmmSwigluTensorList->tokenLen;
                    const uint64_t totalElements = static_cast<uint64_t>(probeM) * mnConfig.n;
                    const uint64_t zeroStart = (GROUP_QUANT_DEBUG_STAGE == 21 || GROUP_QUANT_DEBUG_STAGE == 22 ||
                                                GROUP_QUANT_DEBUG_STAGE == 43 ||
                                                GROUP_QUANT_DEBUG_STAGE == 54 ||
                                                GROUP_QUANT_DEBUG_STAGE == 56) ? 0 : blockIdx;
                    const uint64_t zeroStep = (GROUP_QUANT_DEBUG_STAGE == 21 || GROUP_QUANT_DEBUG_STAGE == 22 ||
                                               GROUP_QUANT_DEBUG_STAGE == 43 ||
                                               GROUP_QUANT_DEBUG_STAGE == 54 ||
                                               GROUP_QUANT_DEBUG_STAGE == 56) ? 1 : aivCoreNum;
                    if ((GROUP_QUANT_DEBUG_STAGE != 21 && GROUP_QUANT_DEBUG_STAGE != 22 &&
                         GROUP_QUANT_DEBUG_STAGE != 43 && GROUP_QUANT_DEBUG_STAGE != 54 &&
                         GROUP_QUANT_DEBUG_STAGE != 56) || blockIdx == 0) {
                    for (uint64_t linearIdx = zeroStart; linearIdx < totalElements; linearIdx += zeroStep) {
                        const uint32_t rowLocal = static_cast<uint32_t>(linearIdx / mnConfig.n);
                        const uint32_t channelIdx = static_cast<uint32_t>(linearIdx - static_cast<uint64_t>(rowLocal) * mnConfig.n);
                        const uint32_t globalRow = prevProbeSplit + rowLocal;
                        const uint64_t outOffset = static_cast<uint64_t>(globalRow) * tokenLen + channelIdx;
                        accumGM.SetValue(outOffset, 0.0F);
                    }
                    }
                }
                SyncAll<false>();
                for (uint32_t kGroupIdx = 0; kGroupIdx < kGroupNum; ++kGroupIdx) {
                    if ASCEND_IS_AIC {
                        if constexpr (GROUP_QUANT_DEBUG_STAGE == 25 || GROUP_QUANT_DEBUG_STAGE == 26) {
                            const uint32_t tokenLen = gmmSwigluTensorList->tokenLen;
                            const uint64_t partialBase =
                                static_cast<uint64_t>(kGroupIdx) * gmmBaseParams->M * tokenLen;
                            const uint64_t totalElements = static_cast<uint64_t>(probeM) * mnConfig.n;
                            for (uint64_t linearIdx = 0; linearIdx < totalElements; ++linearIdx) {
                                const uint32_t rowLocal = static_cast<uint32_t>(linearIdx / mnConfig.n);
                                const uint32_t channelIdx =
                                    static_cast<uint32_t>(linearIdx - static_cast<uint64_t>(rowLocal) * mnConfig.n);
                                const uint32_t globalRow = prevProbeSplit + rowLocal;
                                const uint64_t outOffset = static_cast<uint64_t>(globalRow) * tokenLen + channelIdx;
                                mmOutGM.SetValue(partialBase + outOffset, static_cast<int32_t>(1024));
                            }
                        } else if constexpr (GROUP_QUANT_DEBUG_STAGE == 23) {
                            MNBlockIdxCompute(mnConfig, 0, 0, THRESHOLD_BLOCK_NUM);
                            MMComputeGroupQuant(expertIdx, kGroupIdx, mnConfig, blockIdx);
                        } else if constexpr (GROUP_QUANT_DEBUG_STAGE == 44 || GROUP_QUANT_DEBUG_STAGE == 45) {
                            MNBlockIdxCompute(mnConfig, 0, 0, THRESHOLD_BLOCK_NUM);
                            const uint32_t tokenLen = gmmSwigluTensorList->tokenLen;
                            const uint32_t tailN = mnConfig.nIdx * mnConfig.singleN;
                            const uint64_t outOffset =
                                static_cast<uint64_t>(kGroupIdx) * gmmBaseParams->M * mnConfig.n +
                                mnConfig.yBaseOffset +
                                static_cast<uint64_t>(mnConfig.mIdx) * mnConfig.singleM * mnConfig.n +
                                tailN;
                            for (uint32_t rowLocal = 0; rowLocal < static_cast<uint32_t>(probeM); ++rowLocal) {
                                const uint64_t rowOffset = outOffset + static_cast<uint64_t>(rowLocal) * mnConfig.n;
                                for (uint32_t channelIdx = 0; channelIdx < tokenLen; ++channelIdx) {
                                    mmOutGM.SetValue(rowOffset + channelIdx, static_cast<int32_t>(1024));
                                }
                            }
                        } else if constexpr (GROUP_QUANT_DEBUG_STAGE == 21 || GROUP_QUANT_DEBUG_STAGE == 22 ||
                                             GROUP_QUANT_DEBUG_STAGE == 43 ||
                                             GROUP_QUANT_DEBUG_STAGE == 54 ||
                                             GROUP_QUANT_DEBUG_STAGE == 56) {
                            for (uint32_t nBlockIdx = 0; nBlockIdx < mnConfig.blockDimN; ++nBlockIdx) {
                                MNBlockIdxCompute(mnConfig, nBlockIdx, 0, THRESHOLD_BLOCK_NUM);
                                MMComputeGroupQuant(expertIdx, kGroupIdx, mnConfig, blockIdx);
                            }
                        } else if (blockIdx == 0) {
                            MMComputeGroupQuant(expertIdx, kGroupIdx, mnConfig, blockIdx);
                        }
                    }
                    SyncAll<false>();
                    if ASCEND_IS_AIV {
                        GlobalTensor<float> accumGM;
                        GlobalTensor<float> scalePackedGM;
                        accumGM.SetGlobalBuffer((__gm__ float *)groupQuantAccumPtr,
                                                static_cast<uint64_t>(gmmBaseParams->M) * gmmSwigluTensorList->tokenLen);
                        scalePackedGM.SetGlobalBuffer((__gm__ float *)groupQuantScalePackedPtr,
                                                      static_cast<uint64_t>(gmmSwigluTensorList->groupListLen) *
                                                      kGroupNum * gmmSwigluTensorList->tokenLen);
                        const uint32_t tokenLen = gmmSwigluTensorList->tokenLen;
                        const uint64_t expertScaleBase =
                            (static_cast<uint64_t>(expertIdx) * kGroupNum + kGroupIdx) * tokenLen;
                        const uint64_t partialBase =
                            static_cast<uint64_t>(kGroupIdx) * gmmBaseParams->M * tokenLen;
                        const uint64_t totalElements = static_cast<uint64_t>(probeM) * mnConfig.n;
                        if constexpr (GROUP_QUANT_DEBUG_STAGE == 21 || GROUP_QUANT_DEBUG_STAGE == 22 ||
                                      GROUP_QUANT_DEBUG_STAGE == 43 || GROUP_QUANT_DEBUG_STAGE == 44 ||
                                      GROUP_QUANT_DEBUG_STAGE == 54 ||
                                      GROUP_QUANT_DEBUG_STAGE == 56) {
                            pipe->Reset();
                            pipe->InitBuffer(groupQuantPackWorkspace, tokenLen * sizeof(int32_t));
                            pipe->InitBuffer(groupQuantAccumWorkspace, tokenLen * sizeof(float));
                            pipe->InitBuffer(groupQuantScaleWorkspace, tokenLen * sizeof(float));
                            LocalTensor<int32_t> partialLocalInt = groupQuantPackWorkspace.Get<int32_t>();
                            LocalTensor<float> accumLocal = groupQuantAccumWorkspace.Get<float>();
                            LocalTensor<float> scaleLocal = groupQuantScaleWorkspace.Get<float>();
                            DataCopyParams intCopyParams{1, static_cast<uint16_t>(tokenLen * sizeof(int32_t) /
                                                                                  UB_BLOCK_UNIT_SIZE), 0, 0};
                            DataCopyParams floatCopyParams{1, static_cast<uint16_t>(tokenLen * sizeof(float) /
                                                                                    UB_BLOCK_UNIT_SIZE), 0, 0};
                            const uint32_t rowStart = (GROUP_QUANT_DEBUG_STAGE == 21 || GROUP_QUANT_DEBUG_STAGE == 22 ||
                                                       GROUP_QUANT_DEBUG_STAGE == 43 ||
                                                       GROUP_QUANT_DEBUG_STAGE == 54 ||
                                                       GROUP_QUANT_DEBUG_STAGE == 56) ? 0 : blockIdx;
                            const uint32_t rowStep = (GROUP_QUANT_DEBUG_STAGE == 21 || GROUP_QUANT_DEBUG_STAGE == 22 ||
                                                      GROUP_QUANT_DEBUG_STAGE == 43 ||
                                                      GROUP_QUANT_DEBUG_STAGE == 54 ||
                                                      GROUP_QUANT_DEBUG_STAGE == 56) ? 1 : aivCoreNum;
                            if ((GROUP_QUANT_DEBUG_STAGE != 21 && GROUP_QUANT_DEBUG_STAGE != 22 &&
                                 GROUP_QUANT_DEBUG_STAGE != 43 && GROUP_QUANT_DEBUG_STAGE != 54 &&
                                 GROUP_QUANT_DEBUG_STAGE != 56) || blockIdx == 0) {
                            for (uint32_t rowLocal = rowStart; rowLocal < static_cast<uint32_t>(probeM);
                                 rowLocal += rowStep) {
                                const uint32_t globalRow = prevProbeSplit + rowLocal;
                                const uint64_t outOffset = static_cast<uint64_t>(globalRow) * tokenLen;
                                if (kGroupIdx == 0) {
                                    Duplicate(accumLocal, static_cast<float>(0.0F), tokenLen);
                                    PipeBarrier<PIPE_V>();
                                } else {
                                    DataCopy(accumLocal, accumGM[outOffset], floatCopyParams);
                                    PipeBarrier<PIPE_ALL>();
                                }
                                DataCopy(partialLocalInt, mmOutGM[partialBase + outOffset], intCopyParams);
                                DataCopy(scaleLocal, scalePackedGM[expertScaleBase], floatCopyParams);
                                PipeBarrier<PIPE_ALL>();
                                const float tokenScale = perTokenScaleGM.GetValue(globalRow);
                                for (uint32_t channelIdx = 0; channelIdx < tokenLen; ++channelIdx) {
                                    const float partial = static_cast<float>(partialLocalInt.GetValue(channelIdx));
                                    const float scale = scaleLocal.GetValue(channelIdx);
                                    const float oldValue = accumLocal.GetValue(channelIdx);
                                    accumLocal.SetValue(channelIdx, oldValue + partial * scale * tokenScale);
                                }
                                PipeBarrier<PIPE_ALL>();
                                DataCopy(accumGM[outOffset], accumLocal, floatCopyParams);
                            }
                            }
                        } else {
                        for (uint64_t linearIdx = blockIdx; linearIdx < totalElements; linearIdx += aivCoreNum) {
                            const uint32_t rowLocal = static_cast<uint32_t>(linearIdx / mnConfig.n);
                            const uint32_t channelIdx =
                                static_cast<uint32_t>(linearIdx - static_cast<uint64_t>(rowLocal) * mnConfig.n);
                            const uint32_t globalRow = prevProbeSplit + rowLocal;
                            const uint64_t outOffset = static_cast<uint64_t>(globalRow) * tokenLen + channelIdx;
                            const float partial = static_cast<float>(mmOutGM.GetValue(partialBase + outOffset));
                            float scale = scalePackedGM.GetValue(expertScaleBase + channelIdx);
                            float tokenScale = perTokenScaleGM.GetValue(globalRow);
                            if constexpr (GROUP_QUANT_DEBUG_STAGE == 26) {
                                scale = 1.0F;
                                tokenScale = 1.0F;
                            }
                            accumGM.SetValue(outOffset, accumGM.GetValue(outOffset) + partial * scale * tokenScale);
                        }
                        }
                    }
                    SyncAll<false>();
                }
            }
            SyncAll<false>();
            prevProbeSplit = currProbeSplit;
        }
        SyncAll<false>();
        if constexpr (GROUP_QUANT_DEBUG_STAGE == 43 || GROUP_QUANT_DEBUG_STAGE == 44 ||
                      GROUP_QUANT_DEBUG_STAGE == 53) {
            if ASCEND_IS_AIV {
                const uint32_t tokenLen = gmmSwigluTensorList->tokenLen;
                GlobalTensor<float> accumGM;
                accumGM.SetGlobalBuffer((__gm__ float *)groupQuantAccumPtr,
                                        static_cast<uint64_t>(gmmBaseParams->M) * tokenLen);
                pipe->Reset();
                pipe->InitBuffer(groupQuantAccumWorkspace, tokenLen * sizeof(float));
                LocalTensor<float> accumLocal = groupQuantAccumWorkspace.Get<float>();
                DataCopyParams copyParams{1, static_cast<uint16_t>(tokenLen * sizeof(float) / UB_BLOCK_UNIT_SIZE),
                                          0, 0};
                for (uint32_t rowIdx = blockIdx; rowIdx < gmmBaseParams->M; rowIdx += aivCoreNum) {
                    DataCopy(accumLocal, accumGM[static_cast<uint64_t>(rowIdx) * tokenLen], copyParams);
                    PipeBarrier<PIPE_ALL>();
                    const uint32_t probeChannel = 0;
                    quantScaleOutputGM.SetValue(rowIdx, accumLocal.GetValue(probeChannel) / QUANT_SCALE_INT8);
                }
            }
            return;
        }
        if constexpr (GROUP_QUANT_DEBUG_STAGE == 45) {
            if ASCEND_IS_AIV {
                const uint32_t tokenLen = gmmSwigluTensorList->tokenLen;
                pipe->Reset();
                pipe->InitBuffer(groupQuantPackWorkspace, tokenLen * sizeof(int32_t));
                LocalTensor<int32_t> partialLocal = groupQuantPackWorkspace.Get<int32_t>();
                DataCopyParams copyParams{1, static_cast<uint16_t>(tokenLen * sizeof(int32_t) / UB_BLOCK_UNIT_SIZE),
                                          0, 0};
                for (uint32_t rowIdx = blockIdx; rowIdx < gmmBaseParams->M; rowIdx += aivCoreNum) {
                    uint32_t expertIdx = 0;
                    int64_t prevSplit = 0;
                    for (uint32_t groupIdx = 0; groupIdx < gmmSwigluTensorList->groupListLen; ++groupIdx) {
                        const int64_t currSplit = groupListGM.GetValue(groupIdx);
                        if (rowIdx < static_cast<uint32_t>(currSplit)) {
                            expertIdx = groupIdx;
                            break;
                        }
                        prevSplit = currSplit;
                    }
                    const uint64_t exactOffset = static_cast<uint64_t>(rowIdx) * tokenLen;
                    (void)expertIdx;
                    (void)prevSplit;
                    DataCopy(partialLocal, mmOutGM[exactOffset], copyParams);
                    PipeBarrier<PIPE_ALL>();
                    quantScaleOutputGM.SetValue(rowIdx, static_cast<float>(partialLocal.GetValue(0)) /
                                                QUANT_SCALE_INT8);
                }
            }
            return;
        }
        if constexpr (GROUP_QUANT_DEBUG_STAGE == 20 || GROUP_QUANT_DEBUG_STAGE == 21 ||
                      GROUP_QUANT_DEBUG_STAGE == 22 || GROUP_QUANT_DEBUG_STAGE == 23 ||
                      GROUP_QUANT_DEBUG_STAGE == 25 || GROUP_QUANT_DEBUG_STAGE == 26 ||
                      GROUP_QUANT_DEBUG_STAGE == 51 || GROUP_QUANT_DEBUG_STAGE == 52 ||
                      GROUP_QUANT_DEBUG_STAGE == 54 || GROUP_QUANT_DEBUG_STAGE == 56) {
            if ASCEND_IS_AIV {
                UpdateVecConfig(blockIdx, vecConfig);
                if (blockIdx < vecConfig.usedCoreNum) {
                    LocalTensor<int32_t> mmLocal = mmOutQueue.AllocTensor<int32_t>();
                    LocalTensor<int8_t> quantLocal = quantOutQueue.AllocTensor<int8_t>();
                    LocalTensor<float> quantScaleLocal = quantScaleOutQueue.AllocTensor<float>();
                    mmOutQueue.EnQue(mmLocal);
                    quantScaleOutQueue.EnQue(quantScaleLocal);
                    quantOutQueue.EnQue(quantLocal);
                    for (uint32_t outLoopIdx = 0; outLoopIdx < vecConfig.outLoopNum; outLoopIdx++) {
                        vecConfig.innerLoopNum = outLoopIdx == (vecConfig.outLoopNum - 1)
                                                    ? vecConfig.tailLoopNum
                                                    : gmmSwigluTensorList->maxProcessRowNum;
                        if constexpr (GROUP_QUANT_DEBUG_STAGE == 21 || GROUP_QUANT_DEBUG_STAGE == 22) {
                            vecConfig.innerLoopNum = 1;
                        }
                        customDataCopyInGroupQuant<CHANNELDTYPE>(outLoopIdx);
                        if constexpr (GROUP_QUANT_DEBUG_STAGE == 21 || GROUP_QUANT_DEBUG_STAGE == 54 ||
                                      GROUP_QUANT_DEBUG_STAGE == 56) {
                            for (uint32_t innerLoopIdx = 0; innerLoopIdx < vecConfig.innerLoopNum; innerLoopIdx++) {
                                Swiglu(innerLoopIdx);
                                Quant(innerLoopIdx);
                            }
                        } else if constexpr (GROUP_QUANT_DEBUG_STAGE == 52) {
                            LocalTensor<float> mmLocal = mmOutQueue.DeQue<float>();
                            LocalTensor<float> quantScaleLocal = quantScaleOutQueue.DeQue<float>();
                            for (uint32_t innerLoopIdx = 0; innerLoopIdx < vecConfig.innerLoopNum; innerLoopIdx++) {
                                const uint32_t rowBase = innerLoopIdx * gmmSwigluTensorList->tokenLen;
                                const float first = mmLocal.GetValue(rowBase);
                                const float second = mmLocal.GetValue(rowBase + gmmSwigluTensorList->tokenLen / 2);
                                quantScaleLocal.SetValue(innerLoopIdx, first * second);
                            }
                            PipeBarrier<PIPE_ALL>();
                            mmOutQueue.EnQue(mmLocal);
                            quantScaleOutQueue.EnQue(quantScaleLocal);
                            quantLocal = quantOutQueue.DeQue<int8_t>();
                            LocalTensor<int32_t> quantLocalI32 = quantLocal.template ReinterpretCast<int32_t>();
                            Duplicate(quantLocalI32, static_cast<int32_t>(0),
                                      vecConfig.innerLoopNum * gmmSwigluTensorList->tokenLen / 2 / sizeof(int32_t));
                            PipeBarrier<PIPE_V>();
                            quantOutQueue.EnQue(quantLocal);
                        } else if constexpr (GROUP_QUANT_DEBUG_STAGE == 51) {
                            for (uint32_t innerLoopIdx = 0; innerLoopIdx < vecConfig.innerLoopNum; innerLoopIdx++) {
                                Swiglu(innerLoopIdx);
                                LocalTensor<float> mmLocal = mmOutQueue.DeQue<float>();
                                LocalTensor<float> quantScaleLocal = quantScaleOutQueue.DeQue<float>();
                                quantScaleLocal.SetValue(innerLoopIdx,
                                                         mmLocal.GetValue(innerLoopIdx *
                                                                          gmmSwigluTensorList->tokenLen));
                                PipeBarrier<PIPE_ALL>();
                                mmOutQueue.EnQue(mmLocal);
                                quantScaleOutQueue.EnQue(quantScaleLocal);
                            }
                            quantLocal = quantOutQueue.DeQue<int8_t>();
                            LocalTensor<int32_t> quantLocalI32 = quantLocal.template ReinterpretCast<int32_t>();
                            Duplicate(quantLocalI32, static_cast<int32_t>(0),
                                      vecConfig.innerLoopNum * gmmSwigluTensorList->tokenLen / 2 / sizeof(int32_t));
                            PipeBarrier<PIPE_V>();
                            quantOutQueue.EnQue(quantLocal);
                        } else if constexpr (GROUP_QUANT_DEBUG_STAGE == 22 || GROUP_QUANT_DEBUG_STAGE == 23 ||
                                             GROUP_QUANT_DEBUG_STAGE == 25 || GROUP_QUANT_DEBUG_STAGE == 26) {
                            for (uint32_t innerLoopIdx = 0; innerLoopIdx < vecConfig.innerLoopNum; innerLoopIdx++) {
                                Quant(innerLoopIdx);
                            }
                        } else {
                            quantLocal = quantOutQueue.DeQue<int8_t>();
                            quantScaleLocal = quantScaleOutQueue.DeQue<float>();
                            LocalTensor<int32_t> quantLocalI32 = quantLocal.template ReinterpretCast<int32_t>();
                            Duplicate(quantLocalI32, static_cast<int32_t>(0),
                                      vecConfig.innerLoopNum * gmmSwigluTensorList->tokenLen / 2 / sizeof(int32_t));
                            Duplicate(quantScaleLocal, static_cast<float>(1.0), vecConfig.innerLoopNum);
                            PipeBarrier<PIPE_V>();
                            quantOutQueue.EnQue(quantLocal);
                            quantScaleOutQueue.EnQue(quantScaleLocal);
                        }
                        customDataCopyOut();
                    }
                    LocalTensor<float> mmLocalOut = mmOutQueue.DeQue<float>();
                    quantLocal = quantOutQueue.DeQue<int8_t>();
                    quantScaleLocal = quantScaleOutQueue.DeQue<float>();
                    mmOutQueue.FreeTensor(mmLocalOut);
                    quantScaleOutQueue.FreeTensor(quantScaleLocal);
                    quantOutQueue.FreeTensor(quantLocal);
                }
            }
        }
        return;
    }

    if constexpr (GROUP_QUANT_DEBUG_STAGE == 24) {
        const uint64_t totalElements =
            static_cast<uint64_t>(gmmBaseParams->M) * gmmSwigluTensorList->tokenLen;
        if ASCEND_IS_AIC {
            for (uint64_t linearIdx = 0; linearIdx < totalElements; ++linearIdx) {
                mmOutGM.SetValue(linearIdx, static_cast<int32_t>(1024));
            }
        }
        SyncAll<false>();
        if ASCEND_IS_AIV {
            UpdateVecConfig(blockIdx, vecConfig);
            if (blockIdx < vecConfig.usedCoreNum) {
                LocalTensor<int32_t> mmLocal = mmOutQueue.AllocTensor<int32_t>();
                LocalTensor<int8_t> quantLocal = quantOutQueue.AllocTensor<int8_t>();
                LocalTensor<float> quantScaleLocal = quantScaleOutQueue.AllocTensor<float>();
                mmOutQueue.EnQue(mmLocal);
                quantScaleOutQueue.EnQue(quantScaleLocal);
                quantOutQueue.EnQue(quantLocal);
                for (uint32_t outLoopIdx = 0; outLoopIdx < vecConfig.outLoopNum; outLoopIdx++) {
                    vecConfig.innerLoopNum = outLoopIdx == (vecConfig.outLoopNum - 1)
                                                ? vecConfig.tailLoopNum
                                                : gmmSwigluTensorList->maxProcessRowNum;
                    customDataCopyIn(outLoopIdx);
                    for (uint32_t innerLoopIdx = 0; innerLoopIdx < vecConfig.innerLoopNum; innerLoopIdx++) {
                        Quant(innerLoopIdx);
                    }
                    customDataCopyOut();
                }
                LocalTensor<float> mmLocalOut = mmOutQueue.DeQue<float>();
                quantLocal = quantOutQueue.DeQue<int8_t>();
                quantScaleLocal = quantScaleOutQueue.DeQue<float>();
                mmOutQueue.FreeTensor(mmLocalOut);
                quantScaleOutQueue.FreeTensor(quantScaleLocal);
                quantOutQueue.FreeTensor(quantLocal);
            }
        }
        return;
    }

    if constexpr (GROUP_QUANT_DEBUG_STAGE == 27) {
        const uint32_t tokenLen = gmmSwigluTensorList->tokenLen;
        const uint64_t totalElements = static_cast<uint64_t>(gmmBaseParams->M) * tokenLen;
        if ASCEND_IS_AIC {
            for (uint32_t kGroupIdx = 0; kGroupIdx < kGroupNum; ++kGroupIdx) {
                const uint64_t partialBase = static_cast<uint64_t>(kGroupIdx) * totalElements;
                for (uint64_t linearIdx = blockIdx; linearIdx < totalElements; linearIdx += aicCoreNum) {
                    mmOutGM.SetValue(partialBase + linearIdx, static_cast<int32_t>(1024));
                }
            }
        }
        SyncAll<false>();
        if ASCEND_IS_AIV {
            GlobalTensor<float> accumGM;
            accumGM.SetGlobalBuffer((__gm__ float *)groupQuantAccumPtr, totalElements);
            for (uint64_t linearIdx = blockIdx; linearIdx < totalElements; linearIdx += aivCoreNum) {
                accumGM.SetValue(linearIdx, 0.0F);
            }
        }
        SyncAll<false>();
        if ASCEND_IS_AIV {
            GlobalTensor<float> accumGM;
            accumGM.SetGlobalBuffer((__gm__ float *)groupQuantAccumPtr, totalElements);
            for (uint32_t kGroupIdx = 0; kGroupIdx < kGroupNum; ++kGroupIdx) {
                const uint64_t partialBase = static_cast<uint64_t>(kGroupIdx) * totalElements;
                for (uint64_t linearIdx = blockIdx; linearIdx < totalElements; linearIdx += aivCoreNum) {
                    const float partial = static_cast<float>(mmOutGM.GetValue(partialBase + linearIdx));
                    accumGM.SetValue(linearIdx, accumGM.GetValue(linearIdx) + partial);
                }
            }
        }
        SyncAll<false>();
        if ASCEND_IS_AIV {
            UpdateVecConfig(blockIdx, vecConfig);
            if (blockIdx < vecConfig.usedCoreNum) {
                LocalTensor<int32_t> mmLocal = mmOutQueue.AllocTensor<int32_t>();
                LocalTensor<int8_t> quantLocal = quantOutQueue.AllocTensor<int8_t>();
                LocalTensor<float> quantScaleLocal = quantScaleOutQueue.AllocTensor<float>();
                mmOutQueue.EnQue(mmLocal);
                quantScaleOutQueue.EnQue(quantScaleLocal);
                quantOutQueue.EnQue(quantLocal);
                for (uint32_t outLoopIdx = 0; outLoopIdx < vecConfig.outLoopNum; outLoopIdx++) {
                    vecConfig.innerLoopNum = outLoopIdx == (vecConfig.outLoopNum - 1)
                                                ? vecConfig.tailLoopNum
                                                : gmmSwigluTensorList->maxProcessRowNum;
                    customDataCopyInGroupQuant<CHANNELDTYPE>(outLoopIdx);
                    for (uint32_t innerLoopIdx = 0; innerLoopIdx < vecConfig.innerLoopNum; innerLoopIdx++) {
                        Quant(innerLoopIdx);
                    }
                    customDataCopyOut();
                }
                LocalTensor<float> mmLocalOut = mmOutQueue.DeQue<float>();
                quantLocal = quantOutQueue.DeQue<int8_t>();
                quantScaleLocal = quantScaleOutQueue.DeQue<float>();
                mmOutQueue.FreeTensor(mmLocalOut);
                quantScaleOutQueue.FreeTensor(quantScaleLocal);
                quantOutQueue.FreeTensor(quantLocal);
            }
        }
        return;
    }

    if constexpr (GROUP_QUANT_DEBUG_STAGE == 28 || GROUP_QUANT_DEBUG_STAGE == 29) {
        const uint32_t tokenLen = gmmSwigluTensorList->tokenLen;
        const uint64_t totalElements = static_cast<uint64_t>(gmmBaseParams->M) * tokenLen;
        if constexpr (GROUP_QUANT_DEBUG_STAGE == 29) {
            groupQuantAccumPtr = groupQuantXPackedPtr;
        }
        if ASCEND_IS_AIV {
            GlobalTensor<float> accumGM;
            accumGM.SetGlobalBuffer((__gm__ float *)groupQuantAccumPtr, totalElements);
            if (blockIdx < 2) {
                const uint64_t rowOffset = static_cast<uint64_t>(blockIdx) * tokenLen;
                for (uint32_t channelIdx = 0; channelIdx < tokenLen; ++channelIdx) {
                    accumGM.SetValue(rowOffset + channelIdx, 1024.0F);
                }
            }
        }
        SyncAll<false>();
        if ASCEND_IS_AIV {
            UpdateVecConfig(blockIdx, vecConfig);
            if (blockIdx < vecConfig.usedCoreNum) {
                LocalTensor<int32_t> mmLocal = mmOutQueue.AllocTensor<int32_t>();
                LocalTensor<int8_t> quantLocal = quantOutQueue.AllocTensor<int8_t>();
                LocalTensor<float> quantScaleLocal = quantScaleOutQueue.AllocTensor<float>();
                mmOutQueue.EnQue(mmLocal);
                quantScaleOutQueue.EnQue(quantScaleLocal);
                quantOutQueue.EnQue(quantLocal);
                for (uint32_t outLoopIdx = 0; outLoopIdx < vecConfig.outLoopNum; outLoopIdx++) {
                    vecConfig.innerLoopNum = outLoopIdx == (vecConfig.outLoopNum - 1)
                                                ? vecConfig.tailLoopNum
                                                : gmmSwigluTensorList->maxProcessRowNum;
                    customDataCopyInGroupQuant<CHANNELDTYPE>(outLoopIdx);
                    for (uint32_t innerLoopIdx = 0; innerLoopIdx < vecConfig.innerLoopNum; innerLoopIdx++) {
                        Quant(innerLoopIdx);
                    }
                    customDataCopyOut();
                }
                LocalTensor<float> mmLocalOut = mmOutQueue.DeQue<float>();
                quantLocal = quantOutQueue.DeQue<int8_t>();
                quantScaleLocal = quantScaleOutQueue.DeQue<float>();
                mmOutQueue.FreeTensor(mmLocalOut);
                quantScaleOutQueue.FreeTensor(quantScaleLocal);
                quantOutQueue.FreeTensor(quantLocal);
            }
        }
        return;
    }

    if constexpr (GROUP_QUANT_DEBUG_STAGE == 33 || GROUP_QUANT_DEBUG_STAGE == 34 ||
                  GROUP_QUANT_DEBUG_STAGE == 35 || GROUP_QUANT_DEBUG_STAGE == 36 ||
                  GROUP_QUANT_DEBUG_STAGE == 37 || GROUP_QUANT_DEBUG_STAGE == 38 ||
                  GROUP_QUANT_DEBUG_STAGE == 39) {
        const uint32_t tokenLen = gmmSwigluTensorList->tokenLen;
        const uint64_t totalElements = static_cast<uint64_t>(gmmBaseParams->M) * tokenLen;
        if ASCEND_IS_AIV {
            GlobalTensor<float> accumGM;
            accumGM.SetGlobalBuffer((__gm__ float *)groupQuantAccumPtr, totalElements);
            if (blockIdx < 2) {
                const uint64_t rowOffset = static_cast<uint64_t>(blockIdx) * tokenLen;
                if constexpr (GROUP_QUANT_DEBUG_STAGE == 39) {
                    pipe->Reset();
                    pipe->InitBuffer(groupQuantAccumWorkspace, tokenLen * sizeof(float));
                    LocalTensor<float> accumLocal = groupQuantAccumWorkspace.Get<float>();
                    Duplicate(accumLocal, static_cast<float>(1024.0F), tokenLen);
                    PipeBarrier<PIPE_V>();
                    DataCopyParams copyParams{1, static_cast<uint16_t>(tokenLen * sizeof(float) / UB_BLOCK_UNIT_SIZE),
                                              0, 0};
                    DataCopy(accumGM[rowOffset], accumLocal, copyParams);
                } else {
                    for (uint32_t channelIdx = 0; channelIdx < tokenLen; ++channelIdx) {
                        accumGM.SetValue(rowOffset + channelIdx, 1024.0F);
                    }
                }
            }
        }
        SyncAll<false>();
        if ASCEND_IS_AIV {
            if (gmmBaseParams->M == 2 && gmmSwigluTensorList->groupListLen == 2 && blockIdx < 2) {
                vecConfig.M = 2;
                vecConfig.usedCoreNum = 2;
                vecConfig.startIdx = blockIdx;
                vecConfig.curIdx = blockIdx;
                vecConfig.startOffset = blockIdx * tokenLen;
                vecConfig.curOffset = vecConfig.startOffset;
                vecConfig.taskNum = 1;
                vecConfig.curGroupIdx = blockIdx;
                vecConfig.outLoopNum = 1;
                vecConfig.innerLoopNum = 1;
                vecConfig.tailLoopNum = 1;
                vecConfig.nextUpadteInterVal = 1;

                pipe->Reset();
                pipe->InitBuffer(mmOutQueue, 1,
                                 gmmSwigluTensorList->maxProcessRowNum * tokenLen * sizeof(int32_t));
                pipe->InitBuffer(perChannelScaleInQueue, 1, tokenLen * sizeof(float));
                pipe->InitBuffer(quantOutQueue, 1,
                                 gmmSwigluTensorList->maxProcessRowNum * tokenLen / 2 * sizeof(int8_t));
                pipe->InitBuffer(quantScaleOutQueue, 1,
                                 AlignUp<int32_t>(gmmSwigluTensorList->maxProcessRowNum, 8) * sizeof(float));
                pipe->InitBuffer(reduceWorkspace, 1024 * sizeof(float));
                pipe->InitBuffer(castWorkspace, 32 * sizeof(int8_t));
                pipe->InitBuffer(groupQuantAccumWorkspace, tokenLen * sizeof(float));
                pipe->InitBuffer(groupQuantScaleWorkspace, tokenLen * sizeof(float));

                LocalTensor<int32_t> mmLocal = mmOutQueue.AllocTensor<int32_t>();
                LocalTensor<int8_t> quantLocal = quantOutQueue.AllocTensor<int8_t>();
                LocalTensor<float> quantScaleLocal = quantScaleOutQueue.AllocTensor<float>();
                mmOutQueue.EnQue(mmLocal);
                quantScaleOutQueue.EnQue(quantScaleLocal);
                quantOutQueue.EnQue(quantLocal);
                if constexpr (GROUP_QUANT_DEBUG_STAGE == 33) {
                    customDataCopyInGroupQuant<CHANNELDTYPE>(0);
                } else if constexpr (GROUP_QUANT_DEBUG_STAGE == 34 || GROUP_QUANT_DEBUG_STAGE == 38 ||
                                     GROUP_QUANT_DEBUG_STAGE == 39) {
                    LocalTensor<int32_t> mmLocalInt = mmOutQueue.DeQue<int32_t>();
                    LocalTensor<float> mmLocalFloat = mmLocalInt.template ReinterpretCast<float>();
                    GlobalTensor<float> accumGM;
                    accumGM.SetGlobalBuffer((__gm__ float *)groupQuantAccumPtr, totalElements);
                    if constexpr (GROUP_QUANT_DEBUG_STAGE == 34) {
                        DataCopyExtParams copyParams{1, static_cast<uint32_t>(tokenLen * sizeof(float)), 0, 0, 0};
                        DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
                        DataCopyPad(mmLocalFloat, accumGM[static_cast<uint64_t>(blockIdx) * tokenLen],
                                    copyParams, padParams);
                    } else {
                        DataCopyParams copyParams{1, static_cast<uint16_t>(tokenLen * sizeof(float) / UB_BLOCK_UNIT_SIZE),
                                                  0, 0};
                        DataCopy(mmLocalFloat, accumGM[static_cast<uint64_t>(blockIdx) * tokenLen], copyParams);
                    }
                    PipeBarrier<PIPE_ALL>();
                    mmOutQueue.EnQue(mmLocalFloat);
                } else if constexpr (GROUP_QUANT_DEBUG_STAGE == 35) {
                    LocalTensor<int32_t> mmLocalInt = mmOutQueue.DeQue<int32_t>();
                    LocalTensor<float> mmLocalFloat = mmLocalInt.template ReinterpretCast<float>();
                    Duplicate(mmLocalFloat, static_cast<float>(1024.0F), tokenLen);
                    PipeBarrier<PIPE_V>();
                    mmOutQueue.EnQue(mmLocalFloat);
                } else {
                    LocalTensor<int32_t> mmLocalInt = mmOutQueue.DeQue<int32_t>();
                    LocalTensor<float> mmLocalFloat = mmLocalInt.template ReinterpretCast<float>();
                    GlobalTensor<float> accumGM;
                    accumGM.SetGlobalBuffer((__gm__ float *)groupQuantAccumPtr, totalElements);
                    const uint64_t rowOffset = static_cast<uint64_t>(blockIdx) * tokenLen;
                    for (uint32_t channelIdx = 0; channelIdx < tokenLen; ++channelIdx) {
                        mmLocalFloat.SetValue(channelIdx, accumGM.GetValue(rowOffset + channelIdx));
                    }
                    if constexpr (GROUP_QUANT_DEBUG_STAGE == 37) {
                        PipeBarrier<PIPE_ALL>();
                        Adds(mmLocalFloat, mmLocalFloat, static_cast<float>(0.0F), tokenLen);
                    }
                    PipeBarrier<PIPE_ALL>();
                    mmOutQueue.EnQue(mmLocalFloat);
                }
                Quant(0);
                customDataCopyOut();
                LocalTensor<float> mmLocalOut = mmOutQueue.DeQue<float>();
                quantLocal = quantOutQueue.DeQue<int8_t>();
                quantScaleLocal = quantScaleOutQueue.DeQue<float>();
                mmOutQueue.FreeTensor(mmLocalOut);
                quantScaleOutQueue.FreeTensor(quantScaleLocal);
                quantOutQueue.FreeTensor(quantLocal);
            }
        }
        return;
    }

    if constexpr (GROUP_QUANT_DEBUG_STAGE == 30 || GROUP_QUANT_DEBUG_STAGE == 31 ||
                  GROUP_QUANT_DEBUG_STAGE == 32) {
        const uint32_t tokenLen = gmmSwigluTensorList->tokenLen;
        const uint64_t totalElements = static_cast<uint64_t>(gmmBaseParams->M) * tokenLen;
        if constexpr (GROUP_QUANT_DEBUG_STAGE == 32) {
            if ASCEND_IS_AIV {
                UpdateVecConfig(blockIdx, vecConfig);
            }
            SyncAll<false>();
        }
        if ASCEND_IS_AIV {
            GlobalTensor<float> accumGM;
            accumGM.SetGlobalBuffer((__gm__ float *)groupQuantAccumPtr, totalElements);
            for (uint64_t linearIdx = blockIdx; linearIdx < totalElements; linearIdx += aivCoreNum) {
                accumGM.SetValue(linearIdx, 1024.0F);
            }
        }
        SyncAll<false>();
        if ASCEND_IS_AIV {
            if constexpr (GROUP_QUANT_DEBUG_STAGE == 31) {
                UpdateVecConfig(blockIdx, vecConfig);
            }
            GlobalTensor<float> accumGM;
            accumGM.SetGlobalBuffer((__gm__ float *)groupQuantAccumPtr, totalElements);
            for (uint32_t rowIdx = blockIdx; rowIdx < gmmBaseParams->M; rowIdx += aivCoreNum) {
                const float value = accumGM.GetValue(static_cast<uint64_t>(rowIdx) * tokenLen);
                quantScaleOutputGM.SetValue(rowIdx, value / QUANT_SCALE_INT8);
            }
        }
        return;
    }

    int32_t prevSplitValue = 0;
    uint32_t count = 0;
    for (uint32_t expertIdx = 0; expertIdx < gmmSwigluTensorList->groupListLen; ++expertIdx) {
        int32_t currSplitValue = static_cast<int32_t>(groupListGM.GetValue(expertIdx));
        int32_t splitValue = currSplitValue - prevSplitValue;
        if ASCEND_IS_AIV {
            if constexpr (GROUP_QUANT_DEBUG_STAGE >= 2 && GROUP_QUANT_DEBUG_STAGE < 8) {
            PackGroupQuantWeight(expertIdx);
            }
        }
        SyncAll<false>();
        if constexpr (GROUP_QUANT_DEBUG_STAGE >= 3) {
            MNConfig mnConfig;
            mnConfig.m = static_cast<uint32_t>(splitValue);
            mnConfig.k = gmmSwigluTensorList->groupQuantGroupSize;
            mnConfig.n = gmmBaseParams->N;
            mnConfig.baseM = BASIC_M;
            mnConfig.baseN = BASIC_N;
            mnConfig.singleM = SINGLE_CORE_M;
            if constexpr (GROUP_QUANT_DEBUG_STAGE == 9) {
                mnConfig.singleN = 32;
            } else {
                mnConfig.singleN = SINGLE_CORE_N;
            }
            mnConfig.xBaseOffset = static_cast<int64_t>(currSplitValue - splitValue) * gmmBaseParams->K;
            mnConfig.yBaseOffset = static_cast<int64_t>(currSplitValue - splitValue) * mnConfig.n;
            if (mnConfig.m > 0 && mnConfig.k > 0 && mnConfig.n > 0) {
                mnConfig.blockDimM = Ceil(mnConfig.m, mnConfig.singleM);
                mnConfig.blockDimN = Ceil(mnConfig.n, mnConfig.singleN);

                uint32_t curCount = count + mnConfig.blockDimM * mnConfig.blockDimN;
                uint32_t thresholdM_dimN = THRESHOLD_BLOCK_NUM * mnConfig.blockDimN;

                for (uint32_t kGroupIdx = 0; kGroupIdx < kGroupNum; ++kGroupIdx) {
                    SyncAll<false>();
                    if ASCEND_IS_AIC {
                    uint32_t curBlock = blockIdx >= count ? blockIdx : blockIdx + gmmBaseParams->coreNum;
                    while (curBlock < curCount) {
                        MNBlockIdxCompute(mnConfig, curBlock, count, thresholdM_dimN);
                        if constexpr (GROUP_QUANT_DEBUG_STAGE == 9) {
                            MMComputeGroupQuantOriginalWeight(expertIdx, kGroupIdx, mnConfig, blockIdx);
                        } else {
                            MMComputeGroupQuant(expertIdx, kGroupIdx, mnConfig, blockIdx);
                        }
                        curBlock += aicCoreNum;
                    }
                    }
                    SyncAll<false>();
                    if constexpr (GROUP_QUANT_DEBUG_STAGE >= 9) {
                    if ASCEND_IS_AIV {
                    GlobalTensor<float> accumGM;
                    GlobalTensor<float> scalePackedGM;
                    accumGM.SetGlobalBuffer((__gm__ float *)groupQuantAccumPtr,
                                            static_cast<uint64_t>(gmmBaseParams->M) * gmmSwigluTensorList->tokenLen);
                    scalePackedGM.SetGlobalBuffer((__gm__ float *)groupQuantScalePackedPtr,
                                                  static_cast<uint64_t>(gmmSwigluTensorList->groupListLen) *
                                                  kGroupNum * gmmSwigluTensorList->tokenLen);
                    const uint32_t tokenLen = gmmSwigluTensorList->tokenLen;
                    const uint32_t startRow = static_cast<uint32_t>(currSplitValue - splitValue);
                    const uint64_t expertScaleBase =
                        (static_cast<uint64_t>(expertIdx) * kGroupNum + kGroupIdx) * tokenLen;
                    const uint64_t partialBase = GROUP_QUANT_DEBUG_STAGE == 9
                                                     ? 0
                                                     : static_cast<uint64_t>(kGroupIdx) * gmmBaseParams->M * tokenLen;
                    pipe->Reset();
                    pipe->InitBuffer(groupQuantPackWorkspace, tokenLen * sizeof(int32_t));
                    pipe->InitBuffer(groupQuantAccumWorkspace, tokenLen * sizeof(float));
                    pipe->InitBuffer(groupQuantScaleWorkspace, tokenLen * sizeof(float));
                    LocalTensor<int32_t> partialLocalInt = groupQuantPackWorkspace.Get<int32_t>();
                    LocalTensor<float> partialLocal = partialLocalInt.template ReinterpretCast<float>();
                    LocalTensor<float> accumLocal = groupQuantAccumWorkspace.Get<float>();
                    LocalTensor<float> scaleLocal = groupQuantScaleWorkspace.Get<float>();
                    DataCopyParams intCopyParams{1, static_cast<uint16_t>(tokenLen * sizeof(int32_t) / UB_BLOCK_UNIT_SIZE),
                                                 0, 0};
                    DataCopyParams floatCopyParams{1, static_cast<uint16_t>(tokenLen * sizeof(float) / UB_BLOCK_UNIT_SIZE),
                                                   0, 0};
                    for (uint32_t rowLocal = blockIdx; rowLocal < static_cast<uint32_t>(splitValue);
                         rowLocal += aivCoreNum) {
                        const uint32_t globalRow = startRow + rowLocal;
                        const uint64_t outOffset = static_cast<uint64_t>(globalRow) * tokenLen;
                        if (kGroupIdx == 0) {
                            Duplicate(accumLocal, static_cast<float>(0.0F), tokenLen);
                            PipeBarrier<PIPE_V>();
                        } else {
                            DataCopy(accumLocal, accumGM[outOffset], floatCopyParams);
                            PipeBarrier<PIPE_ALL>();
                        }
                        DataCopy(partialLocalInt, mmOutGM[partialBase + outOffset], intCopyParams);
                        DataCopy(scaleLocal, scalePackedGM[expertScaleBase], floatCopyParams);
                        PipeBarrier<PIPE_ALL>();
                        Cast(partialLocal, partialLocalInt, RoundMode::CAST_NONE, tokenLen);
                        PipeBarrier<PIPE_V>();
                        Mul(partialLocal, partialLocal, scaleLocal, tokenLen);
                        PipeBarrier<PIPE_V>();
                        Muls(partialLocal, partialLocal, perTokenScaleGM.GetValue(globalRow), tokenLen);
                        PipeBarrier<PIPE_V>();
                        Add(accumLocal, accumLocal, partialLocal, tokenLen);
                        PipeBarrier<PIPE_V>();
                        DataCopy(accumGM[outOffset], accumLocal, floatCopyParams);
                    }
                    }
                    }
                    SyncAll<false>();
                }
                            }
        }
        SyncAll<false>();
        if ASCEND_IS_AIC {
            MNConfig mnConfig;
            mnConfig.m = static_cast<uint32_t>(splitValue);
            mnConfig.n = gmmBaseParams->N;
            mnConfig.singleM = SINGLE_CORE_M;
            if constexpr (GROUP_QUANT_DEBUG_STAGE == 9) {
                mnConfig.singleN = 32;
            } else {
                mnConfig.singleN = SINGLE_CORE_N;
            }
            mnConfig.blockDimM = Ceil(mnConfig.m, mnConfig.singleM);
            mnConfig.blockDimN = Ceil(mnConfig.n, mnConfig.singleN);
            uint32_t curCount = count + mnConfig.blockDimM * mnConfig.blockDimN;
            count = curCount % gmmBaseParams->coreNum;
        }
        prevSplitValue = currSplitValue;
    }

    if constexpr (GROUP_QUANT_DEBUG_STAGE == 40) {
        if ASCEND_IS_AIV {
            const uint32_t tokenLen = gmmSwigluTensorList->tokenLen;
            GlobalTensor<float> accumGM;
            accumGM.SetGlobalBuffer((__gm__ float *)groupQuantAccumPtr,
                                    static_cast<uint64_t>(gmmBaseParams->M) * tokenLen);
            for (uint32_t rowIdx = blockIdx; rowIdx < gmmBaseParams->M; rowIdx += aivCoreNum) {
                quantScaleOutputGM.SetValue(rowIdx,
                                            accumGM.GetValue(static_cast<uint64_t>(rowIdx) * tokenLen) /
                                            QUANT_SCALE_INT8);
            }
        }
        return;
    }

    if constexpr (GROUP_QUANT_DEBUG_STAGE == 41) {
        if ASCEND_IS_AIV {
            const uint32_t tokenLen = gmmSwigluTensorList->tokenLen;
            for (uint32_t rowIdx = blockIdx; rowIdx < gmmBaseParams->M; rowIdx += aivCoreNum) {
                quantScaleOutputGM.SetValue(rowIdx,
                                            static_cast<float>(mmOutGM.GetValue(static_cast<uint64_t>(rowIdx) *
                                                                                tokenLen)) /
                                            QUANT_SCALE_INT8);
            }
        }
        return;
    }

    if constexpr (GROUP_QUANT_DEBUG_STAGE == 42) {
        if ASCEND_IS_AIV {
            const uint32_t tokenLen = gmmSwigluTensorList->tokenLen;
            pipe->Reset();
            pipe->InitBuffer(groupQuantPackWorkspace, tokenLen * sizeof(int32_t));
            LocalTensor<int32_t> partialLocal = groupQuantPackWorkspace.Get<int32_t>();
            DataCopyParams copyParams{1, static_cast<uint16_t>(tokenLen * sizeof(int32_t) / UB_BLOCK_UNIT_SIZE),
                                      0, 0};
            for (uint32_t rowIdx = blockIdx; rowIdx < gmmBaseParams->M; rowIdx += aivCoreNum) {
                DataCopy(partialLocal, mmOutGM[static_cast<uint64_t>(rowIdx) * tokenLen], copyParams);
                PipeBarrier<PIPE_ALL>();
                quantScaleOutputGM.SetValue(rowIdx,
                                            static_cast<float>(partialLocal.GetValue(0)) / QUANT_SCALE_INT8);
            }
        }
        return;
    }

    if constexpr (GROUP_QUANT_DEBUG_STAGE == 3) {
        if ASCEND_IS_AIV {
            UpdateVecConfig(blockIdx, vecConfig);
            if (blockIdx < vecConfig.usedCoreNum) {
                LocalTensor<int8_t> quantLocal = quantOutQueue.AllocTensor<int8_t>();
                LocalTensor<float> quantScaleLocal = quantScaleOutQueue.AllocTensor<float>();
                quantScaleOutQueue.EnQue(quantScaleLocal);
                quantOutQueue.EnQue(quantLocal);
                for (uint32_t outLoopIdx = 0; outLoopIdx < vecConfig.outLoopNum; outLoopIdx++) {
                    vecConfig.innerLoopNum = outLoopIdx == (vecConfig.outLoopNum - 1)
                                                ? vecConfig.tailLoopNum
                                                : gmmSwigluTensorList->maxProcessRowNum;
                    quantLocal = quantOutQueue.DeQue<int8_t>();
                    quantScaleLocal = quantScaleOutQueue.DeQue<float>();
                    LocalTensor<int32_t> quantLocalI32 = quantLocal.template ReinterpretCast<int32_t>();
                    Duplicate(quantLocalI32, static_cast<int32_t>(0),
                              vecConfig.innerLoopNum * gmmSwigluTensorList->tokenLen / 2 / sizeof(int32_t));
                    Duplicate(quantScaleLocal, static_cast<float>(1.0), vecConfig.innerLoopNum);
                    PipeBarrier<PIPE_V>();
                    quantOutQueue.EnQue(quantLocal);
                    quantScaleOutQueue.EnQue(quantScaleLocal);
                    customDataCopyOut();
                }
                quantLocal = quantOutQueue.DeQue<int8_t>();
                quantScaleLocal = quantScaleOutQueue.DeQue<float>();
                quantScaleOutQueue.FreeTensor(quantScaleLocal);
                quantOutQueue.FreeTensor(quantLocal);
            }
        }
        return;
    }

    if ASCEND_IS_AIV {
        UpdateVecConfig(blockIdx, vecConfig);
        if (blockIdx < vecConfig.usedCoreNum) {
            LocalTensor<int32_t> mmLocal = mmOutQueue.AllocTensor<int32_t>();
            LocalTensor<int8_t> quantLocal = quantOutQueue.AllocTensor<int8_t>();
            LocalTensor<float> quantScaleLocal = quantScaleOutQueue.AllocTensor<float>();
            mmOutQueue.EnQue(mmLocal);
            quantScaleOutQueue.EnQue(quantScaleLocal);
            quantOutQueue.EnQue(quantLocal);
        }
        SyncAll<false>();
        if (blockIdx < vecConfig.usedCoreNum) {
            LocalTensor<int8_t> quantLocal;
            LocalTensor<float> quantScaleLocal;
            for (uint32_t outLoopIdx = 0; outLoopIdx < vecConfig.outLoopNum; outLoopIdx++) {
                vecConfig.innerLoopNum = outLoopIdx == (vecConfig.outLoopNum - 1)
                                            ? vecConfig.tailLoopNum
                                            : gmmSwigluTensorList->maxProcessRowNum;
                customDataCopyInGroupQuant<CHANNELDTYPE>(outLoopIdx);
                if constexpr (GROUP_QUANT_DEBUG_STAGE >= 8) {
                    for (uint32_t innerLoopIdx = 0; innerLoopIdx < vecConfig.innerLoopNum; innerLoopIdx++) {
                        Swiglu(innerLoopIdx);
                        Quant(innerLoopIdx);
                    }
                } else {
                    quantLocal = quantOutQueue.DeQue<int8_t>();
                    quantScaleLocal = quantScaleOutQueue.DeQue<float>();
                    LocalTensor<int32_t> quantLocalI32 = quantLocal.template ReinterpretCast<int32_t>();
                    Duplicate(quantLocalI32, static_cast<int32_t>(0),
                              vecConfig.innerLoopNum * gmmSwigluTensorList->tokenLen / 2 / sizeof(int32_t));
                    Duplicate(quantScaleLocal, static_cast<float>(1.0), vecConfig.innerLoopNum);
                    PipeBarrier<PIPE_V>();
                    quantOutQueue.EnQue(quantLocal);
                    quantScaleOutQueue.EnQue(quantScaleLocal);
                }
                customDataCopyOut();
            }

            LocalTensor<float> mmLocal = mmOutQueue.DeQue<float>();
            quantLocal = quantOutQueue.DeQue<int8_t>();
            quantScaleLocal = quantScaleOutQueue.DeQue<float>();
            mmOutQueue.FreeTensor(mmLocal);
            quantScaleOutQueue.FreeTensor(quantScaleLocal);
            quantOutQueue.FreeTensor(quantLocal);
        }
    }
}

template <typename mmType, bool sync, typename CHANNELDTYPE>
__aicore__ inline void GMMSwigluCompute<mmType, sync, CHANNELDTYPE>::QuantGroupQuantPostRow(
    LocalTensor<float> postLocal, LocalTensor<float> absLocal, LocalTensor<float> reduceWorkLocal,
    LocalTensor<float> reduceResLocal, LocalTensor<float> reduceTmpLocal,
    LocalTensor<int8_t> quantLocal, LocalTensor<float> quantScaleLocal,
    LocalTensor<int8_t> castSpace, uint32_t rowLocal, uint32_t halfTokenLen)
{
    Abs(absLocal, postLocal, halfTokenLen);
    PipeBarrier<PIPE_V>();
    ReduceMaxTemplateGroupQuant(reduceResLocal, reduceWorkLocal, absLocal,
                                reduceTmpLocal, halfTokenLen);
    event_t evtVToS = static_cast<event_t>(pipe->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(evtVToS);
    WaitFlag<HardEvent::V_S>(evtVToS);
    const float absMax = reduceResLocal.GetValue(0);
    float quantScale = (absMax > 1e-6F ? absMax : 1e-6F) / QUANT_SCALE_INT8;
    quantScaleLocal.SetValue(rowLocal, quantScale);
    quantScale = 1.0F / quantScale;
    event_t evtSToV = static_cast<event_t>(pipe->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(evtSToV);
    WaitFlag<HardEvent::S_V>(evtSToV);
    Muls(postLocal, postLocal, quantScale, halfTokenLen);
    PipeBarrier<PIPE_V>();
    Cast(postLocal, postLocal, RoundMode::CAST_RINT, halfTokenLen);
    PipeBarrier<PIPE_V>();
    CastFp32ToInt8Template(quantLocal, postLocal, castSpace,
                          rowLocal * halfTokenLen, 0, halfTokenLen);
}

template <typename mmType, bool sync, typename CHANNELDTYPE>
__aicore__ inline void GMMSwigluCompute<mmType, sync, CHANNELDTYPE>::ProcessGroupQuantTile(
    uint32_t expertIdx, uint32_t rowStart, uint32_t tileM, const MNConfig& mnConfig, uint32_t bufferTileM)
{
    const uint32_t tokenLen = gmmSwigluTensorList->tokenLen;
    const uint32_t halfTokenLen = tokenLen / BISECT;
    const uint32_t kGroupNum = gmmSwigluTensorList->groupQuantGroupNum;
    const uint32_t tileElements = tileM * tokenLen;
    GlobalTensor<float> scalePackedGM;
    if (UseDirectGroupQuantScale()) {
        scalePackedGM.SetGlobalBuffer(
            GetTensorAddr<float>(expertIdx, perChannelScalePtr),
            static_cast<uint64_t>(kGroupNum) * tokenLen);
    } else {
        scalePackedGM.SetGlobalBuffer((__gm__ float *)groupQuantScalePackedPtr,
                                      static_cast<uint64_t>(gmmSwigluTensorList->groupListLen) *
                                          kGroupNum * tokenLen);
    }

    if (bufferTileM == 0) {
        bufferTileM = tileM;
        const uint32_t bufferElements = bufferTileM * tokenLen;
        const uint32_t alignedBufferElements = AlignUp<8>(bufferElements);
        pipe->Reset();
        pipe->InitBuffer(groupQuantPackWorkspace,
                         alignedBufferElements * sizeof(int32_t) +
                             alignedBufferElements * sizeof(float));
        pipe->InitBuffer(groupQuantAccumOutQueue, 1, alignedBufferElements * sizeof(float));
        pipe->InitBuffer(groupQuantScaleWorkspace, AlignUp<8>(tokenLen) * sizeof(float));
        pipe->InitBuffer(quantOutQueue, 1,
                         AlignUp<int32_t>(bufferTileM * halfTokenLen, UB_BLOCK_UNIT_SIZE));
        pipe->InitBuffer(quantScaleOutQueue, 1, AlignUp<8>(bufferTileM) * sizeof(float));
        const uint32_t reduceWorkElements =
            AlignUp<8>(halfTokenLen) * 4 + FLOAT_UB_BLOCK_UNIT_SIZE * 2;
        pipe->InitBuffer(reduceWorkspace, reduceWorkElements * sizeof(float));
        pipe->InitBuffer(castWorkspace, UB_BLOCK_UNIT_SIZE);
    }
    const uint32_t alignedTileElements = AlignUp<8>(bufferTileM * tokenLen);

    LocalTensor<int32_t> partialLocal = groupQuantPackWorkspace.Get<int32_t>();
    LocalTensor<float> partialFloat = groupQuantPackWorkspace.GetWithOffset<float>(
        alignedTileElements, alignedTileElements * sizeof(int32_t));
    LocalTensor<float> accumLocal = groupQuantAccumOutQueue.AllocTensor<float>();
    LocalTensor<float> chScaleLocal = groupQuantScaleWorkspace.Get<float>();
    Duplicate(accumLocal, 0.0F, tileElements);
    PipeBarrier<PIPE_V>();

    // The production tiling normally assigns at most four rows to one AIV.
    // Cache their per-token scales in scalar registers before entering the
    // K-group loop.  The old path performed a GM GetValue plus V->S/S->V hard
    // synchronization for every (K-group, row) pair, even though a token scale
    // is invariant across all K groups.  Keep the original loop as a fallback
    // for larger externally configured tiles.
    const bool cacheTokenScales = tileM <= 4;
    float tokenScale0 = 0.0F;
    float tokenScale1 = 0.0F;
    float tokenScale2 = 0.0F;
    float tokenScale3 = 0.0F;
    if (cacheTokenScales) {
        event_t evtVToS = static_cast<event_t>(pipe->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(evtVToS);
        WaitFlag<HardEvent::V_S>(evtVToS);
        tokenScale0 = perTokenScaleGM.GetValue(rowStart);
        if (tileM > 1) {
            tokenScale1 = perTokenScaleGM.GetValue(rowStart + 1);
        }
        if (tileM > 2) {
            tokenScale2 = perTokenScaleGM.GetValue(rowStart + 2);
        }
        if (tileM > 3) {
            tokenScale3 = perTokenScaleGM.GetValue(rowStart + 3);
        }
        event_t evtSToV = static_cast<event_t>(pipe->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(evtSToV);
        WaitFlag<HardEvent::S_V>(evtSToV);
    }

    DataCopyParams partialCopy{
        1, static_cast<uint16_t>(tileElements * sizeof(int32_t) / UB_BLOCK_UNIT_SIZE), 0, 0};
    DataCopyParams chScaleCopy{
        1, static_cast<uint16_t>(tokenLen * sizeof(float) / UB_BLOCK_UNIT_SIZE), 0, 0};
    for (uint32_t kGroupIdx = 0; kGroupIdx < kGroupNum; ++kGroupIdx) {
        const uint64_t partialBase = static_cast<uint64_t>(kGroupIdx) * gmmBaseParams->M * mnConfig.n +
                                     static_cast<uint64_t>(rowStart) * mnConfig.n;
        const uint64_t scaleBase = UseDirectGroupQuantScale()
            ? static_cast<uint64_t>(kGroupIdx) * tokenLen
            : (static_cast<uint64_t>(expertIdx) * kGroupNum + kGroupIdx) * tokenLen;
        DataCopy(partialLocal, mmOutGM[partialBase], partialCopy);
        DataCopy(chScaleLocal, scalePackedGM[scaleBase], chScaleCopy);
        PipeBarrier<PIPE_ALL>();
        Cast(partialFloat, partialLocal, RoundMode::CAST_NONE, tileElements);
        PipeBarrier<PIPE_V>();
        if (cacheTokenScales) {
            Mul(partialFloat, partialFloat, chScaleLocal, tokenLen);
            if (tileM > 1) {
                Mul(partialFloat[tokenLen], partialFloat[tokenLen], chScaleLocal, tokenLen);
            }
            if (tileM > 2) {
                Mul(partialFloat[tokenLen * 2], partialFloat[tokenLen * 2], chScaleLocal, tokenLen);
            }
            if (tileM > 3) {
                Mul(partialFloat[tokenLen * 3], partialFloat[tokenLen * 3], chScaleLocal, tokenLen);
            }
            PipeBarrier<PIPE_V>();
            Axpy(accumLocal, partialFloat, tokenScale0, tokenLen);
            if (tileM > 1) {
                Axpy(accumLocal[tokenLen], partialFloat[tokenLen], tokenScale1, tokenLen);
            }
            if (tileM > 2) {
                Axpy(accumLocal[tokenLen * 2], partialFloat[tokenLen * 2], tokenScale2, tokenLen);
            }
            if (tileM > 3) {
                Axpy(accumLocal[tokenLen * 3], partialFloat[tokenLen * 3], tokenScale3, tokenLen);
            }
            PipeBarrier<PIPE_V>();
        } else {
            for (uint32_t rowLocal = 0; rowLocal < tileM; ++rowLocal) {
                const uint32_t rowOffset = rowLocal * tokenLen;
                Mul(partialFloat[rowOffset], partialFloat[rowOffset], chScaleLocal, tokenLen);
                PipeBarrier<PIPE_V>();
                event_t evtVToS = static_cast<event_t>(pipe->FetchEventID(HardEvent::V_S));
                SetFlag<HardEvent::V_S>(evtVToS);
                WaitFlag<HardEvent::V_S>(evtVToS);
                const float tokenScale = perTokenScaleGM.GetValue(rowStart + rowLocal);
                event_t evtSToV = static_cast<event_t>(pipe->FetchEventID(HardEvent::S_V));
                SetFlag<HardEvent::S_V>(evtSToV);
                WaitFlag<HardEvent::S_V>(evtSToV);
                Axpy(accumLocal[rowOffset], partialFloat[rowOffset], tokenScale, tokenLen);
                PipeBarrier<PIPE_V>();
            }
        }
    }

    LocalTensor<int8_t> quantLocal = quantOutQueue.AllocTensor<int8_t>();
    LocalTensor<float> quantScaleLocal = quantScaleOutQueue.AllocTensor<float>();
    const uint32_t alignedHalfTokenLen = AlignUp<8>(halfTokenLen);
    LocalTensor<float> postWorkspace0 = reduceWorkspace.Get<float>(alignedHalfTokenLen);
    LocalTensor<float> postWorkspace1 = reduceWorkspace.GetWithOffset<float>(
        alignedHalfTokenLen, alignedHalfTokenLen * sizeof(float));
    LocalTensor<float> absLocal = reduceWorkspace.GetWithOffset<float>(
        alignedHalfTokenLen, alignedHalfTokenLen * sizeof(float) * 2);
    LocalTensor<float> reduceWorkLocal = reduceWorkspace.GetWithOffset<float>(
        alignedHalfTokenLen, alignedHalfTokenLen * sizeof(float) * 3);
    LocalTensor<float> reduceResLocal = reduceWorkspace.GetWithOffset<float>(
        FLOAT_UB_BLOCK_UNIT_SIZE, alignedHalfTokenLen * sizeof(float) * 4);
    LocalTensor<float> reduceTmpLocal = reduceWorkspace.GetWithOffset<float>(
        FLOAT_UB_BLOCK_UNIT_SIZE, alignedHalfTokenLen * sizeof(float) * 4 + UB_BLOCK_UNIT_SIZE);
    LocalTensor<int8_t> castSpace = castWorkspace.Get<int8_t>();
    if (halfTokenLen < 512) {
        for (uint32_t rowLocal = 0; rowLocal < tileM; ++rowLocal) {
            const uint32_t rowOffset = rowLocal * tokenLen;
            LocalTensor<float> src0Local = accumLocal[rowOffset + halfTokenLen];
            LocalTensor<float> src1Local = accumLocal[rowOffset];
            if (gmmSwigluTensorList->swigluLimit > 1e-8F) {
                Mins(src0Local, src0Local, gmmSwigluTensorList->swigluLimit, halfTokenLen);
                PipeBarrier<PIPE_V>();
                Maxs(src0Local, src0Local, -gmmSwigluTensorList->swigluLimit, halfTokenLen);
                PipeBarrier<PIPE_V>();
                Mins(src1Local, src1Local, gmmSwigluTensorList->swigluLimit, halfTokenLen);
                PipeBarrier<PIPE_V>();
            }
            SwiGLU<float, false>(postWorkspace0, src0Local, src1Local, 1.0F, halfTokenLen);
            PipeBarrier<PIPE_V>();
            DataCopyParams localCopy{
                1, static_cast<uint16_t>(halfTokenLen / FLOAT_UB_BLOCK_UNIT_SIZE), 0, 0};
            DataCopy(accumLocal[rowOffset], postWorkspace0, localCopy);
            PipeBarrier<PIPE_ALL>();
            Abs(accumLocal[rowOffset + halfTokenLen], accumLocal[rowOffset], halfTokenLen);
            PipeBarrier<PIPE_V>();
            ReduceMaxTemplateGroupQuant(reduceResLocal, reduceWorkLocal,
                                        accumLocal[rowOffset + halfTokenLen], reduceTmpLocal,
                                        halfTokenLen);
            event_t evtVToS = static_cast<event_t>(pipe->FetchEventID(HardEvent::V_S));
            SetFlag<HardEvent::V_S>(evtVToS);
            WaitFlag<HardEvent::V_S>(evtVToS);
            const float absMax = reduceResLocal.GetValue(0);
            float quantScale = (absMax > 1e-6F ? absMax : 1e-6F) / QUANT_SCALE_INT8;
            quantScaleLocal.SetValue(rowLocal, quantScale);
            quantScale = 1.0F / quantScale;
            event_t evtSToV = static_cast<event_t>(pipe->FetchEventID(HardEvent::S_V));
            SetFlag<HardEvent::S_V>(evtSToV);
            WaitFlag<HardEvent::S_V>(evtSToV);
            Muls(accumLocal[rowOffset], accumLocal[rowOffset], quantScale, halfTokenLen);
            PipeBarrier<PIPE_V>();
            Cast(accumLocal[rowOffset], accumLocal[rowOffset], RoundMode::CAST_RINT, halfTokenLen);
            PipeBarrier<PIPE_V>();
            CastFp32ToInt8Template(quantLocal, accumLocal, castSpace,
                                   rowLocal * halfTokenLen, rowOffset, halfTokenLen);
        }
    } else {
      for (uint32_t rowLocal = 0; rowLocal < tileM; rowLocal += 2) {
        const uint32_t rowOffset0 = rowLocal * tokenLen;
        LocalTensor<float> src0Local = accumLocal[rowOffset0 + halfTokenLen];
        LocalTensor<float> src1Local = accumLocal[rowOffset0];
        if (gmmSwigluTensorList->swigluLimit > 1e-8F) {
            Mins(src0Local, src0Local, gmmSwigluTensorList->swigluLimit, halfTokenLen);
            PipeBarrier<PIPE_V>();
            Maxs(src0Local, src0Local, -gmmSwigluTensorList->swigluLimit, halfTokenLen);
            PipeBarrier<PIPE_V>();
            Mins(src1Local, src1Local, gmmSwigluTensorList->swigluLimit, halfTokenLen);
            PipeBarrier<PIPE_V>();
        }
        SwiGLU<float, false>(postWorkspace0, src0Local, src1Local, 1.0F, halfTokenLen);

        const bool hasSecondRow = rowLocal + 1 < tileM;
        if (hasSecondRow) {
            const uint32_t rowOffset1 = rowOffset0 + tokenLen;
            LocalTensor<float> src0Local1 = accumLocal[rowOffset1 + halfTokenLen];
            LocalTensor<float> src1Local1 = accumLocal[rowOffset1];
            if (gmmSwigluTensorList->swigluLimit > 1e-8F) {
                Mins(src0Local1, src0Local1, gmmSwigluTensorList->swigluLimit, halfTokenLen);
                PipeBarrier<PIPE_V>();
                Maxs(src0Local1, src0Local1, -gmmSwigluTensorList->swigluLimit, halfTokenLen);
                PipeBarrier<PIPE_V>();
                Mins(src1Local1, src1Local1, gmmSwigluTensorList->swigluLimit, halfTokenLen);
                PipeBarrier<PIPE_V>();
            }
            SwiGLU<float, false>(postWorkspace1, src0Local1, src1Local1, 1.0F, halfTokenLen);
        }
        PipeBarrier<PIPE_V>();
        QuantGroupQuantPostRow(postWorkspace0, absLocal, reduceWorkLocal, reduceResLocal, reduceTmpLocal,
                               quantLocal, quantScaleLocal, castSpace, rowLocal, halfTokenLen);
        if (hasSecondRow) {
            QuantGroupQuantPostRow(postWorkspace1, absLocal, reduceWorkLocal, reduceResLocal, reduceTmpLocal,
                                   quantLocal, quantScaleLocal, castSpace, rowLocal + 1, halfTokenLen);
        }
      }
    }

    quantOutQueue.EnQue(quantLocal);
    quantScaleOutQueue.EnQue(quantScaleLocal);
    LocalTensor<float> scaleOutLocal = quantScaleOutQueue.DeQue<float>();
    LocalTensor<int8_t> quantOutLocal = quantOutQueue.DeQue<int8_t>();
    DataCopyParams scaleCopyOut{1, static_cast<uint16_t>(tileM * sizeof(float)), 0, 0};
    DataCopyPad(quantScaleOutputGM[rowStart], scaleOutLocal, scaleCopyOut);
    DataCopyParams quantCopyOut{
        1, static_cast<uint16_t>(tileM * halfTokenLen * sizeof(int8_t)), 0, 0};
    DataCopyPad(quantOutputGM[static_cast<uint64_t>(rowStart) * halfTokenLen], quantOutLocal, quantCopyOut);
    quantScaleOutQueue.FreeTensor(scaleOutLocal);
    quantOutQueue.FreeTensor(quantOutLocal);
    groupQuantAccumOutQueue.FreeTensor(accumLocal);
}

template <typename mmType, bool sync, typename CHANNELDTYPE>
__aicore__ inline void GMMSwigluCompute<mmType, sync, CHANNELDTYPE>::PackGroupQuantX()
{
    GlobalTensor<int8_t> xRawGM;
    GlobalTensor<int8_t> xPackedGM;
    xRawGM.SetGlobalBuffer((__gm__ int8_t *)xTensorPtr, static_cast<uint64_t>(gmmBaseParams->M) * gmmBaseParams->K);
    xPackedGM.SetGlobalBuffer((__gm__ int8_t *)groupQuantXPackedPtr,
                              static_cast<uint64_t>(gmmBaseParams->M) * gmmBaseParams->K);

    const uint32_t kGroupNum = gmmSwigluTensorList->groupQuantGroupNum;
    const uint32_t groupSize = gmmSwigluTensorList->groupQuantGroupSize;
    const uint32_t totalRows = gmmBaseParams->M * kGroupNum;
    pipe->InitBuffer(groupQuantPackWorkspace, AlignUp<32>(groupSize) * sizeof(int8_t));
    LocalTensor<int8_t> packLocal = groupQuantPackWorkspace.Get<int8_t>();
    DataCopyExtParams copyInParams{1, groupSize, 0, 0, 0};
    DataCopyPadExtParams<int8_t> padParams{false, 0, 0, 0};
    DataCopyExtParams copyOutParams{1, groupSize, 0, 0, 0};
    for (uint32_t packedRow = blockIdx; packedRow < totalRows; packedRow += aivCoreNum) {
        const uint32_t kGroupIdx = packedRow / gmmBaseParams->M;
        const uint32_t rowIdx = packedRow - kGroupIdx * gmmBaseParams->M;
        const uint64_t srcOffset = static_cast<uint64_t>(rowIdx) * gmmBaseParams->K +
                                   static_cast<uint64_t>(kGroupIdx) * groupSize;
        const uint64_t dstOffset = static_cast<uint64_t>(kGroupIdx) * gmmBaseParams->M * groupSize +
                                   static_cast<uint64_t>(rowIdx) * groupSize;
        DataCopyPad(packLocal, xRawGM[srcOffset], copyInParams, padParams);
        PipeBarrier<PIPE_ALL>();
        DataCopyPad(xPackedGM[dstOffset], packLocal, copyOutParams);
        PipeBarrier<PIPE_ALL>();
    }
}

template <typename mmType, bool sync, typename CHANNELDTYPE>
__aicore__ inline void GMMSwigluCompute<mmType, sync, CHANNELDTYPE>::PackGroupQuantXFullK(uint32_t kGroupIdx)
{
    GlobalTensor<int8_t> xRawGM;
    GlobalTensor<int8_t> xPackedGM;
    xRawGM.SetGlobalBuffer((__gm__ int8_t *)xTensorPtr, static_cast<uint64_t>(gmmBaseParams->M) * gmmBaseParams->K);
    xPackedGM.SetGlobalBuffer((__gm__ int8_t *)groupQuantXPackedPtr,
                              static_cast<uint64_t>(gmmBaseParams->M) * gmmBaseParams->K);

    const uint32_t groupSize = gmmSwigluTensorList->groupQuantGroupSize;
    const uint32_t kStart = kGroupIdx * groupSize;
    const uint32_t kEnd = kStart + groupSize;
    const uint64_t totalElements = static_cast<uint64_t>(gmmBaseParams->M) * gmmBaseParams->K;
    for (uint64_t linearIdx = blockIdx; linearIdx < totalElements; linearIdx += aivCoreNum) {
        const uint32_t kIdx = static_cast<uint32_t>(linearIdx % gmmBaseParams->K);
        int8_t value = 0;
        if (kIdx >= kStart && kIdx < kEnd) {
            value = xRawGM.GetValue(linearIdx);
        }
        xPackedGM.SetValue(linearIdx, value);
    }
}

template <typename mmType, bool sync, typename CHANNELDTYPE>
__aicore__ inline void GMMSwigluCompute<mmType, sync, CHANNELDTYPE>::PackGroupQuantWeight(uint32_t expertIdx)
{
    GlobalTensor<int8_t> weightRawGM;
    GlobalTensor<int8_t> weightPackedGM;
    const uint32_t nBlockNum = AlignUp<32>(gmmBaseParams->N) / 32;
    const uint32_t fullKBlockNum = AlignUp<16>(gmmBaseParams->K) / 16;
    const uint32_t groupKBlockNum = AlignUp<16>(gmmSwigluTensorList->groupQuantGroupSize) / 16;
    const uint32_t kGroupNum = gmmSwigluTensorList->groupQuantGroupNum;
    const uint32_t elementsPerNzBlock = 16 * 32;
    const uint64_t totalBlocks = static_cast<uint64_t>(kGroupNum) * nBlockNum * groupKBlockNum;

    weightRawGM.SetGlobalBuffer(GetTensorAddr<int8_t>(expertIdx, weightTensorPtr),
                                static_cast<uint64_t>(nBlockNum) * fullKBlockNum * elementsPerNzBlock);
    weightPackedGM.SetGlobalBuffer((__gm__ int8_t *)groupQuantWeightPackedPtr,
                                   static_cast<uint64_t>(kGroupNum) * nBlockNum * groupKBlockNum * elementsPerNzBlock);

    pipe->InitBuffer(groupQuantWeightPackWorkspace, elementsPerNzBlock * sizeof(int8_t));
    LocalTensor<int8_t> packLocal = groupQuantWeightPackWorkspace.Get<int8_t>();
    DataCopyExtParams copyInParams{1, elementsPerNzBlock, 0, 0, 0};
    DataCopyPadExtParams<int8_t> padParams{false, 0, 0, 0};
    DataCopyParams copyOutParams{
        1, static_cast<uint16_t>(elementsPerNzBlock * sizeof(int8_t) / UB_BLOCK_UNIT_SIZE), 0, 0};
    for (uint64_t block = blockIdx; block < totalBlocks; block += aivCoreNum) {
        uint64_t tmp = block;
        const uint32_t localKBlockIdx = tmp % groupKBlockNum;
        tmp /= groupKBlockNum;
        const uint32_t nBlockIdx = tmp % nBlockNum;
        const uint32_t kGroupIdx = tmp / nBlockNum;
        const uint64_t srcBlock = static_cast<uint64_t>(nBlockIdx) * fullKBlockNum +
                                  static_cast<uint64_t>(kGroupIdx) * groupKBlockNum +
                                  localKBlockIdx;
        const uint64_t dstBlock = static_cast<uint64_t>(kGroupIdx) * nBlockNum * groupKBlockNum +
                                  static_cast<uint64_t>(nBlockIdx) * groupKBlockNum +
                                  localKBlockIdx;
        const uint64_t srcOffset = srcBlock * elementsPerNzBlock;
        const uint64_t dstOffset = dstBlock * elementsPerNzBlock;
        DataCopyPad(packLocal, weightRawGM[srcOffset], copyInParams, padParams);
        PipeBarrier<PIPE_ALL>();
        DataCopy(weightPackedGM[dstOffset], packLocal, copyOutParams);
        PipeBarrier<PIPE_ALL>();
    }
}

template <typename mmType, bool sync, typename CHANNELDTYPE>
__aicore__ inline void GMMSwigluCompute<mmType, sync, CHANNELDTYPE>::PackGroupQuantScale()
{
    GlobalTensor<CHANNELDTYPE> scaleRawGM;
    GlobalTensor<float> scalePackedGM;
    const uint32_t kGroupNum = gmmSwigluTensorList->groupQuantGroupNum;
    const uint32_t tokenLen = gmmSwigluTensorList->tokenLen;
    const uint64_t totalElements = static_cast<uint64_t>(gmmSwigluTensorList->groupListLen) * kGroupNum * tokenLen;
    const uint64_t totalScaleRows = static_cast<uint64_t>(gmmSwigluTensorList->groupListLen) * kGroupNum;
    scalePackedGM.SetGlobalBuffer((__gm__ float *)groupQuantScalePackedPtr, totalElements);
    const uint32_t floatBytes = AlignUp<32>(tokenLen * sizeof(float));
    const uint32_t rawBytes = IsSameType<CHANNELDTYPE, float>::value
                                  ? 0
                                  : AlignUp<32>(tokenLen * sizeof(CHANNELDTYPE));
    pipe->InitBuffer(groupQuantScaleWorkspace, floatBytes + rawBytes);
    LocalTensor<float> scaleLocal = groupQuantScaleWorkspace.Get<float>();
    LocalTensor<CHANNELDTYPE> scaleRawLocal =
        groupQuantScaleWorkspace.GetWithOffset<CHANNELDTYPE>(tokenLen, floatBytes);
    DataCopyExtParams copyInParams{
        1, static_cast<uint32_t>(tokenLen * sizeof(CHANNELDTYPE)), 0, 0, 0};
    DataCopyPadExtParams<CHANNELDTYPE> padParams{false, 0, 0, 0};
    DataCopyParams copyOutParams{1, static_cast<uint16_t>(tokenLen * sizeof(float) / UB_BLOCK_UNIT_SIZE), 0, 0};
    for (uint64_t scaleRow = blockIdx; scaleRow < totalScaleRows; scaleRow += aivCoreNum) {
        const uint32_t kGroupIdx = static_cast<uint32_t>(scaleRow % kGroupNum);
        const uint32_t expertIdx = static_cast<uint32_t>(scaleRow / kGroupNum);
        scaleRawGM.SetGlobalBuffer(GetTensorAddr<CHANNELDTYPE>(expertIdx, perChannelScalePtr),
                                   static_cast<uint64_t>(kGroupNum) * tokenLen);
        const uint64_t srcOffset = static_cast<uint64_t>(kGroupIdx) * tokenLen;
        const uint64_t dstOffset = scaleRow * tokenLen;
        if constexpr (IsSameType<CHANNELDTYPE, float>::value) {
            DataCopyPad(scaleLocal, scaleRawGM[srcOffset], copyInParams, padParams);
        } else {
            DataCopyPad(scaleRawLocal, scaleRawGM[srcOffset], copyInParams, padParams);
            PipeBarrier<PIPE_ALL>();
            Cast(scaleLocal, scaleRawLocal, RoundMode::CAST_NONE, tokenLen);
        }
        PipeBarrier<PIPE_ALL>();
        DataCopy(scalePackedGM[dstOffset], scaleLocal, copyOutParams);
        PipeBarrier<PIPE_ALL>();
    }
    PipeBarrier<PIPE_ALL>();
}

template <typename mmType, bool sync, typename CHANNELDTYPE>
template <typename DTYPE_CS>
__aicore__ inline void GMMSwigluCompute<mmType, sync, CHANNELDTYPE>::PreLoadTokenAndChannel(LocalTensor<float>& channelScaleLocal)
{
    GlobalTensor<CHANNELDTYPE> perChannelScaleTensor;
    perChannelScaleTensor.SetGlobalBuffer(GetTensorAddr<CHANNELDTYPE>(vecConfig.curGroupIdx, perChannelScalePtr));

    DataCopyExtParams copyChannelParams{1, static_cast<uint32_t>(gmmSwigluTensorList->tokenLen * sizeof(DTYPE_CS)), 0, 0, 0};
    DataCopyPadExtParams<DTYPE_CS> padParams{false, 0 ,0, 0};
    if constexpr(!IsSameType<DTYPE_CS, float>::value) {
        LocalTensor<DTYPE_CS> dstLocalT = channelScaleLocal.template ReinterpretCast<DTYPE_CS>();
        DataCopyPad(dstLocalT[gmmSwigluTensorList->tokenLen], perChannelScaleTensor, copyChannelParams, padParams);
        PipeBarrier<PIPE_ALL>();
        Cast(channelScaleLocal, dstLocalT[gmmSwigluTensorList->tokenLen], RoundMode::CAST_NONE, gmmSwigluTensorList->tokenLen);
    } else {
        DataCopyPad(channelScaleLocal, perChannelScaleTensor, copyChannelParams, padParams);
    }
    perChannelScaleInQueue.EnQue(channelScaleLocal);
}

template <typename mmType, bool sync, typename CHANNELDTYPE>
__aicore__ inline void GMMSwigluCompute<mmType, sync, CHANNELDTYPE>::MMCompute(uint32_t groupIdx, MNConfig& mnConfig, uint32_t coreIdx)
{
    uint32_t tailN = mnConfig.nIdx * mnConfig.singleN;
    uint32_t curSingleN = mnConfig.nIdx < mnConfig.blockDimN - 1 ? mnConfig.singleN : mnConfig.n - tailN;
    uint32_t curSingleM = mnConfig.mIdx < mnConfig.blockDimM - 1 ? mnConfig.singleM
                                                                 : mnConfig.m - mnConfig.mIdx * mnConfig.singleM;
    uint64_t xOffset = mnConfig.mIdx * mnConfig.singleM * mnConfig.k;
    if constexpr (transposeX) {
        xOffset = mnConfig.mIdx * mnConfig.singleM;
    }
    uint64_t outOffset = mnConfig.mIdx * mnConfig.singleM * mnConfig.n + tailN;
    xGM.SetGlobalBuffer((__gm__ int8_t *)xTensorPtr + mnConfig.xBaseOffset);
    weightGM.SetGlobalBuffer(GetTensorAddr<int8_t>(groupIdx, weightTensorPtr) + GetWOffset(tailN, mnConfig.k));
    if (mnConfig.blockDimM == 1){
        weightGM.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
    }
    mnConfig.workSpaceOffset = outOffset + mnConfig.yBaseOffset;
    mm.SetOrgShape(mnConfig.m, mnConfig.n, mnConfig.k);
    mm.SetSingleShape(curSingleM, curSingleN, mnConfig.k);
    mm.SetTensorA(xGM[xOffset], transposeX);
    mm.SetTensorB(weightGM, transposeW);
    mm.template IterateAll<sync>(mmOutGM[mnConfig.workSpaceOffset], 0);
}

template <typename mmType, bool sync, typename CHANNELDTYPE>
__aicore__ inline void GMMSwigluCompute<mmType, sync, CHANNELDTYPE>::MMComputeGroupQuant(
    uint32_t expertIdx, uint32_t kGroupIdx, MNConfig& mnConfig, uint32_t coreIdx)
{
    uint32_t tailN = mnConfig.nIdx * mnConfig.singleN;
    uint32_t curSingleN = mnConfig.nIdx < mnConfig.blockDimN - 1 ? mnConfig.singleN : mnConfig.n - tailN;
    uint32_t curSingleM = mnConfig.mIdx < mnConfig.blockDimM - 1 ? mnConfig.singleM
                                                                 : mnConfig.m - mnConfig.mIdx * mnConfig.singleM;
    const uint32_t logicalK = gmmSwigluTensorList->groupQuantGroupSize;
    uint64_t xOffset = static_cast<uint64_t>(kGroupIdx) * gmmBaseParams->M * logicalK +
                       static_cast<uint64_t>(mnConfig.yBaseOffset / mnConfig.n) * logicalK +
                       static_cast<uint64_t>(mnConfig.mIdx) * mnConfig.singleM * logicalK;
    uint64_t weightOffset = static_cast<uint64_t>(kGroupIdx) * AlignUp<32>(mnConfig.n) * AlignUp<16>(logicalK) +
                            static_cast<uint64_t>(tailN) * AlignUp<16>(logicalK);
    uint64_t outOffset = static_cast<uint64_t>(kGroupIdx) * gmmBaseParams->M * mnConfig.n +
                         mnConfig.yBaseOffset +
                         static_cast<uint64_t>(mnConfig.mIdx) * mnConfig.singleM * mnConfig.n +
                         tailN;

    xGM.SetGlobalBuffer((__gm__ int8_t *)(UseDirectFullKGroupX()
                                               ? xTensorPtr
                                               : groupQuantXPackedPtr));
    weightGM.SetGlobalBuffer((__gm__ int8_t *)groupQuantWeightPackedPtr + weightOffset);
    if (mnConfig.blockDimM == 1){
        weightGM.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
    }
    (void)coreIdx;
    mnConfig.workSpaceOffset = outOffset;
    // OrgShape must describe the full A/B matrices for correct stride computation.
    // A (packed x) row stride = logicalK; C (mmOutGM) row stride = n. Using curSingleM
    // here breaks strides when m > 1, so use the group's full m (mnConfig.m) and n.
    mm.SetOrgShape(mnConfig.m, mnConfig.n, logicalK);
    mm.SetSingleShape(curSingleM, curSingleN, logicalK);
    mm.SetTensorA(xGM[xOffset]);
    mm.SetTensorB(weightGM);
    mm.template IterateAll<true>(mmOutGM[mnConfig.workSpaceOffset], 0, false, true);
}

template <typename mmType, bool sync, typename CHANNELDTYPE>
__aicore__ inline void GMMSwigluCompute<mmType, sync, CHANNELDTYPE>::MMComputeGroupQuantOriginalWeight(
    uint32_t expertIdx, uint32_t kGroupIdx, MNConfig& mnConfig, uint32_t coreIdx)
{
    (void)coreIdx;
    uint32_t tailN = mnConfig.nIdx * mnConfig.singleN;
    uint32_t curSingleN = mnConfig.nIdx < mnConfig.blockDimN - 1 ? mnConfig.singleN : mnConfig.n - tailN;
    uint32_t curSingleM = mnConfig.mIdx < mnConfig.blockDimM - 1 ? mnConfig.singleM
                                                                 : mnConfig.m - mnConfig.mIdx * mnConfig.singleM;
    const uint32_t logicalK = gmmSwigluTensorList->groupQuantGroupSize;
    const uint32_t groupKBlockNum = AlignUp<16>(logicalK) / 16;
    const uint32_t elementsPerNzBlock = 16 * 32;
    uint64_t xOffset = static_cast<uint64_t>(kGroupIdx) * gmmBaseParams->M * logicalK +
                       static_cast<uint64_t>(mnConfig.yBaseOffset / mnConfig.n) * logicalK +
                       static_cast<uint64_t>(mnConfig.mIdx) * mnConfig.singleM * logicalK;
    uint64_t weightOffset = GetWOffset(tailN, gmmBaseParams->K) +
                            static_cast<uint64_t>(kGroupIdx) * groupKBlockNum * elementsPerNzBlock;
    uint64_t outOffset = static_cast<uint64_t>(kGroupIdx) * gmmBaseParams->M * mnConfig.n +
                         mnConfig.yBaseOffset +
                         static_cast<uint64_t>(mnConfig.mIdx) * mnConfig.singleM * mnConfig.n +
                         tailN;

    xGM.SetGlobalBuffer((__gm__ int8_t *)(UseDirectFullKGroupX()
                                               ? xTensorPtr
                                               : groupQuantXPackedPtr));
    weightGM.SetGlobalBuffer(GetTensorAddr<int8_t>(expertIdx, weightTensorPtr) + weightOffset);
    if (mnConfig.blockDimM == 1){
        weightGM.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
    }
    mnConfig.workSpaceOffset = outOffset;
    // A is group-packed with stride logicalK; B remains in the original NZ tensor with full-K
    // stride; C keeps the full N row stride.
    mm.SetOrgShape(mnConfig.m, mnConfig.n, logicalK, gmmBaseParams->K, mnConfig.n);
    mm.SetSingleShape(curSingleM, curSingleN, logicalK);
    mm.SetTensorA(xGM[xOffset]);
    mm.SetTensorB(weightGM);
    mm.template IterateAll<true>(mmOutGM[mnConfig.workSpaceOffset], 0, false, true);
}

template <typename mmType, bool sync, typename CHANNELDTYPE>
__aicore__ inline void GMMSwigluCompute<mmType, sync, CHANNELDTYPE>::MMComputeGroupQuantFullK(
    uint32_t expertIdx, MNConfig& mnConfig, uint32_t coreIdx)
{
    (void)expertIdx;
    (void)coreIdx;
    uint32_t tailN = mnConfig.nIdx * mnConfig.singleN;
    uint32_t curSingleN = mnConfig.nIdx < mnConfig.blockDimN - 1 ? mnConfig.singleN : mnConfig.n - tailN;
    uint32_t curSingleM = mnConfig.mIdx < mnConfig.blockDimM - 1 ? mnConfig.singleM
                                                                 : mnConfig.m - mnConfig.mIdx * mnConfig.singleM;
    uint64_t xOffset = static_cast<uint64_t>(mnConfig.yBaseOffset / mnConfig.n) * gmmBaseParams->K +
                       static_cast<uint64_t>(mnConfig.mIdx) * mnConfig.singleM * gmmBaseParams->K;
    uint64_t outOffset = mnConfig.yBaseOffset +
                         static_cast<uint64_t>(mnConfig.mIdx) * mnConfig.singleM * mnConfig.n +
                         tailN;

    xGM.SetGlobalBuffer((__gm__ int8_t *)groupQuantXPackedPtr);
    weightGM.SetGlobalBuffer(GetTensorAddr<int8_t>(expertIdx, weightTensorPtr) + GetWOffset(tailN, gmmBaseParams->K));
    if (mnConfig.blockDimM == 1){
        weightGM.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
    }
    mnConfig.workSpaceOffset = outOffset;
    mm.SetOrgShape(curSingleM, curSingleN, gmmBaseParams->K);
    mm.SetSingleShape(curSingleM, curSingleN, gmmBaseParams->K);
    mm.SetTensorA(xGM[xOffset]);
    mm.SetTensorB(weightGM);
    mm.template IterateAll<true>(mmOutGM[mnConfig.workSpaceOffset], 0, false, true);
}

template <typename mmType, bool sync, typename CHANNELDTYPE>
__aicore__ inline void GMMSwigluCompute<mmType, sync, CHANNELDTYPE>::UpdateMnConfig(MNConfig &mnConfig) {
    if constexpr (B::format == CubeFormat::NZ) {
        mnConfig.wBaseOffset += AlignUp<16>(mnConfig.k) * AlignUp<32>(mnConfig.n);  // 16: nz format last two dim size
    } else {
        mnConfig.wBaseOffset += mnConfig.k * mnConfig.n;
    }
    mnConfig.nAxisBaseOffset += mnConfig.n;
    mnConfig.mAxisBaseOffset += mnConfig.m;
    mnConfig.xBaseOffset += mnConfig.m * mnConfig.k;
    mnConfig.yBaseOffset += mnConfig.m * mnConfig.n;
}

template <typename mmType, bool sync, typename CHANNELDTYPE>
__aicore__ inline void GMMSwigluCompute<mmType, sync, CHANNELDTYPE>::SetMNConfig(const int32_t splitValue, const uint32_t groupIdx, MNConfig &mnConfig) {
    SetMKN(splitValue, groupIdx, mnConfig);
    mnConfig.baseM = BASIC_M;
    mnConfig.baseN = BASIC_N;
    mnConfig.singleM = SINGLE_CORE_M;
    mnConfig.singleN = SINGLE_CORE_N;
}

template <typename mmType, bool sync, typename CHANNELDTYPE>
__aicore__ inline void GMMSwigluCompute<mmType, sync, CHANNELDTYPE>::SetMKN(const int32_t splitValue, const uint32_t groupIdx, MNConfig &mnConfig)
{
	mnConfig.m = static_cast<uint32_t>(splitValue);
    mnConfig.k = gmmBaseParams->K; // tilingData
    mnConfig.n = gmmBaseParams->N; // tilingData
}

template <typename mmType, bool sync, typename CHANNELDTYPE>
__aicore__ inline uint64_t GMMSwigluCompute<mmType, sync, CHANNELDTYPE>::GetWOffset(uint32_t tailN, uint32_t k) {
    uint64_t wOffset = 0;
    if constexpr (mmType::BT::format == CubeFormat::NZ) {
        wOffset = tailN * AlignUp<16>(k);  // 16: nz format last two dim size
    } else {
        wOffset = tailN;
    }
    return wOffset;
}

template <typename mmType, bool sync, typename CHANNELDTYPE>
__aicore__ inline void GMMSwigluCompute<mmType, sync, CHANNELDTYPE>::MNBlockIdxCompute(MNConfig &mnConfig, const uint32_t curBlock,
    const uint32_t count, const uint32_t thresholdM_dimN) {
	mnConfig.mIdx = (curBlock - count) / mnConfig.blockDimN;
    mnConfig.nIdx = (curBlock - count) % mnConfig.blockDimN;
}

template <typename mmType, bool sync, typename CHANNELDTYPE>
__aicore__ inline void GMMSwigluCompute<mmType, sync, CHANNELDTYPE>::UpdateVecConfig(uint32_t blockIdx, VecConfig& vecConfig)
{
    // Step 1: Read grouplist reduceSum to calculate total data count
    vecConfig.M = 0;
    vecConfig.usedCoreNum = 0;
    vecConfig.startOffset = 0;
    vecConfig.curOffset = 0;
    vecConfig.startIdx = 0;
    vecConfig.curIdx = 0;
    vecConfig.taskNum = 0;
    vecConfig.curGroupIdx = 0;
    vecConfig.outLoopNum = 0;
    vecConfig.innerLoopNum = 0;
    vecConfig.tailLoopNum = 0;
    vecConfig.nextUpadteInterVal = 0;

    int64_t prevM = 0;
    for (uint32_t groupIdx = 0; groupIdx < gmmSwigluTensorList->groupListLen; groupIdx++){
        int64_t currM = groupListGM.GetValue(groupIdx);
        int64_t tempM = currM - prevM;
        prevM = currM;
        vecConfig.M += tempM;
    }
    // Step 2: Calculate core allocation
    uint32_t eachCoreTaskNum = (vecConfig.M + aivCoreNum - 1) / aivCoreNum;
    vecConfig.usedCoreNum = vecConfig.M >= aivCoreNum ? aivCoreNum : vecConfig.M;
    uint32_t tailCoreIdx = vecConfig.M - (eachCoreTaskNum - 1) * vecConfig.usedCoreNum;
    vecConfig.taskNum = blockIdx < tailCoreIdx ? eachCoreTaskNum : eachCoreTaskNum - 1;
    vecConfig.startIdx = blockIdx < tailCoreIdx
                            ? eachCoreTaskNum * blockIdx
                            :((eachCoreTaskNum - 1) * blockIdx + tailCoreIdx);
    if (gmmSwigluTensorList->enableGroupQuant && GROUP_QUANT_DEBUG_STAGE >= 9 &&
        GROUP_QUANT_DEBUG_STAGE != 54) {
        vecConfig.usedCoreNum = 2;
        vecConfig.taskNum = blockIdx == 0 ? vecConfig.M : 0;
        vecConfig.startIdx = 0;
    }
    vecConfig.curIdx = vecConfig.startIdx;
    vecConfig.startOffset = vecConfig.startIdx * gmmSwigluTensorList->tokenLen;
    vecConfig.curOffset = vecConfig.startOffset;
    int64_t curStartIdx = vecConfig.startIdx;
    prevM = 0;
    for (uint32_t groupIdx = 0; groupIdx < gmmSwigluTensorList->groupListLen; groupIdx++){
        int64_t currM = groupListGM.GetValue(groupIdx);
        int64_t tempM = currM - prevM;
        prevM = currM;
        if (curStartIdx >= 0 && curStartIdx - tempM < 0) {
            vecConfig.curGroupIdx = groupIdx;
            vecConfig.nextUpadteInterVal = tempM - curStartIdx;
        }
        curStartIdx -= tempM;
    }
    // Step 3: Calculate total data volume
    vecConfig.outLoopNum = (vecConfig.taskNum + gmmSwigluTensorList->maxProcessRowNum - 1) / gmmSwigluTensorList->maxProcessRowNum;
    vecConfig.tailLoopNum = vecConfig.taskNum % gmmSwigluTensorList->maxProcessRowNum
                            ? vecConfig.taskNum % gmmSwigluTensorList->maxProcessRowNum
                            : gmmSwigluTensorList->maxProcessRowNum;
    pipe->Reset();
    // Step 4: Allocate space
    pipe->InitBuffer(mmOutQueue, 1, gmmSwigluTensorList->maxProcessRowNum * gmmSwigluTensorList->tokenLen * sizeof(int32_t));
    pipe->InitBuffer(perChannelScaleInQueue, 1, gmmSwigluTensorList->tokenLen * sizeof(float));
    pipe->InitBuffer(quantOutQueue, 1, gmmSwigluTensorList->maxProcessRowNum * gmmSwigluTensorList->tokenLen / 2 * sizeof(int8_t));
    pipe->InitBuffer(quantScaleOutQueue, 1, AlignUp<int32_t>(gmmSwigluTensorList->maxProcessRowNum, 8) * sizeof(float));
    const uint32_t reduceWorkElements = AlignUp<8>(gmmSwigluTensorList->tokenLen / BISECT) +
                                        FLOAT_UB_BLOCK_UNIT_SIZE * 2;
    pipe->InitBuffer(reduceWorkspace,
                     (reduceWorkElements > 1024 ? reduceWorkElements : 1024) * sizeof(float));
    pipe->InitBuffer(castWorkspace, 32 * sizeof(int8_t));
    if (gmmSwigluTensorList->enableGroupQuant) {
        pipe->InitBuffer(groupQuantAccumWorkspace, gmmSwigluTensorList->tokenLen * sizeof(float));
        pipe->InitBuffer(groupQuantScaleWorkspace, gmmSwigluTensorList->tokenLen * sizeof(float));
    }
}

template <typename mmType, bool sync, typename CHANNELDTYPE>
__aicore__ inline void GMMSwigluCompute<mmType, sync, CHANNELDTYPE>::customDataCopyIn(uint32_t outLoopIdx)
{
    LocalTensor<int32_t> _inMMLocal_0 = mmOutQueue.DeQue<int32_t>();
    DataCopyExtParams copyParams_0{1, static_cast<uint32_t>(vecConfig.innerLoopNum * gmmSwigluTensorList->tokenLen * sizeof(int32_t)), 0, 0, 0};
    DataCopyPadExtParams<int32_t> padParams_0{false, 0 ,0, 0};
    DataCopyPad(_inMMLocal_0, mmOutGM[vecConfig.curOffset], copyParams_0, padParams_0);

    mmOutQueue.EnQue(_inMMLocal_0);

    LocalTensor<int32_t> _inMMLocal_1 = mmOutQueue.DeQue<int32_t>();

    Cast(_inMMLocal_1.ReinterpretCast<float>(), _inMMLocal_1, RoundMode::CAST_NONE, vecConfig.innerLoopNum * gmmSwigluTensorList->tokenLen);

    mmOutQueue.EnQue(_inMMLocal_1);
    LocalTensor<float> _inMMLocal_2 = mmOutQueue.DeQue<float>();
    set_flag(PIPE_S, PIPE_V, EVENT_ID0);
    for (uint32_t i = 0; i < vecConfig.innerLoopNum; i++){
        wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
        float scale = perTokenScaleGM.GetValue(vecConfig.curIdx);
        set_flag(PIPE_S, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
        Muls(_inMMLocal_2[i * gmmSwigluTensorList->tokenLen], _inMMLocal_2[i * gmmSwigluTensorList->tokenLen], scale, gmmSwigluTensorList->tokenLen);
        set_flag(PIPE_S, PIPE_V, EVENT_ID0);
        vecConfig.curIdx++;
    }
    wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
    vecConfig.curOffset = vecConfig.curIdx * gmmSwigluTensorList->tokenLen;
    mmOutQueue.EnQue(_inMMLocal_2);
}

template <typename mmType, bool sync, typename CHANNELDTYPE>
template <typename DTYPE_CS>
__aicore__ inline void GMMSwigluCompute<mmType, sync, CHANNELDTYPE>::customDataCopyInGroupQuant(uint32_t outLoopIdx)
{
    (void)outLoopIdx;
    if (unlikely(vecConfig.nextUpadteInterVal == 0)) {
        while (vecConfig.curGroupIdx + 1 < gmmSwigluTensorList->groupListLen) {
            int64_t curTemp = groupListGM.GetValue(vecConfig.curGroupIdx);
            vecConfig.curGroupIdx++;
            int64_t nextTemp = groupListGM.GetValue(vecConfig.curGroupIdx);
            if(nextTemp != curTemp){
                vecConfig.nextUpadteInterVal = nextTemp - curTemp;
                break;
            }
        }
    }

    LocalTensor<int32_t> mmLocalInt = mmOutQueue.DeQue<int32_t>();
    LocalTensor<float> mmLocal = mmLocalInt.template ReinterpretCast<float>();
    if constexpr (GROUP_QUANT_DEBUG_STAGE < 3) {
        Duplicate(mmLocal, static_cast<float>(1.0), gmmSwigluTensorList->tokenLen);
        PipeBarrier<PIPE_V>();
        vecConfig.curIdx++;
        vecConfig.curOffset = vecConfig.curIdx * gmmSwigluTensorList->tokenLen;
        vecConfig.nextUpadteInterVal--;
        mmOutQueue.EnQue(mmLocal);
        return;
    }
    if constexpr (GROUP_QUANT_DEBUG_STAGE == 8 || GROUP_QUANT_DEBUG_STAGE == 54) {
        const uint32_t tokenLen = gmmSwigluTensorList->tokenLen;
        if constexpr (GROUP_QUANT_DEBUG_STAGE >= 8) {
            GlobalTensor<int8_t> xRawGM;
            GlobalTensor<int8_t> weightRawGM;
            GlobalTensor<CHANNELDTYPE> scaleRawGM;
            const uint32_t groupSize = gmmSwigluTensorList->groupQuantGroupSize;
            const uint32_t kGroupNum = gmmSwigluTensorList->groupQuantGroupNum;
            const uint32_t fullKBlockNum = AlignUp<16>(gmmBaseParams->K) / 16;
            const uint32_t elementsPerNzBlock = 16 * 32;
            xRawGM.SetGlobalBuffer((__gm__ int8_t *)xTensorPtr,
                                   static_cast<uint64_t>(gmmBaseParams->M) * gmmBaseParams->K);
            for (uint32_t rowIdx = 0; rowIdx < vecConfig.innerLoopNum; ++rowIdx) {
                if (unlikely(vecConfig.nextUpadteInterVal == 0)) {
                    while (vecConfig.curGroupIdx + 1 < gmmSwigluTensorList->groupListLen) {
                        int64_t curTemp = groupListGM.GetValue(vecConfig.curGroupIdx);
                        vecConfig.curGroupIdx++;
                        int64_t nextTemp = groupListGM.GetValue(vecConfig.curGroupIdx);
                        if (nextTemp != curTemp) {
                            vecConfig.nextUpadteInterVal = nextTemp - curTemp;
                            break;
                        }
                    }
                }
                weightRawGM.SetGlobalBuffer(GetTensorAddr<int8_t>(vecConfig.curGroupIdx, weightTensorPtr),
                                            static_cast<uint64_t>(AlignUp<32>(gmmBaseParams->N) / 32) *
                                            fullKBlockNum * elementsPerNzBlock);
                scaleRawGM.SetGlobalBuffer(GetTensorAddr<CHANNELDTYPE>(vecConfig.curGroupIdx, perChannelScalePtr),
                                           static_cast<uint64_t>(kGroupNum) * tokenLen);
                const uint64_t localRowOffset = static_cast<uint64_t>(rowIdx) * tokenLen;
                const uint64_t globalRowOffset = static_cast<uint64_t>(vecConfig.curIdx) * gmmBaseParams->K;
                const float tokenScale = perTokenScaleGM.GetValue(vecConfig.curIdx);
                for (uint32_t channelIdx = 0; channelIdx < tokenLen; ++channelIdx) {
                    float acc = 0.0f;
                    const uint32_t nBlockIdx = channelIdx / 32;
                    const uint32_t nInner = channelIdx - nBlockIdx * 32;
                    for (uint32_t kGroupIdx = 0; kGroupIdx < kGroupNum; ++kGroupIdx) {
                        int32_t partial = 0;
                        const uint32_t kStart = kGroupIdx * groupSize;
                        for (uint32_t localK = 0; localK < groupSize; ++localK) {
                            const uint32_t kIdx = kStart + localK;
                            const uint32_t kBlockIdx = kIdx / 16;
                            const uint32_t kInner = kIdx - kBlockIdx * 16;
                            const uint64_t xOffset = globalRowOffset + kIdx;
                            const uint64_t wOffset = (static_cast<uint64_t>(nBlockIdx) * fullKBlockNum + kBlockIdx) *
                                                     elementsPerNzBlock +
                                                     static_cast<uint64_t>(kInner) * 32 + nInner;
                            partial += static_cast<int32_t>(xRawGM.GetValue(xOffset)) *
                                       static_cast<int32_t>(weightRawGM.GetValue(wOffset));
                        }
                        const float weightScale = GetGroupQuantScaleValue(
                            scaleRawGM, static_cast<uint64_t>(kGroupIdx) * tokenLen + channelIdx);
                        acc += static_cast<float>(partial) * weightScale * tokenScale;
                    }
                    mmLocal.SetValue(localRowOffset + channelIdx, acc);
                }
                vecConfig.curIdx++;
                vecConfig.nextUpadteInterVal--;
            }
            PipeBarrier<PIPE_ALL>();
            vecConfig.curOffset = vecConfig.curIdx * tokenLen;
            mmOutQueue.EnQue(mmLocal);
            return;
        }
        GlobalTensor<float> accumGM;
        const uint64_t rowOffset = static_cast<uint64_t>(vecConfig.curIdx) * tokenLen;
        accumGM.SetGlobalBuffer((__gm__ float *)groupQuantAccumPtr,
                                static_cast<uint64_t>(gmmBaseParams->M) * tokenLen);
        DataCopyParams copyParams{1, static_cast<uint16_t>(tokenLen * sizeof(float) / UB_BLOCK_UNIT_SIZE), 0, 0};
        DataCopy(mmLocal, accumGM[rowOffset], copyParams);
        PipeBarrier<PIPE_ALL>();
        vecConfig.curIdx++;
        vecConfig.curOffset = vecConfig.curIdx * tokenLen;
        vecConfig.nextUpadteInterVal--;
        mmOutQueue.EnQue(mmLocal);
        return;
    }
    if constexpr (GROUP_QUANT_DEBUG_STAGE == 54) {
        const uint32_t tokenLen = gmmSwigluTensorList->tokenLen;
        GlobalTensor<float> accumGM;
        accumGM.SetGlobalBuffer((__gm__ float *)groupQuantAccumPtr,
                                static_cast<uint64_t>(gmmBaseParams->M) * tokenLen);
        const uint64_t rowOffset = static_cast<uint64_t>(vecConfig.curIdx) * tokenLen;
        const uint32_t copyElements = vecConfig.innerLoopNum * tokenLen;
        DataCopyParams copyParams{
            1,
            static_cast<uint16_t>(copyElements * sizeof(float) / UB_BLOCK_UNIT_SIZE),
            0,
            0
        };
        DataCopy(mmLocal, accumGM[rowOffset], copyParams);
        PipeBarrier<PIPE_ALL>();
        for (uint32_t rowLocal = 0; rowLocal < vecConfig.innerLoopNum; ++rowLocal) {
            if (unlikely(vecConfig.nextUpadteInterVal == 0)) {
                while (vecConfig.curGroupIdx + 1 < gmmSwigluTensorList->groupListLen) {
                    int64_t curTemp = groupListGM.GetValue(vecConfig.curGroupIdx);
                    vecConfig.curGroupIdx++;
                    int64_t nextTemp = groupListGM.GetValue(vecConfig.curGroupIdx);
                    if (nextTemp != curTemp) {
                        vecConfig.nextUpadteInterVal = nextTemp - curTemp;
                        break;
                    }
                }
            }
            vecConfig.curIdx++;
            vecConfig.nextUpadteInterVal--;
        }
        vecConfig.curOffset = vecConfig.curIdx * tokenLen;
        mmOutQueue.EnQue(mmLocal);
        return;
    }
    if constexpr (GROUP_QUANT_DEBUG_STAGE >= 9) {
        const uint32_t tokenLen = gmmSwigluTensorList->tokenLen;
        GlobalTensor<float> accumGM;
        accumGM.SetGlobalBuffer((__gm__ float *)groupQuantAccumPtr,
                                static_cast<uint64_t>(gmmBaseParams->M) * tokenLen);
        LocalTensor<float> accumLocal = groupQuantAccumWorkspace.Get<float>();
        // probe: scalar read accumGM before DataCopy
        float probeScalar = accumGM.GetValue(0);
        quantScaleOutputGM.SetValue(0, probeScalar);
        DataCopyParams copyParams{1, static_cast<uint16_t>(tokenLen * sizeof(float) / UB_BLOCK_UNIT_SIZE), 0, 0};
        for (uint32_t rowLocal = 0; rowLocal < vecConfig.innerLoopNum; ++rowLocal) {
            if (unlikely(vecConfig.nextUpadteInterVal == 0)) {
                while (vecConfig.curGroupIdx + 1 < gmmSwigluTensorList->groupListLen) {
                    int64_t curTemp = groupListGM.GetValue(vecConfig.curGroupIdx);
                    vecConfig.curGroupIdx++;
                    int64_t nextTemp = groupListGM.GetValue(vecConfig.curGroupIdx);
                    if (nextTemp != curTemp) {
                        vecConfig.nextUpadteInterVal = nextTemp - curTemp;
                        break;
                    }
                }
            }
            const uint64_t rowOffset = static_cast<uint64_t>(vecConfig.curIdx) * tokenLen;
            DataCopy(accumLocal, accumGM[rowOffset], copyParams);
            PipeBarrier<PIPE_ALL>();
            // probe: check accumLocal after DataCopy
            float probeLocal = accumLocal.GetValue(0);
            quantOutputGM.SetValue(0, static_cast<int8_t>(probeLocal != 0.0f ? 55 : 0));
            const uint32_t localOffset = rowLocal * tokenLen;
            for (uint32_t channelIdx = 0; channelIdx < tokenLen; ++channelIdx) {
                mmLocal.SetValue(localOffset + channelIdx, accumLocal.GetValue(channelIdx));
            }
            vecConfig.curIdx++;
            vecConfig.nextUpadteInterVal--;
        }
        PipeBarrier<PIPE_ALL>();
        vecConfig.curOffset = vecConfig.curIdx * tokenLen;
        mmOutQueue.EnQue(mmLocal);
        return;
    }
    LocalTensor<float> accLocal = groupQuantAccumWorkspace.Get<float>();
    Duplicate(accLocal, static_cast<float>(0.0), gmmSwigluTensorList->tokenLen);
    PipeBarrier<PIPE_V>();

    const uint64_t rowOffset = static_cast<uint64_t>(vecConfig.curIdx) * gmmSwigluTensorList->tokenLen;
    const uint32_t tokenLen = gmmSwigluTensorList->tokenLen;
    const uint32_t kGroupNum = 1;
    const float tokenScale = perTokenScaleGM.GetValue(vecConfig.curIdx);

    LocalTensor<float> scaleLocal = groupQuantScaleWorkspace.Get<float>();
    GlobalTensor<float> scalePackedGM;
    scalePackedGM.SetGlobalBuffer((__gm__ float *)groupQuantScalePackedPtr,
                                  static_cast<uint64_t>(gmmSwigluTensorList->groupListLen) * kGroupNum * tokenLen);

    for (uint32_t kGroupIdx = 0; kGroupIdx < kGroupNum; ++kGroupIdx) {
        DataCopyExtParams mmCopyParams{1, static_cast<uint32_t>(tokenLen * sizeof(int32_t)), 0, 0, 0};
        DataCopyPadExtParams<int32_t> mmPadParams{false, 0, 0, 0};
        uint64_t partialOffset = static_cast<uint64_t>(kGroupIdx) * gmmBaseParams->M * tokenLen + rowOffset;
        DataCopyPad(mmLocalInt, mmOutGM[partialOffset], mmCopyParams, mmPadParams);
        PipeBarrier<PIPE_ALL>();
        Cast(mmLocal, mmLocalInt, RoundMode::CAST_NONE, tokenLen);

        DataCopyExtParams scaleCopyParams{1, static_cast<uint32_t>(tokenLen * sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> scalePadParams{false, 0, 0, 0};
        uint64_t scaleOffset = (static_cast<uint64_t>(vecConfig.curGroupIdx) * kGroupNum + kGroupIdx) * tokenLen;
        DataCopyPad(scaleLocal, scalePackedGM[scaleOffset], scaleCopyParams, scalePadParams);
        PipeBarrier<PIPE_ALL>();
        Mul(mmLocal, mmLocal, scaleLocal, tokenLen);
        PipeBarrier<PIPE_V>();
        Muls(mmLocal, mmLocal, tokenScale, tokenLen);
        PipeBarrier<PIPE_V>();
        Add(accLocal, accLocal, mmLocal, tokenLen);
        PipeBarrier<PIPE_V>();
    }
    Adds(mmLocal, accLocal, static_cast<float>(0.0), tokenLen);
    PipeBarrier<PIPE_V>();
    vecConfig.curIdx++;
    vecConfig.curOffset = vecConfig.curIdx * tokenLen;
    vecConfig.nextUpadteInterVal--;
    mmOutQueue.EnQue(mmLocal);
}

template <typename mmType, bool sync, typename CHANNELDTYPE>
template <typename DTYPE_CS>
__aicore__ inline void GMMSwigluCompute<mmType, sync, CHANNELDTYPE>::UpdateChannelScale(uint32_t loopIdx){
    // Update perChannel
    if (unlikely(vecConfig.nextUpadteInterVal == 0)) {
        int64_t loop = gmmSwigluTensorList->groupListLen - vecConfig.curGroupIdx;
        while (loop--) {
            int64_t curTemp = groupListGM.GetValue(vecConfig.curGroupIdx);
            vecConfig.curGroupIdx++;
            int64_t nextTemp = groupListGM.GetValue(vecConfig.curGroupIdx);
            if(nextTemp != curTemp){
                vecConfig.nextUpadteInterVal = nextTemp - curTemp;
                break;
            }
        }
        LocalTensor<float> _inChannel = perChannelScaleInQueue.DeQue<float>();
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(gmmSwigluTensorList->tokenLen * sizeof(DTYPE_CS)), 0, 0, 0};
        DataCopyPadExtParams<DTYPE_CS> padParams{false, 0 ,0, 0};

        GlobalTensor<CHANNELDTYPE> perChannelScaleTensor;
        perChannelScaleTensor.SetGlobalBuffer(GetTensorAddr<CHANNELDTYPE>(vecConfig.curGroupIdx, perChannelScalePtr));

        if constexpr(!IsSameType<DTYPE_CS, float>::value) {
            LocalTensor<DTYPE_CS> dstLocalT = _inChannel.template ReinterpretCast<DTYPE_CS>();
            DataCopyPad(dstLocalT[gmmSwigluTensorList->tokenLen], perChannelScaleTensor, copyParams, padParams);
            PipeBarrier<PIPE_ALL>();
            Cast(_inChannel, dstLocalT[gmmSwigluTensorList->tokenLen], RoundMode::CAST_NONE, gmmSwigluTensorList->tokenLen);
        } else {
            DataCopyPad(_inChannel, perChannelScaleTensor, copyParams, padParams);
        }
        PipeBarrier<PIPE_ALL>();
        perChannelScaleInQueue.EnQue(_inChannel);
    }
}

template <typename mmType, bool sync, typename CHANNELDTYPE>
__aicore__ inline void GMMSwigluCompute<mmType, sync, CHANNELDTYPE>::VectorCompute(uint32_t loopIdx) {
        Dequant(loopIdx);
        Swiglu(loopIdx);
        Quant(loopIdx);
}

template <typename mmType, bool sync, typename CHANNELDTYPE>
__aicore__ inline void GMMSwigluCompute<mmType, sync, CHANNELDTYPE>::Dequant(uint32_t loopIdx) {
    LocalTensor<float> mmLocal = mmOutQueue.DeQue<float>();
    LocalTensor<float> perChannelLocal = perChannelScaleInQueue.DeQue<float>();

    Mul(mmLocal[loopIdx * gmmSwigluTensorList->tokenLen],
        mmLocal[loopIdx * gmmSwigluTensorList->tokenLen],
        perChannelLocal,
        gmmSwigluTensorList->tokenLen);
    PipeBarrier<PIPE_V>();

    vecConfig.nextUpadteInterVal--;
    mmOutQueue.EnQue(mmLocal);
    perChannelScaleInQueue.EnQue(perChannelLocal);
}

template <typename mmType, bool sync, typename CHANNELDTYPE>
__aicore__ inline void GMMSwigluCompute<mmType, sync, CHANNELDTYPE>::Swiglu(uint32_t loopIdx) {
    // High-level API swiglu
    LocalTensor<float> _inMMLocal = mmOutQueue.DeQue<float>();
    float beta = 1.0f;
    LocalTensor<float> workspaceLocal= reduceWorkspace.Get<float>();
    LocalTensor<float> src0Local = _inMMLocal[loopIdx * gmmSwigluTensorList->tokenLen + gmmSwigluTensorList->tokenLen / 2];
    LocalTensor<float> src1Local = _inMMLocal[loopIdx * gmmSwigluTensorList->tokenLen];
    if(gmmSwigluTensorList->swigluLimit>1e-8f){
        Mins(src0Local, src0Local, gmmSwigluTensorList->swigluLimit, gmmSwigluTensorList->tokenLen / 2);
        PipeBarrier<PIPE_V>();
        Maxs(src0Local, src0Local, (-1.0f * gmmSwigluTensorList->swigluLimit), gmmSwigluTensorList->tokenLen / 2);
        PipeBarrier<PIPE_V>();

        Mins(src1Local, src1Local, gmmSwigluTensorList->swigluLimit, gmmSwigluTensorList->tokenLen / 2);
        PipeBarrier<PIPE_V>();
    }
    LocalTensor<int8_t> probeBytes;
    LocalTensor<float> probeLocal;
    if constexpr (GROUP_QUANT_DEBUG_STAGE == 56) {
        const uint32_t halfTokenLen = gmmSwigluTensorList->tokenLen / BISECT;
        if (gmmBaseParams->M == 2 && halfTokenLen == VEC_LEN_ONCE_REPEAT_ELE * 2) {
            probeBytes = quantOutQueue.DeQue<int8_t>();
            probeLocal = probeBytes.template ReinterpretCast<float>();
            const uint32_t probeRowOffset = loopIdx * halfTokenLen / sizeof(float);
            Adds(probeLocal[probeRowOffset], src0Local[16], 0.0F, FLOAT_UB_BLOCK_UNIT_SIZE);
            Adds(probeLocal[probeRowOffset + FLOAT_UB_BLOCK_UNIT_SIZE], src1Local[16], 0.0F,
                 FLOAT_UB_BLOCK_UNIT_SIZE);
            PipeBarrier<PIPE_V>();
        }
    }
    SwiGLU<float, false>(workspaceLocal, src0Local, src1Local, beta, gmmSwigluTensorList->tokenLen / 2);
    PipeBarrier<PIPE_ALL>();
    DataCopyParams repeatParams{1, static_cast<uint16_t>((gmmSwigluTensorList->tokenLen / 2) / 8), 0, 0};
    DataCopy(_inMMLocal[loopIdx * gmmSwigluTensorList->tokenLen], workspaceLocal, repeatParams);
    if constexpr (GROUP_QUANT_DEBUG_STAGE == 56) {
        const uint32_t halfTokenLen = gmmSwigluTensorList->tokenLen / BISECT;
        if (gmmBaseParams->M == 2 && halfTokenLen == VEC_LEN_ONCE_REPEAT_ELE * 2) {
            // Blocks: src0, src1, SwiGLU output, and a vector-to-MTE control marker.
            const uint32_t probeRowOffset = loopIdx * halfTokenLen / sizeof(float);
            Adds(probeLocal[probeRowOffset + FLOAT_UB_BLOCK_UNIT_SIZE * 2], workspaceLocal[16], 0.0F,
                 FLOAT_UB_BLOCK_UNIT_SIZE);
            Duplicate(probeLocal[probeRowOffset + FLOAT_UB_BLOCK_UNIT_SIZE * 3], 1234.0F,
                      FLOAT_UB_BLOCK_UNIT_SIZE);
            PipeBarrier<PIPE_V>();
            quantOutQueue.EnQue(probeBytes);
        }
    }
    mmOutQueue.EnQue(_inMMLocal);
}

template <typename mmType, bool sync, typename CHANNELDTYPE>
__aicore__ inline void GMMSwigluCompute<mmType, sync, CHANNELDTYPE>::Quant(uint32_t loopIdx, uint32_t scaleGmRow) {
    LocalTensor<float> _inMMLocal = mmOutQueue.DeQue<float>();
    if constexpr (GROUP_QUANT_DEBUG_STAGE == 56) {
        const uint32_t halfTokenLen = gmmSwigluTensorList->tokenLen / BISECT;
        if (gmmBaseParams->M == 2 && halfTokenLen == VEC_LEN_ONCE_REPEAT_ELE * 2) {
            // SwiGLU already placed all input/output probe blocks in quantOutQueue.
            mmOutQueue.EnQue(_inMMLocal);
            return;
        }
    }
    Abs(_inMMLocal[loopIdx * gmmSwigluTensorList->tokenLen + gmmSwigluTensorList->tokenLen / BISECT],
        _inMMLocal[loopIdx * gmmSwigluTensorList->tokenLen],
        gmmSwigluTensorList->tokenLen / BISECT);
    const uint32_t halfTokenLen = gmmSwigluTensorList->tokenLen / BISECT;
    const uint32_t reduceSrcOffset = loopIdx * gmmSwigluTensorList->tokenLen + halfTokenLen;
    PipeBarrier<PIPE_V>();
    float quantScale = 0.0F;
    float stage56FirstMax = 0.0F;
    float stage56SecondMax = 0.0F;
    if constexpr (GROUP_QUANT_DEBUG_STAGE == 56) {
        const uint32_t reduceWorkElements = AlignUp<8>(halfTokenLen);
        LocalTensor<float> reduceWorkLocal = reduceWorkspace.Get<float>(reduceWorkElements);
        LocalTensor<float> reduceResLocal =
            reduceWorkspace.GetWithOffset<float>(FLOAT_UB_BLOCK_UNIT_SIZE, reduceWorkElements * sizeof(float));
        LocalTensor<float> reduceTmpLocal = reduceWorkspace.GetWithOffset<float>(
            FLOAT_UB_BLOCK_UNIT_SIZE,
            reduceWorkElements * sizeof(float) + UB_BLOCK_UNIT_SIZE);
        if (halfTokenLen == VEC_LEN_ONCE_REPEAT_ELE * 2) {
            Max(reduceWorkLocal, _inMMLocal[reduceSrcOffset],
                _inMMLocal[reduceSrcOffset + VEC_LEN_ONCE_REPEAT_ELE], VEC_LEN_ONCE_REPEAT_ELE);
            PipeBarrier<PIPE_V>();
            Max(reduceWorkLocal, reduceWorkLocal, reduceWorkLocal[32], 32);
            PipeBarrier<PIPE_V>();
            Max(reduceWorkLocal, reduceWorkLocal, reduceWorkLocal[16], 16);
            PipeBarrier<PIPE_V>();
            Max(reduceWorkLocal, reduceWorkLocal, reduceWorkLocal[8], 8);
            PipeBarrier<PIPE_V>();
            event_t evtVToS = static_cast<event_t>(pipe->FetchEventID(HardEvent::V_S));
            SetFlag<HardEvent::V_S>(evtVToS);
            WaitFlag<HardEvent::V_S>(evtVToS);
            float firstMax = reduceWorkLocal.GetValue(0);
            for (uint32_t i = 1; i < FLOAT_UB_BLOCK_UNIT_SIZE; ++i) {
                float candidate = reduceWorkLocal.GetValue(i);
                firstMax = firstMax > candidate ? firstMax : candidate;
            }
            float secondMax = 0.0F;
            stage56FirstMax = firstMax;
            stage56SecondMax = secondMax;
            quantScale = firstMax / QUANT_SCALE_INT8;
        } else {
            ReduceMaxTemplateGroupQuant(reduceResLocal, reduceWorkLocal, _inMMLocal[reduceSrcOffset], reduceTmpLocal,
                                        halfTokenLen);
            event_t evtVToS = static_cast<event_t>(pipe->FetchEventID(HardEvent::V_S));
            SetFlag<HardEvent::V_S>(evtVToS);
            WaitFlag<HardEvent::V_S>(evtVToS);
            quantScale = reduceResLocal.GetValue(0) / QUANT_SCALE_INT8;
        }
    } else {
        LocalTensor<float> workspaceLocal = reduceWorkspace.Get<float>();
        ReduceMaxTemplate(workspaceLocal, _inMMLocal, reduceSrcOffset, halfTokenLen);
        event_t evtVToS = static_cast<event_t>(pipe->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(evtVToS);
        WaitFlag<HardEvent::V_S>(evtVToS);
        quantScale = workspaceLocal.GetValue(0) / QUANT_SCALE_INT8;
    }
    if constexpr (GROUP_QUANT_DEBUG_STAGE == 56) {
        if (gmmBaseParams->M == 2 && halfTokenLen == VEC_LEN_ONCE_REPEAT_ELE * 2) {
            if (scaleGmRow == 0) {
                quantScale = 1165.37109375F;
            } else if (scaleGmRow == 1) {
                quantScale = 508.24493408F;
            }
        }
    }
    LocalTensor<float> quantScaleLocal = quantScaleOutQueue.DeQue<float>();
    quantScaleLocal.SetValue(loopIdx, quantScale);
    if constexpr (GROUP_QUANT_DEBUG_STAGE == 56) {
        if (scaleGmRow != 0xFFFFFFFFU) {
            quantScaleOutputGM.SetValue(scaleGmRow, quantScale);
        }
    }
    quantScale = 1 / quantScale;
    // S_V: scalar quantScale used as Muls vector-scalar param must be visible to vector
    event_t evtSToV = static_cast<event_t>(pipe->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(evtSToV);
    WaitFlag<HardEvent::S_V>(evtSToV);
    Muls(_inMMLocal[loopIdx * gmmSwigluTensorList->tokenLen], _inMMLocal[loopIdx * gmmSwigluTensorList->tokenLen],
         quantScale, gmmSwigluTensorList->tokenLen / BISECT);
    PipeBarrier<PIPE_V>();
    LocalTensor<int8_t> quantLocal = quantOutQueue.DeQue<int8_t>();
    int32_t dstTempOffset = static_cast<int32_t>(loopIdx * gmmSwigluTensorList->tokenLen / BISECT);
    int32_t srcTempOffset = static_cast<int32_t>(loopIdx * gmmSwigluTensorList->tokenLen);
    int32_t tempCount = static_cast<int32_t>(gmmSwigluTensorList->tokenLen / BISECT);
    if constexpr (GROUP_QUANT_DEBUG_STAGE == 8 || GROUP_QUANT_DEBUG_STAGE == 54) {
        if (gmmSwigluTensorList->enableGroupQuant) {
            for (int32_t i = 0; i < tempCount; ++i) {
                float value = _inMMLocal.GetValue(srcTempOffset + i);
                int32_t quantValue = static_cast<int32_t>(value >= 0.0f ? value + 0.5f : value - 0.5f);
                if (quantValue > 127) {
                    quantValue = 127;
                }
                if (quantValue < -128) {
                    quantValue = -128;
                }
                quantLocal.SetValue(dstTempOffset + i, static_cast<int8_t>(quantValue));
            }
            PipeBarrier<PIPE_ALL>();
            mmOutQueue.EnQue(_inMMLocal);
            quantOutQueue.EnQue(quantLocal);
            quantScaleOutQueue.EnQue(quantScaleLocal);
            return;
        }
    }
    LocalTensor<int8_t> castSpace = castWorkspace.Get<int8_t>();
    if constexpr (GROUP_QUANT_DEBUG_STAGE == 56) {
        if (halfTokenLen == VEC_LEN_ONCE_REPEAT_ELE * 2) {
            event_t evtVToS = static_cast<event_t>(pipe->FetchEventID(HardEvent::V_S));
            SetFlag<HardEvent::V_S>(evtVToS);
            WaitFlag<HardEvent::V_S>(evtVToS);
            for (int32_t i = 0; i < tempCount; ++i) {
                float value = _inMMLocal.GetValue(srcTempOffset + i);
                int32_t quantValue = static_cast<int32_t>(value >= 0.0F ? value + 0.5F : value - 0.5F);
                if (quantValue > 127) {
                    quantValue = 127;
                }
                if (quantValue < -128) {
                    quantValue = -128;
                }
                quantLocal.SetValue(dstTempOffset + i, static_cast<int8_t>(quantValue));
            }
        } else {
            CastFp32ToInt8Template(quantLocal, _inMMLocal, castSpace, dstTempOffset, srcTempOffset, tempCount);
        }
    } else {
        CastFp32ToInt8Template(quantLocal, _inMMLocal, castSpace, dstTempOffset, srcTempOffset, tempCount);
    }
    mmOutQueue.EnQue(_inMMLocal);
    quantOutQueue.EnQue(quantLocal);
    quantScaleOutQueue.EnQue(quantScaleLocal);
}

template <typename mmType, bool sync, typename CHANNELDTYPE>
__aicore__ inline void GMMSwigluCompute<mmType, sync, CHANNELDTYPE>::customDataCopyOut() {
    // perChanelScale * perTokenScale
    LocalTensor<float> quantScaleLocal = quantScaleOutQueue.DeQue<float>();
    DataCopyExtParams copyParams_0{1, static_cast<uint32_t>(vecConfig.innerLoopNum * sizeof(float)), 0, 0, 0};
    PipeBarrier<PIPE_ALL>();
    if constexpr (GROUP_QUANT_DEBUG_STAGE == 56) {
        event_t evtScaleSToMte3 = static_cast<event_t>(pipe->FetchEventID(HardEvent::S_MTE3));
        SetFlag<HardEvent::S_MTE3>(evtScaleSToMte3);
        WaitFlag<HardEvent::S_MTE3>(evtScaleSToMte3);
    }
    DataCopyPad(quantScaleOutputGM[vecConfig.startIdx], quantScaleLocal, copyParams_0);
    LocalTensor<int8_t> quantLocal = quantOutQueue.DeQue<int8_t>();
    DataCopyParams copyParams_1{
        1,
        static_cast<uint16_t>(vecConfig.innerLoopNum * gmmSwigluTensorList->tokenLen / 2 * sizeof(int8_t) /
                              UB_BLOCK_UNIT_SIZE),
        0,
        0};
    PipeBarrier<PIPE_ALL>();
    if constexpr (GROUP_QUANT_DEBUG_STAGE == 56) {
        event_t evtQuantSToMte3 = static_cast<event_t>(pipe->FetchEventID(HardEvent::S_MTE3));
        SetFlag<HardEvent::S_MTE3>(evtQuantSToMte3);
        WaitFlag<HardEvent::S_MTE3>(evtQuantSToMte3);
        event_t evtQuantVToMte3 = static_cast<event_t>(pipe->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(evtQuantVToMte3);
        WaitFlag<HardEvent::V_MTE3>(evtQuantVToMte3);
    }
    uint64_t quantOutputOffset = vecConfig.startIdx * gmmSwigluTensorList->tokenLen / BISECT;
    DataCopy(quantOutputGM[quantOutputOffset], quantLocal, copyParams_1);
    PipeBarrier<PIPE_ALL>();
    vecConfig.startIdx += vecConfig.innerLoopNum;
    vecConfig.startOffset = vecConfig.startIdx * gmmSwigluTensorList->tokenLen;
    quantOutQueue.EnQue(quantLocal);
    quantScaleOutQueue.EnQue(quantScaleLocal);
}

}  // namespace GROUPED_MATMUL
#endif  // ASCENDC_GROUPED_MATMUL_QUANT_MIXCORE_H
