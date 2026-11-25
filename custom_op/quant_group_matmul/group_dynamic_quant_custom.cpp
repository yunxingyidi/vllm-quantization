#include "kernel_operator.h"

using AscendC::GlobalTensor;
using AscendC::LocalTensor;
using AscendC::TPosition;

__aicore__ inline uint64_t AlignUp64(uint64_t value, uint64_t align)
{
    return (value + align - 1) / align * align;
}

template <typename T>
__aicore__ inline void DataCopyPad2D(
    const GlobalTensor<T> dst,
    const LocalTensor<T> src,
    uint32_t dim1,
    uint32_t dim0,
    uint32_t srcDim0,
    uint32_t dstDim0)
{
    AscendC::DataCopyExtParams params;
    params.blockCount = dim1;
    params.blockLen = dim0 * sizeof(T);
    params.srcStride = (srcDim0 - dim0) * sizeof(T) / 32;
    params.dstStride = (dstDim0 - dim0) * sizeof(T);
    DataCopyPad(dst, src, params);
}

struct GroupDynamicQuantTilingData {
    uint32_t m;
    uint32_t k;
    uint32_t groupSize;
    uint32_t groupNum;
    uint32_t coreNum;
};

__aicore__ inline void CopyGroupDynamicQuantTiling(
    GroupDynamicQuantTilingData *tiling,
    GM_ADDR tilingGM)
{
    uint32_t *ptr = reinterpret_cast<uint32_t *>(tiling);
    auto tiling32 = reinterpret_cast<__gm__ uint32_t *>(tilingGM);
    for (uint32_t i = 0; i < sizeof(GroupDynamicQuantTilingData) / sizeof(uint32_t); ++i, ++ptr) {
        *ptr = *(tiling32 + i);
    }
}

template <typename T>
class GroupDynamicQuant {
public:
    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR y,
        GM_ADDR scale,
        GroupDynamicQuantTilingData *tilingData,
        AscendC::TPipe *tPipe)
    {
        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(x));
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(y));
        scaleGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(scale));
        tiling = tilingData;
        pipe = tPipe;
        coreIdx = AscendC::GetBlockIdx();
        if ASCEND_IS_AIV {
            coreIdx /= AscendC::GetTaskRation();
        }
        coreNum = tiling->coreNum;

        alignedGroupSize = AlignUp64(tiling->groupSize, 8);
        constexpr uint32_t SCALE_BATCH = 32;
        const uint32_t qRowBytes = AlignUp64(tiling->groupSize, 32);
        const uint32_t floatBytes = AlignUp64(alignedGroupSize * sizeof(float), 32);
        pipe->InitBuffer(qRowBuffer, qRowBytes);
        pipe->InitBuffer(scaleBuffer, SCALE_BATCH * sizeof(float));
        pipe->InitBuffer(bf16StageBuffer, floatBytes);
        pipe->InitBuffer(floatWorkBuffer, floatBytes);
        pipe->InitBuffer(absWorkBuffer, floatBytes);
        qRowLocal = qRowBuffer.Get<int8_t>();
        scaleLocal = scaleBuffer.Get<float>();
        bf16StageLocal = bf16StageBuffer.Get<T>();
        floatWorkLocal = floatWorkBuffer.Get<float>();
        absWorkLocal = absWorkBuffer.Get<float>();
    }

    __aicore__ inline void Process()
    {
        const uint32_t totalRows = tiling->m * tiling->groupNum;
        const uint32_t rowsPerCore = AscendC::Ceil(totalRows, coreNum);
        const uint32_t rowStart = coreIdx * rowsPerCore;
        const uint32_t rowEnd =
            rowStart + rowsPerCore < totalRows ? rowStart + rowsPerCore : totalRows;
        constexpr uint32_t SCALE_BATCH = 32;
        uint32_t scaleBatchStart = rowStart;
        uint32_t scaleBatchCount = 0;

        for (uint32_t packedRow = rowStart; packedRow < rowEnd; ++packedRow) {
            ProcessRow(packedRow);
            if (scaleBatchCount == 0) {
                scaleBatchStart = packedRow;
            }
            scaleLocal.SetValue(scaleBatchCount, currentScale);
            ++scaleBatchCount;
            if (scaleBatchCount == SCALE_BATCH) {
                AscendC::PipeBarrier<PIPE_V>();
                DataCopy(scaleGm[scaleBatchStart], scaleLocal, SCALE_BATCH);
                scaleBatchCount = 0;
            }
        }
        if (scaleBatchCount > 0) {
            AscendC::PipeBarrier<PIPE_V>();
            DataCopyPad2D(scaleGm[scaleBatchStart], scaleLocal, 1, scaleBatchCount,
                          scaleBatchCount, scaleBatchCount);
        }
    }

private:
    __aicore__ inline void ProcessRow(uint32_t packedRow)
    {
        const uint32_t groupIdx = packedRow / tiling->m;
        const uint32_t tokenIdx = packedRow - groupIdx * tiling->m;
        const uint64_t inputOffset =
            static_cast<uint64_t>(tokenIdx) * tiling->k +
            static_cast<uint64_t>(groupIdx) * tiling->groupSize;
        const uint64_t outputOffset =
            static_cast<uint64_t>(packedRow) * tiling->groupSize;

        AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(tiling->groupSize * sizeof(T)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<T> padParams{true, 0, 0, 0};

        DataCopyPad(bf16StageLocal, xGm[inputOffset], copyParams, padParams);
        AscendC::PipeBarrier<PIPE_MTE2>();
        Cast(floatWorkLocal, bf16StageLocal, AscendC::RoundMode::CAST_NONE, alignedGroupSize);
        AscendC::PipeBarrier<PIPE_V>();

        // Abs + ReduceMax on a separate buffer to preserve floatWorkLocal
        Abs(absWorkLocal, floatWorkLocal, alignedGroupSize);
        AscendC::PipeBarrier<PIPE_V>();
        LocalTensor<float> reduceTmp = bf16StageLocal.template ReinterpretCast<float>();
        ReduceMax(absWorkLocal, absWorkLocal, reduceTmp, static_cast<int32_t>(alignedGroupSize), false);
        AscendC::PipeBarrier<PIPE_V>();
        float maxAbs = absWorkLocal.GetValue(0);

        currentScale = maxAbs / 127.0f;

        // Quantize directly from preserved floatWorkLocal - no second GM load
        if (currentScale > 0.0f) {
            Muls(floatWorkLocal, floatWorkLocal, 1.0f / currentScale, alignedGroupSize);
            AscendC::PipeBarrier<PIPE_V>();
        }
        Maxs(floatWorkLocal, floatWorkLocal, -127.0f, alignedGroupSize);
        Mins(floatWorkLocal, floatWorkLocal, 127.0f, alignedGroupSize);
        AscendC::PipeBarrier<PIPE_V>();
        LocalTensor<int16_t> i16Tmp = floatWorkLocal.ReinterpretCast<int16_t>();
        Cast(i16Tmp, floatWorkLocal, AscendC::RoundMode::CAST_RINT, alignedGroupSize);
        AscendC::PipeBarrier<PIPE_V>();
        Cast(qRowLocal, i16Tmp, AscendC::RoundMode::CAST_RINT, alignedGroupSize);
        AscendC::PipeBarrier<PIPE_V>();

        DataCopy(yGm[outputOffset], qRowLocal, tiling->groupSize);
        AscendC::PipeBarrier<PIPE_MTE3>();
    }

private:
    GlobalTensor<T> xGm;
    GlobalTensor<int8_t> yGm;
    GlobalTensor<float> scaleGm;
    AscendC::TPipe *pipe;
    AscendC::TBuf<> qRowBuffer;
    AscendC::TBuf<TPosition::VECCALC> scaleBuffer;
    AscendC::TBuf<TPosition::VECCALC> bf16StageBuffer;
    AscendC::TBuf<TPosition::VECCALC> floatWorkBuffer;
    AscendC::TBuf<TPosition::VECCALC> absWorkBuffer;
    LocalTensor<int8_t> qRowLocal;
    LocalTensor<float> scaleLocal;
    LocalTensor<T> bf16StageLocal;
    LocalTensor<float> floatWorkLocal;
    LocalTensor<float> absWorkLocal;
    GroupDynamicQuantTilingData *tiling;
    uint32_t coreIdx;
    uint32_t coreNum;
    uint32_t alignedGroupSize;
    float currentScale;
};

extern "C" __global__ __aicore__ void group_dynamic_quant_custom(
    GM_ADDR x,
    GM_ADDR y,
    GM_ADDR scale,
    GM_ADDR workspace,
    GM_ADDR tilingGm)
{
    if ASCEND_IS_AIC {
        return;
    }

    GroupDynamicQuantTilingData tilingData;
    CopyGroupDynamicQuantTiling(&tilingData, tilingGm);
    AscendC::TPipe pipe;
    GroupDynamicQuant<bfloat16_t> op;
    op.Init(x, y, scale, &tilingData, &pipe);
    op.Process();
}
