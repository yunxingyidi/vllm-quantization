#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"

using namespace AscendC;
using namespace matmul;

namespace {
constexpr uint32_t K = 2048;
constexpr uint32_t MAX_FULL_N = 1536;
constexpr uint32_t MAX_HALF_N = 768;
constexpr uint32_t FP32_PER_BLOCK = 8;
constexpr uint32_t N0 = 16;
constexpr uint32_t FP32_PER_REPEAT = 64;
constexpr uint32_t GROUP_PARTIAL_FLOATS = 4096;
// Eight floats are reserved for the scalar result and a full vector repeat is
// reserved for partial maxima.  The latter is zero-filled before the first
// reduction so the final reduction always consumes 64 initialized elements.
constexpr uint32_t REDUCE_BUFFER_FLOATS = FP32_PER_BLOCK + FP32_PER_REPEAT;
constexpr float QUANT_BOUND = 127.0F;

using AType = MatmulType<TPosition::GM, CubeFormat::ND, int8_t>;
using BType = MatmulType<TPosition::GM, CubeFormat::NZ, int8_t>;
// dav-2201 FixPipe only supports FP16 local Cube output. Apply the packed
// per-channel weight scale during L0C->VECIN so raw INT32 dots do not saturate
// before dequantization. Token scaling and the remaining epilogue stay FP32.
using CType = MatmulType<TPosition::VECIN, CubeFormat::NZ, half>;
using BiasType = MatmulType<TPosition::GM, CubeFormat::ND, int32_t>;
// `Matmul` selects MatmulServiceAux on AIC and MatmulClient on AIV for a
// registered MIX kernel.  Using MatmulImpl directly would bypass KFC and make
// the AIC dereference an AIV-local VECIN address.
using MM = Matmul<AType, BType, CType, BiasType, CFG_MDL>;

template <typename T>
__aicore__ inline __gm__ T *TensorListAddr(uint16_t index, GM_ADDR list)
{
    __gm__ uint64_t *descriptor = reinterpret_cast<__gm__ uint64_t *>(list);
    const uint64_t addressTableOffset = *descriptor;
    __gm__ uint64_t *addressTable = descriptor + (addressTableOffset >> 3);
    return reinterpret_cast<__gm__ T *>(*(addressTable + index));
}

__aicore__ inline uint32_t Min(uint32_t a, uint32_t b) { return a < b ? a : b; }

class KernelGmm1VecInFullK {
public:
    TPipe pipe_;
    MM mm_;

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR weightScale,
                                GM_ADDR xScale, GM_ADDR groupList, GM_ADDR y,
                                GM_ADDR yScale,
                                GM_ADDR userWorkspace,
                                const Gmm1VecInFullKTilingData &tiling)
    {
        tiling_ = tiling;
        x_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(x), static_cast<uint64_t>(tiling_.totalM) * K);
        xBf16_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(x),
                              static_cast<uint64_t>(tiling_.totalM) * K);
        xScale_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(xScale), tiling_.totalM);
        groups_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(groupList), tiling_.expertNum);
        y_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(y),
                           static_cast<uint64_t>(tiling_.totalM) * tiling_.halfN);
        yScale_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(yScale), tiling_.totalM);
        weightList_ = weight;
        weightScale_.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t *>(weightScale),
            static_cast<uint64_t>(tiling_.expertNum) * tiling_.kGroupNum * tiling_.fullN);

        const uint32_t matmulTileN = tiling_.baseN;
        const uint32_t cubeTileElements = tiling_.baseM * matmulTileN;
        const uint32_t alignedHalfN =
            (tiling_.halfN + FP32_PER_REPEAT - 1) / FP32_PER_REPEAT * FP32_PER_REPEAT;
        const uint32_t rowStride = 2 * alignedHalfN;
        const uint32_t outputTileElements = tiling_.baseM * rowStride;
        // DAV_2201 AIC has no UB allocation right.  These buffers back the
        // VECIN result and vector epilogue, so only the paired AIV allocates
        // them; MatmulImpl itself initializes its AIC-side Cube resources.
        if ASCEND_IS_AIV {
            pipe_.InitBuffer(cubeNzBuf_, cubeTileElements * sizeof(half));
            if (tiling_.nzEpilogue != 0) {
                const uint32_t paddedM = (tiling_.baseM + 7) / 8 * 8;
                pipe_.InitBuffer(postRowBuf_, cubeTileElements * sizeof(float));
                const uint32_t scratchElements = tiling_.nzEpilogue == 2
                    ? Min(GROUP_PARTIAL_FLOATS, cubeTileElements) : tiling_.baseM * N0;
                pipe_.InitBuffer(dequantBuf_, scratchElements * sizeof(float));
                pipe_.InitBuffer(quantBuf_, tiling_.baseM * tiling_.halfN);
                pipe_.InitBuffer(reduceBuf_, 2 * paddedM * sizeof(float));
                pipe_.InitBuffer(xScaleBuf_, paddedM * sizeof(float));
                pipe_.InitBuffer(scaleOutBuf_, paddedM * sizeof(float));
            } else {
            // Store up and gate in one row-major [M, fullN] buffer. For
            // 256 < fullN <= 768 the Matmul service traverses the complete
            // N plane in one request; wider and narrow-tail shapes retain
            // the compact base-tile path.
            pipe_.InitBuffer(dequantBuf_, outputTileElements * sizeof(float));
            // A single output row is sufficient: SwiGLU is evaluated and quantized
            // row by row after both input halves are ready.
            // Also serves as a full [BASE_M, TILE_N] FP32 staging tile while
            // accumulating K groups. Casting the complete FixPipe tile once
            // removes per-row Cast issue and barrier overhead.
            pipe_.InitBuffer(postRowBuf_, cubeTileElements * sizeof(float));
            pipe_.InitBuffer(quantHalfRowBuf_, alignedHalfN * sizeof(half));
            pipe_.InitBuffer(quantBuf_, tiling_.baseM * alignedHalfN * sizeof(int8_t));
            // Keep the scalar result disjoint from a full-repeat partial-max
            // workspace.  A full repeat avoids masked tail reads in reduction 2.
            pipe_.InitBuffer(reduceBuf_, REDUCE_BUFFER_FLOATS * sizeof(float));
            // Stage one M tile of activation scales in UB.  The epilogue
            // consumes every scale repeatedly across the full output row;
            // scalar GM reads in that loop serialize MTE2 and scalar issue.
            pipe_.InitBuffer(xScaleBuf_, tiling_.baseM * sizeof(float));
            pipe_.InitBuffer(scaleOutBuf_, tiling_.baseM * sizeof(float));
            if (false) {
                pipe_.InitBuffer(routeInputBuf_, K * sizeof(bfloat16_t));
                pipe_.InitBuffer(routeValueBuf_, K * sizeof(float));
                pipe_.InitBuffer(routeScratchBuf_, K * sizeof(float));
                pipe_.InitBuffer(routeQuantBuf_, K * sizeof(int8_t));
            }
            }
        }

        mm_.Init(&tiling_.gateTiling, &pipe_);
    }

    __aicore__ inline void Process()
    {
        const uint32_t core = GetBlockIdx();
        const uint32_t coreNum = tiling_.coreNum;
        uint32_t previousEnd = 0;
        uint32_t logicalTile = 0;
        for (uint32_t expert = 0; expert < tiling_.expertNum; ++expert) {
            const int64_t signedEnd = groups_.GetValue(expert);
            if (signedEnd < static_cast<int64_t>(previousEnd) ||
                signedEnd > static_cast<int64_t>(tiling_.totalM)) return;
            const uint32_t expertEnd = static_cast<uint32_t>(signedEnd);
            for (uint32_t row = previousEnd; row < expertEnd;
                 row += tiling_.baseM, ++logicalTile) {
                if ((logicalTile % coreNum) == core) {
                    ComputeTile(expert, row, Min(tiling_.baseM, expertEnd - row));
                }
            }
            previousEnd = expertEnd;
        }
        mm_.End();
    }

private:
    __aicore__ inline void ComputeNzTile(uint32_t expert, uint32_t rowStart, uint32_t tileM)
    {
        const uint32_t fullN = tiling_.fullN;
        const uint32_t halfN = tiling_.halfN;
        const uint32_t halfElems = tileM * halfN;
        const uint32_t paddedM = (tileM + 7) / 8 * 8;
        LocalTensor<half> cube = cubeNzBuf_.Get<half>();
        LocalTensor<float> values = postRowBuf_.Get<float>();
        LocalTensor<float> token = xScaleBuf_.Get<float>();
        LocalTensor<float> scales = scaleOutBuf_.Get<float>();
        LocalTensor<float> broadcast = dequantBuf_.Get<float>();
        LocalTensor<float> maxima = reduceBuf_.Get<float>();
        LocalTensor<float> blockMax = maxima[paddedM];
        LocalTensor<int8_t> quant = quantBuf_.Get<int8_t>();

        RunMatmul(expert, rowStart, tileM, fullN, 0, 0, cube);
        const uint32_t elements = tileM * fullN;
        Cast(values, cube, RoundMode::CAST_NONE, elements);
        PipeBarrier<PIPE_V>();
        if (tiling_.nzEpilogue == 2) {
            // Keep only one full FP32 accumulator. A bounded partial scratch
            // makes M32 fit UB without changing any group scale or add order.
            const uint32_t scratchElements = Min(GROUP_PARTIAL_FLOATS,
                tiling_.baseM * fullN);
            for (uint32_t group = 1; group < tiling_.kGroupNum; ++group) {
                RunMatmul(expert, rowStart, tileM, fullN, 0, group, cube);
                for (uint32_t offset = 0; offset < elements; offset += scratchElements) {
                    const uint32_t count = Min(scratchElements, elements - offset);
                    Cast(broadcast, cube[offset], RoundMode::CAST_NONE, count);
                    PipeBarrier<PIPE_V>();
                    Add(values[offset], values[offset], broadcast, count);
                    PipeBarrier<PIPE_V>();
                }
            }
        }
        DataCopyPad(token, xScale_[rowStart],
            DataCopyExtParams{1, static_cast<uint32_t>(tileM * sizeof(float)), 0, 0, 0},
            DataCopyPadExtParams<float>{true, 0, 0, 0.0F});
        const event_t load = static_cast<event_t>(pipe_.FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(load);
        WaitFlag<HardEvent::MTE2_V>(load);
        const uint32_t broadcastDst[2] = {tileM, N0};
        const uint32_t broadcastSrc[2] = {tileM, 1};
        // Cube source is dead after Cast. Reuse it as broadcast scratch, then
        // as a disjoint FP32 SwiGLU destination of exactly half the width.
        LocalTensor<uint8_t> tokenScratch = cube.ReinterpretCast<uint8_t>();
        BroadCast<float, 2, 1>(broadcast, token, broadcastDst, broadcastSrc, tokenScratch);
        PipeBarrier<PIPE_V>();
        const BinaryRepeatParams nzScale{1, 1, 1,
            static_cast<uint8_t>(2 * tileM), static_cast<uint8_t>(2 * tileM), 0};
        for (uint32_t rowPart = 0; rowPart < tileM * N0; rowPart += FP32_PER_REPEAT) {
            const uint64_t mask = Min(FP32_PER_REPEAT, tileM * N0 - rowPart);
            Mul(values[rowPart], values[rowPart], broadcast[rowPart], mask,
                static_cast<uint8_t>(fullN / N0), nzScale);
        }
        PipeBarrier<PIPE_V>();
        LocalTensor<float> post = cubeNzBuf_.Get<float>();
        LocalTensor<float> first = values;
        LocalTensor<float> second = values[halfElems];
        first.SetSize(halfElems);
        second.SetSize(halfElems);
        post.SetSize(halfElems);
        // Match the legacy operand order and keep all three tensors disjoint.
        SwiGLU<float, false>(post, second, first, 1.0F,
                            broadcast.ReinterpretCast<uint8_t>(), halfElems);
        PipeBarrier<PIPE_V>();

        // Both FP32 inputs are dead. Only the half-width result needs ND
        // storage for row reductions and the existing GMM2 activation ABI.
        const CopyRepeatParams ndCopy{1, 1,
            static_cast<uint16_t>(halfN / FP32_PER_BLOCK), 2};
        for (uint32_t block = 0; block < halfN / N0; ++block) {
            Copy(values[block * N0], post[block * tileM * N0], N0,
                 static_cast<uint8_t>(tileM), ndCopy);
        }
        PipeBarrier<PIPE_V>();
        Abs(post, values, halfElems);
        Duplicate(maxima, 0.0F, paddedM);
        Duplicate(blockMax, 0.0F, paddedM);
        PipeBarrier<PIPE_V>();
        for (uint32_t col = 0; col < halfN; col += FP32_PER_REPEAT) {
            WholeReduceMax(blockMax, post[col], FP32_PER_REPEAT,
                static_cast<uint8_t>(tileM), 1, 1,
                static_cast<uint8_t>(halfN / FP32_PER_BLOCK),
                ReduceOrder::ORDER_ONLY_VALUE);
            PipeBarrier<PIPE_V>();
            Max(maxima, maxima, blockMax, paddedM);
            PipeBarrier<PIPE_V>();
        }
        // Retain the legacy scalar division/rounding, but batch the V/S
        // hand-off once per tile rather than once per token and operation.
        const event_t toScalar = static_cast<event_t>(pipe_.FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(toScalar);
        WaitFlag<HardEvent::V_S>(toScalar);
        for (uint32_t row = 0; row < tileM; ++row) {
            const float absMax = maxima.GetValue(row);
            const float scale = (absMax > 1.0e-6F ? absMax : 1.0e-6F) / QUANT_BOUND;
            scales.SetValue(row, scale);
            token.SetValue(row, 1.0F / scale);
        }
        const event_t toVector = static_cast<event_t>(pipe_.FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(toVector);
        WaitFlag<HardEvent::S_V>(toVector);
        const uint32_t normDst[2] = {tileM, halfN};
        // Upper half of values and the old abs/SwiGLU result are now dead.
        LocalTensor<uint8_t> normScratch = values[halfElems].ReinterpretCast<uint8_t>();
        BroadCast<float, 2, 1>(post, token, normDst, broadcastSrc, normScratch);
        PipeBarrier<PIPE_V>();
        Mul(values, values, post, halfElems);
        PipeBarrier<PIPE_V>();
        // Reuse the old FP32 broadcast allocation as the FP16 quant stage.
        Cast(cube, values, RoundMode::CAST_RINT, halfElems);
        PipeBarrier<PIPE_V>();
        Cast(quant, cube, RoundMode::CAST_RINT, halfElems);
        PipeBarrier<PIPE_V>();
        const event_t store = static_cast<event_t>(pipe_.FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(store);
        WaitFlag<HardEvent::V_MTE3>(store);
        DataCopy(y_[static_cast<uint64_t>(rowStart) * halfN], quant, halfElems);
        DataCopyPad(yScale_[rowStart], scales,
            DataCopyExtParams{1, static_cast<uint32_t>(tileM * sizeof(float)), 0, 0, 0});
        const event_t stored = static_cast<event_t>(pipe_.FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(stored);
        WaitFlag<HardEvent::MTE3_V>(stored);
    }

    __aicore__ inline void RunMatmul(uint32_t expert, uint32_t rowStart,
                                     uint32_t tileM, uint32_t tileN,
                                     uint32_t nOffset, uint32_t kGroup,
                                     const LocalTensor<half> &dstNz)
    {
        const uint32_t kGroupSize = tiling_.kGroupSize;
        const uint64_t groupWeightOffset =
            static_cast<uint64_t>(kGroup) * (kGroupSize / 16) * 16 * 32;
        GlobalTensor<int8_t> weight;
        weight.SetGlobalBuffer(TensorListAddr<int8_t>(expert, weightList_) +
                               static_cast<uint64_t>(nOffset) * K + groupWeightOffset);
        // A and B remain in their original full-K storage.  The five-argument
        // OrgShape preserves both row strides while SingleShape limits the
        // current Cube reduction to one K-scale group.
        // OrgShape is invariant across K groups and full M tiles. The KFC
        // service retains it; send an update only when a tail changes M/N.
        const bool reuseOrgShape = (tiling_.reserved & 2U) != 0;
        if (!reuseOrgShape || cachedOrgM_ != tileM || cachedOrgN_ != tileN) {
            mm_.SetOrgShape(tileM, tileN, K, K, tileN);
            cachedOrgM_ = tileM;
            cachedOrgN_ = tileN;
        }
        mm_.SetSingleShape(tileM, tileN, kGroupSize);
        if (false) {
            mm_.SetTensorA(stagedX_[static_cast<uint64_t>(kGroup) * kGroupSize]);
        } else {
            mm_.SetTensorA(x_[static_cast<uint64_t>(rowStart) * K +
                              static_cast<uint64_t>(kGroup) * kGroupSize]);
        }
        mm_.SetTensorB(weight);
        const uint64_t scaleOffset =
            (static_cast<uint64_t>(expert) * tiling_.kGroupNum + kGroup) *
                tiling_.fullN + nOffset;
        // npu_trans_quant_param packs each FP32 channel scale into the UINT64
        // format consumed by FixPipe VDEQF16. This both avoids raw-dot FP16
        // saturation and removes the AIV-side channel-scale multiply.
        mm_.SetQuantVector(weightScale_[scaleOffset]);
        mm_.IterateAll(dstNz);
    }

    __aicore__ inline void AccumulateFixpipe(const LocalTensor<float> &dst,
                                             const LocalTensor<half> &src,
                                             const LocalTensor<float> &partial,
                                             uint32_t tileM,
                                             uint32_t tileN, uint32_t nOffset,
                                             bool firstGroup)
    {
        const uint32_t alignedHalfN =
            (tiling_.halfN + FP32_PER_REPEAT - 1) / FP32_PER_REPEAT * FP32_PER_REPEAT;
        const uint32_t rowStride = 2 * alignedHalfN;
        Cast(partial, src, RoundMode::CAST_NONE, tileM * tileN);
        PipeBarrier<PIPE_V>();
        const CopyRepeatParams copyParams{
            1, 1, static_cast<uint16_t>(rowStride / FP32_PER_BLOCK), 2};
        const BinaryRepeatParams addParams{
            1, 1, 1, static_cast<uint8_t>(rowStride / FP32_PER_BLOCK),
            static_cast<uint8_t>(rowStride / FP32_PER_BLOCK), 2};
        for (uint32_t nBlock = 0; nBlock < tileN / N0; ++nBlock) {
            const uint32_t globalColumn = nOffset + nBlock * N0;
            const uint32_t dstColumn = globalColumn < tiling_.halfN
                ? globalColumn
                : alignedHalfN + globalColumn - tiling_.halfN;
            const uint32_t srcOffset = nBlock * tileM * N0;
            if (firstGroup) {
                Copy(dst[dstColumn], partial[srcOffset], N0,
                     static_cast<uint8_t>(tileM), copyParams);
            } else {
                Add(dst[dstColumn], dst[dstColumn], partial[srcOffset], N0,
                    static_cast<uint8_t>(tileM), addParams);
            }
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void AccumulateNzGroups(
        uint32_t expert, uint32_t rowStart, uint32_t tileM,
        const LocalTensor<half> &cube, const LocalTensor<float> &accumulator,
        const LocalTensor<float> &partial)
    {
        const uint32_t fullN = tiling_.fullN;
        const uint32_t elements = tileM * fullN;
        for (uint32_t group = 0; group < tiling_.kGroupNum; ++group) {
            RunMatmul(expert, rowStart, tileM, fullN, 0, group, cube);
            Cast(partial, cube, RoundMode::CAST_NONE, elements);
            PipeBarrier<PIPE_V>();
            if (group == 0) {
                // Copy preserves first-group bits, including signed zero.
                // Full-N eligibility guarantees a whole number of repeats.
                for (uint32_t offset = 0; offset < elements;) {
                    const uint32_t repeats = Min(255, (elements - offset) / FP32_PER_REPEAT);
                    Copy(accumulator[offset], partial[offset], FP32_PER_REPEAT,
                         static_cast<uint8_t>(repeats), CopyRepeatParams{1, 1, 8, 8});
                    offset += repeats * FP32_PER_REPEAT;
                }
            } else {
                Add(accumulator, accumulator, partial, elements);
            }
            PipeBarrier<PIPE_V>();
        }
        // Only one NZ->ND conversion after all groups. The old partial buffer
        // becomes the ND accumulator; the old NZ accumulator is now dead and
        // can serve as the disjoint legacy SwiGLU destination.
        const CopyRepeatParams ndCopy{1, 1,
            static_cast<uint16_t>(fullN / FP32_PER_BLOCK), 2};
        for (uint32_t block = 0; block < fullN / N0; ++block) {
            Copy(partial[block * N0], accumulator[block * tileM * N0], N0,
                 static_cast<uint8_t>(tileM), ndCopy);
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ApplyTokenScale(const LocalTensor<float> &dst,
                                            const LocalTensor<float> &tokenScales,
                                            uint32_t tileM, uint32_t tileN,
                                            uint32_t dstColumn)
    {
        const uint32_t alignedHalfN =
            (tiling_.halfN + FP32_PER_REPEAT - 1) / FP32_PER_REPEAT * FP32_PER_REPEAT;
        const uint32_t rowStride = 2 * alignedHalfN;
        for (uint32_t row = 0; row < tileM; ++row) {
            const event_t vToS = static_cast<event_t>(pipe_.FetchEventID(HardEvent::V_S));
            SetFlag<HardEvent::V_S>(vToS);
            WaitFlag<HardEvent::V_S>(vToS);
            const float tokenScale = tokenScales.GetValue(row);
            const event_t sToV = static_cast<event_t>(pipe_.FetchEventID(HardEvent::S_V));
            SetFlag<HardEvent::S_V>(sToV);
            WaitFlag<HardEvent::S_V>(sToV);
            const uint32_t dstOffset = row * rowStride + dstColumn;
            Muls(dst[dstOffset], dst[dstOffset], tokenScale, tileN);
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void QuantizeRow(const LocalTensor<float> &post,
                                       const LocalTensor<float> &scratch,
                                       const LocalTensor<int8_t> &quant,
                                       const LocalTensor<float> &scales,
                                       uint32_t row)
    {
        LocalTensor<float> reduceResult = reduceBuf_.Get<float>();
        LocalTensor<float> reduceWork = reduceResult[FP32_PER_BLOCK];
        LocalTensor<half> quantHalf = quantHalfRowBuf_.Get<half>();
        const uint32_t halfN = tiling_.halfN;
        const uint32_t quantRowStride =
            (halfN + FP32_PER_REPEAT - 1) / FP32_PER_REPEAT * FP32_PER_REPEAT;
        const uint32_t reduceCount = (halfN + FP32_PER_REPEAT - 1) / FP32_PER_REPEAT;
        const uint32_t alignedHalfN = reduceCount * FP32_PER_REPEAT;
        Abs(scratch, post, halfN);
        if (alignedHalfN != halfN) {
            Duplicate(scratch[halfN], 0.0F, alignedHalfN - halfN);
        }
        Duplicate(reduceWork, 0.0F, FP32_PER_REPEAT);
        PipeBarrier<PIPE_V>();
        WholeReduceMax(reduceWork, scratch, FP32_PER_REPEAT,
                       static_cast<uint8_t>(reduceCount), 1, 1, 8,
                       ReduceOrder::ORDER_ONLY_VALUE);
        PipeBarrier<PIPE_V>();
        WholeReduceMax(reduceResult, reduceWork, FP32_PER_REPEAT,
                       1, 1, 1, 8, ReduceOrder::ORDER_ONLY_VALUE);
        const event_t vToS = static_cast<event_t>(pipe_.FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(vToS);
        WaitFlag<HardEvent::V_S>(vToS);
        const float absMax = reduceResult.GetValue(0);
        const float scale = (absMax > 1.0e-6F ? absMax : 1.0e-6F) / QUANT_BOUND;
        scales.SetValue(row, scale);
        const float reciprocal = 1.0F / scale;
        const event_t sToV = static_cast<event_t>(pipe_.FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(sToV);
        WaitFlag<HardEvent::S_V>(sToV);
        Muls(post, post, reciprocal, halfN);
        PipeBarrier<PIPE_V>();
        // reciprocal is derived from this row's absmax, so the scaled values
        // are already in [-127, 127].  Avoid a redundant Maxs/Mins pass while
        // retaining the original FP32 scaling and rounding order.
        Cast(quantHalf, post, RoundMode::CAST_RINT, halfN);
        PipeBarrier<PIPE_V>();
        if (quantRowStride != halfN) {
            Duplicate(quantHalf[halfN], static_cast<half>(0.0), quantRowStride - halfN);
            PipeBarrier<PIPE_V>();
        }
        // FP16 -> INT8 Cast repeats require a 64-element-aligned destination
        // row start.  In particular TP8 has halfN=96, so its compact row
        // stride cannot be used inside VECOUT.
        Cast(quant[row * quantRowStride], quantHalf, RoundMode::CAST_RINT, quantRowStride);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline bool PrepareByPermInput(
        uint32_t rowStart, uint32_t tileM,
        const LocalTensor<float> &tokenScales)
    {
        LocalTensor<bfloat16_t> input = routeInputBuf_.Get<bfloat16_t>();
        LocalTensor<half> quantHalf = routeInputBuf_.Get<half>();
        LocalTensor<float> values = routeValueBuf_.Get<float>();
        LocalTensor<float> scratch = routeScratchBuf_.Get<float>();
        LocalTensor<int8_t> quant = routeQuantBuf_.Get<int8_t>();
        LocalTensor<float> reduceResult = reduceBuf_.Get<float>();
        const DataCopyExtParams inputCopy{
            1, static_cast<uint32_t>(K * sizeof(bfloat16_t)), 0, 0, 0};

        for (uint32_t row = 0; row < tileM; ++row) {
            const int64_t tokenId = sortedTokenIds_.GetValue(rowStart + row);
            if (tokenId < 0 || tokenId >= static_cast<int64_t>(tiling_.totalM)) {
                return false;
            }
            DataCopyPad(input, xBf16_[static_cast<uint64_t>(tokenId) * K],
                        inputCopy,
                        DataCopyPadExtParams<bfloat16_t>{false, 0, 0, 0.0F});
            const event_t mte2ToV =
                static_cast<event_t>(pipe_.FetchEventID(HardEvent::MTE2_V));
            SetFlag<HardEvent::MTE2_V>(mte2ToV);
            WaitFlag<HardEvent::MTE2_V>(mte2ToV);
            Cast(values, input, RoundMode::CAST_NONE, K);
            PipeBarrier<PIPE_V>();
            Abs(scratch, values, K);
            PipeBarrier<PIPE_V>();
            ReduceMax(reduceResult, scratch, scratch, K);
            const event_t vToS =
                static_cast<event_t>(pipe_.FetchEventID(HardEvent::V_S));
            SetFlag<HardEvent::V_S>(vToS);
            WaitFlag<HardEvent::V_S>(vToS);
            const float absMax = reduceResult.GetValue(0);
            const float scale =
                (absMax > 1.0e-6F ? absMax : 1.0e-6F) / QUANT_BOUND;
            tokenScales.SetValue(row, scale);
            const event_t sToV =
                static_cast<event_t>(pipe_.FetchEventID(HardEvent::S_V));
            SetFlag<HardEvent::S_V>(sToV);
            WaitFlag<HardEvent::S_V>(sToV);
            Duplicate(scratch, scale, K);
            PipeBarrier<PIPE_V>();
            Div(scratch, values, scratch, K);
            PipeBarrier<PIPE_V>();
            LocalTensor<int32_t> quantInt = scratch.ReinterpretCast<int32_t>();
            Cast(quantInt, scratch, RoundMode::CAST_RINT, K);
            PipeBarrier<PIPE_V>();
            SetDeqScale(static_cast<half>(1.0F));
            Cast(quantHalf, quantInt, RoundMode::CAST_ROUND, K);
            PipeBarrier<PIPE_V>();
            Cast(quant, quantHalf, RoundMode::CAST_TRUNC, K);
            const event_t vToMte3 =
                static_cast<event_t>(pipe_.FetchEventID(HardEvent::V_MTE3));
            SetFlag<HardEvent::V_MTE3>(vToMte3);
            WaitFlag<HardEvent::V_MTE3>(vToMte3);
            DataCopy(stagedX_[static_cast<uint64_t>(row) * K], quant,
                     DataCopyParams{1, static_cast<uint16_t>(K / 32), 0, 0});
            // The paired AIC receives the staging address only after this
            // scalar-side wait, so its MTE2 cannot observe a partial row.
            const event_t mte3ToS =
                static_cast<event_t>(pipe_.FetchEventID(HardEvent::MTE3_S));
            SetFlag<HardEvent::MTE3_S>(mte3ToS);
            WaitFlag<HardEvent::MTE3_S>(mte3ToS);
        }
        return true;
    }

    __aicore__ inline void ComputeTile(uint32_t expert, uint32_t rowStart, uint32_t tileM)
    {
        if (tiling_.nzEpilogue != 0) {
            ComputeNzTile(expert, rowStart, tileM);
            return;
        }
        LocalTensor<half> cubeNz = cubeNzBuf_.Get<half>();
        LocalTensor<float> dequant = dequantBuf_.Get<float>();
        LocalTensor<int8_t> quant = quantBuf_.Get<int8_t>();
        LocalTensor<float> scales = scaleOutBuf_.Get<float>();
        LocalTensor<float> tokenScales = xScaleBuf_.Get<float>();
        LocalTensor<float> partial = postRowBuf_.Get<float>();
        if (false &&
            !PrepareByPermInput(rowStart, tileM, tokenScales)) {
            return;
        }
        const uint32_t alignedHalfN =
            (tiling_.halfN + FP32_PER_REPEAT - 1) / FP32_PER_REPEAT * FP32_PER_REPEAT;
        const uint32_t rowStride = 2 * alignedHalfN;
        // Host tiling selects the widest ND local-output tile that fits the
        // AIV UB budget for the adaptive M tile.
        const uint32_t matmulTileN = tiling_.baseN;
        const uint32_t nTiles = (tiling_.fullN + matmulTileN - 1) / matmulTileN;
        const bool nzAccum = (tiling_.reserved & 1U) != 0;
        if (nzAccum) {
            AccumulateNzGroups(expert, rowStart, tileM, cubeNz, dequant, partial);
            dequant = postRowBuf_.Get<float>();
        } else {
            for (uint32_t kGroup = 0; kGroup < tiling_.kGroupNum; ++kGroup) {
                for (uint32_t nTile = 0; nTile < nTiles; ++nTile) {
                    const uint32_t column = nTile * matmulTileN;
                    const uint32_t tileN = Min(matmulTileN, tiling_.fullN - column);
                    RunMatmul(expert, rowStart, tileM, tileN, column, kGroup, cubeNz);
                    AccumulateFixpipe(dequant, cubeNz, partial, tileM, tileN,
                                      column, kGroup == 0);
                }
            }
        }
        if (true) {
            DataCopyPad(tokenScales, xScale_[rowStart],
                        DataCopyExtParams{1, static_cast<uint32_t>(tileM * sizeof(float)), 0, 0, 0},
                        DataCopyPadExtParams<float>{false, 0, 0, 0.0F});
            const event_t mte2ToS = static_cast<event_t>(pipe_.FetchEventID(HardEvent::MTE2_S));
            SetFlag<HardEvent::MTE2_S>(mte2ToS);
            WaitFlag<HardEvent::MTE2_S>(mte2ToS);
        }
        if (tiling_.halfN == alignedHalfN) {
            ApplyTokenScale(dequant, tokenScales, tileM, tiling_.fullN, 0);
        } else {
            for (uint32_t row = 0; row < tileM; ++row) {
                Duplicate(dequant[row * rowStride + tiling_.halfN], 0.0F,
                          alignedHalfN - tiling_.halfN);
                Duplicate(dequant[row * rowStride + alignedHalfN + tiling_.halfN],
                          0.0F, alignedHalfN - tiling_.halfN);
            }
            PipeBarrier<PIPE_V>();
            ApplyTokenScale(dequant, tokenScales, tileM, alignedHalfN, 0);
            ApplyTokenScale(dequant, tokenScales, tileM, alignedHalfN, alignedHalfN);
        }

        LocalTensor<float> postRow = nzAccum
            ? dequantBuf_.Get<float>() : postRowBuf_.Get<float>();
        for (uint32_t row = 0; row < tileM; ++row) {
            // Keep SwiGLU inputs and output disjoint.  `up` becomes scratch
            // only after its row was consumed, so all intermediates stay UB.
            LocalTensor<float> up = dequant[row * rowStride];
            LocalTensor<float> gate = dequant[row * rowStride + alignedHalfN];
            SwiGLU<float, false>(postRow, gate, up, 1.0F, alignedHalfN);
            PipeBarrier<PIPE_V>();
            QuantizeRow(postRow, up, quant, scales, row);
        }

        const event_t event = static_cast<event_t>(pipe_.FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(event);
        WaitFlag<HardEvent::V_MTE3>(event);
        const uint32_t quantRowStride =
            (tiling_.halfN + FP32_PER_REPEAT - 1) / FP32_PER_REPEAT * FP32_PER_REPEAT;
        for (uint32_t row = 0; row < tileM; ++row) {
            DataCopy(y_[static_cast<uint64_t>(rowStart + row) * tiling_.halfN],
                     quant[row * quantRowStride],
                     DataCopyParams{1, static_cast<uint16_t>(tiling_.halfN / 32), 0, 0});
        }
        DataCopyPad(yScale_[rowStart], scales,
                    DataCopyExtParams{1, static_cast<uint32_t>(tileM * sizeof(float)), 0, 0, 0});
    }

    Gmm1VecInFullKTilingData tiling_;
    uint32_t cachedOrgM_ = 0;
    uint32_t cachedOrgN_ = 0;
    GM_ADDR weightList_ = nullptr;
    // Keep the FixPipe destination in a generic UB allocation.  This matches
    // the CANN VECIN MatMul reference kernels; TPosition::VECIN describes the
    // MatMul endpoint, while the backing LocalTensor still comes from UB.
    TBuf<> cubeNzBuf_;
    TBuf<TPosition::VECCALC> dequantBuf_, postRowBuf_, quantHalfRowBuf_, reduceBuf_, xScaleBuf_;
    TBuf<TPosition::VECCALC> routeInputBuf_, routeValueBuf_, routeScratchBuf_;
    TBuf<TPosition::VECOUT> quantBuf_, scaleOutBuf_, routeQuantBuf_;
    GlobalTensor<int8_t> x_, stagedX_, y_;
    GlobalTensor<bfloat16_t> xBf16_;
    GlobalTensor<uint64_t> weightScale_;
    GlobalTensor<float> xScale_, yScale_;
    GlobalTensor<int64_t> groups_;
    GlobalTensor<int64_t> sortedTokenIds_;
};
}  // namespace

extern "C" __global__ __aicore__ void gmm1_vecin_fullk(
    GM_ADDR x, GM_ADDR weight, GM_ADDR weightScale, GM_ADDR xScale,
    GM_ADDR groupList, GM_ADDR y, GM_ADDR yScale,
    GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
    GET_TILING_DATA(data, tiling);
    KernelGmm1VecInFullK op;
    // On AIC this starts the Matmul KFC service loop and returns from the
    // kernel when the paired AIV client calls End().  On AIV it registers the
    // client, after which the VECIN destination address is sent to FixPipe.
    REGIST_MATMUL_OBJ(&op.pipe_, GetSysWorkSpacePtr(), op.mm_, &data.gateTiling);
    op.Init(x, weight, weightScale, xScale, groupList, y,
            yScale, GetUserWorkspace(workspace), data);
    op.Process();
}
