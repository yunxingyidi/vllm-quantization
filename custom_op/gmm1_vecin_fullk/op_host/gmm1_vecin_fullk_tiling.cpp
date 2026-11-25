#include "register/op_impl_registry.h"
#include "tiling/tiling_base.h"
#include "tiling/platform/platform_ascendc.h"
#include "gmm1_vecin_fullk_tiling.h"

#include <algorithm>
#include <cstdlib>
#include <cstdio>

using namespace ge;
using namespace AscendC;
using namespace matmul_tiling;

namespace optiling {
namespace {
constexpr int64_t K = 2048;
constexpr int64_t MAX_FULL_N = 1536;
constexpr int64_t TILE_N = 256;
constexpr int64_t BASE_M = 16;
constexpr int64_t MIN_PARALLEL_M = 4;
constexpr int64_t FP32_PER_REPEAT = 64;
constexpr int64_t UB_BUDGET = 183 * 1024;
constexpr uint64_t SYS_WORKSPACE = 16UL * 1024 * 1024;
constexpr int64_t BY_PERM_UB_BYTES = K * (sizeof(uint16_t) + 2 * sizeof(float) + sizeof(int8_t));
struct CompileInfo { uint32_t aicNum = 0; };

uint32_t ReadEnvUint(const char *name, uint32_t fallback)
{
    const char *value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') return fallback;
    char *end = nullptr;
    const unsigned long parsed = std::strtoul(value, &end, 10);
    return end == value || *end != '\0' || parsed > UINT32_MAX
        ? fallback : static_cast<uint32_t>(parsed);
}

int64_t EstimateNzUbBytes(int64_t tileM, int64_t fullN, bool grouped = false)
{
    const int64_t paddedM = (tileM + 7) / 8 * 8;
    // FP16 NZ, FP32 NZ/ND working storage, INT8 ND, Mx16 broadcast,
    // token/output scale and two vector reduction results. InitBuffer aligns
    // every allocation to 32 bytes.
    const int64_t scratchElements = grouped ? std::min<int64_t>(4096, tileM * fullN)
        : tileM * 16;
    return tileM * fullN * 6 + tileM * (fullN / 2) +
        scratchElements * 4 + 4 * paddedM * 4;
}

int64_t EstimateUbBytes(int64_t tileM, int64_t tileN, int64_t halfN, bool byPerm)
{
    const int64_t alignedHalfN = (halfN + FP32_PER_REPEAT - 1) / FP32_PER_REPEAT * FP32_PER_REPEAT;
    const int64_t rowStride = 2 * alignedHalfN;
    return tileM * tileN * (sizeof(uint16_t) + sizeof(float)) +
           tileM * rowStride * sizeof(float) +
           tileM * alignedHalfN * sizeof(int8_t) +
           alignedHalfN * sizeof(uint16_t) +
           (8 + FP32_PER_REPEAT + tileM) * sizeof(float) +
           (byPerm ? BY_PERM_UB_BYTES : 0);
}

int64_t SelectMatmulN(int64_t fullN, int64_t halfN, int64_t tileM, bool byPerm)
{
    constexpr int64_t candidates[] = {1536, 768, 512, 384, 256, 192, 128, 96, 64};
    for (const int64_t candidate : candidates) {
        const int64_t tileN = std::min(fullN, candidate);
        if (tileN > 0 && tileN % 16 == 0 &&
            EstimateUbBytes(tileM, tileN, halfN, byPerm) <= UB_BUDGET) {
            return tileN;
        }
    }
    return std::min(TILE_N, halfN);
}

graphStatus DoTiling(gert::TilingContext *ctx)
{
    // Required inputs that follow a dynamic TensorList must be resolved by
    // IR index; GetInputShape() addresses the instantiated input array.
    const gert::StorageShape *xStorage = ctx->GetRequiredInputShape(0);
    const gert::StorageShape *groupsStorage = ctx->GetRequiredInputShape(4);
    if (xStorage == nullptr || groupsStorage == nullptr) return GRAPH_FAILED;
    const gert::Shape &x = xStorage->GetStorageShape();
    const gert::Shape &groups = groupsStorage->GetStorageShape();
    if (x.GetDimNum() != 2 || x.GetDim(0) <= 0 || x.GetDim(1) != K ||
        groups.GetDimNum() != 1 || groups.GetDim(0) <= 0) return GRAPH_FAILED;
    const ge::DataType xDtype = ctx->GetInputDesc(0)->GetDataType();
    // Dynamic format selection can expose the selected variant dtype only
    // after tiling.  x_scale presence is the invariant public discriminator:
    // legacy INT8 requires it, while BF16 by-perm forbids it.
    const bool byPerm = false;
    if ((byPerm && xDtype != ge::DT_BF16) || (!byPerm && xDtype != ge::DT_INT8)) {
        return GRAPH_FAILED;
    }
    const gert::StorageShape *weightStorage = ctx->GetDynamicInputShape(1, 0);
    if (weightStorage == nullptr) return GRAPH_FAILED;
    const gert::Shape &weight = weightStorage->GetStorageShape();
    if (weight.GetDimNum() != 4 || weight.GetDim(0) <= 0 ||
        weight.GetDim(1) != 128 || weight.GetDim(2) != 16 || weight.GetDim(3) != 32) return GRAPH_FAILED;
    const int64_t fullN = weight.GetDim(0) * 32;
    if (fullN > MAX_FULL_N || fullN % 64 != 0) return GRAPH_FAILED;
    const int64_t halfN = fullN / 2;
    const gert::StorageShape *scaleStorage = ctx->GetRequiredInputShape(2);
    if (scaleStorage == nullptr) return GRAPH_FAILED;
    const gert::Shape &scale = scaleStorage->GetStorageShape();
    int64_t scaleElements = 1;
    if (scale.GetDimNum() == 0) return GRAPH_FAILED;
    for (size_t dim = 0; dim < scale.GetDimNum(); ++dim) {
        if (scale.GetDim(dim) <= 0) return GRAPH_FAILED;
        scaleElements *= scale.GetDim(dim);
    }
    const int64_t expertNum = groups.GetDim(0);
    if (scaleElements % (expertNum * fullN) != 0) return GRAPH_FAILED;
    const int64_t kGroupNum = scaleElements / (expertNum * fullN);
    if (kGroupNum <= 0 || K % kGroupNum != 0) return GRAPH_FAILED;
    const int64_t kGroupSize = K / kGroupNum;
    if (kGroupSize % 16 != 0) return GRAPH_FAILED;
    const CompileInfo *compile = ctx->GetCompileInfo<CompileInfo>();
    if (compile == nullptr || compile->aicNum == 0) return GRAPH_FAILED;

    Gmm1VecInFullKTilingData data;
    const int64_t totalM = x.GetDim(0);
    const int64_t parallelBaseM = std::min<int64_t>(BASE_M, std::max<int64_t>(
        MIN_PARALLEL_M, (totalM + compile->aicNum - 1) / compile->aicNum));
    // Repeating a synchronous KFC sequence on several M shards serializes at
    // high K-group counts and duplicates weight traffic. Keep those tiles
    // aggregated; expose M parallelism for the per-channel/small-loop path.
    int64_t runtimeBaseM = std::min<int64_t>(
        totalM, kGroupNum > 1 ? BASE_M : parallelBaseM);
    platform_ascendc::PlatformAscendC platform(ctx->GetPlatformInfo());
    uint64_t physicalUb = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, physicalUb);
    const int64_t usableUb = std::min<int64_t>(UB_BUDGET,
        physicalUb > 9 * 1024 ? physicalUb - 9 * 1024 : 0);
    const bool nzEpilogue = ReadEnvUint("GROUP_QUANT_GMM1_NZ_EPILOGUE", 1) != 0 &&
        kGroupNum == 1 && halfN % FP32_PER_REPEAT == 0;
    const bool groupNzEpilogue = ReadEnvUint("GROUP_QUANT_GMM1_GROUP_NZ", 1) != 0 &&
        kGroupNum > 1 && fullN <= 768 && halfN % FP32_PER_REPEAT == 0;
    const bool useNzEpilogue = nzEpilogue || groupNzEpilogue;
    if (useNzEpilogue) {
        const int64_t limit = std::max<int64_t>(4, std::min<int64_t>(32,
            ReadEnvUint(groupNzEpilogue ? "GROUP_QUANT_GMM1_GROUP_BASE_M" :
                "GROUP_QUANT_GMM1_NZ_BASE_M", 32)));
        runtimeBaseM = groupNzEpilogue ? std::min<int64_t>(totalM, limit)
            : std::min<int64_t>(totalM, std::min<int64_t>(limit,
                std::max<int64_t>(MIN_PARALLEL_M,
                    (totalM + compile->aicNum - 1) / compile->aicNum)));
        while (runtimeBaseM > 1 && EstimateNzUbBytes(runtimeBaseM, fullN, groupNzEpilogue) > usableUb) {
            runtimeBaseM = std::max<int64_t>(1, runtimeBaseM / 2);
        }
        if (EstimateNzUbBytes(runtimeBaseM, fullN, groupNzEpilogue) > usableUb) return GRAPH_FAILED;
    }
    const int64_t matmulN = useNzEpilogue ? fullN :
        SelectMatmulN(fullN, halfN, runtimeBaseM, byPerm);

    data.set_totalM(static_cast<uint32_t>(totalM));
    data.set_expertNum(static_cast<uint32_t>(groups.GetDim(0)));
    data.set_coreNum(compile->aicNum);
    data.set_baseM(static_cast<uint32_t>(runtimeBaseM));
    data.set_baseN(static_cast<uint32_t>(matmulN));
    data.set_k(K);
    data.set_fullN(static_cast<uint32_t>(fullN));
    data.set_halfN(static_cast<uint32_t>(halfN));
    data.set_kGroupNum(static_cast<uint32_t>(kGroupNum));
    data.set_kGroupSize(static_cast<uint32_t>(kGroupSize));
    data.set_nzEpilogue(groupNzEpilogue ? 2 : (nzEpilogue ? 1 : 0));
    // Full-N grouped tiles can accumulate directly in NZ without enlarging
    // the existing accumulator or partial buffers. Keep legacy selectable.
    const bool nzAccum = !groupNzEpilogue && ReadEnvUint("GROUP_QUANT_GMM1_NZ_ACCUM", 1) != 0 &&
        kGroupNum > 1 && fullN <= 768 && matmulN == fullN &&
        halfN % FP32_PER_REPEAT == 0;
    const bool reuseOrgShape = groupNzEpilogue &&
        ReadEnvUint("GROUP_QUANT_GMM1_REUSE_ORG_SHAPE", 1) != 0;
    data.set_reserved((nzAccum ? 1U : 0U) | (reuseOrgShape ? 2U : 0U));

    MatmulApiTiling mm(platform);
    mm.SetAType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_INT8);
    mm.SetBType(TPosition::GM, CubeFormat::NZ, matmul_tiling::DataType::DT_INT8);
    mm.SetCType(TPosition::VECIN, CubeFormat::NZ, matmul_tiling::DataType::DT_FLOAT16);
    mm.SetBias(false);
    mm.SetShape(runtimeBaseM, matmulN, kGroupSize);
    mm.SetOrgShape(runtimeBaseM, matmulN, kGroupSize);
    mm.SetBufferSpace(-1, -1, -1);
    if (mm.GetTiling(data.gateTiling) == -1) return GRAPH_FAILED;
    data.upTiling = data.gateTiling;
    if (ReadEnvUint("GROUP_QUANT_GMM1_LOG_TILING", 0) != 0) {
        std::fprintf(stderr, "GMM1 Cube: base=%u/%u/%u dbL0C=%u reuseOrg=%u\n",
            data.gateTiling.get_baseM(), data.gateTiling.get_baseN(), data.gateTiling.get_baseK(),
            data.gateTiling.get_dbL0C(), reuseOrgShape ? 1U : 0U);
        std::fprintf(stderr, "GMM1 tiling: M=%lld N=%lld G=%lld tileM=%lld tileN=%lld cores=%u nz=%u nzAccum=%u UB=%lld\n",
            static_cast<long long>(totalM), static_cast<long long>(fullN),
            static_cast<long long>(kGroupNum), static_cast<long long>(runtimeBaseM),
            static_cast<long long>(matmulN), compile->aicNum, groupNzEpilogue ? 2U : (nzEpilogue ? 1U : 0U),
            nzAccum ? 1U : 0U,
            static_cast<long long>(useNzEpilogue ? EstimateNzUbBytes(runtimeBaseM, fullN, groupNzEpilogue)
                : EstimateUbBytes(runtimeBaseM, matmulN, halfN, byPerm)));
    }

    size_t *workspace = ctx->GetWorkspaceSizes(1);
    const uint64_t stagingBytes = byPerm
        ? static_cast<uint64_t>(compile->aicNum) * runtimeBaseM * K * sizeof(int8_t)
        : 0;
    workspace[0] = SYS_WORKSPACE + stagingBytes;
    ctx->SetBlockDim(compile->aicNum);
    ctx->SetTilingKey(0);
    data.SaveToBuffer(ctx->GetRawTilingData()->GetData(), ctx->GetRawTilingData()->GetCapacity());
    ctx->GetRawTilingData()->SetDataSize(data.GetDataSize());
    return GRAPH_SUCCESS;
}

graphStatus Parse(gert::TilingParseContext *ctx)
{
    CompileInfo *info = ctx->GetCompiledInfo<CompileInfo>();
    if (info == nullptr) return GRAPH_FAILED;
    platform_ascendc::PlatformAscendC platform(ctx->GetPlatformInfo());
    info->aicNum = platform.GetCoreNumAic();
    return info->aicNum == 0 ? GRAPH_FAILED : GRAPH_SUCCESS;
}

}  // namespace

IMPL_OP_OPTILING(Gmm1VecinFullk).Tiling(DoTiling).TilingParse<CompileInfo>(Parse);
}  // namespace optiling
