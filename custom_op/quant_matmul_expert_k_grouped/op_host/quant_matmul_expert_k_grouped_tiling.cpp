#include <algorithm>
#include <climits>
#include <cstdlib>
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"
#include "quant_matmul_expert_k_grouped_tiling.h"

namespace optiling {
namespace {
constexpr size_t X_INDEX = 0;
constexpr size_t WEIGHT_INDEX = 1;
constexpr size_t GROUP_LIST_INDEX = 2;
constexpr size_t WEIGHT_SCALE_INDEX = 3;
constexpr size_t X_SCALE_INDEX = 4;
constexpr uint32_t DEFAULT_CORE_NUM = 20;
constexpr uint32_t DEFAULT_PARALL_NUM = 4;
constexpr uint32_t BASE_M = 64;
constexpr uint32_t BASE_N = 512;
constexpr uint32_t BASE_K = 128;
constexpr uint32_t UB_CAL_SIZE = 16 * 256;
constexpr uint32_t UB_REST_BYTES = 118784;

struct CompileInfo {
    uint32_t aicNum = 0;
    uint64_t sysWorkspaceSize = 0;
};

uint32_t ReadEnvUint(const char *name, uint32_t defaultValue, bool allowZero = false)
{
    const char *value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return defaultValue;
    }
    char *end = nullptr;
    const unsigned long parsed = std::strtoul(value, &end, 10);
    if (end == value || *end != '\0' || (!allowZero && parsed == 0) || parsed > UINT32_MAX) {
        return defaultValue;
    }
    return static_cast<uint32_t>(parsed);
}

ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    const auto *compileInfo = context->GetCompileInfo<CompileInfo>();
    const auto *x = context->GetInputTensor(X_INDEX);
    const auto *weight = context->GetInputTensor(WEIGHT_INDEX);
    const auto *groupList = context->GetInputTensor(GROUP_LIST_INDEX);
    const auto *weightScale = context->GetInputTensor(WEIGHT_SCALE_INDEX);
    const auto *xScale = context->GetInputTensor(X_SCALE_INDEX);
    if (compileInfo == nullptr || x == nullptr || weight == nullptr || groupList == nullptr ||
        weightScale == nullptr || xScale == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const auto &xShape = x->GetStorageShape();
    const auto &weightShape = weight->GetStorageShape();
    const auto &groupListShape = groupList->GetStorageShape();
    const auto &weightScaleShape = weightScale->GetStorageShape();
    const auto &xScaleShape = xScale->GetStorageShape();
    const bool cumulativeGroupList = groupListShape.GetDimNum() == 2 && groupListShape.GetDim(1) == 1;
    if ((xShape.GetDimNum() != 2 && xShape.GetDimNum() != 3) || weightShape.GetDimNum() != 1 ||
        (groupListShape.GetDimNum() != 1 && !cumulativeGroupList) || weightScaleShape.GetDimNum() != 2 ||
        xScaleShape.GetDimNum() != 1) {
        return ge::GRAPH_FAILED;
    }

    const int64_t expertNum = groupListShape.GetDim(0);
    const int64_t scaleGroups = weightScaleShape.GetDim(0);
    const int64_t n = weightScaleShape.GetDim(1);
    const int64_t packedRows = xShape.GetDim(0);
    const bool ndInput = xShape.GetDimNum() == 3;
    const int64_t groupSize = xShape.GetDim(ndInput ? 2 : 1);
    if (expertNum <= 0 || scaleGroups <= 0 || scaleGroups % expertNum != 0 ||
        n <= 0 || packedRows <= 0 || groupSize <= 0) {
        return ge::GRAPH_FAILED;
    }
    const int64_t kGroupNum = scaleGroups / expertNum;
    if ((!ndInput && packedRows % kGroupNum != 0) ||
        (ndInput && xShape.GetDim(1) != kGroupNum)) {
        return ge::GRAPH_FAILED;
    }
    const int64_t m = ndInput ? packedRows : packedRows / kGroupNum;
    if (xScaleShape.GetDim(0) != m) {
        return ge::GRAPH_FAILED;
    }
    const int64_t expectedWeightElements = expertNum * kGroupNum * n * groupSize;
    if (weightShape.GetDim(0) != expectedWeightElements) {
        return ge::GRAPH_FAILED;
    }

    QuantMatmulExpertKGroupedTilingData tilingData;
    const uint32_t requestedCoreNum = ReadEnvUint("GROUP_QUANT_GMM2_CORE_NUM", DEFAULT_CORE_NUM);
    const uint32_t coreNum = std::max<uint32_t>(1, std::min<uint32_t>(compileInfo->aicNum, requestedCoreNum));
    const uint32_t parallNum = std::max<uint32_t>(1, std::min<uint32_t>(
        ReadEnvUint("GROUP_QUANT_GMM2_PARALL_NUM", DEFAULT_PARALL_NUM), 8));
    // Tiling key 0 keeps the production 1:2 full GM-workspace pipeline. Key 1
    // is a 1:1 bounded ping-pong pipeline. On Ascend 910B the Matmul client
    // cannot asynchronously produce directly into a LocalTensor, so key 1
    // uses two base-tile staging slots per core and consumes them through two
    // VECIN slots. The bounded staging area enables Cube/Vector overlap without
    // restoring the full M*N INT32 workspace.
    // Both binaries are emitted from the same operator package; the runtime
    // switch makes precision/performance A/B reversible without rebuilding.
    const auto *weightScaleDesc = context->GetInputDesc(WEIGHT_SCALE_INDEX);
    if (weightScaleDesc == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const bool bf16WeightScale = weightScaleDesc->GetDataType() == ge::DT_BF16;
    const bool useVecinInt32 = !bf16WeightScale &&
        ReadEnvUint("GROUP_QUANT_GMM2_LOCAL_INT32", 0) != 0;
    // The experimental local-int32 template has a different A-address path.
    if (ndInput && useVecinInt32) {
        return ge::GRAPH_FAILED;
    }
    uint32_t baseM = useVecinInt32
        ? ReadEnvUint("GROUP_QUANT_GMM2_LOCAL_BASE_M", 32)
        : ReadEnvUint("GROUP_QUANT_GMM2_BASE_M", BASE_M);
    uint32_t baseN = useVecinInt32
        ? ReadEnvUint("GROUP_QUANT_GMM2_LOCAL_BASE_N", 512)
        : ReadEnvUint("GROUP_QUANT_GMM2_BASE_N", BASE_N);
    uint32_t baseK = ReadEnvUint("GROUP_QUANT_GMM2_BASE_K", BASE_K);
    baseM = std::min<uint32_t>(baseM, 256);
    baseN = std::min<uint32_t>(baseN, 512);
    baseK = std::min<uint32_t>(baseK, 256);
    baseM = std::max<uint32_t>(16, (baseM / 16) * 16);
    baseN = std::max<uint32_t>(32, (baseN / 32) * 32);
    baseK = std::max<uint32_t>(16, (baseK / 16) * 16);
    constexpr uint32_t batchedBaseN = 256;
    const bool batchExpertNTiles = !useVecinInt32 && kGroupNum == 1 &&
        m <= std::min(baseM, BASE_M) && n > batchedBaseN && n % batchedBaseN == 0 &&
        ReadEnvUint("GROUP_QUANT_GMM2_BATCH_EXPERT_N_TILES", 0) != 0;
    if (batchExpertNTiles) {
        baseN = batchedBaseN;
    }
    // Keep the Cube base tile independent from the AIC/AIV hand-off tile.
    // A larger logical M tile lets one cross-core flag cover several baseM
    // matmul iterations, while the Matmul API still uses baseM x baseN.
    uint32_t singleM = ReadEnvUint("GROUP_QUANT_GMM2_SINGLE_M", baseM);
    singleM = std::min<uint32_t>(singleM, 256);
    singleM = std::max<uint32_t>(baseM, (singleM / baseM) * baseM);
    const uint32_t enableTilePipeline = std::min<uint32_t>(
        ReadEnvUint("GROUP_QUANT_GMM2_ENABLE_TILE_PIPELINE", 1), 1);
    tilingData.set_coreNum(coreNum);
    tilingData.set_groupNum(static_cast<uint32_t>(expertNum));
    tilingData.set_totalInGroup(static_cast<uint32_t>(m));
    tilingData.set_k(static_cast<uint32_t>(groupSize));
    tilingData.set_n(static_cast<uint32_t>(n));
    tilingData.set_groupAxis(2);
    tilingData.set_kGroupSize(static_cast<uint32_t>(groupSize));
    tilingData.set_kGroupNum(static_cast<uint32_t>(kGroupNum));
    tilingData.set_ubCalSize(UB_CAL_SIZE);
    tilingData.set_ubRestBytes(UB_REST_BYTES);
    tilingData.set_parallNum(parallNum);
    tilingData.set_enableTilePipeline(enableTilePipeline);
    tilingData.set_tileDynamicSchedule(0);
    tilingData.set_reserved0(singleM);
    const bool streamExpertTiles = ReadEnvUint("GROUP_QUANT_GMM2_STREAM_EXPERT_TILES", 0) != 0;
    const bool expertMOuter = ReadEnvUint("GROUP_QUANT_GMM2_EXPERT_M_OUTER", 0) != 0;
    const uint32_t expertSchedule = streamExpertTiles ? 2U : static_cast<uint32_t>(expertMOuter);
    const bool dualAiv = ReadEnvUint("GROUP_QUANT_GMM2_DUAL_AIV", 1, true) != 0;
    // Keep the legacy single-consumer epilogue selectable in the same binary.
    const uint32_t workspacePipelineFlags =
        (dualAiv ? 1U : 0U) | (ndInput ? 2U : 0U) | (bf16WeightScale ? 4U : 0U);
    tilingData.set_reserved2((useVecinInt32 ? 0U : workspacePipelineFlags) |
        (cumulativeGroupList ? 8U : 0U));

    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    matmul_tiling::MultiCoreMatmulTiling mmTiling(platform);
    mmTiling.SetDim(1);
    mmTiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                      matmul_tiling::DataType::DT_INT8, false);
    mmTiling.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::NZ,
                      matmul_tiling::DataType::DT_INT8, false);
    mmTiling.SetCType(useVecinInt32 ? matmul_tiling::TPosition::VECIN : matmul_tiling::TPosition::GM,
                      matmul_tiling::CubeFormat::ND,
                      matmul_tiling::DataType::DT_INT32);
    mmTiling.SetBias(false);
    // A ND rows span all groups; B remains a single compact K group.
    mmTiling.SetOrgShape(m, n, ndInput ? kGroupNum * groupSize : groupSize,
                         groupSize);
    mmTiling.SetShape(m, n, groupSize);
    mmTiling.SetFixSplit(std::min<int64_t>(baseM, m), std::min<int64_t>(baseN, n),
                         std::min<int64_t>(baseK, groupSize));
    if (mmTiling.GetTiling(tilingData.mmTilingData) == -1) {
        return ge::GRAPH_FAILED;
    }
    const bool useBatchedSchedule = batchExpertNTiles && tilingData.mmTilingData.get_baseM() >= m;
    tilingData.set_reserved1(useBatchedSchedule ? 3U : expertSchedule);
    tilingData.mmTilingData.set_dbL0C(1);
    tilingData.mmTilingData.set_stepKa(4);
    tilingData.mmTilingData.set_stepKb(4);
    tilingData.mmTilingData.set_depthA1(8);
    tilingData.mmTilingData.set_depthB1(8);
    tilingData.mmTilingData.set_stepM(1);
    tilingData.mmTilingData.set_stepN(1);

    const uint64_t mmWorkspace = static_cast<uint64_t>(parallNum) *
                                 static_cast<uint64_t>(tilingData.mmTilingData.get_baseN()) *
                                 static_cast<uint64_t>(singleM) *
                                 sizeof(int32_t) * coreNum * kGroupNum;
    auto workspaceSizes = context->GetWorkspaceSizes(1);
    const uint64_t localPipelineWorkspace =
        2ULL * static_cast<uint64_t>(tilingData.mmTilingData.get_baseM()) *
        static_cast<uint64_t>(tilingData.mmTilingData.get_baseN()) *
        sizeof(int32_t) * coreNum;
    workspaceSizes[0] = compileInfo->sysWorkspaceSize +
        (useVecinInt32 ? localPipelineWorkspace : mmWorkspace);
    // Key 2 keeps the batched GM workspace pipeline with one physical AIV.
    const uint64_t tilingKey = useVecinInt32 ? 1 : (useBatchedSchedule && !dualAiv ? 2 : 0);
    context->SetTilingKey(tilingKey);
    context->SetBlockDim(coreNum);
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(),
                            context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingParse(gert::TilingParseContext *context)
{
    auto *platformInfo = context->GetPlatformInfo();
    auto *compileInfo = context->GetCompiledInfo<CompileInfo>();
    if (platformInfo == nullptr || compileInfo == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto platform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->aicNum = platform.GetCoreNumAic();
    compileInfo->sysWorkspaceSize = platform.GetLibApiWorkSpaceSize();
    return ge::GRAPH_SUCCESS;
}
} // namespace

IMPL_OP_OPTILING(QuantMatmulExpertKGrouped)
    .Tiling(TilingFunc)
    .TilingParse<CompileInfo>(TilingParse);
} // namespace optiling
