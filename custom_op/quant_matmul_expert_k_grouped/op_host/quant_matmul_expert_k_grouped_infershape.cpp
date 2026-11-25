#include "register/op_impl_registry.h"

namespace ops {
namespace {
constexpr size_t X_INDEX = 0;
constexpr size_t GROUP_LIST_INDEX = 2;
constexpr size_t WEIGHT_SCALE_INDEX = 3;

ge::graphStatus InferShape(gert::InferShapeContext *context)
{
    const auto *xShape = context->GetInputShape(X_INDEX);
    const auto *groupListShape = context->GetInputShape(GROUP_LIST_INDEX);
    const auto *weightScaleShape = context->GetInputShape(WEIGHT_SCALE_INDEX);
    if (xShape == nullptr || groupListShape == nullptr || weightScaleShape == nullptr ||
        (xShape->GetDimNum() != 2 && xShape->GetDimNum() != 3) || groupListShape->GetDimNum() != 1 ||
        weightScaleShape->GetDimNum() != 2) {
        return ge::GRAPH_FAILED;
    }
    const int64_t expertNum = groupListShape->GetDim(0);
    const int64_t scaleGroups = weightScaleShape->GetDim(0);
    if (expertNum <= 0 || scaleGroups <= 0 || scaleGroups % expertNum != 0) {
        return ge::GRAPH_FAILED;
    }
    const int64_t kGroupNum = scaleGroups / expertNum;
    const int64_t packedRows = xShape->GetDim(0);
    const bool ndInput = xShape->GetDimNum() == 3;
    if (packedRows < 0 || (ndInput ? xShape->GetDim(1) != kGroupNum : packedRows % kGroupNum != 0)) {
        return ge::GRAPH_FAILED;
    }
    auto *outShape = context->GetOutputShape(0);
    outShape->SetDimNum(2);
    outShape->SetDim(0, ndInput ? packedRows : packedRows / kGroupNum);
    outShape->SetDim(1, weightScaleShape->GetDim(1));
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, ge::DT_BF16);
    return ge::GRAPH_SUCCESS;
}
} // namespace

IMPL_OP_INFERSHAPE(QuantMatmulExpertKGrouped)
    .InferShape(InferShape)
    .InferDataType(InferDataType);
} // namespace ops
