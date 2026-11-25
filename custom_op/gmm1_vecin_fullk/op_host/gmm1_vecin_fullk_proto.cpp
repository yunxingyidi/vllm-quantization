#include "register/op_impl_registry.h"

namespace ops {
static ge::graphStatus InferShapeProto(gert::InferShapeContext *ctx)
{
    const gert::Shape *x = ctx->GetInputShape(0);
    const gert::Shape *weight = ctx->GetDynamicInputShape(1, 0);
    if (x == nullptr || weight == nullptr || weight->GetDimNum() != 4) {
        return ge::GRAPH_FAILED;
    }
    gert::Shape *y = ctx->GetOutputShape(0);
    y->SetDimNum(2);
    const int64_t outputM = x->GetDim(0);
    y->SetDim(0, outputM);
    y->SetDim(1, weight->GetDim(0) * weight->GetDim(3) / 2);
    gert::Shape *scale = ctx->GetOutputShape(1);
    scale->SetDimNum(1);
    scale->SetDim(0, outputM);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDtypeProto(gert::InferDataTypeContext *ctx)
{
    ctx->SetOutputDataType(0, ge::DT_INT8);
    ctx->SetOutputDataType(1, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(Gmm1VecinFullk).InferShape(InferShapeProto).InferDataType(InferDtypeProto);
}  // namespace ops
