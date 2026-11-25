#include "register/op_def_registry.h"

namespace ops {
class Gmm1VecinFullk : public OpDef {
public:
    explicit Gmm1VecinFullk(const char *name) : OpDef(name)
    {
        this->Input("x").ParamType(REQUIRED).DataType({ge::DT_INT8}).Format({ge::FORMAT_ND});
        this->Input("weight").ParamType(DYNAMIC).DataType({ge::DT_INT8}).Format({ge::FORMAT_FRACTAL_NZ});
        // Pack all experts into one [E,G,N] tensor.  Keeping both weight and
        // scale as 128-item dynamic lists would exceed the runtime's 256-input
        // launcher limit for Qwen3-30B-A3B.
        this->Input("weight_scale").ParamType(REQUIRED).DataType({ge::DT_INT64}).Format({ge::FORMAT_ND});
        this->Input("x_scale").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("group_list").ParamType(REQUIRED).DataType({ge::DT_INT64}).Format({ge::FORMAT_ND});
        this->Output("y").ParamType(REQUIRED).DataType({ge::DT_INT8}).Format({ge::FORMAT_ND});
        this->Output("y_scale").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        OpAICoreConfig config;
        config.DynamicCompileStaticFlag(true).DynamicFormatFlag(true).DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false).PrecisionReduceFlag(true);
        this->AICore().AddConfig("ascend910b", config);
    }
};
OP_ADD(Gmm1VecinFullk);
}  // namespace ops
