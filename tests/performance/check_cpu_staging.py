"""Verify CPU staging preserves all quantized parameters on a small MoE layer."""
import json
from pathlib import Path
from types import SimpleNamespace
import torch
import torch_npu
from Smoothquant.custom_moe_quant import GroupQuantFusedMoEMethod
from vllm.model_executor.model_loader.utils import device_loading_context
from tests.performance.resource_probe import install_cpu_staging

torch.manual_seed(42)
torch_npu.npu.config.allow_internal_format=True
method=object.__new__(GroupQuantFusedMoEMethod)
method.quant_config=SimpleNamespace(quantize_weights=True,weight_bits=8,residual_bits=4,residual_group=2048)
method.dynamic_eplb=False
sources={'w13_weight':torch.randn(4,1536,2048,dtype=torch.bfloat16)*.03,'w2_weight':torch.randn(4,2048,768,dtype=torch.bfloat16)*.03}

def make(staged):
    if staged:install_cpu_staging()
    with torch.device('npu:0'):
        layer=torch.nn.Module()
        for name,tensor in method.get_weight(4,768,2048,torch.bfloat16).items():
            assert tensor.device.type==('cpu' if staged else 'npu')
            tensor.copy_(sources[name]);layer.register_parameter(name,torch.nn.Parameter(tensor,requires_grad=False))
        for name,tensor in method.get_dynamic_quant_param(4,768,2048,torch.bfloat16).items():
            layer.register_parameter(name,torch.nn.Parameter(tensor,requires_grad=False))
    with device_loading_context(layer,torch.device('npu:0')):method.process_weights_after_loading(layer)
    assert all(p.device.type=='npu' for p in layer.parameters())
    assert not hasattr(layer,'w13_weight') and not hasattr(layer,'w2_weight')
    torch.npu.synchronize()
    return layer

normal=make(False);staged=make(True)
a=dict(normal.named_parameters());b=dict(staged.named_parameters());assert a.keys()==b.keys()
for name in a:assert torch.equal(a[name].cpu(),b[name].cpu()),name
assert torch.equal(normal.w13_weight_fixpipe_scale.cpu(),staged.w13_weight_fixpipe_scale.cpu())
result={'result':'PASS','parameters_compared':list(a),'extra_compared':['w13_weight_fixpipe_scale'],'all_runtime_parameters_on_npu':True,'bf16_sources_deleted':True,'bitwise_equal':True,'shape':{'experts':4,'hidden':2048,'intermediate':768}}
print(json.dumps(result,indent=2))
