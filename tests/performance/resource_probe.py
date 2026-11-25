"""Opt-in lifecycle snapshots and quiescent-boundary worker memory RPC.

No execute_model hook: no per-step synchronization in the timing experiment.
Allocator peaks cover activations plus workspaces, not pure operator workspace.
"""
import functools
import json
import os
from pathlib import Path
import time

STAGES = []


def snapshot(label):
    import torch_npu
    npu = torch_npu.npu
    npu.synchronize()
    free, total = npu.mem_get_info()
    result = dict(label=label, pid=os.getpid(), logical_device=npu.current_device(),
                  wall_time=time.time(), allocated=npu.memory_allocated(),
                  reserved=npu.memory_reserved(), peak_allocated=npu.max_memory_allocated(),
                  peak_reserved=npu.max_memory_reserved(), device_used=total-free,
                  device_total=total)
    return result


def install():
    if not os.environ.get('GROUP_QUANT_RESOURCE_OUTPUT'):
        return
    if os.environ.get('GROUP_QUANT_STAGE_BF16_ON_CPU') == '1':
        install_cpu_staging()
    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner
    if getattr(NPUModelRunner, '_resource_probe_installed', False):
        return
    NPUModelRunner._resource_probe_installed = True
    for name in ('load_model', 'profile_run', 'initialize_kv_cache', 'capture_model'):
        original = getattr(NPUModelRunner, name)

        @functools.wraps(original)
        def wrapped(self, *args, _original=original, _name=name, **kwargs):
            STAGES.append(snapshot(_name + '_before'))
            result = _original(self, *args, **kwargs)
            STAGES.append(snapshot(_name + '_after'))
            out = Path(os.environ['GROUP_QUANT_RESOURCE_OUTPUT'])
            out.mkdir(parents=True, exist_ok=True)
            (out / f'lifecycle_{os.getpid()}.json').write_text(json.dumps(STAGES, indent=2))
            return result

        setattr(NPUModelRunner, name, wrapped)


def install_cpu_staging():
    """Keep only checkpoint source parameters on host, not runtime weights.

    vLLM's existing device_loading_context moves a module's CPU parameters to
    NPU for processing. GroupQuant deletes these BF16 source parameters after
    quantization; its preallocated INT8 runtime parameters remain on NPU.
    """
    import torch
    from Smoothquant.custom_moe_quant import GroupQuantFusedMoEMethod
    if getattr(GroupQuantFusedMoEMethod, '_resource_cpu_staging', False):
        return
    original = GroupQuantFusedMoEMethod.get_weight

    @functools.wraps(original)
    def staged(self, *args, **kwargs):
        with torch.device('cpu'):
            return original(self, *args, **kwargs)

    GroupQuantFusedMoEMethod.get_weight = staged
    GroupQuantFusedMoEMethod._resource_cpu_staging = True


class ResourceWorkerExtension:
    def resource_snapshot(self, label='snapshot', reset='0'):
        import torch_npu
        value = snapshot(label)
        value['rank'] = self.rank
        value['stages'] = STAGES
        if reset == '1':
            torch_npu.npu.reset_peak_memory_stats()
        return value
