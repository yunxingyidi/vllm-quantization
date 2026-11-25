"""Opt-in hardware counter capture, separate from unprofiled benchmark timing."""
from pathlib import Path
import os


def capture_operator(fn):
    output = os.environ.get("GROUP_QUANT_OPERATOR_PROFILE")
    if not output:
        return
    import torch
    import torch_npu
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    metrics = getattr(torch_npu.profiler.AiCMetrics,
                      os.environ.get("GROUP_QUANT_PROFILE_METRIC", "PipeUtilization"))
    # Capture steady graph replays; input generation, compilation, allocation
    # during capture, and warmup remain outside the measured profile window.
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        keepalive = fn()
    for _ in range(10):
        graph.replay()
    torch.npu.synchronize()
    config = torch_npu.profiler._ExperimentalConfig(
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        aic_metrics=metrics)
    with torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.CPU,
                    torch_npu.profiler.ProfilerActivity.NPU],
        record_shapes=True,
        schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=1, repeat=1),
        experimental_config=config,
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(output)),
    ) as prof:
        with torch.profiler.record_function("paired_operator_steady_10_replays"):
            for _ in range(10):
                graph.replay()
            torch.npu.synchronize()
        prof.step()
    assert keepalive is not None
