"""Set PipeUtilization on the built-in vLLM Ascend worker profiler only.

Loaded exclusively when this directory is explicitly added to PYTHONPATH for
the diagnostic capture.  Does not edit the installed vLLM package.
"""
import os

if os.getenv("GROUP_QUANT_PROFILE_PIPE_UTILIZATION") == "1":
    import torch_npu.profiler

    _original_config = torch_npu.profiler._ExperimentalConfig

    def _pipe_config(*args, **kwargs):
        if kwargs.get("aic_metrics") == torch_npu.profiler.AiCMetrics.AiCoreNone:
            kwargs["aic_metrics"] = torch_npu.profiler.AiCMetrics.PipeUtilization
        return _original_config(*args, **kwargs)

    torch_npu.profiler._ExperimentalConfig = _pipe_config

    # The bundled worker calls start()/stop() but does not call step(). Use a
    # one-step schedule and close the active window immediately before stop;
    # this avoids ending the capture while its action is RECORD. Parsing still
    # happens offline because vLLM workers are daemon processes.
    _original_profile = torch_npu.profiler.profile

    def _scheduled_profile(*args, **kwargs):
        if os.getenv("GROUP_QUANT_SERVING_PROFILE_SHAPES") == "1":
            kwargs["record_shapes"] = True
        if "schedule" not in kwargs and kwargs.get("on_trace_ready") is not None:
            kwargs["schedule"] = torch_npu.profiler.schedule(
                wait=0, warmup=0, active=1, repeat=1)
            prof = _original_profile(*args, **kwargs)
            original_stop = prof.stop

            def _complete_and_stop():
                if not prof.stopped:
                    prof.step()
                return original_stop()

            prof.stop = _complete_and_stop
            return prof
        return _original_profile(*args, **kwargs)

    torch_npu.profiler.profile = _scheduled_profile
