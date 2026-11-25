"""Register this project's quantization, then invoke the unmodified vLLM CLI.

Top-level registration also runs when Python spawn imports the main module.
No benchmark, sampling, HTTP, scheduling or timing code is replaced.
"""
from pathlib import Path
import os
import sys
from types import SimpleNamespace

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
if os.getenv('GROUP_QUANT_CLI_REGISTER','0')=='1':
    from ppl_eval import configure_expert_gmm2_op,configure_gmm1_vecin_fullk
    options=SimpleNamespace(expert_gmm2_op=True,gmm1_vecin_fullk=True)
    configure_expert_gmm2_op(options)
    configure_gmm1_vecin_fullk(options)
    import Smoothquant.group_quant_config  # noqa: F401
    from vllm.platforms import current_platform
    if 'group_quant' not in current_platform.supported_quantization:
        current_platform.supported_quantization.append('group_quant')

if os.environ.get('GROUP_QUANT_RESOURCE_OUTPUT'):
    from tests.performance.resource_probe import install
    install()

if __name__=='__main__':
    from vllm.entrypoints.cli.main import main
    main()
