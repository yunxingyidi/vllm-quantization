"""Offline-parse vLLM daemon-worker NPU profiles serially."""
from pathlib import Path
import sys

from torch_npu.profiler.profiler import analyse

for root in sys.argv[1:]:
    for profile in sorted(Path(root).glob('*_ascend_pt')):
        print(f'ANALYSE {profile}', flush=True)
        analyse(str(profile), max_process_number=2)
        print(f'FINISHED {profile}', flush=True)
