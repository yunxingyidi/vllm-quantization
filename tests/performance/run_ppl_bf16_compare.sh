#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
export ASCEND_RT_VISIBLE_DEVICES=4,5
export GROUP_QUANT_GMM2_GROUP_SIZE=0 GROUP_QUANT_GMM1_BACKEND=auto
export GROUP_QUANT_DISPATCH_SCATTER=0 GROUP_QUANT_BENCH_NATIVE_GMM2=0
export GROUP_QUANT_PROFILE=0 GROUP_QUANT_PROFILE_PIPE_UTILIZATION=0
export VLLM_ASCEND_ENABLE_FUSED_MC2=0 DYNAMIC_EPLB=false
unset VLLM_TORCH_PROFILER_DIR
source "$repo_root/env_vllm_ascend.sh"
set -euo pipefail
out_dir="${PPL_COMPARE_OUTPUT:-tests/artifacts/ppl_bf16_20261102}"
mkdir -p "$out_dir"
python - "$out_dir" <<'PY'
import hashlib,json,sys
from pathlib import Path
from importlib.metadata import version
import torch
from tests.ppl_data import get_test_loader
from ppl_eval import build_eval_windows
model='/storage2/Qwen3-30B-A3B/'
x=get_test_loader('wikitext2',seqlen=2048,model=model).input_ids
windows=build_eval_windows(x,2048,None)
manifest={'model':model,'dataset':'wikitext2','seqlen':2048,'windows':len(windows),'prompt_tokens':sum(map(len,windows)),
'predicted_tokens':sum(len(w)-1 for w in windows),'input_sha256':hashlib.sha256(torch.tensor(windows,dtype=torch.int64).numpy().tobytes()).hexdigest(),
'group_size_gmm1':2048,'group_size_gmm2':384,'activation_scale':'per-token','quantize_moe':True,'quantize_linear':False,
'packages':{p:version(p) for p in ['torch','torch-npu','vllm','vllm-ascend']}}
Path(sys.argv[1],'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(manifest,flush=True)
PY
common=(--model /storage2/Qwen3-30B-A3B/ --dataset wikitext2 --seqlen 2048
        --batch-size 6 --dtype bf16 --tensor-parallel-size 2 --seed 0
        --max-num-seqs 6 --max-num-batched-tokens 12294 --gpu-memory-utilization 0.70
        --no-enforce-eager --no-enable-prefix-caching --no-enable-chunked-prefill)
for mode in bf16 quant; do
    args=("${common[@]}")
    if [[ "$mode" == quant ]]; then
        args+=(--quantization group_quant --quantize-moe --no-quantize-linear --expert-gmm2-op --residual-group 2048)
    else
        args+=(--no-expert-gmm2-op --no-gmm1-vecin-fullk)
    fi
    printf '%q ' python ppl_eval.py "${args[@]}" > "$out_dir/${mode}_command.sh"
    printf '\n' >> "$out_dir/${mode}_command.sh"
    echo "PPL_START $mode"
    PPL_RESULT_PATH="$out_dir/$mode.json" python ppl_eval.py "${args[@]}" > "$out_dir/$mode.log" 2>&1
    cat "$out_dir/$mode.json"
    echo "PPL_COMPLETE $mode"
done
