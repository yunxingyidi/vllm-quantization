#!/usr/bin/env bash
# Run one of the two comparable PPL workloads under an external profiler.
set -euo pipefail

# msprof's --application launches only an executable and does not forward
# positional arguments. Environment variables keep this runner usable both
# directly and through msprof.
mode="${PROFILE_PPL_MODE:-${1:-}}"
windows="${PROFILE_PPL_WINDOWS:-${2:-100}}"
if [[ -z "${mode}" ]]; then
    echo "usage: PROFILE_PPL_MODE=<fp16|group_quant> $0 [mode] [windows]" >&2
    exit 2
fi
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

source "${repo_root}/env_vllm_ascend.sh"
cd "${repo_root}"

common_args=(
    --model /storage2/Qwen3-30B-A3B
    --dataset wikitext2
    --seqlen 2048
    --max-windows "${windows}"
    --batch-size 1
    --tensor-parallel-size 2
    --max-num-seqs 1
    --gpu-memory-utilization 0.90
    --enforce-eager
    --no-enable-prefix-caching
    --no-enable-chunked-prefill
)

case "${mode}" in
    fp16)
        PPL_RESULT_PATH="${PPL_RESULT_PATH:-tests/artifacts/ppl_fp16_unquantized_${windows}w_profile.json}" \
            python ppl_eval.py "${common_args[@]}" --dtype fp16 \
            --no-expert-gmm2-op --no-gmm1-vecin-fullk
        ;;
    group_quant)
        PPL_RESULT_PATH="${PPL_RESULT_PATH:-tests/artifacts/ppl_moe_gmm1_vecin_fused_${windows}w_profile.json}" \
            python ppl_eval.py "${common_args[@]}" --dtype bf16 \
            --quantization group_quant --weight-bits 8 --act-bits 8 \
            --residual-bits 4 --residual-group 4096 --quantize-moe \
            --no-quantize-linear --expert-gmm2-op --gmm1-vecin-fullk
        ;;
    *)
        echo "unsupported mode: ${mode}; expected fp16 or group_quant" >&2
        exit 2
        ;;
esac
