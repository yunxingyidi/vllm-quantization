#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${repo_root}"

# The default path enables expert-aware GMM2 and enables fused GMM1 where its
# installed vLLM-Ascend op-api supports K-group scales.  Both paths use native
# per-token activation scales; weight scales remain K-grouped.
# GMM1 automatically uses native full-K or custom NZ/M32; GMM2 uses dual AIV.
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-4,5}"
source "${repo_root}/env_vllm_ascend.sh"
set -euo pipefail

model_path="${MODEL_PATH:-/storage2/Qwen3-30B-A3B/}"
dataset="${DATASET:-wikitext2}"
seqlen="${SEQLEN:-2048}"
max_windows="${MAX_WINDOWS:-100}"
tp_size="${TP_SIZE:-2}"
batch_size="${BATCH_SIZE:-6}"
max_num_seqs="${MAX_NUM_SEQS:-${batch_size}}"
max_num_batched_tokens="${MAX_NUM_BATCHED_TOKENS:-$((batch_size * (seqlen + 1)))}"
# Requested MoE group size. Each TP-local projection automatically falls back
# to gcd(K_local, GROUP_SIZE), so Qwen3-30B-A3B uses one full-K 2048 group for
# GMM1. GMM2 independently defaults to one full TP-local K=384 group through
# GROUP_QUANT_GMM2_GROUP_SIZE=0, removing its cross-group partial/FixPipe work.
group_size="${GROUP_SIZE:-2048}"
enforce_eager="${ENFORCE_EAGER:-0}"

if (( batch_size < 1 || max_num_seqs < batch_size )); then
    echo "Require BATCH_SIZE >= 1 and MAX_NUM_SEQS >= BATCH_SIZE" >&2
    exit 2
fi
if (( max_num_batched_tokens < batch_size * (seqlen + 1) )); then
    echo "MAX_NUM_BATCHED_TOKENS is too small for BATCH_SIZE x (SEQLEN + 1)" >&2
    exit 2
fi
if (( group_size < 16 || group_size % 16 != 0 )); then
    echo "GROUP_SIZE must be a positive multiple of 16" >&2
    exit 2
fi
if [[ "${enforce_eager}" != "0" && "${enforce_eager}" != "1" ]]; then
    echo "ENFORCE_EAGER must be 0 or 1" >&2
    exit 2
fi

execution_args=(--no-enforce-eager)
if [[ "${enforce_eager}" == "1" ]]; then
    execution_args=(--enforce-eager)
fi

quant_args=(--quantization group_quant --quantize-moe --no-quantize-linear --expert-gmm2-op --residual-group "${group_size}")
default_gpu_memory_utilization=0.98

parallel_args=()
if [[ "${ENABLE_EXPERT_PARALLEL:-0}" == "1" ]]; then
    parallel_args+=(--enable-expert-parallel)
fi
if [[ "${DYNAMIC_EPLB}" == "true" || "${DYNAMIC_EPLB}" == "1" ]]; then
    if [[ "${ENABLE_EXPERT_PARALLEL:-0}" != "1" ]]; then
        echo "DYNAMIC_EPLB requires ENABLE_EXPERT_PARALLEL=1" >&2
        exit 2
    fi
    if [[ -n "${EPLB_ADDITIONAL_CONFIG:-}" ]]; then
        eplb_config="${EPLB_ADDITIONAL_CONFIG}"
    else
        printf -v eplb_config \
            '{"eplb_config":{"dynamic_eplb":true,"expert_heat_collection_interval":400,"algorithm_execution_interval":30,"eplb_policy_type":2,"num_redundant_experts":%d}}' \
            "${tp_size}"
    fi
    parallel_args+=(--additional-config "${eplb_config}")
fi

python ppl_eval.py \
    --model "${model_path}" \
    --dataset "${dataset}" \
    --seqlen "${seqlen}" \
    --max-windows "${max_windows}" \
    --batch-size "${batch_size}" \
    --dtype bf16 \
    --tensor-parallel-size "${tp_size}" \
    "${parallel_args[@]}" \
    "${quant_args[@]}" \
    --max-num-seqs "${max_num_seqs}" \
    --max-num-batched-tokens "${max_num_batched_tokens}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-${default_gpu_memory_utilization}}" \
    "${execution_args[@]}" \
    --no-enable-prefix-caching \
    --no-enable-chunked-prefill \
    "$@"
