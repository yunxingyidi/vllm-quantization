#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
op_name="grouped_matmul_swiglu_quant_weight_nz_tensor_list"
log_dir="${LOG_DIR:-${repo_root}/tests/artifacts/${op_name}_stage9}"
python_bin="${PYTHON:-python}"
rebuild_stage9="${REBUILD_STAGE9:-0}"

mkdir -p "${log_dir}"

cd "${repo_root}"
if [[ -f "${repo_root}/env_vllm_ascend.sh" ]]; then
  source "${repo_root}/env_vllm_ascend.sh"
fi

if [[ "${rebuild_stage9}" == "1" ]]; then
  GROUP_QUANT_STAGE=9 "${repo_root}/custom_op/scripts/build_groupquant_gmm_swiglu.sh" \
    "${log_dir}/build_stage9.log"
fi

source "${repo_root}/custom_op/build_root/install/vendors/custom_transformer/bin/set_env.bash"

stage_header="${repo_root}/custom_op/build_root/csrc/gmm/${op_name}/op_kernel/${op_name}.h"
if ! grep -q "#define GROUP_QUANT_STAGE 9" "${stage_header}"; then
  echo "stage9 is not installed; run with REBUILD_STAGE9=1 or build GROUP_QUANT_STAGE=9 first" >&2
  grep -n "#define GROUP_QUANT_STAGE" "${stage_header}" >&2 || true
  exit 1
fi

run_case() {
  local name="$1"
  shift
  local log_path="${log_dir}/${name}.log"
  echo "[STAGE9] ${name}"
  timeout 300s "${python_bin}" -m tests.test_groupquant_gmm_swiglu_op "$@" > "${log_path}" 2>&1
  grep -q "quant_output_equal=True" "${log_path}"
  grep -E "config:|scale_dtype|group_size|quant_output_equal|scale_mean_abs_diff|scale_max_abs_diff|avg_latency_ms|throughput_tokens_per_s|effective_tops" "${log_path}" | tail -n 12
}

run_perf() {
  local name="$1"
  shift
  local log_path="${log_dir}/${name}.log"
  echo "[STAGE9-PERF] ${name}"
  timeout 600s "${python_bin}" -m tests.test_groupquant_gmm_swiglu_op --perf-only "$@" > "${log_path}" 2>&1
  grep -E "config:|group_size|performance:|avg_latency_ms|throughput_tokens_per_s|effective_tops" "${log_path}" | tail -n 10
}

run_case ksplit_fp32 \
  --m 2 --k 512 --n 256 --experts 2 --group-size 128 \
  --scale-dtype fp32 --warmup 0 --iters 1

run_case ksplit_fp16 \
  --m 2 --k 512 --n 256 --experts 2 --group-size 128 \
  --scale-dtype fp16 --warmup 0 --iters 1

run_case zero_token_expert \
  --m 2 --k 512 --n 256 --experts 4 --group-size 128 \
  --scale-dtype fp32 --warmup 0 --iters 1

run_case multi_row_multi_expert \
  --m 128 --k 512 --n 256 --experts 4 --group-size 128 \
  --scale-dtype fp32 --warmup 1 --iters 5

run_perf moe_like_decode_small \
  --m 128 --k 4096 --n 4096 --experts 4 --group-size 128 \
  --scale-dtype fp32 --warmup 1 --iters 5

run_perf moe_like_decode_mid \
  --m 512 --k 4096 --n 4096 --experts 4 --group-size 128 \
  --scale-dtype fp32 --warmup 1 --iters 5

run_perf moe_like_wide \
  --m 128 --k 4096 --n 8192 --experts 4 --group-size 128 \
  --scale-dtype fp32 --warmup 1 --iters 3

echo "[STAGE9] all cases passed; logs: ${log_dir}"
