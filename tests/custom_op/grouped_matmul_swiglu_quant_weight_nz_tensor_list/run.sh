#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
log_dir="${LOG_DIR:-${repo_root}/tests/artifacts/grouped_matmul_swiglu_quant_weight_nz_tensor_list}"
python_bin="${PYTHON:-python}"
mkdir -p "${log_dir}"

cd "${repo_root}"
if [[ -f "${repo_root}/env_vllm_ascend.sh" ]]; then
  source "${repo_root}/env_vllm_ascend.sh"
fi
source "${repo_root}/custom_op/build_root/install/vendors/custom_transformer/bin/set_env.bash"

run_case() {
  local name="$1"
  shift
  local log_path="${log_dir}/${name}.log"
  echo "[ST] ${name}"
  timeout 180s "${python_bin}" -m tests.test_groupquant_gmm_swiglu_op "$@" > "${log_path}" 2>&1
  grep -q "quant_output_equal=True" "${log_path}"
  grep -E "config:|quant_output_equal|scale_mean_abs_diff|scale_max_abs_diff|avg_latency_ms" "${log_path}" | tail -n 8
}

run_expected_bf16_reject() {
  local log_path="${log_dir}/bf16_reject.log"
  echo "[ST] bf16_reject"
  set +e
  timeout 180s "${python_bin}" -m tests.test_groupquant_gmm_swiglu_op \
    --m 2 --k 512 --n 256 --experts 2 --group-size 128 \
    --scale-dtype bf16 --warmup 0 --iters 1 > "${log_path}" 2>&1
  local rc=$?
  set -e
  if [[ ${rc} -eq 0 ]]; then
    echo "bf16_reject expected failure but command succeeded" >&2
    tail -n 40 "${log_path}" >&2
    return 1
  fi
  grep -q "not implemented for DT_BFLOAT16" "${log_path}"
  grep -E "scale_dtype|not implemented for DT_BFLOAT16|RuntimeError" "${log_path}" | tail -n 6
}

run_case ksplit_fp32 \
  --m 2 --k 512 --n 256 --experts 2 --group-size 128 \
  --scale-dtype fp32 --warmup 0 --iters 1

run_case single_group_fp32 \
  --m 2 --k 512 --n 256 --experts 2 --group-size 512 \
  --scale-dtype fp32 --warmup 0 --iters 1

run_case ksplit_fp16 \
  --m 2 --k 512 --n 256 --experts 2 --group-size 128 \
  --scale-dtype fp16 --warmup 0 --iters 1

run_case zero_token_expert \
  --m 2 --k 512 --n 256 --experts 4 --group-size 128 \
  --scale-dtype fp32 --warmup 0 --iters 1

run_case multi_row \
  --m 32 --k 512 --n 256 --experts 4 --group-size 128 \
  --scale-dtype fp32 --warmup 0 --iters 1

run_case tiled_m256_multi_expert \
  --m 256 --k 512 --n 256 --experts 4 --group-size 128 \
  --scale-dtype fp32 --warmup 0 --iters 1

run_case tail_tile_m257 \
  --m 257 --k 512 --n 256 --experts 4 --group-size 128 \
  --scale-dtype fp32 --warmup 0 --iters 1

run_case multi_n_block \
  --m 2 --k 512 --n 1024 --experts 1 --group-size 128 \
  --scale-dtype fp32 --warmup 0 --iters 1

run_case group64_fp16 \
  --m 16 --k 512 --n 256 --experts 4 --group-size 64 \
  --scale-dtype fp16 --warmup 0 --iters 1

run_expected_bf16_reject

echo "[ST] all cases passed; logs: ${log_dir}"
