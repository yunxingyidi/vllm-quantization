#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
build_root="${GROUP_QUANT_BUILD_ROOT:-${repo_root}/custom_op/build_root/csrc}"
gmm1_op="grouped_matmul_swiglu_quant_weight_nz_tensor_list"
gmm1_vecin_op="gmm1_vecin_fullk"
gmm2_op="quant_matmul_expert_k_grouped"
log_path="${1:-/tmp/groupquant_moe_fused_build.log}"

cd "${repo_root}"
source "${repo_root}/env_vllm_ascend.sh"
set -euo pipefail

bash "${repo_root}/custom_op/scripts/prepare_vllm_ascend_csrc.sh" \
    "${repo_root}" "${build_root}"

rsync -a --delete "${repo_root}/custom_op/${gmm1_op}/" \
    "${build_root}/gmm/${gmm1_op}/"
rsync -a --delete "${repo_root}/custom_op/${gmm1_vecin_op}/" \
    "${build_root}/gmm/${gmm1_vecin_op}/"
rsync -a --delete "${repo_root}/custom_op/${gmm2_op}/" \
    "${build_root}/gmm/${gmm2_op}/"

cd "${build_root}"
rm -rf build prepare_build output build_out
bash build.sh --pkg --ops="${gmm1_op},${gmm1_vecin_op},${gmm2_op}" --soc=ascend910b \
    >"${log_path}" 2>&1

bash "${repo_root}/custom_op/scripts/install_groupquant_gmm_swiglu.sh"
cd "${repo_root}/custom_op/${gmm2_op}"
VLLM_ASCEND_CSRC="${VLLM_ASCEND_CSRC:-${repo_root}/../vllm-ascend/csrc}" \
    python setup.py build_ext
cd "${repo_root}/custom_op/${gmm1_vecin_op}"
VLLM_ASCEND_CSRC="${VLLM_ASCEND_CSRC:-${repo_root}/../vllm-ascend/csrc}" \
    python setup.py build_ext
echo "combined build log: ${log_path}"
echo "built operators: ${gmm1_op}, ${gmm1_vecin_op}, ${gmm2_op}"
echo "source ${repo_root}/custom_op/build_root/install/vendors/custom_transformer/bin/set_env.bash"
