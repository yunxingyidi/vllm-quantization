#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
op_name="grouped_matmul_swiglu_quant_weight_nz_tensor_list"
source_op_dir="${repo_root}/custom_op/${op_name}"
build_root="${GROUP_QUANT_BUILD_ROOT:-${repo_root}/custom_op/build_root/csrc}"
build_op_dir="${build_root}/gmm/${op_name}"
log_path="${1:-/tmp/gmm_build.log}"
group_quant_stage="${GROUP_QUANT_STAGE:-8}"
kernel_header="${build_op_dir}/op_kernel/${op_name}.h"
tiling_cpp="${build_op_dir}/op_host/${op_name}_tiling.cpp"

cd "${repo_root}"
if [[ -f "${repo_root}/env_vllm_ascend.sh" ]]; then
  source "${repo_root}/env_vllm_ascend.sh" || exit $?
fi
set -euo pipefail
source "${repo_root}/custom_op/scripts/groupquant_soc.sh"
soc="$(groupquant_build_soc)"
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-8}"

bash "${repo_root}/custom_op/scripts/prepare_vllm_ascend_csrc.sh" \
  "${repo_root}" "${build_root}"

rsync -a --delete "${source_op_dir}/" "${build_op_dir}/"
if [[ "${group_quant_stage}" != "8" ]]; then
  python - "${kernel_header}" "${group_quant_stage}" <<'PY'
import pathlib
import re
import sys

path = pathlib.Path(sys.argv[1])
stage = sys.argv[2]
text = path.read_text()
text = re.sub(r"#define GROUP_QUANT_STAGE\s+\d+", f"#define GROUP_QUANT_STAGE {stage}", text, count=1)
path.write_text(text)
PY
fi
if [[ "${group_quant_stage}" -eq 14 || "${group_quant_stage}" -eq 15 || "${group_quant_stage}" -eq 16 ||
      "${group_quant_stage}" -eq 17 || "${group_quant_stage}" -eq 18 || "${group_quant_stage}" -eq 19 ||
      "${group_quant_stage}" -eq 20 || "${group_quant_stage}" -eq 21 || "${group_quant_stage}" -eq 22 ||
      "${group_quant_stage}" -eq 23 || "${group_quant_stage}" -eq 24 || "${group_quant_stage}" -eq 25 ||
      "${group_quant_stage}" -eq 26 || "${group_quant_stage}" -eq 27 || "${group_quant_stage}" -eq 28 ||
      "${group_quant_stage}" -eq 29 || "${group_quant_stage}" -eq 30 || "${group_quant_stage}" -eq 31 ||
      "${group_quant_stage}" -eq 32 || "${group_quant_stage}" -eq 33 || "${group_quant_stage}" -eq 34 ||
      "${group_quant_stage}" -eq 35 || "${group_quant_stage}" -eq 36 || "${group_quant_stage}" -eq 37 ||
      "${group_quant_stage}" -eq 38 || "${group_quant_stage}" -eq 39 || "${group_quant_stage}" -eq 40 ||
      "${group_quant_stage}" -eq 41 || "${group_quant_stage}" -eq 42 || "${group_quant_stage}" -eq 43 ||
      "${group_quant_stage}" -eq 44 || "${group_quant_stage}" -eq 45 || "${group_quant_stage}" -eq 46 ||
      "${group_quant_stage}" -eq 47 || "${group_quant_stage}" -eq 48 || "${group_quant_stage}" -eq 49 ||
      "${group_quant_stage}" -eq 50 || "${group_quant_stage}" -eq 51 || "${group_quant_stage}" -eq 52 ||
      "${group_quant_stage}" -eq 53 || "${group_quant_stage}" -eq 54 || "${group_quant_stage}" -eq 55 ||
      "${group_quant_stage}" -eq 56 ]]; then
  python - "${tiling_cpp}" <<'PY'
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
text = path.read_text()
text = text.replace(
    "const int64_t matmulBaseN = enableGroupQuant ? 32 : compileInfoPtr->baseN_;",
    "const int64_t matmulBaseN = compileInfoPtr->baseN_;",
)
path.write_text(text)
PY
fi
cd "${build_root}"
rm -rf build prepare_build output build_out
bash build.sh --pkg --ops="${op_name}" --soc="${soc}" \
  -j"${CMAKE_BUILD_PARALLEL_LEVEL}" > "${log_path}" 2>&1
"${repo_root}/custom_op/scripts/install_groupquant_gmm_swiglu.sh"

echo "build log: ${log_path}"
echo "GROUP_QUANT_STAGE=${group_quant_stage}"
