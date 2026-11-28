#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
build_root="${GROUP_QUANT_BUILD_ROOT:-${repo_root}/custom_op/build_root/csrc}"
package_vendor="$(find "${build_root}/build/_CPack_Packages/Linux/External" \
  -maxdepth 4 -type d -path '*/packages/vendors/custom_transformer' \
  -print -quit 2>/dev/null || true)"
install_vendor="${repo_root}/custom_op/build_root/install/vendors/custom_transformer"

if [[ -z "${package_vendor}" || ! -d "${package_vendor}" ]]; then
  echo "package vendor not found: ${package_vendor}" >&2
  exit 1
fi

chmod -R u+w "${install_vendor}" 2>/dev/null || true
rm -rf "${install_vendor}"
mkdir -p "$(dirname "${install_vendor}")"
rsync -a --delete "${package_vendor}/" "${install_vendor}/"

mkdir -p "${install_vendor}/bin"
cat > "${install_vendor}/bin/set_env.bash" <<SETENV
#!/usr/bin/env bash
_custom_transformer_root="${install_vendor}"
export LD_LIBRARY_PATH="\${_custom_transformer_root}/op_api/lib:\${_custom_transformer_root}/op_proto/lib/linux/aarch64\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}"
export ASCEND_CUSTOM_OPP_PATH="\${_custom_transformer_root}\${ASCEND_CUSTOM_OPP_PATH:+:\${ASCEND_CUSTOM_OPP_PATH}}"
export GROUP_QUANT_CUSTOM_OPAPI_PATH="\${_custom_transformer_root}/op_api/lib/libcust_opapi.so"
unset _custom_transformer_root
SETENV
chmod +x "${install_vendor}/bin/set_env.bash"

stat -c '%y %s %n' \
  "${build_root}/build/libcust_opapi.so" \
  "${install_vendor}/op_api/lib/libcust_opapi.so"
cmp -s "${build_root}/build/libcust_opapi.so" "${install_vendor}/op_api/lib/libcust_opapi.so"
echo "custom_transformer installed to ${install_vendor}"
echo "source ${install_vendor}/bin/set_env.bash"
