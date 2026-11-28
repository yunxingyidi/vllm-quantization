#!/usr/bin/env bash
set -euo pipefail

repo_root="$1"
build_root="$2"
vllm_ascend_csrc="${VLLM_ASCEND_CSRC:-${repo_root}/../vllm-ascend/csrc}"

if [[ ! -f "${vllm_ascend_csrc}/build.sh" ]]; then
    echo "Missing vllm-ascend csrc build tree: ${vllm_ascend_csrc}" >&2
    echo "Set VLLM_ASCEND_CSRC to a vllm-ascend/csrc compatible with the installed CANN." >&2
    exit 1
fi

patch_source="${vllm_ascend_csrc}/cmake/third_party/build/modules/patch"
if [[ ! -f "${patch_source}/protobuf_25.1_change_version.patch" ]]; then
    echo "VLLM_ASCEND_CSRC requires the operator packaging tree with protobuf patches: ${patch_source}" >&2
    exit 1
fi

if [[ ! -f "${build_root}/build.sh" ]]; then
    mkdir -p "${build_root}"
    # Leading slashes limit these exclusions to generated directories at the
    # csrc root.  The old unanchored `--exclude build` also removed the source
    # directory cmake/third_party/build, including required protobuf patches.
    rsync -a \
        --exclude '/build/' \
        --exclude '/build_out/' \
        --exclude '/build_cache/' \
        --exclude '/output/' \
        --exclude '/prepare_build/' \
        "${vllm_ascend_csrc}/" "${build_root}/"
fi

# Repair build roots created by the old exclusion rule without recopying all
# source files or overwriting the custom operators staged below gmm/.
patch_destination="${build_root}/cmake/third_party/build/modules/patch"
mkdir -p "${patch_destination}"
rsync -a "${patch_source}/" "${patch_destination}/"

# vllm-ascend main currently names ASCEND950, while CANN 8.5.1 does not expose
# that enum. Both 910B and 910C use the non-RegBase path. Apply the
# compatibility change only to the disposable build tree.
tiling_util="${build_root}/common/src/tiling_base/tiling_util.cpp"
if grep -q 'SocVersion::ASCEND950' "${tiling_util}"; then
    patch -d "${build_root}" -p1 < \
        "${repo_root}/custom_op/patches/cann_8_5_no_ascend950.patch"
fi
