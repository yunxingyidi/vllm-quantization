#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
PROJECT_DIR="${REPO_ROOT}/custom_op/gmm1_vecin_fullk"
CANN_PATH=${ASCEND_HOME_PATH:-/home/zhangtairan/Ascend/cann-8.5.1}
VENDOR_DIR=${GMM1_VENDOR_DIR:-${CANN_PATH}/opp/vendors/custom_transformer}
OPAPI_DIR=${GMM1_OPAPI_DIR:-${VENDOR_DIR}/op_api/lib}
OPP_PATH=${GMM1_OPP_PATH:-${VENDOR_DIR}}
BUILD_DIR="${SCRIPT_DIR}/build"
OUT_BIN="${BUILD_DIR}/test_aclnn_gmm1_vecin_fullk"
mkdir -p "${BUILD_DIR}"

g++ -std=c++17 -O2 "${SCRIPT_DIR}/test_aclnn_gmm1_vecin_fullk.cpp" \
    -I"${CANN_PATH}/aarch64-linux/include" \
    -I"${PROJECT_DIR}/op_host/op_api" \
    -L"${OPAPI_DIR}" -L"${CANN_PATH}/aarch64-linux/lib64" \
    -Wl,-rpath,"${OPAPI_DIR}:${CANN_PATH}/aarch64-linux/lib64" \
    -lcust_opapi -lascendcl -lnnopbase -o "${OUT_BIN}"

export ASCEND_CUSTOM_OPP_PATH="${OPP_PATH}:${ASCEND_CUSTOM_OPP_PATH:-}"
# Standalone ACLNN can otherwise register a same-named vendor from the SDK
# before ASCEND_CUSTOM_OPP_PATH is scanned. Load the selected package's host
# libraries before ACL initialization, matching the package used for op-api.
PROTO_LIB="${VENDOR_DIR}/op_proto/lib/linux/aarch64/libcust_opsproto_rt2.0.so"
TILING_LIB="${VENDOR_DIR}/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64/libcust_opmaster_rt2.0.so"
for host_lib in "${PROTO_LIB}" "${TILING_LIB}"; do
    if [[ ! -f "${host_lib}" ]]; then
        echo "Missing selected vendor host library: ${host_lib}" >&2
        exit 1
    fi
done
test_preload="${PROTO_LIB}:${TILING_LIB}${LD_PRELOAD:+:${LD_PRELOAD}}"
for test_case in \
    single-expert-full-tile \
    single-expert-m-tail \
    multi-expert-boundary \
    random-nz-precision \
    tp2-n-tail \
    tp4-n-tail \
    tp8-reduce-tail \
    tp1-kgroup32-random \
    tp2-kgroup32-random \
    tp4-kgroup32-random \
    tp8-kgroup32-random \
    fp16-fixpipe-overflow; do
    LD_PRELOAD="${test_preload}" "${OUT_BIN}" "${test_case}"
done

if [[ "${GMM1_RUN_BENCHMARK:-0}" == "1" ]]; then
    LD_PRELOAD="${test_preload}" "${OUT_BIN}" benchmark-full-tile
    LD_PRELOAD="${test_preload}" "${OUT_BIN}" benchmark-tp2-kgroup32
    LD_PRELOAD="${test_preload}" "${OUT_BIN}" benchmark-tp8-kgroup32
fi
