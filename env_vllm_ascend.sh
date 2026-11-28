#!/usr/bin/env bash

_env_vllm_ascend_is_sourced() {
    [[ "${BASH_SOURCE[0]}" != "${0}" ]]
}

_env_vllm_ascend_finish() {
    local status="$1"
    if _env_vllm_ascend_is_sourced; then
        return "${status}"
    fi
    exit "${status}"
}

_env_vllm_ascend_require_file() {
    local path="$1"
    local label="$2"
    if [[ ! -f "${path}" ]]; then
        echo "Missing ${label}: ${path}" >&2
        return 1
    fi
}

_env_vllm_ascend_main() {
    local conda_root="${HOME}/miniconda3"
    local default_conda_env=vllm
    local default_cann_home="${HOME}/Ascend/cann-8.5.1"
    local default_atb_home="${HOME}/Ascend/nnal/atb/8.5.1/atb"
    if [[ ! -d "${conda_root}/envs/vllm/conda-meta" && -d "${conda_root}/envs/vllmq/conda-meta" ]]; then
        default_conda_env=vllmq
    fi
    if [[ ! -f "${default_cann_home}/set_env.sh" && -f "${HOME}/Ascend/cann/set_env.sh" ]]; then
        default_cann_home="${HOME}/Ascend/cann"
    fi
    if [[ ! -f "${default_atb_home}/set_env.sh" && -f "${HOME}/Ascend/nnal/atb/set_env.sh" ]]; then
        default_atb_home="${HOME}/Ascend/nnal/atb"
    fi
    local conda_env_name="${CONDA_ENV_NAME:-${default_conda_env}}"
    local cann_home="${CANN_HOME:-${default_cann_home}}"
    local atb_home="${ATB_HOME:-${default_atb_home}}"
    local asdsip_home="${ASDSIP_HOME:-${HOME}/Ascend/nnal/asdsip}"
    local atb_cxx_abi="${ATB_CXX_ABI:-0}"

    _env_vllm_ascend_require_file "${conda_root}/etc/profile.d/conda.sh" "conda init script" || return 1
    _env_vllm_ascend_require_file "${cann_home}/set_env.sh" "CANN env script" || return 1
    _env_vllm_ascend_require_file "${atb_home}/set_env.sh" "ATB env script" || return 1
    _env_vllm_ascend_require_file "${asdsip_home}/set_env.sh" "ASDSIP env script" || return 1

    # Ascend env scripts access unset shell variables internally, so avoid
    # propagating nounset/errexit from the caller into those sourced scripts.
    set +u
    set +e

    source "${conda_root}/etc/profile.d/conda.sh" || return 1
    conda activate "${conda_env_name}" || return 1

    source "${cann_home}/set_env.sh" || return 1
    source "${atb_home}/set_env.sh" "--cxx_abi=${atb_cxx_abi}" || return 1
    source "${asdsip_home}/set_env.sh" || return 1

    # This server is shared by multiple HCCL jobs. Let the OS allocate an
    # isolated host/NPU socket port unless the caller supplied a fixed range.
    export HCCL_HOST_SOCKET_PORT_RANGE="${HCCL_HOST_SOCKET_PORT_RANGE:-auto}"
    export HCCL_NPU_SOCKET_PORT_RANGE="${HCCL_NPU_SOCKET_PORT_RANGE:-auto}"

    # torch 2.9 workers must not inherit an initialized OpenMP pool through
    # fork.  Spawn also matches vLLM-Ascend's supported multiprocess path.
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
    export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
    export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"

    local repo_root
    local custom_opp_env
    repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    custom_opp_env="${repo_root}/custom_op/build_root/install/vendors/custom_transformer/bin/set_env.bash"
    if [[ -f "${custom_opp_env}" ]]; then
        source "${custom_opp_env}" || return 1
        export GROUP_QUANT_CUSTOM_OPAPI_PATH="${repo_root}/custom_op/build_root/install/vendors/custom_transformer/op_api/lib/libcust_opapi.so"
        echo "Custom OPP environment loaded: ${custom_opp_env}"
    fi

    # Stable GroupQuant inference switches.  Defaults can be overridden before
    # sourcing this file; debug-only workspace/profile modes stay disabled.
    export GROUP_QUANT_USE_EXPERT_GMM2_OP="${GROUP_QUANT_USE_EXPERT_GMM2_OP:-1}"
    # This replaces the GMM1 BF16 output plus GMM2 dynamic-quant sequence
    # with a single GMM1 epilogue that emits GMM2-ready INT8+scale. Unsupported
    # shapes retain the generic GMM1 path through the Python dispatch guard.
    export GROUP_QUANT_USE_GMM1_VECIN_FULLK="${GROUP_QUANT_USE_GMM1_VECIN_FULLK:-1}"
    # Fuse GMM1 input dynamic quantization into MoE routing (quant_mode=1).
    # GroupQuant activation contract is per-token.  Fused GMM1 preserves that
    # contract and feeds its native [M] output scale directly into GMM2.
    local vllm_ascend_version
    local fused_gmm1_default=1
    vllm_ascend_version="$(python -c 'from importlib.metadata import version; print(version("vllm-ascend"))' 2>/dev/null || true)"
    # The 0.19.1rc1 wheel preloads a same-named, per-channel-only GMM1 op-api.
    # It cannot consume GroupQuant's [K/group, N] scales.  Keep GMM2 enabled,
    # but use the correct unfused GMM1 fallback unless explicitly overridden.
    if [[ "${vllm_ascend_version}" == "0.19.1rc1" ]]; then
        fused_gmm1_default=0
    fi
    export GROUP_QUANT_USE_FUSED_SWIGLU_OP="${GROUP_QUANT_USE_FUSED_SWIGLU_OP:-${fused_gmm1_default}}"

    # Keep FUSED_MC2 opt-in.  On GroupQuant it uses MC2 dispatch/combine with
    # the custom per-group GMM chain because dispatch_ffn_combine has no
    # per-K-group scale ABI.  Dynamic EPLB is configured through vLLM's
    # additional_config and can coexist with this decomposed path.
    export VLLM_ASCEND_ENABLE_FUSED_MC2="${VLLM_ASCEND_ENABLE_FUSED_MC2:-0}"
    export DYNAMIC_EPLB="${DYNAMIC_EPLB:-false}"

    # GMM1: GroupedMatmul + SwiGLU + dynamic quant.  Tuned on the Qwen3-30B-A3B
    # MoE shapes K=2048, N=1536, E=128 with one full-K group.  Using the
    # full N=1536 projection as a logical task avoids the extra N-direction
    # hand-off imposed by the older 768 split (validated on the PPL prefill
    # workload with identical output PPL).
    export GROUP_QUANT_GMM1_GLOBAL_SCHEDULE="${GROUP_QUANT_GMM1_GLOBAL_SCHEDULE:-1}"
    export GROUP_QUANT_GMM1_CORE_NUM="${GROUP_QUANT_GMM1_CORE_NUM:-24}"
    export GROUP_QUANT_GMM1_SINGLE_N="${GROUP_QUANT_GMM1_SINGLE_N:-1536}"
    export GROUP_QUANT_GMM1_UB_BUDGET_KB="${GROUP_QUANT_GMM1_UB_BUDGET_KB:-96}"
    export GROUP_QUANT_GMM1_PARTIAL_TILE_KB="${GROUP_QUANT_GMM1_PARTIAL_TILE_KB:-64}"
    # Single-K-group NZ epilogue; unsupported widths/groups retain the legacy
    # implementation. Keep independent switches for same-binary A/B tests.
    # NZ epilogue and adaptive M32 are host defaults; overrides are optional.

    # GMM2: expert-aware K-group matmul.  The large routed-token shape has
    # enough M/N tiles to occupy all 20 AICs.  A 64x512 tile reduces scalar
    # scheduling overhead, and four pipeline slots are sufficient to hide the
    # AIC/AIV hand-off without doubling the per-op workspace.  Group size 0
    # means one full TP-local K group (384 for Qwen3-30B-A3B GMM2), removing
    # cross-group GM partials while GMM1 keeps its independent full-K=2048.
    export GROUP_QUANT_GMM2_GROUP_SIZE="${GROUP_QUANT_GMM2_GROUP_SIZE:-0}"
    export GROUP_QUANT_GMM2_CORE_NUM="${GROUP_QUANT_GMM2_CORE_NUM:-20}"
    export GROUP_QUANT_GMM2_PARALL_NUM="${GROUP_QUANT_GMM2_PARALL_NUM:-4}"
    export GROUP_QUANT_GMM2_BASE_M="${GROUP_QUANT_GMM2_BASE_M:-64}"
    export GROUP_QUANT_GMM2_BASE_N="${GROUP_QUANT_GMM2_BASE_N:-512}"
    export GROUP_QUANT_GMM2_BASE_K="${GROUP_QUANT_GMM2_BASE_K:-128}"
    # Logical AIC/AIV hand-off tile.  Keep it equal to baseM until an
    # end-to-end validation promotes a larger two-level M tile.
    export GROUP_QUANT_GMM2_SINGLE_M="${GROUP_QUANT_GMM2_SINGLE_M:-64}"
    export GROUP_QUANT_GMM2_ENABLE_TILE_PIPELINE="${GROUP_QUANT_GMM2_ENABLE_TILE_PIPELINE:-1}"
    # Spread M tiles globally across experts to prevent small-expert work from
    # serializing on a subset of cores.  Set to 0 to use the legacy mapping.
    export GROUP_QUANT_GMM2_EXPERT_M_OUTER="${GROUP_QUANT_GMM2_EXPERT_M_OUTER:-1}"
    # Partition rows across both AIV sub-blocks of the 1:2 GM pipeline.
    # Keep the legacy epilogue available for same-binary correctness/perf A/B.
    # Dual-AIV is the host default; set GROUP_QUANT_GMM2_DUAL_AIV=0 to revert.
    export GROUP_QUANT_PROFILE="${GROUP_QUANT_PROFILE:-0}"
    export GROUP_QUANT_CHECK_NONFINITE="${GROUP_QUANT_CHECK_NONFINITE:-0}"

    echo "Activated conda env: ${conda_env_name}"
    echo "CANN_HOME=${cann_home}"
    echo "ATB_HOME_PATH=${ATB_HOME_PATH:-unset}"
    echo "ASDSIP_HOME_PATH=${ASDSIP_HOME_PATH:-unset}"
    echo "HCCL_HOST_SOCKET_PORT_RANGE=${HCCL_HOST_SOCKET_PORT_RANGE}"
    echo "HCCL_NPU_SOCKET_PORT_RANGE=${HCCL_NPU_SOCKET_PORT_RANGE}"
    echo "Worker multiprocessing=${VLLM_WORKER_MULTIPROC_METHOD}, OMP_NUM_THREADS=${OMP_NUM_THREADS}"
    echo "GroupQuant GMM1: enabled=${GROUP_QUANT_USE_FUSED_SWIGLU_OP}, backend=${GROUP_QUANT_GMM1_BACKEND:-auto}, vecin-fullk=${GROUP_QUANT_USE_GMM1_VECIN_FULLK}, cores=${GROUP_QUANT_GMM1_CORE_NUM}, singleN=${GROUP_QUANT_GMM1_SINGLE_N}, ub=${GROUP_QUANT_GMM1_UB_BUDGET_KB}KiB, partial=${GROUP_QUANT_GMM1_PARTIAL_TILE_KB}KiB, nz-epilogue=${GROUP_QUANT_GMM1_NZ_EPILOGUE:-1}, nz-maxM=${GROUP_QUANT_GMM1_NZ_BASE_M:-32}, nz-accum=${GROUP_QUANT_GMM1_NZ_ACCUM:-1}, group-nz=${GROUP_QUANT_GMM1_GROUP_NZ:-1}, reuse-org=${GROUP_QUANT_GMM1_REUSE_ORG_SHAPE:-1}"
    echo "GroupQuant GMM2: enabled=${GROUP_QUANT_USE_EXPERT_GMM2_OP}, group=${GROUP_QUANT_GMM2_GROUP_SIZE} (0=full local K), cores=${GROUP_QUANT_GMM2_CORE_NUM}, parall=${GROUP_QUANT_GMM2_PARALL_NUM}, base=${GROUP_QUANT_GMM2_BASE_M}x${GROUP_QUANT_GMM2_BASE_N}x${GROUP_QUANT_GMM2_BASE_K}, singleM=${GROUP_QUANT_GMM2_SINGLE_M}, global-m-tiles=${GROUP_QUANT_GMM2_EXPERT_M_OUTER}, dual-aiv=${GROUP_QUANT_GMM2_DUAL_AIV:-1}, nd-input=${GROUP_QUANT_GMM2_ND_INPUT:-1}"

    echo "FUSED_MC2 request=${VLLM_ASCEND_ENABLE_FUSED_MC2}, Dynamic EPLB request=${DYNAMIC_EPLB}"

    python - <<'PY'
import torch_npu.op_plugin.atb._atb_ops  # noqa: F401
print("ATB import check: ok")
PY
}

_env_vllm_ascend_main
_env_vllm_ascend_finish $?
