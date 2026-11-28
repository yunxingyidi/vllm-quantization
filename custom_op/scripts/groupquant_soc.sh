#!/usr/bin/env bash

# The build target is the SoC family name used by the CANN operator builder.
groupquant_build_soc() {
    local soc="${GROUP_QUANT_SOC:-}"
    if [[ -z "${soc}" ]]; then
        soc="$(python - <<'PY'
import torch_npu

version = torch_npu.npu.get_soc_version()
if 220 <= version <= 225:
    print("ascend910b")
elif 250 <= version <= 255:
    print("ascend910_93")
else:
    raise SystemExit(f"Unsupported Ascend SoC version {version}. Set GROUP_QUANT_SOC for cross-compilation.")
PY
)" || return 1
    fi
    case "${soc}" in
        ascend910b|ascend910_93) printf '%s\n' "${soc}" ;;
        *) echo "Unsupported GROUP_QUANT_SOC: ${soc}" >&2; return 1 ;;
    esac
}
