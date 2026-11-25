#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec bash "${repo_root}/custom_op/scripts/build_groupquant_moe_fused.sh" \
    "${1:-/tmp/groupquant_moe_fused_build.log}"
