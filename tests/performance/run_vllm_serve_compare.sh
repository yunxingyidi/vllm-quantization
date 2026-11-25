#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
source "$repo_root/env_vllm_ascend.sh"
set -euo pipefail
python tests/performance/run_vllm_serve_compare.py
