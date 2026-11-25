#!/usr/bin/env bash
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cmake -S "$here" -B "$here/build" -DASC_DIR="$ASCEND_HOME_PATH/compiler/tikcpp/ascendc_kernel_cmake"
cmake --build "$here/build" -j4
