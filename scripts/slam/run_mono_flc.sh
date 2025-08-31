#!/usr/bin/env bash
set -euo pipefail

# Run Jetson-SLAM monocular on a recorded FLC session
# Usage:
#   ./run_mono_flc.sh [RUN_DIR]
# Default RUN_DIR: ../../data/flc_run (expects images/*.png and times.txt)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="${1:-${SCRIPT_DIR}/../../data/flc_run}"
IMG_DIR="${RUN_DIR}/images"
TIMES_FILE="${RUN_DIR}/times.txt"
VOCAB="${SCRIPT_DIR}/../../Vocabulary/ORBvoc.txt"
SETTINGS="${SCRIPT_DIR}/../../Examples/Monocular/FLC.yaml"
BIN="${SCRIPT_DIR}/../../Examples/Monocular/mono_euroc"

if [[ ! -x "${BIN}" ]]; then
  echo "ERROR: mono_euroc binary not found at ${BIN}. Build Jetson-SLAM first (run build.sh)." >&2
  exit 1
fi

if [[ ! -f "${SETTINGS}" ]]; then
  echo "ERROR: Settings YAML missing: ${SETTINGS}" >&2
  exit 1
fi

if [[ ! -d "${IMG_DIR}" || ! -f "${TIMES_FILE}" ]]; then
  echo "ERROR: RUN_DIR must contain images/ and times.txt. Given: ${RUN_DIR}" >&2
  exit 1
fi

# Pass nanosecond timestamps directly; mono_euroc converts to seconds internally.
pushd "${SCRIPT_DIR}/../../Examples/Monocular" >/dev/null
"${BIN}" "${VOCAB}" "${SETTINGS}" "${IMG_DIR}" "${TIMES_FILE}"
popd >/dev/null
