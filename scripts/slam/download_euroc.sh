#!/usr/bin/env bash
set -euo pipefail

# EuRoC ASL dataset downloader (zip format, non-ROS bag)
# Usage:
#   ./download_euroc.sh MH_01_easy [DEST_DIR]
#   ./download_euroc.sh V1_01_easy [DEST_DIR]
# Default DEST_DIR: ../../data/euroc
#
# Sources:
# - ETH ASL EuRoC dataset index
# - Example: MH_01_easy.zip, V1_01_easy.zip, etc.

SEQ_NAME="${1:-}"
DEST_DIR="${2:-../../data/euroc}"

if [[ -z "${SEQ_NAME}" ]]; then
  echo "ERROR: Provide a sequence name (e.g., MH_01_easy, MH_03_medium, V1_01_easy)." >&2
  exit 1
fi

# Normalize DEST_DIR to absolute path
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_DEST_REL="${SCRIPT_DIR}/../../data/euroc"
if [[ "${DEST_DIR}" == "" || "${DEST_DIR}" == "../../data/euroc" ]]; then
  DEST_DIR="${DEFAULT_DEST_REL}"
fi
mkdir -p "${DEST_DIR}"

# Map sequence to ASL zip URL (zip contains mav0/* with data.csv)
# Construct URL based on known structure
# Machine Hall: MH_0X_{easy,medium,difficult}
# Vicon Rooms: V1_0X_*, V2_0X_*
base_url="http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset"

# Determine subfolder and file
if [[ "${SEQ_NAME}" =~ ^MH_0[1-5]_ ]]; then
  subdir="machine_hall/${SEQ_NAME}"
elif [[ "${SEQ_NAME}" =~ ^V1_0[1-3]_ ]]; then
  subdir="vicon_room1/${SEQ_NAME}"
elif [[ "${SEQ_NAME}" =~ ^V2_0[1-3]_ ]]; then
  subdir="vicon_room2/${SEQ_NAME}"
else
  echo "ERROR: Unsupported sequence name: ${SEQ_NAME}" >&2
  exit 1
fi

zip_url="${base_url}/${subdir}/${SEQ_NAME}.zip"
zip_out="${DEST_DIR}/${SEQ_NAME}.zip"
seq_dir="${DEST_DIR}/${SEQ_NAME}"

echo "Downloading: ${zip_url}"
if command -v curl >/dev/null 2>&1; then
  curl -fL --retry 3 -o "${zip_out}" "${zip_url}"
elif command -v wget >/dev/null 2>&1; then
  wget -O "${zip_out}" "${zip_url}"
else
  echo "ERROR: Need curl or wget installed." >&2
  exit 1
fi

# Verify zip
if ! unzip -tq "${zip_out}" >/dev/null; then
  echo "ERROR: Corrupt zip: ${zip_out}" >&2
  exit 1
fi

# Extract (creates mav0/ structure)
mkdir -p "${seq_dir}"
unzip -q -o "${zip_out}" -d "${seq_dir}"

# If timestamps file exists in dataset, prefer its cam0 timestamps (integer nanoseconds to match filenames)
if [[ -f "${seq_dir}/mav0/cam0/data.csv" ]]; then
  out_times="${seq_dir}/times_cam0.txt"
  awk -F, 'NR>1 {print $1}' "${seq_dir}/mav0/cam0/data.csv" > "${out_times}" || true
fi

## Derive fallback timestamps file name for examples (MH01/V101/V201)
tf_base=""
if [[ "${SEQ_NAME}" =~ ^MH_0([1-5])_ ]]; then
  tf_base="MH0${BASH_REMATCH[1]}"
elif [[ "${SEQ_NAME}" =~ ^V1_0([1-3])_ ]]; then
  tf_base="V10${BASH_REMATCH[1]}"
elif [[ "${SEQ_NAME}" =~ ^V2_0([1-3])_ ]]; then
  tf_base="V20${BASH_REMATCH[1]}"
fi

cat <<MSG

Done.
Sequence: ${SEQ_NAME}
Extracted: ${seq_dir}
Left images:   ${seq_dir}/mav0/cam0/data
Right images:  ${seq_dir}/mav0/cam1/data
Timestamps:    ${seq_dir}/times_cam0.txt (generated) or use Examples/Stereo/EuRoC_TimeStamps/${tf_base}.txt

Example run (Stereo):
  cd ${SCRIPT_DIR}/../../Examples/Stereo
  ./stereo_euroc ../../Vocabulary/ORBvoc.txt ./EuRoC.yaml \
      ${seq_dir}/mav0/cam0/data \
      ${seq_dir}/mav0/cam1/data \
      ${seq_dir}/times_cam0.txt

Reference dataset page: https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets
MSG

