#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_FILE="${ROOT_DIR}/sam_vit_b_01ec64.pth"
URL="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"

if [[ -f "${OUT_FILE}" ]]; then
  echo "[OK] 已存在：${OUT_FILE}"
  exit 0
fi

echo "[INFO] 下载 SAM checkpoint（vit_b）..."
echo "       ${URL}"
echo "       -> ${OUT_FILE}"

curl -L "${URL}" -o "${OUT_FILE}"

echo "[OK] 下载完成：${OUT_FILE}"

