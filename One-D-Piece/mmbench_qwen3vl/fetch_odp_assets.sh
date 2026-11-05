#!/usr/bin/env bash
# =============================================================================
# Fetch the non-git assets for the One-D-Piece <-> Qwen3-VL MMBench link:
# data presence check + (opt-in) model weight download.
#
# Python dependency installation is NOT handled here anymore -- just do:
#   python -m venv ~/envs/qwen3vl && source ~/envs/qwen3vl/bin/activate
#   pip install -r requirements.txt
#
# Usage:
#   bash fetch_odp_assets.sh                  # just verify MMBench images
#   bash fetch_odp_assets.sh --weights        # also download ODP + Qwen3-VL weights
#   bash fetch_odp_assets.sh --no-data-check --weights
#
# What it does (each step is skippable):
#   1. verify the MMBench images are present (manifest.jsonl is in git, but the
#      ~96 MB jpgs are NOT -- each developer fetches them separately, see README);
#   2. (opt-in, --weights) download the model weights that are NOT in git:
#        - ODP L-256 tokenizer  (~2.4 GB, public HuggingFace) -> aicodec model package
#        - Qwen3-VL-30B         (~60 GB, set QWEN_REPO / place manually) -> One-D-Piece/
# =============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ONE_D_PIECE_DIR="$(cd "${HERE}/.." && pwd)"            # .../One-D-Piece
REPO_ROOT="$(cd "${ONE_D_PIECE_DIR}/.." && pwd)"       # repo root

DO_WEIGHTS=0
DO_DATA_CHECK=1
for arg in "$@"; do
  case "$arg" in
    --weights) DO_WEIGHTS=1 ;;
    --no-data-check) DO_DATA_CHECK=0 ;;
    -h|--help)
      grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
  esac
done

echo "[fetch] repo root       : ${REPO_ROOT}"
echo "[fetch] One-D-Piece dir : ${ONE_D_PIECE_DIR}"

# ---------------------------------------------------------------------------
# 1) MMBench images: NOT in git / LFS. manifest.jsonl is versioned, but the
#    ~96 MB jpgs must be provided separately (see One-D-Piece/README.md).
#    Verify they are already present rather than downloading them.
# ---------------------------------------------------------------------------
DATA_DIR="${ONE_D_PIECE_DIR}/MMBench_DEV_EN"
MANIFEST="${DATA_DIR}/manifest.jsonl"
if [[ "${DO_DATA_CHECK}" == "1" ]]; then
  if [[ ! -f "${MANIFEST}" ]]; then
    echo "[FATAL] manifest missing: ${MANIFEST}" >&2
    echo "        It should come from git; run this from a fresh clone." >&2
    exit 1
  fi
  N_MANIFEST="$(grep -c . "${MANIFEST}" || true)"
  N_JPG="$(find "${DATA_DIR}" -maxdepth 1 -type f -name '*.jpg' 2>/dev/null | wc -l | tr -d ' ')"
  echo "[fetch] manifest rows: ${N_MANIFEST}, jpgs present: ${N_JPG}"
  if [[ "${N_JPG}" -lt "${N_MANIFEST}" ]]; then
    cat >&2 <<EOF
[FATAL] MMBench images are not in git or Git LFS (LFS is disabled on this server).
        Expected ${N_MANIFEST} jpgs in ${DATA_DIR}, found ${N_JPG}.
        Copy the MMBench_DEV_EN image set from the team shared drive / an
        existing checkout, or regenerate from the official OpenCompass
        mmbench_dev_en release. Filenames must match the manifest "image" field.
        See One-D-Piece/README.md ("Data and weights").
        Re-run after placing the images (or pass --no-data-check to skip).
EOF
    exit 1
  fi
  echo "[fetch] MMBench dataset present (${N_JPG} images)."
fi

# ---------------------------------------------------------------------------
# 2) Model weights (NOT tracked in git) -- opt in with --weights
#    ODP tokenizer weight lives in the aicodec model package (bridge.py's
#    sibling): aicodec/aicodec/models/one_d_piece/checkpoints/pytorch_model.bin
#    Qwen3-VL weights live under One-D-Piece/ (see README "Data and weights").
# ---------------------------------------------------------------------------
if [[ "${DO_WEIGHTS}" == "1" ]]; then
  ODP_CKPT_DIR="${REPO_ROOT}/aicodec/aicodec/models/one_d_piece/checkpoints"
  ODP_CKPT="${ODP_CKPT_DIR}/pytorch_model.bin"
  if [[ -s "${ODP_CKPT}" ]]; then
    echo "[fetch] ODP checkpoint already present: ${ODP_CKPT}"
  else
    echo "[fetch] downloading ODP L-256 tokenizer from HuggingFace..."
    mkdir -p "${ODP_CKPT_DIR}"
    # Public repo: turing-motors/One-D-Piece-L-256. The bridge expects the
    # state dict at <ODP_CHECKPOINT> (see README / ODP_CHECKPOINT env var).
    dl="$(ODP_HF_REPO="${ODP_HF_REPO:-turing-motors/One-D-Piece-L-256}" \
          ODP_HF_FILE="${ODP_HF_FILE:-pytorch_model.bin}" \
          python - <<'PY'
import os
from huggingface_hub import hf_hub_download
print(hf_hub_download(
    repo_id=os.environ["ODP_HF_REPO"], filename=os.environ["ODP_HF_FILE"]))
PY
)"
    cp "${dl}" "${ODP_CKPT}"
  fi

  QWEN_DIR="${ONE_D_PIECE_DIR}/Qwen3-VL-30B-A3B-Instruct"
  if [[ -d "${QWEN_DIR}" && -f "${QWEN_DIR}/config.json" ]]; then
    echo "[fetch] Qwen3-VL weights already present: ${QWEN_DIR}"
  else
    QWEN_REPO="${QWEN_REPO:-}"
    if [[ -z "${QWEN_REPO}" ]]; then
      echo "[WARN] QWEN_REPO not set; cannot auto-download the ~60GB Qwen3-VL model."
      echo "       Place it at: ${QWEN_DIR}"
      echo "       or re-run with QWEN_REPO=<hf-repo-id> bash $0 --weights"
    else
      echo "[fetch] snapshot_download ${QWEN_REPO} -> ${QWEN_DIR}"
      python - "${QWEN_REPO}" "${QWEN_DIR}" <<'PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download(repo_id=sys.argv[1], local_dir=sys.argv[2])
PY
    fi
  fi
fi

cat <<EOF

[done] assets checked/fetched.

Next steps:
  source ~/envs/qwen3vl/bin/activate   # or your venv/conda env
  pip install -r "${HERE}/requirements.txt"
  bash "${HERE}/run_odp_mmbench.sh" --baseline-only --limit 8   # smoke test
EOF
