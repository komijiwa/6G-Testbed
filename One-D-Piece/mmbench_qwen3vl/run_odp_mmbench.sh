#!/bin/bash
# MMBench + Qwen3-VL evaluation driver for the One-D-Piece ODP-UDP link
# (eval-only; run after the UDP pipeline has produced a reconstructed folder,
# or directly against the baseline MMBench images).
#
# Folder relations, all resolved relative to this script (HERE):
#   HERE                                      One-D-Piece/mmbench_qwen3vl
#   ${HERE}/mmbench_eval.py                   eval script (same folder)
#   ${HERE}/../MMBench_DEV_EN                 default data folder (images + manifest.jsonl)
#   ${HERE}/../Qwen3-VL-30B-A3B-Instruct      model weights (not in git; fetch_odp_assets.sh --weights)
#
# Reconstruction is no longer part of this driver; feed it ready-made data
# folders instead (each holding images + manifest.jsonl).
#
#   # 1) score the default data folder as baseline
#   bash run_odp_mmbench.sh --baseline-only
#
#   # 2) score a One-D-Piece reconstructed folder (e.g. produced by
#   #    run_odp_mmbench_pipeline.py / server/odp_image_udp_server.py)
#   bash run_odp_mmbench.sh --data-dir /path/to/folder --tag my_cond
#
#   # 3) print the result table
#   bash run_odp_mmbench.sh --summary
#
#   # smoke test first:  bash run_odp_mmbench.sh --baseline-only --limit 20
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ONE_D_PIECE_DIR="$(dirname "${HERE}")"            # .../One-D-Piece

MODEL_PATH="${MODEL_PATH:-${ONE_D_PIECE_DIR}/Qwen3-VL-30B-A3B-Instruct}"
DEFAULT_DATA_DIR="${DEFAULT_DATA_DIR:-${ONE_D_PIECE_DIR}/MMBench_DEV_EN}"
EVAL_SCRIPT="${HERE}/mmbench_eval.py"
RESULT_ROOT="${RESULT_ROOT:-${HERE}/results}"
BASELINE_TAG="original256"

# ---- defaults, all overridable by flags ---------------------------------
DATA_DIR=""; TAG=""; LIMIT=""; TP=1; GPU_UTIL=0.9; SEED=42
MODE="single"; SKIP_BASELINE=0

usage() { sed -n '2,23p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

while [[ $# -gt 0 ]]; do  # $#: number of unprocessed arguments left.
  case "$1" in  # $1: the first unprocessed argument.
    --data-dir)      DATA_DIR="$2";   shift 2 ;;
    --tag)           TAG="$2";        shift 2 ;;
    --limit)         LIMIT="$2";      shift 2 ;;
    --model)         MODEL_PATH="$2"; shift 2 ;;
    --seed)          SEED="$2";       shift 2 ;;
    --tp)            TP="$2";         shift 2 ;;
    --gpu-util)      GPU_UTIL="$2";   shift 2 ;;
    --baseline-only) MODE="baseline"; shift ;;
    --summary)       MODE="summary";  shift ;;
    --skip-baseline) SKIP_BASELINE=1; shift ;;
    -h|--help)       usage 0 ;;
    *) echo "[FATAL] unknown option: $1" >&2; usage 1 ;;
  esac
done

for required in "${MODEL_PATH}" "${EVAL_SCRIPT}"; do
  [[ -e "${required}" ]] || { echo "[FATAL] missing: ${required}" >&2; exit 1; }
done

# Resolve the interpreter once: some hosts only ship 'python3'.
if [[ -z "${PYBIN:-}" ]]; then
  if command -v python >/dev/null 2>&1; then PYBIN=python
  elif command -v python3 >/dev/null 2>&1; then PYBIN=python3
  else echo "[FATAL] no python interpreter on PATH" >&2; exit 1; fi
fi

mkdir -p "${RESULT_ROOT}"

# vLLM V1 forks its EngineCore by default; that crashes if a CUDA context
# already exists in the launching process. Spawn is the safe method.
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

# Score a data folder with Qwen3-VL. $1=data_dir $2=tag
run_eval () {
  local data_dir="$1" tag="$2"
  local out="${RESULT_ROOT}/${tag}"
  if [[ -f "${out}/summary.json" ]]; then
    echo "[SKIP] eval ${tag} (summary.json exists)"
    return
  fi
  [[ -d "${data_dir}" ]] || { echo "[FATAL] no such folder: ${data_dir}" >&2; exit 1; }
  local extra=()
  [[ -n "${LIMIT}" ]] && extra+=(--limit "${LIMIT}")
  local base="${RESULT_ROOT}/${BASELINE_TAG}"
  if [[ "${tag}" != "${BASELINE_TAG}" && -f "${base}/predictions.jsonl" ]]; then
    extra+=(--baseline "${base}")
  fi
  echo "[EVAL] ${tag} <- ${data_dir}"
  "${PYBIN}" "${EVAL_SCRIPT}" \
    --data-dir "${data_dir}" --output-dir "${out}" --model "${MODEL_PATH}" \
    --tensor-parallel-size "${TP}" --gpu-memory-utilization "${GPU_UTIL}" \
    --seed "${SEED}" "${extra[@]}"
}

ensure_baseline () {
  [[ "${SKIP_BASELINE}" -eq 1 ]] && return 0
  if [[ -f "${RESULT_ROOT}/${BASELINE_TAG}/summary.json" ]]; then
    echo "[SKIP] baseline (already scored)"
    return 0
  fi
  run_eval "${DEFAULT_DATA_DIR}" "${BASELINE_TAG}"
}

print_summary () {
  echo
  printf '%-18s %-9s %-8s %s\n' CONDITION RAW_ACC INVALID FILTERED_ACC
  for f in "${RESULT_ROOT}"/*/summary.json; do
    [[ -e "${f}" ]] || continue
    ${PYBIN} - "${f}" <<'PY'
import json, sys, pathlib
p = pathlib.Path(sys.argv[1]); d = json.loads(p.read_text(encoding="utf-8"))
f = d.get("filtered_accuracy")
print("%-18s %-9.4f %-8d %s" % (
    p.parent.name, d["raw_accuracy"], d["invalid_predictions"],
    "%.4f (n=%d)" % (f["accuracy"], f["baseline_correct_total"]) if f else "-"))
PY
  done
}

# ---- dispatch ------------------------------------------------------------
case "${MODE}" in
  summary)
    print_summary
    ;;
  baseline)
    ensure_baseline
    print_summary
    ;;
  single)
    if [[ -n "${DATA_DIR}" ]]; then
      run_eval "${DATA_DIR}" "${TAG:-$(basename "${DATA_DIR}")}"
    else
      echo "[FATAL] need --data-dir, --baseline-only or --summary" >&2
      usage 1
    fi
    print_summary
    ;;
esac
