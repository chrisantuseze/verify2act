#!/usr/bin/env bash
# label_and_merge.sh — Compute reachability labels for pre-collected worker
# datasets and merge them into a single dataset.
#
# Targets the nut_assembly_workers_6 worker dirs produced by a prior
# parallel_collect.sh run that was stopped before Phase 2 / Phase 3.
#
# Pipeline:
#   Phase 1 (parallel): compute_labels.py × WORKERS
#   Phase 2 (serial):   merge_datasets.py
#
# Usage:
#   bash label_and_merge.sh [OPTIONS]
# Example:
#   bash label_and_merge.sh --worker-base-dir dataset/nut_assembly_workers --workers 2 
#
# Options (all optional — defaults shown):
#   --worker-base-dir  PATH  Directory containing worker_<i> subdirs  (default: dataset/nut_assembly_workers_6)
#   --workers          N     Number of worker dirs to process          (default: 6)
#   --output-dir       PATH  Final merged dataset root                 (default: dataset/nut_assembly_6)
#   --env              STR   Environment name                          (default: ClutteredNutAssembly)
#   --label-horizon    N     Expert rollout horizon for labelling      (default: 300)
#   --num-round        N     Round nuts per scene                      (default: 2)
#   --num-square       N     Square nuts per scene                     (default: 2)
#   --stacking-prob    F     Initial stacking probability              (default: 0.6)
#   --nut-type-mode    STR   roundnut | squarenut | random | alternate (default: random)
#   --base-seed        N     Seed for worker 0; workers get base+i     (default: 42)
#   --no-labels            Skip Phase 1 (assume labels already computed)
#   --no-merge             Skip Phase 2 (keep worker dirs, no merge)
#   --keep-workers         Keep worker dirs after merging

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Defaults ─────────────────────────────────────────────────────────────────
WORKER_BASE_DIR="dataset/nut_assembly_workers"
WORKERS=6
OUTPUT_DIR="dataset/nut_assembly_workers_consolidated"
ENV="ClutteredNutAssembly"
LABEL_HORIZON=300
NUM_ROUND=2
NUM_SQUARE=2
STACKING_PROB=0.6
NUT_TYPE_MODE="random"
BASE_SEED=44
NO_LABELS=false
NO_MERGE=false
KEEP_WORKERS=false

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --worker-base-dir) WORKER_BASE_DIR="$2"; shift 2 ;;
        --workers)         WORKERS="$2";         shift 2 ;;
        --output-dir)      OUTPUT_DIR="$2";       shift 2 ;;
        --env)             ENV="$2";              shift 2 ;;
        --label-horizon)   LABEL_HORIZON="$2";    shift 2 ;;
        --num-round)       NUM_ROUND="$2";        shift 2 ;;
        --num-square)      NUM_SQUARE="$2";       shift 2 ;;
        --stacking-prob)   STACKING_PROB="$2";    shift 2 ;;
        --nut-type-mode)   NUT_TYPE_MODE="$2";    shift 2 ;;
        --base-seed)       BASE_SEED="$2";        shift 2 ;;
        --no-labels)       NO_LABELS=true;        shift   ;;
        --no-merge)        NO_MERGE=true;         shift   ;;
        --keep-workers)    KEEP_WORKERS=true;     shift   ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

# Resolve worker base dir relative to the script location if not absolute
if [[ "${WORKER_BASE_DIR}" != /* ]]; then
    WORKER_BASE_DIR="${SCRIPT_DIR}/${WORKER_BASE_DIR}"
fi
if [[ "${OUTPUT_DIR}" != /* ]]; then
    OUTPUT_DIR="${SCRIPT_DIR}/${OUTPUT_DIR}"
fi

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export OPENBLAS_NUM_THREADS=2
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# ── Banner ────────────────────────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════════════"
echo "  Label + Merge pre-collected worker datasets"
echo "  Worker base dir: ${WORKER_BASE_DIR}"
echo "  Workers        : ${WORKERS}"
echo "  Environment    : ${ENV}"
echo "  Label horizon  : ${LABEL_HORIZON}"
echo "  Nuts (R/S)     : ${NUM_ROUND} / ${NUM_SQUARE}"
echo "  Stacking prob  : ${STACKING_PROB}"
echo "  Nut type mode  : ${NUT_TYPE_MODE}"
echo "  Output dir     : ${OUTPUT_DIR}"
echo "  Compute labels : $( [[ "${NO_LABELS}" == "true" ]] && echo 'no (--no-labels)' || echo 'yes' )"
echo "══════════════════════════════════════════════════════════"
echo ""

# ── Phase 1: label computation (parallel) ────────────────────────────────────
if [[ "${NO_LABELS}" == "true" ]]; then
    echo "[label_and_merge] --no-labels set; skipping label computation."
else
    echo "[label_and_merge] Phase 1: computing reachability labels in parallel..."
    declare -a LABEL_PIDS
    declare -a LABEL_LOGS

    for (( i=0; i<WORKERS; i++ )); do
        WORKER_DIR="${WORKER_BASE_DIR}/worker_${i}"
        LABEL_LOG="${WORKER_BASE_DIR}/worker_${i}_labels.log"
        LABEL_LOGS+=("${LABEL_LOG}")
        SEED=$(( BASE_SEED + i ))

        if [[ ! -d "${WORKER_DIR}/episodes" ]]; then
            echo "[label_and_merge] Worker ${i}: no episodes/ dir — skipping."
            LABEL_PIDS+=(-1)
            continue
        fi

        # Skip if labels already exist and are non-empty
        if [[ -s "${WORKER_DIR}/labels.jsonl" ]]; then
            echo "[label_and_merge] Worker ${i}: labels.jsonl already exists — skipping."
            LABEL_PIDS+=(-1)
            continue
        fi

        echo "[label_and_merge] Labelling worker ${i} (seed=${SEED}) -> ${WORKER_DIR}"

        python "${SCRIPT_DIR}/compute_labels.py" \
            --dataset-dir    "${WORKER_DIR}" \
            --env            "${ENV}" \
            --horizon        "${LABEL_HORIZON}" \
            --num-round      "${NUM_ROUND}" \
            --num-square     "${NUM_SQUARE}" \
            --initial-stacking-prob "${STACKING_PROB}" \
            --nut-type-mode  "${NUT_TYPE_MODE}" \
            --seed           "${SEED}" \
            > "${LABEL_LOG}" 2>&1 &

        LABEL_PIDS+=($!)
    done

    echo "[label_and_merge] Waiting for label workers..."
    LABEL_FAILED=0
    for (( i=0; i<WORKERS; i++ )); do
        PID="${LABEL_PIDS[$i]}"
        [[ "${PID}" == "-1" ]] && continue
        if wait "${PID}"; then
            echo "[label_and_merge] Label worker ${i} (PID ${PID}) finished OK"
        else
            EXIT_CODE=$?
            echo "[label_and_merge] Label worker ${i} (PID ${PID}) FAILED (exit ${EXIT_CODE})"
            echo "                  See log: ${LABEL_LOGS[$i]}"
            LABEL_FAILED=$(( LABEL_FAILED + 1 ))
        fi
    done

    if (( LABEL_FAILED > 0 )); then
        echo "[label_and_merge] WARNING: ${LABEL_FAILED}/${WORKERS} label workers failed."
    else
        echo "[label_and_merge] All label workers completed successfully."
    fi
fi

# ── Phase 2: merge ────────────────────────────────────────────────────────────
if [[ "${NO_MERGE}" == "true" ]]; then
    echo "[label_and_merge] --no-merge set; skipping merge."
    exit 0
fi

VALID_DIRS=()
for (( i=0; i<WORKERS; i++ )); do
    WDIR="${WORKER_BASE_DIR}/worker_${i}"
    if [[ -d "${WDIR}/episodes" ]]; then
        VALID_DIRS+=("${WDIR}")
    else
        echo "[label_and_merge] Skipping empty worker dir (no episodes/): ${WDIR}"
    fi
done

if (( ${#VALID_DIRS[@]} == 0 )); then
    echo "[label_and_merge] ERROR: No worker produced any episodes. Nothing to merge." >&2
    exit 1
fi

echo ""
echo "[label_and_merge] Phase 2: merging ${#VALID_DIRS[@]} worker dataset(s) -> ${OUTPUT_DIR}"

if python "${SCRIPT_DIR}/merge_datasets.py" \
    --source-dirs "${VALID_DIRS[@]}" \
    --output-dir  "${OUTPUT_DIR}" \
    --overwrite; then

    echo "[label_and_merge] Merge complete. Final dataset: ${OUTPUT_DIR}"

    if [[ "${KEEP_WORKERS}" == "false" ]]; then
        echo "[label_and_merge] Removing worker dirs (use --keep-workers to retain them)."
        rm -rf "${WORKER_BASE_DIR}"
    else
        echo "[label_and_merge] Worker dirs kept in ${WORKER_BASE_DIR}/"
    fi
else
    echo "[label_and_merge] ERROR: Merge failed. Worker dirs preserved in ${WORKER_BASE_DIR}/" >&2
    exit 1
fi

echo ""
echo "Done."
