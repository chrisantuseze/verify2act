#!/usr/bin/env bash
# parallel_collect.sh — Run batch_collect.py across N worker processes in headless
# mode, compute reachability labels per worker in parallel, then merge everything
# into a single dataset.
#
# Pipeline:
#   Phase 1 (parallel): batch_collect.py × WORKERS
#   Phase 2 (parallel): compute_labels.py × WORKERS  (on each worker's own dir)
#   Phase 3 (serial):   merge_datasets.py             (merges episodes + labels)
#
# Usage:
#   bash parallel_collect.sh [OPTIONS]
#
# Options (all optional — defaults shown):
#   --workers          N     Number of parallel processes          (default: 4)
#   --total-episodes   N     Total episodes across ALL workers     (default: 400)
#   --output-dir       PATH  Final merged dataset root             (default: dataset/nut_assembly)
#   --env              STR   Environment name                      (default: ClutteredNutAssembly)
#   --max-timesteps    N     Max steps per episode                 (default: 1000)
#   --policy-mode      STR   expert | noisy                        (default: expert)
#   --noise-sigma      F     Noise sigma (noisy mode only)         (default: 0.05)
#   --transition-mode  STR   keyframe | dense                      (default: keyframe)
#   --num-round        N     Round nuts per scene                  (default: 2)
#   --num-square       N     Square nuts per scene                 (default: 2)
#   --stacking-prob    F     Initial stacking probability          (default: 0.6)
#   --nut-type-mode    STR   roundnut | squarenut | random | alternate (default: random)
#   --image-size       N     Image resolution (pixels)             (default: 512)
#   --base-seed        N     Seed for worker 0; workers get base+i (default: 42)
#   --label-horizon    N     Expert rollout horizon for labelling  (default: 300)
#   --no-labels            Skip Phase 2 (no reachability labels computed)
#   --no-merge             Skip Phase 3 (keep worker dirs, no merge)
#   --keep-workers         Keep worker dirs after merging
#
# Examples:
#   # Collect + label + merge: 1000 episodes across 8 workers
#   bash parallel_collect.sh --workers 8 --total-episodes 1000
#
#   # Skip labelling (label later with compute_labels.py on the merged dataset)
#   bash parallel_collect.sh --workers 4 --total-episodes 200 --no-labels
#
#   # Noisy policy, custom output
#   bash parallel_collect.sh --workers 4 --total-episodes 200 \
#       --policy-mode noisy --noise-sigma 0.05 --output-dir dataset/noisy

set -euo pipefail

# ── Defaults ─────────────────────────────────────────────────────────────────
WORKERS=4
TOTAL_EPISODES=400
OUTPUT_DIR="dataset/nut_assembly"
ENV="ClutteredNutAssembly"
MAX_TIMESTEPS=1000
POLICY_MODE="expert"
NOISE_SIGMA=0.05
TRANSITION_MODE="keyframe"
NUM_ROUND=2
NUM_SQUARE=2
STACKING_PROB=0.6
NUT_TYPE_MODE="random"
IMAGE_SIZE=512
BASE_SEED=42
NO_MERGE=false
KEEP_WORKERS=false
NO_LABELS=false
LABEL_HORIZON=300

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --workers)          WORKERS="$2";          shift 2 ;;
        --total-episodes)   TOTAL_EPISODES="$2";   shift 2 ;;
        --output-dir)       OUTPUT_DIR="$2";        shift 2 ;;
        --env)              ENV="$2";               shift 2 ;;
        --max-timesteps)    MAX_TIMESTEPS="$2";     shift 2 ;;
        --policy-mode)      POLICY_MODE="$2";       shift 2 ;;
        --noise-sigma)      NOISE_SIGMA="$2";       shift 2 ;;
        --transition-mode)  TRANSITION_MODE="$2";   shift 2 ;;
        --num-round)        NUM_ROUND="$2";         shift 2 ;;
        --num-square)       NUM_SQUARE="$2";        shift 2 ;;
        --stacking-prob)    STACKING_PROB="$2";     shift 2 ;;
        --nut-type-mode)    NUT_TYPE_MODE="$2";     shift 2 ;;
        --image-size)       IMAGE_SIZE="$2";        shift 2 ;;
        --base-seed)        BASE_SEED="$2";         shift 2 ;;
        --label-horizon)    LABEL_HORIZON="$2";     shift 2 ;;
        --no-labels)        NO_LABELS=true;         shift   ;;
        --no-merge)         NO_MERGE=true;          shift   ;;
        --keep-workers)     KEEP_WORKERS=true;      shift   ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

# ── Derived values ────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Worker dirs live BESIDE the output dir (not inside it) so that overwriting
# or deleting the merged output can never accidentally destroy collected data.
WORKER_BASE_DIR="${OUTPUT_DIR%/}_workers"

# MuJoCo headless rendering: use EGL (GPU, no display required) unless the
# caller has already set MUJOCO_GL.  Workers inherit this via environment.
export MUJOCO_GL="${MUJOCO_GL:-egl}"

# Each worker process must not spawn dozens of BLAS/OMP threads - doing so
# exhausts RLIMIT_NPROC when many workers run in parallel (8 workers × 64
# OpenBLAS threads alone would need >500 threads).  Single-threaded BLAS is
# fine here because the simulation itself is the bottleneck, not BLAS.
export OPENBLAS_NUM_THREADS=2
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Distribute episodes as evenly as possible across workers
BASE_EPS=$(( TOTAL_EPISODES / WORKERS ))
REMAINDER=$(( TOTAL_EPISODES % WORKERS ))

# ── Banner ────────────────────────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════════════"
echo "  Parallel headless data collection"
echo "  Workers        : ${WORKERS}"
echo "  Total episodes : ${TOTAL_EPISODES}  (base ${BASE_EPS}/worker, +1 for first ${REMAINDER})"
echo "  Environment    : ${ENV}"
echo "  Policy mode    : ${POLICY_MODE} (sigma=${NOISE_SIGMA})"
echo "  Transition mode: ${TRANSITION_MODE}"
echo "  Output dir     : ${OUTPUT_DIR}"
echo "  Worker dirs    : ${WORKER_BASE_DIR}/worker_<i>"
echo "  Compute labels : $( [[ "${NO_LABELS}" == "true" ]] && echo 'no (--no-labels)' || echo "yes (horizon=${LABEL_HORIZON})" )"
echo "══════════════════════════════════════════════════════════"
echo ""

mkdir -p "${WORKER_BASE_DIR}"

# ── Launch workers ────────────────────────────────────────────────────────────
declare -a PIDS
declare -a LOG_FILES
declare -a WORKER_DIRS

for (( i=0; i<WORKERS; i++ )); do
    WORKER_DIR="${WORKER_BASE_DIR}/worker_${i}"
    LOG_FILE="${WORKER_BASE_DIR}/worker_${i}.log"
    SEED=$(( BASE_SEED + i ))

    # Workers 0..(REMAINDER-1) get one extra episode
    if (( i < REMAINDER )); then
        EPS=$(( BASE_EPS + 1 ))
    else
        EPS="${BASE_EPS}"
    fi

    WORKER_DIRS+=("${WORKER_DIR}")
    LOG_FILES+=("${LOG_FILE}")

    echo "[launcher] Starting worker ${i}: ${EPS} episodes, seed=${SEED} -> ${WORKER_DIR}"

    python "${SCRIPT_DIR}/batch_collect.py" \
        --env              "${ENV}" \
        --headless \
        --num-episodes     "${EPS}" \
        --max-timesteps    "${MAX_TIMESTEPS}" \
        --output-dir       "${WORKER_DIR}" \
        --policy-mode      "${POLICY_MODE}" \
        --noise-sigma      "${NOISE_SIGMA}" \
        --transition-mode  "${TRANSITION_MODE}" \
        --num-round        "${NUM_ROUND}" \
        --num-square       "${NUM_SQUARE}" \
        --initial-stacking-prob "${STACKING_PROB}" \
        --nut-type-mode    "${NUT_TYPE_MODE}" \
        --image-size       "${IMAGE_SIZE}" \
        --seed             "${SEED}" \
        > "${LOG_FILE}" 2>&1 &

    PIDS+=($!)
done

# ── Wait for all workers ──────────────────────────────────────────────────────
echo ""
echo "[launcher] All ${WORKERS} workers launched. Waiting..."
FAILED=0

for (( i=0; i<WORKERS; i++ )); do
    PID="${PIDS[$i]}"
    if wait "${PID}"; then
        echo "[launcher] Worker ${i} (PID ${PID}) finished OK"
    else
        EXIT_CODE=$?
        echo "[launcher] Worker ${i} (PID ${PID}) FAILED (exit ${EXIT_CODE})"
        echo "           See log: ${LOG_FILES[$i]}"
        FAILED=$(( FAILED + 1 ))
    fi
done

echo ""
if (( FAILED > 0 )); then
    echo "[launcher] WARNING: ${FAILED}/${WORKERS} workers failed."
    echo "           Check logs in ${WORKER_BASE_DIR}/"
fi

# ── Phase 2: label computation (parallel, one process per worker dir) ────────
if [[ "${NO_LABELS}" == "true" ]]; then
    echo "[launcher] --no-labels set; skipping reachability label computation."
else
    echo ""
    echo "[launcher] Phase 2: computing reachability labels in parallel..."
    declare -a LABEL_PIDS
    declare -a LABEL_LOG_FILES

    for (( i=0; i<WORKERS; i++ )); do
        WORKER_DIR="${WORKER_BASE_DIR}/worker_${i}"
        LABEL_LOG="${WORKER_BASE_DIR}/worker_${i}_labels.log"
        LABEL_LOG_FILES+=("${LABEL_LOG}")

        # Skip workers that produced no episodes
        if [[ ! -d "${WORKER_DIR}/episodes" ]]; then
            echo "[launcher] Worker ${i}: no episodes dir, skipping labels."
            LABEL_PIDS+=(-1)
            continue
        fi

        SEED=$(( BASE_SEED + i ))
        echo "[launcher] Labelling worker ${i} -> ${WORKER_DIR}"

        python "${SCRIPT_DIR}/compute_labels.py" \
            --dataset-dir  "${WORKER_DIR}" \
            --env          "${ENV}" \
            --horizon      "${LABEL_HORIZON}" \
            --num-round    "${NUM_ROUND}" \
            --num-square   "${NUM_SQUARE}" \
            --initial-stacking-prob "${STACKING_PROB}" \
            --nut-type-mode "${NUT_TYPE_MODE}" \
            --seed         "${SEED}" \
            > "${LABEL_LOG}" 2>&1 &

        LABEL_PIDS+=($!)
    done

    echo "[launcher] All label workers launched. Waiting..."
    LABEL_FAILED=0
    for (( i=0; i<WORKERS; i++ )); do
        PID="${LABEL_PIDS[$i]}"
        [[ "${PID}" == "-1" ]] && continue
        if wait "${PID}"; then
            echo "[launcher] Label worker ${i} (PID ${PID}) finished OK"
        else
            EXIT_CODE=$?
            echo "[launcher] Label worker ${i} (PID ${PID}) FAILED (exit ${EXIT_CODE})"
            echo "           See log: ${LABEL_LOG_FILES[$i]}"
            LABEL_FAILED=$(( LABEL_FAILED + 1 ))
        fi
    done

    if (( LABEL_FAILED > 0 )); then
        echo "[launcher] WARNING: ${LABEL_FAILED}/${WORKERS} label workers failed."
    fi
fi

# ── Phase 3: merge ────────────────────────────────────────────────────────────
if [[ "${NO_MERGE}" == "true" ]]; then
    echo "[launcher] --no-merge set; skipping merge. Worker dirs kept in ${WORKER_BASE_DIR}/"
    exit 0
fi

# Find non-empty worker dirs (workers that produced episodes)
VALID_DIRS=()
for WDIR in "${WORKER_DIRS[@]}"; do
    if [[ -d "${WDIR}/episodes" ]]; then
        VALID_DIRS+=("${WDIR}")
    else
        echo "[launcher] Skipping empty worker dir (no episodes/): ${WDIR}"
    fi
done

if (( ${#VALID_DIRS[@]} == 0 )); then
    echo "[launcher] ERROR: No worker produced any episodes. Nothing to merge." >&2
    exit 1
fi

echo "[launcher] Merging ${#VALID_DIRS[@]} worker dataset(s) -> ${OUTPUT_DIR}"
if python "${SCRIPT_DIR}/merge_datasets.py" \
    --source-dirs "${VALID_DIRS[@]}" \
    --output-dir  "${OUTPUT_DIR}" \
    --overwrite; then

    echo "[launcher] Merge complete. Final dataset: ${OUTPUT_DIR}"

    if [[ "${KEEP_WORKERS}" == "false" ]]; then
        echo "[launcher] Removing worker dirs (use --keep-workers to retain them)."
        rm -rf "${WORKER_BASE_DIR}"
    else
        echo "[launcher] Worker dirs kept in ${WORKER_BASE_DIR}/"
    fi
else
    echo "[launcher] ERROR: Merge failed (exit $?). Worker dirs preserved in ${WORKER_BASE_DIR}/" >&2
    exit 1
fi

echo ""
echo "Done."
