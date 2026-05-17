#!/usr/bin/env bash
# merge_all.sh — Merge all seed datasets (transitions + labels + episodes)
# in the local dataset/ directory into a single consolidated dataset.
#
# Usage:
#   bash merge_all.sh [--output-dir PATH] [--overwrite] [--skip-labels] [--skip-subskill]
#
# Defaults:
#   --output-dir  dataset/nut_assembly_merged
#   (no --overwrite)
#   (labels are merged)
#   (transitions_subskill.jsonl is merged)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# dataset/ lives next to the scripts/ directory (parent of SCRIPT_DIR)
DATASET_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)/dataset"
OUTPUT_DIR="${DATASET_DIR}/nut_assembly_merged"
EXTRA_ARGS=()

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --output-dir)   OUTPUT_DIR="$2"; shift 2 ;;
        --overwrite)    EXTRA_ARGS+=("--overwrite"); shift ;;
        --skip-labels)  EXTRA_ARGS+=("--skip-labels"); shift ;;
        --skip-subskill) EXTRA_ARGS+=("--skip-subskill"); shift ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

# Resolve output dir relative to script if not absolute
if [[ "${OUTPUT_DIR}" != /* ]]; then
    OUTPUT_DIR="${SCRIPT_DIR}/${OUTPUT_DIR}"
fi

# ── Discover source datasets ──────────────────────────────────────────────────
# A valid dataset dir must contain an episodes/ subdirectory.
SOURCE_DIRS=()
while IFS= read -r -d '' dir; do
    if [[ -d "${dir}/episodes" ]]; then
        SOURCE_DIRS+=("${dir}")
    fi
done < <(find "${DATASET_DIR}" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z)

# Exclude the output dir itself in case it already exists inside dataset/
FILTERED_DIRS=()
for d in "${SOURCE_DIRS[@]}"; do
    if [[ "$(realpath "${d}")" != "$(realpath "${OUTPUT_DIR}" 2>/dev/null || echo __none__)" ]]; then
        FILTERED_DIRS+=("${d}")
    fi
done

if (( ${#FILTERED_DIRS[@]} == 0 )); then
    echo "ERROR: No datasets with episodes/ found under ${DATASET_DIR}" >&2
    exit 1
fi

# ── Detect a large pre-merged base dataset ───────────────────────────────────
# If nut_assembly_merged_prev exists among the source dirs, promote it to the
# base (output) directory instead of re-copying all its episodes.  We simply
# rename it to OUTPUT_DIR (an instant mv on the same filesystem) and then only
# append the remaining smaller datasets to it.
PREV_DIR="${DATASET_DIR}/nut_assembly_merged_prev"
USE_BASE_DIR=false
if [[ -d "${PREV_DIR}/episodes" ]]; then
    # Remove prev from the list of sources so it is not copied again.
    NEW_FILTERED=()
    for d in "${FILTERED_DIRS[@]}"; do
        if [[ "$(realpath "${d}")" != "$(realpath "${PREV_DIR}")" ]]; then
            NEW_FILTERED+=("${d}")
        fi
    done
    FILTERED_DIRS=("${NEW_FILTERED[@]}")

    # Handle the case where OUTPUT_DIR already exists.
    if [[ -d "${OUTPUT_DIR}" ]]; then
        if [[ " ${EXTRA_ARGS[*]} " == *" --overwrite "* ]]; then
            echo "--overwrite: removing existing output dir ${OUTPUT_DIR}"
            rm -rf "${OUTPUT_DIR}"
        else
            echo "ERROR: Output directory already exists: ${OUTPUT_DIR}" >&2
            echo "       Pass --overwrite to replace it." >&2
            exit 1
        fi
    fi

    echo "Promoting ${PREV_DIR} → ${OUTPUT_DIR}  (rename, no copy)"
    mv "${PREV_DIR}" "${OUTPUT_DIR}"
    USE_BASE_DIR=true
fi

# ── Banner ────────────────────────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════════════"
echo "  Merge all seed datasets"
echo "  Source datasets (${#FILTERED_DIRS[@]}):"
for d in "${FILTERED_DIRS[@]}"; do
    echo "    ${d}"
done
echo "  Output dir: ${OUTPUT_DIR}"
if [[ "${USE_BASE_DIR}" == true ]]; then
    echo "  Mode: append to base (nut_assembly_merged_prev promoted, no re-copy)"
else
    echo "  Mode: fresh merge"
fi
echo "══════════════════════════════════════════════════════════"
echo ""

# ── Run merge ─────────────────────────────────────────────────────────────────
MERGE_SCRIPT="$(cd "${SCRIPT_DIR}/.." && pwd)/merge_datasets.py"
if [[ ! -f "${MERGE_SCRIPT}" ]]; then
    echo "ERROR: merge script not found: ${MERGE_SCRIPT}" >&2
    exit 1
fi

if [[ "${USE_BASE_DIR}" == true ]]; then
    if (( ${#FILTERED_DIRS[@]} == 0 )); then
        echo "No additional datasets to append — base dataset is already up to date."
    else
        python "${MERGE_SCRIPT}" \
            --source-dirs "${FILTERED_DIRS[@]}" \
            --base-dir    "${OUTPUT_DIR}" \
            "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
    fi
else
    python "${MERGE_SCRIPT}" \
        --source-dirs "${FILTERED_DIRS[@]}" \
        --output-dir  "${OUTPUT_DIR}" \
        "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
fi

echo ""
echo "Done. Merged dataset written to: ${OUTPUT_DIR}"
