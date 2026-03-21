#!/usr/bin/env bash
# merge_all.sh — Merge all seed datasets (transitions + labels + episodes)
# in the local dataset/ directory into a single consolidated dataset.
#
# Usage:
#   bash merge_all.sh [--output-dir PATH] [--overwrite] [--skip-labels]
#
# Defaults:
#   --output-dir  dataset/nut_assembly_merged
#   (no --overwrite)
#   (labels are merged)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_DIR="${SCRIPT_DIR}/dataset"
OUTPUT_DIR="${DATASET_DIR}/nut_assembly_merged"
EXTRA_ARGS=()

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --output-dir)   OUTPUT_DIR="$2"; shift 2 ;;
        --overwrite)    EXTRA_ARGS+=("--overwrite"); shift ;;
        --skip-labels)  EXTRA_ARGS+=("--skip-labels"); shift ;;
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

# ── Banner ────────────────────────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════════════"
echo "  Merge all seed datasets"
echo "  Source datasets (${#FILTERED_DIRS[@]}):"
for d in "${FILTERED_DIRS[@]}"; do
    echo "    ${d}"
done
echo "  Output dir: ${OUTPUT_DIR}"
echo "══════════════════════════════════════════════════════════"
echo ""

# ── Run merge ─────────────────────────────────────────────────────────────────
python "${SCRIPT_DIR}/merge_datasets.py" \
    --source-dirs "${FILTERED_DIRS[@]}" \
    --output-dir  "${OUTPUT_DIR}" \
    "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"

echo ""
echo "Done. Merged dataset written to: ${OUTPUT_DIR}"
