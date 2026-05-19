#!/usr/bin/env python3
"""
Script to filter the heavy CALVIN dataset down to a lightweight representation.
Extracts only the 'rgb_static' key from required episodes, saving massive disk space.
"""

import os
import argparse
import shutil
import numpy as np
from pathlib import Path
from tqdm import tqdm

def filter_dataset(input_dir: str, output_dir: str, history_len: int, compress: bool):
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_path}")

    # 1. Locate and load auto_lang_ann.npy
    lang_ann_path = input_path / "lang_annotations" / "auto_lang_ann.npy"
    if not lang_ann_path.exists():
        lang_ann_path = input_path / "auto_lang_ann.npy"

    if not lang_ann_path.exists():
        raise FileNotFoundError(f"Could not find auto_lang_ann.npy in {input_path}")

    print(f"Loading annotations from: {lang_ann_path}")
    annotations = np.load(lang_ann_path, allow_pickle=True).item()
    indices = annotations["info"]["indx"]
    print(f"Loaded {len(indices)} transitions.")

    # 2. Identify all unique required frame indices
    required_indices = set()
    for start_idx, end_idx in indices:
        # Target image
        required_indices.add(end_idx)
        # History images [I_{start_idx - H + 1}, ..., I_{start_idx}]
        for j in range(history_len):
            required_indices.add(max(0, start_idx - j))

    print(f"Identified {len(required_indices)} unique required frames.")

    # 3. Create output directories and copy annotations/metadata
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create matching lang_annotations subdirectory in output
    output_lang_dir = output_path / "lang_annotations"
    output_lang_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(lang_ann_path, output_lang_dir / "auto_lang_ann.npy")
    print(f"Copied auto_lang_ann.npy to: {output_lang_dir / 'auto_lang_ann.npy'}")

    # Also copy other common CALVIN metadata files if they exist (ep_lens.npy, etc.)
    for meta_file in ["ep_lens.npy", "ep_start_end_ids.npy"]:
        meta_src = input_path / meta_file
        if meta_src.exists():
            shutil.copy2(meta_src, output_path / meta_file)
            print(f"Copied {meta_file} metadata file.")

    # 4. Extract and write lightweight npz files
    print("\nFiltering and saving episode files...")
    original_total_size = 0
    filtered_total_size = 0
    missing_count = 0

    sorted_indices = sorted(list(required_indices))
    for idx in tqdm(sorted_indices, desc="Processing episodes"):
        filename = f"episode_{idx:07d}.npz"
        src_file = input_path / filename
        dst_file = output_path / filename

        if not src_file.exists():
            missing_count += 1
            continue

        # Keep track of original file size
        original_total_size += src_file.stat().st_size

        # Load original
        try:
            data = np.load(src_file, allow_pickle=True)
            if "rgb_static" not in data:
                print(f"\nWarning: 'rgb_static' not found in {filename}. Skipping.")
                continue
            
            rgb_static = data["rgb_static"]

            # Save only rgb_static
            if compress:
                np.savez_compressed(dst_file, rgb_static=rgb_static)
            else:
                np.savez(dst_file, rgb_static=rgb_static)

            filtered_total_size += dst_file.stat().st_size
        except Exception as e:
            print(f"\nError processing {filename}: {e}. Skipping.")

    # 5. Print out report
    print("\n" + "="*50)
    print("Dataset filtering completed successfully!")
    print("="*50)
    print(f"Input Directory:  {input_dir}")
    print(f"Output Directory: {output_dir}")
    print(f"Processed:        {len(sorted_indices) - missing_count} / {len(sorted_indices)} requested files.")
    if missing_count > 0:
        print(f"Missing files:    {missing_count} (gracefully skipped)")
    
    orig_mb = original_total_size / (1024 * 1024)
    filt_mb = filtered_total_size / (1024 * 1024)
    savings = (1 - (filtered_total_size / original_total_size)) * 100 if original_total_size > 0 else 0

    print(f"Original Size:    {orig_mb:.2f} MB")
    print(f"Filtered Size:    {filt_mb:.2f} MB")
    print(f"Size Reduction:   {savings:.2f}%")
    print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter CALVIN dataset down to rgb_static only.")
    parser.add_argument("--input-dir", type=str, required=True, help="Path to original CALVIN split training/validation folder")
    parser.add_argument("--output-dir", type=str, required=True, help="Path to output lightweight training/validation folder")
    parser.add_argument("--history-len", type=int, default=3, help="History window length used by dataset loader")
    parser.add_argument("--no-compress", action="store_true", help="Disable npz compression (saves faster but takes slightly more space)")
    
    args = parser.parse_args()
    
    filter_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        history_len=args.history_len,
        compress=not args.no_compress
    )
