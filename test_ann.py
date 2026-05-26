import numpy as np
from pathlib import Path

root = Path("calvin/dataset/task_ABC_D_filtered/training")
lang_ann_path = root / "lang_annotations" / "auto_lang_ann.npy"
if not lang_ann_path.exists():
    lang_ann_path = root / "auto_lang_ann.npy"

annotations = np.load(lang_ann_path, allow_pickle=True).item()
indices = annotations["info"]["indx"]

print("Number of transitions in auto_lang_ann.npy:", len(indices))
print("First 10 transitions:", indices[:10])

valid_files = set()
for p in root.glob("episode_*.npz"):
    try:
        valid_files.add(int(p.stem.split('_')[1]))
    except:
        pass

print("Number of valid files:", len(valid_files))

missing_starts = 0
missing_ends = 0
for start_idx, end_idx in indices:
    if start_idx not in valid_files: missing_starts += 1
    if end_idx not in valid_files: missing_ends += 1

print("Transitions with missing start_idx:", missing_starts)
print("Transitions with missing end_idx:", missing_ends)
