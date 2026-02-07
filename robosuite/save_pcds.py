# save_pcds.py
import pickle, numpy as np
from pathlib import Path
import open3d as o3d

ep_path = Path("data_capture/dataset/nut_assembly/episodes/episode_00002_subsampled.pkl")
out_dir = Path("debug_pcds")
out_dir.mkdir(exist_ok=True)

with open(ep_path, "rb") as f:
    data, attrs = pickle.load(f)

# pick a timestep (e.g., 0)
t = 2
object_names = [attrs["segmentation_labels"][k] for k in sorted(attrs["segmentation_labels"])]

for i, name in enumerate(object_names, start=1):
    pts = data[f"point_cloud_{i}"][t]  # (N,3)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    o3d.io.write_point_cloud(str(out_dir / f"{t:03d}_{name}.ply"), pcd)
    print(f"saved {name} -> {out_dir}/{t:03d}_{name}.ply")

print("Done.")