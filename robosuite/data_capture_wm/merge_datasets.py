"""
Merge multiple world-model datasets into one consolidated dataset.

This utility merges dataset roots that follow the `data_capture_wm` layout:

  dataset/
    episodes/ep_00000/
    transitions.jsonl
    metadata.json
    labels.jsonl (optional)

During merge it:
  1. Renumbers episodes sequentially (`ep_00000`, `ep_00001`, ...)
  2. Copies all episode folders into a single output dataset
  3. Rewrites `episode_id` and episode-relative paths in `transitions.jsonl`
  4. Rewrites `episode_id` in `labels.jsonl` when present
  5. Regenerates output `metadata.json` stats from merged contents

Example:
  python merge_datasets.py \
      --source-dirs dataset_a dataset_b dataset_c \
      --output-dir merged_dataset
"""

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Dict, List, Optional, Tuple


EPISODE_PATH_KEYS = ("image_t", "image_t1", "state_t", "state_t1", "goal_image")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge multiple WM datasets")
    parser.add_argument(
        "--source-dirs",
        nargs="+",
        required=True,
        help="One or more dataset roots to merge",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output dataset root to create",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete output directory first if it already exists",
    )
    parser.add_argument(
        "--skip-labels",
        action="store_true",
        help="Do not merge labels.jsonl even if present",
    )
    parser.add_argument(
        "--skip-subskill",
        action="store_true",
        help="Do not merge transitions_subskill.jsonl even if present",
    )
    return parser.parse_args()


def _episode_sort_key(episode_name: str) -> Tuple[int, int, str]:
    if episode_name.startswith("ep_"):
        try:
            return (0, int(episode_name[3:]), episode_name)
        except ValueError:
            pass
    return (1, 0, episode_name)


def list_episode_dirs(dataset_dir: Path) -> List[Path]:
    episodes_dir = dataset_dir / "episodes"
    if not episodes_dir.exists():
        return []
    eps = [path for path in episodes_dir.iterdir() if path.is_dir()]
    eps.sort(key=lambda path: _episode_sort_key(path.name))
    return eps


def rewrite_episode_relpath(path_str: str, id_map: Dict[str, str]) -> str:
    p = PurePosixPath(path_str)
    parts = list(p.parts)
    if len(parts) >= 2 and parts[0] == "episodes" and parts[1] in id_map:
        parts[1] = id_map[parts[1]]
        return str(PurePosixPath(*parts))
    return path_str


def read_json(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    with open(path, "r") as f:
        return json.load(f)


def count_jsonl_lines(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with open(path, "r") as f:
        for _ in f:
            count += 1
    return count


def aggregate_field(values: List):
    uniq = []
    for value in values:
        if value not in uniq:
            uniq.append(value)
    if not uniq:
        return None
    if len(uniq) == 1:
        return uniq[0]
    return "mixed"


def main() -> None:
    args = parse_args()

    source_dirs = [Path(p).resolve() for p in args.source_dirs]
    output_dir = Path(args.output_dir).resolve()
    output_episodes_dir = output_dir / "episodes"

    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output directory exists: {output_dir}. Use --overwrite to replace it."
            )
        shutil.rmtree(output_dir)

    output_episodes_dir.mkdir(parents=True, exist_ok=True)
    output_transitions = output_dir / "transitions.jsonl"
    output_subskill = output_dir / "transitions_subskill.jsonl"
    output_labels = output_dir / "labels.jsonl"

    next_episode_idx = 0
    total_transitions_written = 0
    total_subskill_written = 0
    total_labels_written = 0
    total_episodes_written = 0
    successful_episodes = 0

    source_metadatas: List[Dict] = []
    source_reports: List[Dict] = []

    with open(output_transitions, "w") as transitions_out:
        labels_out = None
        if not args.skip_labels:
            labels_out = open(output_labels, "w")
        subskill_out = None
        if not args.skip_subskill:
            subskill_out = open(output_subskill, "w")

        try:
            for src_dir in source_dirs:
                if not src_dir.exists():
                    raise FileNotFoundError(f"Source dataset not found: {src_dir}")

                src_metadata = read_json(src_dir / "metadata.json")
                if src_metadata is not None:
                    source_metadatas.append(src_metadata)

                src_episode_dirs = list_episode_dirs(src_dir)
                id_map: Dict[str, str] = {}
                copied_episode_meta_paths: List[Path] = []

                for src_episode_dir in src_episode_dirs:
                    old_id = src_episode_dir.name
                    new_id = f"ep_{next_episode_idx:05d}"
                    next_episode_idx += 1

                    id_map[old_id] = new_id
                    dst_episode_dir = output_episodes_dir / new_id
                    shutil.copytree(src_episode_dir, dst_episode_dir)
                    copied_episode_meta_paths.append(dst_episode_dir / "meta.json")

                # Rewrite per-episode metadata now that local id map is complete.
                for meta_path in copied_episode_meta_paths:
                    if not meta_path.exists():
                        continue
                    with open(meta_path, "r") as f:
                        episode_meta = json.load(f)

                    old_episode_id = episode_meta.get("episode_id", "")
                    new_episode_id = id_map.get(old_episode_id)
                    if new_episode_id is not None:
                        episode_meta["episode_id"] = new_episode_id

                    goal_image = episode_meta.get("goal_image")
                    if isinstance(goal_image, str) and goal_image:
                        episode_meta["goal_image"] = rewrite_episode_relpath(
                            goal_image, id_map
                        )

                    with open(meta_path, "w") as f:
                        json.dump(episode_meta, f, indent=2)

                    total_episodes_written += 1
                    if bool(episode_meta.get("success", False)):
                        successful_episodes += 1

                src_transitions_path = src_dir / "transitions.jsonl"
                transitions_written_this_source = 0
                if src_transitions_path.exists():
                    with open(src_transitions_path, "r") as src_tf:
                        for line in src_tf:
                            if not line.strip():
                                continue
                            row = json.loads(line)

                            old_episode_id = row.get("episode_id")
                            if old_episode_id not in id_map:
                                continue

                            row["episode_id"] = id_map[old_episode_id]
                            for key in EPISODE_PATH_KEYS:
                                value = row.get(key)
                                if isinstance(value, str) and value:
                                    row[key] = rewrite_episode_relpath(value, id_map)

                            transitions_out.write(json.dumps(row) + "\n")
                            transitions_written_this_source += 1

                subskill_written_this_source = 0
                src_subskill_path = src_dir / "transitions_subskill.jsonl"
                if subskill_out is not None and src_subskill_path.exists():
                    with open(src_subskill_path, "r") as src_sf:
                        for line in src_sf:
                            if not line.strip():
                                continue
                            row = json.loads(line)
                            old_episode_id = row.get("episode_id")
                            if old_episode_id not in id_map:
                                continue
                            row["episode_id"] = id_map[old_episode_id]
                            for key in EPISODE_PATH_KEYS:
                                value = row.get(key)
                                if isinstance(value, str) and value:
                                    row[key] = rewrite_episode_relpath(value, id_map)
                            subskill_out.write(json.dumps(row) + "\n")
                            subskill_written_this_source += 1

                labels_written_this_source = 0
                src_labels_path = src_dir / "labels.jsonl"
                if labels_out is not None and src_labels_path.exists():
                    with open(src_labels_path, "r") as src_lf:
                        for line in src_lf:
                            if not line.strip():
                                continue
                            row = json.loads(line)
                            old_episode_id = row.get("episode_id")
                            if old_episode_id not in id_map:
                                continue
                            row["episode_id"] = id_map[old_episode_id]
                            labels_out.write(json.dumps(row) + "\n")
                            labels_written_this_source += 1

                total_transitions_written += transitions_written_this_source
                total_subskill_written += subskill_written_this_source
                total_labels_written += labels_written_this_source

                source_reports.append(
                    {
                        "source_dir": str(src_dir),
                        "episodes_copied": len(src_episode_dirs),
                        "transitions_copied": transitions_written_this_source,
                        "subskill_transitions_copied": subskill_written_this_source,
                        "labels_copied": labels_written_this_source,
                    }
                )

                print(
                    f"Merged {src_dir}: "
                    f"episodes={len(src_episode_dirs)}, "
                    f"transitions={transitions_written_this_source}, "
                    f"subskill={subskill_written_this_source}, "
                    f"labels={labels_written_this_source}"
                )
        finally:
            if subskill_out is not None:
                subskill_out.close()
            if labels_out is not None:
                labels_out.close()

    # If subskill merging was enabled but nothing was written, remove empty file.
    if not args.skip_subskill and output_subskill.exists() and total_subskill_written == 0:
        output_subskill.unlink()

    # If labels merging was enabled but no labels were written, remove empty file.
    if not args.skip_labels and output_labels.exists() and total_labels_written == 0:
        output_labels.unlink()

    merged_meta = {
        "env_name": aggregate_field([md.get("env_name") for md in source_metadatas]),
        "env_config": aggregate_field([md.get("env_config") for md in source_metadatas]),
        "policy_mode": aggregate_field([md.get("policy_mode") for md in source_metadatas]),
        "noise_sigma": aggregate_field([md.get("noise_sigma") for md in source_metadatas]),
        "transition_mode": aggregate_field([md.get("transition_mode") for md in source_metadatas]),
        "camera": aggregate_field([md.get("camera") for md in source_metadatas]),
        "image_size": aggregate_field([md.get("image_size") for md in source_metadatas]),
        "stats": {
            "total": total_episodes_written,
            "success": successful_episodes,
            "failed": total_episodes_written - successful_episodes,
            "transitions": total_transitions_written,
            "subskill_transitions": total_subskill_written,
        },
        "timestamp": datetime.now().isoformat(),
        "merged_from": [str(path) for path in source_dirs],
        "sources": source_reports,
    }

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(merged_meta, f, indent=2)

    print("\nMerge complete")
    print(f"  Output:      {output_dir}")
    print(f"  Episodes:    {total_episodes_written}")
    print(f"  Transitions: {count_jsonl_lines(output_transitions)}")
    if not args.skip_subskill and output_subskill.exists():
        print(f"  Subskill:    {count_jsonl_lines(output_subskill)}")
    if not args.skip_labels and output_labels.exists():
        print(f"  Labels:      {count_jsonl_lines(output_labels)}")


if __name__ == "__main__":
    main()
