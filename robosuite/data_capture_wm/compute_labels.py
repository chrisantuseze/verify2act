"""
Reachability Label Computation (Stage 2 post-processing)

For each transition in transitions.jsonl:
  1. Restore sim state from state_{t+1}.npz
  2. Run expert policy for up to H steps
  3. If env reports task success → label_reachable = 1, else 0
  4. Write labels to labels.jsonl

Usage:
    xvfb-run -a python compute_labels.py \
        --dataset-dir dataset/nut_assembly \
        --env ClutteredNutAssembly \
        --horizon 300 \
        --output labels.jsonl
"""

import os
if 'MUJOCO_GL' not in os.environ:
    os.environ['MUJOCO_GL'] = 'glx'

import sys
import json
import argparse
import io
from contextlib import redirect_stdout, redirect_stderr
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List
from tqdm import tqdm
import os

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # robosuite/
sys.path.insert(0, str(Path(__file__).resolve().parent))         # data_capture_wm/


def restore_sim_state(env, npz_path: str):
    """Load qpos/qvel from .npz and set the simulator to that state."""
    data = np.load(npz_path)
    qpos = data["qpos"]
    qvel = data["qvel"]

    sim_qpos_dim = env.sim.data.qpos.shape[0]
    sim_qvel_dim = env.sim.data.qvel.shape[0]

    if qpos.shape[0] != sim_qpos_dim or qvel.shape[0] != sim_qvel_dim:
        raise ValueError(
            f"State dimension mismatch for {npz_path}: "
            f"saved qpos/qvel=({qpos.shape[0]}, {qvel.shape[0]}), "
            f"env expects=({sim_qpos_dim}, {sim_qvel_dim})."
        )

    state = env.sim.get_state()
    if hasattr(state, "_replace"):
        new_state = state._replace(qpos=qpos, qvel=qvel)
        env.sim.set_state(new_state)
    else:
        env.sim.data.qpos[:] = qpos
        env.sim.data.qvel[:] = qvel
    env.sim.forward()


def load_metadata(dataset_dir: Path) -> Dict:
    metadata_path = dataset_dir / "metadata.json"
    if not metadata_path.exists():
        return {}
    with open(metadata_path, "r") as f:
        return json.load(f)


def get_saved_state_dims(dataset_dir: Path, transitions: List[Dict]) -> Tuple[int, int]:
    if not transitions:
        raise ValueError("No transitions found in transitions.jsonl")
    first_state = dataset_dir / transitions[0]["state_t1"]
    arr = np.load(first_state)
    return int(arr["qpos"].shape[0]), int(arr["qvel"].shape[0])


def group_transitions_by_dims(
    dataset_dir: Path, transitions: List[Dict]
) -> Dict[Tuple[int, int], List[Dict]]:
    """Group transitions by their saved qpos/qvel dimensions.

    This handles merged datasets collected under different env configurations
    (e.g. different numbers of nuts → different state-space sizes).
    """
    groups: Dict[Tuple[int, int], List[Dict]] = {}
    for tr in tqdm(transitions, desc="Inspecting state dims", leave=False):
        state_path = dataset_dir / tr["state_t1"]
        arr = np.load(state_path)
        dims = (int(arr["qpos"].shape[0]), int(arr["qvel"].shape[0]))
        groups.setdefault(dims, []).append(tr)
    return groups


def detect_cluttered_counts(
    target_qpos_dim: int,
    target_qvel_dim: int,
    initial_stacking_prob: float,
    nut_type_mode: str,
    horizon: int,
    max_round: int,
    max_square: int,
):
    from run_cluttered_nutassembly import create_environment

    matches = []
    for num_round in range(1, max_round + 1):
        for num_square in range(0, max_square + 1):
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                env = create_environment(
                    env_name="ClutteredNutAssembly",
                    num_round_nuts=num_round,
                    num_square_nuts=num_square,
                    initial_stacking_prob=initial_stacking_prob,
                    nut_type_mode=nut_type_mode,
                    horizon=horizon,
                )
                env.reset()
                dims = (int(env.sim.data.qpos.shape[0]), int(env.sim.data.qvel.shape[0]))
                env.close()
            if dims == (target_qpos_dim, target_qvel_dim):
                matches.append((num_round, num_square))
    return matches


def assert_env_matches_saved_dims(env, saved_dims: Tuple[int, int], context: str):
    env_dims = (int(env.sim.data.qpos.shape[0]), int(env.sim.data.qvel.shape[0]))
    if env_dims != saved_dims:
        raise ValueError(
            f"{context}: saved qpos/qvel={saved_dims}, env qpos/qvel={env_dims}. "
            "Use the exact collection environment configuration."
        )


def check_reachability(
    env,
    expert_policy_factory,
    state_npz: str,
    horizon: int,
) -> bool:
    """
    Restore sim to the given state and run expert for up to *horizon* steps.
    Returns True if the task success flag fires within that budget.
    """
    restore_sim_state(env, state_npz)

    # Need to re-derive observations after state restore
    obs = env._get_observations()
    policy = expert_policy_factory(env, data_collection_mode=True)
    policy.obs = obs

    for _ in range(horizon):
        action, policy_done = policy.step()
        obs, reward, env_done, info = env.step(action)
        policy.obs = obs
        if env_done:
            return True
        if policy_done:
            break
    return False


def main():
    parser = argparse.ArgumentParser(description="Compute reachability labels")
    parser.add_argument("--dataset-dir", type=str, required=True)
    parser.add_argument("--env", type=str, required=True,
                        choices=["ClutteredNutAssembly", "NutAssembly",
                                 "Stack", "Stack3", "PickPlace"])
    parser.add_argument("--horizon", type=int, default=300)
    parser.add_argument("--output", type=str, default="labels.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    # ClutteredNutAssembly params
    parser.add_argument("--num-round", type=int, default=6)
    parser.add_argument("--num-square", type=int, default=2)
    parser.add_argument("--initial-stacking-prob", type=float, default=0.6)
    parser.add_argument(
        "--nut-type-mode",
        type=str,
        default="roundnut",
        choices=["roundnut", "squarenut", "random", "alternate"],
        help="Nut type mode for ClutteredNutAssembly",
    )
    parser.add_argument("--auto-detect-cluttered-config", action="store_true")
    parser.add_argument("--max-round-search", type=int, default=8)
    parser.add_argument("--max-square-search", type=int, default=4)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    transitions_path = dataset_dir / "transitions.jsonl"
    if not transitions_path.exists():
        print(f"Error: {transitions_path} not found.")
        sys.exit(1)

    # Read all transitions
    transitions = []
    with open(transitions_path) as f:
        for line in f:
            transitions.append(json.loads(line.strip()))

    print(f"Loaded {len(transitions)} transitions from {transitions_path}")
    metadata = load_metadata(dataset_dir)

    # Group transitions by their saved state dims to handle merged datasets
    dim_groups = group_transitions_by_dims(dataset_dir, transitions)
    print(f"Found {len(dim_groups)} distinct state-dim group(s): {list(dim_groups.keys())}")

    np.random.seed(args.seed)

    # Compute labels
    labels_path = dataset_dir / args.output
    count_pos, count_neg, count_skip = 0, 0, 0
    all_rows: Dict[Tuple[str, int], Dict] = {}  # keyed by (episode_id, timestep) for ordered output

    for saved_dims, group_transitions in dim_groups.items():
        print(f"\nProcessing {len(group_transitions)} transitions with dims qpos={saved_dims[0]}, qvel={saved_dims[1]}")

        # ---- Build environment for this dimension group ----
        if args.env == "ClutteredNutAssembly":
            from run_cluttered_nutassembly import create_environment
            env_config = {
                "num_round_nuts": args.num_round,
                "num_square_nuts": args.num_square,
                "initial_stacking_prob": args.initial_stacking_prob,
                "nut_type_mode": args.nut_type_mode,
            }
            md_cfg = metadata.get("env_config", {})
            if md_cfg.get("env_name") == "ClutteredNutAssembly":
                env_config.update(
                    {
                        "num_round_nuts": int(md_cfg.get("num_round_nuts", env_config["num_round_nuts"])),
                        "num_square_nuts": int(md_cfg.get("num_square_nuts", env_config["num_square_nuts"])),
                        "initial_stacking_prob": float(md_cfg.get("initial_stacking_prob", env_config["initial_stacking_prob"])),
                        "nut_type_mode": md_cfg.get("nut_type_mode", env_config["nut_type_mode"]),
                    }
                )

            env = create_environment(
                env_name="ClutteredNutAssembly",
                num_round_nuts=env_config["num_round_nuts"],
                num_square_nuts=env_config["num_square_nuts"],
                initial_stacking_prob=env_config["initial_stacking_prob"],
                nut_type_mode=env_config["nut_type_mode"],
                horizon=args.horizon,
                has_renderer=False,  # no need to render for label computation
                has_offscreen_renderer=True,
            )
            env.reset()

            try:
                assert_env_matches_saved_dims(env, saved_dims, "Initial cluttered config")
            except ValueError:
                env.close()
                matches = detect_cluttered_counts(
                    target_qpos_dim=saved_dims[0],
                    target_qvel_dim=saved_dims[1],
                    initial_stacking_prob=env_config["initial_stacking_prob"],
                    nut_type_mode=env_config["nut_type_mode"],
                    horizon=args.horizon,
                    max_round=args.max_round_search,
                    max_square=args.max_square_search,
                )
                if len(matches) != 1:
                    print(
                        f"  WARNING: Could not uniquely infer ClutteredNutAssembly config "
                        f"for dims {saved_dims}. Candidates={matches}. Skipping this group."
                    )
                    count_skip += len(group_transitions)
                    continue

                inferred_round, inferred_square = matches[0]
                print(
                    f"  Auto-detected cluttered config: num_round={inferred_round}, "
                    f"num_square={inferred_square}"
                )
                env = create_environment(
                    env_name="ClutteredNutAssembly",
                    num_round_nuts=inferred_round,
                    num_square_nuts=inferred_square,
                    initial_stacking_prob=env_config["initial_stacking_prob"],
                    nut_type_mode=env_config["nut_type_mode"],
                    horizon=args.horizon,
                )
                env.reset()
                assert_env_matches_saved_dims(env, saved_dims, "Auto-detected cluttered config")

            from policy_wrappers import create_nut_assembly_policy as expert_factory
        elif args.env == "NutAssembly":
            from run_nutassembly import create_environment, HeuristicNutAssemblyPolicy
            env = create_environment(env_name="NutAssembly")
            env.reset()
            assert_env_matches_saved_dims(env, saved_dims, "NutAssembly config")

            def expert_factory(env, data_collection_mode=True):
                return HeuristicNutAssemblyPolicy(env)
        elif args.env in ("Stack", "Stack3"):
            from run_stack import create_environment
            env = create_environment(args.env)
            env.reset()
            assert_env_matches_saved_dims(env, saved_dims, f"{args.env} config")
            from policy_wrappers import create_stack_policy as expert_factory
        elif args.env == "PickPlace":
            from run_pickplace import create_environment
            env = create_environment("PickPlaceCan")
            env.reset()
            assert_env_matches_saved_dims(env, saved_dims, "PickPlace config")
            from policy_wrappers import create_pickplace_policy as expert_factory
        else:
            raise ValueError(f"Unsupported env: {args.env}")

        # ---- Label this group ----
        for tr in tqdm(group_transitions, desc=f"Labeling dims={saved_dims}"):
            state_path = str(dataset_dir / tr["state_t1"])
            reachable = check_reachability(
                env, expert_factory, state_path, args.horizon
            )
            label = 1 if reachable else 0
            if label == 1:
                count_pos += 1
            else:
                count_neg += 1
            all_rows[(tr["episode_id"], tr["timestep"])] = {
                "episode_id": tr["episode_id"],
                "timestep": tr["timestep"],
                "label_reachable": label,
            }

        env.close()

    # Write output in the original transition order
    with open(labels_path, "w") as out_f:
        for tr in transitions:
            key = (tr["episode_id"], tr["timestep"])
            if key not in all_rows:
                continue  # was skipped (dim mismatch with no unique match)
            out_f.write(json.dumps(all_rows[key]) + "\n")
            out_f.flush()
            try:
                os.fsync(out_f.fileno())
            except Exception:
                pass

    total = count_pos + count_neg
    print(f"\nDone. Wrote {total} labels to {labels_path}")
    print(f"  Positive (reachable): {count_pos} ({count_pos/total*100:.1f}%)")
    print(f"  Negative:             {count_neg} ({count_neg/total*100:.1f}%)")
    if count_skip:
        print(f"  Skipped (dim mismatch, no unique env match): {count_skip}")


if __name__ == "__main__":
    main()
