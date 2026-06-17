#!/usr/bin/env python3
import os
import sys
from pathlib import Path

# Add calvin_env to python path
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "calvin" / "calvin_env"))

def main():
    try:
        import hydra
        from omegaconf import OmegaConf
    except ImportError:
        print("Error: Please activate your conda environment (e.g. verify2act) containing hydra and omegaconf.")
        sys.exit(1)

    print("Composing environment configuration using Hydra...")
    
    # Initialize hydra with calvin_env configuration path
    config_dir = str(repo_root / "calvin" / "calvin_env" / "conf")
    
    # Clear global hydra state if already initialized
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize_config_dir(config_dir=config_dir)
    
    # Compose the configuration with necessary overrides
    cfg = hydra.compose(
        config_name="config_data_collection",
        overrides=[
            "cameras=static_and_gripper",
            "use_vr=false",
            "hydra/job_logging=default",
            "hydra/hydra_logging=default"
        ]
    )
    
    # Resolve the main config sections needed by play_table_env.py
    resolved_cfg = OmegaConf.create({
        "cameras": cfg.cameras,
        "env": cfg.env,
        "scene": cfg.scene,
        "robot": cfg.robot,
        "seed": cfg.seed,
        "use_vr": cfg.use_vr,
        "data_path": cfg.data_path,
    })
    
    # Resolve all interpolations in-place
    OmegaConf.resolve(resolved_cfg)
    yaml_content = OmegaConf.to_yaml(resolved_cfg)
    
    # Target dataset directories to populate
    target_dirs = [
        repo_root / "calvin" / "dataset" / "task_ABCD_D_filtered" / "training" / ".hydra",
        repo_root / "calvin" / "dataset" / "task_ABCD_D_filtered" / "validation" / ".hydra",
        repo_root / "calvin" / "dataset" / "task_ABC_D_filtered" / "training" / ".hydra",
        repo_root / "calvin" / "dataset" / "task_ABC_D_filtered" / "validation" / ".hydra",
    ]
    
    created_any = False
    for path in target_dirs:
        # Check if the parent directory (e.g. validation or training) exists before writing config
        if path.parent.exists():
            path.mkdir(parents=True, exist_ok=True)
            file_path = path / "merged_config.yaml"
            file_path.write_text(yaml_content)
            print(f"Successfully generated config: {file_path}")
            created_any = True
        else:
            print(f"Skipping (directory not found): {path.parent}")
            
    if created_any:
        print("\nCalvin environment configurations populated successfully!")
    else:
        print("\nWarning: No dataset directories found. Make sure the CALVIN datasets are downloaded/extracted.")

if __name__ == "__main__":
    main()
