import os
import sys
import torch

# Add verify2act path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from verify2act.dino_wm_baseline.dynamics import BaselineDINOWM

def test_baseline():
    print("Testing BaselineDINOWM initialization...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize model
    model = BaselineDINOWM(
        dino_channels=1024,
        clip_channels=512,
        action_dim=64,
        history_len=3,
        num_patches=256,
        depth=2,  # shallow for test speed
        heads=4,
        mlp_dim=512,
        concat_dim=0
    ).to(device)

    print("BaselineDINOWM initialized successfully!")

    # Create mock inputs
    B, T, num_patches, channels = 2, 4, 256, 1024
    xt_history = torch.randn(B, T, num_patches, channels, device=device)
    action_tokens = torch.randn(B, 7, 512, device=device)

    print(f"Mock inputs shapes: xt_history {xt_history.shape}, action_tokens {action_tokens.shape}")

    # Test forward pass (training loop interface)
    print("Testing forward pass...")
    z_pred, z_tgt, loss = model(xt_history, action_tokens)
    print(f"Forward completed! loss: {loss.item():.4f}")

    # Test step pass (inference rollout interface)
    print("Testing step/rollout pass...")
    pred_visual = model.step(xt_history, action_tokens)
    print(f"Step completed! pred_visual shape: {pred_visual.shape}")
    assert pred_visual.shape == (B, num_patches, channels), f"Unexpected shape {pred_visual.shape}"
    print("All tests passed successfully!")

if __name__ == "__main__":
    test_baseline()
