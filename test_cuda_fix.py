#!/usr/bin/env python
"""
Quick test to verify the CUDA → CPU fallback fix works correctly.
This simulates the exact scenario from your training code.
"""

import torch
import torch.nn as nn
import sys
import os

# Add the path to your modules
sys.path.insert(0, '/home/e_chrisantus/Projects/multi-object-manip/verify2act/Points2Plans/relational_dynamics')

print("=" * 70)
print("Testing CUDA → CPU Fallback Fix")
print("=" * 70)

try:
    from model.pointconv_util_groupnorm import PointConvDensitySetAbstraction
    print("✓ Successfully imported PointConvDensitySetAbstraction")
except Exception as e:
    print(f"✗ Failed to import: {e}")
    sys.exit(1)

# Test with CUDA if available
if not torch.cuda.is_available():
    print("\n⚠️  CUDA not available. This test requires a GPU.")
    print("   The fix is designed for CUDA operations, so CPU-only testing")
    print("   won't reproduce the original error.")
    sys.exit(0)

print(f"\n✓ CUDA is available")
print(f"  Device: {torch.cuda.get_device_name(0)}")
print(f"  PyTorch version: {torch.__version__}")
print(f"  CUDA version: {torch.version.cuda}")

# Create the model that was crashing
print("\n[1] Creating PointConvDensitySetAbstraction model...")
try:
    # These are typical parameters from your model
    model = PointConvDensitySetAbstraction(
        npoint=64,
        nsample=16, 
        in_channel=35,  # This is the problematic channel count
        mlp=[64],
        bandwidth=0.1,
        group_all=False
    )
    model = model.cuda()
    print("  ✓ Model created and moved to CUDA")
except Exception as e:
    print(f"  ✗ Model creation failed: {e}")
    sys.exit(1)

# Create input tensors matching your error scenario
print("\n[2] Creating test input tensors...")
try:
    # Simulate the second layer from your error log:
    # "MLP layer 0 input shape: torch.Size([8, 35, 16, 64])"
    xyz = torch.randn(8, 3, 128).cuda()
    points = torch.randn(8, 32, 128).cuda()
    print(f"  xyz shape: {xyz.shape} on {xyz.device}")
    print(f"  points shape: {points.shape} on {points.device}")
except Exception as e:
    print(f"  ✗ Tensor creation failed: {e}")
    sys.exit(1)

# Run the forward pass - this is where it was crashing
print("\n[3] Running forward pass (this is where the crash occurred)...")
try:
    with torch.no_grad():
        new_xyz, new_points = model(xyz, points)
    print(f"  ✓ Forward pass completed successfully!")
    print(f"  Output shapes: xyz={new_xyz.shape}, points={new_points.shape}")
    print(f"  Output device: {new_xyz.device}")
except Exception as e:
    print(f"  ✗ Forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test with gradient computation
print("\n[4] Testing with gradient computation...")
try:
    xyz = torch.randn(8, 3, 128).cuda().requires_grad_(True)
    points = torch.randn(8, 32, 128).cuda().requires_grad_(True)
    
    new_xyz, new_points = model(xyz, points)
    loss = new_points.sum()
    loss.backward()
    
    print(f"  ✓ Backward pass completed successfully!")
    print(f"  Gradients computed: xyz.grad={xyz.grad is not None}, points.grad={points.grad is not None}")
except Exception as e:
    print(f"  ✗ Gradient computation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 70)
print("✓ ALL TESTS PASSED!")
print("=" * 70)
print("\nThe CPU fallback fix is working correctly.")
print("You can now run your full training code.")
print("\nNote: Training will be slower due to CPU fallback for Conv2d operations.")
print("Consider the alternative solutions in ILLEGAL_INSTRUCTION_FIX.md if speed is critical.")
