#!/usr/bin/env python
"""
Test script to diagnose the Illegal instruction error in Conv2d operations.
This will help identify if the issue is with specific tensor sizes, operations, or CPU instructions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os

print("=" * 60)
print("PyTorch Diagnostic Test")
print("=" * 60)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
print(f"CPU threads: {torch.get_num_threads()}")
print("=" * 60)

# Test 1: Simple Conv2d operation
print("\n[Test 1] Simple Conv2d operation...")
try:
    conv = nn.Conv2d(6, 32, 1)
    x = torch.randn(8, 6, 8, 128)
    print(f"  Input: {x.shape}, dtype: {x.dtype}, contiguous: {x.is_contiguous()}")
    
    # Try with CPU explicitly
    conv_cpu = conv.cpu()
    x_cpu = x.cpu()
    out = conv_cpu(x_cpu)
    print(f"  ✓ Conv2d passed: {out.shape}")
except Exception as e:
    print(f"  ✗ Conv2d failed: {e}")

# Test 2: Conv2d + GroupNorm
print("\n[Test 2] Conv2d + GroupNorm...")
try:
    conv = nn.Conv2d(6, 32, 1).cpu()
    bn = nn.GroupNorm(1, 32).cpu()
    x = torch.randn(8, 6, 8, 128).cpu()
    
    out = conv(x)
    print(f"  After conv: {out.shape}")
    out = bn(out)
    print(f"  ✓ Conv2d + GroupNorm passed: {out.shape}")
except Exception as e:
    print(f"  ✗ Conv2d + GroupNorm failed: {e}")

# Test 3: Full pipeline (Conv2d + GroupNorm + ReLU)
print("\n[Test 3] Conv2d + GroupNorm + ReLU...")
try:
    conv = nn.Conv2d(6, 32, 1).cpu()
    bn = nn.GroupNorm(1, 32).cpu()
    x = torch.randn(8, 6, 8, 128).cpu()
    
    out = F.relu(bn(conv(x)))
    print(f"  ✓ Full pipeline passed: {out.shape}")
except Exception as e:
    print(f"  ✗ Full pipeline failed: {e}")

# Test 4: Exact shapes from your error (second layer)
print("\n[Test 4] Exact shapes from error log (35 channels)...")
try:
    conv = nn.Conv2d(35, 64, 1).cpu()
    bn = nn.GroupNorm(1, 64).cpu()
    x = torch.randn(8, 35, 16, 64).cpu()
    
    print(f"  Input: {x.shape}, dtype: {x.dtype}, contiguous: {x.is_contiguous()}")
    
    # Make sure tensor is contiguous
    x = x.contiguous()
    
    out = conv(x)
    print(f"  After conv: {out.shape}")
    out = bn(out)
    print(f"  After BN: {out.shape}")
    out = F.relu(out)
    print(f"  ✓ Exact shape test passed: {out.shape}")
except Exception as e:
    print(f"  ✗ Exact shape test failed: {e}")
    import traceback
    traceback.print_exc()

# Test 5: Try with different number of threads
print("\n[Test 5] Testing with reduced CPU threads...")
for num_threads in [1, 2, 4]:
    try:
        torch.set_num_threads(num_threads)
        print(f"  Testing with {num_threads} thread(s)...")
        
        conv = nn.Conv2d(35, 64, 1).cpu()
        bn = nn.GroupNorm(1, 64).cpu()
        x = torch.randn(8, 35, 16, 64).cpu().contiguous()
        
        out = F.relu(bn(conv(x)))
        print(f"    ✓ Passed with {num_threads} thread(s): {out.shape}")
        break  # If successful, stop testing
    except Exception as e:
        print(f"    ✗ Failed with {num_threads} thread(s): {e}")

# Test 6: Try disabling MKL (if using Intel MKL)
print("\n[Test 6] Testing with MKL-DNN disabled...")
try:
    # Disable MKL-DNN
    torch.backends.mkldnn.enabled = False
    print(f"  MKL-DNN disabled: {not torch.backends.mkldnn.enabled}")
    
    conv = nn.Conv2d(35, 64, 1).cpu()
    bn = nn.GroupNorm(1, 64).cpu()
    x = torch.randn(8, 35, 16, 64).cpu().contiguous()
    
    out = F.relu(bn(conv(x)))
    print(f"  ✓ Test passed with MKL-DNN disabled: {out.shape}")
except Exception as e:
    print(f"  ✗ Test failed: {e}")

print("\n" + "=" * 60)
print("Diagnostic test completed")
print("=" * 60)
