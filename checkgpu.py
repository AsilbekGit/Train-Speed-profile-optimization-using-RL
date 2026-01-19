#!/usr/bin/env python3
"""
GPU Check Script
================
Run this to verify GPU is available for DQN training

Usage:
    python check_gpu.py
"""

import sys

print("="*70)
print("GPU CHECK FOR DQN TRAINING")
print("="*70)

# Check PyTorch
try:
    import torch
    print(f"\n✓ PyTorch installed: {torch.__version__}")
except ImportError:
    print("\n✗ PyTorch NOT installed!")
    print("\nInstall PyTorch with CUDA support:")
    print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    print("\nOr for CUDA 12.1:")
    print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    sys.exit(1)

# Check CUDA
print(f"\n{'='*70}")
print("CUDA STATUS")
print("="*70)

if torch.cuda.is_available():
    print(f"✓ CUDA available: True")
    print(f"✓ CUDA version: {torch.version.cuda}")
    print(f"✓ cuDNN version: {torch.backends.cudnn.version()}")
    print(f"✓ Number of GPUs: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"\nGPU {i}: {props.name}")
        print(f"  Memory: {props.total_memory / 1e9:.1f} GB")
        print(f"  Compute Capability: {props.major}.{props.minor}")
        print(f"  Multi-processors: {props.multi_processor_count}")
    
    # Test GPU computation
    print(f"\n{'='*70}")
    print("GPU BENCHMARK")
    print("="*70)
    
    import time
    
    # Small matrix multiplication test
    size = 5000
    print(f"\nMatrix multiplication test ({size}x{size})...")
    
    # CPU
    a_cpu = torch.randn(size, size)
    b_cpu = torch.randn(size, size)
    
    start = time.time()
    c_cpu = torch.mm(a_cpu, b_cpu)
    cpu_time = time.time() - start
    print(f"  CPU time: {cpu_time*1000:.1f} ms")
    
    # GPU
    a_gpu = a_cpu.cuda()
    b_gpu = b_cpu.cuda()
    torch.cuda.synchronize()
    
    start = time.time()
    c_gpu = torch.mm(a_gpu, b_gpu)
    torch.cuda.synchronize()
    gpu_time = time.time() - start
    print(f"  GPU time: {gpu_time*1000:.1f} ms")
    print(f"  Speedup: {cpu_time/gpu_time:.1f}x")
    
    print(f"\n{'='*70}")
    print("✓ GPU is ready for DQN training!")
    print("="*70)
    print("\nRun DQN training with:")
    print("  python train_dqn.py --episodes 5000")
    
else:
    print("✗ CUDA NOT available!")
    print("\nPossible reasons:")
    print("  1. No NVIDIA GPU installed")
    print("  2. NVIDIA drivers not installed")
    print("  3. PyTorch installed without CUDA support")
    
    print("\nTo install PyTorch with CUDA:")
    print("  pip uninstall torch")
    print("  pip install torch --index-url https://download.pytorch.org/whl/cu118")
    
    print("\nDQN will run on CPU (slower but works)")

print()