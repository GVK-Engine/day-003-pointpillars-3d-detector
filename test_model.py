"""
test_model.py
=============
Test PointPillars model architecture and dataset loading.
No GPU required. Runs on CPU.

Run this FIRST before training to verify everything works.

Author  : Vamshikrishna Gadde
Program : MS Robotics, Arizona State University
Series  : Day 3 of 90 — Perception Series
"""

import torch
import os
from model import PointPillars
from dataset import (
    MAX_PILLARS, MAX_POINTS, KITTI_ROOT, KITTIDataset
)

print("\n" + "="*60)
print("  PointPillars — Architecture + Dataset Test")
print("="*60)

# ── TEST 1: MODEL ARCHITECTURE ────────────────────────────────────────

print("\n  TEST 1: Model Architecture")
print("  " + "─"*40)

model  = PointPillars()
params = model.count_parameters()

print(f"  Model built successfully")
print(f"  Total parameters : {params:,}")
print(f"  Model size       : ~{params * 4 / 1024 / 1024:.1f} MB")

# Dummy forward pass
B             = 2
pillars       = torch.randn(B, MAX_PILLARS, MAX_POINTS, 9)
pillar_coords = torch.zeros(B, MAX_PILLARS, 3, dtype=torch.int32)
n_pillars     = [150, 180]

print(f"\n  Input shape: {pillars.shape}")

with torch.no_grad():
    out = model(pillars, pillar_coords, n_pillars)

print(f"  Output shapes:")
for k, v in out.items():
    print(f"    {k:15s}: {v.shape}")

print(f"\n  TEST 1 PASSED — Model architecture works!")

# ── TEST 2: DATASET LOADING ───────────────────────────────────────────

print(f"\n  TEST 2: Dataset Loading")
print("  " + "─"*40)

if not os.path.exists(KITTI_ROOT):
    print(f"  KITTI data not found at: {KITTI_ROOT}")
    print(f"  Skipping dataset test.")
else:
    dataset = KITTIDataset(KITTI_ROOT, split='train')
    sample  = dataset[0]

    print(f"  Frame ID      : {sample['frame_id']}")
    print(f"  Points loaded : {sample['n_points']:,}")
    print(f"  Pillars       : {sample['n_pillars']}")
    print(f"  Pillar shape  : {sample['pillars'].shape}")
    print(f"  Objects found : {len(sample['labels'])}")

    for obj in sample['labels'][:5]:
        h, w, l = obj['box3d'][0], obj['box3d'][1], obj['box3d'][2]
        x, y, z = obj['box3d'][3], obj['box3d'][4], obj['box3d'][5]
        print(f"    {obj['class_name']:12s} "
              f"pos=({x:.1f},{y:.1f},{z:.1f}) "
              f"size={l:.1f}x{w:.1f}x{h:.1f}m")

    print(f"\n  TEST 2 PASSED — Dataset loading works!")

# ── SUMMARY ───────────────────────────────────────────────────────────

print(f"\n" + "="*60)
print(f"  ALL TESTS PASSED")
print(f"  Ready to train with: py -3.11 train.py")
print("="*60 + "\n")