"""
dataset.py
==========
KITTI 3D Object Detection Dataset Loader.

What this does:
  Reads KITTI LiDAR .bin files
  Reads KITTI 3D label .txt files
  Reads calibration files
  Converts raw data into pillars for PointPillars

KITTI label format (one line per object):
  Class Truncated Occluded Alpha
  Left Top Right Bottom
  Height Width Length X Y Z RotationY

Author  : Vamshikrishna Gadde
Program : MS Robotics, Arizona State University
Series  : Day 3 of 90 — Perception Series
"""

import numpy as np
import torch
from torch.utils.data import Dataset
import os

# ── CONSTANTS ─────────────────────────────────────────────────────────

CLASSES     = ['Car', 'Pedestrian', 'Cyclist']
CLASS_TO_ID = {c: i for i, c in enumerate(CLASSES)}

POINT_CLOUD_RANGE = [0, -39.68, -3, 69.12, 39.68, 1]
VOXEL_SIZE        = [0.16, 0.16, 4]
MAX_POINTS        = 32
MAX_PILLARS       = 12000

# ── KITTI ROOT PATH ───────────────────────────────────────────────────

KITTI_ROOT = r"D:\kitti\kitti_object"

# ── CALIBRATION ───────────────────────────────────────────────────────

def load_calibration(calib_path):
    """
    Load KITTI calibration file.
    Returns P2 (camera projection), R0 (rectification), V2C (velo to cam).
    """
    data = {}
    with open(calib_path, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            if ':' in line:
                key, value = line.split(':', 1)
                data[key.strip()] = np.array(
                    value.strip().split(), dtype=np.float32
                )

    P2  = data['P2'].reshape(3, 4)
    R0  = data['R0_rect'].reshape(3, 3)
    V2C = data['Tr_velo_to_cam'].reshape(3, 4)

    V2C_4x4          = np.eye(4, dtype=np.float32)
    V2C_4x4[:3, :]   = V2C

    R0_4x4            = np.eye(4, dtype=np.float32)
    R0_4x4[:3, :3]    = R0

    return P2, R0_4x4, V2C_4x4

# ── LABELS ────────────────────────────────────────────────────────────

def load_labels(label_path):
    """
    Load KITTI 3D object labels.
    Returns list of objects with class and 3D box info.
    """
    objects = []
    if not os.path.exists(label_path):
        return objects

    with open(label_path, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) < 15:
                continue

            class_name = parts[0]
            if class_name == 'Van':
                class_name = 'Car'
            if class_name not in CLASSES:
                continue

            objects.append({
                'class_name': class_name,
                'class_id':   CLASS_TO_ID[class_name],
                'truncated':  float(parts[1]),
                'occluded':   int(parts[2]),
                'box2d':      np.array(parts[4:8],  dtype=np.float32),
                'box3d':      np.array(parts[8:15], dtype=np.float32),
            })

    return objects

# ── POINT CLOUD ───────────────────────────────────────────────────────

def load_pointcloud(lidar_path):
    """Load KITTI velodyne .bin file. Returns (N, 4) array."""
    return np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)

def filter_pointcloud(pts):
    """Keep only points inside detection range."""
    x_min, y_min, z_min, x_max, y_max, z_max = POINT_CLOUD_RANGE
    mask = (
        (pts[:, 0] >= x_min) & (pts[:, 0] <= x_max) &
        (pts[:, 1] >= y_min) & (pts[:, 1] <= y_max) &
        (pts[:, 2] >= z_min) & (pts[:, 2] <= z_max)
    )
    return pts[mask]

# ── PILLARIZATION ─────────────────────────────────────────────────────

def create_pillars(pts):
    """
    Convert point cloud into pillar representation.

    Divides the ground plane into a grid.
    Each grid cell = one pillar (vertical column).
    Points inside each pillar are encoded as 9 features.

    Returns:
      pillars       : (MAX_PILLARS, MAX_POINTS, 9)
      pillar_coords : (MAX_PILLARS, 3)
      n_pillars     : int
    """
    x_min, y_min, z_min, x_max, y_max, z_max = POINT_CLOUD_RANGE
    vx, vy, vz = VOXEL_SIZE

    nx = int((x_max - x_min) / vx)
    ny = int((y_max - y_min) / vy)

    # Assign each point to a pillar
    ix = np.clip(
        np.floor((pts[:, 0] - x_min) / vx).astype(int), 0, nx - 1
    )
    iy = np.clip(
        np.floor((pts[:, 1] - y_min) / vy).astype(int), 0, ny - 1
    )

    pillar_idx = ix * ny + iy

    # Group points by pillar
    pillars_dict = {}
    for i, idx in enumerate(pillar_idx):
        if idx not in pillars_dict:
            pillars_dict[idx] = []
        if len(pillars_dict[idx]) < MAX_POINTS:
            pillars_dict[idx].append(i)

    valid_pillars = list(pillars_dict.items())[:MAX_PILLARS]
    n_pillars     = len(valid_pillars)

    pillars       = np.zeros(
        (MAX_PILLARS, MAX_POINTS, 9), dtype=np.float32
    )
    pillar_coords = np.zeros((MAX_PILLARS, 3), dtype=np.int32)

    for p_idx, (grid_idx, point_indices) in enumerate(valid_pillars):
        gx = grid_idx // ny
        gy = grid_idx %  ny

        pillar_coords[p_idx] = [0, gy, gx]

        px_center = x_min + (gx + 0.5) * vx
        py_center = y_min + (gy + 0.5) * vy

        for i, pt_idx in enumerate(point_indices):
            x, y, z, r = pts[pt_idx]
            pillars[p_idx, i] = [
                x, y, z, r,
                x - px_center,
                y - py_center,
                z - z_min,
                np.sqrt(x**2 + y**2),
                (z_min + z_max) / 2,
            ]

    return pillars, pillar_coords, n_pillars

# ── DATASET CLASS ─────────────────────────────────────────────────────

class KITTIDataset(Dataset):
    """
    PyTorch Dataset for KITTI 3D Object Detection.

    Loads LiDAR point clouds, converts to pillars,
    and returns with 3D object labels.
    """

    def __init__(self, root_dir, split='train'):
        self.root_dir  = root_dir
        self.lidar_dir = os.path.join(root_dir, 'training', 'velodyne')
        self.label_dir = os.path.join(root_dir, 'training', 'label_2')
        self.calib_dir = os.path.join(root_dir, 'training', 'calib')

        all_files = sorted([
            f.replace('.bin', '')
            for f in os.listdir(self.lidar_dir)
            if f.endswith('.bin')
        ])

        split_idx = int(len(all_files) * 0.8)
        if split == 'train':
            self.frames = all_files[:split_idx]
        else:
            self.frames = all_files[split_idx:]

        print(f"  Dataset ({split}): {len(self.frames)} frames")

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frame_id   = self.frames[idx]
        lidar_path = os.path.join(
            self.lidar_dir, frame_id + '.bin'
        )
        label_path = os.path.join(
            self.label_dir, frame_id + '.txt'
        )
        calib_path = os.path.join(
            self.calib_dir, frame_id + '.txt'
        )

        pts    = filter_pointcloud(load_pointcloud(lidar_path))
        labels = load_labels(label_path)

        pillars, pillar_coords, n_pillars = create_pillars(pts)

        return {
            'frame_id':      frame_id,
            'pillars':       torch.from_numpy(pillars),
            'pillar_coords': torch.from_numpy(pillar_coords),
            'n_pillars':     n_pillars,
            'labels':        labels,
            'n_points':      len(pts),
        }

# ── TEST ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  KITTI Dataset Test")
    print("="*60)

    if not os.path.exists(KITTI_ROOT):
        print(f"\n  Data not found at: {KITTI_ROOT}")
        print(f"  Check your path.")
    else:
        dataset = KITTIDataset(KITTI_ROOT, split='train')
        sample  = dataset[0]

        print(f"\n  Frame ID     : {sample['frame_id']}")
        print(f"  Points       : {sample['n_points']:,}")
        print(f"  Pillars      : {sample['n_pillars']}")
        print(f"  Pillar shape : {sample['pillars'].shape}")
        print(f"  Objects      : {len(sample['labels'])}")

        for obj in sample['labels'][:3]:
            print(f"    {obj['class_name']:12s} "
                  f"box={obj['box3d']}")

        print("\n  Dataset working correctly!")