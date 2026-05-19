"""
model.py
========
PointPillars 3D Object Detection Model.

Three components:
  1. PillarFeatureNetwork  — learns features from each pillar
  2. PointPillarsScatter   — creates the 2D pseudo-image
  3. Backbone + Head       — detects objects in pseudo-image

Author  : Vamshikrishna Gadde
Program : MS Robotics, Arizona State University
Series  : Day 3 of 90 — Perception Series
"""

import torch
import torch.nn as nn
from dataset import (
    POINT_CLOUD_RANGE, VOXEL_SIZE,
    MAX_PILLARS, MAX_POINTS, CLASSES
)

# ── PILLAR FEATURE NETWORK ────────────────────────────────────────────

class PillarFeatureNetwork(nn.Module):
    """
    Learns a 64-dimensional feature vector for each pillar.

    Input:  (B, MAX_PILLARS, MAX_POINTS, 9)
    Output: (B, MAX_PILLARS, 64)

    For each pillar:
      Run each point through Linear → BN → ReLU
      Max pool over all points → one feature per pillar
    """

    def __init__(self, in_channels=9, out_channels=64):
        super().__init__()
        self.out_channels = out_channels
        self.linear = nn.Linear(in_channels, out_channels, bias=False)
        self.bn     = nn.BatchNorm1d(out_channels)
        self.relu   = nn.ReLU()

    def forward(self, pillars, n_pillars):
        B, P, N, C = pillars.shape

        # Flatten to (B*P*N, C) for linear layer
        x = pillars.view(B * P * N, C)
        x = self.relu(self.bn(self.linear(x)))

        # Reshape back to (B, P, N, 64)
        x = x.view(B, P, N, self.out_channels)

        # Max pool over points: (B, P, 64)
        x = x.max(dim=2)[0]

        return x

# ── PILLAR SCATTER ────────────────────────────────────────────────────

class PointPillarsScatter(nn.Module):
    """
    Scatter pillar features into a 2D spatial grid.

    Creates a top-down pseudo-image from pillar features.
    Each pillar's 64 features go into its grid cell position.

    Input:  pillar_features (B, MAX_PILLARS, 64)
            pillar_coords   (B, MAX_PILLARS, 3)
    Output: pseudo-image    (B, 64, ny, nx)
    """

    def __init__(self):
        super().__init__()
        x_min, y_min, z_min, x_max, y_max, z_max = POINT_CLOUD_RANGE
        vx, vy, _ = VOXEL_SIZE
        self.nx = int((x_max - x_min) / vx)
        self.ny = int((y_max - y_min) / vy)

    def forward(self, pillar_features, pillar_coords, n_pillars):
        B = pillar_features.shape[0]
        C = pillar_features.shape[2]

        canvas = torch.zeros(
            B, C, self.ny, self.nx,
            dtype=pillar_features.dtype,
            device=pillar_features.device
        )

        for b in range(B):
            n      = n_pillars[b]
            coords = pillar_coords[b, :n]
            feats  = pillar_features[b, :n]
            y_idx  = coords[:, 1]
            x_idx  = coords[:, 2]
            canvas[b, :, y_idx, x_idx] = feats.t()

        return canvas

# ── 2D CNN BACKBONE ───────────────────────────────────────────────────

class Backbone(nn.Module):
    """
    2D CNN that processes the pseudo-image.

    Three blocks of convolutions at different scales.
    Upsamples and concatenates for multi-scale detection.

    Output channels: 384 (128 x 3 scales)
    """

    def __init__(self, in_channels=64):
        super().__init__()

        def block(in_ch, out_ch, stride=1):
            return nn.Sequential(
                nn.Conv2d(
                    in_ch, out_ch, 3,
                    stride=stride, padding=1, bias=False
                ),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        self.block1 = nn.Sequential(
            block(in_channels, 64, stride=2),
            block(64, 64),
            block(64, 64),
        )
        self.block2 = nn.Sequential(
            block(64, 128, stride=2),
            block(128, 128),
            block(128, 128),
        )
        self.block3 = nn.Sequential(
            block(128, 256, stride=2),
            block(256, 256),
            block(256, 256),
        )

        self.up1 = nn.ConvTranspose2d(64,  128, 2, stride=2)
        self.up2 = nn.ConvTranspose2d(128, 128, 2, stride=2)
        self.up3 = nn.ConvTranspose2d(256, 128, 4, stride=4)

        self.out_channels = 384

    def forward(self, x):
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)

        u1 = self.up1(x1)
        u2 = self.up2(x2)
        u3 = self.up3(x3)

        h = min(u1.shape[2], u2.shape[2], u3.shape[2])
        w = min(u1.shape[3], u2.shape[3], u3.shape[3])

        u1 = u1[:, :, :h, :w]
        u2 = u2[:, :, :h, :w]
        u3 = u3[:, :, :h, :w]

        return torch.cat([u1, u2, u3], dim=1)

# ── DETECTION HEAD ────────────────────────────────────────────────────

class DetectionHead(nn.Module):
    """
    Predicts 3D bounding boxes from feature map.

    For each grid cell and anchor predicts:
      cls_preds : object class scores
      box_preds : 3D box offsets (dx,dy,dz,dw,dl,dh,heading)
      dir_preds : heading direction (forward or backward)
    """

    def __init__(self, in_channels=384, n_classes=3, n_anchors=2):
        super().__init__()
        self.cls_head = nn.Conv2d(
            in_channels, n_anchors * n_classes, 1
        )
        self.box_head = nn.Conv2d(
            in_channels, n_anchors * 7, 1
        )
        self.dir_head = nn.Conv2d(
            in_channels, n_anchors * 2, 1
        )

    def forward(self, x):
        return {
            'cls_preds': self.cls_head(x),
            'box_preds': self.box_head(x),
            'dir_preds': self.dir_head(x),
        }

# ── FULL MODEL ────────────────────────────────────────────────────────

class PointPillars(nn.Module):
    """
    Full PointPillars 3D Object Detection Model.

    Pipeline:
      Pillars → PFN → Scatter → Backbone → Head → Predictions
    """

    def __init__(self):
        super().__init__()
        self.pfn      = PillarFeatureNetwork(9, 64)
        self.scatter  = PointPillarsScatter()
        self.backbone = Backbone(64)
        self.head     = DetectionHead(
            self.backbone.out_channels, len(CLASSES), 2
        )

    def forward(self, pillars, pillar_coords, n_pillars):
        feats      = self.pfn(pillars, n_pillars)
        pseudo_img = self.scatter(feats, pillar_coords, n_pillars)
        features   = self.backbone(pseudo_img)
        return self.head(features)

    def count_parameters(self):
        return sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )

# ── TEST ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  PointPillars Model Test")
    print("="*60)

    model  = PointPillars()
    params = model.count_parameters()

    print(f"\n  Parameters : {params:,}")
    print(f"  Model size : ~{params * 4 / 1024 / 1024:.1f} MB")

    B             = 2
    pillars       = torch.randn(B, MAX_PILLARS, MAX_POINTS, 9)
    pillar_coords = torch.zeros(B, MAX_PILLARS, 3, dtype=torch.int32)
    n_pillars     = [100, 120]

    print(f"\n  Input shape: {pillars.shape}")
    print(f"  Running forward pass...")

    with torch.no_grad():
        out = model(pillars, pillar_coords, n_pillars)

    print(f"\n  Output shapes:")
    for k, v in out.items():
        print(f"    {k:15s}: {v.shape}")

    print(f"\n  Model working correctly!")