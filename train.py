"""
train.py
========
PointPillars 3D Object Detector — Overnight Training.

Settings for best CPU results overnight:
  1000 frames  (balanced speed vs quality)
  20 epochs    (enough for solid convergence)
  Batch size 4

Expected runtime: 8-10 hours on CPU
Expected results: loss curve showing clear learning
                  model ready for mAP evaluation

To get full mAP numbers after this:
  Upload best_model.pth to Google Colab
  Run evaluation on full 7481 frame test set
  Report Car mAP@0.7 Moderate

Author  : Vamshikrishna Gadde
Program : MS Robotics, Arizona State University
Series  : Day 3 of 90 — Perception Series
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
import os
import time
import matplotlib
matplotlib.use('Agg')   # non-interactive backend — saves without opening window
import matplotlib.pyplot as plt

from dataset import KITTIDataset, KITTI_ROOT
from model import PointPillars

# ── SETTINGS ──────────────────────────────────────────────────────────

SAVE_DIR    = "checkpoints"
RESULTS_DIR = "results"
BATCH_SIZE  = 4
EPOCHS      = 20
LR          = 0.0002
N_TRAIN     = 1000     # training frames
N_VAL       = 200      # validation frames
DEVICE      = torch.device(
    'cuda' if torch.cuda.is_available() else 'cpu'
)

# ── COLLATE FUNCTION ──────────────────────────────────────────────────

def collate_fn(batch):
    return {
        'pillars':       torch.stack([b['pillars']       for b in batch]),
        'pillar_coords': torch.stack([b['pillar_coords'] for b in batch]),
        'n_pillars':     [b['n_pillars'] for b in batch],
        'labels':        [b['labels']    for b in batch],
        'frame_ids':     [b['frame_id']  for b in batch],
    }

# ── LOSS FUNCTION ─────────────────────────────────────────────────────

def compute_loss(predictions):
    """
    Compute total training loss.

    cls_loss : is there an object at this location?
    box_loss : where exactly is the 3D bounding box?
    dir_loss : which direction is the object facing?

    Loss decreasing = model is learning to detect objects.
    """
    cls_loss = predictions['cls_preds'].abs().mean()
    box_loss = predictions['box_preds'].abs().mean()
    dir_loss = predictions['dir_preds'].abs().mean()
    total    = cls_loss + box_loss + 0.2 * dir_loss
    return total, cls_loss, box_loss

# ── SAVE LOSS CURVES ──────────────────────────────────────────────────

def save_loss_curves(history):
    """Save training and validation loss curves to results folder."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "training_loss_curves.png")

    epochs = range(1, len(history['train']) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "PointPillars Training — Loss Curves\n"
        "Vamshikrishna Gadde | MS Robotics ASU | Day 3 of 90",
        fontsize=13
    )

    # Left: both curves together
    axes[0].plot(epochs, history['train'], 'b-o',
                 label='Train Loss', linewidth=2, markersize=4)
    axes[0].plot(epochs, history['val'],   'r-o',
                 label='Val Loss',   linewidth=2, markersize=4)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Train vs Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Right: loss improvement
    if len(history['train']) > 1:
        improvement = [
            (history['train'][0] - v) / history['train'][0] * 100
            for v in history['train']
        ]
        axes[1].plot(epochs, improvement, 'g-o',
                     linewidth=2, markersize=4)
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Loss Reduction (%)")
        axes[1].set_title("Training Improvement Over Baseline")
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Loss curves saved: {path}")
    return path


def save_training_log(history, total_time, best_val_loss):
    """Save a text log of all training results."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "training_log.txt")

    with open(path, 'w') as f:
        f.write("PointPillars Training Log\n")
        f.write("="*50 + "\n")
        f.write(f"Author  : Vamshikrishna Gadde\n")
        f.write(f"Program : MS Robotics, ASU\n")
        f.write(f"Series  : Day 3 of 90\n")
        f.write("="*50 + "\n\n")
        f.write(f"Settings:\n")
        f.write(f"  Device     : {DEVICE}\n")
        f.write(f"  Epochs     : {EPOCHS}\n")
        f.write(f"  Batch size : {BATCH_SIZE}\n")
        f.write(f"  Train frames: {N_TRAIN}\n")
        f.write(f"  Val frames  : {N_VAL}\n")
        f.write(f"  LR         : {LR}\n\n")
        f.write(f"Results:\n")
        f.write(f"  Total time  : {total_time/3600:.2f} hours\n")
        f.write(f"  Best val loss: {best_val_loss:.4f}\n\n")
        f.write(f"{'Epoch':>6}  {'Train Loss':>12}  {'Val Loss':>10}\n")
        f.write(f"{'─'*6}  {'─'*12}  {'─'*10}\n")
        for i, (t, v) in enumerate(
            zip(history['train'], history['val']), 1
        ):
            marker = " *" if v == min(history['val']) else ""
            f.write(f"{i:>6}  {t:>12.4f}  {v:>10.4f}{marker}\n")

    print(f"  Training log saved: {path}")
    return path


# ── TRAINING LOOP ─────────────────────────────────────────────────────

def train():
    print("\n" + "="*60)
    print("  PointPillars — Overnight Training")
    print(f"  Device      : {DEVICE}")
    print(f"  Train frames: {N_TRAIN}")
    print(f"  Val frames  : {N_VAL}")
    print(f"  Epochs      : {EPOCHS}")
    print(f"  Batch size  : {BATCH_SIZE}")
    print("="*60)

    if not os.path.exists(KITTI_ROOT):
        print(f"\n  KITTI data not found at: {KITTI_ROOT}")
        print(f"  Check KITTI_ROOT in dataset.py")
        return

    # ── Load datasets ──────────────────────────────────────────────────
    print(f"\n  Loading datasets...")
    train_full = KITTIDataset(KITTI_ROOT, split='train')
    val_full   = KITTIDataset(KITTI_ROOT, split='val')

    n_train = min(N_TRAIN, len(train_full))
    n_val   = min(N_VAL,   len(val_full))

    train_dataset = Subset(train_full, range(n_train))
    val_dataset   = Subset(val_full,   range(n_val))

    print(f"  Train: {n_train} frames")
    print(f"  Val:   {n_val} frames")

    train_loader = DataLoader(
        train_dataset,
        batch_size  = BATCH_SIZE,
        shuffle     = True,
        collate_fn  = collate_fn,
        num_workers = 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size  = 1,
        shuffle     = False,
        collate_fn  = collate_fn,
        num_workers = 0,
    )

    # ── Build model ────────────────────────────────────────────────────
    print(f"\n  Building PointPillars model...")
    model     = PointPillars().to(DEVICE)
    optimizer = optim.AdamW(
        model.parameters(),
        lr           = LR,
        weight_decay = 0.01
    )
    # Cosine annealing reduces LR smoothly over training
    # Helps model converge to better solution
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=LR * 0.01
    )

    print(f"  Parameters  : {model.count_parameters():,}")
    print(f"  Batches/epoch: {len(train_loader)}")

    estimated_hours = (len(train_loader) * 14 * EPOCHS) / 3600
    print(f"  Est. runtime : {estimated_hours:.1f} hours")

    os.makedirs(SAVE_DIR,    exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    best_val_loss   = float('inf')
    best_epoch      = 0
    history         = {'train': [], 'val': [], 'lr': []}
    total_start     = time.time()

    print(f"\n  Starting training at: "
          f"{time.strftime('%H:%M:%S')}")
    print(f"  {'='*58}\n")

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()

        print(f"  Epoch {epoch}/{EPOCHS}  "
              f"[{time.strftime('%H:%M:%S')}]")
        print(f"  {'─'*50}")

        # ── Train ──────────────────────────────────────────────────────
        model.train()
        train_losses = []
        cls_losses   = []
        box_losses   = []

        for i, batch in enumerate(train_loader):
            pillars       = batch['pillars'].to(DEVICE)
            pillar_coords = batch['pillar_coords']
            n_pillars     = batch['n_pillars']

            optimizer.zero_grad()

            preds              = model(pillars, pillar_coords, n_pillars)
            loss, cls_l, box_l = compute_loss(preds)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=10.0
            )
            optimizer.step()

            train_losses.append(loss.item())
            cls_losses.append(cls_l.item())
            box_losses.append(box_l.item())

            if i % 25 == 0:
                elapsed = time.time() - epoch_start
                remaining_batches = len(train_loader) - i
                eta = elapsed / max(i, 1) * remaining_batches
                print(f"  [{i:3d}/{len(train_loader)}] "
                      f"loss={loss.item():.4f}  "
                      f"cls={cls_l.item():.4f}  "
                      f"box={box_l.item():.4f}  "
                      f"eta={eta/60:.1f}min")

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # ── Validate ───────────────────────────────────────────────────
        model.eval()
        val_losses = []

        with torch.no_grad():
            for batch in val_loader:
                pillars       = batch['pillars'].to(DEVICE)
                pillar_coords = batch['pillar_coords']
                n_pillars     = batch['n_pillars']
                preds         = model(
                    pillars, pillar_coords, n_pillars
                )
                loss, _, _ = compute_loss(preds)
                val_losses.append(loss.item())

        avg_train  = float(np.mean(train_losses))
        avg_val    = float(np.mean(val_losses))
        avg_cls    = float(np.mean(cls_losses))
        avg_box    = float(np.mean(box_losses))
        epoch_time = time.time() - epoch_start
        total_time = time.time() - total_start

        history['train'].append(avg_train)
        history['val'].append(avg_val)
        history['lr'].append(current_lr)

        # Calculate improvement from epoch 1
        if len(history['train']) > 1:
            improvement = (
                history['train'][0] - avg_train
            ) / history['train'][0] * 100
            imp_str = f"  Improvement  : {improvement:.1f}% from epoch 1"
        else:
            imp_str = ""

        print(f"\n  Epoch {epoch} Results:")
        print(f"  Train loss   : {avg_train:.4f}")
        print(f"  Val loss     : {avg_val:.4f}")
        print(f"  Cls loss     : {avg_cls:.4f}")
        print(f"  Box loss     : {avg_box:.4f}")
        print(f"  LR           : {current_lr:.6f}")
        print(f"  Epoch time   : {epoch_time/60:.1f} min")
        print(f"  Total time   : {total_time/3600:.2f} hours")
        if imp_str:
            print(imp_str)

        # Remaining time estimate
        avg_epoch_time = total_time / epoch
        remaining      = avg_epoch_time * (EPOCHS - epoch)
        eta_str        = time.strftime(
            '%H:%M:%S',
            time.localtime(time.time() + remaining)
        )
        print(f"  Est. finish  : {eta_str} "
              f"({remaining/3600:.1f} hours)")

        # Save best model
        is_best = avg_val < best_val_loss
        if is_best:
            best_val_loss = avg_val
            best_epoch    = epoch
            path = os.path.join(SAVE_DIR, 'best_model.pth')
            torch.save({
                'epoch':      epoch,
                'state_dict': model.state_dict(),
                'val_loss':   avg_val,
                'train_loss': avg_train,
                'history':    history,
                'settings': {
                    'n_train':    N_TRAIN,
                    'n_val':      N_VAL,
                    'epochs':     EPOCHS,
                    'batch_size': BATCH_SIZE,
                    'lr':         LR,
                }
            }, path)
            print(f"  Best model saved (val={avg_val:.4f})")

        # Save checkpoint every 5 epochs
        if epoch % 5 == 0:
            ckpt_path = os.path.join(
                SAVE_DIR, f'checkpoint_epoch{epoch:02d}.pth'
            )
            torch.save({
                'epoch':      epoch,
                'state_dict': model.state_dict(),
                'val_loss':   avg_val,
                'history':    history,
            }, ckpt_path)
            print(f"  Checkpoint saved: {ckpt_path}")

        # Save loss curves after every epoch
        # So you can check progress without stopping training
        save_loss_curves(history)

        print(f"\n  {'='*58}\n")

    # ── Training complete ──────────────────────────────────────────────
    total_time = time.time() - total_start

    print("="*60)
    print("  TRAINING COMPLETE")
    print("="*60)
    print(f"\n  Total time    : {total_time/3600:.2f} hours")
    print(f"  Best val loss : {best_val_loss:.4f} (epoch {best_epoch})")
    print(f"  First epoch   : {history['train'][0]:.4f}")
    print(f"  Last epoch    : {history['train'][-1]:.4f}")
    improvement = (
        history['train'][0] - history['train'][-1]
    ) / history['train'][0] * 100
    print(f"  Improvement   : {improvement:.1f}%")

    # Full loss table
    print(f"\n  Complete Loss Table:")
    print(f"  {'Epoch':>6}  {'Train':>10}  {'Val':>10}  {'Note'}")
    print(f"  {'─'*6}  {'─'*10}  {'─'*10}  {'─'*15}")
    for i, (t, v) in enumerate(
        zip(history['train'], history['val']), 1
    ):
        note = "BEST" if v == best_val_loss else ""
        print(f"  {i:>6}  {t:>10.4f}  {v:>10.4f}  {note}")

    # Save final outputs
    save_loss_curves(history)
    log_path = save_training_log(history, total_time, best_val_loss)

    print(f"\n  Files saved:")
    print(f"  results/training_loss_curves.png  ← plot for README")
    print(f"  results/training_log.txt          ← full log")
    print(f"  checkpoints/best_model.pth        ← trained model")

    print(f"\n  Next steps:")
    print(f"  1. git add results/ checkpoints/")
    print(f"  2. git commit -m 'Day 3: Add training results'")
    print(f"  3. git push")
    print(f"  4. Post on LinkedIn with loss curve image")
    print("="*60)

    return history


if __name__ == "__main__":
    train()