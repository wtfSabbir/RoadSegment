"""
SegFormer Training Script: 6-Channel Curb-Gutter Segmentation
================================================================
Step 3 of the curb-gutter segmentation pipeline.

What this script does:
  1. Loads the patch dataset prepared by prepare_dataset.py
  2. Modifies SegFormer-B2's first conv layer to accept 6 channels
     (HAG, verticality, intensity, R, G, B) while keeping pretrained
     weights for the RGB channels intact
  3. Trains with weighted CrossEntropyLoss (handles class imbalance)
  4. Uses gradient accumulation to simulate larger batch size on 8GB VRAM
  5. Saves checkpoints, logs to TensorBoard, tracks best validation mIoU
  6. Includes data augmentation suited for top-down LiDAR rasters

Usage:
  python train_segformer.py \\
    --data_dir   /path/to/your/dataset/ \\
    --output_dir /path/to/checkpoints/ \\
    --epochs 100 \\
    --lr 6e-5 \\
    --batch_size 1 \\
    --grad_accum_steps 8 \\
    --model_size b2

Dependencies:
  pip install torch torchvision transformers accelerate
  pip install scikit-learn opencv-python tqdm tensorboard
"""

import argparse
import os
import json
import random
import numpy as np
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from transformers import SegformerForSemanticSegmentation, SegformerConfig
from model_arch import build_model, CHANNEL_ORDER

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x


# ─── Dataset ───────────────────────────────────────────────────────────────

class CurbGutterDataset(Dataset):
    """
    Loads 6-channel .npy image patches and corresponding .png label masks.

    Augmentations applied (train split only):
      - Random horizontal flip
      - Random vertical flip
      - Random 90-degree rotation
      - Random brightness jitter on RGB channels only
    """

    def __init__(self, data_dir, split='train', augment=None):
        self.data_dir = Path(data_dir)
        self.split    = split
        self.augment  = augment if augment is not None else (split == 'train')

        self.image_dir = self.data_dir / 'images' / split
        self.mask_dir  = self.data_dir / 'masks'  / split

        self.image_files = sorted(self.image_dir.glob("*.npy"))
        if len(self.image_files) == 0:
            raise RuntimeError(f"No .npy files found in {self.image_dir}")

        print(f"[{split}] Loaded {len(self.image_files)} patches")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        import cv2

        img_path  = self.image_files[idx]
        mask_path = self.mask_dir / (img_path.stem + ".png")

        image = np.load(img_path).astype(np.float32)        # (H, W, 6)
        mask  = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)  # (H, W) uint8

        if self.augment:
            image, mask = self._augment(image, mask)

        # To CHW tensor
        image_t = torch.from_numpy(image.transpose(2, 0, 1).copy())  # (6, H, W)
        mask_t  = torch.from_numpy(mask.copy()).long()               # (H, W)

        return image_t, mask_t

    def _augment(self, image, mask):
        # Horizontal flip
        if random.random() < 0.5:
            image = np.flip(image, axis=1)
            mask  = np.flip(mask,  axis=1)

        # Vertical flip
        if random.random() < 0.5:
            image = np.flip(image, axis=0)
            mask  = np.flip(mask,  axis=0)

        # 90-degree rotation
        k = random.choice([0, 1, 2, 3])
        if k > 0:
            image = np.rot90(image, k, axes=(0, 1))
            mask  = np.rot90(mask,  k, axes=(0, 1))

        # Brightness jitter on RGB channels (indices 3, 4, 5) only
        if random.random() < 0.3:
            factor = random.uniform(0.85, 1.15)
            image = image.copy()
            image[:, :, 3:6] = np.clip(image[:, :, 3:6] * factor, 0, 1)

        return image.copy(), mask.copy()


# ─── Model setup ───────────────────────────────────────────────────────────
# build_model() now lives in model_arch.py — the SAME file inference.py
# imports from. This guarantees training and inference always build the
# identical architecture. Do not redefine build_model() here.


# ─── Loss ──────────────────────────────────────────────────────────────────

def build_loss(class_weights_path, device, ignore_index=0):
    """
    Build weighted CrossEntropyLoss.
    Loads weights computed by prepare_dataset.py if available.
    Class 0 (unlabeled) is ignored in the loss.
    """
    weights = torch.ones(3, device=device)  # [unlabeled, ground, road]

    if class_weights_path and os.path.exists(class_weights_path):
        with open(class_weights_path) as f:
            data = json.load(f)
        if 'train' in data:
            weights[1] = data['train']['ground']
            weights[2] = data['train']['road']
            print(f"Loaded class weights: ground={weights[1]:.3f}, road={weights[2]:.3f}")
    else:
        print("No class_weights.json found — using uniform weights")

    weights[0] = 0.0  # unlabeled gets zero weight, combined with ignore_index

    return nn.CrossEntropyLoss(weight=weights, ignore_index=ignore_index)


# ─── Metrics ───────────────────────────────────────────────────────────────

def compute_iou(pred, target, num_classes=3, ignore_index=0):
    """Compute per-class IoU and mean IoU, ignoring the ignore_index class."""
    ious = []
    for cls in range(num_classes):
        if cls == ignore_index:
            continue
        pred_mask   = (pred == cls)
        target_mask = (target == cls)
        intersection = (pred_mask & target_mask).sum().item()
        union         = (pred_mask | target_mask).sum().item()
        if union == 0:
            continue
        ious.append(intersection / union)

    mean_iou = np.mean(ious) if ious else 0.0
    return mean_iou, ious


# ─── Training loop ─────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device,
                    grad_accum_steps, scaler, epoch, writer, global_step, scheduler):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    pbar = tqdm(loader, desc=f"Epoch {epoch} [train]")
    for i, (images, masks) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            outputs = model(pixel_values=images)
            logits  = outputs.logits  # (B, num_labels, H/4, W/4) — SegFormer downsamples

            # Upsample logits to match mask resolution
            logits_upsampled = F.interpolate(
                logits, size=masks.shape[-2:], mode='bilinear', align_corners=False)

            loss = criterion(logits_upsampled, masks)
            loss = loss / grad_accum_steps

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (i + 1) % grad_accum_steps == 0:
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
            # CRITICAL FIX: step the scheduler once per OPTIMIZER step,
            # not once per epoch. warmup_steps/total_steps were computed
            # in units of optimizer steps, so the scheduler must match.
            scheduler.step()
            global_step += 1

        total_loss += loss.item() * grad_accum_steps

        if global_step % 20 == 0:
            writer.add_scalar('train/loss_step', loss.item() * grad_accum_steps, global_step)

        pbar.set_postfix({'loss': f"{loss.item() * grad_accum_steps:.4f}"})

    avg_loss = total_loss / len(loader)
    return avg_loss, global_step


@torch.no_grad()
def validate(model, loader, criterion, device, epoch, writer):
    model.eval()
    total_loss = 0.0
    all_ious   = []

    pbar = tqdm(loader, desc=f"Epoch {epoch} [val]")
    for images, masks in pbar:
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device, non_blocking=True)

        outputs = model(pixel_values=images)
        logits  = outputs.logits
        logits_upsampled = F.interpolate(
            logits, size=masks.shape[-2:], mode='bilinear', align_corners=False)

        loss = criterion(logits_upsampled, masks)
        total_loss += loss.item()

        preds = logits_upsampled.argmax(dim=1)
        for p, m in zip(preds, masks):
            mean_iou, _ = compute_iou(p, m)
            all_ious.append(mean_iou)

        pbar.set_postfix({'loss': f"{loss.item():.4f}"})

    avg_loss = total_loss / len(loader)
    avg_iou  = np.mean(all_ious) if all_ious else 0.0

    writer.add_scalar('val/loss', avg_loss, epoch)
    writer.add_scalar('val/mean_iou', avg_iou, epoch)

    return avg_loss, avg_iou


# ─── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train SegFormer for curb-gutter segmentation")
    parser.add_argument("--data_dir",        required=True, help="Dataset folder from prepare_dataset.py")
    parser.add_argument("--output_dir",      required=True, help="Where to save checkpoints/logs")
    parser.add_argument("--run",             required=True,
                        help="Name for this training run, e.g. Test_Run1_512Patches. "
                             "Creates output_dir/<run>/ containing config.json, best.pt, last.pt")
    parser.add_argument("--model_size",      default="b2", choices=["b0", "b1", "b2", "b3", "b4", "b5"])
    parser.add_argument("--resolution",      type=float, default=0.02,
                        help="Raster resolution in meters used by prepare_dataset.py for this data "
                             "(recorded in config.json so inference matches exactly)")
    parser.add_argument("--k",               type=int,   default=50,
                        help="k-neighbors used for verticality computation in prepare_dataset.py "
                             "(recorded in config.json so inference matches exactly)")
    parser.add_argument("--epochs",          type=int,   default=100)
    parser.add_argument("--lr",              type=float, default=6e-5)
    parser.add_argument("--batch_size",      type=int,   default=8)
    parser.add_argument("--grad_accum_steps",type=int,   default=2)
    parser.add_argument("--num_workers",     type=int,   default=4)
    parser.add_argument("--weight_decay",    type=float, default=0.01)
    parser.add_argument("--warmup_epochs",   type=int,   default=5)
    parser.add_argument("--mixed_precision", action="store_true", default=True)
    parser.add_argument("--resume",          default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--seed",            type=int,   default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Named run folder ─────────────────────────────────────────────────
    # output_dir/<run>/  holds config.json, best.pt, last.pt, tensorboard logs.
    # This is the ONE folder you point inference.py at via --model_folder.
    run_dir = Path(args.output_dir) / args.run
    run_dir.mkdir(parents=True, exist_ok=True)

    config_path = run_dir / "config.json"
    if config_path.exists():
        print(f"NOTE: {config_path} already exists (resuming or overwriting a previous run "
              f"with the same --run name). It will be overwritten with current args.")

    training_config = {
        "run_name":     args.run,
        "model_size":   args.model_size,
        "num_channels": 6,
        "num_labels":   3,
        "resolution":   args.resolution,
        "k_neighbors":  args.k,
        "channel_order": CHANNEL_ORDER,
        "label_map":    {"0": "unlabeled", "1": "ground", "2": "road"},
        "patch_size":   None,  # filled in below from actual data, if discoverable
        "trained_epochs_requested": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "warmup_epochs": args.warmup_epochs,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "data_dir": str(args.data_dir),
        "created": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }

    # Inference-time settings: left as placeholders here. Edit config.json
    # by hand before running inference, or override via inference.py CLI flags.
    inference_config = {
        "input_folder":     "",
        "single_file":      "",
        "output_folder":    "",
        "run_option":       "folder",   # "folder" or "single"
        "inference_stride": 256,
        "min_valid_ratio":  0.1,
    }

    full_config = {
        "training_config":  training_config,
        "inference_config": inference_config,
    }

    with open(config_path, "w") as f:
        json.dump(full_config, f, indent=2)
    print(f"Config saved: {config_path}")

    output_dir = run_dir  # checkpoints save directly into the run folder

    # ── Datasets ──────────────────────────────────────────────────────────
    train_ds = CurbGutterDataset(args.data_dir, split='train')
    val_ds   = CurbGutterDataset(args.data_dir, split='val')

    # Detect actual patch size from real data rather than guessing —
    # this is what inference.py needs to reconstruct the same patch grid.
    sample_image, _ = train_ds[0]
    detected_patch_size = sample_image.shape[-1]  # (C, H, W), H==W expected
    training_config["patch_size"] = detected_patch_size
    with open(config_path, "w") as f:
        json.dump(full_config, f, indent=2)
    print(f"Detected patch_size={detected_patch_size} from training data, config.json updated")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True)

    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")
    print(f"Effective batch size: {args.batch_size * args.grad_accum_steps}")

    # ── Model ─────────────────────────────────────────────────────────────
    model = build_model(model_size=args.model_size, num_channels=6, num_labels=3)
    model.to(device)

    # ── Loss ──────────────────────────────────────────────────────────────
    class_weights_path = Path(args.data_dir) / "class_weights.json"
    criterion = build_loss(class_weights_path, device)

    # ── Optimizer & Scheduler ────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    total_steps  = (len(train_loader) // args.grad_accum_steps) * args.epochs
    warmup_steps = (len(train_loader) // args.grad_accum_steps) * args.warmup_epochs

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.cuda.amp.GradScaler() if args.mixed_precision else None

    # ── Resume ────────────────────────────────────────────────────────────
    start_epoch = 0
    best_iou    = 0.0
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_iou    = ckpt.get('best_iou', 0.0)

    # ── TensorBoard ───────────────────────────────────────────────────────
    log_dir = output_dir / "tensorboard" / datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(log_dir)
    print(f"TensorBoard logs: {log_dir}")
    print(f"Run: tensorboard --logdir {output_dir / 'tensorboard'}")

    global_step = start_epoch * (len(train_loader) // args.grad_accum_steps)

    # ── Training loop ────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs):
        train_loss, global_step = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            args.grad_accum_steps, scaler, epoch, writer, global_step, scheduler)

        val_loss, val_iou = validate(model, val_loader, criterion, device, epoch, writer)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch {epoch:3d} | train_loss={train_loss:.4f} | "
              f"val_loss={val_loss:.4f} | val_mIoU={val_iou:.4f} | lr={current_lr:.2e}")

        writer.add_scalar('train/loss_epoch', train_loss, epoch)
        writer.add_scalar('train/lr', current_lr, epoch)

        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_iou': val_iou,
            'best_iou': best_iou,
            'args': vars(args),
        }

        torch.save(checkpoint, output_dir / "last.pt")

        if val_iou > best_iou:
            best_iou = val_iou
            checkpoint['best_iou'] = best_iou
            torch.save(checkpoint, output_dir / "best.pt")
            print(f"  → New best model saved (mIoU={best_iou:.4f})")

    writer.close()
    print(f"\nTraining complete. Best val mIoU: {best_iou:.4f}")
    print(f"Best checkpoint: {output_dir / 'best.pt'}")
    print(f"Run folder (use this as --model_folder for inference): {output_dir}")


if __name__ == "__main__":
    main()
