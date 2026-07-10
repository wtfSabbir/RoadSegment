"""
Dataset Preparation Script: LAZ → SegFormer Training Patches
=============================================================
Step 2 of the curb-gutter segmentation pipeline.

What this script does:
  1. Reads all LAZ tiles from an input folder
  2. Converts each to a 6-channel 2.5D raster (same as audit script)
  3. Cuts each raster into 512x512 patches with 50% overlap
  4. Saves image patches (6-channel .npy OR 3-channel RGB .png) + label masks
  5. Filters out patches that are mostly empty or mostly unlabeled
  6. Produces a train/val split with a summary report

Output structure (you point --output to YOUR folder):
  your_folder/
    images/
      train/   ← 6-channel .npy files (for SegFormer with custom channels)
      val/
    masks/
      train/   ← single-channel label .png files (0=unlabeled, 1=ground, 2=road)
      val/
    rgb_preview/  ← RGB .png previews so you can visually check patches
      train/
      val/
    dataset_report.txt  ← summary stats

Usage:
  python prepare_dataset.py \\
    --input  /path/to/laz/tiles/ \\
    --output /path/to/your/dataset/ \\
    --resolution 0.02 \\
    --patch_size 512 \\
    --stride 256 \\
    --val_split 0.15 \\
    --k 50 \\
    --min_valid_ratio 0.3 \\
    --min_road_ratio 0.02

Dependencies:
  pip install laspy lazrs-python numpy scipy matplotlib scikit-learn opencv-python tqdm
"""

import argparse
import os
import sys
import json
import random
import numpy as np
from pathlib import Path
from datetime import datetime
import cv2

try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm not installed
    def tqdm(x, **kwargs):
        return x

# ─── Label config ─────────────────────────────────────────────────────────────
# Adjust to match YOUR classification field values
LABEL_MAP = {
    0: "unlabeled",
    1: "ground",   # sidewalk + everything else
    2: "road",
}

# ─── Core feature computation (same logic as audit script) ────────────────────

def load_laz(path):
    try:
        import laspy
    except ImportError:
        print("ERROR: pip install laspy lazrs-python")
        sys.exit(1)
    return laspy.read(str(path))


def get_arrays(las):
    """Extract numpy arrays from laspy object safely."""
    x = np.array(las.x, dtype=np.float64)
    y = np.array(las.y, dtype=np.float64)
    z = np.array(las.z, dtype=np.float64)

    intensity = None
    if hasattr(las, 'intensity'):
        intensity = np.array(las.intensity, dtype=np.float32)
        mx = intensity.max()
        if mx > 0:
            intensity /= mx

    rgb = None
    if all(hasattr(las, c) for c in ['red', 'green', 'blue']):
        r = np.array(las.red,   dtype=np.float32)
        g = np.array(las.green, dtype=np.float32)
        b = np.array(las.blue,  dtype=np.float32)
        mx = max(r.max(), g.max(), b.max())
        if mx > 0:
            r /= mx; g /= mx; b /= mx
        rgb = np.stack([r, g, b], axis=1)

    labels = None
    if hasattr(las, 'classification'):
        labels = np.array(las.classification, dtype=np.uint8)

    return x, y, z, intensity, rgb, labels


def compute_verticality(x, y, z, k=50, batch_size=100_000):
    """
    Compute verticality per point using PCA on k-NN neighborhood.
    Returns float32 array in [0, 1]. 0=flat surface, 1=vertical surface.
    """
    from sklearn.neighbors import KDTree

    pts = np.column_stack([x, y, z]).astype(np.float32)
    tree = KDTree(pts)
    _, indices = tree.query(pts, k=k + 1)
    indices = indices[:, 1:]  # remove self

    n = len(pts)
    verticality = np.zeros(n, dtype=np.float32)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        neighbors = pts[indices[start:end]]          # (batch, k, 3)
        centroid  = neighbors.mean(axis=1, keepdims=True)
        centered  = neighbors - centroid
        cov = np.einsum('bki,bkj->bij', centered, centered) / k
        try:
            eigvals = np.linalg.eigvalsh(cov)        # (batch, 3) ascending
            lam0 = eigvals[:, 0]
            lam2 = eigvals[:, 2]
            denom = np.where(lam2 > 1e-9, lam2, 1e-9)
            verticality[start:end] = np.clip(lam0 / denom, 0, 1)
        except np.linalg.LinAlgError:
            pass

    return verticality


def compute_height_above_ground(x, y, z, cell_size=1.0):
    """Height above local minimum z in each 1m grid cell."""
    xi = np.floor((x - x.min()) / cell_size).astype(np.int32)
    yi = np.floor((y - y.min()) / cell_size).astype(np.int32)
    key = xi.astype(np.int64) * 100000 + yi.astype(np.int64)

    hag = np.zeros(len(z), dtype=np.float32)
    for uk in np.unique(key):
        mask = key == uk
        hag[mask] = z[mask] - z[mask].min()

    # Normalize to [0, 1]
    mx = hag.max()
    if mx > 0:
        hag /= mx
    return hag


def build_raster(x, y, z, intensity, rgb, labels, verticality, hag, resolution):
    """
    Build 6-channel bird's-eye raster + label mask.
    Channels: [hag, verticality, intensity, R, G, B]
    For each pixel: highest-z point wins.
    Returns raster (H, W, 6) float32, label_mask (H, W) uint8
    """
    x_min, y_min = x.min(), y.min()
    cols = int(np.ceil((x.max() - x_min) / resolution)) + 1
    rows = int(np.ceil((y.max() - y_min) / resolution)) + 1

    col_idx = np.floor((x - x_min) / resolution).astype(np.int32)
    row_idx = np.floor((y - y_min) / resolution).astype(np.int32)
    flat    = row_idx * cols + col_idx
    n_px    = rows * cols

    # Highest-z wins per pixel
    z_grid  = np.full(n_px, -np.inf, dtype=np.float32)
    winner  = np.full(n_px, -1,      dtype=np.int32)
    order   = np.argsort(z)
    for pt_i in order:
        fi = flat[pt_i]
        if 0 <= fi < n_px and z[pt_i] > z_grid[fi]:
            z_grid[fi] = z[pt_i]
            winner[fi] = pt_i

    valid_flat = np.where(winner >= 0)[0]
    valid_pts  = winner[valid_flat]
    ri = valid_flat // cols
    ci = valid_flat  % cols

    raster = np.zeros((rows, cols, 6), dtype=np.float32)
    raster[ri, ci, 0] = hag[valid_pts]
    raster[ri, ci, 1] = verticality[valid_pts]
    raster[ri, ci, 2] = intensity[valid_pts] if intensity is not None else 0.0
    if rgb is not None:
        raster[ri, ci, 3] = rgb[valid_pts, 0]
        raster[ri, ci, 4] = rgb[valid_pts, 1]
        raster[ri, ci, 5] = rgb[valid_pts, 2]

    # Label mask — majority vote per pixel
    label_mask = np.zeros((rows, cols), dtype=np.uint8)
    if labels is not None:
        flat_sorted_idx = np.argsort(flat)
        sorted_flat     = flat[flat_sorted_idx]
        sorted_labels   = labels[flat_sorted_idx]
        splits          = np.where(np.diff(sorted_flat))[0] + 1
        groups          = np.split(sorted_labels, splits)
        unique_pixels   = sorted_flat[np.concatenate([[0], splits])]
        for pix, grp in zip(unique_pixels, groups):
            if 0 <= pix < n_px:
                r2 = pix // cols
                c2 = pix  % cols
                label_mask[r2, c2] = np.bincount(grp.astype(np.int32),
                                                  minlength=3).argmax()

    return raster, label_mask, (x_min, y_min, rows, cols)


# ─── Patching ─────────────────────────────────────────────────────────────────

def extract_patches(raster, label_mask, patch_size, stride,
                    min_valid_ratio, min_road_ratio):
    """
    Slide a window over the raster and extract valid patches.
    Returns list of (image_patch, mask_patch) numpy arrays.

    Filters:
      - min_valid_ratio : minimum fraction of non-zero pixels in the image patch
      - min_road_ratio  : minimum fraction of road pixels in the mask
                          (ensures patch has something useful to learn from)
    """
    H, W = raster.shape[:2]
    patches = []

    for r in range(0, H - patch_size + 1, stride):
        for c in range(0, W - patch_size + 1, stride):
            img_patch  = raster[r:r+patch_size, c:c+patch_size]      # (P, P, 6)
            mask_patch = label_mask[r:r+patch_size, c:c+patch_size]  # (P, P)

            # Filter: too many empty pixels
            valid_pixels = (img_patch[:, :, 0] > 0).sum()
            if valid_pixels / (patch_size * patch_size) < min_valid_ratio:
                continue

            # Filter: no road pixels at all (patch is pure background)
            road_pixels = (mask_patch == 2).sum()
            if road_pixels / (patch_size * patch_size) < min_road_ratio:
                continue

            patches.append((img_patch.copy(), mask_patch.copy()))

    return patches


# ─── Saving ───────────────────────────────────────────────────────────────────

def save_patch(img_patch, mask_patch, img_path, mask_path, preview_path=None):
    """Save one patch pair."""
    np.save(img_path,  img_patch)    # (512, 512, 6) float32
    # Save mask as PNG (0, 1, 2 values — uint8)
    
    cv2.imwrite(str(mask_path), mask_patch)

    if preview_path is not None:
        # Save RGB preview
        rgb = (img_patch[:, :, 3:6] * 255).astype(np.uint8)
        cv2.imwrite(str(preview_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Prepare SegFormer training dataset from LAZ tiles")
    parser.add_argument("--input",      required=True,
                        help="Folder containing .laz or .las files")
    parser.add_argument("--output",     required=True,
                        help="Output dataset folder")
    parser.add_argument("--resolution", type=float, default=0.02,
                        help="Raster resolution in meters (default: 0.02)")
    parser.add_argument("--patch_size", type=int,   default=512,
                        help="Patch size in pixels (default: 512)")
    parser.add_argument("--stride",     type=int,   default=256,
                        help="Stride for patch extraction (default: 256 = 50%% overlap)")
    parser.add_argument("--k",          type=int,   default=50,
                        help="Neighbors for verticality (default: 50)")
    parser.add_argument("--val_split",  type=float, default=0.15,
                        help="Fraction of tiles to use for validation (default: 0.15). "
                             "Ignored if --val_tiles is provided.")
    parser.add_argument("--val_tiles", nargs="+", default=None,
                        help="Explicit tile stems (no extension) to use as validation. "
                             "Everything else goes to train. Overrides --val_split. "
                             "Example: --val_tiles tile_003 tile_007 tile_012")
    parser.add_argument("--min_valid_ratio", type=float, default=0.25,
                        help="Min fraction of non-empty pixels per patch (default: 0.3)")
    parser.add_argument("--min_road_ratio",  type=float, default=0.02,
                        help="Min fraction of road pixels per patch (default: 0.02)")
    parser.add_argument("--save_preview",    action="store_true",
                        help="Save RGB preview PNGs (slower but useful for debugging)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for train/val split")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # ── Find all LAZ files ────────────────────────────────────────────────────
    input_dir = Path(args.input)
    laz_files = sorted(list(input_dir.glob("*.laz")) +
                       list(input_dir.glob("*.las")) +
                       list(input_dir.glob("*.LAZ")) +
                       list(input_dir.glob("*.LAS")))

    if not laz_files:
        print(f"ERROR: No .laz/.las files found in {input_dir}")
        sys.exit(1)

    print(f"\nFound {len(laz_files)} LAZ files")
    print(f"Resolution   : {args.resolution}m")
    print(f"Patch size   : {args.patch_size}×{args.patch_size} px "
          f"= {args.patch_size * args.resolution:.1f}×{args.patch_size * args.resolution:.1f}m")
    print(f"Stride       : {args.stride} px = {args.stride * args.resolution:.1f}m")
    print(f"Val split    : {args.val_split:.0%}")

    # ── Train / val split at tile level ──────────────────────────────────────
    # Split at TILE level, not patch level — prevents data leakage.
    # If --val_tiles is provided, use those exact tiles for val (by stem name).
    # Otherwise fall back to random --val_split fraction.
    if args.val_tiles is not None:
        # Normalise: strip extensions from user-provided names so comparison
        # works whether they typed "tile_003" or "tile_003.laz"
        requested_val_stems = set(Path(t).stem for t in args.val_tiles)
        val_files   = set(str(f) for f in laz_files if f.stem in requested_val_stems)
        train_files = set(str(f) for f in laz_files if f.stem not in requested_val_stems)

        # Warn about any requested stems that weren't found
        found_stems = set(Path(f).stem for f in val_files)
        missing = requested_val_stems - found_stems
        if missing:
            print(f"WARNING: These --val_tiles names were not found in input folder: {missing}")

        print(f"Split mode   : EXPLICIT (--val_tiles provided)")
    else:
        shuffled = laz_files.copy()
        random.shuffle(shuffled)
        n_val   = max(1, int(len(shuffled) * args.val_split))
        val_files   = set(str(f) for f in shuffled[:n_val])
        train_files = set(str(f) for f in shuffled[n_val:])
        print(f"Split mode   : RANDOM (--val_split={args.val_split:.0%})")

    print(f"Train tiles  : {len(train_files)}")
    print(f"Val tiles    : {len(val_files)}")
    if args.val_tiles is not None:
        print(f"Val tile names: {sorted([Path(f).stem for f in val_files])}")

    # ── Create output dirs ────────────────────────────────────────────────────
    out = Path(args.output)
    for split in ['train', 'val']:
        (out / 'images' / split).mkdir(parents=True, exist_ok=True)
        (out / 'masks'  / split).mkdir(parents=True, exist_ok=True)
        if args.save_preview:
            (out / 'rgb_preview' / split).mkdir(parents=True, exist_ok=True)

    # ── Process each tile ─────────────────────────────────────────────────────
    stats = {
        'train': {'tiles': 0, 'patches': 0, 'skipped': 0,
                  'class_pixels': {0: 0, 1: 0, 2: 0}},
        'val':   {'tiles': 0, 'patches': 0, 'skipped': 0,
                  'class_pixels': {0: 0, 1: 0, 2: 0}},
    }
    patch_counter = {'train': 0, 'val': 0}

    for laz_path in tqdm(laz_files, desc="Processing tiles"):
        split = 'val' if str(laz_path) in val_files else 'train'
        tile_name = laz_path.stem

        print(f"\n{'='*60}")
        print(f"[{split.upper()}] {tile_name}")

        try:
            # 1. Load
            las = load_laz(laz_path)
            x, y, z, intensity, rgb, labels = get_arrays(las)
            print(f"  Points: {len(x):,}")

            if labels is None:
                print("  WARNING: No classification field — skipping tile")
                continue

            # 2. Features
            print("  Computing verticality...")
            vert = compute_verticality(x, y, z, k=args.k)
            print("  Computing height above ground...")
            hag  = compute_height_above_ground(x, y, z)

            # 3. Rasterize
            print("  Rasterizing...")
            raster, label_mask, meta = build_raster(
                x, y, z, intensity, rgb, labels, vert, hag, args.resolution)
            print(f"  Raster: {raster.shape[0]}×{raster.shape[1]} px")

            # 4. Extract patches
            patches = extract_patches(
                raster, label_mask,
                patch_size=args.patch_size,
                stride=args.stride,
                min_valid_ratio=args.min_valid_ratio,
                min_road_ratio=args.min_road_ratio,
            )
            n_total_possible = (
                ((raster.shape[0] - args.patch_size) // args.stride + 1) *
                ((raster.shape[1] - args.patch_size) // args.stride + 1)
            )
            n_skipped = n_total_possible - len(patches)
            print(f"  Patches: {len(patches)} kept, {n_skipped} skipped (empty/no road)")

            # 5. Save patches
            for img_patch, mask_patch in patches:
                idx = patch_counter[split]
                fname = f"{tile_name}_patch{idx:05d}"

                img_path  = out / 'images' / split / f"{fname}.npy"
                mask_path = out / 'masks'  / split / f"{fname}.png"
                prev_path = (out / 'rgb_preview' / split / f"{fname}.png"
                             if args.save_preview else None)

                save_patch(img_patch, mask_patch, img_path, mask_path, prev_path)
                patch_counter[split] += 1

                # Accumulate class stats
                for cls in [0, 1, 2]:
                    stats[split]['class_pixels'][cls] += int((mask_patch == cls).sum())

            stats[split]['tiles']   += 1
            stats[split]['patches'] += len(patches)
            stats[split]['skipped'] += n_skipped

        except Exception as e:
            print(f"  ERROR processing {tile_name}: {e}")
            import traceback; traceback.print_exc()
            continue

    # ── Report ────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("DATASET PREPARATION COMPLETE")
    print(f"{'='*60}")

    report_lines = [
        f"Dataset prepared: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Resolution: {args.resolution}m | Patch: {args.patch_size}px | Stride: {args.stride}px",
        f"",
    ]

    for split in ['train', 'val']:
        s = stats[split]
        total_px = sum(s['class_pixels'].values())
        report_lines += [
            f"[{split.upper()}]",
            f"  Tiles processed : {s['tiles']}",
            f"  Patches saved   : {s['patches']}",
            f"  Patches skipped : {s['skipped']}",
        ]
        if total_px > 0:
            report_lines += [
                f"  Class distribution:",
                f"    unlabeled : {s['class_pixels'][0]:>10,}  "
                f"({100*s['class_pixels'][0]/total_px:5.1f}%)",
                f"    ground    : {s['class_pixels'][1]:>10,}  "
                f"({100*s['class_pixels'][1]/total_px:5.1f}%)",
                f"    road      : {s['class_pixels'][2]:>10,}  "
                f"({100*s['class_pixels'][2]/total_px:5.1f}%)",
            ]

            # Compute class weights for training
            freqs = np.array([s['class_pixels'][i] for i in [1, 2]], dtype=np.float64)
            freqs = freqs / freqs.sum()
            weights = 1.0 / (freqs + 1e-6)
            weights = weights / weights.sum() * len(weights)
            report_lines += [
                f"",
                f"  Suggested loss weights (for CrossEntropyLoss):",
                f"    ground : {weights[0]:.4f}",
                f"    road   : {weights[1]:.4f}",
                f"    (ignore_index=0 for unlabeled)",
            ]
        report_lines.append("")

    report_lines += [
        f"Output structure:",
        f"  {args.output}/images/train/  ← {patch_counter['train']} .npy files (512×512×6)",
        f"  {args.output}/images/val/    ← {patch_counter['val']} .npy files",
        f"  {args.output}/masks/train/   ← {patch_counter['train']} .png files (0/1/2)",
        f"  {args.output}/masks/val/     ← {patch_counter['val']} .png files",
        f"",
        f"Next step: train SegFormer with these patches.",
        f"Use ignore_index=0 in your loss function to skip unlabeled pixels.",
    ]

    report_text = "\n".join(report_lines)
    print(report_text)

    report_path = out / "dataset_report.txt"
    report_path.write_text(report_text)
    print(f"\nReport saved: {report_path}")

    # Save class weights as JSON for training script to load
    weights_data = {}
    for split in ['train', 'val']:
        s = stats[split]
        freqs = np.array([s['class_pixels'][i] for i in [1, 2]], dtype=np.float64)
        if freqs.sum() > 0:
            freqs = freqs / freqs.sum()
            w = 1.0 / (freqs + 1e-6)
            w = w / w.sum() * 2
            weights_data[split] = {'ground': float(w[0]), 'road': float(w[1])}

    weights_path = out / "class_weights.json"
    with open(weights_path, 'w') as f:
        json.dump(weights_data, f, indent=2)
    print(f"Class weights saved: {weights_path}")


if __name__ == "__main__":
    main()