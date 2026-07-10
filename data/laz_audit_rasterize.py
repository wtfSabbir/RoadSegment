"""
LAZ Tile Audit & 2.5D Rasterization Script
============================================
Step 1 of the curb-gutter segmentation pipeline.

What this script does:
  1. Reads a LAZ/LAS file using laspy
  2. Audits the tile (density, class distribution, RGB quality)
  3. Computes per-point features: verticality, height above ground, normal Z
  4. Projects to bird's-eye-view raster (multi-channel)
  5. Outputs a label mask image
  6. Saves visualizations so you can inspect boundary sharpness

Usage:
  python laz_audit_rasterize.py --input your_tile.laz --resolution 0.05

Dependencies:
  pip install laspy lazrs-python numpy scipy matplotlib scikit-learn opencv-python
"""

import argparse
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

# ─── Label map ────────────────────────────────────────────────────────────────
# Adjust these to match YOUR dataset's label integers
LABEL_MAP = {
    0: ("unlabeled",  "gray"),
    1: ("road",       "gray"),
    2: ("gutter",     "cyan"),
    3: ("curb",       "red"),
    4: ("sidewalk",   "orange"),
    5: ("grass/soil", "green"),
    # Add more if needed
}

# ─── Helpers ──────────────────────────────────────────────────────────────────

def load_laz(path: str):
    """Load LAZ/LAS file and return laspy point cloud object."""
    try:
        import laspy
    except ImportError:
        print("ERROR: laspy not installed. Run: pip install laspy lazrs-python")
        sys.exit(1)

    print(f"\n[1/6] Loading: {path}")
    las = laspy.read(path)
    print(f"      Point count : {len(las.points):,}")
    print(f"      LAS version : {las.header.version}")

    # Print available dimensions
    dims = [d.name for d in las.point_format.dimensions]
    print(f"      Dimensions  : {dims}")
    return las


def audit_tile(las):
    """Print a full audit of the tile."""
    print("\n[2/6] Auditing tile...")

    x = np.array(las.x)
    y = np.array(las.y)
    z = np.array(las.z)

    x_range = x.max() - x.min()
    y_range = y.max() - y.min()
    area_m2 = x_range * y_range
    density  = len(x) / area_m2 if area_m2 > 0 else 0

    print(f"      X range     : {x.min():.2f} → {x.max():.2f}  ({x_range:.2f} m)")
    print(f"      Y range     : {y.min():.2f} → {y.max():.2f}  ({y_range:.2f} m)")
    print(f"      Z range     : {z.min():.2f} → {z.max():.2f}  ({z.max()-z.min():.2f} m)")
    print(f"      Area        : {area_m2:.1f} m²")
    print(f"      Density     : {density:.1f} pts/m²")

    # RGB quality check
    has_rgb = all(hasattr(las, c) for c in ['red', 'green', 'blue'])
    if has_rgb:
        r = np.array(las.red)
        g = np.array(las.green)
        b = np.array(las.blue)
        # LAS stores RGB as 16-bit; normalize
        max_val = max(r.max(), g.max(), b.max())
        if max_val > 255:
            print(f"      RGB range   : 16-bit (max={max_val}) → will normalize to 0-255")
        else:
            print(f"      RGB range   : 8-bit  (max={max_val})")
        rgb_var = np.var(r.astype(float)) + np.var(g.astype(float)) + np.var(b.astype(float))
        print(f"      RGB variance: {rgb_var:.0f}  ({'good' if rgb_var > 1000 else 'LOW - colors may be washed out'})")
    else:
        print("      RGB         : NOT FOUND")

    # Label distribution
    if hasattr(las, 'classification'):
        labels = np.array(las.classification)
        unique, counts = np.unique(labels, return_counts=True)
        total = len(labels)
        print(f"\n      Label distribution:")
        for u, c in zip(unique, counts):
            name = LABEL_MAP.get(int(u), (str(u), "white"))[0]
            bar  = "█" * int(40 * c / total)
            print(f"        [{int(u):2d}] {name:<12} {c:>10,}  ({100*c/total:5.1f}%)  {bar}")
    else:
        print("      Labels      : classification field NOT FOUND")
        print("                    Check your LAZ dimension names above")

    return {
        'x': x, 'y': y, 'z': z,
        'has_rgb': has_rgb,
        'density': density,
        'x_range': x_range,
        'y_range': y_range
    }


def compute_features(las, info, k_neighbors=20):
    """
    Compute per-point geometric features:
      - height_above_ground : z minus local ground estimate
      - verticality         : how vertical the local surface is (0=flat, 1=vertical)
      - normal_z            : Z component of estimated surface normal
    """
    print(f"\n[3/6] Computing geometric features (k={k_neighbors} neighbors)...")
    print(f"      This may take a minute on large tiles...")

    from sklearn.neighbors import KDTree

    x, y, z = info['x'], info['y'], info['z']
    pts = np.column_stack([x, y, z])

    # KDTree for neighbor search
    tree = KDTree(pts)
    distances, indices = tree.query(pts, k=k_neighbors + 1)  # +1 because point queries itself
    indices = indices[:, 1:]  # remove self

    n_pts = len(pts)
    verticality  = np.zeros(n_pts, dtype=np.float32)
    normal_z     = np.ones(n_pts,  dtype=np.float32)

    # Process in batches to avoid memory spike
    batch_size = 50_000
    for start in range(0, n_pts, batch_size):
        end = min(start + batch_size, n_pts)
        idx_batch = indices[start:end]          # (batch, k)
        neighbors = pts[idx_batch]              # (batch, k, 3)

        # PCA on local neighborhood → smallest eigenvalue direction = normal
        # Center neighbors
        centroid = neighbors.mean(axis=1, keepdims=True)   # (batch, 1, 3)
        centered = neighbors - centroid                     # (batch, k, 3)

        # Covariance matrix per point: (batch, 3, 3)
        cov = np.einsum('bki,bkj->bij', centered, centered) / k_neighbors

        # Eigenvalues (sorted ascending by numpy)
        try:
            eigvals = np.linalg.eigvalsh(cov)   # (batch, 3)
            lam0 = eigvals[:, 0]                # smallest
            lam2 = eigvals[:, 2]                # largest

            # Verticality = 1 - |normal_z|
            # Approximate: if smallest eigenvalue direction is mostly Z → surface is horizontal
            # We use linearity/planarity proxy: (lam2 - lam0) / lam2
            denom = np.where(lam2 > 1e-9, lam2, 1e-9)
            vert  = lam0 / denom  # small for planar, larger for linear/vertical
            verticality[start:end] = np.clip(vert, 0, 1)
        except np.linalg.LinAlgError:
            pass  # leave as zeros for degenerate cases

        if start % 200_000 == 0:
            print(f"      Processed {start:,} / {n_pts:,} points...")

    # Height above ground: subtract 1st percentile of z in local XY neighborhood
    # Simple approach: subtract minimum z in each 1m cell
    print("      Computing height above ground...")
    cell_size = 1.0
    xi = np.floor((x - x.min()) / cell_size).astype(int)
    yi = np.floor((y - y.min()) / cell_size).astype(int)
    cell_key = xi * 100000 + yi
    height_above_ground = np.zeros(n_pts, dtype=np.float32)

    unique_cells = np.unique(cell_key)
    for uc in unique_cells:
        mask = cell_key == uc
        z_min = z[mask].min()
        height_above_ground[mask] = z[mask] - z_min

    print(f"      Verticality range   : {verticality.min():.3f} → {verticality.max():.3f}")
    print(f"      Height above ground : {height_above_ground.min():.3f} → {height_above_ground.max():.3f} m")

    return verticality, height_above_ground, normal_z


def rasterize(las, info, verticality, height_above_ground, resolution=0.05):
    """
    Project point cloud to bird's-eye-view raster.

    Channels:
      0: height_above_ground (normalized)
      1: verticality         (normalized)
      2: intensity           (normalized)
      3: R (normalized 0-1)
      4: G (normalized 0-1)
      5: B (normalized 0-1)

    For each pixel, takes the value from the HIGHEST z point
    (most informative for curb detection).
    """
    print(f"\n[4/6] Rasterizing at {resolution}m resolution...")

    x, y, z = info['x'], info['y'], info['z']

    x_min, y_min = x.min(), y.min()
    cols = int(np.ceil((x.max() - x_min) / resolution)) + 1
    rows = int(np.ceil((y.max() - y_min) / resolution)) + 1
    print(f"      Raster size : {rows} rows × {cols} cols")
    print(f"      Memory est. : {rows * cols * 6 * 4 / 1e6:.1f} MB (6 channels, float32)")

    # Pixel indices for each point
    col_idx = np.floor((x - x_min) / resolution).astype(int)
    row_idx = np.floor((y - y_min) / resolution).astype(int)

    # For each pixel, keep the highest-z point
    flat_idx  = row_idx * cols + col_idx
    n_pixels  = rows * cols

    # Height mask: track which point "wins" each pixel
    z_grid = np.full(n_pixels, -np.inf, dtype=np.float32)
    winner = np.full(n_pixels, -1, dtype=np.int32)

    sort_order = np.argsort(z)  # process low-z first so high-z overwrites
    for pt_i in sort_order:
        fi = flat_idx[pt_i]
        if 0 <= fi < n_pixels:
            if z[pt_i] > z_grid[fi]:
                z_grid[fi] = z[pt_i]
                winner[fi] = pt_i

    # Build 6-channel raster
    raster = np.zeros((rows, cols, 6), dtype=np.float32)
    valid  = winner >= 0
    valid_flat = np.where(valid)[0]
    valid_pts  = winner[valid_flat]

    r_idx = valid_flat // cols
    c_idx = valid_flat  % cols

    raster[r_idx, c_idx, 0] = height_above_ground[valid_pts]
    raster[r_idx, c_idx, 1] = verticality[valid_pts]

    # Intensity
    if hasattr(las, 'intensity'):
        intensity = np.array(las.intensity, dtype=np.float32)
        int_max   = intensity.max()
        if int_max > 0:
            intensity /= int_max
        raster[r_idx, c_idx, 2] = intensity[valid_pts]

    # RGB
    if info['has_rgb']:
        r = np.array(las.red,   dtype=np.float32)
        g = np.array(las.green, dtype=np.float32)
        b = np.array(las.blue,  dtype=np.float32)
        max_val = max(r.max(), g.max(), b.max())
        if max_val > 0:
            r /= max_val; g /= max_val; b /= max_val
        raster[r_idx, c_idx, 3] = r[valid_pts]
        raster[r_idx, c_idx, 4] = g[valid_pts]
        raster[r_idx, c_idx, 5] = b[valid_pts]

    # Normalize height channel
    h_max = raster[:, :, 0].max()
    if h_max > 0:
        raster[:, :, 0] /= h_max

    print(f"      Raster built. Non-empty pixels: {valid.sum():,} / {n_pixels:,} ({100*valid.sum()/n_pixels:.1f}%)")
    return raster, (x_min, y_min, resolution, rows, cols)


def build_label_mask(las, info, raster_meta):
    """Build label mask image aligned to the raster."""
    print("\n[5/6] Building label mask...")

    if not hasattr(las, 'classification'):
        print("      WARNING: No classification field found — skipping label mask")
        return None

    x, y = info['x'], info['y']
    x_min, y_min, resolution, rows, cols = raster_meta
    labels = np.array(las.classification, dtype=np.uint8)

    col_idx = np.floor((x - x_min) / resolution).astype(int)
    row_idx = np.floor((y - y_min) / resolution).astype(int)

    # For label mask: majority vote per pixel (most common label wins)
    label_grid = np.zeros((rows, cols), dtype=np.uint8)
    from scipy import stats as scipy_stats

    flat_idx = row_idx * cols + col_idx
    n_pixels = rows * cols

    # Group by pixel and take majority
    order = np.argsort(flat_idx)
    sorted_flat  = flat_idx[order]
    sorted_labels= labels[order]

    splits = np.where(np.diff(sorted_flat))[0] + 1
    groups = np.split(sorted_labels, splits)
    unique_pixels = sorted_flat[np.concatenate([[0], splits])]

    for pix, grp in zip(unique_pixels, groups):
        if 0 <= pix < n_pixels:
            r = pix // cols
            c = pix  % cols
            label_grid[r, c] = np.bincount(grp).argmax()

    unique_labels, counts = np.unique(label_grid, return_counts=True)
    print("      Label mask pixel distribution:")
    for u, c in zip(unique_labels, counts):
        name = LABEL_MAP.get(int(u), (str(u), "white"))[0]
        print(f"        [{int(u):2d}] {name:<12} {c:>8,} pixels")

    return label_grid


def visualize_and_save(raster, label_mask, output_dir, resolution):
    """Save visualization images."""
    print(f"\n[6/6] Saving visualizations to: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    channel_names = [
        "height_above_ground",
        "verticality",
        "intensity",
        "red",
        "green",
        "blue"
    ]

    # Save individual channels
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"2.5D Raster Channels (resolution={resolution}m)", fontsize=14)

    for i, (ax, name) in enumerate(zip(axes.flat, channel_names)):
        ch = raster[:, :, i]
        im = ax.imshow(ch, cmap='viridis', origin='lower', vmin=0, vmax=1)
        ax.set_title(name)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    ch_path = os.path.join(output_dir, "channels.png")
    plt.savefig(ch_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"      Saved: {ch_path}")

    # RGB composite
    rgb = raster[:, :, 3:6]
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(rgb, origin='lower')
    ax.set_title("RGB Composite")
    ax.axis('off')
    rgb_path = os.path.join(output_dir, "rgb_composite.png")
    plt.savefig(rgb_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"      Saved: {rgb_path}")

    # Verticality — most important for curb detection
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(raster[:, :, 1], cmap='hot', origin='lower', vmin=0, vmax=1)
    ax.set_title("Verticality Channel\n(bright = vertical surface = CURB)", fontsize=12)
    ax.axis('off')
    plt.colorbar(im, ax=ax, label='Verticality (0=flat, 1=vertical)')
    vert_path = os.path.join(output_dir, "verticality_highlight.png")
    plt.savefig(vert_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"      Saved: {vert_path}")

    # Overlay: verticality on top of RGB with label mask
    if label_mask is not None:
        # Build color label image
        color_mask = np.zeros((*label_mask.shape, 3), dtype=np.float32)
        label_colors = {
            0: [0.5, 0.5, 0.5],  # unlabeled - gray
            1: [0.4, 0.4, 0.4],  # road - dark gray
            2: [0.0, 1.0, 1.0],  # gutter - cyan
            3: [1.0, 0.0, 0.0],  # curb - red
            4: [1.0, 0.6, 0.0],  # sidewalk - orange
            5: [0.0, 0.8, 0.0],  # grass - green
        }
        for lbl, color in label_colors.items():
            mask = label_mask == lbl
            color_mask[mask] = color

        fig, axes = plt.subplots(1, 2, figsize=(18, 7))
        fig.suptitle("Segmentation Label Mask vs RGB", fontsize=14)

        axes[0].imshow(raster[:, :, 3:6], origin='lower')
        axes[0].set_title("RGB Composite")
        axes[0].axis('off')

        axes[1].imshow(color_mask, origin='lower')
        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='gray',   label='unlabeled'),
            Patch(facecolor='dimgray',label='road'),
            Patch(facecolor='cyan',   label='gutter'),
            Patch(facecolor='red',    label='curb'),
            Patch(facecolor='orange', label='sidewalk'),
            Patch(facecolor='green',  label='grass/soil'),
        ]
        axes[1].legend(handles=legend_elements, loc='lower right', fontsize=9)
        axes[1].set_title("Label Mask")
        axes[1].axis('off')

        plt.tight_layout()
        mask_path = os.path.join(output_dir, "label_mask_vs_rgb.png")
        plt.savefig(mask_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"      Saved: {mask_path}")

    # Save raster as numpy for later use
    npy_path = os.path.join(output_dir, "raster.npy")
    np.save(npy_path, raster)
    print(f"      Saved: {npy_path}  (shape={raster.shape})")

    if label_mask is not None:
        mask_npy_path = os.path.join(output_dir, "label_mask.npy")
        np.save(mask_npy_path, label_mask)
        print(f"      Saved: {mask_npy_path}")

    print("\n✓ Done. Check the output folder for visualizations.")
    print("  KEY THING TO CHECK: In 'verticality_highlight.png',")
    print("  you should see bright lines exactly where the curb faces are.")
    print("  If you do → the 2.5D approach will work.")
    print("  If you don't → we need to adjust the k_neighbors or resolution.")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LAZ Tile Audit & 2.5D Rasterization")
    parser.add_argument("--input",      required=True,  help="Path to .laz or .las file")
    parser.add_argument("--resolution", type=float, default=0.05,
                        help="Raster resolution in meters (default: 0.05m = 5cm)")
    parser.add_argument("--k",          type=int,   default=20,
                        help="Number of neighbors for feature computation (default: 20)")
    parser.add_argument("--output",     default="./raster_output",
                        help="Output directory for visualizations")
    parser.add_argument("--max_points", type=int,   default=2_000_000,
                        help="Subsample to this many points for speed during audit (default: 2M)")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"ERROR: File not found: {args.input}")
        sys.exit(1)

    # Load
    las = load_laz(args.input)

    # Subsample if tile is very large (for audit speed)
    n_pts = len(las.points)
    if n_pts > args.max_points:
        print(f"\n      Subsampling from {n_pts:,} → {args.max_points:,} points for speed")
        print(f"      (Use --max_points {n_pts} to process all points)")
        idx = np.random.choice(n_pts, args.max_points, replace=False)
        # We'll work with numpy arrays directly
        las_x = np.array(las.x)[idx]
        las_y = np.array(las.y)[idx]
        las_z = np.array(las.z)[idx]

        # Rebuild a simple namespace for downstream
        class SubLas:
            pass
        sub = SubLas()
        sub.x = las_x; sub.y = las_y; sub.z = las_z
        sub.point_format = las.point_format
        sub.header = las.header
        sub.points = las.points[idx]
        if hasattr(las, 'red'):
            sub.red   = np.array(las.red)[idx]
            sub.green = np.array(las.green)[idx]
            sub.blue  = np.array(las.blue)[idx]
        if hasattr(las, 'intensity'):
            sub.intensity = np.array(las.intensity)[idx]
        if hasattr(las, 'classification'):
            sub.classification = np.array(las.classification)[idx]
        las = sub

    # Audit
    info = audit_tile(las)

    # Features
    verticality, height_above_ground, normal_z = compute_features(las, info, k_neighbors=args.k)

    # Rasterize
    raster, raster_meta = rasterize(las, info, verticality, height_above_ground, resolution=args.resolution)

    # Label mask
    label_mask = build_label_mask(las, info, raster_meta)

    # Visualize
    visualize_and_save(raster, label_mask, args.output, args.resolution)


if __name__ == "__main__":
    main()
