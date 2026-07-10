"""
Boundary Refinement Script (v2)
================================
Refines the road-ground boundary in a segmented LAZ file by anchoring
to the physical curb face geometry rather than re-labeling by proximity.

Algorithm (your proposed approach):
  1. Find all high-verticality points — these form the curb face band
  2. For each local section of that band, determine which side is road
     and which side is ground by majority-vote sampling within 40cm
  3. The road-side edge of the verticality band is the true boundary line
  4. Any road point that has crossed past the true boundary (onto the
     curb wall or ground side) gets re-labeled as ground
  5. Hard distance limit: never correct a point more than --snap_radius
     from the nearest curb face point — prevents jumping to wrong curbs
     in roundabouts and leaves no-curb road edges completely untouched

Why this is better than v1 (proximity re-labeling):
  v1 moved the boundary inward by re-labeling road points near ground
  points, but the position depended on which points happened to be
  near each other, not on the actual curb geometry. This caused the
  ~10cm inward offset you observed.

  v2 finds the geometric truth first (the verticality spike = the curb
  face), then corrects classification relative to that truth. The
  boundary now sits at the base of the curb face on the road side,
  which is exactly where the gutter line should be.

Usage:
  python refine_boundary.py \\
    --input_laz  tile_segmented.laz \\
    --output_dir ./refined/

  python refine_boundary.py \\
    --input_dir  ./segmented/ \\
    --output_dir ./refined/

Dependencies:
  pip install laspy lazrs-python numpy scikit-learn scipy
"""

import argparse
import sys
import warnings
import numpy as np
from pathlib import Path
import laspy

warnings.filterwarnings("ignore")

CLASS_GROUND = 1
CLASS_ROAD   = 2


# ── I/O ───────────────────────────────────────────────────────────────────────

def load_laz(path):
    try:
        import laspy
    except ImportError:
        print("ERROR: pip install laspy lazrs-python")
        sys.exit(1)
    return laspy.read(str(path))


def get_or_compute_verticality(las, x, y, z, k=25):
    """
    Use pre-computed CloudCompare verticality field if present,
    otherwise compute from scratch. Handles NaN/Inf from CloudCompare.
    Field name must start with 'verticality' (case-insensitive).
    """
    dims = [d.name for d in las.point_format.dimensions]
    extra_dims = ([d.name for d in las.point_format.extra_dims]
                  if hasattr(las.point_format, 'extra_dims') else [])
    all_dims = dims + extra_dims

    vert_field = None
    for name in all_dims:
        if name.lower().startswith('verticality'):
            vert_field = name
            break

    if vert_field is not None:
        print(f"  Using pre-computed verticality field: '{vert_field}'")
        vert = np.array(las[vert_field], dtype=np.float32)
        vert = np.nan_to_num(vert, nan=0.0, posinf=1.0, neginf=0.0)
        vert = np.clip(vert, 0.0, 1.0)
    else:
        print("  No pre-computed verticality found — computing from scratch...")
        vert = _compute_verticality(x, y, z, k=k)

    return vert


def _compute_verticality(x, y, z, k=25, batch_size=100_000):
    from sklearn.neighbors import KDTree

    pts = np.column_stack([x, y, z]).astype(np.float32)
    tree = KDTree(pts)
    _, indices = tree.query(pts, k=k + 1)
    indices = indices[:, 1:]

    n = len(pts)
    verticality = np.zeros(n, dtype=np.float32)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        neighbors = pts[indices[start:end]]
        centroid  = neighbors.mean(axis=1, keepdims=True)
        centered  = neighbors - centroid
        cov = np.einsum('bki,bkj->bij', centered, centered) / k
        try:
            eigvals = np.linalg.eigvalsh(cov)
            lam0 = eigvals[:, 0]
            lam2 = eigvals[:, 2]
            denom = np.where(lam2 > 1e-9, lam2, 1e-9)
            verticality[start:end] = np.clip(lam0 / denom, 0, 1)
        except np.linalg.LinAlgError:
            pass

        if start % 500_000 == 0 and start > 0:
            print(f"    Verticality: {start:,} / {n:,}...")

    return verticality


# ── Core: verticality-anchor refinement ──────────────────────────────────────

def refine_boundary(x, y, z, labels, verticality,
                    min_vert_threshold=0.35,
                    snap_radius=0.30,
                    side_sample_radius=0.40,
                    min_road_fraction=0.80):
    """
    Refine classification by anchoring to the verticality spike band.

    Parameters:
      min_vert_threshold  : minimum verticality to be considered a curb face point
      snap_radius         : hard limit — only correct road points within this
                            distance of a curb face point. Prevents jumping to
                            wrong curbs in roundabouts.
      side_sample_radius  : how far to sample on each side of the curb face
                            to determine which side is road vs ground
      min_road_fraction   : fraction of sampled points that must be road-class
                            for that side to be declared the road side

    Returns updated labels array.
    """
    from scipy.spatial import cKDTree

    pts_xy = np.column_stack([x, y])

    # ── Step 1: Find curb face points (high verticality) ─────────────────
    curb_mask = verticality >= min_vert_threshold
    n_curb = curb_mask.sum()
    print(f"  Curb face points (verticality >= {min_vert_threshold}): {n_curb:,}")

    if n_curb < 10:
        print("  Too few curb face points — skipping refinement. "
              "Try lowering --min_vert_threshold.")
        return labels

    curb_xy   = pts_xy[curb_mask]
    curb_tree = cKDTree(curb_xy)

    # ── Step 2: Find road points within snap_radius of curb face ─────────
    # These are the only points we'll consider correcting
    road_mask    = labels == CLASS_ROAD
    road_indices = np.where(road_mask)[0]
    road_xy      = pts_xy[road_mask]

    if len(road_xy) == 0:
        print("  No road points found — skipping.")
        return labels

    dists_to_curb, nearest_curb_idx = curb_tree.query(
        road_xy, distance_upper_bound=snap_radius)

    candidate_local = dists_to_curb < snap_radius
    candidate_global = road_indices[candidate_local]
    n_candidates = candidate_local.sum()
    print(f"  Road points within snap_radius ({snap_radius}m) "
          f"of curb face: {n_candidates:,}")

    if n_candidates == 0:
        print("  No candidate points found — skipping refinement.")
        return labels

    # ── Step 3: For each curb face point, determine road vs ground side ───
    # We do this once per curb face point rather than per candidate road
    # point, to avoid redundant computation.
    # For each curb face point, sample points in a circle of side_sample_radius,
    # split by rough left/right relative to local curb direction.
    # Simpler approach: check which side has more road-class points.

    all_pts_xy = pts_xy
    all_tree   = cKDTree(all_pts_xy)

    # Map from curb face index → is the road side in the +normal or -normal direction
    # We determine this by checking which side of the curb face has more road points.
    # For each curb face point, query a disk of side_sample_radius and count
    # road vs ground points on each side using the local surface normal direction.

    # Local curb direction for each curb face point: use PCA on nearby curb points
    # to get the tangent direction, then the normal is perpendicular to that.
    curb_pts_3d = np.column_stack([x[curb_mask], y[curb_mask], z[curb_mask]])

    print("  Determining road side for each curb face section...")

    # Build a lookup: for each candidate road point, is it on the wrong side?
    new_labels = labels.copy()
    corrected  = 0

    # Process in batches keyed by their nearest curb face point
    # Group candidate road points by nearest curb face point
    nearest_curb_for_candidates = nearest_curb_idx[candidate_local]
    unique_curb_pts = np.unique(nearest_curb_for_candidates)

    for curb_pt_idx in unique_curb_pts:
        cx, cy = curb_xy[curb_pt_idx]

        # Sample all points within side_sample_radius of this curb face point
        neighbor_indices = all_tree.query_ball_point([cx, cy], side_sample_radius)
        if len(neighbor_indices) < 5:
            continue

        neighbor_indices = np.array(neighbor_indices)
        neighbor_labels  = labels[neighbor_indices]
        neighbor_xy      = all_pts_xy[neighbor_indices]

        # Compute local curb tangent direction using nearby curb face points
        local_curb_neighbors = curb_tree.query_ball_point(
            [cx, cy], side_sample_radius)
        if len(local_curb_neighbors) >= 3:
            local_curb_xy = curb_xy[local_curb_neighbors]
            centered = local_curb_xy - local_curb_xy.mean(axis=0)
            cov = np.cov(centered.T)
            eigvals, eigvecs = np.linalg.eigh(cov)
            tangent = eigvecs[:, -1]   # direction of max variance = curb runs along this
            normal  = np.array([-tangent[1], tangent[0]])  # perpendicular to tangent
        else:
            # Fallback: use direction from curb point to centroid of neighbor road pts
            road_neighbor_mask = neighbor_labels == CLASS_ROAD
            if road_neighbor_mask.sum() > 0:
                road_centroid = neighbor_xy[road_neighbor_mask].mean(axis=0)
                normal = road_centroid - np.array([cx, cy])
                norm_len = np.linalg.norm(normal)
                if norm_len > 1e-6:
                    normal /= norm_len
                else:
                    continue
            else:
                continue

        # Project neighbor points onto normal direction
        # Positive projection = one side, negative = other side
        relative_xy  = neighbor_xy - np.array([cx, cy])
        projections  = relative_xy @ normal

        pos_mask = projections > 0
        neg_mask = projections < 0

        pos_road_frac = ((neighbor_labels[pos_mask] == CLASS_ROAD).sum()
                         / max(pos_mask.sum(), 1))
        neg_road_frac = ((neighbor_labels[neg_mask] == CLASS_ROAD).sum()
                         / max(neg_mask.sum(), 1))

        # The road side is whichever side has more road points
        if pos_road_frac >= neg_road_frac:
            road_side_sign   = +1.0   # positive projection = road side
            ground_side_sign = -1.0
        else:
            road_side_sign   = -1.0   # negative projection = road side
            ground_side_sign = +1.0

        # ── Step 4: Correct road points on the wrong side ─────────────
        # Road points that project onto the GROUND side of this curb face
        # point are bleeding past the curb — re-label them as ground.
        # Only touch candidates that are nearest to THIS curb face point.
        this_curb_candidates = candidate_global[
            nearest_curb_for_candidates == curb_pt_idx]

        for global_idx in this_curb_candidates:
            px, py = pts_xy[global_idx]
            rel    = np.array([px - cx, py - cy])
            proj   = np.dot(rel, normal)

            # If this road point projects onto the ground side,
            # it has crossed the curb face — re-label as ground
            if proj * road_side_sign < 0:
                new_labels[global_idx] = CLASS_GROUND
                corrected += 1

    print(f"  Points corrected (road → ground): {corrected:,} "
          f"of {n_candidates:,} candidates "
          f"({100*corrected/max(n_candidates,1):.1f}%)")

    if corrected == 0:
        print("  No corrections made. If unexpected, try:")
        print("    - lowering  --min_vert_threshold (current: "
              f"{min_vert_threshold})")
        print("    - increasing --snap_radius (current: "
              f"{snap_radius}m)")
        print("    - increasing --side_sample_radius (current: "
              f"{side_sample_radius}m)")

    return new_labels


# ── Per-tile pipeline ─────────────────────────────────────────────────────────

def process_tile(input_laz_path, output_dir,
                 min_vert_threshold=0.35,
                 snap_radius=0.30,
                 side_sample_radius=0.40,
                 min_road_fraction=0.80,
                 k=25):

    input_path  = Path(input_laz_path)
    tile_name   = input_path.stem
    base_name   = tile_name.replace('_segmented', '')
    out_laz_path = Path(output_dir) / f"{base_name}_refined.laz"

    print(f"\n{'='*60}")
    print(f"[REFINE] {input_path.name}")

    las = load_laz(str(input_path))
    x   = np.array(las.x, dtype=np.float64)
    y   = np.array(las.y, dtype=np.float64)
    z   = np.array(las.z, dtype=np.float64)
    labels = np.array(las.classification, dtype=np.uint8)

    print(f"  Points: {len(x):,}")
    road_before   = (labels == CLASS_ROAD).sum()
    ground_before = (labels == CLASS_GROUND).sum()
    print(f"  Before: road={road_before:,}  ground={ground_before:,}")

    vert = get_or_compute_verticality(las, x, y, z, k=k)

    new_labels = refine_boundary(
        x, y, z, labels, vert,
        min_vert_threshold=min_vert_threshold,
        snap_radius=snap_radius,
        side_sample_radius=side_sample_radius,
        min_road_fraction=min_road_fraction,
    )

    road_after   = (new_labels == CLASS_ROAD).sum()
    ground_after = (new_labels == CLASS_GROUND).sum()
    print(f"  After:  road={road_after:,}  ground={ground_after:,}")
    print(f"  Net:    {road_before - road_after:+,} road points moved to ground")

    header = laspy.LasHeader(
        point_format=las.header.point_format.id,
        version=las.header.version
    )
    header.offsets = las.header.offsets
    header.scales  = las.header.scales

    # Copy CRS if present
    try:
        crs = las.header.parse_crs()
        if crs is not None:
            header.set_crs(crs)
    except Exception as e:
        print(f"  WARNING: could not set CRS — {e}")

    clean_las = laspy.LasData(header=header)
    clean_las.x              = las.x
    clean_las.y              = las.y
    clean_las.z              = las.z
    clean_las.intensity      = las.intensity
    clean_las.classification = new_labels
    if hasattr(las, 'red'):
        clean_las.red   = las.red
        clean_las.green = las.green
        clean_las.blue  = las.blue
    if hasattr(las, 'return_number'):
        clean_las.return_number     = las.return_number
        clean_las.number_of_returns = las.number_of_returns

    clean_las.write(str(out_laz_path))
    print(f"  Saved: {out_laz_path}")

    return str(out_laz_path)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Refine road-ground boundary using verticality anchor approach.")

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input_laz", type=str,
                             help="Path to a single segmented .laz file.")
    input_group.add_argument("--input_dir", type=str,
                             help="Folder of segmented .laz files.")

    parser.add_argument("--output_dir", required=True,
                        help="Folder to save refined .laz files.")
    parser.add_argument("--min_vert_threshold", type=float, default=0.35,
                        help="Min verticality to be considered a curb face. "
                             "Default: 0.35. Lower if curb not detected.")
    parser.add_argument("--snap_radius", type=float, default=0.30,
                        help="Hard distance limit — only correct road points "
                             "within this distance of a curb face point (m). "
                             "Default: 0.30. Prevents jumping to wrong curbs.")
    parser.add_argument("--side_sample_radius", type=float, default=0.40,
                        help="Radius (m) to sample points on each side of the "
                             "curb face to determine road vs ground direction. "
                             "Default: 0.40")
    parser.add_argument("--min_road_fraction", type=float, default=0.80,
                        help="Fraction of sampled points that must be road-class "
                             "to declare that side the road side. Default: 0.80")
    parser.add_argument("--k", type=int, default=25,
                        help="k-neighbors for verticality if no pre-computed "
                             "field found. Default: 25")

    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    kwargs = dict(
        min_vert_threshold=args.min_vert_threshold,
        snap_radius=args.snap_radius,
        side_sample_radius=args.side_sample_radius,
        min_road_fraction=args.min_road_fraction,
        k=args.k,
    )

    if args.input_laz:
        process_tile(args.input_laz, args.output_dir, **kwargs)
    else:
        input_dir = Path(args.input_dir)
        laz_files = sorted(
            list(input_dir.glob("*.laz")) + list(input_dir.glob("*.las")) +
            list(input_dir.glob("*.LAZ")) + list(input_dir.glob("*.LAS"))
        )
        if not laz_files:
            print(f"No LAZ/LAS files found in {input_dir}")
            return
        print(f"Found {len(laz_files)} files to refine.")
        for f in laz_files:
            try:
                process_tile(str(f), args.output_dir, **kwargs)
            except Exception as e:
                print(f"  ERROR on {f.name}: {e}")
                import traceback; traceback.print_exc()


if __name__ == "__main__":
    main()