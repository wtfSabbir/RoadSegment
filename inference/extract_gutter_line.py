import os
import json
import laspy
import warnings
import argparse
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN
from scipy.signal import savgol_filter

warnings.filterwarnings("ignore")

CLASS_STREET = 1
CLASS_CURB = 2

def load_laz(path: str):
    las = laspy.read(path)
    coords = np.stack([las.x, las.y, las.z], axis=1).astype(np.float64)
    classification = np.array(las.classification, dtype=np.int32)
    
    # extract EPSG dynamically
    epsg_code = None
    try:
        crs = las.header.parse_crs()
        if crs is not None:
            epsg_code = crs.to_epsg()
    except Exception as e:
        print(f"Could not parse CRS from header: {e}")

    if epsg_code is None:
        print("No valid EPSG found in LAZ file. Defaulting to 3945.")
        epsg_code = 3945
    else:
        print(f"Successfully extracted EPSG:{epsg_code} from LAZ header.")

    return coords, classification, epsg_code

def extract_proximity_boundaries(street_pts, curb_pts, search_radius=0.15):
    """Idea 1: Uses a KD-Tree to find where street and curb touch."""
    print(f"\n[1] Building KD-Tree for {len(curb_pts):,} Curb points...")
    curb_tree = cKDTree(curb_pts[:, :2])  # 2D tree for XY distance

    print(f"[2] Querying {len(street_pts):,} Street points against Curb...")
    dists, indices = curb_tree.query(street_pts[:, :2], distance_upper_bound=search_radius)

    valid_mask = dists < search_radius
    border_street = street_pts[valid_mask]
    nearest_curb = curb_pts[indices[valid_mask]]

    midpoints = (border_street + nearest_curb) / 2.0
    print(f"    -> Found {len(midpoints):,} raw boundary midpoints.")
    return midpoints

def order_points_into_line(points):
    """Sorts a scatter of points into a connected line using a Nearest-Neighbor walk."""
    if len(points) < 3:
        return points
        
    start_idx = np.argmin(points[:, 0])
    ordered = [points[start_idx]]
    unvisited = np.delete(points, start_idx, axis=0)
    
    while len(unvisited) > 0:
        last_pt = ordered[-1]
        dists = np.linalg.norm(unvisited[:, :2] - last_pt[:2], axis=1)
        next_idx = np.argmin(dists)
        
        # If the next point is more than 2 meters away, it's a gap (break the line)
        if dists[next_idx] > 2.0:
            break
            
        ordered.append(unvisited[next_idx])
        unvisited = np.delete(unvisited, next_idx, axis=0)
        
    return np.array(ordered)

def process_and_smooth(midpoints, grid_size=0.15, eps=1.5, min_samples=5):
    """Thins out the dense band of points and smooths them into lines."""
    print("[3] Clustering separate boundaries (e.g., left vs right side)...")
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(midpoints[:, :2])
    
    lines = []
    for lbl in set(labels):
        if lbl == -1: continue # Skip noise
        
        cluster_pts = midpoints[labels == lbl]
        
        # THINNING (Voxel Downsampling)
        coords_scaled = np.round(cluster_pts / grid_size).astype(int)
        _, unique_indices = np.unique(coords_scaled, axis=0, return_index=True)
        thinned_pts = cluster_pts[unique_indices]
        
        # ORDERING 
        ordered_pts = order_points_into_line(thinned_pts)
        
        if len(ordered_pts) > 10:
            # SMOOTHING
            window = min(15, len(ordered_pts) - (len(ordered_pts)%2 == 0))
            sx = savgol_filter(ordered_pts[:, 0], window, 3)
            sy = savgol_filter(ordered_pts[:, 1], window, 3)
            sz = savgol_filter(ordered_pts[:, 2], min(window, 5), 2)
            lines.append(np.stack([sx, sy, sz], axis=1))
            
    return lines

def process_single_file(input_laz, output_dir, search_radius, grid_size, eps, min_samples):
    """Wraps the core logic to process a single file and save the output."""
    input_path = Path(input_laz)
    output_path = Path(output_dir) / f"{input_path.stem}.geojson"
    
    print(f"\n{'='*50}\nProcessing: {input_path.name}")
    
    coords, classification, epsg_code = load_laz(str(input_path))
    street_pts = coords[classification == CLASS_STREET]
    curb_pts = coords[classification == CLASS_CURB]

    # Run Pipeline
    midpoints = extract_proximity_boundaries(street_pts, curb_pts, search_radius=search_radius)
    
    if len(midpoints) == 0:
        print("No boundary points found. Skipping export.")
        return

    final_lines = process_and_smooth(midpoints, grid_size=grid_size, eps=eps, min_samples=min_samples)
    
    print(f"\n[4] Exporting {len(final_lines)} continuous lines to {output_path.name}...")
    
    # Export to GeoJSON
    features = []
    for i, line in enumerate(final_lines):
        coords_list = [[float(x), float(y), float(z)] for x, y, z in line]
        features.append({
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": coords_list},
            "properties": {"Layer": "VOTROTTO_FE", "source": "Idea1_Proximity"}
        })
        
    with open(output_path, "w") as f:
        json.dump({
            "type": "FeatureCollection", 
            "crs": {"type": "name", "properties": {"name": f"urn:ogc:def:crs:EPSG::{epsg_code}"}},
            "features": features
        }, f, indent=2)
    print("Done!")

def main():
    parser = argparse.ArgumentParser(description="Extract Curb-Gutter boundaries from LAZ files.")
    
    # Mutually exclusive group: Must provide EITHER a single file OR a directory
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--single_file", type=str, help="Path to a single .laz file.")
    input_group.add_argument("--input_dir", type=str, help="Path to a folder containing .laz files.")
    
    # Output directory
    parser.add_argument("--output_dir", type=str, required=True, help="Folder to save the .geojson outputs.")
    
    # Optional Algorithmic Parameters
    parser.add_argument("--search_radius", type=float, default=0.25, help="Max distance (m) to find curb from street. Default: 0.20")
    parser.add_argument("--grid_size", type=float, default=0.20, help="Voxel size (m) for thinning points. Default: 0.15")
    parser.add_argument("--eps", type=float, default=0.5, help="DBSCAN max distance to connect a line. Default: 1.5")
    parser.add_argument("--min_samples", type=int, default=3, help="DBSCAN min points to form a valid line. Default: 5")

    args = parser.parse_args()

    # Ensure output directory exists
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.single_file:
        process_single_file(args.single_file, args.output_dir, args.search_radius, args.grid_size, args.eps, args.min_samples)
        
    elif args.input_dir:
        input_dir = Path(args.input_dir)
        # Find all .laz or .las files in the directory
        laz_files = list(input_dir.glob("*.laz")) + list(input_dir.glob("*.las"))
        
        if not laz_files:
            print(f"No .laz or .las files found in {args.input_dir}")
            return
            
        print(f"Found {len(laz_files)} files to process in {args.input_dir}")
        for laz_file in laz_files:
            process_single_file(laz_file, args.output_dir, args.search_radius, args.grid_size, args.eps, args.min_samples)

if __name__ == "__main__":
    main()