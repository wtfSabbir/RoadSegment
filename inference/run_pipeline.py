"""
Pipeline Orchestrator
======================
Chains inference → boundary refinement → vectorisation in a single command.

By default only inference runs. Add flags to enable additional stages:
  --refine_boundary    Run geometric boundary refinement after inference
  --vectorisation      Run gutter line extraction on the final LAZ

Output files:
  Inference:        {output_dir}/{tile}_segmented.laz + PNG visualizations
  + refine:         {output_dir}/{tile}_refined.laz
  + vectorisation:  {output_dir}/{tile}.geojson

Use --clean_intermediates to delete intermediate LAZ files after completion.

Usage examples:

  # Inference only
  python run_pipeline.py \\
    --model_folder /path/to/checkpoints/MyRun/ \\
    --single_file  /path/to/tile.laz \\
    --output_dir   /path/to/output/

  # Full pipeline
  python run_pipeline.py \\
    --model_folder /path/to/checkpoints/MyRun/ \\
    --input_folder /path/to/raw/tiles/ \\
    --output_dir   /path/to/output/ \\
    --refine_boundary \\
    --vectorisation

  # Inference + vectorisation, skip refinement
  python run_pipeline.py \\
    --model_folder /path/to/checkpoints/MyRun/ \\
    --single_file  /path/to/tile.laz \\
    --output_dir   /path/to/output/ \\
    --vectorisation
"""

import argparse
import json
import sys
import traceback
from pathlib import Path
from datetime import datetime
import laspy


# ── Stage imports ─────────────────────────────────────────────────────────────

def _import_inference():
    try:
        import inference
        return inference
    except ImportError as e:
        print(f"ERROR: Could not import inference.py — {e}")
        sys.exit(1)

def _import_refine():
    try:
        import refine_boundary
        return refine_boundary
    except ImportError as e:
        print(f"ERROR: Could not import refine_boundary.py — {e}")
        sys.exit(1)

def _import_vectorise():
    try:
        import extract_gutter_line
        return extract_gutter_line
    except ImportError as e:
        print(f"ERROR: Could not import extract_gutter_line.py — {e}")
        sys.exit(1)


# ── Logging ───────────────────────────────────────────────────────────────────

def _banner(stage_name, tile_name):
    width = 60
    print(f"\n{'='*width}")
    print(f"  STAGE : {stage_name}")
    print(f"  TILE  : {tile_name}")
    print(f"  TIME  : {datetime.now().strftime('%H:%M:%S')}")
    print('='*width)

def _stage_done(stage_name, output_path):
    print(f"\n  ✓ {stage_name} complete → {Path(output_path).name}")

def _stage_skip(stage_name, reason):
    print(f"\n  — {stage_name} SKIPPED: {reason}")


# ── Single tile ───────────────────────────────────────────────────────────────

def run_single_tile(laz_path, output_dir, args,
                    inference_mod, refine_mod, vectorise_mod,
                    inference_kwargs, refine_kwargs, vectorise_kwargs):

    tile_name  = Path(laz_path).stem
    output_dir = Path(output_dir)
    current_laz = None

    # ── Stage 1: Inference ──────────────────────────────────────────────
    _banner("INFERENCE", tile_name)
    try:
        inference_mod.process_tile(
            laz_path=str(laz_path),
            **inference_kwargs,
        )
        segmented_laz = output_dir / f"{tile_name}_segmented.laz"
        current_laz = segmented_laz
        _stage_done("Inference", current_laz)
    except Exception as e:
        print(f"\n  ERROR in Inference stage: {e}")
        traceback.print_exc()
        print("  Pipeline aborted for this tile.")
        return None

    # ── Stage 2: Boundary Refinement ────────────────────────────────────
    if args.refine_boundary:
        _banner("BOUNDARY REFINEMENT", tile_name)
        try:
            refined_path = refine_mod.process_tile(
                input_laz_path=str(current_laz),
                output_dir=str(output_dir),
                **refine_kwargs,
            )
            current_laz = Path(refined_path)
            _stage_done("Boundary Refinement", current_laz)
        except Exception as e:
            print(f"\n  ERROR in Boundary Refinement stage: {e}")
            traceback.print_exc()
            print("  Continuing with segmented LAZ.")
    else:
        _stage_skip("Boundary Refinement", "use --refine_boundary to enable")

    # ── Stage 3: Vectorisation ───────────────────────────────────────────
    if args.vectorisation:
        _banner("VECTORISATION", tile_name)
        try:
            # extract_gutter_line.py uses process_single_file (not process_tile)
            vectorise_mod.process_single_file(
                input_laz=str(current_laz),
                output_dir=str(output_dir),
                **vectorise_kwargs,
            )
            base = Path(current_laz).stem.replace('_refined', '').replace('_segmented', '')
            geojson_path = output_dir / f"{base}.geojson"
            _stage_done("Vectorisation", geojson_path)
        except Exception as e:
            print(f"\n  ERROR in Vectorisation stage: {e}")
            traceback.print_exc()
    else:
        _stage_skip("Vectorisation", "use --vectorisation to enable")

    # ── Clean intermediates ──────────────────────────────────────────────
    if args.clean_intermediates:
        segmented = output_dir / f"{tile_name}_segmented.laz"
        base      = tile_name.replace('_segmented', '')
        refined   = output_dir / f"{base}_refined.laz"

        if args.refine_boundary and refined.exists() and segmented.exists():
            segmented.unlink()
            print(f"  Cleaned: {segmented.name}")

        if args.vectorisation and args.refine_boundary and refined.exists():
            refined.unlink()
            print(f"  Cleaned: {refined.name}")

    return str(current_laz)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Curb-gutter extraction pipeline orchestrator.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Input
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--single_file",  type=str,
                             help="Path to a single raw .laz file.")
    input_group.add_argument("--input_folder", type=str,
                             help="Folder of raw .laz files.")

    # Required
    parser.add_argument("--model_folder", required=True,
                        help="Run folder with config.json + checkpoint.")
    parser.add_argument("--output_dir",   required=True,
                        help="Folder to save all outputs.")

    # Stage flags
    parser.add_argument("--refine_boundary", action="store_true",
                        help="Run geometric boundary refinement after inference.")
    parser.add_argument("--vectorisation",   action="store_true",
                        help="Run gutter line extraction on the final LAZ.")

    # Inference args
    parser.add_argument("--checkpoint", default="last.pt",
                        choices=["best.pt", "last.pt"],
                        help="Checkpoint to load. Default: last.pt")

    # Refinement args
    parser.add_argument("--min_vert_threshold", type=float, default=0.35,
                        help="[Refine] Min verticality for a real curb face. "
                             "Default: 0.35")
    parser.add_argument("--snap_radius", type=float, default=0.30,
                        help="[Refine] Hard distance limit (m) — only correct "
                             "road points within this distance of a curb face. "
                             "Default: 0.30")
    parser.add_argument("--side_sample_radius", type=float, default=0.40,
                        help="[Refine] Radius (m) to sample points on each side "
                             "of the curb face. Default: 0.40")
    parser.add_argument("--min_road_fraction", type=float, default=0.80,
                        help="[Refine] Min road-class fraction to declare road "
                             "side. Default: 0.80")
    parser.add_argument("--refine_k", type=int, default=25,
                        help="[Refine] k-neighbors for verticality if no "
                             "pre-computed field. Default: 25")

    # Vectorisation args
    parser.add_argument("--search_radius", type=float, default=0.25,
                        help="[Vectorise] Road-to-ground search radius (m). "
                             "Default: 0.25")
    parser.add_argument("--grid_size",     type=float, default=0.20,
                        help="[Vectorise] Voxel thinning size (m). Default: 0.20")
    parser.add_argument("--eps",           type=float, default=0.5,
                        help="[Vectorise] DBSCAN radius (m). Default: 0.5")
    parser.add_argument("--min_samples",   type=int,   default=3,
                        help="[Vectorise] DBSCAN min points. Default: 3")

    # Output control
    parser.add_argument("--clean_intermediates", action="store_true",
                        help="Delete intermediate LAZ files after completion.")

    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Collect files
    if args.single_file:
        laz_files = [Path(args.single_file)]
        if not laz_files[0].exists():
            print(f"ERROR: File not found: {laz_files[0]}")
            sys.exit(1)
    else:
        input_dir = Path(args.input_folder)
        laz_files = sorted(
            list(input_dir.glob("*.laz")) + list(input_dir.glob("*.las")) +
            list(input_dir.glob("*.LAZ")) + list(input_dir.glob("*.LAS"))
        )
        if not laz_files:
            print(f"ERROR: No LAZ/LAS files found in {input_dir}")
            sys.exit(1)

    # Print plan
    print("\n" + "="*60)
    print("  CURB-GUTTER EXTRACTION PIPELINE")
    print("="*60)
    print(f"  Tiles        : {len(laz_files)}")
    print(f"  Model folder : {args.model_folder}")
    print(f"  Output dir   : {args.output_dir}")
    print(f"  Checkpoint   : {args.checkpoint}")
    print(f"\n  Stages:")
    print(f"    [✓] Inference")
    print(f"    [{'✓' if args.refine_boundary else '✗'}] Boundary Refinement")
    print(f"    [{'✓' if args.vectorisation   else '✗'}] Vectorisation")
    print("="*60)

    # Import modules
    inference_mod = _import_inference()
    refine_mod    = _import_refine()    if args.refine_boundary else None
    vectorise_mod = _import_vectorise() if args.vectorisation   else None

    # Load model once, reuse across all tiles
    import torch
    from model_arch import build_model

    model_folder = Path(args.model_folder)
    config_path  = model_folder / "config.json"
    if not config_path.exists():
        print(f"ERROR: {config_path} not found.")
        sys.exit(1)

    with open(config_path) as f:
        full_config = json.load(f)

    tc = full_config["training_config"]
    ic = full_config["inference_config"]

    print(f"\nLoaded config: {tc['run_name']}")
    print(f"  model_size={tc['model_size']} | "
          f"resolution={tc['resolution']}m | "
          f"patch_size={tc['patch_size']}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    model = build_model(
        model_size=tc["model_size"],
        num_channels=tc["num_channels"],
        num_labels=tc["num_labels"],
    )
    checkpoint_path = model_folder / args.checkpoint
    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    print(f"  Checkpoint: {checkpoint_path.name} "
          f"(epoch={ckpt.get('epoch')}, "
          f"val_iou={ckpt.get('val_iou', ckpt.get('best_iou', 0.0)):.4f})")

    patch_size = tc["patch_size"]
    if patch_size is None:
        print("ERROR: config.json patch_size is null.")
        sys.exit(1)

    inference_stride = ic.get("inference_stride", 256)

    # Build kwargs per stage
    inference_kwargs = dict(
        model=model,
        device=device,
        resolution=tc["resolution"],
        k=tc["k_neighbors"],
        patch_size=patch_size,
        inference_stride=inference_stride,
        output_dir=args.output_dir,
    )

    refine_kwargs = dict(
        min_vert_threshold=args.min_vert_threshold,
        snap_radius=args.snap_radius,
        side_sample_radius=args.side_sample_radius,
        min_road_fraction=args.min_road_fraction,
        k=args.refine_k,
    )

    vectorise_kwargs = dict(
        search_radius=args.search_radius,
        grid_size=args.grid_size,
        eps=args.eps,
        min_samples=args.min_samples,
    )

    # Process tiles
    total   = len(laz_files)
    success = 0
    failed  = 0
    start_time = datetime.now()

    for i, laz_path in enumerate(laz_files, 1):
        print(f"\n\n{'#'*60}")
        print(f"  Tile {i}/{total}: {laz_path.name}")
        print('#'*60)

        result = run_single_tile(
            laz_path=laz_path,
            output_dir=args.output_dir,
            args=args,
            inference_mod=inference_mod,
            refine_mod=refine_mod,
            vectorise_mod=vectorise_mod,
            inference_kwargs=inference_kwargs,
            refine_kwargs=refine_kwargs,
            vectorise_kwargs=vectorise_kwargs,
        )

        if result is not None:
            success += 1
        else:
            failed += 1

    elapsed = datetime.now() - start_time
    print(f"\n\n{'='*60}")
    print(f"  PIPELINE COMPLETE")
    print(f"  Succeeded : {success} / {total}")
    print(f"  Failed    : {failed} / {total}")
    print(f"  Elapsed   : {str(elapsed).split('.')[0]}")
    print(f"  Output    : {args.output_dir}")
    print('='*60)


if __name__ == "__main__":
    main()