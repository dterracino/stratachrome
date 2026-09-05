"""
pipeline.py
-----------
End-to-end command-line runner for Stratachrome. Segments an image, computes
optical lightness using color_tools, maps two-tier depth with solid pedestal
enforcement, constructs a manifold mesh, and packages a Bambu-compatible 3MF.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from PIL import Image

from stratachrome.color_engine import (
    build_layer_schedule,
    create_filament,
    extract_perceptual_lightness,
)
from stratachrome.depth_mapper import TwoTierDepthMapper
from stratachrome.export_3mf import export_bambu_3mf
from stratachrome.mesh_builder import PhysicalDimensions, WatertightMeshBuilder
from stratachrome.optical_model import (
    LightnessLayerMapper,
    simulate_tier_stack,
)
from stratachrome.segmentation import (
    ForegroundSegmenter,
    SegmentationConfig,
    partition_lightness_channels,
)


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stratachrome: Two-Tier Automated Multi-Color 3MF Generator."
    )
    parser.add_argument("-i", "--input", type=Path, required=True, help="Input RGB image path.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("output/project.3mf"),
        help="Output 3MF path (default: output/project.3mf).",
    )
    parser.add_argument(
        "-s",
        "--size",
        type=float,
        default=150.0,
        help="Target physical size in mm for the maximum image dimension (default: 150.0).",
    )
    parser.add_argument(
        "--max-dim",
        type=int,
        default=1000,
        help="Maximum raster dimension for mesh grid (default: 1000).",
    )
    parser.add_argument(
        "--layer-height",
        type=float,
        default=0.10,
        help="Standard layer step height in mm (default: 0.10).",
    )
    parser.add_argument(
        "--first-layer",
        type=float,
        default=0.20,
        help="First layer bed-contact height in mm (default: 0.20).",
    )
    parser.add_argument(
        "--swap-mode",
        type=str,
        choices=["ams", "manual"],
        default="ams",
        help="Filament change mode: 'ams' for multi-material auto-switching, 'manual' for single-extruder pause triggers (default: ams).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device ('cuda', 'cpu'). Auto-detected if omitted.",
    )
    return parser.parse_args()


def _load_and_rescale_image(path: Path, max_dim: int) -> Image.Image:
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    with Image.open(path) as img:
        img = img.convert("RGB")
        w, h = img.size
        if max(w, h) > max_dim:
            scale = max_dim / max(w, h)
            new_size = (int(round(w * scale)), int(round(h * scale)))
            img = img.resize(new_size, Image.Resampling.BILINEAR)
        return img


def main() -> int:
    args = _parse_arguments()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"1. Loading image '{args.input.name}' (constrained to max {args.max_dim}px)...")
    source_image = _load_and_rescale_image(args.input, args.max_dim)
    w_px, h_px = source_image.size

    if w_px >= h_px:
        width_mm = args.size
        height_mm = round(args.size * (h_px / w_px), 2)
    else:
        height_mm = args.size
        width_mm = round(args.size * (w_px / h_px), 2)

    print(f"   Image grid: {w_px}x{h_px} -> Print size: {width_mm:.1f}mm x {height_mm:.1f}mm")
    print(f"   Vertical resolution: First layer={args.first_layer:.2f}mm, Step={args.layer_height:.2f}mm")
    print(f"   Swap mode: {args.swap_mode.upper()}")

    print("2. Extracting alpha matte via BiRefNet...")
    segmenter = ForegroundSegmenter(SegmentationConfig(device=args.device, feather_radius=2))
    seg_result = segmenter.extract_matte(source_image)

    print("3. Extracting CIELCh lightness matrix via color_tools...")
    full_l = extract_perceptual_lightness(source_image)
    bg_l, fg_l = partition_lightness_channels(full_l, seg_result.matte)

    print("4. Calibrating filament optical models via color_tools...")
    bg_raw = [
        create_filament("Bambu Charcoal", "#1F1F1F", td=0.6),
        create_filament("Bambu Ash Gray", "#757575", td=2.0),
        create_filament("Bambu Jade White", "#F5F5F5", td=5.0),
    ]
    bg_filaments = build_layer_schedule(bg_raw, layer_steps=[0, 5, 9])
    bg_states = simulate_tier_stack(
        bg_filaments,
        total_layers=12,
        step_height_mm=args.layer_height,
        first_layer_height_mm=args.first_layer,
    )
    bg_mapper = LightnessLayerMapper(bg_states)

    fg_raw = [
        create_filament("Bambu Dark Blue", "#0B2545", td=0.8),
        create_filament("Bambu Crimson", "#8B0000", td=1.8),
        create_filament("Bambu Butter Yellow", "#FFD166", td=3.5),
    ]
    fg_filaments = build_layer_schedule(fg_raw, layer_steps=[0, 4, 8])
    fg_states = simulate_tier_stack(
        fg_filaments,
        total_layers=12,
        step_height_mm=args.layer_height,
        first_layer_height_mm=args.layer_height,
        initial_substrate_l=bg_states[-1].simulated_l,
    )
    fg_mapper = LightnessLayerMapper(fg_states)

    print("5. Generating two-tier heightmap with solid pedestal support...")
    depth_mapper = TwoTierDepthMapper(
        bg_mapper=bg_mapper,
        fg_mapper=fg_mapper,
        bg_states=bg_states,
        fg_states=fg_states,
        step_height_mm=args.layer_height,
        first_layer_height_mm=args.first_layer,
    )
    height_result = depth_mapper.generate_heightmap(bg_l, fg_l, seg_result.matte)
    print(f"   Max height: {height_result.max_height_mm:.2f}mm ({height_result.total_layers} layers)")

    print("6. Building watertight manifold triangle mesh...")
    dimensions = PhysicalDimensions(width_mm=width_mm, height_mm=height_mm, base_floor_z_mm=0.0)
    mesh_builder = WatertightMeshBuilder(dimensions)
    mesh = mesh_builder.build_mesh(height_result.z_grid)
    print(f"   Vertices: {mesh.vertex_count:,} | Triangles: {mesh.face_count:,}")

    print(f"7. Packaging Bambu Studio / Orca Slicer 3MF into '{args.output}'...")
    export_bambu_3mf(mesh, height_result.swap_schedule, args.output, swap_mode=args.swap_mode)

    print("\n✓ Stratachrome export complete!")
    if args.swap_mode == "manual":
        print("Embedded layer pause schedule (Manual Mode):")
        for swap in height_result.swap_schedule:
            print(f"  - Layer {swap.global_layer_idx:2d} @ {swap.z_height_mm:5.2f}mm -> {swap.filament_name} ({swap.tier_name})")
    else:
        print("Configured for AMS multi-material execution (no manual pauses required).")

    return 0


if __name__ == "__main__":
    sys.exit(main())