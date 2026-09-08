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

import numpy as np
from PIL import Image
from color_tools import FilamentRecord

from stratachrome.color_engine import (
    extract_perceptual_lab,
    plan_tier_colors,
)
from stratachrome.depth_mapper import SingleTierDepthMapper, TwoTierDepthMapper
from stratachrome.bambu_exporter import export_bambu_project
from stratachrome.mesh_builder import PhysicalDimensions, WatertightMeshBuilder
from stratachrome.optical_model import (
    OptimizedTierSchedule,
)


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stratachrome: Automated Multi-Color 3MF Relief Generator."
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
        "--tier-mode",
        choices=("single", "two"),
        default="two",
        help=(
            "Relief mode: 'single' analyzes the full image without segmentation; "
            "'two' separates and stacks background/foreground tiers (default: two)."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device ('cuda', 'cpu'). Auto-detected if omitted.",
    )
    parser.add_argument(
        "--colors-per-tier",
        type=int,
        choices=tuple(range(2, 9)),
        default=4,
        help="Maximum filament colors per tier, from 2 to 8 (default: 4).",
    )
    parser.add_argument(
        "--max-layers-per-tier",
        type=int,
        default=120,
        help=(
            "Maximum total layers available to each tier; the optimizer may use "
            "fewer when extra thickness is not useful (default: 120)."
        ),
    )
    parser.add_argument(
        "--td-scale",
        type=float,
        default=1.0,
        help=(
            "Experimental multiplier applied to catalog transmission distances "
            "during optical simulation (default: 1.0)."
        ),
    )
    return parser.parse_args()


def _load_and_rescale_image(path: Path, max_dim: int) -> Image.Image:
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    with Image.open(path) as loaded_image:
        image = loaded_image.convert("RGB")
        w, h = image.size
        if max(w, h) > max_dim:
            scale = max_dim / max(w, h)
            new_size = (int(round(w * scale)), int(round(h * scale)))
            image = image.resize(new_size, Image.Resampling.BILINEAR)
        return image


def _filament_display_name(filament: FilamentRecord) -> str:
    """Return the full catalog identity used in schedule output."""
    return " ".join(
        part.strip()
        for part in (
            filament.maker,
            filament.type,
            filament.finish,
            filament.color,
        )
        if part and part.strip()
    )


def _print_tier_schedule(
    tier_name: str,
    schedule: OptimizedTierSchedule,
    max_layers: int,
) -> None:
    stopping_reason = (
        "tier budget reached" if len(schedule.states) == max_layers else "fit converged"
    )
    print(
        f"   {tier_name}: {len(schedule.assignments)} colors, "
        f"{len(schedule.states)}/{max_layers} layers ({stopping_reason}), "
        f"mean Delta E 2000={schedule.mean_delta_e:.2f}, "
        f"objective={schedule.objective_score:.2f}"
    )
    for assignment, layer_count in zip(schedule.assignments, schedule.layer_counts):
        filament = assignment.filament
        print(
            f"     L{assignment.start_layer:03d} +{layer_count:3d}  "
            f"{_filament_display_name(filament)} "
            f"({filament.hex}, TD={filament.td_value:g})"
        )


def _print_height_usage(z_grid: np.ndarray) -> None:
    """Report how much image area terminates at each printable elevation."""
    heights, counts = np.unique(z_grid, return_counts=True)
    total = int(np.sum(counts))
    print(f"   Used height levels: {len(heights)}")
    for height, count in zip(heights, counts):
        print(f"     Z={height:.2f}mm: {count:,} pixels ({100.0 * count / total:.2f}%)")


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
    print(
        f"   Vertical resolution: First layer={args.first_layer:.2f}mm, Step={args.layer_height:.2f}mm"
    )
    print(f"   Swap mode: {args.swap_mode.upper()} | Tier mode: {args.tier_mode.upper()}")
    print(f"   Optical TD scale: {args.td_scale:g}")

    if args.tier_mode == "two":
        from stratachrome.segmentation import ForegroundSegmenter, SegmentationConfig

        print("2. Extracting foreground matte via BiRefNet...")
        segmenter = ForegroundSegmenter(SegmentationConfig(device=args.device, feather_radius=2))
        matte = segmenter.extract_matte(source_image).matte
    else:
        print("2. Single-tier mode selected; skipping foreground extraction.")
        matte = np.zeros((h_px, w_px), dtype=np.float32)

    print("3. Converting the source image to CIELAB via color_tools...")
    full_lab = extract_perceptual_lab(source_image)

    tier_label = "background" if args.tier_mode == "two" else "single"
    print(f"4. Selecting and optimizing {tier_label} tier colors...")
    bg_plan = plan_tier_colors(
        source_image,
        full_lab,
        matte,
        foreground=False,
        max_colors=args.colors_per_tier,
        step_height_mm=args.layer_height,
        first_layer_height_mm=args.first_layer,
        max_layers_per_tier=args.max_layers_per_tier,
        td_scale=args.td_scale,
    )
    bg_schedule = bg_plan.schedule
    bg_states = list(bg_schedule.states)

    if args.tier_mode == "two":
        print("5. Selecting and optimizing foreground tier colors...")
        fg_plan = plan_tier_colors(
            source_image,
            full_lab,
            matte,
            foreground=True,
            max_colors=args.colors_per_tier,
            step_height_mm=args.layer_height,
            first_layer_height_mm=args.layer_height,
            initial_substrate_lab=bg_states[-1].simulated_lab,
            max_layers_per_tier=args.max_layers_per_tier,
            td_scale=args.td_scale,
            preferred_filaments=bg_plan.palette.filaments,
        )
        fg_schedule = fg_plan.schedule
        fg_states = list(fg_schedule.states)
        _print_tier_schedule("Background", bg_schedule, args.max_layers_per_tier)
        _print_tier_schedule("Foreground", fg_schedule, args.max_layers_per_tier)
        print("6. Generating discrete two-tier L* heightmap with pedestal support...")
        depth_mapper = TwoTierDepthMapper(
            bg_states=bg_states,
            fg_states=fg_states,
            step_height_mm=args.layer_height,
            first_layer_height_mm=args.first_layer,
        )
        height_result = depth_mapper.generate_heightmap(full_lab, full_lab, matte)
    else:
        _print_tier_schedule("Single", bg_schedule, args.max_layers_per_tier)
        print("5. Generating single-tier L* heightmap...")
        single_mapper = SingleTierDepthMapper(
            states=bg_states,
            step_height_mm=args.layer_height,
            first_layer_height_mm=args.first_layer,
        )
        height_result = single_mapper.generate_heightmap(full_lab)
    print(
        f"   Max height: {height_result.max_height_mm:.2f}mm ({height_result.total_layers} layers)"
    )
    _print_height_usage(height_result.z_grid)

    print("7. Building watertight manifold triangle mesh...")
    dimensions = PhysicalDimensions(width_mm=width_mm, height_mm=height_mm, base_floor_z_mm=0.0)
    mesh_builder = WatertightMeshBuilder(dimensions)
    mesh = mesh_builder.build_mesh(
        height_result.z_grid,
        progress=lambda message: print(f"   {message}", flush=True),
    )
    print(f"   Vertices: {mesh.vertex_count:,} | Triangles: {mesh.face_count:,}")

    print(f"8. Packaging Bambu Studio / Orca Slicer 3MF into '{args.output}'...")
    export_bambu_project(
        mesh,
        height_result.swap_schedule,
        args.output,
        swap_mode=args.swap_mode,
        step_height_mm=args.layer_height,
        first_layer_height_mm=args.first_layer,
    )

    print("\nStratachrome export complete!")
    if args.swap_mode == "manual":
        print("Embedded layer pause schedule (Manual Mode):")
        for swap in height_result.swap_schedule:
            print(
                f"  - Layer {swap.global_layer_idx:2d} @ {swap.z_height_mm:5.2f}mm -> {swap.filament_name} ({swap.tier_name})"
            )
    else:
        print("Configured for AMS multi-material execution (no manual pauses required).")

    return 0


if __name__ == "__main__":
    sys.exit(main())
