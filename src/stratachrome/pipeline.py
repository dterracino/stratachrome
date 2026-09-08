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
    allocate_lookahead_tier_layers,
    extract_perceptual_lab,
    plan_geometry_first_tier_colors,
    plan_tier_colors,
)
from stratachrome.color_diagnostics import DELTA_E_HEATMAP_MAX, save_color_diagnostics
from stratachrome.depth_mapper import (
    GeometryFirstTwoTierDepthMapper,
    SingleTierDepthMapper,
    TwoTierDepthMapper,
    map_tier_lightness_to_layer_indices,
)
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
        "--mapping-mode",
        choices=("optical", "geometry-first", "lookahead"),
        default="optical",
        help=(
            "Mapping strategy: 'optical' lets simulated color choose relief height; "
            "'geometry-first' fixes equal-sized L* tiers; 'lookahead' divides a fixed "
            "total across tiers from optical demand (default: optical)."
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
            "Layers available to each tier; optical mode may use fewer, while "
            "geometry-first uses exactly this count (default: 120)."
        ),
    )
    parser.add_argument(
        "--total-layers",
        type=int,
        default=None,
        help="Fixed total layer count required by lookahead mapping mode.",
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
    parser.add_argument(
        "--color-diagnostics",
        action="store_true",
        help="Save predicted-color and fixed-scale Delta E heatmap PNGs beside the 3MF.",
    )
    args = parser.parse_args()
    if args.mapping_mode == "lookahead":
        minimum_layers = 1 if args.tier_mode == "single" else 2
        if args.total_layers is None:
            parser.error("--mapping-mode lookahead requires --total-layers.")
        if args.total_layers < minimum_layers:
            parser.error(
                f"--total-layers must be at least {minimum_layers} "
                f"for {args.tier_mode}-tier lookahead mode."
            )
    elif args.total_layers is not None:
        parser.error("--total-layers is only valid with --mapping-mode lookahead.")
    return args


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
    print(
        f"   Swap mode: {args.swap_mode.upper()} | Tier mode: {args.tier_mode.upper()} "
        f"| Mapping mode: {args.mapping_mode.upper()}"
    )
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

    background_layer_budget = args.max_layers_per_tier
    foreground_layer_budget = args.max_layers_per_tier
    if args.mapping_mode == "lookahead":
        if args.tier_mode == "single":
            background_layer_budget = args.total_layers
            print(f"   Fixed single-tier geometry: {background_layer_budget} total layers.")
        else:
            print("   Looking ahead with optical schedules to divide the total layer budget...")
            lookahead_bg_plan = plan_tier_colors(
                source_image,
                full_lab,
                matte,
                foreground=False,
                max_colors=args.colors_per_tier,
                step_height_mm=args.layer_height,
                first_layer_height_mm=args.first_layer,
                max_layers_per_tier=args.total_layers,
                td_scale=args.td_scale,
            )
            lookahead_bg_states = list(lookahead_bg_plan.schedule.states)
            lookahead_fg_plan = plan_tier_colors(
                source_image,
                full_lab,
                matte,
                foreground=True,
                max_colors=args.colors_per_tier,
                step_height_mm=args.layer_height,
                first_layer_height_mm=args.layer_height,
                initial_substrate_lab=lookahead_bg_states[-1].simulated_lab,
                max_layers_per_tier=args.total_layers,
                td_scale=args.td_scale,
                preferred_filaments=lookahead_bg_plan.palette.filaments,
            )
            background_demand = len(lookahead_bg_plan.schedule.states)
            foreground_demand = len(lookahead_fg_plan.schedule.states)
            background_layer_budget, foreground_layer_budget = allocate_lookahead_tier_layers(
                args.total_layers,
                background_demand,
                foreground_demand,
            )
            print(
                f"   Optical demand: background={background_demand}, "
                f"foreground={foreground_demand} layers."
            )
            print(
                f"   Fixed geometry split: background={background_layer_budget}, "
                f"foreground={foreground_layer_budget} "
                f"({args.total_layers} total layers)."
            )

    geometry_layer_indices: dict[str, np.ndarray] = {}
    if args.mapping_mode in {"geometry-first", "lookahead"}:
        print("   Fixing tier geometry from CIELAB L* before color allocation...")
        geometry_layer_indices["background"] = map_tier_lightness_to_layer_indices(
            full_lab,
            matte,
            foreground=False,
            layer_count=background_layer_budget,
        )
        if args.tier_mode == "two":
            geometry_layer_indices["foreground"] = map_tier_lightness_to_layer_indices(
                full_lab,
                matte,
                foreground=True,
                layer_count=foreground_layer_budget,
            )

    tier_label = "background" if args.tier_mode == "two" else "single"
    print(f"4. Selecting and optimizing {tier_label} tier colors...")
    if args.mapping_mode in {"geometry-first", "lookahead"}:
        bg_plan = plan_geometry_first_tier_colors(
            source_image,
            full_lab,
            matte,
            geometry_layer_indices["background"],
            foreground=False,
            total_layers=background_layer_budget,
            max_colors=args.colors_per_tier,
            step_height_mm=args.layer_height,
            first_layer_height_mm=args.first_layer,
            td_scale=args.td_scale,
        )
    else:
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
        if args.mapping_mode in {"geometry-first", "lookahead"}:
            fg_plan = plan_geometry_first_tier_colors(
                source_image,
                full_lab,
                matte,
                geometry_layer_indices["foreground"],
                foreground=True,
                total_layers=foreground_layer_budget,
                max_colors=args.colors_per_tier,
                step_height_mm=args.layer_height,
                first_layer_height_mm=args.layer_height,
                initial_substrate_lab=bg_states[-1].simulated_lab,
                td_scale=args.td_scale,
                preferred_filaments=bg_plan.palette.filaments,
            )
        else:
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
        _print_tier_schedule("Background", bg_schedule, background_layer_budget)
        _print_tier_schedule("Foreground", fg_schedule, foreground_layer_budget)
        print("6. Generating two-tier heightmap with pedestal support...")
        depth_mapper_class = (
            GeometryFirstTwoTierDepthMapper
            if args.mapping_mode in {"geometry-first", "lookahead"}
            else TwoTierDepthMapper
        )
        depth_mapper = depth_mapper_class(
            bg_states=bg_states,
            fg_states=fg_states,
            step_height_mm=args.layer_height,
            first_layer_height_mm=args.first_layer,
        )
        height_result = depth_mapper.generate_heightmap(full_lab, full_lab, matte)
        color_diagnostics = (
            depth_mapper.generate_color_diagnostics(full_lab, full_lab, matte)
            if args.color_diagnostics
            else None
        )
    else:
        _print_tier_schedule("Single", bg_schedule, background_layer_budget)
        print("5. Generating single-tier L* heightmap...")
        single_mapper = SingleTierDepthMapper(
            states=bg_states,
            step_height_mm=args.layer_height,
            first_layer_height_mm=args.first_layer,
        )
        height_result = single_mapper.generate_heightmap(full_lab)
        color_diagnostics = (
            single_mapper.generate_color_diagnostics(full_lab) if args.color_diagnostics else None
        )
    print(
        f"   Max height: {height_result.max_height_mm:.2f}mm ({height_result.total_layers} layers)"
    )
    _print_height_usage(height_result.z_grid)

    if color_diagnostics is not None:
        preview_path, heatmap_path = save_color_diagnostics(color_diagnostics, args.output)
        print(
            "   Color error: "
            f"mean={color_diagnostics.mean_delta_e:.2f}, "
            f"95th percentile={color_diagnostics.percentile_95_delta_e:.2f}, "
            f"max={color_diagnostics.max_delta_e:.2f}"
        )
        print(f"   Predicted-color preview: {preview_path}")
        print(f"   Delta E heatmap (saturates at {DELTA_E_HEATMAP_MAX:g}): " f"{heatmap_path}")

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
