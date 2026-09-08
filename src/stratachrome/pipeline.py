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

from stratachrome.cli_defaults import (
    DEFAULT_COLORS_PER_TIER,
    DEFAULT_FIRST_LAYER_HEIGHT_MM,
    DEFAULT_LAYER_HEIGHT_MM,
    DEFAULT_MAX_LAYERS_PER_TIER,
    DEFAULT_MAX_DIM,
    DEFAULT_SIZE_MM,
    DEFAULT_TOTAL_LAYERS,
)
from stratachrome.cli_paths import resolve_output_file
from stratachrome.color_engine import (
    FilamentMatch,
    TierPalette,
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
    parser.add_argument("input", type=Path, help="Input RGB image path.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help=("Output 3MF file or directory. Defaults to " "output/<input-stem>_stratachrome.3mf."),
    )
    parser.add_argument(
        "-s",
        "--size",
        type=float,
        default=DEFAULT_SIZE_MM,
        help=f"Target physical size in mm for the maximum image dimension (default: {DEFAULT_SIZE_MM}).",
    )
    parser.add_argument(
        "--max-dim",
        type=int,
        default=DEFAULT_MAX_DIM,
        help=f"Maximum raster dimension for mesh grid (default: {DEFAULT_MAX_DIM}).",
    )
    parser.add_argument(
        "--layer-height",
        type=float,
        default=DEFAULT_LAYER_HEIGHT_MM,
        help=f"Standard layer step height in mm (default: {DEFAULT_LAYER_HEIGHT_MM:.2f}).",
    )
    parser.add_argument(
        "--first-layer",
        type=float,
        default=DEFAULT_FIRST_LAYER_HEIGHT_MM,
        help=(
            "First layer bed-contact height in mm "
            f"(default: {DEFAULT_FIRST_LAYER_HEIGHT_MM:.2f})."
        ),
    )
    parser.add_argument(
        "--swap-mode",
        type=str,
        choices=["auto", "manual"],
        default="auto",
        help="Filament change mode: 'auto' for multi-material switching, 'manual' for single-extruder pause triggers (default: auto).",
    )
    parser.add_argument(
        "--tier-mode",
        choices=("single", "dual"),
        default="dual",
        help=(
            "Relief mode: 'single' analyzes the full image without segmentation; "
            "'dual' separates and stacks background/foreground tiers (default: dual)."
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
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="Torch compute device (default: auto).",
    )
    parser.add_argument(
        "--colors-per-tier",
        type=int,
        choices=tuple(range(2, 17)),
        default=DEFAULT_COLORS_PER_TIER,
        help=f"Maximum filament colors per tier, from 2 to 16 (default: {DEFAULT_COLORS_PER_TIER}).",
    )
    parser.add_argument(
        "--max-layers-per-tier",
        type=int,
        default=DEFAULT_MAX_LAYERS_PER_TIER,
        help=(
            "Layers available to each tier; optical mode may use fewer, while "
            f"geometry-first uses exactly this count (default: {DEFAULT_MAX_LAYERS_PER_TIER})."
        ),
    )
    parser.add_argument(
        "--total-layers",
        type=int,
        default=DEFAULT_TOTAL_LAYERS,
        help=(
            "Fixed total layer count used by lookahead mode and ignored by other "
            f"mapping modes (default: {DEFAULT_TOTAL_LAYERS})."
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
    parser.add_argument(
        "--color-diagnostics",
        action="store_true",
        help="Save predicted-color and fixed-scale Delta E heatmap PNGs beside the 3MF.",
    )
    args = parser.parse_args()
    args.output = resolve_output_file(args.input, args.output, "stratachrome", ".3mf")
    if args.mapping_mode == "lookahead":
        minimum_layers = 1 if args.tier_mode == "single" else 2
        if args.total_layers < minimum_layers:
            parser.error(
                f"--total-layers must be at least {minimum_layers} "
                f"for {args.tier_mode}-tier lookahead mode."
            )
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


def _warn_palette_lightness_collisions(
    tier_name: str,
    palette: TierPalette,
    lab_image: np.ndarray,
    matte: np.ndarray,
    *,
    foreground: bool,
    layer_count: int,
) -> None:
    """Warn when selected 64px target colors round to the same height layer."""
    tier_mask = matte >= 0.5 if foreground else matte < 0.5
    tier_lightness = np.asarray(lab_image[..., 0][tier_mask], dtype=np.float64)
    minimum = float(tier_lightness.min())
    maximum = float(tier_lightness.max())
    lightness_range = maximum - minimum

    matches_by_layer: dict[int, list[FilamentMatch]] = {}
    for match in palette.matches:
        if lightness_range <= 1e-8 or layer_count == 1:
            layer_index = 0
        else:
            normalized = np.clip((match.target.lab[0] - minimum) / lightness_range, 0.0, 1.0)
            layer_index = int(np.rint(normalized * (layer_count - 1)))
        matches_by_layer.setdefault(layer_index, []).append(match)

    for layer_index, matches in matches_by_layer.items():
        if len(matches) < 2:
            continue
        for first_index, first in enumerate(matches[:-1]):
            for second in matches[first_index + 1 :]:
                first_rgb = "#" + "".join(f"{value:02X}" for value in first.target.rgb)
                second_rgb = "#" + "".join(f"{value:02X}" for value in second.target.rgb)
                print(
                    f"WARNING: {tier_name} 64px colors {first_rgb} "
                    f"(L*={first.target.lab[0]:.2f}, matched to "
                    f"{_filament_display_name(first.filament)}) and {second_rgb} "
                    f"(L*={second.target.lab[0]:.2f}, matched to "
                    f"{_filament_display_name(second.filament)}) map to the same "
                    f"height layer {layer_index + 1}/{layer_count}.",
                    file=sys.stderr,
                )


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

    if args.tier_mode == "dual":
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
        if args.tier_mode == "dual":
            geometry_layer_indices["foreground"] = map_tier_lightness_to_layer_indices(
                full_lab,
                matte,
                foreground=True,
                layer_count=foreground_layer_budget,
            )

    tier_label = "background" if args.tier_mode == "dual" else "single"
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
    _warn_palette_lightness_collisions(
        tier_label.title(),
        bg_plan.palette,
        full_lab,
        matte,
        foreground=False,
        layer_count=len(bg_states),
    )

    if args.tier_mode == "dual":
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
        _warn_palette_lightness_collisions(
            "Foreground",
            fg_plan.palette,
            full_lab,
            matte,
            foreground=True,
            layer_count=len(fg_states),
        )
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
        (
            preview_path,
            heatmap_path,
            source_64px_path,
            preview_64px_path,
            source_histogram_path,
            preview_histogram_path,
        ) = save_color_diagnostics(color_diagnostics, args.output, source_image)
        print(
            "   Color error: "
            f"mean={color_diagnostics.mean_delta_e:.2f}, "
            f"95th percentile={color_diagnostics.percentile_95_delta_e:.2f}, "
            f"max={color_diagnostics.max_delta_e:.2f}"
        )
        print(f"   Predicted-color preview: {preview_path}")
        print(f"   Delta E heatmap (saturates at {DELTA_E_HEATMAP_MAX:g}): " f"{heatmap_path}")
        print(f"   64px source image: {source_64px_path}")
        print(f"   64px predicted-color image: {preview_64px_path}")
        print(f"   Source-color histogram: {source_histogram_path}")
        print(f"   Predicted-color histogram: {preview_histogram_path}")

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
        print("Configured for automatic multi-material execution (no manual pauses required).")

    return 0


if __name__ == "__main__":
    sys.exit(main())
