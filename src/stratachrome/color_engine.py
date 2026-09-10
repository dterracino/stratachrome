"""Tier color analysis and filament matching through color_tools."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from PIL import Image
from color_tools import (
    FilamentCollections,
    FilamentPalette,
    FilamentRecord,
    rgb_to_lab,
)
from color_tools.image import DominantColor

from stratachrome.color_algorithms import (
    DEFAULT_COLOR_ALGORITHM,
    ColorPaletteAlgorithm,
    get_color_algorithm,
)
from stratachrome.optical_model import (
    OptimizedTierSchedule,
    optimize_geometry_first_schedule,
    optimize_tier_schedule,
)

_PALETTE_ANALYSIS_MAX_DIMENSION = 64
_NEW_FILAMENT_REUSE_PENALTY = 0.75


@dataclass(frozen=True)
class FilamentMatch:
    """One dominant tier color and its selected printable filament."""

    target: DominantColor
    filament: FilamentRecord
    delta_e: float


@dataclass(frozen=True)
class TierPalette:
    """Perceptually selected, unique filament palette for one image tier."""

    matches: tuple[FilamentMatch, ...]

    @property
    def filaments(self) -> tuple[FilamentRecord, ...]:
        return tuple(match.filament for match in self.matches)


@dataclass(frozen=True)
class TierColorPlan:
    """Jointly selected palette and optimized physical layer schedule."""

    palette: TierPalette
    schedule: OptimizedTierSchedule
    selection_score: float


def allocate_lookahead_tier_layers(
    total_layers: int,
    background_demand: int,
    foreground_demand: int,
) -> tuple[int, int]:
    """Fit optical tier demands proportionally into one fixed layer total."""
    if total_layers < 2:
        raise ValueError("Two-tier lookahead mode requires at least two total layers.")
    if background_demand < 1 or foreground_demand < 1:
        raise ValueError("Tier layer demands must be positive.")

    demand_total = background_demand + foreground_demand
    background_layers = round(total_layers * background_demand / demand_total)
    background_layers = min(total_layers - 1, max(1, background_layers))
    return background_layers, total_layers - background_layers


def _quantized_tier_colors(
    image: Image.Image,
    matte: np.ndarray,
    *,
    foreground: bool,
    color_count: int,
) -> tuple[DominantColor, ...]:
    """Downsample one tier and quantize it to representative RGB colors."""
    if not 1 <= color_count <= 32:
        raise ValueError("color_count must be between 1 and 32.")
    rgb_image = image.convert("RGB")
    if matte.shape != (rgb_image.height, rgb_image.width):
        raise ValueError("Image and matte dimensions must match for palette analysis.")

    scale = _PALETTE_ANALYSIS_MAX_DIMENSION / max(rgb_image.size)
    small_size = (
        max(1, round(rgb_image.width * scale)),
        max(1, round(rgb_image.height * scale)),
    )
    small_rgb = rgb_image.resize(small_size, Image.Resampling.NEAREST)
    matte_image = Image.fromarray(
        np.rint(np.clip(matte, 0.0, 1.0) * 255.0).astype(np.uint8),
        mode="L",
    ).resize(small_size, Image.Resampling.NEAREST)
    small_array = np.asarray(small_rgb, dtype=np.uint8)
    small_matte = np.asarray(matte_image, dtype=np.uint8)
    tier_mask = small_matte >= 128 if foreground else small_matte < 128
    tier_pixels = small_array[tier_mask]
    if len(tier_pixels) == 0:
        tier_name = "foreground" if foreground else "background"
        raise ValueError(f"Downsampled image contains no {tier_name} pixels.")

    pixel_strip = Image.fromarray(tier_pixels.reshape(1, -1, 3), mode="RGB")
    quantized = pixel_strip.quantize(
        colors=color_count,
        method=Image.Quantize.MEDIANCUT,
        dither=Image.Dither.NONE,
    )
    palette_values = quantized.getpalette()
    color_counts = quantized.getcolors(maxcolors=color_count)
    if palette_values is None or color_counts is None:
        raise RuntimeError("Pillow did not return the quantized tier palette.")

    targets: list[DominantColor] = []
    sorted_counts = sorted(color_counts, key=lambda item: item[0], reverse=True)
    for population_count, palette_value in sorted_counts:
        if not isinstance(palette_value, int):
            raise RuntimeError("Pillow returned a non-indexed quantized palette.")
        palette_index = palette_value
        offset = palette_index * 3
        channels = palette_values[offset : offset + 3]
        if len(channels) != 3:
            raise RuntimeError("Pillow returned an incomplete quantized palette.")
        rgb = (int(channels[0]), int(channels[1]), int(channels[2]))
        population = population_count / len(tier_pixels)
        targets.append(
            DominantColor(
                rgb=rgb,
                lab=rgb_to_lab(rgb),
                population=population,
                dominance=population,
                global_salience=0.0,
                local_contrast=0.0,
                spatial_distribution=0.0,
                spatial_coherence=0.0,
                lightness_contrast=0.0,
                focal_importance=0.0,
            )
        )
    return tuple(targets)


def extract_image_palette(
    image: Image.Image,
    color_count: int,
) -> tuple[DominantColor, ...]:
    """Quantize an entire image using the same analysis path as tier planning."""
    matte = np.zeros((image.height, image.width), dtype=np.float32)
    return _quantized_tier_colors(
        image,
        matte,
        foreground=False,
        color_count=color_count,
    )


def extract_perceptual_lab(image: Image.Image | np.ndarray) -> np.ndarray:
    """Convert an RGB image to an H x W x 3 CIELAB array via color_tools."""
    if isinstance(image, Image.Image):
        rgb_array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    else:
        rgb_array = np.asarray(image, dtype=np.uint8)
        if rgb_array.ndim != 3 or rgb_array.shape[-1] != 3:
            raise ValueError("image array must have shape (height, width, 3).")

    height, width, _ = rgb_array.shape
    flat_rgb = rgb_array.reshape(-1, 3)
    unique_rgb, inverse_indices = np.unique(flat_rgb, axis=0, return_inverse=True)
    unique_lab = np.asarray(
        [rgb_to_lab((int(red), int(green), int(blue))) for red, green, blue in unique_rgb],
        dtype=np.float64,
    )
    return unique_lab[inverse_indices].reshape(height, width, 3)


def extract_perceptual_lightness(image: Image.Image | np.ndarray) -> np.ndarray:
    """Return the CIELAB L* channel for compatibility with existing callers."""
    return extract_perceptual_lab(image)[..., 0].astype(np.float32)


def build_tier_image(
    image: Image.Image,
    matte: np.ndarray,
    *,
    foreground: bool,
) -> Image.Image:
    """Create an RGBA image whose alpha selects exactly one segmentation tier."""
    rgb_image = image.convert("RGB")
    if matte.shape != (rgb_image.height, rgb_image.width):
        raise ValueError(
            f"Matte shape {matte.shape} does not match image shape "
            f"{(rgb_image.height, rgb_image.width)}."
        )
    tier_mask = matte >= 0.5 if foreground else matte < 0.5
    if not np.any(tier_mask):
        tier_name = "foreground" if foreground else "background"
        raise ValueError(f"Segmentation produced no {tier_name} pixels.")

    rgba = np.empty((rgb_image.height, rgb_image.width, 4), dtype=np.uint8)
    rgba[..., :3] = np.asarray(rgb_image, dtype=np.uint8)
    rgba[..., 3] = np.where(tier_mask, 255, 0).astype(np.uint8)
    return Image.fromarray(rgba, mode="RGBA")


def build_tier_palette_previews(
    image: Image.Image,
    matte: np.ndarray,
    *,
    foreground: bool,
    colors_per_tier: int,
) -> tuple[Image.Image, Image.Image]:
    """Build a 64px tier colormap and a full-resolution quantized tier image."""
    if not 1 <= colors_per_tier <= 16:
        raise ValueError("colors_per_tier must be between 1 and 16.")

    tier_image = build_tier_image(image, matte, foreground=foreground)
    scale = _PALETTE_ANALYSIS_MAX_DIMENSION / max(tier_image.size)
    small_size = (
        max(1, round(tier_image.width * scale)),
        max(1, round(tier_image.height * scale)),
    )
    colormap = tier_image.resize(small_size, Image.Resampling.NEAREST)
    colormap_array = np.asarray(colormap, dtype=np.uint8)
    tier_mask = colormap_array[..., 3] > 0
    tier_pixels = colormap_array[..., :3][tier_mask]

    pixel_strip = Image.fromarray(tier_pixels.reshape(1, -1, 3), mode="RGB")
    quantized_strip = pixel_strip.quantize(
        colors=colors_per_tier * 2,
        method=Image.Quantize.MEDIANCUT,
        dither=Image.Dither.NONE,
    ).convert("RGB")
    palette_colors = np.unique(
        np.asarray(quantized_strip, dtype=np.uint8).reshape(-1, 3),
        axis=0,
    ).astype(np.int32)
    tier_array = np.asarray(tier_image, dtype=np.uint8)
    full_tier_mask = tier_array[..., 3] > 0
    full_tier_pixels = tier_array[..., :3][full_tier_mask].astype(np.int32)
    best_distances = np.full(len(full_tier_pixels), np.iinfo(np.int32).max, dtype=np.int32)
    nearest_colors = np.empty_like(full_tier_pixels)
    for palette_color in palette_colors:
        distances = np.sum((full_tier_pixels - palette_color) ** 2, axis=1)
        closer = distances < best_distances
        best_distances[closer] = distances[closer]
        nearest_colors[closer] = palette_color

    quantized_array = tier_array.copy()
    quantized_array[..., :3][full_tier_mask] = nearest_colors.astype(np.uint8)
    quantized = Image.fromarray(quantized_array, mode="RGBA")
    return colormap, quantized


def select_tier_palette(
    image: Image.Image,
    matte: np.ndarray,
    *,
    foreground: bool,
    color_count: int = 4,
    collection: Sequence[FilamentRecord] | None = None,
    preferred_filaments: Sequence[FilamentRecord] = (),
    new_filament_penalty: float = _NEW_FILAMENT_REUSE_PENALTY,
    color_algorithm: ColorPaletteAlgorithm | None = None,
) -> TierPalette:
    """Generate adjusted tier colors and match them to printable filaments."""
    if not 1 <= color_count <= 16:
        raise ValueError("color_count must be between 1 and 16.")
    if new_filament_penalty < 0.0:
        raise ValueError("new_filament_penalty cannot be negative.")
    available = tuple(
        collection if collection is not None else FilamentCollections.BAMBU_PLA_BASICMATTE
    )
    available = tuple(
        filament
        for filament in available
        if filament.td_value is not None and filament.td_value > 0.0
    )
    if not available:
        raise ValueError("The filament search collection has no records with TD values.")

    algorithm = color_algorithm or get_color_algorithm(DEFAULT_COLOR_ALGORITHM)
    generated_palette = algorithm.generate_palette(
        image,
        color_count * 2,
        matte=matte,
        foreground=foreground,
    )
    targets = generated_palette.as_dominant_colors()
    palette = FilamentPalette(list(available))
    available_ids = {filament.id for filament in available}
    preferred_ids = {
        filament.id for filament in preferred_filaments if filament.id in available_ids
    }
    used_ids: set[str] = set()
    used_rgb: set[tuple[int, int, int]] = set()
    matches: list[FilamentMatch] = []
    for target in targets:
        candidates = palette.nearest_filaments(
            target.rgb,
            metric="de2000",
            count=len(available),
            owned=False,
        )
        filament, distance = min(
            candidates,
            key=lambda candidate: (
                candidate[1]
                + (
                    0.0
                    if not preferred_ids or candidate[0].id in preferred_ids
                    else new_filament_penalty
                ),
                candidate[1],
            ),
        )
        if filament.id in used_ids or filament.rgb in used_rgb:
            continue
        used_ids.add(filament.id)
        used_rgb.add(filament.rgb)
        matches.append(FilamentMatch(target, filament, float(distance)))
        if len(matches) == color_count:
            break

    if not matches:
        raise ValueError("No quantized tier colors could be matched to a filament.")
    matches.sort(key=lambda match: match.filament.lab[0])
    return TierPalette(tuple(matches))


def sample_tier_lab(
    lab_image: np.ndarray,
    matte: np.ndarray,
    *,
    foreground: bool,
    max_samples: int = 4096,
) -> np.ndarray:
    """Return a deterministic, bounded Lab sample from one segmentation tier."""
    if lab_image.ndim != 3 or lab_image.shape[-1] != 3:
        raise ValueError("lab_image must have shape (height, width, 3).")
    if lab_image.shape[:2] != matte.shape:
        raise ValueError("lab_image and matte dimensions must match.")
    if max_samples < 1:
        raise ValueError("max_samples must be positive.")

    tier_mask = matte >= 0.5 if foreground else matte < 0.5
    samples = np.asarray(lab_image[tier_mask], dtype=np.float64)
    if len(samples) == 0:
        tier_name = "foreground" if foreground else "background"
        raise ValueError(f"Segmentation produced no {tier_name} samples.")
    if len(samples) <= max_samples:
        return samples

    indices = np.linspace(0, len(samples) - 1, num=max_samples, dtype=np.int64)
    return samples[indices]


def sample_tier_lab_with_layers(
    lab_image: np.ndarray,
    layer_indices: np.ndarray,
    matte: np.ndarray,
    *,
    foreground: bool,
    max_samples: int = 4096,
) -> tuple[np.ndarray, np.ndarray]:
    """Return aligned target Lab colors and geometry-selected layer indices."""
    if lab_image.ndim != 3 or lab_image.shape[-1] != 3:
        raise ValueError("lab_image must have shape (height, width, 3).")
    if lab_image.shape[:2] != matte.shape or layer_indices.shape != matte.shape:
        raise ValueError("lab_image, layer_indices, and matte dimensions must match.")
    if max_samples < 1:
        raise ValueError("max_samples must be positive.")

    tier_mask = matte >= 0.5 if foreground else matte < 0.5
    samples = np.asarray(lab_image[tier_mask], dtype=np.float64)
    sampled_layers = np.asarray(layer_indices[tier_mask], dtype=np.int32)
    if len(samples) == 0:
        tier_name = "foreground" if foreground else "background"
        raise ValueError(f"Segmentation produced no {tier_name} samples.")
    if len(samples) <= max_samples:
        return samples, sampled_layers

    indices = np.linspace(0, len(samples) - 1, num=max_samples, dtype=np.int64)
    return samples[indices], sampled_layers[indices]


def plan_tier_colors(
    image: Image.Image,
    lab_image: np.ndarray,
    matte: np.ndarray,
    *,
    foreground: bool,
    max_colors: int = 4,
    step_height_mm: float = 0.10,
    first_layer_height_mm: float = 0.20,
    initial_substrate_lab: tuple[float, float, float] | None = None,
    layer_penalty: float = 0.25,
    max_layers_per_tier: int = 120,
    td_scale: float = 1.0,
    collection: Sequence[FilamentRecord] | None = None,
    preferred_filaments: Sequence[FilamentRecord] = (),
    new_filament_penalty: float = _NEW_FILAMENT_REUSE_PENALTY,
    color_algorithm: ColorPaletteAlgorithm | None = None,
) -> TierColorPlan:
    """Quantize a tier, match its filaments, and optimize physical thickness."""
    targets = sample_tier_lab(
        lab_image,
        matte,
        foreground=foreground,
    )
    candidate_palette = select_tier_palette(
        image,
        matte,
        foreground=foreground,
        color_count=max_colors,
        collection=collection,
        preferred_filaments=preferred_filaments,
        new_filament_penalty=new_filament_penalty,
        color_algorithm=color_algorithm,
    )

    matches = list(candidate_palette.matches)
    while matches:
        working_palette = TierPalette(
            tuple(sorted(matches, key=lambda match: match.filament.lab[0]))
        )
        try:
            schedule = optimize_tier_schedule(
                working_palette.filaments,
                targets,
                step_height_mm=step_height_mm,
                first_layer_height_mm=first_layer_height_mm,
                initial_substrate_lab=initial_substrate_lab,
                target_by_filament={
                    match.filament.id: match.target.lab for match in working_palette.matches
                },
                layer_penalty=layer_penalty,
                max_layers_per_tier=max_layers_per_tier,
                td_scale=td_scale,
            )
        except ValueError as error:
            if "cannot satisfy the TD minimum" not in str(error):
                raise
            removable = [
                match for match in matches if match.filament.color.strip().casefold() != "black"
            ]
            if not removable:
                raise
            matches.remove(min(removable, key=lambda match: match.target.population))
            continue
        return TierColorPlan(working_palette, schedule, schedule.mean_delta_e)
    raise ValueError("No optically valid quantized tier palette could be constructed.")


def plan_geometry_first_tier_colors(
    image: Image.Image,
    lab_image: np.ndarray,
    matte: np.ndarray,
    geometry_layer_indices: np.ndarray,
    *,
    foreground: bool,
    total_layers: int,
    max_colors: int = 4,
    step_height_mm: float = 0.10,
    first_layer_height_mm: float = 0.20,
    initial_substrate_lab: tuple[float, float, float] | None = None,
    td_scale: float = 1.0,
    collection: Sequence[FilamentRecord] | None = None,
    preferred_filaments: Sequence[FilamentRecord] = (),
    new_filament_penalty: float = _NEW_FILAMENT_REUSE_PENALTY,
    color_algorithm: ColorPaletteAlgorithm | None = None,
) -> TierColorPlan:
    """Fit movable filament boundaries to a fixed geometry layer grid."""
    targets, target_layers = sample_tier_lab_with_layers(
        lab_image,
        geometry_layer_indices,
        matte,
        foreground=foreground,
    )
    candidate_palette = select_tier_palette(
        image,
        matte,
        foreground=foreground,
        color_count=max_colors,
        collection=collection,
        preferred_filaments=preferred_filaments,
        new_filament_penalty=new_filament_penalty,
        color_algorithm=color_algorithm,
    )

    matches = list(candidate_palette.matches)
    while matches:
        working_palette = TierPalette(
            tuple(sorted(matches, key=lambda match: match.filament.lab[0]))
        )
        try:
            schedule = optimize_geometry_first_schedule(
                working_palette.filaments,
                targets,
                target_layers,
                total_layers=total_layers,
                step_height_mm=step_height_mm,
                first_layer_height_mm=first_layer_height_mm,
                initial_substrate_lab=initial_substrate_lab,
                td_scale=td_scale,
            )
        except ValueError as error:
            if "cannot satisfy the TD minimum" not in str(error):
                raise
            removable = [
                match for match in matches if match.filament.color.strip().casefold() != "black"
            ]
            if not removable:
                raise
            matches.remove(min(removable, key=lambda match: match.target.population))
            continue
        return TierColorPlan(working_palette, schedule, schedule.mean_delta_e)
    raise ValueError("No optically valid quantized tier palette could be constructed.")
