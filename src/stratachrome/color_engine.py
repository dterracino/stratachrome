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
from color_tools.image import DominantColor, dominant_colors

from stratachrome.optical_model import OptimizedTierSchedule, optimize_tier_schedule


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


def select_tier_palette(
    image: Image.Image,
    matte: np.ndarray,
    *,
    foreground: bool,
    color_count: int = 4,
    collection: Sequence[FilamentRecord] | None = None,
) -> TierPalette:
    """Find dominant tier colors and match them to unique real filaments."""
    if not 1 <= color_count <= 4:
        raise ValueError("color_count must be between 1 and 4.")

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

    tier_image = build_tier_image(image, matte, foreground=foreground)
    targets = dominant_colors(tier_image, count=color_count)
    palette = FilamentPalette(list(available))
    used_ids: set[str] = set()
    used_rgb: set[tuple[int, int, int]] = set()
    matches: list[FilamentMatch] = []

    for target in targets:
        candidates = palette.nearest_filaments(
            target.rgb,
            metric="de2000",
            count=min(50, len(available)),
            owned=False,
        )
        selected = next(
            (
                (filament, distance)
                for filament, distance in candidates
                if filament.id not in used_ids and filament.rgb not in used_rgb
            ),
            None,
        )
        if selected is None:
            continue
        filament, distance = selected
        used_ids.add(filament.id)
        used_rgb.add(filament.rgb)
        matches.append(FilamentMatch(target, filament, float(distance)))

    if not matches:
        raise ValueError("No dominant tier colors could be matched to a filament.")

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
    color_penalty: float = 0.50,
    max_layers_per_filament: int = 120,
    collection: Sequence[FilamentRecord] | None = None,
) -> TierColorPlan:
    """Choose the requested dominant palette and optimize every color's layer count."""
    candidate_palette = select_tier_palette(
        image,
        matte,
        foreground=foreground,
        color_count=max_colors,
        collection=collection,
    )
    targets = sample_tier_lab(
        lab_image,
        matte,
        foreground=foreground,
    )
    schedule = optimize_tier_schedule(
        candidate_palette.filaments,
        targets,
        step_height_mm=step_height_mm,
        first_layer_height_mm=first_layer_height_mm,
        initial_substrate_lab=initial_substrate_lab,
        layer_penalty=layer_penalty,
        max_layers_per_filament=max_layers_per_filament,
    )
    score = (
        schedule.mean_delta_e
        + layer_penalty * len(schedule.states)
        + color_penalty * (len(candidate_palette.filaments) - 1)
    )
    return TierColorPlan(candidate_palette, schedule, round(score, 4))
