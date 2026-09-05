"""
color_engine.py
---------------
Integrates color_tools.conversions to handle exact CIELCh / CIELAB transforms,
extract perceptual lightness channels, and automatically derive filament
L* values directly from hex codes or RGB tuples.
"""

from __future__ import annotations

from typing import Iterable, Sequence
import numpy as np
from PIL import Image

from color_tools.conversions import hex_to_rgb, rgb_to_lch

from stratachrome.optical_model import Filament, FilamentLayerAssignment


def _resolve_l_from_hex(hex_str: str) -> float:
    """Derives CIELCh L* directly from a hex color string using color_tools."""
    clean_hex = hex_str if hex_str.startswith("#") else f"#{hex_str}"
    rgb_tuple = hex_to_rgb(clean_hex)
    if rgb_tuple is None:
        raise ValueError(f"Invalid hex string could not be parsed by color_tools: '{hex_str}'")
    lch_tuple = rgb_to_lch(rgb_tuple)
    # rgb_to_lch returns (L, C, h) where L is perceptual lightness [0.0, 100.0]
    return float(lch_tuple[0])


def _resolve_l_from_rgb(rgb: tuple[int, int, int]) -> float:
    """Derives CIELCh L* directly from an RGB tuple using color_tools."""
    lch_tuple = rgb_to_lch(rgb)
    return float(lch_tuple[0])


def create_filament(name: str, hex_color: str, td: float) -> Filament:
    """Instantiates a Filament by calculating intrinsic L* via color_tools.

    Args:
        name: Human-readable filament label.
        hex_color: Hex color string (e.g., '#1F1F1F' or '1F1F1F').
        td: Physical transmission distance in millimeters.

    Returns:
        Calibrated Filament instance with intrinsic L* computed.
    """
    clean_hex = hex_color if hex_color.startswith("#") else f"#{hex_color}"
    l_val = _resolve_l_from_hex(clean_hex)
    clamped_l = max(0.0, min(100.0, l_val))
    return Filament(
        name=name,
        hex_color=clean_hex,
        l_value=round(clamped_l, 2),
        td=td,
    )


def extract_perceptual_lightness(image: Image.Image | np.ndarray) -> np.ndarray:
    """Extracts a 2D matrix of perceptual L* values [0.0, 100.0] from an RGB image.

    Applies color_tools.conversions.rgb_to_lch via a lookup table of unique RGB
    triplets to ensure instant processing for 1000x1000 rasters.
    """
    if isinstance(image, Image.Image):
        rgb_arr = np.array(image.convert("RGB"), dtype=np.uint8)
    else:
        rgb_arr = image.astype(np.uint8)

    rows, cols, _ = rgb_arr.shape

    # Find unique RGB colors in the image to minimize redundant conversion calls
    flat_rgb = rgb_arr.reshape(-1, 3)
    unique_rgb, inverse_indices = np.unique(flat_rgb, axis=0, return_inverse=True)

    # Compute exact color_tools rgb_to_lch for each unique RGB
    unique_l = np.empty(len(unique_rgb), dtype=np.float32)
    for idx, (r, g, b) in enumerate(unique_rgb):
        lch = rgb_to_lch((int(r), int(g), int(b)))
        unique_l[idx] = float(lch[0])

    # Reconstruct full 2D L* matrix from unique color indices
    l_grid = unique_l[inverse_indices].reshape(rows, cols)
    return np.clip(l_grid, 0.0, 100.0).astype(np.float32)


def auto_sort_filaments_by_lightness(filaments: Iterable[Filament]) -> list[Filament]:
    """Sorts filaments monotonically by ascending L* to ensure correct opacity stacking."""
    return sorted(filaments, key=lambda f: f.l_value)


def build_layer_schedule(
    ordered_filaments: Sequence[Filament],
    layer_steps: Sequence[int],
) -> list[FilamentLayerAssignment]:
    """Pairs sorted filaments with assigned slicer layer heights.

    Args:
        ordered_filaments: Filaments to sort and assign.
        layer_steps: Target start layer index for each filament (first must be 0).

    Returns:
        List of validated FilamentLayerAssignment objects.
    """
    if len(ordered_filaments) != len(layer_steps):
        raise ValueError(
            f"Count mismatch: {len(ordered_filaments)} filaments vs {len(layer_steps)} layer steps"
        )
    if layer_steps[0] != 0:
        raise ValueError("The first filament in any tier must be assigned to start at layer 0.")

    sorted_fils = auto_sort_filaments_by_lightness(ordered_filaments)

    return [
        FilamentLayerAssignment(filament=fil, start_layer=step)
        for fil, step in zip(sorted_fils, layer_steps)
    ]