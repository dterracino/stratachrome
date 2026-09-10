"""Map full-resolution images onto small perceptual color palettes."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from PIL import Image
from color_tools import delta_e_2000_array, rgb_to_lab

RgbColor = tuple[int, int, int]

_DISTANCE_CHUNK_SIZE = 16_384


def recolor_image_to_palette(
    image: Image.Image,
    palette: Sequence[RgbColor],
) -> Image.Image:
    """Replace each source pixel with its nearest palette color by CIEDE2000."""
    if not palette:
        raise ValueError("At least one palette color is required for recoloring.")

    source = np.asarray(image.convert("RGB"), dtype=np.uint8)
    height, width, _ = source.shape
    unique_rgb, inverse_indices = np.unique(
        source.reshape(-1, 3),
        axis=0,
        return_inverse=True,
    )
    unique_lab = np.asarray(
        [
            rgb_to_lab((int(red), int(green), int(blue)))
            for red, green, blue in unique_rgb
        ],
        dtype=np.float64,
    )
    palette_rgb = np.asarray(palette, dtype=np.uint8)
    if palette_rgb.ndim != 2 or palette_rgb.shape[1] != 3:
        raise ValueError("Palette colors must be RGB triples.")
    palette_lab = np.asarray(
        [rgb_to_lab((int(red), int(green), int(blue))) for red, green, blue in palette_rgb],
        dtype=np.float64,
    )

    nearest_indices = np.empty(len(unique_rgb), dtype=np.intp)
    for start in range(0, len(unique_rgb), _DISTANCE_CHUNK_SIZE):
        stop = min(start + _DISTANCE_CHUNK_SIZE, len(unique_rgb))
        distances = delta_e_2000_array(
            unique_lab[start:stop, None, :],
            palette_lab[None, :, :],
        )
        nearest_indices[start:stop] = np.argmin(distances, axis=1)

    recolored = palette_rgb[nearest_indices][inverse_indices].reshape(height, width, 3)
    return Image.fromarray(recolored, mode="RGB")
