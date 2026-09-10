"""Alternate dominant-color extraction algorithms."""

from __future__ import annotations

import numpy as np
from PIL import Image
from scipy.cluster.vq import kmeans2

from stratachrome.color_algorithms import ExtractedPaletteColor

_CONSTANT_CHANNEL_EPSILON = 1e-12
_KMEANS_ITERATIONS = 30
_KMEANS_RANDOM_SEED = 42


KMeansDominantColor = ExtractedPaletteColor


def extract_kmeans_dominant_colors(
    image: Image.Image,
    color_count: int,
    *,
    matte: np.ndarray | None = None,
    foreground: bool = False,
) -> tuple[KMeansDominantColor, ...]:
    """Cluster standardized full-resolution RGB pixels into dominant colors."""
    if not 1 <= color_count <= 32:
        raise ValueError("color_count must be between 1 and 32.")

    rgb_image = image.convert("RGB")
    rgb_array = np.asarray(rgb_image, dtype=np.float64)
    if matte is None:
        pixels = rgb_array.reshape(-1, 3)
    else:
        if matte.shape != (rgb_image.height, rgb_image.width):
            raise ValueError("Image and matte dimensions must match for palette analysis.")
        tier_mask = matte >= 0.5 if foreground else matte < 0.5
        pixels = rgb_array[tier_mask]
    if len(pixels) == 0:
        raise ValueError("Cannot extract colors from an empty image.")

    channel_std = np.std(pixels, axis=0)
    scaling_std = np.where(
        channel_std > _CONSTANT_CHANNEL_EPSILON,
        channel_std,
        1.0,
    )
    standardized = pixels / scaling_std
    distinct_pixels = np.unique(standardized, axis=0)
    cluster_count = min(color_count, len(distinct_pixels))

    centers, labels = kmeans2(
        standardized,
        cluster_count,
        iter=_KMEANS_ITERATIONS,
        minit="++",
        missing="raise",
        rng=np.random.default_rng(_KMEANS_RANDOM_SEED),
    )
    populations = np.bincount(labels, minlength=len(centers))
    ordered_indices = np.argsort(-populations, kind="stable")

    dominant: list[KMeansDominantColor] = []
    seen_rgb: set[tuple[int, int, int]] = set()
    for center_index in ordered_indices:
        center_rgb = np.clip(
            np.rint(centers[center_index] * scaling_std),
            0,
            255,
        ).astype(np.uint8)
        rgb = (int(center_rgb[0]), int(center_rgb[1]), int(center_rgb[2]))
        if rgb in seen_rgb:
            continue
        seen_rgb.add(rgb)
        dominant.append(
            KMeansDominantColor(
                rgb=rgb,
                population=float(populations[center_index] / len(pixels)),
            )
        )
    return tuple(dominant)
