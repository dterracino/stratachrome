"""Pluggable dominant-color extraction with shared LCh redistribution."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np
from PIL import Image
from color_tools import lch_to_lab
from color_tools.image import DominantColor

from stratachrome.color_spacing import PaletteColorRemap, redistribute_palette_lightness

RgbColor = tuple[int, int, int]

DEFAULT_COLOR_ALGORITHM = "median-cut"


@dataclass(frozen=True)
class ExtractedPaletteColor:
    """One dominant RGB color and its fraction of the selected pixels."""

    rgb: RgbColor
    population: float


@dataclass(frozen=True)
class GeneratedPalette:
    """Palette colors after the shared LCh redistribution policy."""

    algorithm: str
    colors: tuple[PaletteColorRemap, ...]
    populations: tuple[float, ...]

    def as_dominant_colors(self) -> tuple[DominantColor, ...]:
        """Adapt adjusted colors to the records consumed by filament matching."""
        targets: list[DominantColor] = []
        for color, population in zip(self.colors, self.populations):
            targets.append(
                DominantColor(
                    rgb=color.adjusted_rgb,
                    lab=lch_to_lab(color.adjusted_lch),
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
        return tuple(
            sorted(
                targets,
                key=lambda target: target.population,
                reverse=True,
            )
        )


class ColorPaletteAlgorithm(ABC):
    """Base pipeline for extraction followed by common LCh redistribution."""

    key: str
    display_name: str

    def generate_palette(
        self,
        image: Image.Image,
        color_count: int,
        *,
        matte: np.ndarray | None = None,
        foreground: bool = False,
    ) -> GeneratedPalette:
        """Extract dominant colors and apply the shared L* allocation."""
        extracted = self._extract_colors(
            image,
            color_count,
            matte=matte,
            foreground=foreground,
        )
        if not extracted:
            raise ValueError(f"{self.display_name} did not extract any colors.")

        remapped = redistribute_palette_lightness(tuple(color.rgb for color in extracted))
        population_by_rgb = {color.rgb: color.population for color in extracted}
        populations = tuple(population_by_rgb[color.original_rgb] for color in remapped)
        return GeneratedPalette(
            algorithm=self.key,
            colors=remapped,
            populations=populations,
        )

    @abstractmethod
    def _extract_colors(
        self,
        image: Image.Image,
        color_count: int,
        *,
        matte: np.ndarray | None,
        foreground: bool,
    ) -> tuple[ExtractedPaletteColor, ...]:
        """Return dominant RGB colors before shared lightness redistribution."""


class MedianCutColorAlgorithm(ColorPaletteAlgorithm):
    """Existing 64px nearest-neighbor and Pillow median-cut extraction."""

    key = "median-cut"
    display_name = "64px median cut"

    def _extract_colors(
        self,
        image: Image.Image,
        color_count: int,
        *,
        matte: np.ndarray | None,
        foreground: bool,
    ) -> tuple[ExtractedPaletteColor, ...]:
        from stratachrome.color_engine import _quantized_tier_colors

        selected_matte = (
            np.zeros((image.height, image.width), dtype=np.float32)
            if matte is None
            else matte
        )
        colors = _quantized_tier_colors(
            image,
            selected_matte,
            foreground=foreground if matte is not None else False,
            color_count=color_count,
        )
        return tuple(
            ExtractedPaletteColor(color.rgb, color.population) for color in colors
        )


class KMeansColorAlgorithm(ColorPaletteAlgorithm):
    """Full-resolution standardized RGB K-means++ extraction."""

    key = "kmeans"
    display_name = "standardized RGB K-means++"

    def _extract_colors(
        self,
        image: Image.Image,
        color_count: int,
        *,
        matte: np.ndarray | None,
        foreground: bool,
    ) -> tuple[ExtractedPaletteColor, ...]:
        from stratachrome.dominant_colors import extract_kmeans_dominant_colors

        return extract_kmeans_dominant_colors(
            image,
            color_count,
            matte=matte,
            foreground=foreground,
        )


_ALGORITHMS: Mapping[str, ColorPaletteAlgorithm] = MappingProxyType(
    {
        MedianCutColorAlgorithm.key: MedianCutColorAlgorithm(),
        KMeansColorAlgorithm.key: KMeansColorAlgorithm(),
    }
)
COLOR_ALGORITHM_NAMES = tuple(_ALGORITHMS)


def get_color_algorithm(name: str) -> ColorPaletteAlgorithm:
    """Return a registered color algorithm by CLI-safe name."""
    try:
        return _ALGORITHMS[name]
    except KeyError as error:
        choices = ", ".join(COLOR_ALGORITHM_NAMES)
        raise ValueError(f"Unknown color algorithm {name!r}; choose from {choices}.") from error
