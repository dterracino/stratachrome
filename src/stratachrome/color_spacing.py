"""Perceptual lightness redistribution for image-derived palettes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from color_tools import (
    find_nearest_in_gamut,
    is_in_srgb_gamut,
    lab_to_lch,
    lch_to_lab,
    lch_to_rgb,
    rgb_to_lch,
)

RgbColor = tuple[int, int, int]
LchColor = tuple[float, float, float]


@dataclass(frozen=True)
class PaletteColorRemap:
    """One palette color before and after lightness redistribution."""

    original_rgb: RgbColor
    original_lch: LchColor
    adjusted_rgb: RgbColor
    adjusted_lch: LchColor

    @property
    def chroma_was_reduced(self) -> bool:
        """Return whether sRGB gamut fitting reduced this color's chroma."""
        return self.adjusted_lch[1] < self.original_lch[1] - 1e-9


def _validate_rgb(rgb: RgbColor) -> None:
    if len(rgb) != 3 or any(
        not isinstance(channel, int) or not 0 <= channel <= 255 for channel in rgb
    ):
        raise ValueError("Palette colors must be RGB integer triples from 0 through 255.")


def _fit_chroma_to_srgb(lch: LchColor) -> LchColor:
    """Hold L* and hue fixed while reducing chroma to the sRGB boundary."""
    candidate_lab = lch_to_lab(lch)
    if is_in_srgb_gamut(candidate_lab):
        return lch

    fitted_lab = find_nearest_in_gamut(candidate_lab)
    _, fitted_chroma, _ = lab_to_lch(fitted_lab)
    return (lch[0], min(lch[1], fitted_chroma), lch[2])


def redistribute_palette_lightness(
    colors: Sequence[RgbColor],
) -> tuple[PaletteColorRemap, ...]:
    """Evenly space palette L* values while preserving order, hue, and chroma."""
    if not colors:
        raise ValueError("At least one palette color is required.")

    originals: list[tuple[RgbColor, LchColor]] = []
    for rgb in colors:
        _validate_rgb(rgb)
        originals.append((rgb, rgb_to_lch(rgb)))
    originals.sort(key=lambda item: item[1][0])

    minimum_lightness = originals[0][1][0]
    maximum_lightness = originals[-1][1][0]
    interval = (
        0.0
        if len(originals) == 1
        else (maximum_lightness - minimum_lightness) / (len(originals) - 1)
    )

    remapped: list[PaletteColorRemap] = []
    for index, (original_rgb, original_lch) in enumerate(originals):
        target_lightness = minimum_lightness + index * interval
        candidate = (target_lightness, original_lch[1], original_lch[2])
        adjusted_lch = _fit_chroma_to_srgb(candidate)
        adjusted_rgb = lch_to_rgb(adjusted_lch)
        remapped.append(
            PaletteColorRemap(
                original_rgb=original_rgb,
                original_lch=original_lch,
                adjusted_rgb=adjusted_rgb,
                adjusted_lch=adjusted_lch,
            )
        )
    return tuple(remapped)
