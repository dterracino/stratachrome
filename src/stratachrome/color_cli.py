"""Compare an image's extracted palette with lightness-spaced colors."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence

from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError

from stratachrome.cli_defaults import DEFAULT_COLORS_PER_TIER
from stratachrome.cli_paths import resolve_output_file
from stratachrome.color_algorithms import get_color_algorithm
from stratachrome.color_spacing import PaletteColorRemap
from stratachrome.palette_recolor import recolor_image_to_palette

_DEFAULT_PALETTE_COLORS = DEFAULT_COLORS_PER_TIER * 2
_MARGIN = 24
_SWATCH_WIDTH = 144
_SWATCH_HEIGHT = 82
_SECTION_GAP = 24
_TITLE_HEIGHT = 28
_PROGRESS_STAGE_COUNT = 9


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare an image palette with evenly spaced L* colors."
    )
    parser.add_argument("input", type=Path, help="Path to the input image file.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output PNG file or directory (default: output/<image>_colors.png).",
    )
    parser.add_argument(
        "--colors",
        type=int,
        choices=range(2, 33),
        default=_DEFAULT_PALETTE_COLORS,
        metavar="2-32",
        help=f"Number of image colors to extract (default: {_DEFAULT_PALETTE_COLORS}).",
    )
    args = parser.parse_args()
    args.output = resolve_output_file(args.input, args.output, "colors", ".png")
    return args


def _text_color(rgb: tuple[int, int, int]) -> tuple[int, int, int]:
    luminance = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
    return (0, 0, 0) if luminance >= 145.0 else (255, 255, 255)


def _draw_section(
    draw: ImageDraw.ImageDraw,
    remapped: Sequence[PaletteColorRemap],
    *,
    top: int,
    adjusted: bool,
    title: str,
) -> None:
    font = ImageFont.load_default()
    draw.text((_MARGIN, top), title, fill=(25, 25, 25), font=font)
    swatch_top = top + _TITLE_HEIGHT
    for index, color in enumerate(remapped):
        rgb = color.adjusted_rgb if adjusted else color.original_rgb
        lch = color.adjusted_lch if adjusted else color.original_lch
        left = _MARGIN + index * _SWATCH_WIDTH
        right = left + _SWATCH_WIDTH
        draw.rectangle((left, swatch_top, right, swatch_top + _SWATCH_HEIGHT), fill=rgb)
        draw.text(
            (left + 8, swatch_top + 8),
            f"#{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}",
            fill=_text_color(rgb),
            font=font,
        )
        draw.text(
            (left + 8, swatch_top + 30),
            f"L {lch[0]:.1f}  C {lch[1]:.1f}\nh {lch[2]:.1f}",
            fill=_text_color(rgb),
            font=font,
        )


def _render_comparison(
    remapped: Sequence[PaletteColorRemap],
    kmeans_remapped: Sequence[PaletteColorRemap],
) -> Image.Image:
    section_height = _TITLE_HEIGHT + _SWATCH_HEIGHT
    palette_size = max(len(remapped), len(kmeans_remapped))
    width = max(640, 2 * _MARGIN + palette_size * _SWATCH_WIDTH)
    height = 2 * _MARGIN + 3 * section_height + 2 * _SECTION_GAP
    comparison = Image.new("RGB", (width, height), (242, 242, 242))
    draw = ImageDraw.Draw(comparison)
    _draw_section(
        draw,
        remapped,
        top=_MARGIN,
        adjusted=False,
        title="Original colors: 64px median cut",
    )
    _draw_section(
        draw,
        remapped,
        top=_MARGIN + section_height + _SECTION_GAP,
        adjusted=True,
        title="Remapped colors: 64px median cut",
    )
    _draw_section(
        draw,
        kmeans_remapped,
        top=_MARGIN + 2 * (section_height + _SECTION_GAP),
        adjusted=True,
        title="Remapped colors: standardized RGB K-means",
    )
    return comparison


def _load_image(path: Path) -> Image.Image:
    try:
        with Image.open(path) as source:
            return source.convert("RGB")
    except (OSError, UnidentifiedImageError) as error:
        raise ValueError(f"Could not load image {path}: {error}") from error


def _recolor_output_path(comparison_path: Path, algorithm: str) -> Path:
    return comparison_path.with_name(f"{comparison_path.stem}_{algorithm}.png")


def _report_progress(stage: int, message: str) -> None:
    print(f"[{stage}/{_PROGRESS_STAGE_COUNT}] {message}", flush=True)


def main() -> int:
    args = _parse_arguments()
    if not args.input.is_file():
        sys.stderr.write(f"Input image not found: {args.input}\n")
        return 1

    try:
        _report_progress(1, f"Loading image: {args.input}")
        image = _load_image(args.input)

        _report_progress(
            2,
            f"Generating and redistributing {args.colors} 64px median-cut colors...",
        )
        median_algorithm = get_color_algorithm("median-cut")
        median_palette = median_algorithm.generate_palette(
            image,
            args.colors,
        )
        _report_progress(3, "The 64px palette is ready.")
        remapped = median_palette.colors

        _report_progress(
            4,
            f"Generating and redistributing {args.colors} full-resolution K-means++ colors...",
        )
        kmeans_algorithm = get_color_algorithm("kmeans")
        kmeans_palette = kmeans_algorithm.generate_palette(
            image,
            args.colors,
        )
        _report_progress(5, "The K-means palette is ready.")
        kmeans_remapped = kmeans_palette.colors

        _report_progress(6, "Rendering the palette comparison...")
        comparison = _render_comparison(remapped, kmeans_remapped)

        _report_progress(7, "Recoloring the image with the 64px palette...")
        median_cut_image = recolor_image_to_palette(
            image,
            tuple(color.adjusted_rgb for color in remapped),
        )

        _report_progress(8, "Recoloring the image with the K-means palette...")
        kmeans_image = recolor_image_to_palette(
            image,
            tuple(color.adjusted_rgb for color in kmeans_remapped),
        )
        median_cut_path = _recolor_output_path(args.output, "64px")
        kmeans_path = _recolor_output_path(args.output, "kmeans")

        _report_progress(9, f"Saving three PNG files to {args.output.parent}...")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        comparison.save(args.output, format="PNG")
        median_cut_image.save(median_cut_path, format="PNG")
        kmeans_image.save(kmeans_path, format="PNG")
    except ValueError as error:
        sys.stderr.write(f"Color analysis failed: {error}\n")
        return 1

    print(f"Saved color comparison: {args.output}")
    print(f"Saved 64px palette recoloring: {median_cut_path}")
    print(f"Saved K-means palette recoloring: {kmeans_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
