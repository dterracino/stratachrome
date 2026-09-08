"""Render RGB channel histograms for color diagnostic images."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

_HISTOGRAM_WIDTH = 768
_HISTOGRAM_HEIGHT = 432
_MARGIN_LEFT = 56
_MARGIN_RIGHT = 24
_MARGIN_TOP = 24
_MARGIN_BOTTOM = 48
_CHART_GAP = 16
_CHANNEL_COLORS = np.asarray(((210, 45, 55), (30, 155, 80), (35, 90, 210)), dtype=np.uint8)


def rgb_histogram(image: Image.Image | np.ndarray) -> np.ndarray:
    """Count RGB channel values, excluding fully transparent pixels."""
    pixels = np.asarray(image, dtype=np.uint8)
    if pixels.ndim != 3 or pixels.shape[2] not in {3, 4}:
        raise ValueError("image must have RGB or RGBA shape (height, width, channels).")

    included = pixels[..., 3] > 0 if pixels.shape[2] == 4 else np.ones(pixels.shape[:2], dtype=bool)
    rgb_pixels = pixels[..., :3][included]
    if len(rgb_pixels) == 0:
        raise ValueError("image contains no visible pixels.")

    return np.asarray(
        [np.bincount(rgb_pixels[:, channel], minlength=256) for channel in range(3)],
        dtype=np.int64,
    )


def render_rgb_histogram(image: Image.Image | np.ndarray) -> np.ndarray:
    """Render separate red, green, and blue channel distributions as continuous curves."""
    histogram = rgb_histogram(image).astype(np.float64)
    plot_width = _HISTOGRAM_WIDTH - _MARGIN_LEFT - _MARGIN_RIGHT
    chart_height = (_HISTOGRAM_HEIGHT - _MARGIN_TOP - _MARGIN_BOTTOM - 2 * _CHART_GAP) // 3

    canvas = np.full((_HISTOGRAM_HEIGHT, _HISTOGRAM_WIDTH, 3), 248, dtype=np.uint8)
    for channel, color in enumerate(_CHANNEL_COLORS):
        chart_top = _MARGIN_TOP + channel * (chart_height + _CHART_GAP)
        x_axis = chart_top + chart_height
        canvas[chart_top : x_axis + 1, _MARGIN_LEFT] = 55
        canvas[x_axis, _MARGIN_LEFT : _HISTOGRAM_WIDTH - _MARGIN_RIGHT] = 55

        for fraction in (0.5, 1.0):
            row = x_axis - round(chart_height * fraction)
            canvas[row, _MARGIN_LEFT + 1 : _HISTOGRAM_WIDTH - _MARGIN_RIGHT] = 220

        peak = float(histogram[channel].max())
        sample_positions = np.linspace(0.0, 255.0, plot_width)
        curve = np.interp(sample_positions, np.arange(256), histogram[channel])
        y_positions = x_axis - np.rint(curve / peak * chart_height).astype(np.int32)
        previous_y = int(y_positions[0])
        for offset, y_position_value in enumerate(y_positions):
            y_position = int(y_position_value)
            x_position = _MARGIN_LEFT + offset
            segment_top = max(chart_top, min(previous_y, y_position) - 1)
            segment_bottom = min(x_axis, max(previous_y, y_position) + 1)
            canvas[segment_top : segment_bottom + 1, x_position] = color
            previous_y = y_position

    return canvas


def save_rgb_histogram(image: Image.Image | np.ndarray, output_path: Path) -> Path:
    """Render and save an RGB channel histogram PNG."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(render_rgb_histogram(image), mode="RGB").save(output_path)
    return output_path
