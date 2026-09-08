"""Render predicted optical colors and fixed-scale perceptual error maps."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from stratachrome.depth_mapper import ColorDiagnosticResult

DELTA_E_HEATMAP_MAX = 30.0

_HEATMAP_COLORS = np.asarray(
    (
        (0, 0, 0),
        (28, 72, 140),
        (32, 190, 180),
        (246, 215, 70),
        (190, 30, 45),
    ),
    dtype=np.float32,
)


def render_delta_e_heatmap(
    delta_e: np.ndarray,
    maximum: float = DELTA_E_HEATMAP_MAX,
) -> np.ndarray:
    """Map Delta E values to a fixed black-blue-cyan-yellow-red scale."""
    values = np.asarray(delta_e, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("delta_e must be a 2D array.")
    if maximum <= 0.0:
        raise ValueError("maximum must be positive.")

    positions = np.clip(values / maximum, 0.0, 1.0) * (len(_HEATMAP_COLORS) - 1)
    lower = np.floor(positions).astype(np.int32)
    upper = np.minimum(lower + 1, len(_HEATMAP_COLORS) - 1)
    weights = (positions - lower)[..., None]
    rgb = _HEATMAP_COLORS[lower] * (1.0 - weights) + _HEATMAP_COLORS[upper] * weights
    return np.rint(rgb).astype(np.uint8)


def save_color_diagnostics(
    diagnostics: ColorDiagnosticResult,
    project_path: Path,
) -> tuple[Path, Path]:
    """Save predicted-color and Delta E PNGs beside a project output path."""
    preview_path = project_path.with_name(f"{project_path.stem}_color_preview.png")
    heatmap_path = project_path.with_name(f"{project_path.stem}_delta_e_heatmap.png")
    preview_path.parent.mkdir(parents=True, exist_ok=True)

    Image.fromarray(diagnostics.preview_rgb).save(preview_path)
    Image.fromarray(render_delta_e_heatmap(diagnostics.delta_e)).save(heatmap_path)
    return preview_path, heatmap_path