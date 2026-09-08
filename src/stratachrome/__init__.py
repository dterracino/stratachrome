"""
stratachrome
------------
Single- or two-tier automated multi-color 3D relief generator.
Models Beer-Lambert optical transmission, optionally isolates foreground and
background via BiRefNet, builds watertight meshes, and exports Bambu/Orca 3MF.
"""

from __future__ import annotations

from typing import Any

from stratachrome.color_engine import (
    FilamentMatch,
    TierColorPlan,
    TierPalette,
    build_tier_image,
    extract_perceptual_lab,
    extract_perceptual_lightness,
    plan_tier_colors,
    sample_tier_lab,
    select_tier_palette,
)
from stratachrome.depth_mapper import (
    HeightmapResult,
    SingleTierDepthMapper,
    SwapEvent,
    TierHeightBudget,
    TwoTierDepthMapper,
)
from stratachrome.export_3mf import export_bambu_3mf
from stratachrome.mesh_builder import (
    PhysicalDimensions,
    TriangleMesh,
    WatertightMeshBuilder,
    export_binary_stl,
)
from stratachrome.optical_model import (
    ColorLayerMapper,
    FilamentLayerAssignment,
    LayerOpticalState,
    OptimizedTierSchedule,
    assignments_from_layer_counts,
    optimize_tier_schedule,
    simulate_tier_stack,
)

__version__ = "0.1.0"

_SEGMENTATION_EXPORTS = {
    "ForegroundSegmenter",
    "SegmentationConfig",
    "SegmentationResult",
    "partition_lightness_channels",
}


def __getattr__(name: str) -> Any:
    if name in _SEGMENTATION_EXPORTS:
        from stratachrome import segmentation

        return getattr(segmentation, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Optical & Color
    "FilamentLayerAssignment",
    "LayerOpticalState",
    "OptimizedTierSchedule",
    "ColorLayerMapper",
    "simulate_tier_stack",
    "optimize_tier_schedule",
    "assignments_from_layer_counts",
    "FilamentMatch",
    "TierColorPlan",
    "TierPalette",
    "build_tier_image",
    "extract_perceptual_lab",
    "extract_perceptual_lightness",
    "plan_tier_colors",
    "sample_tier_lab",
    "select_tier_palette",
    # Segmentation
    "ForegroundSegmenter",
    "SegmentationConfig",
    "SegmentationResult",
    "partition_lightness_channels",
    # Depth
    "TierHeightBudget",
    "SwapEvent",
    "HeightmapResult",
    "SingleTierDepthMapper",
    "TwoTierDepthMapper",
    # Geometry & Mesh
    "PhysicalDimensions",
    "TriangleMesh",
    "WatertightMeshBuilder",
    "export_binary_stl",
    # Packaging
    "export_bambu_3mf",
]
