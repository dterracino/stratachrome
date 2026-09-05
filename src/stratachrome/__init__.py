"""
stratachrome
------------
Two-tier automated multi-color 3D relief generator.
Models Beer-Lambert optical transmission, isolates foreground/background
via BiRefNet, builds watertight manifold triangle meshes, and exports
native Bambu/Orca .3mf project containers.
"""

from __future__ import annotations

from stratachrome.color_engine import (
    auto_sort_filaments_by_lightness,
    build_layer_schedule,
    create_filament,
    extract_perceptual_lightness,
)
from stratachrome.depth_mapper import (
    HeightmapResult,
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
    Filament,
    FilamentLayerAssignment,
    LayerOpticalState,
    LightnessLayerMapper,
    simulate_tier_stack,
)
from stratachrome.segmentation import (
    ForegroundSegmenter,
    SegmentationConfig,
    SegmentationResult,
    partition_lightness_channels,
)

__version__ = "0.1.0"

__all__ = [
    # Optical & Color
    "Filament",
    "FilamentLayerAssignment",
    "LayerOpticalState",
    "LightnessLayerMapper",
    "simulate_tier_stack",
    "create_filament",
    "extract_perceptual_lightness",
    "auto_sort_filaments_by_lightness",
    "build_layer_schedule",
    # Segmentation
    "ForegroundSegmenter",
    "SegmentationConfig",
    "SegmentationResult",
    "partition_lightness_channels",
    # Depth
    "TierHeightBudget",
    "SwapEvent",
    "HeightmapResult",
    "TwoTierDepthMapper",
    # Geometry & Mesh
    "PhysicalDimensions",
    "TriangleMesh",
    "WatertightMeshBuilder",
    "export_binary_stl",
    # Packaging
    "export_bambu_3mf",
]