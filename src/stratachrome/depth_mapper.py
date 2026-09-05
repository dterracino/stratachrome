"""
depth_mapper.py
---------------
Transforms zone-separated lightness arrays into physical millimeter heights
using two-tier stacking. Enforces the background ceiling as a solid pedestal
beneath the foreground and provides smooth, anti-sheer boundary blending.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from stratachrome.optical_model import LayerOpticalState, LightnessLayerMapper


@dataclass(frozen=True)
class TierHeightBudget:
    """Defines physical layer counts and vertical dimensions for a printing tier.

    Attributes:
        layer_count: Total discrete slicer layers allocated to this tier.
        step_height_mm: Layer height for standard layers (e.g., 0.10mm).
        first_layer_height_mm: Height for the bed-contact layer (e.g., 0.16mm).
        floor_z_mm: Base elevation where this tier starts.
    """
    layer_count: int
    step_height_mm: float = 0.10
    first_layer_height_mm: float = 0.16
    floor_z_mm: float = 0.0

    def __post_init__(self) -> None:
        if self.layer_count <= 0:
            raise ValueError(f"layer_count must be positive, got {self.layer_count}")
        if self.step_height_mm <= 0.0:
            raise ValueError(f"step_height_mm must be positive, got {self.step_height_mm}")
        if self.first_layer_height_mm <= 0.0:
            raise ValueError(
                f"first_layer_height_mm must be positive, got {self.first_layer_height_mm}"
            )
        if self.floor_z_mm < 0.0:
            raise ValueError(f"floor_z_mm cannot be negative, got {self.floor_z_mm}")

    @property
    def total_height_mm(self) -> float:
        """Computes the total vertical span of this tier in millimeters."""
        if self.floor_z_mm == 0.0:
            return self.first_layer_height_mm + (self.layer_count - 1) * self.step_height_mm
        return self.layer_count * self.step_height_mm

    @property
    def ceiling_z_mm(self) -> float:
        """Computes the absolute maximum Z elevation in millimeters."""
        return self.floor_z_mm + self.total_height_mm


@dataclass(frozen=True)
class SwapEvent:
    """Represents a physical filament change event for slicer metadata.

    Attributes:
        global_layer_idx: Zero-based global slicer layer index.
        z_height_mm: Physical elevation in millimeters where the pause occurs.
        filament_name: Name of the incoming filament.
        filament_hex: Hex code of the incoming filament.
        tier_name: Which zone this filament belongs to ('background' or 'foreground').
    """
    global_layer_idx: int
    z_height_mm: float
    filament_name: str
    filament_hex: str
    tier_name: str


@dataclass(frozen=True)
class HeightmapResult:
    """Container for the computed surface geometry and print schedule.

    Attributes:
        z_grid: 2D float32 array of top surface elevations in millimeters.
        bg_surface_z: 2D float32 array of background elevations.
        fg_surface_z: 2D float32 array of foreground elevations.
        total_layers: Total slicer layers across both tiers.
        max_height_mm: Maximum physical elevation reached.
        swap_schedule: Chronological list of filament swap pauses.
    """
    z_grid: np.ndarray
    bg_surface_z: np.ndarray
    fg_surface_z: np.ndarray
    total_layers: int
    max_height_mm: float
    swap_schedule: list[SwapEvent]


def _validate_image_dimensions(
    bg_l: np.ndarray,
    fg_l: np.ndarray,
    matte: np.ndarray,
) -> None:
    """Verifies that all input matrices share identical 2D spatial dimensions."""
    if bg_l.shape != fg_l.shape or bg_l.shape != matte.shape:
        raise ValueError(
            f"Shape mismatch: bg_l {bg_l.shape}, fg_l {fg_l.shape}, matte {matte.shape}"
        )
    if bg_l.ndim != 2:
        raise ValueError(f"Arrays must be 2D grids, got ndim={bg_l.ndim}")


def _build_tier_height_array(
    budget: TierHeightBudget,
    is_base_tier: bool,
) -> np.ndarray:
    """Generates a 1D array of absolute Z elevations for each layer in a tier."""
    heights = np.empty(budget.layer_count, dtype=np.float32)
    for idx in range(budget.layer_count):
        if is_base_tier and idx == 0:
            heights[idx] = budget.first_layer_height_mm
        elif is_base_tier:
            heights[idx] = budget.first_layer_height_mm + idx * budget.step_height_mm
        else:
            heights[idx] = budget.floor_z_mm + (idx + 1) * budget.step_height_mm
    return heights


def _map_zone_to_elevations(
    target_l: np.ndarray,
    mapper: LightnessLayerMapper,
    height_lut: np.ndarray,
) -> np.ndarray:
    """Maps a 2D L* channel to discrete millimeter elevations using a height LUT."""
    layer_indices = mapper.map_image_lightness_to_layers(target_l)
    clamped_indices = np.clip(layer_indices, 0, len(height_lut) - 1)
    return height_lut[clamped_indices]


def _blend_boundaries(
    bg_z: np.ndarray,
    fg_z: np.ndarray,
    matte: np.ndarray,
) -> np.ndarray:
    """Combines background and foreground surfaces using soft matte blending.

    Formula: Z(x, y) = (1 - M) * Z_bg + M * Z_fg
    """
    clamped_matte = np.clip(matte, 0.0, 1.0).astype(np.float32)
    blended = (1.0 - clamped_matte) * bg_z + clamped_matte * fg_z
    return blended.astype(np.float32)


def _extract_tier_swaps(
    states: list[LayerOpticalState],
    layer_offset: int,
    tier_name: str,
    height_lut: np.ndarray,
) -> list[SwapEvent]:
    """Identifies filament change points across a tier's optical states."""
    swaps: list[SwapEvent] = []
    seen_filaments: set[str] = set()

    for state in states:
        fil = state.active_filament
        if fil.name not in seen_filaments:
            seen_filaments.add(fil.name)
            global_idx = state.layer_index + layer_offset
            z_pos = float(height_lut[state.layer_index])
            swaps.append(
                SwapEvent(
                    global_layer_idx=global_idx,
                    z_height_mm=round(z_pos, 4),
                    filament_name=fil.name,
                    filament_hex=fil.hex_color,
                    tier_name=tier_name,
                )
            )
    return swaps


class TwoTierDepthMapper:
    """Coordinates height generation, stacking, and swap scheduling for 2-tier models."""

    def __init__(
        self,
        bg_mapper: LightnessLayerMapper,
        fg_mapper: LightnessLayerMapper,
        bg_states: list[LayerOpticalState],
        fg_states: list[LayerOpticalState],
        step_height_mm: float = 0.10,
        first_layer_height_mm: float = 0.16,
    ) -> None:
        self._bg_mapper = bg_mapper
        self._fg_mapper = fg_mapper
        self._bg_states = bg_states
        self._fg_states = fg_states
        self._step_height_mm = step_height_mm
        self._first_layer_height_mm = first_layer_height_mm

        self._bg_budget = TierHeightBudget(
            layer_count=len(bg_states),
            step_height_mm=step_height_mm,
            first_layer_height_mm=first_layer_height_mm,
            floor_z_mm=0.0,
        )

        self._fg_budget = TierHeightBudget(
            layer_count=len(fg_states),
            step_height_mm=step_height_mm,
            first_layer_height_mm=step_height_mm,
            floor_z_mm=self._bg_budget.ceiling_z_mm,
        )

        self._bg_height_lut = _build_tier_height_array(self._bg_budget, is_base_tier=True)
        self._fg_height_lut = _build_tier_height_array(self._fg_budget, is_base_tier=False)

    @property
    def bg_budget(self) -> TierHeightBudget:
        return self._bg_budget

    @property
    def fg_budget(self) -> TierHeightBudget:
        return self._fg_budget

    def generate_heightmap(
        self,
        bg_l: np.ndarray,
        fg_l: np.ndarray,
        matte: np.ndarray,
    ) -> HeightmapResult:
        """Assembles a full 3D surface grid from zone lightness channels and matte.

        Args:
            bg_l: 2D float array of background CIELCh L* values (0.0 to 100.0).
            fg_l: 2D float array of foreground CIELCh L* values (0.0 to 100.0).
            matte: 2D float array in [0.0, 1.0] representing foreground opacity.

        Returns:
            HeightmapResult containing the merged Z-grid and print schedule.
        """
        _validate_image_dimensions(bg_l, fg_l, matte)

        # 1. Map zone lightness to physical elevations
        bg_z = _map_zone_to_elevations(bg_l, self._bg_mapper, self._bg_height_lut)
        fg_z = _map_zone_to_elevations(fg_l, self._fg_mapper, self._fg_height_lut)

        # 2. Smoothly blend across feathered boundaries
        z_grid = _blend_boundaries(bg_z, fg_z, matte)

        # 3. Assemble swap schedules
        bg_swaps = _extract_tier_swaps(
            self._bg_states,
            layer_offset=0,
            tier_name="background",
            height_lut=self._bg_height_lut,
        )
        fg_swaps = _extract_tier_swaps(
            self._fg_states,
            layer_offset=self._bg_budget.layer_count,
            tier_name="foreground",
            height_lut=self._fg_height_lut,
        )
        full_schedule = bg_swaps + fg_swaps

        total_layers = self._bg_budget.layer_count + self._fg_budget.layer_count
        max_height = float(np.max(z_grid))

        return HeightmapResult(
            z_grid=z_grid,
            bg_surface_z=bg_z,
            fg_surface_z=fg_z,
            total_layers=total_layers,
            max_height_mm=round(max_height, 4),
            swap_schedule=full_schedule,
        )