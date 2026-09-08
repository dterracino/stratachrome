"""
depth_mapper.py
---------------
Transforms CIELAB lightness arrays into discrete physical elevations for single-
or two-tier reliefs. Two-tier mode places a solid background pedestal beneath
the foreground. A dedicated first-layer height is included in base geometry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
from color_tools import delta_e_2000_array

from stratachrome.optical_model import ColorLayerMapper, LayerOpticalState


@dataclass(frozen=True)
class TierHeightBudget:
    """Defines physical layer counts and vertical dimensions for a printing tier.

    Attributes:
        layer_count: Total discrete slicer layers allocated to this tier.
        step_height_mm: Layer height for standard layers (default: 0.10mm).
        first_layer_height_mm: Height for the bed-contact layer (default: 0.20mm).
        floor_z_mm: Base elevation where this tier starts.
    """

    layer_count: int
    step_height_mm: float = 0.10
    first_layer_height_mm: float = 0.20
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
    filament_type: str = "PLA"
    filament_finish: str = "Basic"
    filament_id: str = ""


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


@dataclass(frozen=True)
class ColorDiagnosticResult:
    """Predicted printed colors and their perceptual error against the source."""

    preview_rgb: np.ndarray
    delta_e: np.ndarray
    mean_delta_e: float
    percentile_95_delta_e: float
    max_delta_e: float


def _validate_image_dimensions(
    bg_lab: np.ndarray,
    fg_lab: np.ndarray,
    matte: np.ndarray,
) -> None:
    """Verify that both Lab images and the matte share spatial dimensions."""
    if bg_lab.shape != fg_lab.shape or bg_lab.shape[:2] != matte.shape:
        raise ValueError(
            f"Shape mismatch: bg_lab {bg_lab.shape}, " f"fg_lab {fg_lab.shape}, matte {matte.shape}"
        )
    if bg_lab.ndim != 3 or bg_lab.shape[-1] != 3 or matte.ndim != 2:
        raise ValueError("Lab images must be H x W x 3 and matte must be H x W.")


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


def _map_lightness_to_layer_indices(
    target_lab: np.ndarray,
    layer_count: int,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Map tier-local CIELAB L* values to discrete layer indices."""
    lightness = np.asarray(target_lab[..., 0], dtype=np.float32)
    samples = lightness if mask is None else lightness[mask]
    if samples.size == 0:
        raise ValueError("Cannot map lightness for an empty tier.")

    minimum = float(np.min(samples))
    maximum = float(np.max(samples))
    if maximum <= minimum:
        layer_indices = np.zeros(lightness.shape, dtype=np.int32)
    else:
        normalized = np.clip((lightness - minimum) / (maximum - minimum), 0.0, 1.0)
        layer_indices = np.rint(normalized * (layer_count - 1)).astype(np.int32)
    return layer_indices


def _map_lightness_to_elevations(
    target_lab: np.ndarray,
    height_lut: np.ndarray,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Map tier-local CIELAB L* values to discrete printable elevations."""
    layer_indices = _map_lightness_to_layer_indices(target_lab, len(height_lut), mask)
    return cast(np.ndarray, height_lut[layer_indices])


def _diagnostics_from_layer_indices(
    target_lab: np.ndarray,
    layer_indices: np.ndarray,
    states: list[LayerOpticalState],
) -> ColorDiagnosticResult:
    """Build predicted RGB and Delta E arrays from selected optical states."""
    state_lab = np.asarray([state.simulated_lab for state in states], dtype=np.float64)
    state_rgb = np.asarray([state.simulated_rgb for state in states], dtype=np.uint8)
    predicted_lab = state_lab[layer_indices]
    delta_e = np.asarray(
        delta_e_2000_array(np.asarray(target_lab, dtype=np.float64), predicted_lab),
        dtype=np.float32,
    )
    return ColorDiagnosticResult(
        preview_rgb=state_rgb[layer_indices],
        delta_e=delta_e,
        mean_delta_e=float(np.mean(delta_e)),
        percentile_95_delta_e=float(np.percentile(delta_e, 95.0)),
        max_delta_e=float(np.max(delta_e)),
    )


def _map_zone_to_elevations(
    target_lab: np.ndarray,
    mapper: ColorLayerMapper,
    height_lut: np.ndarray,
) -> np.ndarray:
    """Map a Lab image to discrete millimeter elevations using a height LUT."""
    layer_indices = mapper.map_image_lab_to_layers(target_lab)
    clamped_indices = np.clip(layer_indices, 0, len(height_lut) - 1)
    return cast(np.ndarray, height_lut[clamped_indices])


def _blend_boundaries(
    bg_z: np.ndarray,
    fg_z: np.ndarray,
    matte: np.ndarray,
) -> np.ndarray:
    """Combine background and foreground surfaces using soft matte blending."""
    clamped_matte = np.clip(matte, 0.0, 1.0).astype(np.float32)
    blended = (1.0 - clamped_matte) * bg_z + clamped_matte * fg_z
    return cast(np.ndarray, blended.astype(np.float32))


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
        if fil.id not in seen_filaments:
            seen_filaments.add(fil.id)
            global_idx = state.layer_index + layer_offset
            z_pos = float(height_lut[state.layer_index])
            swaps.append(
                SwapEvent(
                    global_layer_idx=global_idx,
                    z_height_mm=round(z_pos, 4),
                    filament_name=" ".join(
                        part for part in (fil.maker, fil.type, fil.finish, fil.color) if part
                    ),
                    filament_hex=fil.hex,
                    tier_name=tier_name,
                    filament_type=fil.type,
                    filament_finish=fil.finish,
                    filament_id=fil.id,
                )
            )
    return swaps


class TwoTierDepthMapper:
    """Coordinate optical color mapping, stacking, and swaps for two-tier models."""

    def __init__(
        self,
        bg_states: list[LayerOpticalState],
        fg_states: list[LayerOpticalState],
        step_height_mm: float = 0.10,
        first_layer_height_mm: float = 0.20,
    ) -> None:
        self._bg_mapper = ColorLayerMapper(bg_states)
        self._fg_mapper = ColorLayerMapper(fg_states)
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
        bg_lab: np.ndarray,
        fg_lab: np.ndarray,
        matte: np.ndarray,
    ) -> HeightmapResult:
        """Assembles a full 3D surface grid from zone lightness channels and matte.

        Args:
            bg_lab: Background target colors shaped H x W x 3.
            fg_lab: Foreground target colors shaped H x W x 3.
            matte: 2D float array in [0.0, 1.0] representing foreground opacity.

        Returns:
            HeightmapResult containing the merged Z-grid and print schedule.
        """
        _validate_image_dimensions(bg_lab, fg_lab, matte)

        # 1. Map target colors to the closest simulated optical states.
        bg_z = _map_zone_to_elevations(bg_lab, self._bg_mapper, self._bg_height_lut)
        fg_z = _map_zone_to_elevations(fg_lab, self._fg_mapper, self._fg_height_lut)

        # 2. Smoothly blend across feathered segmentation boundaries.
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

    def generate_color_diagnostics(
        self,
        bg_lab: np.ndarray,
        fg_lab: np.ndarray,
        matte: np.ndarray,
    ) -> ColorDiagnosticResult:
        """Predict the realizable optical state selected for every source pixel."""
        _validate_image_dimensions(bg_lab, fg_lab, matte)
        foreground_mask = matte >= 0.5
        bg_indices = self._bg_mapper.map_image_lab_to_layers(bg_lab)
        fg_indices = self._fg_mapper.map_image_lab_to_layers(fg_lab)
        target_lab = np.where(foreground_mask[..., None], fg_lab, bg_lab)

        bg_state_count = len(self._bg_states)
        combined_states = self._bg_states + self._fg_states
        combined_indices = np.where(
            foreground_mask,
            fg_indices + bg_state_count,
            bg_indices,
        )
        return _diagnostics_from_layer_indices(target_lab, combined_indices, combined_states)


class SingleTierDepthMapper:
    """Generate one lightness-driven relief and its filament schedule."""

    def __init__(
        self,
        states: list[LayerOpticalState],
        step_height_mm: float = 0.10,
        first_layer_height_mm: float = 0.20,
    ) -> None:
        self._states = states
        self._budget = TierHeightBudget(
            layer_count=len(states),
            step_height_mm=step_height_mm,
            first_layer_height_mm=first_layer_height_mm,
        )
        self._height_lut = _build_tier_height_array(self._budget, is_base_tier=True)

    @property
    def budget(self) -> TierHeightBudget:
        return self._budget

    def generate_heightmap(self, lab_image: np.ndarray) -> HeightmapResult:
        """Map the full image L* range onto one discrete tier."""
        if lab_image.ndim != 3 or lab_image.shape[-1] != 3:
            raise ValueError("lab_image must have shape (height, width, 3).")

        z_grid = _map_lightness_to_elevations(lab_image, self._height_lut)
        swaps = _extract_tier_swaps(
            self._states,
            layer_offset=0,
            tier_name="single",
            height_lut=self._height_lut,
        )
        return HeightmapResult(
            z_grid=z_grid,
            bg_surface_z=z_grid,
            fg_surface_z=z_grid,
            total_layers=self._budget.layer_count,
            max_height_mm=round(float(np.max(z_grid)), 4),
            swap_schedule=swaps,
        )

    def generate_color_diagnostics(self, lab_image: np.ndarray) -> ColorDiagnosticResult:
        """Predict colors for the same lightness-selected states used by geometry."""
        if lab_image.ndim != 3 or lab_image.shape[-1] != 3:
            raise ValueError("lab_image must have shape (height, width, 3).")
        layer_indices = _map_lightness_to_layer_indices(lab_image, len(self._states))
        return _diagnostics_from_layer_indices(lab_image, layer_indices, self._states)
