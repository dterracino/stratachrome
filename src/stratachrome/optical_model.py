"""
optical_model.py
----------------
Models filament transmission and perceived surface lightness (L*) using
the Beer-Lambert law. Produces 1D simulation curves and vectorized lookup
tables to map target lightness values to physical slicer layers.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class Filament:
    """Represents the optical and physical properties of a 3D printing filament.

    Attributes:
        name: Human-readable identifier.
        hex_color: Hex color string (e.g., '#FFFFFF').
        l_value: Intrinsic CIELCh L* value (0.0 to 100.0).
        td: Transmission Distance in millimeters (thickness to reach ~10% transmission).
    """
    name: str
    hex_color: str
    l_value: float
    td: float

    def __post_init__(self) -> None:
        if not (0.0 <= self.l_value <= 100.0):
            raise ValueError(f"l_value must be between 0.0 and 100.0, got {self.l_value}")
        if self.td <= 0.0:
            raise ValueError(f"Transmission Distance (td) must be positive, got {self.td}")


@dataclass(frozen=True)
class FilamentLayerAssignment:
    """Defines which layer a specific filament begins printing.

    Attributes:
        filament: The Filament instance being assigned.
        start_layer: The zero-based layer index where this filament begins.
    """
    filament: Filament
    start_layer: int

    def __post_init__(self) -> None:
        if self.start_layer < 0:
            raise ValueError(f"start_layer must be non-negative, got {self.start_layer}")


@dataclass(frozen=True)
class LayerOpticalState:
    """Stores the calculated optical state for a single slicer layer.

    Attributes:
        layer_index: Zero-based layer index.
        height_mm: Physical height in millimeters.
        active_filament: The filament being extruded on this layer.
        simulated_l: Resulting perceived surface lightness (0.0 to 100.0).
    """
    layer_index: int
    height_mm: float
    active_filament: Filament
    simulated_l: float


def _validate_ascending_lightness(assignments: list[FilamentLayerAssignment]) -> None:
    """Verifies that filaments are sequenced in strictly ascending lightness order."""
    if len(assignments) <= 1:
        return

    for i in range(len(assignments) - 1):
        curr_f = assignments[i].filament
        next_f = assignments[i + 1].filament
        if curr_f.l_value > next_f.l_value:
            raise ValueError(
                f"Filaments must be in ascending L* order. "
                f"'{curr_f.name}' (L*={curr_f.l_value:.2f}) is higher than "
                f"'{next_f.name}' (L*={next_f.l_value:.2f})."
            )


def _validate_layer_order(assignments: list[FilamentLayerAssignment]) -> None:
    """Verifies that starting layers are strictly increasing."""
    if len(assignments) <= 1:
        return

    for i in range(len(assignments) - 1):
        curr_layer = assignments[i].start_layer
        next_layer = assignments[i + 1].start_layer
        if next_layer <= curr_layer:
            raise ValueError(
                f"Start layers must strictly increase. "
                f"Found layer {curr_layer} followed by {next_layer}."
            )


def _calculate_transmittance(layer_thickness_mm: float, td_mm: float) -> float:
    """Computes fractional light transmittance through a layer via Beer-Lambert law."""
    if td_mm <= 0.0:
        return 0.0
    exponent = -layer_thickness_mm / td_mm
    return math.pow(10.0, exponent)


def _blend_lightness(substrate_l: float, overlay_l: float, transmittance: float) -> float:
    """Blends overlay lightness on top of a substrate based on transmittance."""
    clamped_t = max(0.0, min(1.0, transmittance))
    return (1.0 - clamped_t) * overlay_l + clamped_t * substrate_l


def _calculate_layer_height(
    layer_idx: int,
    first_layer_height_mm: float,
    step_height_mm: float,
) -> float:
    """Calculates the physical top Z coordinate of a given layer index."""
    if layer_idx == 0:
        return first_layer_height_mm
    return first_layer_height_mm + layer_idx * step_height_mm


def _resolve_active_filament(
    layer_idx: int,
    sorted_assignments: list[FilamentLayerAssignment],
) -> Filament:
    """Finds which filament is active for a given layer index."""
    active = sorted_assignments[0].filament
    for assignment in sorted_assignments:
        if layer_idx >= assignment.start_layer:
            active = assignment.filament
        else:
            break
    return active


def simulate_tier_stack(
    assignments: list[FilamentLayerAssignment],
    total_layers: int,
    step_height_mm: float = 0.10,
    first_layer_height_mm: float = 0.16,
    initial_substrate_l: float | None = None,
) -> list[LayerOpticalState]:
    """Simulates cumulative optical transmission layer-by-layer for a tier.

    Args:
        assignments: List of filament assignments with their starting layers.
        total_layers: Total number of layers to simulate in this tier.
        step_height_mm: Layer height for standard layers (default 0.10mm).
        first_layer_height_mm: Height for the initial base layer (default 0.16mm).
        initial_substrate_l: Lightness of the surface underneath this tier.
                             If None, defaults to the base filament's intrinsic L*.

    Returns:
        A list of LayerOpticalState instances representing each layer.
    """
    if not assignments:
        raise ValueError("At least one FilamentLayerAssignment is required.")
    if total_layers <= 0:
        raise ValueError("total_layers must be greater than zero.")

    sorted_assignments = sorted(assignments, key=lambda a: a.start_layer)
    _validate_layer_order(sorted_assignments)
    _validate_ascending_lightness(sorted_assignments)

    if sorted_assignments[0].start_layer != 0:
        raise ValueError("The first filament assignment must start at layer 0.")

    states: list[LayerOpticalState] = []
    
    current_surface_l = (
        initial_substrate_l
        if initial_substrate_l is not None
        else sorted_assignments[0].filament.l_value
    )

    for layer_idx in range(total_layers):
        thickness = first_layer_height_mm if layer_idx == 0 else step_height_mm
        height_mm = _calculate_layer_height(layer_idx, first_layer_height_mm, step_height_mm)
        active_filament = _resolve_active_filament(layer_idx, sorted_assignments)

        t = _calculate_transmittance(thickness, active_filament.td)
        current_surface_l = _blend_lightness(
            substrate_l=current_surface_l,
            overlay_l=active_filament.l_value,
            transmittance=t,
        )

        states.append(
            LayerOpticalState(
                layer_index=layer_idx,
                height_mm=round(height_mm, 4),
                active_filament=active_filament,
                simulated_l=round(current_surface_l, 4),
            )
        )

    return states


class LightnessLayerMapper:
    """Provides high-performance vectorized mapping from target L* to layer indices."""

    def __init__(self, optical_states: list[LayerOpticalState]) -> None:
        if not optical_states:
            raise ValueError("optical_states cannot be empty.")
        self._states = optical_states
        self._layers = np.array([s.layer_index for s in optical_states], dtype=np.int32)
        self._heights = np.array([s.height_mm for s in optical_states], dtype=np.float32)
        self._l_curves = np.array([s.simulated_l for s in optical_states], dtype=np.float32)

    @property
    def simulated_l_curve(self) -> np.ndarray:
        return self._l_curves.copy()

    @property
    def layer_indices(self) -> np.ndarray:
        return self._layers.copy()

    @property
    def height_array(self) -> np.ndarray:
        return self._heights.copy()

    def map_image_lightness_to_layers(self, target_l_image: np.ndarray) -> np.ndarray:
        """Maps a 2D array of target L* values (0.0 to 100.0) to discrete layer indices."""
        flat_targets = target_l_image.flatten()
        
        idx = np.searchsorted(self._l_curves, flat_targets)
        idx = np.clip(idx, 1, len(self._l_curves) - 1)

        left = self._l_curves[idx - 1]
        right = self._l_curves[idx]
        
        choose_left = (flat_targets - left) <= (right - flat_targets)
        best_indices = np.where(choose_left, idx - 1, idx)

        best_layers = self._layers[best_indices]
        return best_layers.reshape(target_l_image.shape)

    def map_image_lightness_to_heights(self, target_l_image: np.ndarray) -> np.ndarray:
        """Maps a 2D array of target L* values directly to millimeter heights."""
        layer_grid = self.map_image_lightness_to_layers(target_l_image)
        return self._heights[layer_grid]