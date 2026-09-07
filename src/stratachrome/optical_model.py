"""Optical stack simulation and perceptual layer-schedule optimization."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import numpy as np
from color_tools import (
    FilamentRecord,
    delta_e_2000_array,
    lab_to_xyz,
    rgb_to_xyz,
    xyz_to_lab,
    xyz_to_rgb,
)

LabColor = tuple[float, float, float]
XYZColor = tuple[float, float, float]


@dataclass(frozen=True)
class FilamentLayerAssignment:
    """A color_tools filament and the zero-based layer where it starts."""

    filament: FilamentRecord
    start_layer: int

    def __post_init__(self) -> None:
        if self.start_layer < 0:
            raise ValueError(f"start_layer must be non-negative, got {self.start_layer}")
        _require_td(self.filament)


@dataclass(frozen=True)
class LayerOpticalState:
    """Predicted color and active filament at one printable layer."""

    layer_index: int
    height_mm: float
    active_filament: FilamentRecord
    simulated_lab: LabColor
    simulated_rgb: tuple[int, int, int]

    @property
    def simulated_l(self) -> float:
        """Return CIELAB lightness for compatibility and reporting."""
        return self.simulated_lab[0]


@dataclass(frozen=True)
class OptimizedTierSchedule:
    """Optimized assignments, optical states, and fit diagnostics for one tier."""

    assignments: tuple[FilamentLayerAssignment, ...]
    states: tuple[LayerOpticalState, ...]
    layer_counts: tuple[int, ...]
    mean_delta_e: float


def _require_td(filament: FilamentRecord) -> float:
    td = filament.td_value
    if td is None or td <= 0.0:
        raise ValueError(f"Filament '{filament.id}' requires a positive td_value; got {td!r}.")
    return float(td)


def _calculate_transmittance(layer_thickness_mm: float, td_mm: float) -> float:
    """Return the substrate contribution using the project's TD convention."""
    if layer_thickness_mm <= 0.0:
        raise ValueError("layer_thickness_mm must be positive.")
    if td_mm <= 0.0:
        raise ValueError("td_mm must be positive.")
    return math.pow(10.0, -layer_thickness_mm / td_mm)


def _blend_xyz(substrate: XYZColor, overlay: XYZColor, transmittance: float) -> XYZColor:
    """Blend tristimulus values in linear XYZ using transmission as the weight."""
    t = max(0.0, min(1.0, transmittance))
    return tuple(
        (1.0 - t) * overlay_component + t * substrate_component
        for substrate_component, overlay_component in zip(substrate, overlay)
    )  # type: ignore[return-value]


def _calculate_layer_height(
    layer_idx: int,
    first_layer_height_mm: float,
    step_height_mm: float,
) -> float:
    if layer_idx == 0:
        return first_layer_height_mm
    return first_layer_height_mm + layer_idx * step_height_mm


def _validate_assignments(assignments: Sequence[FilamentLayerAssignment]) -> None:
    if not assignments:
        raise ValueError("At least one filament assignment is required.")
    if assignments[0].start_layer != 0:
        raise ValueError("The first filament assignment must start at layer 0.")

    for current, following in zip(assignments, assignments[1:]):
        if following.start_layer <= current.start_layer:
            raise ValueError("Filament start layers must strictly increase.")
        if following.filament.lab[0] < current.filament.lab[0]:
            raise ValueError("Filaments must be ordered by ascending CIELAB L*.")


def _resolve_active_filament(
    layer_idx: int,
    assignments: Sequence[FilamentLayerAssignment],
) -> FilamentRecord:
    active = assignments[0].filament
    for assignment in assignments:
        if layer_idx < assignment.start_layer:
            break
        active = assignment.filament
    return active


def simulate_tier_stack(
    assignments: Sequence[FilamentLayerAssignment],
    total_layers: int,
    step_height_mm: float = 0.10,
    first_layer_height_mm: float = 0.20,
    initial_substrate_lab: LabColor | None = None,
) -> list[LayerOpticalState]:
    """Simulate a fixed bottom-to-top filament schedule in CIE XYZ and Lab."""
    sorted_assignments = tuple(sorted(assignments, key=lambda item: item.start_layer))
    _validate_assignments(sorted_assignments)
    if total_layers <= 0:
        raise ValueError("total_layers must be greater than zero.")
    if step_height_mm <= 0.0 or first_layer_height_mm <= 0.0:
        raise ValueError("Layer heights must be positive.")

    if initial_substrate_lab is None:
        current_xyz = rgb_to_xyz(sorted_assignments[0].filament.rgb)
    else:
        current_xyz = lab_to_xyz(initial_substrate_lab)

    states: list[LayerOpticalState] = []
    for layer_idx in range(total_layers):
        thickness = first_layer_height_mm if layer_idx == 0 else step_height_mm
        filament = _resolve_active_filament(layer_idx, sorted_assignments)
        transmittance = _calculate_transmittance(thickness, _require_td(filament))
        current_xyz = _blend_xyz(
            substrate=current_xyz,
            overlay=rgb_to_xyz(filament.rgb),
            transmittance=transmittance,
        )
        simulated_lab = tuple(float(value) for value in xyz_to_lab(current_xyz))
        states.append(
            LayerOpticalState(
                layer_index=layer_idx,
                height_mm=round(
                    _calculate_layer_height(
                        layer_idx,
                        first_layer_height_mm,
                        step_height_mm,
                    ),
                    4,
                ),
                active_filament=filament,
                simulated_lab=simulated_lab,  # type: ignore[arg-type]
                simulated_rgb=xyz_to_rgb(current_xyz),
            )
        )
    return states


def _mean_nearest_delta_e(target_lab: np.ndarray, state_lab: np.ndarray) -> float:
    distances = delta_e_2000_array(
        target_lab[:, None, :],
        state_lab[None, :, :],
    )
    return float(np.mean(np.min(distances, axis=1)))


def _candidate_layer_limit(
    filament: FilamentRecord,
    step_height_mm: float,
    terminal_transmittance: float,
    tier_layer_budget: int,
) -> int:
    """Bound search where the accumulated overlay is effectively opaque."""
    td = _require_td(filament)
    optical_thickness = -td * math.log10(terminal_transmittance)
    return max(1, min(tier_layer_budget, math.ceil(optical_thickness / step_height_mm)))


def _minimum_layers_for_contribution(
    filament: FilamentRecord,
    *,
    first_layer_height_mm: float,
    step_height_mm: float,
    minimum_color_contribution: float,
) -> int:
    """Return the layers needed for a filament to contribute visibly."""
    required_thickness = -_require_td(filament) * math.log10(
        1.0 - minimum_color_contribution
    )
    if required_thickness <= first_layer_height_mm:
        return 1
    return 1 + math.ceil(
        (required_thickness - first_layer_height_mm) / step_height_mm
    )


def _schedule_score(
    ordered: Sequence[FilamentRecord],
    layer_counts: Sequence[int],
    target_lab: np.ndarray,
    target_by_filament: Mapping[str, LabColor],
    *,
    step_height_mm: float,
    first_layer_height_mm: float,
    initial_substrate_lab: LabColor | None,
    anchor_weight: float,
    layer_penalty: float,
) -> tuple[float, list[LayerOpticalState], float]:
    assignments = assignments_from_layer_counts(ordered, layer_counts)
    states = simulate_tier_stack(
        assignments,
        total_layers=sum(layer_counts),
        step_height_mm=step_height_mm,
        first_layer_height_mm=first_layer_height_mm,
        initial_substrate_lab=initial_substrate_lab,
    )
    state_lab = np.asarray([state.simulated_lab for state in states], dtype=np.float64)
    mean_delta_e = _mean_nearest_delta_e(target_lab, state_lab)

    anchor_errors: list[float] = []
    start = 0
    for filament, count in zip(ordered, layer_counts):
        segment_lab = state_lab[start : start + count]
        anchor = np.asarray(
            target_by_filament.get(filament.id, filament.lab),
            dtype=np.float64,
        )
        distances = delta_e_2000_array(segment_lab, anchor)
        anchor_errors.append(float(np.min(distances)))
        start += count

    score = (
        mean_delta_e
        + anchor_weight * float(np.mean(anchor_errors))
        + layer_penalty * sum(layer_counts)
    )
    return score, states, mean_delta_e


def _optimize_bounded_schedule(
    ordered: Sequence[FilamentRecord],
    minimum_counts: Sequence[int],
    maximum_counts: Sequence[int],
    max_layers_per_tier: int,
    target_lab: np.ndarray,
    target_by_filament: Mapping[str, LabColor],
    *,
    step_height_mm: float,
    first_layer_height_mm: float,
    initial_substrate_lab: LabColor | None,
    anchor_weight: float,
    layer_penalty: float,
    fit_tolerance: float,
) -> tuple[list[int], list[LayerOpticalState], float]:
    """Optimize color thicknesses without exceeding a total tier budget."""
    counts = list(minimum_counts)
    cache: dict[tuple[int, ...], tuple[float, list[LayerOpticalState], float]] = {}

    def score(candidate_counts: Sequence[int]):
        key = tuple(candidate_counts)
        if key not in cache:
            cache[key] = _schedule_score(
                ordered,
                candidate_counts,
                target_lab,
                target_by_filament,
                step_height_mm=step_height_mm,
                first_layer_height_mm=first_layer_height_mm,
                initial_substrate_lab=initial_substrate_lab,
                anchor_weight=anchor_weight,
                layer_penalty=layer_penalty,
            )
        return cache[key]

    best_score, best_states, best_mean_delta_e = score(counts)

    while True:
        best_counts: list[int] | None = None
        move_score = best_score
        move_states = best_states
        move_mean_delta_e = best_mean_delta_e

        candidates: list[list[int]] = []
        if sum(counts) < max_layers_per_tier:
            for receiver, receiver_count in enumerate(counts):
                if receiver_count < maximum_counts[receiver]:
                    candidate = counts.copy()
                    candidate[receiver] += 1
                    candidates.append(candidate)

        for donor, donor_count in enumerate(counts):
            if donor_count > minimum_counts[donor]:
                candidate = counts.copy()
                candidate[donor] -= 1
                candidates.append(candidate)
            for receiver in range(len(counts)):
                if (
                    receiver == donor
                    or donor_count <= minimum_counts[donor]
                    or counts[receiver] >= maximum_counts[receiver]
                ):
                    continue
                candidate = counts.copy()
                candidate[donor] -= 1
                candidate[receiver] += 1
                candidates.append(candidate)

        for candidate in candidates:
            candidate_score, states, mean_delta_e = score(candidate)
            if candidate_score < move_score - fit_tolerance:
                best_counts = candidate
                move_score = candidate_score
                move_states = states
                move_mean_delta_e = mean_delta_e

        if best_counts is None:
            return counts, best_states, best_mean_delta_e
        counts = best_counts
        best_score = move_score
        best_states = move_states
        best_mean_delta_e = move_mean_delta_e


def optimize_tier_schedule(
    filaments: Sequence[FilamentRecord],
    target_lab: np.ndarray,
    *,
    step_height_mm: float = 0.10,
    first_layer_height_mm: float = 0.20,
    initial_substrate_lab: LabColor | None = None,
    target_by_filament: Mapping[str, LabColor] | None = None,
    fit_tolerance: float = 0.02,
    layer_penalty: float = 0.25,
    minimum_color_contribution: float = 0.12,
    anchor_weight: float = 0.50,
    terminal_transmittance: float = 0.02,
    max_layers_per_tier: int = 120,
) -> OptimizedTierSchedule:
    """Choose useful color thicknesses within a maximum tier layer budget.

    Optimization starts with the TD-derived minimum for every selected color.
    It may add layers, remove previously added layers, or move a layer between
    colors while doing so improves the perceptual objective enough to justify
    its physical thickness. The resulting tier may use less than
    ``max_layers_per_tier``, but never more.
    """
    if not filaments:
        raise ValueError("At least one filament is required.")
    if target_lab.ndim != 2 or target_lab.shape[1] != 3 or len(target_lab) == 0:
        raise ValueError("target_lab must be a non-empty array shaped (N, 3).")
    if not 0.0 < terminal_transmittance < 1.0:
        raise ValueError("terminal_transmittance must be between 0 and 1.")
    if fit_tolerance < 0.0:
        raise ValueError("fit_tolerance cannot be negative.")
    if layer_penalty < 0.0:
        raise ValueError("layer_penalty cannot be negative.")
    if not 0.0 < minimum_color_contribution < 1.0:
        raise ValueError("minimum_color_contribution must be between 0 and 1.")
    if anchor_weight < 0.0:
        raise ValueError("anchor_weight cannot be negative.")
    if max_layers_per_tier < 1:
        raise ValueError("max_layers_per_tier must be positive.")

    if step_height_mm <= 0.0 or first_layer_height_mm <= 0.0:
        raise ValueError("Layer heights must be positive.")

    ordered = tuple(sorted(filaments, key=lambda item: item.lab[0]))
    first_thicknesses = [step_height_mm] * len(ordered)
    if initial_substrate_lab is None:
        first_thicknesses[0] = first_layer_height_mm
    minimum_counts = [
        _minimum_layers_for_contribution(
            filament,
            first_layer_height_mm=first_thickness,
            step_height_mm=step_height_mm,
            minimum_color_contribution=minimum_color_contribution,
        )
        for filament, first_thickness in zip(ordered, first_thicknesses)
    ]
    if initial_substrate_lab is None:
        # The solid bed-contact color is already fully represented; there is no
        # underlying tier showing through it.
        minimum_counts[0] = 1
    if sum(minimum_counts) > max_layers_per_tier:
        raise ValueError(
            f"Tier budget of {max_layers_per_tier} layers cannot satisfy the "
            f"TD minimum of {sum(minimum_counts)} layers for {len(ordered)} colors."
        )
    maximum_counts = [
        _candidate_layer_limit(
            filament,
            step_height_mm,
            terminal_transmittance,
            max_layers_per_tier,
        )
        for filament in ordered
    ]
    anchors = dict(target_by_filament or {})
    layer_counts, states, mean_delta_e = _optimize_bounded_schedule(
        ordered,
        minimum_counts,
        maximum_counts,
        max_layers_per_tier,
        target_lab,
        anchors,
        step_height_mm=step_height_mm,
        first_layer_height_mm=first_layer_height_mm,
        initial_substrate_lab=initial_substrate_lab,
        anchor_weight=anchor_weight,
        layer_penalty=layer_penalty,
        fit_tolerance=fit_tolerance,
    )
    assignments = assignments_from_layer_counts(ordered, layer_counts)
    return OptimizedTierSchedule(
        assignments=tuple(assignments),
        states=tuple(states),
        layer_counts=tuple(layer_counts),
        mean_delta_e=round(mean_delta_e, 4),
    )


def assignments_from_layer_counts(
    filaments: Sequence[FilamentRecord],
    layer_counts: Sequence[int],
) -> list[FilamentLayerAssignment]:
    """Convert consecutive layer counts into filament start-layer assignments."""
    if len(filaments) != len(layer_counts):
        raise ValueError("filaments and layer_counts must have the same length.")
    if any(count < 1 for count in layer_counts):
        raise ValueError("Every selected filament must receive at least one layer.")

    assignments: list[FilamentLayerAssignment] = []
    start_layer = 0
    for filament, count in zip(filaments, layer_counts):
        assignments.append(FilamentLayerAssignment(filament, start_layer))
        start_layer += count
    return assignments


class ColorLayerMapper:
    """Map image Lab pixels to the nearest simulated layer using CIEDE2000."""

    def __init__(self, optical_states: Sequence[LayerOpticalState]) -> None:
        if not optical_states:
            raise ValueError("optical_states cannot be empty.")
        self._states = tuple(optical_states)
        self._layers = np.asarray([state.layer_index for state in optical_states], dtype=np.int32)
        self._heights = np.asarray([state.height_mm for state in optical_states], dtype=np.float32)
        self._lab = np.asarray([state.simulated_lab for state in optical_states], dtype=np.float64)

    @property
    def simulated_lab_curve(self) -> np.ndarray:
        return self._lab.copy()

    @property
    def layer_indices(self) -> np.ndarray:
        return self._layers.copy()

    @property
    def height_array(self) -> np.ndarray:
        return self._heights.copy()

    def map_image_lab_to_layers(self, target_lab_image: np.ndarray) -> np.ndarray:
        if target_lab_image.ndim != 3 or target_lab_image.shape[-1] != 3:
            raise ValueError("target_lab_image must have shape (height, width, 3).")
        flat_targets = np.asarray(target_lab_image, dtype=np.float64).reshape(-1, 3)
        best_indices = np.empty(len(flat_targets), dtype=np.int32)

        # Bound the temporary Delta-E matrix for large source images.
        chunk_size = 65_536
        for start in range(0, len(flat_targets), chunk_size):
            stop = min(start + chunk_size, len(flat_targets))
            distances = delta_e_2000_array(
                flat_targets[start:stop, None, :],
                self._lab[None, :, :],
            )
            best_indices[start:stop] = np.argmin(distances, axis=1)

        return self._layers[best_indices].reshape(target_lab_image.shape[:2])
