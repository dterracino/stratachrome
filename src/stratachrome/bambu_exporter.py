"""High-level Bambu Studio 3MF archive writer."""

from __future__ import annotations

import json
from pathlib import Path
import zipfile

from stratachrome.bambu_project import (
    build_content_types_xml,
    build_custom_gcode_xml,
    build_master_assembly_xml,
    build_model_rels_xml,
    build_model_settings_config,
    build_object_geometry_xml,
    build_project_settings_config,
    build_root_rels_xml,
    build_slice_info_config,
    ordered_filaments,
    plate_center_from_settings,
)
from stratachrome.depth_mapper import SwapEvent
from stratachrome.mesh_builder import TriangleMesh


def export_bambu_project(
    mesh: TriangleMesh,
    swap_schedule: list[SwapEvent],
    output_path: Path,
    swap_mode: str = "auto",
    *,
    step_height_mm: float = 0.10,
    first_layer_height_mm: float = 0.20,
) -> None:
    """Export a Bambu project with a base filament and later tool changes."""
    if swap_mode not in {"auto", "manual"}:
        raise ValueError("swap_mode must be 'auto' or 'manual'.")
    if step_height_mm <= 0.0 or first_layer_height_mm <= 0.0:
        raise ValueError("Layer heights must be positive.")

    filaments = ordered_filaments(swap_schedule)
    min_x = float(mesh.vertices[:, 0].min())
    max_x = float(mesh.vertices[:, 0].max())
    min_y = float(mesh.vertices[:, 1].min())
    max_y = float(mesh.vertices[:, 1].max())
    mesh_center_x = (min_x + max_x) / 2.0
    mesh_center_y = (min_y + max_y) / 2.0
    project_settings_config = build_project_settings_config(
        swap_schedule,
        step_height_mm=step_height_mm,
        first_layer_height_mm=first_layer_height_mm,
    )
    plate_center_x, plate_center_y = plate_center_from_settings(json.loads(project_settings_config))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", build_content_types_xml())
        archive.writestr("_rels/.rels", build_root_rels_xml())
        archive.writestr(
            "3D/3dmodel.model",
            build_master_assembly_xml(plate_center_x, plate_center_y),
        )
        archive.writestr("3D/_rels/3dmodel.model.rels", build_model_rels_xml())
        archive.writestr(
            "3D/Objects/object_1.model",
            build_object_geometry_xml(mesh, mesh_center_x, mesh_center_y),
        )
        archive.writestr(
            "Metadata/project_settings.config",
            project_settings_config,
        )
        archive.writestr(
            "Metadata/custom_gcode_per_layer.xml",
            build_custom_gcode_xml(swap_schedule, swap_mode),
        )
        archive.writestr(
            "Metadata/model_settings.config",
            build_model_settings_config(mesh, len(filaments), mesh_center_x, mesh_center_y),
        )
        archive.writestr("Metadata/slice_info.config", build_slice_info_config())
