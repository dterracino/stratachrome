"""
export_3mf.py
-------------
Packages a TriangleMesh and slicer pause schedule into a Bambu Studio / Orca
Slicer compatible .3mf project archive using Open Packaging Conventions (OPC).
Decomposes geometry into 3D/Objects/object_1.model and writes layer-by-layer
pause markers into Metadata.
"""

from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile

from stratachrome.depth_mapper import SwapEvent
from stratachrome.mesh_builder import TriangleMesh


_NS_3MF = "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"
_NS_RELS = "http://schemas.openxmlformats.org/package/2006/relationships"
_NS_TYPES = "http://schemas.openxmlformats.org/package/2006/content-types"


def _build_content_types_xml() -> str:
    """Generates standard [Content_Types].xml defining OPC MIME mappings."""
    types_elem = ET.Element("Types", xmlns=_NS_TYPES)
    
    ET.SubElement(types_elem, "Default", Extension="rels", ContentType="application/vnd.openxmlformats-package.relationships+xml")
    ET.SubElement(types_elem, "Default", Extension="model", ContentType="application/vnd.ms-package.3dmanufacturing-3dmodelxml")
    ET.SubElement(types_elem, "Default", Extension="config", ContentType="text/plain")
    ET.SubElement(types_elem, "Default", Extension="xml", ContentType="application/xml")
    ET.SubElement(types_elem, "Default", Extension="png", ContentType="image/png")

    return ET.tostring(types_elem, encoding="utf-8", xml_declaration=True).decode("utf-8")


def _build_root_rels_xml() -> str:
    """Generates _rels/.rels linking the root package to the primary 3D model."""
    rels_elem = ET.Element("Relationships", xmlns=_NS_RELS)
    
    ET.SubElement(
        rels_elem,
        "Relationship",
        Target="/3D/3dmodel.model",
        Id="rel-1",
        Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel",
    )
    return ET.tostring(rels_elem, encoding="utf-8", xml_declaration=True).decode("utf-8")


def _build_model_rels_xml() -> str:
    """Generates 3D/_rels/3dmodel.model.rels linking assembly to object geometry."""
    rels_elem = ET.Element("Relationships", xmlns=_NS_RELS)
    
    ET.SubElement(
        rels_elem,
        "Relationship",
        Target="/3D/Objects/object_1.model",
        Id="rel-1",
        Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel",
    )
    return ET.tostring(rels_elem, encoding="utf-8", xml_declaration=True).decode("utf-8")


def _build_master_assembly_xml(object_id: int = 1) -> str:
    """Generates 3D/3dmodel.model referencing the external object component."""
    model_elem = ET.Element("model", unit="millimeter", xml_lang="en-US", xmlns=_NS_3MF)
    
    metadata_author = ET.SubElement(model_elem, "metadata", name="Application")
    metadata_author.text = "Stratachrome"

    resources = ET.SubElement(model_elem, "resources")
    build = ET.SubElement(model_elem, "build")
    
    ET.SubElement(
        build,
        "item",
        objectid=str(object_id),
        transform="1 0 0 0 1 0 0 0 1 0 0 0",
    )

    return ET.tostring(model_elem, encoding="utf-8", xml_declaration=True).decode("utf-8")


def _build_object_geometry_xml(mesh: TriangleMesh, object_id: int = 1) -> str:
    """Generates 3D/Objects/object_1.model containing full mesh vertices and faces."""
    model_elem = ET.Element("model", unit="millimeter", xml_lang="en-US", xmlns=_NS_3MF)
    resources = ET.SubElement(model_elem, "resources")
    
    object_elem = ET.SubElement(resources, "object", id=str(object_id), type="model")
    mesh_elem = ET.SubElement(object_elem, "mesh")
    
    # Vertices
    vertices_elem = ET.SubElement(mesh_elem, "vertices")
    for vx, vy, vz in mesh.vertices:
        ET.SubElement(
            vertices_elem,
            "vertex",
            x=f"{vx:.4f}",
            y=f"{vy:.4f}",
            z=f"{vz:.4f}",
        )

    # Triangles with strict CCW winding
    triangles_elem = ET.SubElement(mesh_elem, "triangles")
    for v1, v2, v3 in mesh.faces:
        ET.SubElement(
            triangles_elem,
            "triangle",
            v1=str(v1),
            v2=str(v2),
            v3=str(v3),
        )

    return ET.tostring(model_elem, encoding="utf-8", xml_declaration=True).decode("utf-8")


def _build_custom_gcode_xml(swap_schedule: list[SwapEvent]) -> str:
    """Generates Metadata/custom_gcode_per_layer.xml with slicer pause markers."""
    root = ET.Element("layers")
    
    for swap in swap_schedule:
        if swap.global_layer_idx == 0:
            continue

        layer_elem = ET.SubElement(
            root,
            "layer",
            height=f"{swap.z_height_mm:.3f}",
            type="2",  # Bambu/Orca designation for pause event
        )
        layer_elem.text = f"; Filament swap: {swap.filament_name} ({swap.filament_hex})"

    return ET.tostring(root, encoding="utf-8", xml_declaration=True).decode("utf-8")


def _build_model_settings_config(mesh: TriangleMesh, object_id: int = 1) -> str:
    """Generates minimal Metadata/model_settings.config for Bambu plate placement."""
    min_x = float(mesh.vertices[:, 0].min())
    max_x = float(mesh.vertices[:, 0].max())
    min_y = float(mesh.vertices[:, 1].min())
    max_y = float(mesh.vertices[:, 1].max())
    min_z = float(mesh.vertices[:, 2].min())
    max_z = float(mesh.vertices[:, 2].max())

    center_x = (min_x + max_x) / 2.0
    center_y = (min_y + max_y) / 2.0
    center_z = (min_z + max_z) / 2.0

    lines = [
        "; model_settings.config generated by Stratachrome",
        f"object_id={object_id}",
        "name=stratachrome_relief",
        f"bbox_min={min_x:.4f},{min_y:.4f},{min_z:.4f}",
        f"bbox_max={max_x:.4f},{max_y:.4f},{max_z:.4f}",
        f"center={center_x:.4f},{center_y:.4f},{center_z:.4f}",
        "plate_id=1",
    ]
    return "\n".join(lines)


def export_bambu_3mf(
    mesh: TriangleMesh,
    swap_schedule: list[SwapEvent],
    output_path: Path,
) -> None:
    """Exports a TriangleMesh and swap schedule into a Bambu/Orca compatible 3MF."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", _build_content_types_xml())
        zf.writestr("_rels/.rels", _build_root_rels_xml())
        zf.writestr("3D/3dmodel.model", _build_master_assembly_xml())
        zf.writestr("3D/_rels/3dmodel.model.rels", _build_model_rels_xml())
        zf.writestr("3D/Objects/object_1.model", _build_object_geometry_xml(mesh))
        zf.writestr("Metadata/custom_gcode_per_layer.xml", _build_custom_gcode_xml(swap_schedule))
        zf.writestr("Metadata/model_settings.config", _build_model_settings_config(mesh))