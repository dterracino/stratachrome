"""Bambu Studio project metadata and 3MF assembly builders."""

from __future__ import annotations

from copy import deepcopy
from importlib.resources import files
import json
from typing import Any, Mapping, Sequence
import xml.etree.ElementTree as ET

from stratachrome.depth_mapper import SwapEvent
from stratachrome.mesh_builder import TriangleMesh


NS_3MF = "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"
NS_PRODUCTION = "http://schemas.microsoft.com/3dmanufacturing/production/2015/06"
NS_BAMBU = "http://schemas.bambulab.com/package/2021"
NS_RELS = "http://schemas.openxmlformats.org/package/2006/relationships"
NS_TYPES = "http://schemas.openxmlformats.org/package/2006/content-types"
NS_XML = "http://www.w3.org/XML/1998/namespace"
BAMBU_VERSION = "02.07.01.62"
OBJECT_PATH = "/3D/Objects/object_1.model"

ET.register_namespace("", NS_3MF)
ET.register_namespace("p", NS_PRODUCTION)
ET.register_namespace("BambuStudio", NS_BAMBU)


def serialize_xml(element: ET.Element) -> str:
    return ET.tostring(element, encoding="utf-8", xml_declaration=True).decode("utf-8")


def build_content_types_xml() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<Types xmlns="{NS_TYPES}">\n'
        ' <Default Extension="rels" '
        'ContentType="application/vnd.openxmlformats-package.relationships+xml"/>\n'
        ' <Default Extension="model" '
        'ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>\n'
        ' <Default Extension="gcode" ContentType="text/x.gcode"/>\n'
        '</Types>\n'
    )


def _build_relationships_xml(target: str) -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<Relationships xmlns="{NS_RELS}">\n'
        f' <Relationship Target="{target}" Id="rel-1" '
        'Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>\n'
        '</Relationships>\n'
    )


def build_root_rels_xml() -> str:
    return _build_relationships_xml("/3D/3dmodel.model")


def build_model_rels_xml() -> str:
    return _build_relationships_xml(OBJECT_PATH)


def _model_root() -> ET.Element:
    return ET.Element(
        f"{{{NS_3MF}}}model",
        {
            "unit": "millimeter",
            f"{{{NS_XML}}}lang": "en-US",
            "xmlns:BambuStudio": NS_BAMBU,
            "requiredextensions": "p",
        },
    )


def _add_metadata(parent: ET.Element, name: str, value: str) -> None:
    metadata = ET.SubElement(parent, f"{{{NS_3MF}}}metadata", name=name)
    metadata.text = value


def build_master_assembly_xml(center_x: float, center_y: float) -> str:
    root = _model_root()
    _add_metadata(root, "Application", f"BambuStudio-{BAMBU_VERSION}")
    _add_metadata(root, "BambuStudio:3mfVersion", "1")

    resources = ET.SubElement(root, f"{{{NS_3MF}}}resources")
    assembly = ET.SubElement(
        resources,
        f"{{{NS_3MF}}}object",
        {
            "id": "2",
            f"{{{NS_PRODUCTION}}}UUID": "00000001-61cb-4c03-9d28-80fed5dfa1dc",
            "type": "model",
        },
    )
    components = ET.SubElement(assembly, f"{{{NS_3MF}}}components")
    ET.SubElement(
        components,
        f"{{{NS_3MF}}}component",
        {
            f"{{{NS_PRODUCTION}}}path": OBJECT_PATH,
            "objectid": "1",
            f"{{{NS_PRODUCTION}}}UUID": "00010000-b206-40ff-9872-83e8017abed1",
            "transform": "1 0 0 0 1 0 0 0 1 0 0 0",
        },
    )

    build = ET.SubElement(
        root,
        f"{{{NS_3MF}}}build",
        {f"{{{NS_PRODUCTION}}}UUID": "2c7c17d8-22b5-4d84-8835-1976022ea369"},
    )
    ET.SubElement(
        build,
        f"{{{NS_3MF}}}item",
        {
            "objectid": "2",
            f"{{{NS_PRODUCTION}}}UUID": "00000002-b1ec-4553-aec9-835e5b724bb4",
            "transform": f"1 0 0 0 1 0 0 0 1 {center_x:.6f} {center_y:.6f} 0",
            "printable": "1",
        },
    )
    return serialize_xml(root)


def build_object_geometry_xml(mesh: TriangleMesh, center_x: float, center_y: float) -> str:
    root = _model_root()
    _add_metadata(root, "BambuStudio:3mfVersion", "1")
    resources = ET.SubElement(root, f"{{{NS_3MF}}}resources")
    object_element = ET.SubElement(
        resources,
        f"{{{NS_3MF}}}object",
        {
            "id": "1",
            f"{{{NS_PRODUCTION}}}UUID": "00010000-81cb-4c03-9d28-80fed5dfa1dc",
            "type": "model",
        },
    )
    mesh_element = ET.SubElement(object_element, f"{{{NS_3MF}}}mesh")
    vertices = ET.SubElement(mesh_element, f"{{{NS_3MF}}}vertices")
    for vx, vy, vz in mesh.vertices:
        ET.SubElement(
            vertices,
            f"{{{NS_3MF}}}vertex",
            x=f"{vx - center_x:.4f}",
            y=f"{vy - center_y:.4f}",
            z=f"{vz:.4f}",
        )

    triangles = ET.SubElement(mesh_element, f"{{{NS_3MF}}}triangles")
    for v1, v2, v3 in mesh.faces:
        ET.SubElement(
            triangles,
            f"{{{NS_3MF}}}triangle",
            v1=str(v1),
            v2=str(v2),
            v3=str(v3),
        )
    # The 3MF core schema requires every model part to contain both resources
    # and build. Referenced object parts have an empty build, as emitted by
    # Bambu Studio itself.
    ET.SubElement(root, f"{{{NS_3MF}}}build")
    return serialize_xml(root)


def filament_key(swap: SwapEvent) -> tuple[str, str]:
    return swap.filament_hex.upper(), swap.filament_name


def ordered_filaments(swap_schedule: Sequence[SwapEvent]) -> list[tuple[str, str]]:
    ordered: list[tuple[str, str]] = []
    for swap in swap_schedule:
        key = filament_key(swap)
        if key not in ordered:
            ordered.append(key)
    if not ordered:
        raise ValueError("The swap schedule must contain at least the base filament.")
    return ordered


def plate_center_from_settings(settings: Mapping[str, Any]) -> tuple[float, float]:
    """Return the center of the configured printable polygon."""
    raw_area = settings.get("printable_area")
    if not isinstance(raw_area, list) or not raw_area:
        return 128.0, 128.0

    points: list[tuple[float, float]] = []
    for raw_point in raw_area:
        if not isinstance(raw_point, str):
            return 128.0, 128.0
        try:
            raw_x, raw_y = raw_point.lower().split("x", maxsplit=1)
            points.append((float(raw_x), float(raw_y)))
        except (ValueError, TypeError):
            return 128.0, 128.0

    xs, ys = zip(*points)
    return (min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0


def load_project_settings() -> dict[str, Any]:
    """Load Stratachrome's packaged X1 Carbon 0.4 mm project defaults."""
    resource = (
        files("stratachrome")
        .joinpath("resources")
        .joinpath("bambu_x1c_0.4_project_settings.json")
    )
    try:
        loaded = json.loads(resource.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise RuntimeError("Packaged Bambu project settings are not valid JSON.") from error
    if not isinstance(loaded, dict):
        raise RuntimeError("Packaged Bambu project settings must be a JSON object.")
    # Process settings are embedded as ordinary project values, but the project
    # must not claim or select a named print preset in the user's Bambu Studio.
    loaded.pop("print_settings_id", None)
    inherits_group = loaded.get("inherits_group")
    if isinstance(inherits_group, list) and inherits_group:
        inherits_group[0] = ""
    return loaded


def _repeat_first_value(values: Sequence[Any], count: int) -> list[Any]:
    if not values:
        return [""] * count
    return [deepcopy(values[0]) for _ in range(count)]


def build_project_settings_config(
    swap_schedule: Sequence[SwapEvent],
    *,
    step_height_mm: float,
    first_layer_height_mm: float,
) -> str:
    settings = deepcopy(load_project_settings())
    filaments = ordered_filaments(swap_schedule)
    filament_count = len(filaments)
    source_count = len(settings.get("filament_colour", []))

    if source_count:
        for key, value in list(settings.items()):
            if not isinstance(value, list) or not value:
                continue
            per_filament = len(value) == source_count
            inherited_filament = key.startswith("filament_") and len(value) in {
                source_count,
                source_count + 1,
            }
            if per_filament or inherited_filament:
                settings[key] = _repeat_first_value(value, filament_count)

    colors = [color for color, _ in filaments]
    settings.update(
        {
            "version": str(settings.get("version", BAMBU_VERSION)),
            "layer_height": f"{step_height_mm:g}",
            "initial_layer_print_height": f"{first_layer_height_mm:g}",
            "filament_colour": colors,
            "filament_multi_colour": colors,
            "default_filament_colour": [""] * filament_count,
            "filament_type": ["PLA"] * filament_count,
            "filament_vendor": ["Bambu Lab"] * filament_count,
            "filament_settings_id": ["Bambu PLA Basic @BBL X1C"] * filament_count,
            "filament_ids": [""] * filament_count,
            "filament_is_support": ["0"] * filament_count,
            "filament_soluble": ["0"] * filament_count,
            "filament_map": ["1"] * filament_count,
            "filament_self_index": [str(index + 1) for index in range(filament_count)],
            "single_extruder_multi_material": "1",
        }
    )

    settings["flush_volumes_matrix"] = [
        "0" if source == target else "280"
        for source in range(filament_count)
        for target in range(filament_count)
    ]
    settings["flush_volumes_vector"] = ["140"] * (2 * filament_count)
    return json.dumps(settings, indent=4, ensure_ascii=False)


def build_custom_gcode_xml(
    swap_schedule: Sequence[SwapEvent],
    swap_mode: str,
) -> str:
    root = ET.Element("custom_gcodes_per_layer")
    plate = ET.SubElement(root, "plate")
    ET.SubElement(plate, "plate_info", id="1")

    filaments = ordered_filaments(swap_schedule)
    slot_by_filament = {key: index + 1 for index, key in enumerate(filaments)}
    active = filament_key(swap_schedule[0])
    for swap in swap_schedule[1:]:
        incoming = filament_key(swap)
        if incoming == active:
            continue
        ET.SubElement(
            plate,
            "layer",
            top_z=f"{swap.z_height_mm:.6f}",
            type="2",
            extruder=str(slot_by_filament[incoming]),
            color=incoming[0],
            extra="",
            gcode="tool_change" if swap_mode == "ams" else "pause_print",
        )
        active = incoming

    ET.SubElement(plate, "mode", value="MultiAsSingle")
    return serialize_xml(root)


def build_model_settings_config(
    mesh: TriangleMesh,
    filament_count: int,
    center_x: float,
    center_y: float,
) -> str:
    root = ET.Element("config")
    object_element = ET.SubElement(root, "object", id="2")
    ET.SubElement(object_element, "metadata", key="name", value="stratachrome_relief")
    ET.SubElement(object_element, "metadata", key="extruder", value="1")
    ET.SubElement(object_element, "metadata", face_count=str(mesh.face_count))
    part = ET.SubElement(object_element, "part", id="1", subtype="normal_part")
    part_metadata = {
        "name": "stratachrome_relief",
        "matrix": "1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1",
        "source_file": "stratachrome_relief",
        "source_object_id": "0",
        "source_volume_id": "0",
        "source_offset_x": f"{center_x:.6f}",
        "source_offset_y": f"{center_y:.6f}",
        "source_offset_z": "0",
    }
    for key, value in part_metadata.items():
        ET.SubElement(part, "metadata", key=key, value=value)
    ET.SubElement(
        part,
        "mesh_stat",
        face_count=str(mesh.face_count),
        edges_fixed="0",
        degenerate_facets="0",
        facets_removed="0",
        facets_reversed="0",
        backwards_edges="0",
    )

    plate = ET.SubElement(root, "plate")
    for key, value in (
        ("plater_id", "1"),
        ("plater_name", ""),
        ("locked", "false"),
        ("filament_map_mode", "Auto For Flush"),
        ("filament_maps", " ".join("1" for _ in range(filament_count))),
        ("filament_volume_maps", " ".join("0" for _ in range(filament_count))),
    ):
        ET.SubElement(plate, "metadata", key=key, value=value)
    instance = ET.SubElement(plate, "model_instance")
    ET.SubElement(instance, "metadata", key="object_id", value="2")
    ET.SubElement(instance, "metadata", key="instance_id", value="0")
    ET.SubElement(instance, "metadata", key="identify_id", value="1")

    assemble = ET.SubElement(root, "assemble")
    ET.SubElement(
        assemble,
        "assemble_item",
        object_id="2",
        instance_id="0",
        transform="1 0 0 0 1 0 0 0 1 0 0 0",
        offset="0 0 0",
    )
    return serialize_xml(root)


def build_slice_info_config() -> str:
    root = ET.Element("config")
    header = ET.SubElement(root, "header")
    ET.SubElement(header, "header_item", key="X-BBL-Client-Type", value="slicer")
    ET.SubElement(header, "header_item", key="X-BBL-Client-Version", value=BAMBU_VERSION)
    return serialize_xml(root)
