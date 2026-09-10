"""Structural tests for Bambu Studio project export."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
from typing import TypeVar
import unittest
import xml.etree.ElementTree as ET
import zipfile

import numpy as np

from stratachrome.bambu_exporter import export_bambu_project
from stratachrome.bambu_project import NS_3MF, NS_BAMBU, NS_PRODUCTION, NS_RELS
from stratachrome.depth_mapper import SwapEvent
from stratachrome.mesh_builder import TriangleMesh

_T = TypeVar("_T")


def _required(value: _T | None, description: str) -> _T:
    if value is None:
        raise AssertionError(f"Expected {description} to be present.")
    return value


class BambuExporterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.mesh = TriangleMesh(
            vertices=np.asarray(
                [[0, 0, 0], [10, 0, 0], [0, 20, 0], [0, 0, 1]],
                dtype=np.float32,
            ),
            faces=np.asarray(
                [[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]],
                dtype=np.int32,
            ),
        )
        self.swaps = [
            SwapEvent(0, 0.2, "Bambu Lab Black", "#000000", "background"),
            SwapEvent(3, 0.4, "Bambu Lab Red", "#FF0000", "background"),
            SwapEvent(5, 0.6, "Bambu Lab White", "#FFFFFF", "foreground"),
            SwapEvent(7, 0.8, "Bambu Lab Red", "#FF0000", "foreground"),
        ]

    def test_export_has_valid_assembly_and_bambu_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "project.3mf"
            export_bambu_project(self.mesh, self.swaps, output)

            with zipfile.ZipFile(output) as archive:
                self.assertIsNone(archive.testzip())
                content_types = archive.read("[Content_Types].xml").decode("utf-8")
                self.assertIn("3dmanufacturing-3dmodel+xml", content_types)
                self.assertNotIn("image/png", content_types)

                image_members = [
                    name
                    for name in archive.namelist()
                    if name.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
                ]
                self.assertEqual(image_members, [])

                for relationships_path in (
                    "_rels/.rels",
                    "3D/_rels/3dmodel.model.rels",
                ):
                    relationships_xml = archive.read(relationships_path).decode("utf-8")
                    self.assertIn(f'<Relationships xmlns="{NS_RELS}">', relationships_xml)
                    self.assertNotIn("ns0:", relationships_xml)
                    relationships = ET.fromstring(relationships_xml)
                    for relationship in relationships:
                        target = _required(
                            relationship.get("Target"),
                            "relationship target",
                        )
                        self.assertIn(target.lstrip("/"), archive.namelist())

                master = ET.fromstring(archive.read("3D/3dmodel.model"))
                namespace = {"m": NS_3MF, "p": NS_PRODUCTION}
                resources = master.findall("./m:resources/m:object", namespace)
                self.assertEqual([item.get("id") for item in resources], ["2"])
                build_item = _required(
                    master.find("./m:build/m:item", namespace),
                    "master build item",
                )
                self.assertEqual(build_item.get("objectid"), "2")
                transform_value = _required(
                    build_item.get("transform"),
                    "master build transform",
                )
                transform = [float(value) for value in transform_value.split()]
                self.assertEqual(transform[9:11], [128.0, 128.0])
                component = _required(
                    master.find(".//m:component", namespace),
                    "master component",
                )
                self.assertEqual(component.get("objectid"), "1")
                self.assertEqual(
                    component.get(f"{{{NS_PRODUCTION}}}path"), "/3D/Objects/object_1.model"
                )
                self.assertIn(
                    f'xmlns:BambuStudio="{NS_BAMBU}"',
                    archive.read("3D/3dmodel.model").decode("utf-8"),
                )

                object_model = ET.fromstring(archive.read("3D/Objects/object_1.model"))
                object_build = _required(
                    object_model.find("./m:build", namespace),
                    "object build",
                )
                self.assertEqual(len(object_build), 0)

                settings = json.loads(archive.read("Metadata/project_settings.config"))
                self.assertGreater(len(settings), 100)
                self.assertEqual(settings["printer_model"], "Bambu Lab X1 Carbon")
                self.assertEqual(
                    settings["printer_settings_id"],
                    "Bambu Lab X1 Carbon 0.4 nozzle",
                )
                self.assertEqual(settings["filament_colour"], ["#000000", "#FF0000", "#FFFFFF"])
                for key in (
                    "filament_density",
                    "filament_flow_ratio",
                    "filament_max_volumetric_speed",
                    "filament_retraction_length",
                    "filament_start_gcode",
                ):
                    self.assertEqual(settings[key], [settings[key][0]] * 3)
                self.assertEqual(settings["layer_height"], "0.1")
                self.assertNotIn("print_settings_id", settings)
                self.assertEqual(
                    settings["default_print_profile"],
                    "0.20mm Standard @BBL X1C",
                )
                self.assertEqual(
                    settings["inherits_group"][0],
                    "",
                )

                model_settings = ET.fromstring(archive.read("Metadata/model_settings.config"))
                extruder = _required(
                    model_settings.find("./object/metadata[@key='extruder']"),
                    "extruder metadata",
                )
                self.assertEqual(extruder.get("value"), "1")

    def test_first_color_is_model_color_and_later_transitions_are_events(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "project.3mf"
            export_bambu_project(self.mesh, self.swaps, output)
            with zipfile.ZipFile(output) as archive:
                custom = ET.fromstring(archive.read("Metadata/custom_gcode_per_layer.xml"))

            layers = custom.findall("./plate/layer")
            self.assertEqual(len(layers), 3)
            self.assertEqual([layer.get("extruder") for layer in layers], ["2", "3", "2"])
            self.assertEqual(
                [layer.get("top_z") for layer in layers], ["0.400000", "0.600000", "0.800000"]
            )
            self.assertTrue(all(layer.get("gcode") == "tool_change" for layer in layers))

    def test_export_preserves_basic_and_matte_filament_profiles(self) -> None:
        swaps = [
            SwapEvent(
                0,
                0.2,
                "Bambu Lab PLA Basic Black",
                "#000000",
                "single",
                filament_finish="Basic",
                filament_id="bambu-lab-pla-basic-black",
            ),
            SwapEvent(
                3,
                0.4,
                "Bambu Lab PLA Matte Caramel",
                "#AE835B",
                "single",
                filament_finish="Matte",
                filament_id="bambu-lab-pla-matte-caramel",
            ),
        ]

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "profiles.3mf"
            export_bambu_project(self.mesh, swaps, output)
            with zipfile.ZipFile(output) as archive:
                settings = json.loads(archive.read("Metadata/project_settings.config"))

        self.assertEqual(
            settings["filament_settings_id"],
            ["Bambu PLA Basic @BBL X1C", "Bambu PLA Matte @BBL X1C"],
        )
        self.assertEqual(settings["filament_ids"], ["GFA00", "GFA01"])
        self.assertEqual(settings["filament_density"], ["1.26", "1.32"])
        self.assertEqual(settings["filament_flow_ratio"], ["0.98", "0.98"])
        self.assertEqual(settings["filament_max_volumetric_speed"], ["21", "22"])


if __name__ == "__main__":
    unittest.main()
