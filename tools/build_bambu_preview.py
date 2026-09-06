"""Build a Bambu-validated preview project from generated core parts."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import zipfile

from PIL import Image, ImageOps


def _thumbnail(source: Image.Image, size: int) -> bytes:
    contained = ImageOps.contain(source, (int(size * 0.82), int(size * 0.82)))
    canvas = Image.new("RGBA", (size, size), (30, 30, 30, 255))
    canvas.alpha_composite(contained, ((size - contained.width) // 2, (size - contained.height) // 2))
    output = io.BytesIO()
    canvas.save(output, "PNG")
    return output.getvalue()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--generated", type=Path, required=True)
    parser.add_argument("--source-image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source = Image.open(args.source_image).convert("RGBA")
    core_parts = {
        "3D/3dmodel.model",
        "3D/Objects/object_1.model",
        "Metadata/project_settings.config",
        "Metadata/model_settings.config",
        "Metadata/custom_gcode_per_layer.xml",
        "Metadata/slice_info.config",
    }
    png_sizes = {
        "Metadata/plate_1.png": 512,
        "Metadata/plate_1_small.png": 128,
        "Metadata/plate_no_light_1.png": 512,
        "Metadata/top_1.png": 512,
        "Metadata/pick_1.png": 512,
    }
    plate = {
        "bbox_all": [48, 28, 208, 228],
        "bbox_objects": [
            {
                "area": 32000,
                "bbox": [48, 28, 208, 228],
                "id": 1,
                "layer_height": 0.2,
                "name": "stratachrome_relief",
            }
        ],
        "bed_type": "supertack_plate",
        "filament_colors": [],
        "filament_ids": [],
        "first_extruder": 0,
        "is_seq_print": False,
        "nozzle_diameter": 0.4,
        "version": 2,
    }
    ranges = (
        '<?xml version="1.0" encoding="utf-8"?>'
        '<objects><object id="1"><range min_z="0" max_z="0.2">'
        '<option opt_key="extruder">0</option>'
        '<option opt_key="layer_height">0.2</option>'
        '</range></object></objects>'
    ).encode()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with (
        zipfile.ZipFile(args.template) as template,
        zipfile.ZipFile(args.generated) as generated,
        zipfile.ZipFile(args.output, "w", zipfile.ZIP_DEFLATED) as output,
    ):
        for name in template.namelist():
            data = generated.read(name) if name in core_parts else template.read(name)
            if name in png_sizes:
                data = _thumbnail(source, png_sizes[name])
            elif name == "Metadata/plate_1.json":
                data = json.dumps(plate, separators=(",", ":")).encode()
            elif name == "Metadata/layer_config_ranges.xml":
                data = ranges
            elif name == "Metadata/brim_ear_points.txt":
                data = b"brim_points_format_version=1\n"
            output.writestr(name, data)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
