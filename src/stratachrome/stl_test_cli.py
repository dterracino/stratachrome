"""
stl_test_cli.py
---------------
CLI verification script to generate and export a 3D manifold STL heightmap
from a single image to verify mesh topology and slicer compatibility.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
from PIL import Image

from stratachrome.mesh_builder import PhysicalDimensions, WatertightMeshBuilder, export_binary_stl


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stratachrome: Convert an image to a manifold binary STL heightmap for verification."
    )
    parser.add_argument("-i", "--input", type=Path, required=True, help="Input image file path.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("output/test_model.stl"),
        help="Output STL path (default: output/test_model.stl).",
    )
    parser.add_argument(
        "-s",
        "--size",
        type=float,
        default=100.0,
        help="Target physical size in mm for the maximum image dimension (default: 100.0).",
    )
    parser.add_argument("--max-height", type=float, default=2.4, help="Maximum Z height in mm (default: 2.4).")
    parser.add_argument("--base-height", type=float, default=0.4, help="Solid base height in mm (default: 0.4).")
    parser.add_argument("--max-dimension", type=int, default=400, help="Max raster dimension (default: 400).")
    return parser.parse_args()


def _load_normalized_lightness(path: Path, max_dim: int) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Input file not found: {path}")

    with Image.open(path) as img:
        img = img.convert("L")
        w, h = img.size
        if max(w, h) > max_dim:
            scale = max_dim / max(w, h)
            img = img.resize((int(round(w * scale)), int(round(h * scale))), Image.Resampling.BILINEAR)
        arr = np.array(img, dtype=np.float32)
        return arr / 255.0


def main() -> int:
    args = _parse_arguments()

    try:
        norm_l = _load_normalized_lightness(args.input, args.max_dimension)
    except Exception as err:
        sys.stderr.write(f"Error reading image: {err}\n")
        return 1

    rows, cols = norm_l.shape
    if cols >= rows:
        width_mm = args.size
        height_mm = round(args.size * (rows / cols), 2)
    else:
        height_mm = args.size
        width_mm = round(args.size * (cols / rows), 2)

    print(f"Loaded grid: {cols}x{rows} points -> {width_mm:.1f}mm x {height_mm:.1f}mm")

    z_span = args.max_height - args.base_height
    z_grid = args.base_height + norm_l * z_span

    dims = PhysicalDimensions(
        width_mm=width_mm,
        height_mm=height_mm,
        base_floor_z_mm=0.0,
    )

    print("Building watertight manifold mesh...")
    builder = WatertightMeshBuilder(dims)
    mesh = builder.build_mesh(
        z_grid,
        progress=lambda message: print(f"  {message}", flush=True),
    )
    print(f"Generated {mesh.vertex_count:,} vertices and {mesh.face_count:,} triangles.")

    print(f"Exporting binary STL to {args.output}...")
    export_binary_stl(mesh, args.output)
    print("Done! You can load this STL into Bambu Studio or Orca Slicer to inspect the geometry.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
