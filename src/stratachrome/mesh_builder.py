"""
mesh_builder.py
---------------
Converts a 2D heightmap Z-grid into a watertight, 2-manifold triangular mesh
with a solid flat base, perimeter skirt, and strict counter-clockwise (CCW)
winding. Automatically orients image coordinates to face the front of the build plate.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np


@dataclass(frozen=True)
class PhysicalDimensions:
    """Defines physical print bounding dimensions in millimeters.

    Attributes:
        width_mm: Total print width along the X axis.
        height_mm: Total print depth along the Y axis.
        base_floor_z_mm: Absolute Z elevation of the bottom baseplate floor.
    """
    width_mm: float
    height_mm: float
    base_floor_z_mm: float = 0.0

    def __post_init__(self) -> None:
        if self.width_mm <= 0.0:
            raise ValueError(f"width_mm must be positive, got {self.width_mm}")
        if self.height_mm <= 0.0:
            raise ValueError(f"height_mm must be positive, got {self.height_mm}")
        if self.base_floor_z_mm < 0.0:
            raise ValueError(f"base_floor_z_mm cannot be negative, got {self.base_floor_z_mm}")


@dataclass(frozen=True)
class TriangleMesh:
    """Container for 3D triangle mesh geometry data.

    Attributes:
        vertices: (N, 3) float32 array of vertex [X, Y, Z] coordinates in mm.
        faces: (M, 3) int32 array of vertex indices with CCW winding order.
    """
    vertices: np.ndarray
    faces: np.ndarray

    @property
    def vertex_count(self) -> int:
        return int(self.vertices.shape[0])

    @property
    def face_count(self) -> int:
        return int(self.faces.shape[0])


class WatertightMeshBuilder:
    """Constructs 2-manifold closed meshes from 2D heightmap grids."""

    def __init__(self, dimensions: PhysicalDimensions) -> None:
        self._dims = dimensions

    def build_mesh(self, z_grid: np.ndarray) -> TriangleMesh:
        """Transforms a 2D Z-grid into a watertight triangular mesh.

        Args:
            z_grid: 2D float32 array of surface elevations in millimeters.

        Returns:
            TriangleMesh containing manifold vertices and CCW faces.
        """
        if z_grid.ndim != 2:
            raise ValueError(f"z_grid must be a 2D array, got ndim={z_grid.ndim}")

        # Flip vertically to align image top-edge (row 0) with the front of the build plate (Y=0)
        corrected_grid = np.flipud(z_grid)

        rows, cols = corrected_grid.shape
        dx = self._dims.width_mm / (cols - 1)
        dy = self._dims.height_mm / (rows - 1)
        floor_z = self._dims.base_floor_z_mm

        # 1. Generate grid coordinates (X, Y)
        x_coords = np.linspace(0.0, self._dims.width_mm, cols, dtype=np.float32)
        y_coords = np.linspace(0.0, self._dims.height_mm, rows, dtype=np.float32)
        xx, yy = np.meshgrid(x_coords, y_coords)

        # Flatten surface vertices
        top_verts = np.stack(
            (xx.ravel(), yy.ravel(), corrected_grid.ravel()),
            axis=-1
        ).astype(np.float32)

        # Flatten floor vertices at base_floor_z_mm
        floor_verts = np.stack(
            (xx.ravel(), yy.ravel(), np.full(rows * cols, floor_z, dtype=np.float32)),
            axis=-1
        ).astype(np.float32)

        # Combine vertices: [Top Surface (0 to N-1), Floor (N to 2N-1)]
        vertices = np.vstack((top_verts, floor_verts))
        num_top_verts = rows * cols

        faces_list: list[list[int]] = []

        # 2. Build top surface triangles (CCW winding)
        for r in range(rows - 1):
            for c in range(cols - 1):
                i0 = r * cols + c
                i1 = r * cols + (c + 1)
                i2 = (r + 1) * cols + c
                i3 = (r + 1) * cols + (c + 1)

                # Two triangles per grid cell
                faces_list.append([i0, i2, i1])
                faces_list.append([i1, i2, i3])

        # 3. Build bottom floor triangles (Inverted winding for downward-facing normals)
        for r in range(rows - 1):
            for c in range(cols - 1):
                f0 = num_top_verts + (r * cols + c)
                f1 = num_top_verts + (r * cols + (c + 1))
                f2 = num_top_verts + ((r + 1) * cols + c)
                f3 = num_top_verts + ((r + 1) * cols + (c + 1))

                faces_list.append([f0, f1, f2])
                faces_list.append([f1, f3, f2])

        # 4. Build perimeter skirt connecting top surface edges to bottom floor
        # Top/Bottom edges (along columns)
        for c in range(cols - 1):
            # North edge (row 0)
            t0 = c
            t1 = c + 1
            b0 = num_top_verts + c
            b1 = num_top_verts + (c + 1)
            faces_list.append([t0, b0, t1])
            faces_list.append([t1, b0, b1])

            # South edge (row rows-1)
            r_idx = rows - 1
            t2 = r_idx * cols + c
            t3 = r_idx * cols + (c + 1)
            b2 = num_top_verts + (r_idx * cols + c)
            b3 = num_top_verts + (r_idx * cols + (c + 1))
            faces_list.append([t3, b3, t2])
            faces_list.append([t2, b3, b2])

        # Left/Right edges (along rows)
        for r in range(rows - 1):
            # West edge (col 0)
            t0 = r * cols
            t1 = (r + 1) * cols
            b0 = num_top_verts + (r * cols)
            b1 = num_top_verts + ((r + 1) * cols)
            faces_list.append([t1, b1, t0])
            faces_list.append([t0, b1, b0])

            # East edge (col cols-1)
            c_idx = cols - 1
            t2 = r * cols + c_idx
            t3 = (r + 1) * cols + c_idx
            b2 = num_top_verts + (r * cols + c_idx)
            b3 = num_top_verts + ((r + 1) * cols + c_idx)
            faces_list.append([t2, b2, t3])
            faces_list.append([t3, b2, b3])

        faces = np.array(faces_list, dtype=np.int32)
        return TriangleMesh(vertices=vertices, faces=faces)


def export_binary_stl(mesh: TriangleMesh, output_path: Path) -> None:
    """Writes a TriangleMesh to disk as a binary STL file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "wb") as f:
        # 80-byte header
        header = b"Stratachrome Watertight Manifold Exporter"
        header = header + b"\0" * (80 - len(header))
        f.write(header)

        # 4-byte triangle count
        f.write(np.uint32(mesh.face_count).tobytes())

        # Precompute face normals and pack binary triangles
        v = mesh.vertices
        faces = mesh.faces

        v0 = v[faces[:, 0]]
        v1 = v[faces[:, 1]]
        v2 = v[faces[:, 2]]

        edge1 = v1 - v0
        edge2 = v2 - v0
        normals = np.cross(edge1, edge2)
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        normals = normals / norms

        for idx in range(mesh.face_count):
            n = normals[idx].astype(np.float32)
            p0 = v0[idx].astype(np.float32)
            p1 = v1[idx].astype(np.float32)
            p2 = v2[idx].astype(np.float32)

            f.write(n.tobytes())
            f.write(p0.tobytes())
            f.write(p1.tobytes())
            f.write(p2.tobytes())
            f.write(b"\0\0")  # 2-byte attribute byte count