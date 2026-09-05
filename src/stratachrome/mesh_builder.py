"""
mesh_builder.py
---------------
Transforms a 2D height array into a watertight, 2-manifold triangle mesh.
Generates top relief geometry, vertical boundary walls (skirt), and a flat
bottom base plate with strict counter-clockwise (CCW) vertex winding.
Includes high-performance binary STL file export.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import struct

import numpy as np


@dataclass(frozen=True)
class TriangleMesh:
    """Represents an indexed 3D surface mesh.

    Attributes:
        vertices: Float32 array of shape (N, 3) storing (X, Y, Z) coordinates.
        faces: Int32 array of shape (M, 3) storing vertex indices in CCW order.
    """
    vertices: np.ndarray
    faces: np.ndarray

    def __post_init__(self) -> None:
        if self.vertices.ndim != 2 or self.vertices.shape[1] != 3:
            raise ValueError(f"vertices must have shape (N, 3), got {self.vertices.shape}")
        if self.faces.ndim != 2 or self.faces.shape[1] != 3:
            raise ValueError(f"faces must have shape (M, 3), got {self.faces.shape}")

    @property
    def vertex_count(self) -> int:
        return self.vertices.shape[0]

    @property
    def face_count(self) -> int:
        return self.faces.shape[0]


@dataclass(frozen=True)
class PhysicalDimensions:
    """Defines physical print plate dimensions in millimeters.

    Attributes:
        width_mm: Physical size along the X-axis.
        height_mm: Physical size along the Y-axis.
        base_floor_z_mm: Base elevation for the bottom plate (typically 0.0mm).
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


def _build_xy_coordinates(rows: int, cols: int, width_mm: float, height_mm: float) -> tuple[np.ndarray, np.ndarray]:
    """Computes physical X and Y grid coordinates mapped to millimeter dimensions."""
    x_coords = np.linspace(0.0, width_mm, cols, dtype=np.float32)
    y_coords = np.linspace(0.0, height_mm, rows, dtype=np.float32)
    return np.meshgrid(x_coords, y_coords)


def _build_top_surface_faces(rows: int, cols: int) -> np.ndarray:
    """Generates two CCW triangles for every grid cell on the top surface."""
    r_idx = np.arange(rows - 1, dtype=np.int32)[:, None]
    c_idx = np.arange(cols - 1, dtype=np.int32)[None, :]

    v00 = r_idx * cols + c_idx
    v01 = v00 + 1
    v10 = (r_idx + 1) * cols + c_idx
    v11 = v10 + 1

    t1 = np.stack([v00, v10, v01], axis=-1).reshape(-1, 3)
    t2 = np.stack([v01, v10, v11], axis=-1).reshape(-1, 3)
    return np.vstack([t1, t2])


def _build_bottom_surface_faces(rows: int, cols: int, offset: int) -> np.ndarray:
    """Generates two CCW triangles for every cell on the bottom plate facing -Z."""
    r_idx = np.arange(rows - 1, dtype=np.int32)[:, None]
    c_idx = np.arange(cols - 1, dtype=np.int32)[None, :]

    b00 = offset + (r_idx * cols + c_idx)
    b01 = b00 + 1
    b10 = offset + ((r_idx + 1) * cols + c_idx)
    b11 = b10 + 1

    t1 = np.stack([b00, b01, b10], axis=-1).reshape(-1, 3)
    t2 = np.stack([b01, b11, b10], axis=-1).reshape(-1, 3)
    return np.vstack([t1, t2])


def _build_boundary_quad(
    t_curr: np.ndarray,
    t_next: np.ndarray,
    b_curr: np.ndarray,
    b_next: np.ndarray,
) -> np.ndarray:
    """Constructs two CCW triangles for a vertical perimeter quad wall."""
    t1 = np.stack([t_curr, b_curr, t_next], axis=-1)
    t2 = np.stack([t_next, b_curr, b_next], axis=-1)
    return np.vstack([t1, t2])


def _build_skirt_wall_faces(rows: int, cols: int, offset: int) -> np.ndarray:
    """Generates outward-facing CCW triangles connecting top and bottom perimeter edges."""
    wall_faces: list[np.ndarray] = []

    # 1. North Edge (row = 0, increasing cols: c -> c+1)
    c_north = np.arange(cols - 1, dtype=np.int32)
    t_curr = c_north
    t_next = c_north + 1
    b_curr = offset + t_curr
    b_next = offset + t_next
    wall_faces.append(_build_boundary_quad(t_curr, t_next, b_curr, b_next))

    # 2. East Edge (col = cols - 1, increasing rows: r -> r+1)
    r_east = np.arange(rows - 1, dtype=np.int32)
    t_curr = r_east * cols + (cols - 1)
    t_next = (r_east + 1) * cols + (cols - 1)
    b_curr = offset + t_curr
    b_next = offset + t_next
    wall_faces.append(_build_boundary_quad(t_curr, t_next, b_curr, b_next))

    # 3. South Edge (row = rows - 1, decreasing cols: c+1 -> c)
    c_south = np.arange(cols - 2, -1, -1, dtype=np.int32)
    t_curr = (rows - 1) * cols + (c_south + 1)
    t_next = (rows - 1) * cols + c_south
    b_curr = offset + t_curr
    b_next = offset + t_next
    wall_faces.append(_build_boundary_quad(t_curr, t_next, b_curr, b_next))

    # 4. West Edge (col = 0, decreasing rows: r+1 -> r)
    r_west = np.arange(rows - 2, -1, -1, dtype=np.int32)
    t_curr = (r_west + 1) * cols
    t_next = r_west * cols
    b_curr = offset + t_curr
    b_next = offset + t_next
    wall_faces.append(_build_boundary_quad(t_curr, t_next, b_curr, b_next))

    return np.vstack(wall_faces)


def _validate_manifold_structure(mesh: TriangleMesh) -> None:
    """Verifies that the generated mesh contains no degenerate faces."""
    f = mesh.faces
    has_degenerate = np.any((f[:, 0] == f[:, 1]) | (f[:, 1] == f[:, 2]) | (f[:, 2] == f[:, 0]))
    if has_degenerate:
        raise ValueError("Degenerate triangle detected (face references same vertex multiple times).")


def _compute_triangle_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Computes outward face normal vectors using CCW cross products."""
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    norm = np.linalg.norm(cross, axis=1, keepdims=True)
    norm = np.where(norm == 0.0, 1.0, norm)
    return (cross / norm).astype(np.float32)


def export_binary_stl(mesh: TriangleMesh, output_path: Path) -> None:
    """Exports a TriangleMesh directly to a standard binary STL file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    num_triangles = mesh.face_count
    normals = _compute_triangle_normals(mesh.vertices, mesh.faces)

    tri_v0 = mesh.vertices[mesh.faces[:, 0]]
    tri_v1 = mesh.vertices[mesh.faces[:, 1]]
    tri_v2 = mesh.vertices[mesh.faces[:, 2]]

    # 50 bytes per triangle: normal (3f4), v0 (3f4), v1 (3f4), v2 (3f4), attribute byte count (u2)
    record_dtype = np.dtype([
        ("normal", "<f4", (3,)),
        ("v0", "<f4", (3,)),
        ("v1", "<f4", (3,)),
        ("v2", "<f4", (3,)),
        ("attr", "<u2"),
    ])

    records = np.empty(num_triangles, dtype=record_dtype)
    records["normal"] = normals
    records["v0"] = tri_v0
    records["v1"] = tri_v1
    records["v2"] = tri_v2
    records["attr"] = 0

    header = b"Binary STL export generated by Hueforge-Lite"[:80].ljust(80, b"\0")

    with open(output_path, "wb") as f:
        f.write(header)
        f.write(struct.pack("<I", num_triangles))
        f.write(records.tobytes())


class WatertightMeshBuilder:
    """Builds a closed, 2-manifold triangular mesh from a 2D height grid."""

    def __init__(self, dimensions: PhysicalDimensions) -> None:
        self._dims = dimensions

    def build_mesh(self, z_grid: np.ndarray) -> TriangleMesh:
        """Constructs the manifold 3D mesh."""
        if z_grid.ndim != 2:
            raise ValueError(f"z_grid must be a 2D array, got ndim={z_grid.ndim}")

        rows, cols = z_grid.shape
        if rows < 2 or cols < 2:
            raise ValueError(f"Grid must be at least 2x2, got {rows}x{cols}")

        grid_x, grid_y = _build_xy_coordinates(
            rows=rows,
            cols=cols,
            width_mm=self._dims.width_mm,
            height_mm=self._dims.height_mm,
        )

        flat_x = grid_x.ravel()
        flat_y = grid_y.ravel()
        flat_top_z = z_grid.astype(np.float32).ravel()
        flat_bot_z = np.full_like(flat_top_z, self._dims.base_floor_z_mm)

        top_vertices = np.column_stack([flat_x, flat_y, flat_top_z])
        bottom_vertices = np.column_stack([flat_x, flat_y, flat_bot_z])
        all_vertices = np.vstack([top_vertices, bottom_vertices])

        bottom_offset = rows * cols
        top_faces = _build_top_surface_faces(rows, cols)
        bottom_faces = _build_bottom_surface_faces(rows, cols, offset=bottom_offset)
        skirt_faces = _build_skirt_wall_faces(rows, cols, offset=bottom_offset)

        all_faces = np.vstack([top_faces, bottom_faces, skirt_faces])

        mesh = TriangleMesh(vertices=all_vertices, faces=all_faces)
        _validate_manifold_structure(mesh)
        return mesh