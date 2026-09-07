"""
mesh_builder.py
---------------
Converts a 2D heightmap Z-grid into a closed terraced triangular mesh with one
flat top per source pixel, vertical height transitions, a solid flat base, and
strict counter-clockwise (CCW) winding. Automatically orients image coordinates
to face the front of the build plate.
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
    """Construct closed terraced meshes from 2D pixel-height grids."""

    def __init__(self, dimensions: PhysicalDimensions) -> None:
        self._dims = dimensions

    def build_mesh(self, z_grid: np.ndarray) -> TriangleMesh:
        """Transform pixel elevations into a watertight terraced mesh.

        Args:
            z_grid: 2D float32 array of surface elevations in millimeters.

        Returns:
            TriangleMesh containing manifold vertices and CCW faces.
        """
        if z_grid.ndim != 2:
            raise ValueError(f"z_grid must be a 2D array, got ndim={z_grid.ndim}")
        if z_grid.size == 0:
            raise ValueError("z_grid cannot be empty.")

        # Flip vertically to align image top-edge (row 0) with the front of the build plate (Y=0)
        corrected_grid = np.asarray(np.flipud(z_grid), dtype=np.float32)

        rows, cols = corrected_grid.shape
        floor_z = self._dims.base_floor_z_mm
        if not np.all(np.isfinite(corrected_grid)):
            raise ValueError("z_grid must contain only finite elevations.")
        if np.any(corrected_grid <= floor_z):
            raise ValueError("Every pixel elevation must be above the base floor.")

        # Each source pixel owns one rectangular cell. Corner vertices are shared
        # only when both their position and elevation match, preserving flat tops
        # while allowing vertical walls at discrete height changes.
        point_cols = cols + 1
        row_ids, col_ids = np.indices((rows, cols), dtype=np.int64)
        point_00 = row_ids * point_cols + col_ids
        point_10 = point_00 + 1
        point_01 = point_00 + point_cols
        point_11 = point_01 + 1

        height_values, height_ranks = np.unique(corrected_grid, return_inverse=True)
        height_ranks = height_ranks.reshape(rows, cols)
        level_count = len(height_values)
        corner_keys = np.concatenate(
            [
                (points * level_count + height_ranks).ravel()
                for points in (point_00, point_10, point_01, point_11)
            ]
        )
        unique_keys, inverse = np.unique(corner_keys, return_inverse=True)
        corners = inverse.reshape(4, rows, cols)
        v00, v10, v01, v11 = corners

        top_points = unique_keys // level_count
        top_levels = unique_keys % level_count
        top_vertices = np.column_stack(
            (
                (top_points % point_cols) * (self._dims.width_mm / cols),
                (top_points // point_cols) * (self._dims.height_mm / rows),
                height_values[top_levels],
            )
        ).astype(np.float32)

        face_chunks: list[np.ndarray] = [
            np.column_stack((v00.ravel(), v10.ravel(), v01.ravel())),
            np.column_stack((v10.ravel(), v11.ravel(), v01.ravel())),
        ]

        # Walls between horizontally adjacent pixels.
        if cols > 1:
            left_height = corrected_grid[:, :-1]
            right_height = corrected_grid[:, 1:]
            left_high = left_height > right_height
            right_high = right_height > left_height
            left_0, left_1 = v10[:, :-1], v11[:, :-1]
            right_0, right_1 = v00[:, 1:], v01[:, 1:]
            if np.any(left_high):
                face_chunks.extend(
                    (
                        np.column_stack(
                            (right_0[left_high], right_1[left_high], left_0[left_high])
                        ),
                        np.column_stack(
                            (right_1[left_high], left_1[left_high], left_0[left_high])
                        ),
                    )
                )
            if np.any(right_high):
                face_chunks.extend(
                    (
                        np.column_stack(
                            (left_0[right_high], right_0[right_high], left_1[right_high])
                        ),
                        np.column_stack(
                            (left_1[right_high], right_0[right_high], right_1[right_high])
                        ),
                    )
                )

        # Walls between vertically adjacent pixels.
        if rows > 1:
            south_height = corrected_grid[:-1, :]
            north_height = corrected_grid[1:, :]
            south_high = south_height > north_height
            north_high = north_height > south_height
            south_0, south_1 = v01[:-1, :], v11[:-1, :]
            north_0, north_1 = v00[1:, :], v10[1:, :]
            if np.any(south_high):
                face_chunks.extend(
                    (
                        np.column_stack(
                            (north_0[south_high], south_0[south_high], north_1[south_high])
                        ),
                        np.column_stack(
                            (north_1[south_high], south_0[south_high], south_1[south_high])
                        ),
                    )
                )
            if np.any(north_high):
                face_chunks.extend(
                    (
                        np.column_stack(
                            (south_0[north_high], south_1[north_high], north_0[north_high])
                        ),
                        np.column_stack(
                            (south_1[north_high], north_1[north_high], north_0[north_high])
                        ),
                    )
                )

        # A perimeter-only floor triangulation keeps the base solid without
        # duplicating a full second image-sized vertex grid.
        bottom_points = np.arange(cols + 1, dtype=np.int64)
        right_points = np.arange(1, rows + 1, dtype=np.int64) * point_cols + cols
        top_points_ordered = rows * point_cols + np.arange(cols - 1, -1, -1, dtype=np.int64)
        left_points = np.arange(rows - 1, 0, -1, dtype=np.int64) * point_cols
        perimeter_points = np.concatenate(
            (bottom_points, right_points, top_points_ordered, left_points)
        )
        floor_start = len(top_vertices)
        floor_indices = floor_start + np.arange(len(perimeter_points), dtype=np.int64)
        floor_vertices = np.column_stack(
            (
                (perimeter_points % point_cols) * (self._dims.width_mm / cols),
                (perimeter_points // point_cols) * (self._dims.height_mm / rows),
                np.full(len(perimeter_points), floor_z),
            )
        ).astype(np.float32)
        floor_lookup = np.full((rows + 1) * (cols + 1), -1, dtype=np.int64)
        floor_lookup[perimeter_points] = floor_indices

        def add_perimeter_wall(
            floor_0: np.ndarray,
            floor_1: np.ndarray,
            top_0: np.ndarray,
            top_1: np.ndarray,
            *,
            reverse: bool,
        ) -> None:
            if reverse:
                face_chunks.extend(
                    (
                        np.column_stack((floor_0, top_0, floor_1)),
                        np.column_stack((floor_1, top_0, top_1)),
                    )
                )
            else:
                face_chunks.extend(
                    (
                        np.column_stack((floor_0, floor_1, top_0)),
                        np.column_stack((floor_1, top_1, top_0)),
                    )
                )

        add_perimeter_wall(
            floor_lookup[point_00[0, :]],
            floor_lookup[point_10[0, :]],
            v00[0, :],
            v10[0, :],
            reverse=False,
        )
        add_perimeter_wall(
            floor_lookup[point_01[-1, :]],
            floor_lookup[point_11[-1, :]],
            v01[-1, :],
            v11[-1, :],
            reverse=True,
        )
        add_perimeter_wall(
            floor_lookup[point_00[:, 0]],
            floor_lookup[point_01[:, 0]],
            v00[:, 0],
            v01[:, 0],
            reverse=True,
        )
        add_perimeter_wall(
            floor_lookup[point_10[:, -1]],
            floor_lookup[point_11[:, -1]],
            v10[:, -1],
            v11[:, -1],
            reverse=False,
        )

        center_index = floor_start + len(floor_vertices)
        center_vertex = np.asarray(
            [[self._dims.width_mm / 2.0, self._dims.height_mm / 2.0, floor_z]],
            dtype=np.float32,
        )
        next_floor = np.roll(floor_indices, -1)
        face_chunks.append(
            np.column_stack(
                (
                    np.full(len(floor_indices), center_index, dtype=np.int64),
                    next_floor,
                    floor_indices,
                )
            )
        )

        vertices = np.vstack((top_vertices, floor_vertices, center_vertex))
        faces = np.vstack(face_chunks).astype(np.int32, copy=False)
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
