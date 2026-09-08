"""Build flat-pixel relief meshes and export binary STL files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from scipy import ndimage
from scipy.spatial import Delaunay

MeshProgressCallback = Callable[[str], None]


@dataclass(frozen=True)
class PhysicalDimensions:
    """Physical print dimensions in millimeters."""

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
    """Indexed triangular mesh geometry."""

    vertices: np.ndarray
    faces: np.ndarray

    @property
    def vertex_count(self) -> int:
        return int(self.vertices.shape[0])

    @property
    def face_count(self) -> int:
        return int(self.faces.shape[0])


def _component_boundary_vertices(
    mask: np.ndarray,
    *,
    required_vertices: np.ndarray | None = None,
    simplify_vertical: bool = True,
) -> np.ndarray:
    """Return corners and junctions on the boundary of a cell mask."""
    boundary = np.zeros((mask.shape[0] + 1, mask.shape[1] + 1), dtype=bool)
    above = np.pad(mask[:-1, :], ((1, 0), (0, 0)))
    below = np.pad(mask[1:, :], ((0, 1), (0, 0)))
    left = np.pad(mask[:, :-1], ((0, 0), (1, 0)))
    right = np.pad(mask[:, 1:], ((0, 0), (0, 1)))

    top_edges = mask & ~above
    bottom_edges = mask & ~below
    left_edges = mask & ~left
    right_edges = mask & ~right
    row_ids, col_ids = np.nonzero(top_edges)
    boundary[row_ids, col_ids] = True
    boundary[row_ids, col_ids + 1] = True
    row_ids, col_ids = np.nonzero(bottom_edges)
    boundary[row_ids + 1, col_ids] = True
    boundary[row_ids + 1, col_ids + 1] = True
    row_ids, col_ids = np.nonzero(left_edges)
    boundary[row_ids, col_ids] = True
    boundary[row_ids + 1, col_ids] = True
    row_ids, col_ids = np.nonzero(right_edges)
    boundary[row_ids, col_ids + 1] = True
    boundary[row_ids + 1, col_ids + 1] = True

    padded = np.pad(mask, 1, constant_values=False)
    top_left = padded[:-1, :-1]
    top_right = padded[:-1, 1:]
    bottom_left = padded[1:, :-1]
    bottom_right = padded[1:, 1:]
    straight_horizontal = (
        (top_left == top_right) & (bottom_left == bottom_right) & (top_left != bottom_left)
    )
    straight_vertical = (
        (top_left == bottom_left) & (top_right == bottom_right) & (top_left != top_right)
    )
    removable = straight_horizontal
    if simplify_vertical:
        removable |= straight_vertical
    if required_vertices is not None:
        removable &= ~required_vertices
    boundary &= ~removable
    return np.argwhere(boundary)


def _mask_triangles(
    mask: np.ndarray,
    *,
    required_vertices: np.ndarray | None = None,
    simplify_vertical: bool = True,
) -> list[np.ndarray]:
    """Triangulate connected mask regions from their simplified boundaries."""
    labels, component_count = ndimage.label(mask)
    triangles: list[np.ndarray] = []
    for label_index in range(1, component_count + 1):
        component = labels == label_index
        grid_points = _component_boundary_vertices(
            component,
            required_vertices=required_vertices,
            simplify_vertical=simplify_vertical,
        )
        if len(grid_points) < 3:
            continue
        triangulation = Delaunay(grid_points[:, [1, 0]])
        point_triangles = triangulation.simplices
        centroids = np.mean(grid_points[point_triangles], axis=1)
        cell_rows = np.floor(centroids[:, 0]).astype(np.int64)
        cell_cols = np.floor(centroids[:, 1]).astype(np.int64)
        inside = (
            (cell_rows >= 0)
            & (cell_rows < mask.shape[0])
            & (cell_cols >= 0)
            & (cell_cols < mask.shape[1])
        )
        valid = np.nonzero(inside)[0]
        inside[valid] &= component[cell_rows[valid], cell_cols[valid]]
        triangles.extend(grid_points[point_triangles[inside]])
    return triangles


def _repair_diagonal_contacts(heights: np.ndarray) -> np.ndarray:
    """Bridge diagonal-only contacts that would create non-manifold edges."""
    repaired = heights.copy()
    for _ in range(8):
        changed = False
        for row in range(repaired.shape[0] - 1):
            for col in range(repaired.shape[1] - 1):
                top_left = repaired[row, col]
                top_right = repaired[row, col + 1]
                bottom_left = repaired[row + 1, col]
                bottom_right = repaired[row + 1, col + 1]

                down_height = min(top_left, bottom_right)
                if down_height > max(top_right, bottom_left):
                    if top_right >= bottom_left:
                        repaired[row, col + 1] = down_height
                    else:
                        repaired[row + 1, col] = down_height
                    changed = True

                up_height = min(top_right, bottom_left)
                if up_height > max(top_left, bottom_right):
                    if top_left >= bottom_right:
                        repaired[row, col] = up_height
                    else:
                        repaired[row + 1, col + 1] = up_height
                    changed = True
        if not changed:
            break
    return repaired


class _IndexedMeshAccumulator:
    """Weld coordinate-identical vertices while accumulating triangle faces."""

    def __init__(self) -> None:
        self.vertices: list[tuple[float, float, float]] = []
        self.faces: list[tuple[int, int, int]] = []
        self._indices: dict[tuple[float, float, float], int] = {}

    def vertex(self, x: float, y: float, z: float) -> int:
        point = (x, y, z)
        index = self._indices.get(point)
        if index is None:
            index = len(self.vertices)
            self._indices[point] = index
            self.vertices.append(point)
        return index

    def triangle(
        self,
        first: tuple[float, float, float],
        second: tuple[float, float, float],
        third: tuple[float, float, float],
    ) -> None:
        self.faces.append(
            (
                self.vertex(*first),
                self.vertex(*second),
                self.vertex(*third),
            )
        )


def _add_mask_surface(
    accumulator: _IndexedMeshAccumulator,
    mask: np.ndarray,
    z_height: float,
    pixel_width: float,
    pixel_height: float,
    *,
    upward: bool,
    required_vertices: np.ndarray | None = None,
) -> None:
    """Triangulate connected cell regions while retaining boundary vertices."""
    for points in _mask_triangles(mask, required_vertices=required_vertices):
        first, second, third = points
        signed_area = (second[1] - first[1]) * (third[0] - first[0]) - (second[0] - first[0]) * (
            third[1] - first[1]
        )
        if (signed_area > 0) != upward:
            second, third = third, second
        accumulator.triangle(
            (float(first[1]) * pixel_width, float(first[0]) * pixel_height, z_height),
            (float(second[1]) * pixel_width, float(second[0]) * pixel_height, z_height),
            (float(third[1]) * pixel_width, float(third[0]) * pixel_height, z_height),
        )


def _add_wall_surface(
    accumulator: _IndexedMeshAccumulator,
    mask: np.ndarray,
    levels: np.ndarray,
    horizontal_step: float,
    fixed_coordinate: float,
    *,
    fixed_axis: str,
    positive_normal: bool,
) -> None:
    """Triangulate a connected exposed-wall mask across distance and height."""
    required_vertices = np.zeros(
        (mask.shape[0] + 1, mask.shape[1] + 1),
        dtype=bool,
    )
    if mask.shape[1] > 1:
        profile_changes = np.any(mask[:, :-1] != mask[:, 1:], axis=0)
        required_vertices[:, 1:-1] = profile_changes
    for points in _mask_triangles(
        mask,
        required_vertices=required_vertices,
        simplify_vertical=False,
    ):
        first, second, third = points
        signed_area = (second[1] - first[1]) * (third[0] - first[0]) - (second[0] - first[0]) * (
            third[1] - first[1]
        )
        if fixed_axis == "y":
            correct_winding = (signed_area < 0) == positive_normal
            coordinates = [
                (float(point[1]) * horizontal_step, fixed_coordinate, float(levels[point[0]]))
                for point in (first, second, third)
            ]
        else:
            correct_winding = (signed_area > 0) == positive_normal
            coordinates = [
                (fixed_coordinate, float(point[1]) * horizontal_step, float(levels[point[0]]))
                for point in (first, second, third)
            ]
        if not correct_winding:
            coordinates[1], coordinates[2] = coordinates[2], coordinates[1]
        accumulator.triangle(*coordinates)


def _add_vertical_walls(
    accumulator: _IndexedMeshAccumulator,
    heights: np.ndarray,
    floor_z: float,
    pixel_width: float,
    pixel_height: float,
) -> None:
    """Add vertical boundary faces between neighboring cell elevations."""
    rows, cols = heights.shape
    levels = np.unique(np.concatenate((heights.ravel(), [floor_z])))
    lower_levels = levels[:-1, np.newaxis]

    for row_boundary in range(rows + 1):
        y = row_boundary * pixel_height
        south = np.full(cols, floor_z) if row_boundary == 0 else heights[row_boundary - 1]
        north = np.full(cols, floor_z) if row_boundary == rows else heights[row_boundary]
        for positive_normal, taller, shorter in (
            (True, south, north),
            (False, north, south),
        ):
            mask = (lower_levels < taller) & (lower_levels >= shorter)
            if np.any(mask):
                _add_wall_surface(
                    accumulator,
                    mask,
                    levels,
                    pixel_width,
                    y,
                    fixed_axis="y",
                    positive_normal=positive_normal,
                )

    for col_boundary in range(cols + 1):
        x = col_boundary * pixel_width
        west = np.full(rows, floor_z) if col_boundary == 0 else heights[:, col_boundary - 1]
        east = np.full(rows, floor_z) if col_boundary == cols else heights[:, col_boundary]
        for positive_normal, taller, shorter in (
            (True, west, east),
            (False, east, west),
        ):
            mask = (lower_levels < taller) & (lower_levels >= shorter)
            if np.any(mask):
                _add_wall_surface(
                    accumulator,
                    mask,
                    levels,
                    pixel_height,
                    x,
                    fixed_axis="x",
                    positive_normal=positive_normal,
                )


class WatertightMeshBuilder:
    """Construct a two-manifold relief with exact, merged planar surfaces."""

    def __init__(self, dimensions: PhysicalDimensions) -> None:
        self._dims = dimensions

    def build_mesh(
        self,
        z_grid: np.ndarray,
        progress: MeshProgressCallback | None = None,
    ) -> TriangleMesh:
        """Transform an image-shaped elevation grid into a printable solid.

        Coplanar pixel faces are merged across X, Y, and Z while required
        height junctions are retained to keep the indexed surface manifold.
        """
        if z_grid.ndim != 2:
            raise ValueError(f"z_grid must be a 2D array, got ndim={z_grid.ndim}")
        if z_grid.size == 0:
            raise ValueError("z_grid cannot be empty.")

        corrected_grid = np.asarray(np.flipud(z_grid), dtype=np.float32)
        floor_z = self._dims.base_floor_z_mm
        if not np.all(np.isfinite(corrected_grid)):
            raise ValueError("z_grid must contain only finite elevations.")
        if np.any(corrected_grid <= floor_z):
            raise ValueError("Every pixel elevation must be above the base floor.")

        pixel_rows, pixel_cols = corrected_grid.shape
        corrected_grid = _repair_diagonal_contacts(corrected_grid)
        pixel_width = self._dims.width_mm / pixel_cols
        pixel_height = self._dims.height_mm / pixel_rows
        accumulator = _IndexedMeshAccumulator()
        padded_heights = np.pad(corrected_grid, 1, constant_values=floor_z)
        incident_heights = np.sort(
            np.stack(
                (
                    padded_heights[:-1, :-1],
                    padded_heights[:-1, 1:],
                    padded_heights[1:, :-1],
                    padded_heights[1:, 1:],
                )
            ),
            axis=0,
        )
        height_junctions = np.count_nonzero(np.diff(incident_heights, axis=0) != 0.0, axis=0) >= 2

        unique_heights = np.unique(corrected_grid)
        elevation_report_stride = max(1, int(np.ceil(len(unique_heights) / 20)))
        for elevation_number, height in enumerate(unique_heights, start=1):
            report_elevation = (
                elevation_number == 1
                or elevation_number == len(unique_heights)
                or (elevation_number - 1) % elevation_report_stride == 0
            )
            if progress is not None and report_elevation:
                progress(
                    f"Reducing plateau elevation {elevation_number}/{len(unique_heights)} "
                    f"({float(height):.2f} mm)..."
                )
            _add_mask_surface(
                accumulator,
                corrected_grid == height,
                float(height),
                pixel_width,
                pixel_height,
                upward=True,
                required_vertices=height_junctions,
            )
        top_face_count = len(accumulator.faces)
        if progress is not None:
            progress(f"Reduced top surface to {top_face_count:,} triangles.")
            progress("Building floor and vertical heightfield walls...")
        _add_mask_surface(
            accumulator,
            np.ones(corrected_grid.shape, dtype=bool),
            floor_z,
            pixel_width,
            pixel_height,
            upward=False,
            required_vertices=height_junctions,
        )
        _add_vertical_walls(
            accumulator,
            corrected_grid,
            floor_z,
            pixel_width,
            pixel_height,
        )

        if progress is not None:
            progress("Compacting unused vertices...")
        vertices = np.asarray(accumulator.vertices, dtype=np.float32)
        faces = np.asarray(accumulator.faces, dtype=np.int32)
        if progress is not None:
            progress(f"Mesh complete: {len(vertices):,} vertices, {len(faces):,} triangles.")
        return TriangleMesh(vertices=vertices, faces=faces)


def export_binary_stl(mesh: TriangleMesh, output_path: Path) -> None:
    """Write a TriangleMesh as binary STL."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "wb") as file:
        header = b"Stratachrome Watertight Manifold Exporter"
        file.write(header + b"\0" * (80 - len(header)))
        file.write(np.uint32(mesh.face_count).tobytes())

        vertices = mesh.vertices
        faces = mesh.faces
        vertex_0 = vertices[faces[:, 0]]
        vertex_1 = vertices[faces[:, 1]]
        vertex_2 = vertices[faces[:, 2]]
        normals = np.cross(vertex_1 - vertex_0, vertex_2 - vertex_0)
        lengths = np.linalg.norm(normals, axis=1, keepdims=True)
        lengths[lengths == 0.0] = 1.0
        normals /= lengths

        for index in range(mesh.face_count):
            file.write(normals[index].astype(np.float32).tobytes())
            file.write(vertex_0[index].astype(np.float32).tobytes())
            file.write(vertex_1[index].astype(np.float32).tobytes())
            file.write(vertex_2[index].astype(np.float32).tobytes())
            file.write(b"\0\0")
