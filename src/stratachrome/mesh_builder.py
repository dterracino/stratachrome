"""Build exact stepped relief meshes and export binary STL files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, cast

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


class _MeshAssembler:
    """Weld exact grid/height intersections while accumulating oriented faces."""

    def __init__(self, x_coords: np.ndarray, y_coords: np.ndarray) -> None:
        self._x_coords = x_coords
        self._y_coords = y_coords
        self._vertex_indices: dict[tuple[int, int, float], int] = {}
        self.vertices: list[tuple[float, float, float]] = []
        self.faces: list[tuple[int, int, int]] = []

    def grid_vertex(self, row: int, col: int, z: float) -> int:
        key = (row, col, z)
        existing = self._vertex_indices.get(key)
        if existing is not None:
            return existing
        index = len(self.vertices)
        self._vertex_indices[key] = index
        self.vertices.append((float(self._x_coords[col]), float(self._y_coords[row]), z))
        return index

    def loose_vertex(self, x: float, y: float, z: float) -> int:
        index = len(self.vertices)
        self.vertices.append((x, y, z))
        return index

    def oriented_face(
        self,
        first: int,
        second: int,
        third: int,
        expected_normal: tuple[float, float, float],
    ) -> None:
        points = np.asarray(
            (self.vertices[first], self.vertices[second], self.vertices[third]),
            dtype=np.float64,
        )
        normal = np.cross(points[1] - points[0], points[2] - points[0])
        if float(np.dot(normal, expected_normal)) < 0.0:
            second, third = third, second
            normal = -normal
        if float(np.linalg.norm(normal)) == 0.0:
            raise RuntimeError("Exact stepped mesh construction produced a degenerate triangle.")
        self.faces.append((first, second, third))


def _component_boundary_vertices(mask: np.ndarray) -> np.ndarray:
    """Return every grid vertex on the boundary of a 4-connected cell mask."""
    boundary = np.zeros((mask.shape[0] + 1, mask.shape[1] + 1), dtype=bool)
    above = np.pad(mask[:-1, :], ((1, 0), (0, 0)))
    below = np.pad(mask[1:, :], ((0, 1), (0, 0)))
    left = np.pad(mask[:, :-1], ((0, 0), (1, 0)))
    right = np.pad(mask[:, 1:], ((0, 0), (0, 1)))

    row_ids, col_ids = np.nonzero(mask & ~above)
    boundary[row_ids, col_ids] = True
    boundary[row_ids, col_ids + 1] = True
    row_ids, col_ids = np.nonzero(mask & ~below)
    boundary[row_ids + 1, col_ids] = True
    boundary[row_ids + 1, col_ids + 1] = True
    row_ids, col_ids = np.nonzero(mask & ~left)
    boundary[row_ids, col_ids] = True
    boundary[row_ids + 1, col_ids] = True
    row_ids, col_ids = np.nonzero(mask & ~right)
    boundary[row_ids, col_ids + 1] = True
    boundary[row_ids + 1, col_ids + 1] = True
    return np.argwhere(boundary)


def _add_top_surfaces(
    assembler: _MeshAssembler,
    heights: np.ndarray,
    progress: MeshProgressCallback | None,
) -> None:
    """Triangulate same-height pixel regions using their exact grid boundaries."""
    height_values = np.unique(heights)
    report_stride = max(1, int(np.ceil(len(height_values) / 20)))

    for height_number, height in enumerate(height_values, start=1):
        if progress is not None and (
            height_number == 1
            or height_number == len(height_values)
            or (height_number - 1) % report_stride == 0
        ):
            progress(
                f"Reducing plateau elevation {height_number}/{len(height_values)} "
                f"({float(height):.2f} mm)..."
            )

        labels, component_count = cast(
            tuple[np.ndarray, int],
            ndimage.label(heights == height),
        )
        component_slices = ndimage.find_objects(labels)
        for label_index in range(1, component_count + 1):
            component_slice = component_slices[label_index - 1]
            if component_slice is None:
                continue
            component = labels[component_slice] == label_index
            local_points = _component_boundary_vertices(component)
            row_offset = int(component_slice[0].start)
            col_offset = int(component_slice[1].start)
            grid_points = local_points + np.asarray((row_offset, col_offset))

            triangulation = Delaunay(grid_points[:, [1, 0]])
            point_triangles = triangulation.simplices
            centroids = np.mean(grid_points[point_triangles], axis=1)
            cell_rows = np.floor(centroids[:, 0]).astype(np.int64) - row_offset
            cell_cols = np.floor(centroids[:, 1]).astype(np.int64) - col_offset
            inside = (
                (cell_rows >= 0)
                & (cell_rows < component.shape[0])
                & (cell_cols >= 0)
                & (cell_cols < component.shape[1])
            )
            inside_indices = np.nonzero(inside)[0]
            inside[inside_indices] &= component[
                cell_rows[inside_indices],
                cell_cols[inside_indices],
            ]

            for triangle in point_triangles[inside]:
                vertices = [
                    assembler.grid_vertex(
                        int(grid_points[point_index, 0]),
                        int(grid_points[point_index, 1]),
                        float(height),
                    )
                    for point_index in triangle
                ]
                assembler.oriented_face(
                    vertices[0],
                    vertices[1],
                    vertices[2],
                    expected_normal=(0.0, 0.0, 1.0),
                )


def _grid_point_height_levels(heights: np.ndarray, floor_z: float) -> list[list[set[float]]]:
    """Collect every incident elevation at each XY grid intersection."""
    rows, cols = heights.shape
    levels = [[{floor_z} for _ in range(cols + 1)] for _ in range(rows + 1)]
    for row in range(rows):
        for col in range(cols):
            height = float(heights[row, col])
            levels[row][col].add(height)
            levels[row][col + 1].add(height)
            levels[row + 1][col].add(height)
            levels[row + 1][col + 1].add(height)
    return levels


def _repair_height_saddles(heights: np.ndarray) -> tuple[np.ndarray, int]:
    """Remove diagonal-only layer contacts that cannot form a two-manifold solid."""
    repaired: np.ndarray | None = None
    repair_count = 0

    def values() -> np.ndarray:
        return repaired if repaired is not None else heights

    def raise_bridge(
        first: tuple[int, int],
        second: tuple[int, int],
        target_height: float,
    ) -> bool:
        nonlocal repaired, repair_count
        current = values()
        bridge = first if current[first] >= current[second] else second
        if float(current[bridge]) >= target_height:
            return False
        if repaired is None:
            repaired = heights.copy()
        repaired[bridge] = target_height
        repair_count += 1
        return True

    changed = True
    pass_count = 0
    while changed and pass_count < 8:
        changed = False
        pass_count += 1
        for row in range(heights.shape[0] - 1):
            for col in range(heights.shape[1] - 1):
                current = values()
                top_left = float(current[row, col])
                top_right = float(current[row, col + 1])
                bottom_left = float(current[row + 1, col])
                bottom_right = float(current[row + 1, col + 1])

                down_target = min(top_left, bottom_right)
                if down_target > max(top_right, bottom_left):
                    changed = (
                        raise_bridge(
                            (row, col + 1),
                            (row + 1, col),
                            down_target,
                        )
                        or changed
                    )

                current = values()
                top_left = float(current[row, col])
                top_right = float(current[row, col + 1])
                bottom_left = float(current[row + 1, col])
                bottom_right = float(current[row + 1, col + 1])
                up_target = min(top_right, bottom_left)
                if up_target > max(top_left, bottom_right):
                    changed = (
                        raise_bridge(
                            (row, col),
                            (row + 1, col + 1),
                            up_target,
                        )
                        or changed
                    )

    return (repaired if repaired is not None else heights), repair_count


def _add_vertical_wall(
    assembler: _MeshAssembler,
    start: tuple[int, int],
    end: tuple[int, int],
    lower_z: float,
    upper_z: float,
    point_levels: list[list[set[float]]],
    expected_normal: tuple[float, float, float],
) -> None:
    """Triangulate a vertical wall while honoring all incident Z split points."""
    if upper_z <= lower_z:
        return
    start_col, start_row = start
    end_col, end_row = end
    start_levels = sorted(
        {lower_z, upper_z}
        | {level for level in point_levels[start_row][start_col] if lower_z < level < upper_z}
    )
    end_levels = sorted(
        {lower_z, upper_z}
        | {level for level in point_levels[end_row][end_col] if lower_z < level < upper_z}
    )
    start_vertices = [assembler.grid_vertex(start_row, start_col, z) for z in start_levels]
    end_vertices = [assembler.grid_vertex(end_row, end_col, z) for z in end_levels]

    start_index = 0
    end_index = 0
    while start_index < len(start_levels) - 1 or end_index < len(end_levels) - 1:
        next_start = (
            start_levels[start_index + 1] if start_index < len(start_levels) - 1 else float("inf")
        )
        next_end = end_levels[end_index + 1] if end_index < len(end_levels) - 1 else float("inf")
        if next_start == next_end:
            assembler.oriented_face(
                start_vertices[start_index],
                end_vertices[end_index],
                end_vertices[end_index + 1],
                expected_normal,
            )
            assembler.oriented_face(
                start_vertices[start_index],
                end_vertices[end_index + 1],
                start_vertices[start_index + 1],
                expected_normal,
            )
            start_index += 1
            end_index += 1
        elif next_start < next_end:
            assembler.oriented_face(
                start_vertices[start_index],
                end_vertices[end_index],
                start_vertices[start_index + 1],
                expected_normal,
            )
            start_index += 1
        else:
            assembler.oriented_face(
                start_vertices[start_index],
                end_vertices[end_index],
                end_vertices[end_index + 1],
                expected_normal,
            )
            end_index += 1


def _add_step_walls(
    assembler: _MeshAssembler,
    heights: np.ndarray,
    floor_z: float,
) -> None:
    """Add exact vertical walls on outer and unequal-height pixel boundaries."""
    rows, cols = heights.shape
    point_levels = _grid_point_height_levels(heights, floor_z)
    for row in range(rows):
        for col in range(cols):
            height = float(heights[row, col])
            west = floor_z if col == 0 else float(heights[row, col - 1])
            east = floor_z if col == cols - 1 else float(heights[row, col + 1])
            south = floor_z if row == 0 else float(heights[row - 1, col])
            north = floor_z if row == rows - 1 else float(heights[row + 1, col])
            if height > west:
                _add_vertical_wall(
                    assembler,
                    (col, row),
                    (col, row + 1),
                    west,
                    height,
                    point_levels,
                    (-1.0, 0.0, 0.0),
                )
            if height > east:
                _add_vertical_wall(
                    assembler,
                    (col + 1, row),
                    (col + 1, row + 1),
                    east,
                    height,
                    point_levels,
                    (1.0, 0.0, 0.0),
                )
            if height > south:
                _add_vertical_wall(
                    assembler,
                    (col, row),
                    (col + 1, row),
                    south,
                    height,
                    point_levels,
                    (0.0, -1.0, 0.0),
                )
            if height > north:
                _add_vertical_wall(
                    assembler,
                    (col, row + 1),
                    (col + 1, row + 1),
                    north,
                    height,
                    point_levels,
                    (0.0, 1.0, 0.0),
                )


def _perimeter_grid_points(rows: int, cols: int) -> list[tuple[int, int]]:
    """Return grid perimeter points in counter-clockwise XY order."""
    points = [(0, col) for col in range(cols + 1)]
    points.extend((row, cols) for row in range(1, rows + 1))
    points.extend((rows, col) for col in range(cols - 1, -1, -1))
    points.extend((row, 0) for row in range(rows - 1, 0, -1))
    return points


def _add_floor(
    assembler: _MeshAssembler,
    rows: int,
    cols: int,
    floor_z: float,
    width_mm: float,
    height_mm: float,
) -> None:
    perimeter = [
        assembler.grid_vertex(row, col, floor_z) for row, col in _perimeter_grid_points(rows, cols)
    ]
    center = assembler.loose_vertex(width_mm / 2.0, height_mm / 2.0, floor_z)
    for index, current in enumerate(perimeter):
        assembler.oriented_face(
            center,
            perimeter[(index + 1) % len(perimeter)],
            current,
            expected_normal=(0.0, 0.0, -1.0),
        )


class WatertightMeshBuilder:
    """Construct one exact, two-manifold stepped relief from a pixel heightmap."""

    def __init__(self, dimensions: PhysicalDimensions) -> None:
        self._dims = dimensions

    def build_mesh(
        self,
        z_grid: np.ndarray,
        progress: MeshProgressCallback | None = None,
    ) -> TriangleMesh:
        """Transform an image-shaped elevation grid into a printable solid.

        Pixel tops remain horizontal and full-sized. Unequal neighboring pixels
        meet at true vertical walls on their shared raster boundary.
        """
        if z_grid.ndim != 2:
            raise ValueError(f"z_grid must be a 2D array, got ndim={z_grid.ndim}")
        if z_grid.size == 0:
            raise ValueError("z_grid cannot be empty.")

        heights = np.asarray(np.flipud(z_grid), dtype=np.float32)
        floor_z = self._dims.base_floor_z_mm
        if not np.all(np.isfinite(heights)):
            raise ValueError("z_grid must contain only finite elevations.")
        if np.any(heights <= floor_z):
            raise ValueError("Every pixel elevation must be above the base floor.")
        heights, saddle_repairs = _repair_height_saddles(heights)

        rows, cols = heights.shape
        if progress is not None:
            progress(f"Building exact {cols:,} x {rows:,} stepped-pixel grid...")
            if saddle_repairs:
                progress(
                    f"Resolved {saddle_repairs:,} diagonal corner "
                    "contacts for manifold topology."
                )
        x_coords = np.linspace(0.0, self._dims.width_mm, cols + 1, dtype=np.float64)
        y_coords = np.linspace(0.0, self._dims.height_mm, rows + 1, dtype=np.float64)
        assembler = _MeshAssembler(x_coords, y_coords)

        _add_top_surfaces(assembler, heights, progress)
        if progress is not None:
            progress(f"Reduced top surface to {len(assembler.faces):,} triangles.")
            progress("Building exact vertical step walls...")
        _add_step_walls(assembler, heights, floor_z)
        if progress is not None:
            progress("Building floor...")
        _add_floor(
            assembler,
            rows,
            cols,
            floor_z,
            self._dims.width_mm,
            self._dims.height_mm,
        )

        if progress is not None:
            progress("Compacting unused vertices...")
        vertices = np.asarray(assembler.vertices, dtype=np.float32)
        faces = np.asarray(assembler.faces, dtype=np.int32)
        used_vertices, compact_faces = np.unique(faces, return_inverse=True)
        vertices = vertices[used_vertices]
        faces = compact_faces.reshape(faces.shape).astype(np.int32, copy=False)
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
