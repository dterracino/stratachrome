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


def _flat_cell_samples(length_mm: float, count: int) -> np.ndarray:
    """Return two inset coordinates per pixel with sub-nozzle transition gaps."""
    pixel_size = length_mm / count
    float32_guard = float(np.spacing(np.float32(length_mm))) * 8.0
    inset = min(pixel_size * 0.1, max(pixel_size * 0.001, float32_guard))

    starts = np.arange(count, dtype=np.float64) * pixel_size
    stops = starts + pixel_size
    starts[1:] += inset
    stops[:-1] -= inset
    samples = np.column_stack((starts, stops)).ravel().astype(np.float32)
    if np.any(np.diff(samples) <= 0.0):
        raise ValueError("Pixel resolution is too high for stable float32 mesh coordinates.")
    return samples


def _grid_surface_faces(rows: int, cols: int) -> np.ndarray:
    """Triangulate a regular vertex grid with upward CCW winding."""
    row_ids, col_ids = np.indices((rows - 1, cols - 1), dtype=np.int64)
    lower_left = row_ids * cols + col_ids
    lower_right = lower_left + 1
    upper_left = lower_left + cols
    upper_right = upper_left + 1
    return np.vstack(
        (
            np.column_stack((lower_left.ravel(), lower_right.ravel(), upper_left.ravel())),
            np.column_stack((lower_right.ravel(), upper_right.ravel(), upper_left.ravel())),
        )
    )


def _component_boundary_vertices(mask: np.ndarray) -> np.ndarray:
    """Return the grid vertices on the boundary of a 4-connected cell mask."""
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

    return np.argwhere(boundary)


def _reduced_grid_surface_faces(
    heights: np.ndarray,
    progress: MeshProgressCallback | None = None,
) -> np.ndarray:
    """Triangulate connected coplanar XY regions using boundary vertices only."""
    rows, cols = heights.shape
    upper_left = heights[:-1, :-1]
    flat_cells = (
        (upper_left == heights[:-1, 1:])
        & (upper_left == heights[1:, :-1])
        & (upper_left == heights[1:, 1:])
    )
    height_values, height_ranks = np.unique(upper_left, return_inverse=True)
    flat_ranks = np.where(flat_cells, height_ranks.reshape(upper_left.shape), -1)

    reduced_cells = np.zeros_like(flat_cells)
    reduced_face_chunks: list[np.ndarray] = []

    core_rank = flat_ranks[:-1, :-1]
    reducible_cores = (
        (core_rank >= 0)
        & (core_rank == flat_ranks[:-1, 1:])
        & (core_rank == flat_ranks[1:, :-1])
        & (core_rank == flat_ranks[1:, 1:])
    )
    reducible_ranks = np.unique(core_rank[reducible_cores])
    elevation_report_stride = max(1, int(np.ceil(len(reducible_ranks) / 20)))

    for elevation_number, rank in enumerate(reducible_ranks, start=1):
        report_elevation = (
            elevation_number == 1
            or elevation_number == len(reducible_ranks)
            or (elevation_number - 1) % elevation_report_stride == 0
        )
        if progress is not None and report_elevation:
            progress(
                f"Reducing plateau elevation {elevation_number}/{len(reducible_ranks)} "
                f"({float(height_values[rank]):.2f} mm)..."
            )
        rank_mask = flat_ranks == rank
        labels, component_count = ndimage.label(rank_mask)
        if component_count == 0:
            continue
        rank_cores = (
            rank_mask[:-1, :-1]
            & rank_mask[:-1, 1:]
            & rank_mask[1:, :-1]
            & rank_mask[1:, 1:]
        )
        candidate_labels = np.unique(labels[:-1, :-1][rank_cores])
        component_slices = ndimage.find_objects(labels)
        component_report_stride = max(100, int(np.ceil(len(candidate_labels) / 10)))
        for component_number, label_index in enumerate(candidate_labels, start=1):
            if progress is not None and component_number % component_report_stride == 0:
                progress(
                    f"  Plateau elevation {elevation_number}/{len(reducible_ranks)}: "
                    f"{component_number:,}/{len(candidate_labels):,} plateaus..."
                )
            component_slice = component_slices[int(label_index) - 1]
            if component_slice is None:
                continue
            component = labels[component_slice] == label_index
            local_points = _component_boundary_vertices(component)
            row_offset = component_slice[0].start
            col_offset = component_slice[1].start
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
            point_triangles = point_triangles[inside]

            triangle_points = grid_points[point_triangles]
            edge_1 = triangle_points[:, 1] - triangle_points[:, 0]
            edge_2 = triangle_points[:, 2] - triangle_points[:, 0]
            clockwise = edge_1[:, 1] * edge_2[:, 0] - edge_1[:, 0] * edge_2[:, 1] < 0
            point_triangles[clockwise, 1:3] = point_triangles[clockwise, 2:0:-1]
            reduced_cells[component_slice] |= component
            reduced_face_chunks.append(
                grid_points[point_triangles][..., 0] * cols + grid_points[point_triangles][..., 1]
            )

    regular_faces = _grid_surface_faces(rows, cols)
    cell_count = (rows - 1) * (cols - 1)
    regular_cells = ~reduced_cells.ravel()
    face_chunks = [
        regular_faces[:cell_count][regular_cells],
        regular_faces[cell_count:][regular_cells],
        *reduced_face_chunks,
    ]
    return np.vstack(face_chunks).astype(np.int64, copy=False)


def _perimeter_vertex_indices(rows: int, cols: int) -> np.ndarray:
    """Return grid perimeter indices in counter-clockwise XY order."""
    bottom = np.arange(cols, dtype=np.int64)
    right = np.arange(1, rows, dtype=np.int64) * cols + (cols - 1)
    top = (rows - 1) * cols + np.arange(cols - 2, -1, -1, dtype=np.int64)
    left = np.arange(rows - 2, 0, -1, dtype=np.int64) * cols
    return np.concatenate((bottom, right, top, left))


class WatertightMeshBuilder:
    """Construct a two-manifold relief with a flat top for every source pixel."""

    def __init__(self, dimensions: PhysicalDimensions) -> None:
        self._dims = dimensions

    def build_mesh(
        self,
        z_grid: np.ndarray,
        progress: MeshProgressCallback | None = None,
    ) -> TriangleMesh:
        """Transform an image-shaped elevation grid into a printable solid.

        Every source pixel receives a two-triangle horizontal plateau. Adjacent
        plateaus are joined across a microscopic transition strip instead of a
        zero-width T-junction, keeping the indexed surface strictly manifold.
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
        if progress is not None:
            progress(f"Expanding {pixel_cols:,} x {pixel_rows:,} flat-pixel grid...")
        x_coords = _flat_cell_samples(self._dims.width_mm, pixel_cols)
        y_coords = _flat_cell_samples(self._dims.height_mm, pixel_rows)
        expanded_heights = np.repeat(np.repeat(corrected_grid, 2, axis=0), 2, axis=1)
        xx, yy = np.meshgrid(x_coords, y_coords)
        top_vertices = np.column_stack(
            (xx.ravel(), yy.ravel(), expanded_heights.ravel())
        ).astype(np.float32)

        grid_rows, grid_cols = expanded_heights.shape
        top_faces = _reduced_grid_surface_faces(expanded_heights, progress)
        if progress is not None:
            progress(f"Reduced top surface to {len(top_faces):,} triangles.")
        perimeter_top = _perimeter_vertex_indices(grid_rows, grid_cols)

        if progress is not None:
            progress("Building floor and perimeter shell...")
        floor_start = len(top_vertices)
        floor_indices = floor_start + np.arange(len(perimeter_top), dtype=np.int64)
        floor_vertices = top_vertices[perimeter_top].copy()
        floor_vertices[:, 2] = floor_z

        next_top = np.roll(perimeter_top, -1)
        next_floor = np.roll(floor_indices, -1)
        skirt_faces = np.vstack(
            (
                np.column_stack((perimeter_top, floor_indices, next_top)),
                np.column_stack((next_top, floor_indices, next_floor)),
            )
        )

        center_index = floor_start + len(floor_vertices)
        center_vertex = np.asarray(
            [[self._dims.width_mm / 2.0, self._dims.height_mm / 2.0, floor_z]],
            dtype=np.float32,
        )
        bottom_faces = np.column_stack(
            (
                np.full(len(floor_indices), center_index, dtype=np.int64),
                next_floor,
                floor_indices,
            )
        )

        vertices = np.vstack((top_vertices, floor_vertices, center_vertex))
        faces = np.vstack((top_faces, skirt_faces, bottom_faces))
        if progress is not None:
            progress("Compacting unused vertices...")
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
