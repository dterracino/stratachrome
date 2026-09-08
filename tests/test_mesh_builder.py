"""Geometry correctness tests for watertight heightmap meshes."""

from __future__ import annotations

import unittest

import numpy as np

from stratachrome.mesh_builder import PhysicalDimensions, WatertightMeshBuilder


class WatertightMeshBuilderTests(unittest.TestCase):
    def test_uniform_pixel_stack_merges_into_one_cuboid(self) -> None:
        dimensions = PhysicalDimensions(40.0, 30.0)

        for height in (0.2, 1.2):
            z_grid = np.full((3, 4), height, dtype=np.float32)
            mesh = WatertightMeshBuilder(dimensions).build_mesh(z_grid)

            self.assertEqual(mesh.vertex_count, 8)
            self.assertEqual(mesh.face_count, 12)
            np.testing.assert_allclose(mesh.vertices.min(axis=0), [0.0, 0.0, 0.0])
            np.testing.assert_allclose(
                mesh.vertices.max(axis=0),
                [dimensions.width_mm, dimensions.height_mm, height],
            )

    def test_floor_contacts_bed_and_horizontal_faces_point_outward(self) -> None:
        z_grid = np.asarray([[0.2, 0.3, 0.4], [0.5, 0.6, 0.7]], dtype=np.float32)
        mesh = WatertightMeshBuilder(PhysicalDimensions(20.0, 10.0)).build_mesh(z_grid)

        triangle_vertices = mesh.vertices[mesh.faces]
        normals = np.cross(
            triangle_vertices[:, 1] - triangle_vertices[:, 0],
            triangle_vertices[:, 2] - triangle_vertices[:, 0],
        )
        horizontal = np.all(
            np.isclose(triangle_vertices[:, :, 2], triangle_vertices[:, :1, 2]),
            axis=1,
        )
        top = horizontal & (triangle_vertices[:, 0, 2] > 0.0)
        floor = horizontal & np.isclose(triangle_vertices[:, 0, 2], 0.0)
        self.assertEqual(int(np.count_nonzero(top)), 2 * z_grid.size)
        self.assertTrue(np.all(normals[top, 2] > 0.0))
        self.assertTrue(np.all(normals[floor, 2] < 0.0))

    def test_each_source_pixel_has_a_flat_full_size_top(self) -> None:
        z_grid = np.asarray([[0.2, 0.6], [0.4, 0.8]], dtype=np.float32)
        mesh = WatertightMeshBuilder(PhysicalDimensions(20.0, 10.0)).build_mesh(z_grid)
        triangles = mesh.vertices[mesh.faces]
        horizontal = np.all(
            np.isclose(triangles[:, :, 2], triangles[:, :1, 2]),
            axis=1,
        )
        top = triangles[horizontal & (triangles[:, 0, 2] > 0.0)]

        edge_1 = top[:, 1, :2] - top[:, 0, :2]
        edge_2 = top[:, 2, :2] - top[:, 0, :2]
        areas = np.abs(edge_1[:, 0] * edge_2[:, 1] - edge_1[:, 1] * edge_2[:, 0]) / 2.0
        for height in z_grid.ravel():
            height_faces = top[np.isclose(top[:, 0, 2], height)]
            height_areas = areas[np.isclose(top[:, 0, 2], height)]
            self.assertEqual(len(height_faces), 2)
            self.assertGreater(float(np.sum(height_areas)), 49.0)
            self.assertLessEqual(float(np.sum(height_areas)), 50.0)

    def test_mesh_is_strictly_two_manifold(self) -> None:
        z_grid = np.asarray([[0.2, 0.6, 0.2], [0.7, 0.3, 0.8]], dtype=np.float32)
        mesh = WatertightMeshBuilder(PhysicalDimensions(30.0, 20.0)).build_mesh(z_grid)
        edges = np.sort(
            np.vstack(
                (
                    mesh.faces[:, [0, 1]],
                    mesh.faces[:, [1, 2]],
                    mesh.faces[:, [2, 0]],
                )
            ),
            axis=1,
        )
        _, edge_use_counts = np.unique(edges, axis=0, return_counts=True)

        np.testing.assert_array_equal(edge_use_counts, np.full_like(edge_use_counts, 2))

        triangles = mesh.vertices[mesh.faces]
        normals = np.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
        )
        self.assertTrue(np.all(np.linalg.norm(normals, axis=1) > 0.0))

    def test_coplanar_pixel_region_removes_interior_faces(self) -> None:
        z_grid = np.full((8, 10), 0.4, dtype=np.float32)
        mesh = WatertightMeshBuilder(PhysicalDimensions(20.0, 16.0)).build_mesh(z_grid)
        unreduced_top_faces = 2 * (2 * z_grid.shape[0] - 1) * (2 * z_grid.shape[1] - 1)
        triangles = mesh.vertices[mesh.faces]
        top_faces = np.all(
            np.isclose(triangles[:, :, 2], 0.4),
            axis=1,
        )

        self.assertLess(int(np.count_nonzero(top_faces)), unreduced_top_faces // 4)
        self.assertEqual(len(np.unique(mesh.faces)), mesh.vertex_count)

    def test_mesh_build_reports_reduction_progress(self) -> None:
        messages: list[str] = []
        z_grid = np.full((8, 10), 0.4, dtype=np.float32)

        WatertightMeshBuilder(PhysicalDimensions(20.0, 16.0)).build_mesh(
            z_grid,
            progress=messages.append,
        )

        self.assertTrue(any("Reducing plateau elevation" in message for message in messages))
        self.assertTrue(any("Reduced top surface" in message for message in messages))
        self.assertTrue(any("Compacting unused vertices" in message for message in messages))
        self.assertTrue(messages[-1].startswith("Mesh complete:"))

    def test_progress_is_bounded_for_many_distinct_elevations(self) -> None:
        block_heights = np.arange(100, dtype=np.float32).reshape(10, 10) * 0.1 + 0.2
        z_grid = np.repeat(np.repeat(block_heights, 2, axis=0), 2, axis=1)
        messages: list[str] = []

        WatertightMeshBuilder(PhysicalDimensions(20.0, 20.0)).build_mesh(
            z_grid,
            progress=messages.append,
        )

        elevation_messages = [
            message for message in messages if message.startswith("Reducing plateau elevation")
        ]
        self.assertLessEqual(len(elevation_messages), 22)


if __name__ == "__main__":
    unittest.main()
