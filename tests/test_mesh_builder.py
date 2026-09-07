"""Geometry correctness tests for watertight heightmap meshes."""

from __future__ import annotations

import unittest

import numpy as np

from stratachrome.mesh_builder import PhysicalDimensions, WatertightMeshBuilder


class WatertightMeshBuilderTests(unittest.TestCase):
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
            self.assertAlmostEqual(float(np.sum(height_areas)), 50.0)

    def test_mesh_contains_no_sloped_triangles(self) -> None:
        z_grid = np.asarray([[0.2, 0.6, 0.2], [0.7, 0.3, 0.8]], dtype=np.float32)
        mesh = WatertightMeshBuilder(PhysicalDimensions(30.0, 20.0)).build_mesh(z_grid)
        triangles = mesh.vertices[mesh.faces]

        flat_z = np.all(np.isclose(triangles[:, :, 2], triangles[:, :1, 2]), axis=1)
        flat_x = np.all(np.isclose(triangles[:, :, 0], triangles[:, :1, 0]), axis=1)
        flat_y = np.all(np.isclose(triangles[:, :, 1], triangles[:, :1, 1]), axis=1)
        self.assertTrue(np.all(flat_z | flat_x | flat_y))


if __name__ == "__main__":
    unittest.main()
