"""Geometry correctness tests for watertight heightmap meshes."""

from __future__ import annotations

import unittest

import numpy as np

from stratachrome.mesh_builder import PhysicalDimensions, WatertightMeshBuilder


class WatertightMeshBuilderTests(unittest.TestCase):
    def test_floor_contacts_bed_and_horizontal_faces_point_outward(self) -> None:
        z_grid = np.asarray([[0.2, 0.3, 0.4], [0.5, 0.6, 0.7]], dtype=np.float32)
        mesh = WatertightMeshBuilder(PhysicalDimensions(20.0, 10.0)).build_mesh(z_grid)

        vertex_count_per_surface = z_grid.size
        horizontal_face_count_per_surface = 2 * (z_grid.shape[0] - 1) * (
            z_grid.shape[1] - 1
        )
        floor_vertices = mesh.vertices[vertex_count_per_surface:]
        np.testing.assert_array_equal(
            floor_vertices[:, 2],
            np.zeros(vertex_count_per_surface, dtype=np.float32),
        )

        triangle_vertices = mesh.vertices[mesh.faces]
        normals = np.cross(
            triangle_vertices[:, 1] - triangle_vertices[:, 0],
            triangle_vertices[:, 2] - triangle_vertices[:, 0],
        )
        self.assertTrue(np.all(normals[:horizontal_face_count_per_surface, 2] > 0.0))
        floor_start = horizontal_face_count_per_surface
        floor_end = floor_start + horizontal_face_count_per_surface
        self.assertTrue(np.all(normals[floor_start:floor_end, 2] < 0.0))


if __name__ == "__main__":
    unittest.main()
