"""Tests for alternate dominant-color extraction."""

from __future__ import annotations

import unittest

import numpy as np
from PIL import Image

from stratachrome.dominant_colors import extract_kmeans_dominant_colors


class KMeansDominantColorTests(unittest.TestCase):
    def test_extracts_full_resolution_color_clusters(self) -> None:
        pixels = np.zeros((20, 20, 3), dtype=np.uint8)
        pixels[:10, :] = (230, 30, 20)
        pixels[10:, :] = (20, 60, 220)
        image = Image.fromarray(pixels, mode="RGB")

        colors = extract_kmeans_dominant_colors(image, 2)

        self.assertEqual({color.rgb for color in colors}, {(230, 30, 20), (20, 60, 220)})
        self.assertAlmostEqual(sum(color.population for color in colors), 1.0)

    def test_constant_channels_do_not_produce_invalid_centers(self) -> None:
        pixels = np.zeros((10, 20, 3), dtype=np.uint8)
        pixels[:, :10] = (90, 40, 130)
        pixels[:, 10:] = (90, 200, 130)
        image = Image.fromarray(pixels, mode="RGB")

        colors = extract_kmeans_dominant_colors(image, 2)

        self.assertEqual({color.rgb for color in colors}, {(90, 40, 130), (90, 200, 130)})

    def test_extracts_only_the_requested_segmentation_tier(self) -> None:
        pixels = np.zeros((10, 20, 3), dtype=np.uint8)
        pixels[:, :10] = (220, 30, 20)
        pixels[:, 10:] = (20, 60, 220)
        matte = np.zeros((10, 20), dtype=np.float32)
        matte[:, 10:] = 1.0

        colors = extract_kmeans_dominant_colors(
            Image.fromarray(pixels, mode="RGB"),
            4,
            matte=matte,
            foreground=True,
        )

        self.assertEqual(len(colors), 1)
        self.assertEqual(colors[0].rgb, (20, 60, 220))

    def test_limits_clusters_to_distinct_colors(self) -> None:
        image = Image.new("RGB", (8, 8), (12, 34, 56))

        colors = extract_kmeans_dominant_colors(image, 8)

        self.assertEqual(len(colors), 1)
        self.assertEqual(colors[0].rgb, (12, 34, 56))
        self.assertEqual(colors[0].population, 1.0)

    def test_returns_requested_number_of_nonempty_buckets(self) -> None:
        channel_values = np.arange(256, dtype=np.uint8)
        pixels = np.column_stack(
            (
                channel_values,
                np.roll(channel_values, 37),
                np.roll(channel_values, 91),
            )
        ).reshape(16, 16, 3)
        image = Image.fromarray(pixels, mode="RGB")

        colors = extract_kmeans_dominant_colors(image, 16)

        self.assertEqual(len(colors), 16)
        self.assertTrue(all(color.population > 0.0 for color in colors))

    def test_rejects_invalid_color_count(self) -> None:
        image = Image.new("RGB", (1, 1))

        with self.assertRaisesRegex(ValueError, "between 1 and 32"):
            extract_kmeans_dominant_colors(image, 0)


if __name__ == "__main__":
    unittest.main()
