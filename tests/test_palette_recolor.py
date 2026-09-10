"""Tests for full-resolution perceptual palette recoloring."""

from __future__ import annotations

import unittest

import numpy as np
from PIL import Image

from stratachrome.palette_recolor import recolor_image_to_palette


class PaletteRecolorTests(unittest.TestCase):
    def test_recolors_every_pixel_to_a_palette_member(self) -> None:
        pixels = np.asarray(
            [
                [(0, 0, 0), (40, 40, 40), (200, 200, 200)],
                [(255, 255, 255), (220, 40, 40), (40, 60, 220)],
            ],
            dtype=np.uint8,
        )
        image = Image.fromarray(pixels, mode="RGB")
        palette = ((0, 0, 0), (255, 255, 255), (220, 40, 40))

        recolored = recolor_image_to_palette(image, palette)

        self.assertEqual(recolored.size, image.size)
        result_colors = {
            tuple(int(channel) for channel in color)
            for color in np.asarray(recolored).reshape(-1, 3)
        }
        self.assertTrue(result_colors.issubset(set(palette)))
        self.assertEqual(recolored.getpixel((0, 0)), (0, 0, 0))
        self.assertEqual(recolored.getpixel((0, 1)), (255, 255, 255))
        self.assertEqual(recolored.getpixel((1, 1)), (220, 40, 40))

    def test_rejects_empty_palette(self) -> None:
        with self.assertRaisesRegex(ValueError, "At least one"):
            recolor_image_to_palette(Image.new("RGB", (1, 1)), ())


if __name__ == "__main__":
    unittest.main()
