"""Tests for perceptual palette lightness redistribution."""

from __future__ import annotations

import unittest

from color_tools import is_in_srgb_gamut, lch_to_lab, rgb_to_lch

from stratachrome.color_spacing import redistribute_palette_lightness


def _hue_distance(first: float, second: float) -> float:
    return abs((first - second + 180.0) % 360.0 - 180.0)


class ColorSpacingTests(unittest.TestCase):
    def test_redistributes_sorted_colors_evenly_across_existing_range(self) -> None:
        colors = ((245, 220, 40), (22, 32, 48), (70, 180, 220), (190, 70, 120))

        result = redistribute_palette_lightness(colors)

        original_lightness = [color.original_lch[0] for color in result]
        adjusted_lightness = [color.adjusted_lch[0] for color in result]
        expected_step = (
            original_lightness[-1] - original_lightness[0]
        ) / (len(result) - 1)
        self.assertEqual(original_lightness, sorted(original_lightness))
        for index, lightness in enumerate(adjusted_lightness):
            self.assertAlmostEqual(
                lightness,
                original_lightness[0] + index * expected_step,
                places=10,
            )

    def test_preserves_hue_and_only_reduces_chroma_when_needed(self) -> None:
        result = redistribute_palette_lightness(
            ((0, 0, 0), (255, 0, 0), (255, 255, 255))
        )

        for color in result:
            self.assertLessEqual(color.adjusted_lch[1], color.original_lch[1] + 1e-9)
            self.assertLess(
                _hue_distance(color.adjusted_lch[2], color.original_lch[2]),
                1e-9,
            )
            self.assertTrue(is_in_srgb_gamut(lch_to_lab(color.adjusted_lch)))
        self.assertTrue(result[1].chroma_was_reduced)

    def test_single_color_keeps_its_original_lch(self) -> None:
        rgb = (80, 120, 160)

        result = redistribute_palette_lightness((rgb,))

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].original_rgb, rgb)
        expected_lch = rgb_to_lch(rgb)
        for actual, expected in zip(result[0].adjusted_lch, expected_lch):
            self.assertAlmostEqual(actual, expected, places=10)

    def test_rejects_an_empty_palette(self) -> None:
        with self.assertRaisesRegex(ValueError, "At least one"):
            redistribute_palette_lightness(())

    def test_rejects_invalid_rgb_channels(self) -> None:
        with self.assertRaisesRegex(ValueError, "RGB integer triples"):
            redistribute_palette_lightness(((0, 20, 300),))


if __name__ == "__main__":
    unittest.main()
