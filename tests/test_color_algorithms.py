"""Tests for pluggable palette generation and TD-planning integration."""

from __future__ import annotations

import unittest

import numpy as np
from PIL import Image
from color_tools import FilamentCollections

from stratachrome.color_algorithms import (
    ColorPaletteAlgorithm,
    ExtractedPaletteColor,
    KMeansColorAlgorithm,
    MedianCutColorAlgorithm,
    get_color_algorithm,
)
from stratachrome.color_engine import extract_perceptual_lab, plan_tier_colors


class _FixtureColorAlgorithm(ColorPaletteAlgorithm):
    key = "fixture"
    display_name = "fixture colors"

    def __init__(self) -> None:
        self.call_count = 0

    def _extract_colors(
        self,
        image: Image.Image,
        color_count: int,
        *,
        matte: np.ndarray | None,
        foreground: bool,
    ) -> tuple[ExtractedPaletteColor, ...]:
        self.call_count += 1
        return (
            ExtractedPaletteColor((235, 220, 190), 0.20),
            ExtractedPaletteColor((10, 40, 100), 0.50),
            ExtractedPaletteColor((160, 20, 30), 0.30),
        )


class ColorAlgorithmTests(unittest.TestCase):
    def test_all_algorithms_share_the_base_lightness_redistribution(self) -> None:
        image = Image.new("RGB", (2, 2))
        algorithm = _FixtureColorAlgorithm()

        palette = algorithm.generate_palette(image, 3)

        lightness = [color.adjusted_lch[0] for color in palette.colors]
        self.assertEqual(lightness, sorted(lightness))
        self.assertAlmostEqual(
            lightness[1],
            lightness[0] + (lightness[2] - lightness[0]) / 2.0,
            places=10,
        )

    def test_dominant_adapter_restores_population_priority(self) -> None:
        palette = _FixtureColorAlgorithm().generate_palette(Image.new("RGB", (1, 1)), 3)

        targets = palette.as_dominant_colors()

        self.assertEqual([target.population for target in targets], [0.50, 0.30, 0.20])

    def test_registry_returns_both_concrete_algorithms(self) -> None:
        self.assertIsInstance(get_color_algorithm("median-cut"), MedianCutColorAlgorithm)
        self.assertIsInstance(get_color_algorithm("kmeans"), KMeansColorAlgorithm)

    def test_generated_palette_flows_into_td_schedule_planning(self) -> None:
        algorithm = _FixtureColorAlgorithm()
        image = Image.new("RGB", (12, 12), (90, 90, 90))
        matte = np.zeros((12, 12), dtype=np.float32)
        collection = tuple(
            filament
            for filament in FilamentCollections.BAMBU_PLA_BASICMATTE
            if filament.color in {"Black", "Red", "Jade White"}
        )

        plan = plan_tier_colors(
            image,
            extract_perceptual_lab(image),
            matte,
            foreground=False,
            max_colors=3,
            max_layers_per_tier=20,
            collection=collection,
            color_algorithm=algorithm,
        )

        self.assertEqual(algorithm.call_count, 1)
        self.assertEqual(
            len(plan.schedule.assignments),
            len(plan.palette.filaments),
        )
        self.assertEqual(
            len(plan.schedule.states),
            sum(plan.schedule.layer_counts),
        )
        self.assertTrue(all(count >= 1 for count in plan.schedule.layer_counts))
        target_lightness = sorted(match.target.lab[0] for match in plan.palette.matches)
        self.assertEqual(len(target_lightness), 3)
        self.assertAlmostEqual(
            target_lightness[1],
            target_lightness[0]
            + (target_lightness[2] - target_lightness[0]) / 2.0,
            places=10,
        )


if __name__ == "__main__":
    unittest.main()
