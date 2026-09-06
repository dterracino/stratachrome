"""Integration tests for color_tools-driven palette and optical behavior."""

from __future__ import annotations

import unittest

import numpy as np
from PIL import Image
from color_tools import FilamentCollections

from stratachrome.color_engine import (
    build_tier_image,
    extract_perceptual_lab,
    plan_tier_colors,
    sample_tier_lab,
    select_tier_palette,
)
from stratachrome.optical_model import (
    ColorLayerMapper,
    assignments_from_layer_counts,
    optimize_tier_schedule,
    simulate_tier_stack,
)
from stratachrome.depth_mapper import TwoTierDepthMapper
from stratachrome.pipeline import _filament_display_name


class ColorToolsIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.collection = FilamentCollections.BAMBU_PLA_BASICMATTE

    def _filament(self, color: str):
        return next(record for record in self.collection if record.color == color)

    def test_default_collection_is_printable_and_has_td(self) -> None:
        self.assertTrue(self.collection)
        self.assertTrue(all(record.maker == "Bambu Lab" for record in self.collection))
        self.assertEqual({record.finish for record in self.collection}, {"Basic", "Matte"})
        self.assertTrue(
            all(record.td_value is not None and record.td_value > 0 for record in self.collection)
        )

    def test_schedule_display_name_includes_type_and_finish(self) -> None:
        sky_blue = self._filament("Sky Blue")

        self.assertEqual(
            _filament_display_name(sky_blue),
            "Bambu Lab PLA Matte Sky Blue",
        )

    def test_tier_masks_are_complementary(self) -> None:
        image = Image.new("RGB", (4, 2), (10, 20, 30))
        matte = np.asarray([[0.0, 0.49, 0.5, 1.0]] * 2, dtype=np.float32)

        background = np.asarray(build_tier_image(image, matte, foreground=False))
        foreground = np.asarray(build_tier_image(image, matte, foreground=True))

        np.testing.assert_array_equal(background[..., 3] > 0, matte < 0.5)
        np.testing.assert_array_equal(foreground[..., 3] > 0, matte >= 0.5)
        np.testing.assert_array_equal(
            (background[..., 3] > 0) | (foreground[..., 3] > 0),
            np.ones_like(matte, dtype=bool),
        )

    def test_palette_selection_uses_unique_filaments_sorted_by_lightness(self) -> None:
        pixels = np.zeros((32, 32, 3), dtype=np.uint8)
        pixels[:, :16] = (5, 5, 5)
        pixels[:, 16:] = (240, 235, 220)
        image = Image.fromarray(pixels, mode="RGB")
        matte = np.zeros((32, 32), dtype=np.float32)

        palette = select_tier_palette(
            image,
            matte,
            foreground=False,
            color_count=2,
        )

        self.assertEqual(len(palette.filaments), 2)
        self.assertEqual(len({record.id for record in palette.filaments}), 2)
        self.assertEqual(len({record.rgb for record in palette.filaments}), 2)
        self.assertEqual(
            list(palette.filaments),
            sorted(palette.filaments, key=lambda record: record.lab[0]),
        )

    def test_schedule_uses_td_and_returns_derived_layer_count(self) -> None:
        black = self._filament("Black")
        white = self._filament("Jade White")
        targets = np.asarray([black.lab, white.lab], dtype=np.float64)

        schedule = optimize_tier_schedule(
            (white, black),
            targets,
            max_layers_per_filament=20,
        )

        self.assertEqual(schedule.assignments[0].filament.id, black.id)
        self.assertEqual(len(schedule.states), sum(schedule.layer_counts))
        self.assertGreaterEqual(schedule.layer_counts[1], 1)
        self.assertLessEqual(schedule.layer_counts[1], 20)
        self.assertAlmostEqual(
            schedule.states[-1].height_mm,
            0.2 + 0.1 * (len(schedule.states) - 1),
        )

    def test_tier_planner_selects_palette_size_and_schedule_together(self) -> None:
        pixels = np.zeros((24, 24, 3), dtype=np.uint8)
        pixels[:, :12] = (10, 40, 100)
        pixels[:, 12:] = (235, 220, 190)
        image = Image.fromarray(pixels, mode="RGB")
        lab = extract_perceptual_lab(image)
        matte = np.zeros((24, 24), dtype=np.float32)

        plan = plan_tier_colors(
            image,
            lab,
            matte,
            foreground=False,
            max_colors=4,
            max_layers_per_filament=12,
        )

        self.assertGreaterEqual(len(plan.palette.filaments), 2)
        self.assertLessEqual(len(plan.palette.filaments), 4)
        self.assertEqual(
            len(plan.schedule.states),
            sum(plan.schedule.layer_counts),
        )
        self.assertGreater(plan.selection_score, 0.0)

    def test_layer_mapping_uses_full_lab_distance(self) -> None:
        red = self._filament("Red")
        blue = self._filament("Blue")
        assignments = assignments_from_layer_counts(
            sorted((red, blue), key=lambda record: record.lab[0]),
            (1, 1),
        )
        states = simulate_tier_stack(assignments, total_layers=2)
        mapper = ColorLayerMapper(states)
        targets = np.asarray([[states[0].simulated_lab, states[1].simulated_lab]])

        layers = mapper.map_image_lab_to_layers(targets)

        np.testing.assert_array_equal(layers, np.asarray([[0, 1]], dtype=np.int32))

    def test_depth_mapper_builds_swaps_from_native_filament_records(self) -> None:
        black = self._filament("Black")
        white = self._filament("Jade White")
        assignments = assignments_from_layer_counts((black, white), (1, 1))
        states = simulate_tier_stack(assignments, total_layers=2)
        mapper = ColorLayerMapper(states)
        lab_image = np.asarray([[states[0].simulated_lab, states[1].simulated_lab]])
        matte = np.zeros((1, 2), dtype=np.float32)
        depth_mapper = TwoTierDepthMapper(mapper, mapper, states, states)

        result = depth_mapper.generate_heightmap(lab_image, lab_image, matte)

        self.assertEqual(len(result.swap_schedule), 4)
        self.assertTrue(
            all(swap.filament_name.startswith("Bambu Lab ") for swap in result.swap_schedule)
        )
        self.assertTrue(all(swap.filament_hex.startswith("#") for swap in result.swap_schedule))

    def test_lab_sampling_is_tier_specific_and_bounded(self) -> None:
        image = Image.new("RGB", (20, 10), (120, 30, 200))
        lab = extract_perceptual_lab(image)
        matte = np.zeros((10, 20), dtype=np.float32)
        matte[:, 10:] = 1.0

        background = sample_tier_lab(lab, matte, foreground=False, max_samples=12)
        foreground = sample_tier_lab(lab, matte, foreground=True, max_samples=7)

        self.assertEqual(background.shape, (12, 3))
        self.assertEqual(foreground.shape, (7, 3))


if __name__ == "__main__":
    unittest.main()
