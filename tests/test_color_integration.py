"""Integration tests for color_tools-driven palette and optical behavior."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
from color_tools import FilamentCollections

import stratachrome.color_engine as color_engine
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
from stratachrome.depth_mapper import SingleTierDepthMapper, TwoTierDepthMapper
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

    def test_population_coverage_beats_a_small_salient_highlight(self) -> None:
        pixels = np.full((40, 40, 3), (150, 150, 150), dtype=np.uint8)
        pixels[:8, :] = (245, 245, 245)
        pixels[8:12, :] = (0, 0, 0)
        pixels[12, :16] = (255, 210, 0)
        image = Image.fromarray(pixels, mode="RGB")
        matte = np.ones((40, 40), dtype=np.float32)

        palette = select_tier_palette(
            image,
            matte,
            foreground=True,
            color_count=3,
        )

        selected = {filament.color for filament in palette.filaments}
        self.assertEqual(selected, {"Black", "Ash Gray", "Jade White"})
        self.assertEqual(palette.filaments[0].color, "Black")

    def test_palette_selection_can_use_eight_unique_colors(self) -> None:
        filaments = tuple(
            self._filament(color)
            for color in (
                "Black",
                "Jade White",
                "Red",
                "Bambu Green",
                "Blue",
                "Sunflower Yellow",
                "Ash Gray",
                "Pumpkin Orange",
            )
        )
        pixels = np.empty((24, 80, 3), dtype=np.uint8)
        for index, filament in enumerate(filaments):
            pixels[:, index * 10 : (index + 1) * 10] = filament.rgb

        palette = select_tier_palette(
            Image.fromarray(pixels, mode="RGB"),
            np.ones((24, 80), dtype=np.float32),
            foreground=True,
            color_count=8,
            collection=filaments,
        )

        self.assertEqual(len(palette.filaments), 8)
        self.assertEqual(len({filament.id for filament in palette.filaments}), 8)

    def test_eight_color_ceiling_does_not_force_eight_colors(self) -> None:
        palette = select_tier_palette(
            Image.new("RGB", (24, 24), (250, 250, 248)),
            np.ones((24, 24), dtype=np.float32),
            foreground=True,
            color_count=8,
        )

        self.assertEqual(len(palette.filaments), 1)
        self.assertEqual(palette.filaments[0].color, "Jade White")

    def test_palette_quantizes_twice_the_requested_filament_cap(self) -> None:
        image = Image.new("RGB", (24, 24), (120, 120, 120))
        matte = np.ones((24, 24), dtype=np.float32)

        with patch(
            "stratachrome.color_engine._quantized_tier_colors",
            wraps=color_engine._quantized_tier_colors,
        ) as quantize:
            palette = select_tier_palette(
                image,
                matte,
                foreground=True,
                color_count=4,
            )

        self.assertEqual(quantize.call_args.kwargs["color_count"], 8)
        self.assertLessEqual(len(palette.filaments), 4)

    def test_palette_prefers_a_competitive_reusable_filament(self) -> None:
        jade = self._filament("Jade White")
        ivory = self._filament("Ivory White")
        image = Image.new("RGB", (24, 24), (250, 250, 248))

        palette = select_tier_palette(
            image,
            np.ones((24, 24), dtype=np.float32),
            foreground=True,
            color_count=1,
            collection=(jade, ivory),
            preferred_filaments=(ivory,),
        )

        self.assertEqual(palette.filaments, (ivory,))

    def test_reuse_does_not_force_a_visibly_wrong_filament(self) -> None:
        red = self._filament("Red")
        white = self._filament("Jade White")

        palette = select_tier_palette(
            Image.new("RGB", (24, 24), red.rgb),
            np.ones((24, 24), dtype=np.float32),
            foreground=True,
            color_count=1,
            collection=(red, white),
            preferred_filaments=(white,),
        )

        self.assertEqual(palette.filaments, (red,))

    def test_schedule_uses_td_and_returns_derived_layer_count(self) -> None:
        black = self._filament("Black")
        white = self._filament("Jade White")
        targets = np.asarray([black.lab, white.lab], dtype=np.float64)

        schedule = optimize_tier_schedule(
            (white, black),
            targets,
            max_layers_per_tier=20,
        )

        self.assertEqual(schedule.assignments[0].filament.id, black.id)
        self.assertEqual(len(schedule.states), sum(schedule.layer_counts))
        self.assertGreaterEqual(schedule.layer_counts[1], 4)
        self.assertEqual(sum(schedule.layer_counts), 20)
        self.assertAlmostEqual(
            schedule.states[-1].height_mm,
            0.2 + 0.1 * (len(schedule.states) - 1),
        )

    def test_td_scale_controls_optical_transition_thickness(self) -> None:
        black = self._filament("Black")
        white = self._filament("Jade White")
        targets = np.asarray([black.lab, white.lab], dtype=np.float64)

        catalog_schedule = optimize_tier_schedule(
            (black, white),
            targets,
            max_layers_per_tier=20,
        )
        frontlit_schedule = optimize_tier_schedule(
            (black, white),
            targets,
            max_layers_per_tier=20,
            td_scale=0.1,
        )

        self.assertLess(
            sum(frontlit_schedule.layer_counts),
            sum(catalog_schedule.layer_counts),
        )

    def test_tier_budget_rejects_td_minimums_that_do_not_fit(self) -> None:
        black = self._filament("Black")
        white = self._filament("Jade White")
        targets = np.asarray([black.lab, white.lab], dtype=np.float64)

        with self.assertRaisesRegex(ValueError, "cannot satisfy the TD minimum"):
            optimize_tier_schedule(
                (black, white),
                targets,
                initial_substrate_lab=black.lab,
                max_layers_per_tier=2,
            )

    def test_low_td_color_stops_before_the_tier_budget(self) -> None:
        black = self._filament("Black")
        white = self._filament("Jade White")

        schedule = optimize_tier_schedule(
            (black,),
            np.asarray([black.lab], dtype=np.float64),
            initial_substrate_lab=white.lab,
            max_layers_per_tier=20,
        )

        self.assertEqual(black.td_value, 0.1)
        self.assertLess(sum(schedule.layer_counts), 20)

    def test_tier_planner_retains_palette_when_td_minimums_fit(self) -> None:
        pixels = np.zeros((24, 24, 3), dtype=np.uint8)
        pixels[:12, :12] = (10, 40, 100)
        pixels[:12, 12:] = (235, 220, 190)
        pixels[12:, :12] = (160, 20, 30)
        pixels[12:, 12:] = (20, 150, 60)
        image = Image.fromarray(pixels, mode="RGB")
        lab = extract_perceptual_lab(image)
        matte = np.zeros((24, 24), dtype=np.float32)

        plan = plan_tier_colors(
            image,
            lab,
            matte,
            foreground=False,
            max_colors=4,
            max_layers_per_tier=20,
        )

        self.assertEqual(len(plan.palette.filaments), 4)
        self.assertEqual(
            len(plan.schedule.assignments),
            len(plan.palette.filaments),
        )
        self.assertTrue(all(count >= 1 for count in plan.schedule.layer_counts))
        self.assertEqual(
            len(plan.schedule.states),
            sum(plan.schedule.layer_counts),
        )
        self.assertLessEqual(len(plan.schedule.states), 20)
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

    def test_black_pixels_stop_below_the_next_color_boundary(self) -> None:
        black = self._filament("Black")
        white = self._filament("Jade White")
        assignments = assignments_from_layer_counts((black, white), (2, 4))
        states = simulate_tier_stack(
            assignments,
            total_layers=6,
            first_layer_height_mm=0.10,
            initial_substrate_lab=white.lab,
        )
        mapper = ColorLayerMapper(states)

        black_layer = int(
            mapper.map_image_lab_to_layers(np.asarray([[black.lab]], dtype=np.float64))[0, 0]
        )

        self.assertLess(black_layer, assignments[1].start_layer)

    def test_depth_mapper_builds_swaps_from_native_filament_records(self) -> None:
        black = self._filament("Black")
        white = self._filament("Jade White")
        assignments = assignments_from_layer_counts((black, white), (1, 1))
        states = simulate_tier_stack(assignments, total_layers=2)
        lab_image = np.asarray([[states[0].simulated_lab, states[1].simulated_lab]])
        matte = np.asarray([[0.0, 1.0]], dtype=np.float32)
        depth_mapper = TwoTierDepthMapper(states, states)

        result = depth_mapper.generate_heightmap(lab_image, lab_image, matte)

        self.assertEqual(len(result.swap_schedule), 4)
        self.assertEqual(result.z_grid[0, 0], result.bg_surface_z[0, 0])
        self.assertEqual(result.z_grid[0, 1], result.fg_surface_z[0, 1])
        np.testing.assert_allclose(
            (result.z_grid - 0.2) / 0.1,
            np.rint((result.z_grid - 0.2) / 0.1),
            atol=1e-5,
        )
        self.assertTrue(
            all(swap.filament_name.startswith("Bambu Lab ") for swap in result.swap_schedule)
        )
        self.assertTrue(all(swap.filament_hex.startswith("#") for swap in result.swap_schedule))

    def test_two_tier_depth_blends_feathered_boundaries(self) -> None:
        black = self._filament("Black")
        white = self._filament("Jade White")
        states = simulate_tier_stack(
            assignments_from_layer_counts((black, white), (1, 1)),
            total_layers=2,
        )
        target = np.asarray([[states[0].simulated_lab]])

        result = TwoTierDepthMapper(states, states).generate_heightmap(
            target,
            target,
            np.asarray([[0.5]], dtype=np.float32),
        )

        self.assertAlmostEqual(float(result.z_grid[0, 0]), 0.3)

    def test_single_tier_depth_maps_lightness_monotonically_to_layer_grid(self) -> None:
        black = self._filament("Black")
        white = self._filament("Jade White")
        states = simulate_tier_stack(
            assignments_from_layer_counts((black, white), (1, 3)),
            total_layers=4,
        )
        lab_image = np.asarray(
            [[[10.0, 80.0, 70.0], [50.0, -80.0, -70.0], [90.0, 0.0, 0.0]]],
            dtype=np.float64,
        )

        result = SingleTierDepthMapper(states).generate_heightmap(lab_image)

        np.testing.assert_allclose(result.z_grid, [[0.2, 0.4, 0.5]])
        self.assertEqual(result.total_layers, 4)
        self.assertEqual({swap.tier_name for swap in result.swap_schedule}, {"single"})

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
