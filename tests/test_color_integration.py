"""Integration tests for color_tools-driven palette and optical behavior."""

from __future__ import annotations

from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
from color_tools import FilamentCollections
from color_tools.image import DominantColor

import stratachrome.color_engine as color_engine
from stratachrome.color_diagnostics import save_color_diagnostics
from stratachrome.color_histograms import render_rgb_histogram
from stratachrome.color_engine import (
    FilamentMatch,
    TierPalette,
    allocate_lookahead_tier_layers,
    build_tier_image,
    extract_perceptual_lab,
    plan_geometry_first_tier_colors,
    plan_tier_colors,
    sample_tier_lab,
    select_tier_palette,
)
from stratachrome.optical_model import (
    ColorLayerMapper,
    assignments_from_layer_counts,
    optimize_geometry_first_schedule,
    optimize_tier_schedule,
    simulate_tier_stack,
)
from stratachrome.depth_mapper import (
    GeometryFirstTwoTierDepthMapper,
    SingleTierDepthMapper,
    TwoTierDepthMapper,
)
from stratachrome.pipeline import (
    _filament_display_name,
    _parse_arguments,
    _warn_palette_lightness_collisions,
)


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

    def test_lookahead_split_preserves_matching_optical_demand(self) -> None:
        self.assertEqual(allocate_lookahead_tier_layers(27, 12, 15), (12, 15))

    def test_lookahead_split_scales_demand_to_exact_total(self) -> None:
        background, foreground = allocate_lookahead_tier_layers(27, 15, 17)

        self.assertEqual((background, foreground), (13, 14))
        self.assertEqual(background + foreground, 27)

    def test_lookahead_cli_defaults_to_twenty_seven_total_layers(self) -> None:
        with patch("sys.argv", ["stratachrome", "input.jpg", "--mapping-mode", "lookahead"]):
            args = _parse_arguments()

        self.assertEqual(args.total_layers, 27)

    def test_lookahead_cli_accepts_total_layers(self) -> None:
        with patch(
            "sys.argv",
            [
                "stratachrome",
                "input.jpg",
                "--mapping-mode",
                "lookahead",
                "--total-layers",
                "27",
            ],
        ):
            args = _parse_arguments()

        self.assertEqual(args.mapping_mode, "lookahead")
        self.assertEqual(args.total_layers, 27)

    def test_total_layers_is_ignored_outside_lookahead_mode(self) -> None:
        with patch(
            "sys.argv",
            ["stratachrome", "input.jpg", "--total-layers", "31"],
        ):
            args = _parse_arguments()

        self.assertEqual(args.mapping_mode, "optical")
        self.assertEqual(args.total_layers, 31)

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
        self.assertEqual(selected, {"Black", "Dark Gray", "Jade White"})
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

    def test_palette_quantizes_twice_the_sixteen_filament_cap(self) -> None:
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
                color_count=16,
            )

        self.assertEqual(quantize.call_args.kwargs["color_count"], 32)
        self.assertLessEqual(len(palette.filaments), 16)

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

    def test_geometry_first_schedule_preserves_fixed_layer_count(self) -> None:
        black = self._filament("Black")
        white = self._filament("Jade White")
        targets = np.asarray([black.lab] * 2 + [white.lab] * 6, dtype=np.float64)
        geometry_layers = np.arange(8, dtype=np.int32)

        schedule = optimize_geometry_first_schedule(
            (white, black),
            targets,
            geometry_layers,
            total_layers=8,
        )

        self.assertEqual(sum(schedule.layer_counts), 8)
        self.assertEqual(len(schedule.states), 8)
        self.assertEqual(schedule.assignments[0].filament.id, black.id)
        self.assertGreater(schedule.layer_counts[1], schedule.layer_counts[0])

    def test_geometry_first_schedule_moves_color_boundary_within_fixed_space(self) -> None:
        black = self._filament("Black")
        white = self._filament("Jade White")
        targets = np.asarray([black.lab] * 2 + [white.lab] * 6, dtype=np.float64)

        schedule = optimize_geometry_first_schedule(
            (black, white),
            targets,
            np.arange(8, dtype=np.int32),
            total_layers=8,
        )

        self.assertEqual(schedule.assignments[1].start_layer, schedule.layer_counts[0])
        self.assertLess(schedule.assignments[1].start_layer, 4)
        self.assertEqual(sum(schedule.layer_counts), 8)

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

    def test_palette_lightness_warning_reports_only_same_height_layer(self) -> None:
        def target(rgb: tuple[int, int, int], lightness: float) -> DominantColor:
            return DominantColor(
                rgb=rgb,
                lab=(lightness, 0.0, 0.0),
                population=0.3,
                dominance=0.3,
                global_salience=0.0,
                local_contrast=0.0,
                spatial_distribution=0.0,
                spatial_coherence=0.0,
                lightness_contrast=0.0,
                focal_importance=0.0,
            )

        palette = TierPalette(
            (
                FilamentMatch(target((10, 20, 30), 10.0), self._filament("Black"), 1.0),
                FilamentMatch(target((40, 50, 60), 11.0), self._filament("Jade White"), 1.0),
                FilamentMatch(target((70, 80, 90), 30.0), self._filament("Blue"), 1.0),
            )
        )
        lab_image = np.asarray([[[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]]])
        matte = np.zeros((1, 2), dtype=np.float32)
        stderr = StringIO()

        with redirect_stderr(stderr):
            _warn_palette_lightness_collisions(
                "Background",
                palette,
                lab_image,
                matte,
                foreground=False,
                layer_count=10,
            )

        warning = stderr.getvalue()
        self.assertIn("#0A141E (L*=10.00", warning)
        self.assertIn("#28323C (L*=11.00", warning)
        self.assertIn("same height layer 2/10", warning)
        self.assertNotIn("#46505A", warning)

    def test_geometry_first_tier_planner_fills_precomputed_layer_space(self) -> None:
        black = self._filament("Black")
        white = self._filament("Jade White")
        pixels = np.asarray(
            [[[20, 20, 20], [80, 80, 80], [160, 160, 160], [240, 240, 240]]],
            dtype=np.uint8,
        )
        image = Image.fromarray(pixels, mode="RGB")
        lab = extract_perceptual_lab(image)
        matte = np.zeros((1, 4), dtype=np.float32)
        geometry_layers = np.asarray([[0, 2, 5, 7]], dtype=np.int32)

        plan = plan_geometry_first_tier_colors(
            image,
            lab,
            matte,
            geometry_layers,
            foreground=False,
            total_layers=8,
            max_colors=2,
            collection=(black, white),
        )

        self.assertEqual(sum(plan.schedule.layer_counts), 8)
        self.assertEqual(len(plan.schedule.states), 8)
        self.assertEqual(len(plan.palette.filaments), 2)

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

    def test_geometry_first_two_tier_geometry_is_independent_of_chroma(self) -> None:
        black = self._filament("Black")
        white = self._filament("Jade White")
        states = simulate_tier_stack(
            assignments_from_layer_counts((black, white), (1, 2)),
            total_layers=3,
        )
        neutral_lab = np.asarray(
            [
                [
                    [20.0, 0.0, 0.0],
                    [80.0, 0.0, 0.0],
                    [20.0, 0.0, 0.0],
                    [80.0, 0.0, 0.0],
                ]
            ],
            dtype=np.float64,
        )
        chromatic_lab = np.asarray(
            [
                [
                    [20.0, 90.0, -70.0],
                    [80.0, -90.0, 70.0],
                    [20.0, -90.0, -70.0],
                    [80.0, 90.0, 70.0],
                ]
            ],
            dtype=np.float64,
        )
        matte = np.asarray([[0.0, 0.0, 1.0, 1.0]], dtype=np.float32)
        mapper = GeometryFirstTwoTierDepthMapper(states, states)

        neutral = mapper.generate_heightmap(neutral_lab, neutral_lab, matte)
        chromatic = mapper.generate_heightmap(chromatic_lab, chromatic_lab, matte)

        np.testing.assert_array_equal(neutral.z_grid, chromatic.z_grid)
        np.testing.assert_allclose(neutral.z_grid, [[0.2, 0.4, 0.5, 0.7]])

    def test_color_diagnostics_reconstruct_exact_optical_states(self) -> None:
        black = self._filament("Black")
        white = self._filament("Jade White")
        states = simulate_tier_stack(
            assignments_from_layer_counts((black, white), (1, 1)),
            total_layers=2,
        )
        targets = np.asarray([[states[0].simulated_lab, states[1].simulated_lab]])

        diagnostics = TwoTierDepthMapper(states, states).generate_color_diagnostics(
            targets,
            targets,
            np.asarray([[0.0, 1.0]], dtype=np.float32),
        )

        np.testing.assert_array_equal(
            diagnostics.preview_rgb,
            np.asarray([[states[0].simulated_rgb, states[1].simulated_rgb]], dtype=np.uint8),
        )
        np.testing.assert_allclose(diagnostics.delta_e, 0.0, atol=1e-5)
        self.assertAlmostEqual(diagnostics.mean_delta_e, 0.0)

        with tempfile.TemporaryDirectory() as directory:
            source_rgb = np.asarray([[[12, 34, 56], [78, 90, 123]]], dtype=np.uint8)
            (
                preview_path,
                heatmap_path,
                source_64px_path,
                preview_64px_path,
                source_histogram_path,
                preview_histogram_path,
            ) = save_color_diagnostics(
                diagnostics,
                Path(directory) / "sample.3mf",
                source_rgb,
            )

            self.assertEqual(preview_path.name, "sample_color_preview.png")
            self.assertEqual(heatmap_path.name, "sample_delta_e_heatmap.png")
            self.assertEqual(source_64px_path.name, "sample_source_64px.png")
            self.assertEqual(preview_64px_path.name, "sample_color_preview_64px.png")
            self.assertEqual(source_histogram_path.name, "sample_source_histogram.png")
            self.assertEqual(
                preview_histogram_path.name,
                "sample_color_preview_histogram.png",
            )
            with Image.open(preview_path) as preview:
                np.testing.assert_array_equal(np.asarray(preview), diagnostics.preview_rgb)
            with Image.open(heatmap_path) as heatmap:
                np.testing.assert_array_equal(
                    np.asarray(heatmap),
                    np.zeros((1, 2, 3), dtype=np.uint8),
                )
            with Image.open(source_64px_path) as source_64px:
                self.assertEqual(source_64px.size, (64, 32))
                np.testing.assert_array_equal(np.asarray(source_64px)[0, 0], source_rgb[0, 0])
            with Image.open(preview_64px_path) as preview_64px:
                self.assertEqual(preview_64px.size, (64, 32))
                np.testing.assert_array_equal(
                    np.asarray(preview_64px)[-1, -1],
                    diagnostics.preview_rgb[0, 1],
                )
            with Image.open(source_histogram_path) as histogram:
                self.assertEqual(histogram.size, (768, 432))
            with Image.open(preview_histogram_path) as histogram:
                self.assertEqual(histogram.size, (768, 432))

    def test_color_histogram_renders_three_continuous_channel_charts(self) -> None:
        source_rgb = np.stack(
            (
                np.repeat((0, 64, 255), (128, 64, 64)),
                np.repeat((32, 128, 224), (64, 128, 64)),
                np.repeat((16, 96, 192), (96, 96, 64)),
            ),
            axis=1,
        )
        source_rgb = source_rgb[None, ...].astype(np.uint8)

        rendered = render_rgb_histogram(source_rgb)

        channel_colors = ((210, 45, 55), (30, 155, 80), (35, 90, 210))
        chart_ranges = ((24, 133), (149, 258), (274, 383))
        for color, (chart_top, chart_bottom) in zip(channel_colors, chart_ranges):
            curve = np.all(rendered == color, axis=2)
            occupied_rows = np.flatnonzero(curve.any(axis=1))
            self.assertGreaterEqual(occupied_rows.min(), chart_top)
            self.assertLessEqual(occupied_rows.max(), chart_bottom)
            self.assertTrue(curve[:, 56:744].any(axis=0).all())
            for column in range(57, 744):
                previous_rows = np.flatnonzero(curve[:, column - 1])
                current_rows = np.flatnonzero(curve[:, column])
                self.assertLessEqual(current_rows.min(), previous_rows.max() + 1)
                self.assertLessEqual(previous_rows.min(), current_rows.max() + 1)

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
