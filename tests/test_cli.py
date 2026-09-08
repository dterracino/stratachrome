"""Tests for consistent command-line arguments across Stratachrome tools."""

from __future__ import annotations

from contextlib import nullcontext, redirect_stderr
from io import StringIO
from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

import stratachrome.segment_cli as segment_cli
from stratachrome.pipeline import _parse_arguments as parse_main_arguments
from stratachrome.segment_cli import _parse_arguments as parse_segment_arguments
from stratachrome.stl_test_cli import _parse_arguments as parse_mesh_arguments


class CliConsistencyTests(unittest.TestCase):
    def test_geometry_clis_share_equivalent_defaults(self) -> None:
        with patch("sys.argv", ["stratachrome", "input.png"]):
            main_args = parse_main_arguments()
        with patch("sys.argv", ["stratachrome-mesh", "input.png"]):
            mesh_args = parse_mesh_arguments()

        self.assertEqual(main_args.size, mesh_args.size)
        self.assertEqual(main_args.max_dim, mesh_args.max_dim)
        self.assertEqual(main_args.first_layer, mesh_args.first_layer)
        self.assertEqual(
            (main_args.size, main_args.max_dim, main_args.first_layer), (200.0, 1000, 0.2)
        )

    def test_main_uses_requested_generation_defaults(self) -> None:
        with patch("sys.argv", ["stratachrome", "input.png"]):
            args = parse_main_arguments()

        self.assertEqual(args.colors_per_tier, 4)
        self.assertEqual(args.tier_mode, "dual")
        self.assertEqual(args.swap_mode, "auto")
        self.assertEqual(args.device, "auto")
        self.assertEqual(args.max_layers_per_tier, 20)
        self.assertEqual(args.total_layers, 27)

    def test_main_accepts_sixteen_colors_per_tier(self) -> None:
        with patch("sys.argv", ["stratachrome", "input.png", "--colors-per-tier", "16"]):
            args = parse_main_arguments()

        self.assertEqual(args.colors_per_tier, 16)

    def test_segment_uses_canonical_output_destination(self) -> None:
        with patch(
            "sys.argv",
            ["stratachrome-segment", "input.png", "--output", "previews"],
        ):
            args = parse_segment_arguments()

        self.assertEqual(args.output, Path("previews"))
        self.assertNotIn("output_dir", vars(args))

    def test_segment_main_creates_canonical_output_directory(self) -> None:
        with TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "previews"
            image = Image.new("RGB", (128, 64), (10, 20, 30))
            matte = np.zeros((64, 128), dtype=np.float32)
            matte[:, 64:] = 1.0
            result = Mock(matte=matte)
            segmenter = Mock(device="cpu")
            segmenter.extract_matte.return_value = result
            arguments = Namespace(
                input=Path("input.png"),
                output=output,
                device="auto",
                feather=2,
                threshold=None,
                colors_per_tier=4,
                save_matte=True,
                save_quantized=True,
                save_colormaps=True,
                save_histograms=True,
            )

            with (
                patch.object(segment_cli, "_parse_arguments", return_value=arguments),
                patch.object(segment_cli, "_load_image", return_value=image),
                patch.object(segment_cli, "ForegroundSegmenter", return_value=segmenter),
                patch.object(segment_cli, "_log_hardware_diagnostics"),
                patch.object(segment_cli, "_status_spinner", return_value=nullcontext()),
            ):
                exit_code = segment_cli.main()

            self.assertEqual(exit_code, 0)
            self.assertTrue((output / "input_foreground.png").is_file())
            self.assertTrue((output / "input_background.png").is_file())
            self.assertTrue((output / "input_matte.png").is_file())
            self.assertTrue((output / "input_background_colormap.png").is_file())
            self.assertTrue((output / "input_foreground_colormap.png").is_file())
            self.assertTrue((output / "input_background_quantized.png").is_file())
            self.assertTrue((output / "input_foreground_quantized.png").is_file())
            self.assertTrue((output / "input_background_histogram.png").is_file())
            self.assertTrue((output / "input_foreground_histogram.png").is_file())
            self.assertTrue((output / "input_background_quantized_histogram.png").is_file())
            self.assertTrue((output / "input_foreground_quantized_histogram.png").is_file())
            with Image.open(output / "input_background_colormap.png") as colormap:
                self.assertEqual(colormap.size, (64, 32))
            with Image.open(output / "input_background_quantized.png") as quantized:
                self.assertEqual(quantized.size, image.size)

    def test_segment_uses_auto_device_and_shared_color_default(self) -> None:
        with patch("sys.argv", ["stratachrome-segment", "input.png"]):
            args = parse_segment_arguments()

        self.assertEqual(args.device, "auto")
        self.assertEqual(args.colors_per_tier, 4)
        self.assertFalse(args.save_quantized)
        self.assertFalse(args.save_colormaps)
        self.assertFalse(args.save_histograms)

    def test_omitted_outputs_use_input_derived_names_in_output_directory(self) -> None:
        with patch("sys.argv", ["stratachrome", "images/photo.jpg"]):
            main_args = parse_main_arguments()
        with patch("sys.argv", ["stratachrome-mesh", "images/photo.jpg"]):
            mesh_args = parse_mesh_arguments()
        with patch("sys.argv", ["stratachrome-segment", "images/photo.jpg"]):
            segment_args = parse_segment_arguments()

        self.assertEqual(main_args.output, Path("output/photo_stratachrome.3mf"))
        self.assertEqual(mesh_args.output, Path("output/photo_mesh.stl"))
        self.assertEqual(segment_args.output, Path("output"))

    def test_output_directories_receive_input_derived_filenames(self) -> None:
        with patch("sys.argv", ["stratachrome", "images/photo.jpg", "--output", "exports"]):
            main_args = parse_main_arguments()
        with patch("sys.argv", ["stratachrome-mesh", "images/photo.jpg", "--output", "exports"]):
            mesh_args = parse_mesh_arguments()

        self.assertEqual(main_args.output, Path("exports/photo_stratachrome.3mf"))
        self.assertEqual(mesh_args.output, Path("exports/photo_mesh.stl"))

    def test_bare_output_filenames_are_placed_beside_input(self) -> None:
        with patch("sys.argv", ["stratachrome", "images/photo.jpg", "--output", "custom.3mf"]):
            main_args = parse_main_arguments()
        with patch("sys.argv", ["stratachrome-mesh", "images/photo.jpg", "--output", "custom.stl"]):
            mesh_args = parse_mesh_arguments()

        self.assertEqual(main_args.output, Path("images/custom.3mf"))
        self.assertEqual(mesh_args.output, Path("images/custom.stl"))

    def test_explicit_output_file_paths_are_preserved(self) -> None:
        with patch(
            "sys.argv",
            ["stratachrome", "images/photo.jpg", "--output", "exports/custom.3mf"],
        ):
            main_args = parse_main_arguments()
        with patch(
            "sys.argv",
            ["stratachrome-mesh", "images/photo.jpg", "--output", "exports/custom.stl"],
        ):
            mesh_args = parse_mesh_arguments()

        self.assertEqual(main_args.output, Path("exports/custom.3mf"))
        self.assertEqual(mesh_args.output, Path("exports/custom.stl"))

    def test_existing_dotted_output_directory_receives_derived_filename(self) -> None:
        with TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "exports.v1"
            output.mkdir()

            with patch("sys.argv", ["stratachrome", "images/photo.jpg", "--output", str(output)]):
                args = parse_main_arguments()

            self.assertEqual(args.output, output / "photo_stratachrome.3mf")

    def test_mesh_uses_canonical_geometry_destinations(self) -> None:
        with patch(
            "sys.argv",
            [
                "stratachrome-mesh",
                "input.png",
                "--first-layer",
                "0.24",
                "--max-dim",
                "512",
            ],
        ):
            args = parse_mesh_arguments()

        self.assertEqual(args.first_layer, 0.24)
        self.assertEqual(args.max_dim, 512)
        self.assertNotIn("base_height", vars(args))
        self.assertNotIn("max_dimension", vars(args))

    def test_removed_argument_names_are_rejected(self) -> None:
        invocations = (
            (
                parse_segment_arguments,
                ["stratachrome-segment", "input.png", "--output-dir", "previews"],
            ),
            (
                parse_mesh_arguments,
                ["stratachrome-mesh", "input.png", "--base-height", "0.2"],
            ),
            (
                parse_mesh_arguments,
                ["stratachrome-mesh", "input.png", "--max-dimension", "512"],
            ),
            (parse_main_arguments, ["stratachrome", "--input", "input.png"]),
            (parse_segment_arguments, ["stratachrome-segment", "-i", "input.png"]),
            (parse_mesh_arguments, ["stratachrome-mesh", "--input", "input.png"]),
            (parse_main_arguments, ["stratachrome", "input.png", "--tier-mode", "two"]),
            (parse_main_arguments, ["stratachrome", "input.png", "--swap-mode", "ams"]),
        )

        for parser, argv in invocations:
            with self.subTest(argument=argv[-2]):
                with patch("sys.argv", argv), redirect_stderr(StringIO()):
                    with self.assertRaises(SystemExit) as raised:
                        parser()
                self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
