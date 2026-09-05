"""
segment_cli.py
--------------
Command-line utility to test and visualize foreground/background separation.
Extracts the matte using BiRefNet via segmentation.py and exports isolated
RGBA images for both zones with real-time status feedback.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from pathlib import Path
import sys
import threading
import time
from typing import Generator

import numpy as np
from PIL import Image
import torch

from stratachrome.segmentation import ForegroundSegmenter, SegmentationConfig


@contextmanager
def _status_spinner(message: str) -> Generator[None, None, None]:
    """Displays a live console spinner with elapsed time during blocking tasks."""
    spinner_chars = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    stop_flag = threading.Event()
    start_time = time.perf_counter()

    def _spin() -> None:
        idx = 0
        while not stop_flag.is_set():
            elapsed = time.perf_counter() - start_time
            glyph = spinner_chars[idx % len(spinner_chars)]
            sys.stdout.write(f"\r  {glyph} {message} ({elapsed:4.1f}s)")
            sys.stdout.flush()
            idx += 1
            time.sleep(0.08)

    spin_thread = threading.Thread(target=_spin, daemon=True)
    spin_thread.start()
    try:
        yield
    finally:
        stop_flag.set()
        spin_thread.join()
        total_time = time.perf_counter() - start_time
        sys.stdout.write(f"\r  ✓ {message} ({total_time:.2f}s)\n")
        sys.stdout.flush()


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stratachrome: Extract foreground and background layers from an input image."
    )
    parser.add_argument(
        "-i", "--input",
        type=Path,
        required=True,
        help="Path to the input image file.",
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=Path("output"),
        help="Directory where segmented images will be saved (default: ./output).",
    )
    parser.add_argument(
        "--feather",
        type=int,
        default=2,
        help="Edge feathering radius in pixels (default: 2).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional binarization cut-off (0.0 to 1.0) before feathering.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Compute device ('cuda', 'cpu'). Auto-detected if omitted.",
    )
    parser.add_argument(
        "--save-matte",
        action="store_true",
        help="Also export the raw grayscale alpha matte image.",
    )
    return parser.parse_args()


def _load_image(path: Path) -> Image.Image:
    if not path.is_file():
        raise FileNotFoundError(f"Input image not found: {path}")
    with Image.open(path) as img:
        return img.convert("RGB")


def _apply_alpha_matte(rgb_image: Image.Image, alpha_channel: np.ndarray) -> Image.Image:
    rgb_arr = np.array(rgb_image, dtype=np.uint8)
    alpha_uint8 = (np.clip(alpha_channel, 0.0, 1.0) * 255.0).astype(np.uint8)
    rgba_arr = np.dstack((rgb_arr, alpha_uint8))
    return Image.fromarray(rgba_arr, mode="RGBA")


def _export_matte_preview(matte: np.ndarray, output_path: Path) -> None:
    matte_uint8 = (np.clip(matte, 0.0, 1.0) * 255.0).astype(np.uint8)
    img = Image.fromarray(matte_uint8, mode="L")
    img.save(output_path)


def _log_hardware_diagnostics(target_device: torch.device) -> None:
    """Prints active runtime compute device specifications."""
    if target_device.type == "cuda" and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(target_device)
        mem_info = torch.cuda.get_device_properties(target_device).total_memory / (1024**3)
        print(f"[Hardware] Compute Target: CUDA -> {gpu_name} ({mem_info:.1f} GB VRAM)")
    else:
        print("[Hardware] Compute Target: CPU (Note: transformer inference will be slower without CUDA)")


def main() -> int:
    args = _parse_arguments()

    try:
        source_image = _load_image(args.input)
    except Exception as err:
        sys.stderr.write(f"Error loading input image: {err}\n")
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.input.stem

    config = SegmentationConfig(
        device=args.device,
        feather_radius=args.feather,
        threshold=args.threshold,
    )

    segmenter = ForegroundSegmenter(config=config)
    _log_hardware_diagnostics(segmenter.device)

    print(f"[1/4] Preparing input image '{args.input.name}' ({source_image.width}x{source_image.height})...")

    with _status_spinner("Loading BiRefNet neural network weights into memory"):
        # Explicitly warm up / load model weights
        segmenter._ensure_model_loaded()

    with _status_spinner("Running deep bilateral segmentation inference (1024x1024 patch grid)"):
        result = segmenter.extract_matte(source_image)

    print("[3/4] Processing alpha boundaries and partitioning color plates...")
    fg_image = _apply_alpha_matte(source_image, result.matte)
    fg_path = args.output_dir / f"{stem}_foreground.png"
    fg_image.save(fg_path)

    bg_matte = 1.0 - result.matte
    bg_image = _apply_alpha_matte(source_image, bg_matte)
    bg_path = args.output_dir / f"{stem}_background.png"
    bg_image.save(bg_path)

    if args.save_matte:
        matte_path = args.output_dir / f"{stem}_matte.png"
        _export_matte_preview(result.matte, matte_path)
        print(f"      - Matte layer       -> {matte_path}")

    print(f"      - Foreground layer  -> {fg_path}")
    print(f"      - Background layer  -> {bg_path}")
    print("[4/4] Segmentation completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())