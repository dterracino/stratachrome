"""Compatibility runner for the Bambu project exporter."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Callable

from stratachrome.bambu_exporter import export_bambu_project
from stratachrome import pipeline


def _pop_path_option(name: str) -> Path | None:
    try:
        index = sys.argv.index(name)
    except ValueError:
        return None
    if index + 1 >= len(sys.argv):
        raise ValueError(f"{name} requires a path.")
    value = Path(sys.argv[index + 1])
    del sys.argv[index : index + 2]
    return value


def _float_option(name: str, default: float) -> float:
    try:
        index = sys.argv.index(name)
    except ValueError:
        return default
    if index + 1 >= len(sys.argv):
        raise ValueError(f"{name} requires a number.")
    return float(sys.argv[index + 1])


def main() -> int:
    """Run the existing pipeline while routing packaging to the new exporter."""
    template = _pop_path_option("--bambu-template")
    step_height = _float_option("--layer-height", 0.10)
    first_layer = _float_option("--first-layer", 0.20)

    def export_with_profile(mesh, swaps, output, swap_mode="ams") -> None:
        export_bambu_project(
            mesh,
            swaps,
            output,
            swap_mode,
            step_height_mm=step_height,
            first_layer_height_mm=first_layer,
            template_3mf=template,
        )

    pipeline.export_bambu_3mf = export_with_profile
    return pipeline.main()


if __name__ == "__main__":
    raise SystemExit(main())
