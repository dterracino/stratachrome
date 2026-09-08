"""Shared input-derived output path handling for Stratachrome CLIs."""

from __future__ import annotations

from pathlib import Path

DEFAULT_OUTPUT_DIRECTORY = Path("output")


def resolve_output_file(
    input_path: Path,
    output: Path | None,
    name_suffix: str,
    extension: str,
) -> Path:
    """Resolve a single output file from CLI input and output arguments."""
    default_name = f"{input_path.stem}_{name_suffix}{extension}"
    if output is None:
        return DEFAULT_OUTPUT_DIRECTORY / default_name
    if (output.exists() and output.is_dir()) or not output.suffix:
        return output / default_name
    if output.parent == Path("."):
        return input_path.parent / output.name
    return output


def resolve_output_directory(output: Path | None) -> Path:
    """Resolve the directory used by a multi-file CLI."""
    return DEFAULT_OUTPUT_DIRECTORY if output is None else output
