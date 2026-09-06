"""Compatibility runner for the Bambu project exporter."""

from __future__ import annotations

from stratachrome import pipeline


def main() -> int:
    """Run the standard pipeline, which now owns native Bambu export."""
    return pipeline.main()


if __name__ == "__main__":
    raise SystemExit(main())
