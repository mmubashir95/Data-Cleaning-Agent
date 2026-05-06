"""Helpers for assembling simple cleaning reports."""

from pathlib import Path


def generate_report(report_text: str, output_path: str | Path) -> Path:
    """Write a plain-text report to disk and return its path."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report_text, encoding="utf-8")
    return path
