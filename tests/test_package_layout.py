from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "frost_analysis"


def test_modules_are_grouped_by_responsibility() -> None:
    expected = {
        "pipelines": {"prepare.py", "process.py"},
        "data": {
            "alignment.py",
            "cycles.py",
            "images.py",
            "inventory.py",
            "registry.py",
            "sensors.py",
        },
        "processing": {"baseline.py", "features.py", "missing.py", "resample.py"},
        "analysis": {"correlation.py", "screening.py"},
        "core": {"artifacts.py", "validation.py"},
    }
    for directory, files in expected.items():
        assert {path.name for path in (PACKAGE / directory).glob("*.py")} >= files
    assert not any((PACKAGE / name).is_file() for name in ("prepare.py", "process.py"))


def test_core_has_no_business_imports() -> None:
    forbidden = ("pipelines", "data", "processing", "analysis")
    for path in (PACKAGE / "core").glob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert not any(f"frost_analysis.{name}" in text for name in forbidden)
