from pathlib import Path


def test_core_directories_exist():
    root = Path(__file__).resolve().parents[1]
    for directory in ("configs", "data", "docs", "environment", "results", "scripts", "src"):
        assert (root / directory).is_dir(), directory
