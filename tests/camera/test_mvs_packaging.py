from __future__ import annotations

from pathlib import Path


def test_native_mvs_adapter_is_declared_as_an_installed_python_module():
    root = Path(__file__).parents[2]
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")

    assert 'py-modules = ["ev_vision_mvs_adapter"]' in pyproject
