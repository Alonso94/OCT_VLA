"""Check the package boundary without simulator or learning dependencies."""

import subprocess
import sys
from pathlib import Path

import pytest


def test_package_import_is_dependency_free(tmp_path: Path):
    source = Path(__file__).resolve().parents[1] / "src"
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-B",
            "-c",
            "import pathlib, sys; "
            "sys.path.insert(0, sys.argv[1]); "
            "import oct_vla; "
            "assert pathlib.Path(oct_vla.__file__).resolve() == "
            "pathlib.Path(sys.argv[1]) / 'oct_vla' / '__init__.py'",
            str(source),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    if result.returncode:
        pytest.fail(result.stderr)
    assert result.stdout == ""
    assert result.stderr == ""
