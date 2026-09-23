"""scripts/check_stage2_init.py must stay parseable by LeRobot's `parser.wrap`."""

import ast
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_stage2_init.py"


def test_no_postponed_annotations():
    # parser.wrap reads the wrapped function's annotation as a class; under
    # postponed evaluation it is a string and every INIT_CHECK dies at parse.
    tree = ast.parse(SCRIPT.read_text())
    future = [n for n in tree.body if isinstance(n, ast.ImportFrom) and n.module == "__future__"]
    assert not any(a.name == "annotations" for n in future for a in n.names)
