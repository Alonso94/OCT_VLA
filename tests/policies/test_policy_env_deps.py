"""The policy interpreter must carry every backbone's optional dependencies.

No unit test builds a GR00T action head (it needs the 3B checkpoint), so a
policy venv without ``lerobot[groot]`` passed the whole suite and failed only
inside a GPU job (``diffusers`` missing, 2026-11-10). This catches it here.
"""

import importlib.util

import pytest

pytest.importorskip("torch")
pytest.importorskip("lerobot")

# lerobot[groot] extra: diffusers (flow head), peft, timm, dm-tree, decord.
GROOT_DEPS = ("diffusers", "peft", "timm", "tree", "decord")


@pytest.mark.parametrize("module", GROOT_DEPS)
def test_groot_dependency_installed(module):
    assert importlib.util.find_spec(module) is not None, (
        f"{module!r} missing from the policy venv; install lerobot[groot]"
    )
