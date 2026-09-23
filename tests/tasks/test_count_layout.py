"""The count-shift profiles' x layouts, without a simulator.

`robotwin_env` imports RoboTwin's `envs` package at module scope; the sampler
under test touches none of it, so stub modules stand in.
"""

import random
import sys
import types

import pytest


@pytest.fixture(scope="module")
def env_module():
    stubs = {
        "envs": types.ModuleType("envs"),
        "envs._base_task": types.ModuleType("envs._base_task"),
        "envs.utils": types.ModuleType("envs.utils"),
    }
    stubs["envs._base_task"].Base_Task = type("Base_Task", (), {})
    stubs["envs.utils"].create_actor = stubs["envs.utils"].create_box = None
    saved = {name: sys.modules.get(name) for name in stubs}
    sys.modules.update(stubs)
    sys.modules.pop("oct_vla.tasks.shelf_restock.robotwin_env", None)
    try:
        import oct_vla.tasks.shelf_restock.robotwin_env as module
    except ImportError as error:  # a further simulator-only import
        pytest.skip(f"robotwin_env not importable without RoboTwin: {error}")
    yield module
    sys.modules.pop("oct_vla.tasks.shelf_restock.robotwin_env", None)
    for name, value in saved.items():
        if value is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = value


def _positions(module, task_class, seed):
    task = task_class.__new__(task_class)
    return task._spaced_x_positions(task_class.spec, random.Random(seed))


def test_three_object_draws_are_unchanged(env_module):
    """The training profile's layout must not move: collections in flight and
    every recorded corpus depend on these exact draws."""
    spec = env_module.ShelfRestockTask.spec
    low, high = spec.object_variation.position_x_range
    for seed in range(50):
        rng = random.Random(seed)
        free = (high - low) - env_module.MIN_OBJECT_SEPARATION * 2
        offsets = sorted(rng.uniform(0.0, free) for _ in range(3))
        expected = [low + o + i * env_module.MIN_OBJECT_SEPARATION for i, o in enumerate(offsets)]
        rng.shuffle(expected)
        assert _positions(env_module, env_module.ShelfRestockTask, seed) == expected


def test_two_object_first_target_matches_three_object(env_module):
    two = env_module.ShelfRestockTwoObjectTask
    three = env_module.ShelfRestockTask
    assert two.layout == "nested_in_3" and three.layout == "independent"
    firsts_two = [min(_positions(env_module, two, s)) for s in range(2000)]
    firsts_three = [min(_positions(env_module, three, s)) for s in range(2000)]
    # Same seed, same leading draws: the first target is identical per seed.
    assert firsts_two == firsts_three
    spec = three.spec
    low, _ = spec.object_variation.position_x_range
    assert max(firsts_two) < low + 0.17 + 1e-9


def test_two_object_layout_keeps_separation(env_module):
    for seed in range(500):
        xs = sorted(_positions(env_module, env_module.ShelfRestockTwoObjectTask, seed))
        assert len(xs) == 2
        assert xs[1] - xs[0] >= env_module.MIN_OBJECT_SEPARATION - 1e-12
