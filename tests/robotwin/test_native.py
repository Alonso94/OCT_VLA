import pytest

from oct_vla.robots.robotwin.native import NativePortError, _load_task_class


class DummyExternalTask:
    pass


def test_load_task_class_resolves_external_entrypoint_bypassing_envs():
    loaded = _load_task_class(f"{__name__}:DummyExternalTask")
    assert loaded is DummyExternalTask


def test_load_task_class_reports_missing_external_module():
    with pytest.raises(NativePortError):
        _load_task_class("oct_vla.does_not_exist:Whatever")


def test_load_task_class_reports_missing_external_class():
    with pytest.raises(NativePortError):
        _load_task_class(f"{__name__}:DoesNotExist")


def test_load_task_class_reports_missing_native_task():
    with pytest.raises(NativePortError):
        _load_task_class("no_such_robotwin_task")
