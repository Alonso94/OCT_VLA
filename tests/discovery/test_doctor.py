import json
import subprocess
import sys
from pathlib import Path

import pytest

from oct_vla import cli, doctor
from oct_vla.config import ExternalPaths


@pytest.mark.parametrize("layout", ["task_config", "env_cfg/task_config"])
def test_robotwin_layouts(tmp_path, layout):
    (tmp_path / "envs").mkdir()
    registry = tmp_path / layout
    registry.mkdir(parents=True)
    for name in ("_embodiment_config.yml", "_camera_config.yml"):
        (registry / name).touch()
    assert doctor.task_config_dir(tmp_path) == registry
    (registry / "_camera_config.yml").unlink()
    with pytest.raises(ValueError):
        doctor.task_config_dir(tmp_path)


def test_real_probe_uses_selected_interpreter_without_importing_heavy_packages():
    probe = Path(doctor.__file__).with_name("_probe.py")
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            str(probe),
            json.dumps(
                {
                    "json": "nonexistent-distribution-octvla-test",
                    "octvla_missing_module": "octvla_missing_module",
                }
            ),
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    payload = json.loads(result.stdout)
    assert payload["packages"]["json"]["origin"]
    assert payload["packages"]["octvla_missing_module"]["origin"] is None
    assert payload["executable"] == sys.executable


def test_missing_roots_report_failures_without_creation(tmp_path, monkeypatch):
    monkeypatch.setattr(doctor, "_run", lambda *args: "Test GPU, driver")
    paths = ExternalPaths(data_root=tmp_path / "absent", output_root=tmp_path / "new/output")
    report = doctor.diagnose(paths)
    assert report.exit_code == 1
    assert {c.name for c in report.checks if c.status == "FAIL"} >= {
        "RoboTwin",
        "Dataset root",
        "Asset root",
        "Simulator",
        "Policy",
    }
    assert not paths.output_root.exists()


def test_stale_module_resolution_is_not_hidden_by_version(monkeypatch, tmp_path):
    payload = {
        "python": "3.10.18",
        "executable": "sim/bin/python",
        "packages": {"curobo": {"origin": None, "version": "0.0.0", "revision": None}},
        "stale_paths": [{"file": "editable.pth", "target": "/missing/source"}],
    }
    monkeypatch.setattr(doctor, "_run", lambda *args: json.dumps(payload))
    checks = doctor._environment("Simulator", tmp_path / "python", {}, "3.10", 1)
    assert any(c.name == "Simulator/curobo" and c.status == "FAIL" for c in checks)
    assert any("missing/source" in c.detail and c.status == "WARN" for c in checks)


def test_timeout_is_actionable(monkeypatch):
    def timeout(command, **kwargs):
        assert kwargs["timeout"] == 0.1
        assert kwargs["env"]["GIT_OPTIONAL_LOCKS"] == "0"
        raise subprocess.TimeoutExpired(command, 0.1)

    monkeypatch.setattr(subprocess, "run", timeout)
    with pytest.raises(ValueError, match="timed out"):
        doctor._run(["anything"], 0.1)


def test_cli_json_and_exit_status(monkeypatch, capsys):
    report = doctor.Report(ExternalPaths(), (doctor.Check("test", "FAIL", "missing"),))
    monkeypatch.setattr(cli, "diagnose", lambda paths: report)
    assert cli.main(["doctor", "--json"]) == 1
    assert json.loads(capsys.readouterr().out)["exit_code"] == 1


def test_cli_invalid_configuration_is_usage_error(tmp_path):
    with pytest.raises(SystemExit) as error:
        cli.main(["doctor", "--config", str(tmp_path / "missing.json")])
    assert error.value.code == 2


def test_complete_discovery_reports_success_without_runtime_claim(tmp_path, monkeypatch):
    root = tmp_path / "sim"
    (root / "envs").mkdir(parents=True)
    (root / "assets").mkdir()
    registry = root / "task_config"
    registry.mkdir()
    for name in ("_embodiment_config.yml", "_camera_config.yml"):
        (registry / name).touch()

    def run(command, timeout):
        if command[0] == "nvidia-smi":
            return "Test GPU, driver"
        packages = json.loads(command[-1])
        return json.dumps(
            {
                "python": "3.10.18" if "sapien" in packages else "3.12.3",
                "executable": command[0],
                "stale_paths": [],
                "packages": {
                    name: {"origin": "source.py", "version": "1", "revision": None}
                    for name in packages
                },
            }
        )

    monkeypatch.setattr(doctor, "_run", run)
    report = doctor.diagnose(
        ExternalPaths(
            robotwin_root=root,
            asset_root=root / "assets",
            data_root=tmp_path,
            robotwin_python=Path("sim-python"),
            policy_python=Path("policy-python"),
        )
    )
    assert report.exit_code == 0
    assert any(c.name == "RoboTwin revision" and c.status == "WARN" for c in report.checks)
    assert "not tested" in report.checks[-1].detail
