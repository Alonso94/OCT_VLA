import json
from pathlib import Path

import pytest

from oct_vla.config import read_env_file, resolve_paths


def test_precedence_and_file_relative_paths(tmp_path):
    folder = tmp_path / "configuration"
    folder.mkdir()
    config = folder / "paths.json"
    config.write_text(
        json.dumps(
            {"robotwin_root": "sim", "data_root": "data", "policy_python": "policy/bin/python"}
        )
    )
    env = folder / ".env"
    env.write_text('OCTVLA_DATA_ROOT="file data"\n')
    resolved = resolve_paths(
        {"data_root": "cli"},
        config=config,
        env_file=env,
        environ={"OCTVLA_DATA_ROOT": "environment"},
        cwd=tmp_path,
    )
    assert resolved.data_root == tmp_path / "cli"
    assert resolved.robotwin_root == folder / "sim"
    assert resolved.robotwin_python == folder / "sim/.venv-robotwin/bin/python"
    assert resolved.asset_root == folder / "sim/assets"
    assert resolved.policy_python == folder / "policy/bin/python"
    assert (
        resolve_paths({}, config=config, env_file=env, environ={}, cwd=tmp_path).data_root
        == folder / "file data"
    )
    assert (
        resolve_paths(
            {},
            config=config,
            env_file=env,
            environ={"OCTVLA_DATA_ROOT": "environment"},
            cwd=tmp_path,
        ).data_root
        == tmp_path / "environment"
    )
    assert resolve_paths({}, config=config, environ={}, cwd=tmp_path).data_root == folder / "data"


def test_interpreter_symlink_preserved(tmp_path):
    executable = tmp_path / "python-base"
    executable.touch()
    link = tmp_path / "venv-python"
    link.symlink_to(executable)
    assert resolve_paths({"policy_python": str(link)}, environ={}).policy_python == link


@pytest.mark.parametrize(
    "contents", ["[]", '{"typo": "value"}', '{"data_root": 3}', '{"data_root": ""}', "not json"]
)
def test_bad_config(tmp_path, contents):
    config = tmp_path / "paths.json"
    config.write_text(contents)
    with pytest.raises(ValueError):
        resolve_paths({}, config=config, environ={})


def test_env_is_literal_not_shell(tmp_path):
    env = tmp_path / ".env"
    env.write_text("# ignored\nexport OCTVLA_DATA_ROOT='$(touch SENTINEL)' # comment\n")
    assert read_env_file(env)["OCTVLA_DATA_ROOT"] == "$(touch SENTINEL)"
    resolved = resolve_paths({}, env_file=env, environ={})
    assert resolved.data_root == tmp_path / "$(touch SENTINEL)"
    assert not (tmp_path / "SENTINEL").exists()


@pytest.mark.parametrize(
    "line",
    [
        "UNKNOWN=x",
        "OCTVLA_DATA_ROOT=",
        "OCTVLA_DATA_ROOT=a b",
        "source another-file",
        "OCTVLA_DATA_ROOT='unclosed",
    ],
)
def test_invalid_env_assignments(tmp_path, line):
    env = tmp_path / ".env"
    env.write_text(line)
    with pytest.raises(ValueError):
        read_env_file(env)


def test_does_not_implicitly_load_dotenv_or_create_paths(tmp_path):
    (tmp_path / ".env").write_text("ROBOTWIN_ROOT=somewhere")
    paths = resolve_paths({"output_root": "new/output"}, environ={}, cwd=tmp_path)
    assert paths.robotwin_root is None
    assert not paths.output_root.exists()
    assert paths.as_dict()["output_root"] == str(tmp_path / "new/output")


def test_relative_config_filename(tmp_path):
    (tmp_path / "paths.json").write_text('{"data_root": "data"}')
    assert (
        resolve_paths({}, config=Path("paths.json"), environ={}, cwd=tmp_path).data_root
        == tmp_path / "data"
    )
