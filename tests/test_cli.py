"""Tests for the Click CLI (LM-20 / LM-21)."""

from __future__ import annotations

import numpy as np
import pytest
from click.testing import CliRunner

import lithomaker
import lithomaker.cli as cli_module
from lithomaker.cli import cli

# Small geometry keeps each conversion well under a second.
FAST = ["--size-x", "10", "--dot-size", "1", "--max-layers", "12"]

runner = CliRunner()


def run(*args: str):
    return runner.invoke(cli, [str(arg) for arg in args])


@pytest.fixture
def image(tmp_path, make_image_file) -> str:
    path = make_image_file(np.linspace(0, 1, 100, dtype=np.float32).reshape(10, 10), "photo.png")
    return str(path)


# --------------------------------------------------------------------------
# --help / --version / info (LM-21)
# --------------------------------------------------------------------------
def test_group_help_lists_commands():
    result = run("--help")

    assert result.exit_code == 0
    assert "Usage:" in result.output
    assert "convert" in result.output
    assert "gui" in result.output
    assert "info" in result.output


def test_convert_help_documents_every_option():
    result = run("convert", "--help")

    assert result.exit_code == 0
    for flag in (
        "-i, --input",
        "-o, --output",
        "-b, --black",
        "-w, --white",
        "-g, --gamma",
        "--levels",
        "--auto-levels",
        "--min-layers",
        "--max-layers",
        "--thickness",
        "--size-x",
        "--size-y",
        "--dot-size",
        "--base-thickness",
        "--ascii",
        "--format",
        "--force",
        "--verbose",
        "--stats",
    ):
        assert flag in result.output, flag
    assert "INPUT_IMAGE" in result.output
    assert "OUTPUT.STL" in result.output


def test_version_comes_from_the_package():
    result = run("--version")

    assert result.exit_code == 0
    assert lithomaker.__version__ in result.output


def test_info_reports_versions():
    result = run("info")

    assert result.exit_code == 0
    assert "lithomaker" in result.output
    assert "python" in result.output
    assert "pillow" in result.output
    assert "numpy" in result.output


# --------------------------------------------------------------------------
# happy path (LM-20)
# --------------------------------------------------------------------------
def test_positional_arguments_write_the_stl(image, tmp_path):
    output = tmp_path / "out.stl"

    result = run("convert", image, output, *FAST)

    assert result.exit_code == 0, result.output
    assert output.is_file()
    assert result.stdout.splitlines()[0] == str(output)


def test_input_output_aliases_work(image, tmp_path):
    output = tmp_path / "aliased.stl"

    result = run("convert", "-i", image, "-o", output, *FAST)

    assert result.exit_code == 0, result.output
    assert output.is_file()


def test_output_defaults_to_the_input_stem(image, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    result = run("convert", "photo.png", *FAST)

    assert result.exit_code == 0, result.output
    assert (tmp_path / "photo.stl").is_file()
    assert result.stdout.splitlines()[0] == "photo.stl"


def test_stats_table_is_printed_by_default(image, tmp_path):
    result = run("convert", image, tmp_path / "out.stl", *FAST)

    assert result.exit_code == 0
    assert "vertices" in result.stdout
    assert "triangles" in result.stdout
    assert "z range" in result.stdout
    assert "footprint" in result.stdout
    assert "file size" in result.stdout


def test_no_stats_leaves_only_the_path_on_stdout(image, tmp_path):
    result = run("convert", image, tmp_path / "out.stl", "--no-stats", *FAST)

    assert result.exit_code == 0
    assert result.stdout.splitlines() == [str(tmp_path / "out.stl")]


def test_ascii_flag_writes_ascii_stl(image, tmp_path):
    output = tmp_path / "out.stl"

    result = run("convert", image, output, "--ascii", *FAST)

    assert result.exit_code == 0
    assert output.read_text(encoding="ascii").startswith("solid ")
    assert "ASCII STL" in result.stdout


def test_format_long_form_is_equivalent_to_ascii_flag(image, tmp_path):
    output = tmp_path / "out.stl"

    result = run("convert", image, output, "--format", "ascii", *FAST)

    assert result.exit_code == 0
    assert output.read_text(encoding="ascii").startswith("solid ")


def test_binary_is_the_default(image, tmp_path):
    output = tmp_path / "out.stl"

    run("convert", image, output, *FAST)

    data = output.read_bytes()
    assert not data.lower().startswith(b"solid")
    assert len(data) > 84


def test_explicit_binary_format_is_accepted(image, tmp_path):
    result = run("convert", image, tmp_path / "out.stl", "--binary", "--format", "binary", *FAST)

    assert result.exit_code == 0


def test_verbose_streams_progress_to_stderr(image, tmp_path):
    result = run("convert", image, tmp_path / "out.stl", "-v", *FAST)

    assert result.exit_code == 0
    assert "loading image" in result.stderr
    assert "[100%] done" in result.stderr
    assert "loading image" not in result.stdout


def test_progress_is_silent_without_verbose(image, tmp_path):
    result = run("convert", image, tmp_path / "out.stl", *FAST)

    assert "loading image" not in result.stderr
    assert "[100%]" not in result.stderr


def test_warnings_go_to_stderr(image, tmp_path):
    result = run("convert", image, tmp_path / "out.stl", "--size-y", "40", *FAST)

    assert result.exit_code == 0
    assert "warning:" in result.stderr
    assert "stretched" not in result.stdout


# --------------------------------------------------------------------------
# positional / flag conflicts and usage errors, exit code 2 (D5)
# --------------------------------------------------------------------------
def test_missing_input_is_a_usage_error():
    result = run("convert")

    assert result.exit_code == 2
    assert "missing required argument" in result.output


def test_positional_and_flag_input_conflict(image, tmp_path):
    result = run("convert", image, tmp_path / "out.stl", "-i", "other.png", *FAST)

    assert result.exit_code == 2
    assert "both given" in result.output


def test_positional_and_flag_output_conflict(image, tmp_path):
    result = run("convert", image, tmp_path / "out.stl", "-o", "other.stl", *FAST)

    assert result.exit_code == 2
    assert "both given" in result.output


def test_ascii_and_format_conflict(image, tmp_path):
    result = run("convert", image, tmp_path / "out.stl", "--ascii", "--format", "binary", *FAST)

    assert result.exit_code == 2
    assert "disagree" in result.output


def test_levels_and_auto_flag_conflict(image, tmp_path):
    result = run("convert", image, tmp_path / "out.stl", "--levels", "auto", "--no-auto-levels")

    assert result.exit_code == 2
    assert "disagree" in result.output


@pytest.mark.parametrize(
    "options, expected",
    [
        (["--gamma", "0"], "gamma"),
        (["--thickness", "0"], "thickness"),
        (["--dot-size", "0"], "dot size"),
        (["--base-thickness", "-1"], "base thickness"),
        (["--min-layers", "10", "--max-layers", "5"], "max layers"),
        (["--black", "0.8", "--white", "0.2"], "black point"),
        (["--size-x", "0"], "size X"),
    ],
)
def test_invalid_parameters_are_usage_errors(image, tmp_path, options, expected):
    result = run("convert", image, tmp_path / "out.stl", *options)

    assert result.exit_code == 2
    assert expected in result.output
    assert not (tmp_path / "out.stl").exists()


def test_mesh_size_guard_is_a_usage_error(image, tmp_path):
    result = run("convert", image, tmp_path / "out.stl", "--size-x", "400", "--dot-size", "0.05")

    assert result.exit_code == 2
    assert "mesh too large" in result.output


def test_usage_errors_never_print_a_traceback(image, tmp_path):
    result = run("convert", image, tmp_path / "out.stl", "--gamma", "0")

    assert "Traceback" not in result.output


# --------------------------------------------------------------------------
# runtime errors, exit code 1
# --------------------------------------------------------------------------
def test_missing_image_is_a_runtime_error(tmp_path):
    result = run("convert", tmp_path / "nope.png", tmp_path / "out.stl")

    assert result.exit_code == 1
    assert "not found" in result.output
    assert "Traceback" not in result.output


def test_corrupt_image_is_a_runtime_error(tmp_path):
    path = tmp_path / "broken.png"
    path.write_bytes(b"nope")

    result = run("convert", path, tmp_path / "out.stl")

    assert result.exit_code == 1
    assert "could not read image" in result.output


def test_overwrite_is_refused_without_force(image, tmp_path):
    output = tmp_path / "out.stl"
    output.write_bytes(b"precious")

    result = run("convert", image, output, *FAST)

    assert result.exit_code == 1
    assert "refusing to overwrite" in result.output
    assert "--force" in result.output
    assert output.read_bytes() == b"precious"


def test_force_overwrites_the_output(image, tmp_path):
    output = tmp_path / "out.stl"
    output.write_bytes(b"stale")

    result = run("convert", image, output, "--force", *FAST)

    assert result.exit_code == 0, result.output
    assert output.read_bytes()[:5] != b"stale"


def test_unexpected_errors_are_reported_without_a_traceback(image, tmp_path, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("caboom")

    monkeypatch.setattr(cli_module, "convert", boom)

    result = run("convert", image, tmp_path / "out.stl", *FAST)

    assert result.exit_code == 1
    assert "caboom" in result.output
    assert "Traceback" not in result.output


def test_os_errors_become_one_line_failures(image, tmp_path, monkeypatch):
    def boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(cli_module, "convert", boom)

    result = run("convert", image, tmp_path / "out.stl", *FAST)

    assert result.exit_code == 1
    assert "disk full" in result.output
    assert "Traceback" not in result.output


def test_verbose_reraises_unexpected_errors(image, tmp_path, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("caboom")

    monkeypatch.setattr(cli_module, "convert", boom)

    result = run("convert", image, tmp_path / "out.stl", "-v", *FAST)

    assert isinstance(result.exception, RuntimeError)


# --------------------------------------------------------------------------
# levels switches (D7)
# --------------------------------------------------------------------------
def test_auto_levels_flag_converts_successfully(image, tmp_path):
    result = run("convert", image, tmp_path / "out.stl", "--auto-levels", *FAST)

    assert result.exit_code == 0


def test_auto_levels_with_manual_levels_warns(image, tmp_path):
    result = run("convert", image, tmp_path / "out.stl", "--auto-levels", "-b", "0.2", *FAST)

    assert result.exit_code == 0
    assert "auto levels" in result.stderr


def test_levels_auto_choice_works(image, tmp_path):
    result = run("convert", image, tmp_path / "out.stl", "--levels", "auto", *FAST)

    assert result.exit_code == 0
    assert "warning" not in result.stderr


def test_no_auto_levels_is_explicit_manual(image, tmp_path):
    result = run("convert", image, tmp_path / "out.stl", "--no-auto-levels", *FAST)

    assert result.exit_code == 0


def test_gui_subcommand_launches_the_gui(monkeypatch):
    import lithomaker.gui as gui_module

    calls = {}
    monkeypatch.setattr(gui_module, "main", lambda: calls.setdefault("ran", True))

    result = run("gui")

    assert result.exit_code == 0, result.output
    assert calls["ran"] is True


# --------------------------------------------------------------------------
# console-script entry point
# --------------------------------------------------------------------------
def test_main_uses_the_same_command(image, tmp_path, monkeypatch):
    calls = {}

    def fake_main(*args, **kwargs):
        calls["prog_name"] = kwargs.get("prog_name")

    monkeypatch.setattr(cli_module, "cli", fake_main)

    cli_module.main()

    assert calls["prog_name"] == "lithomaker"


def test_entry_point_is_declared_in_pyproject():
    from pathlib import Path

    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    text = pyproject.read_text()

    assert 'lithomaker = "lithomaker.cli:main"' in text


def test_convert_uses_the_shared_core(image, tmp_path, monkeypatch):
    """The CLI must not grow its own pipeline: it calls lithomaker.core.convert."""
    calls = {}
    original = cli_module.convert

    def spy(image_path, out_path, config, *, overwrite=False, progress=None):
        calls["image"] = str(image_path)
        calls["config"] = config
        return original(image_path, out_path, config, overwrite=overwrite, progress=progress)

    monkeypatch.setattr(cli_module, "convert", spy)

    result = run("convert", image, tmp_path / "out.stl", "--black", "0.1", *FAST)

    assert result.exit_code == 0
    assert calls["image"] == image
    assert calls["config"].levels.black == pytest.approx(0.1)


def test_cli_module_imports_without_numpy_heavy_side_effects():
    assert cli_module.cli.name == "cli"


def test_module_entry_point_runs():
    import subprocess
    import sys

    completed = subprocess.run(
        [sys.executable, "-m", "lithomaker.cli", "--version"],
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert lithomaker.__version__ in completed.stdout
