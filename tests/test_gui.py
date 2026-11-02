"""Tests for the tkinter frontend (LM-30 … LM-33).

The pure helpers and the parity map run everywhere. Widget tests need a display and are
skipped when ``DISPLAY`` is unset, so CI on a headless runner stays green (LM-41).
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

import lithomaker.gui as gui_module
from lithomaker.cli import PARAMETER_ALIASES, PRESENTATION_OPTIONS, convert_command
from lithomaker.config import ConfigError
from lithomaker.core import Heightmap
from lithomaker.gui import (
    PARAMETER_MAP,
    ZOOM_MAX,
    ZOOM_MIN,
    config_from_parameters,
    heightmap_image,
    histogram_points,
    preview_config,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
requires_display = pytest.mark.skipif(not os.environ.get("DISPLAY"), reason="needs a display")


def sample_parameters(**overrides):
    values = {
        "input": "photo.png",
        "output": "out.stl",
        "black": 0.0,
        "white": 1.0,
        "gamma": 1.0,
        "auto_levels": False,
        "min_layers": 4,
        "max_layers": 32,
        "thickness": 0.2,
        "size_x": 120.0,
        "size_y": None,
        "dot_size": 0.4,
        "base_thickness": 1.0,
        "ascii": False,
        "force": False,
    }
    values.update(overrides)
    return values


# --------------------------------------------------------------------------
# Headless import and parity map
# --------------------------------------------------------------------------
def test_gui_imports_without_a_display():
    environment = {key: value for key, value in os.environ.items() if key != "DISPLAY"}

    completed = subprocess.run(
        [sys.executable, "-c", "import lithomaker.gui; print('imported')"],
        capture_output=True,
        text=True,
        env=environment,
        cwd=REPO_ROOT,
    )

    assert completed.returncode == 0, completed.stderr
    assert "imported" in completed.stdout


def test_parity_map_matches_the_cli_parameters():
    aliases = {
        alias: canonical
        for canonical, names in PARAMETER_ALIASES.items()
        for alias in names
    }
    canonical_cli = {
        aliases.get(param.name, param.name)
        for param in convert_command.params
        if param.name not in PRESENTATION_OPTIONS
    }

    assert canonical_cli == set(PARAMETER_MAP)


def test_parity_map_has_no_unknown_cli_flags():
    cli_names = {param.name for param in convert_command.params}

    for names in PARAMETER_ALIASES.values():
        assert set(names) <= cli_names
    assert PRESENTATION_OPTIONS <= cli_names


def test_gui_entry_points_are_declared():
    script = (REPO_ROOT / "pyproject.toml").read_text()

    assert 'lithomaker-gui = "lithomaker.gui:main"' in script


# --------------------------------------------------------------------------
# config_from_parameters
# --------------------------------------------------------------------------
def test_config_from_parameters_builds_the_shared_object():
    config = config_from_parameters(sample_parameters())

    assert config.input == Path("photo.png")
    assert config.output == Path("out.stl")
    assert config.levels.black == 0.0
    assert config.max_layers == 32
    assert config.ascii is False


def test_config_from_parameters_accepts_widget_strings():
    config = config_from_parameters(
        sample_parameters(
            thickness="0.3",
            size_x="80",
            size_y="60",
            dot_size="0.5",
            base_thickness="1.2",
            min_layers="3",
            max_layers="20",
        )
    )

    assert config.thickness == pytest.approx(0.3)
    assert config.size_x == pytest.approx(80.0)
    assert config.size_y == pytest.approx(60.0)
    assert config.min_layers == 3


def test_blank_output_means_default_naming():
    config = config_from_parameters(sample_parameters(output="   "))

    assert config.output is None
    assert config.resolved_output == Path("photo.stl")


def test_blank_size_y_means_aspect_correct():
    assert config_from_parameters(sample_parameters(size_y="")).size_y is None


def test_missing_image_is_reported_clearly():
    with pytest.raises(ConfigError, match="choose an input image"):
        config_from_parameters(sample_parameters(input=""))


def test_bad_numbers_name_the_field():
    with pytest.raises(ConfigError, match="thickness must be a number"):
        config_from_parameters(sample_parameters(thickness="thick"))


def test_blank_number_is_rejected():
    with pytest.raises(ConfigError, match="dot size must not be empty"):
        config_from_parameters(sample_parameters(dot_size="  "))


def test_invalid_relationship_reuses_config_validation():
    with pytest.raises(ConfigError, match="min layers"):
        config_from_parameters(sample_parameters(min_layers=10, max_layers=5))


# --------------------------------------------------------------------------
# pure preview helpers
# --------------------------------------------------------------------------
def test_preview_config_coarsens_the_dot_size():
    config = config_from_parameters(sample_parameters(size_x=160.0, dot_size=0.1))

    coarse = preview_config(config, max_vertices=80)

    assert coarse.dot_size >= 160.0 / 80
    assert config.dot_size == pytest.approx(0.1)  # original untouched (frozen)


def test_preview_config_never_widens_an_already_coarse_grid():
    config = config_from_parameters(sample_parameters(size_x=10.0, dot_size=2.0))

    assert preview_config(config).dot_size == pytest.approx(2.0)


def _heightmap(values: np.ndarray) -> Heightmap:
    return Heightmap(
        values=values,
        grid_shape=values.shape,
        resolved_size=(10.0, 10.0),
        z_min=0.8,
        z_max=4.0,
        warnings=[],
    )


def test_heightmap_image_shows_transmitted_light():
    # values is indexed [x, y]: one column of two X pixels.
    heightmap = _heightmap(np.array([[1.0], [0.0]], dtype=np.float32))

    image = heightmap_image(heightmap)
    pixels = np.asarray(image)

    assert image.mode == "L"
    # Bright source pixel (1.0) is thin -> more light gets through -> lighter preview.
    assert pixels[0, 0] > pixels[0, 1]


def test_transmission_follows_beer_lambert_with_the_base_plate():
    heightmap = _heightmap(np.array([[1.0], [0.0]], dtype=np.float32))

    transmitted = gui_module.transmission(heightmap, base_thickness=2.0)

    thin = 2.0 + heightmap.z_min
    thick = 2.0 + heightmap.z_max
    assert transmitted[0, 0] == pytest.approx(
        np.exp(-gui_module.LIGHT_ABSORPTION_PER_MM * thin)
    )
    assert transmitted[1, 0] == pytest.approx(
        np.exp(-gui_module.LIGHT_ABSORPTION_PER_MM * thick)
    )


def test_base_plate_thickness_dims_the_preview():
    heightmap = _heightmap(np.array([[1.0], [0.5]], dtype=np.float32))

    thin_plate = heightmap_image(heightmap, base_thickness=0.0)
    thick_plate = heightmap_image(heightmap, base_thickness=3.0)

    # More material everywhere -> less transmitted light -> darker preview.
    assert np.asarray(thick_plate).mean() < np.asarray(thin_plate).mean()


def test_heightmap_image_transposes_for_pil_rows():
    values = np.linspace(0.0, 1.0, 12, dtype=np.float32).reshape(4, 3)

    image = heightmap_image(_heightmap(values))

    assert image.size == (4, 3)  # PIL size is (width=nx, height=ny)


def test_histogram_points_are_inside_the_canvas():
    counts = np.array([0, 5, 10, 5, 0])

    points = histogram_points(counts, width=100, height=50)

    assert points[0] == (0.0, 50.0)
    assert points[-1] == (100.0, 50.0)
    xs = [x for x, _ in points]
    ys = [y for _, y in points]
    assert min(xs) >= 0.0 and max(xs) <= 100.0
    assert min(ys) >= 0.0 and max(ys) <= 50.0
    assert min(ys) == pytest.approx(0.0)  # the tallest bin reaches the top


def test_histogram_points_handle_empty_counts():
    assert histogram_points(np.zeros(0), width=10, height=10) == [
        (0.0, 10.0),
        (10.0, 10.0),
    ]


# --------------------------------------------------------------------------
# widget tests (need a display)
# --------------------------------------------------------------------------
@pytest.fixture
def app():
    """Fresh hidden root per test: state isolation matters more than Tk start-up cost.

    Creating a root costs a couple of seconds on a slow X server and microseconds under
    Xvfb, so CI is unaffected; each test still gets a clean application instance.
    """
    import tkinter as tk

    try:
        root = tk.Tk()
    except tk.TclError:  # pragma: no cover - no reachable display
        pytest.skip("cannot connect to a display")
    root.withdraw()
    application = gui_module.LithomakerApp(root)
    application.pack(fill="both", expand=True)
    root.update()
    yield application
    if application._preview_job is not None:
        application.after_cancel(application._preview_job)
        application._preview_job = None
    root.destroy()


def pump(root, predicate, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        root.update()
        if predicate():
            return True
        time.sleep(0.01)
    root.update()
    return predicate()


def make_png(tmp_path, height=10, width=20, name="photo.png") -> Path:
    ramp = np.linspace(0.0, 1.0, width, dtype=np.float32)
    pixels = np.tile(ramp, (height, 1))
    path = tmp_path / name
    Image.fromarray(np.round(pixels * 255).astype(np.uint8)).convert("RGB").save(path)
    return path


@requires_display
def test_every_mapped_parameter_has_a_widget_variable(app):
    import tkinter as tk

    for canonical, attribute in PARAMETER_MAP.items():
        assert isinstance(getattr(app, attribute), tk.Variable), canonical


@requires_display
def test_defaults_match_the_cli(app):
    values = app.read_parameters()

    assert values["black"] == pytest.approx(0.0)
    assert values["white"] == pytest.approx(1.0)
    assert values["gamma"] == pytest.approx(1.0)
    assert values["min_layers"] == 4
    assert values["max_layers"] == 32
    assert values["thickness"] == "0.2"
    assert values["size_x"] == "120"
    assert values["dot_size"] == "0.4"
    assert values["base_thickness"] == "1.0"
    assert values["ascii"] is False


@requires_display
def test_loading_an_image_fills_the_output_and_preview(app, tmp_path):
    path = make_png(tmp_path)

    app.load_image_file(path)

    assert app.image_path_var.get() == str(path)
    assert app.output_path_var.get() == str(path.with_suffix(".stl"))
    assert app._preview_photo is not None  # rendered immediately on load
    assert app.preview_info_var.get().startswith("heightmap")


@requires_display
def test_histogram_is_drawn_after_loading(app, tmp_path):
    app.load_image_file(make_png(tmp_path))

    items = app.histogram_canvas.find_all()

    assert len(items) >= 3  # polygon plus the two black/white markers


@requires_display
def test_heightmap_preview_uses_updated_levels(app, tmp_path, monkeypatch):
    seen = {}
    original = gui_module.build_heightmap

    def spy(image, config):
        seen["gamma"] = config.levels.gamma
        return original(image, config)

    monkeypatch.setattr(gui_module, "build_heightmap", spy)
    app.load_image_file(make_png(tmp_path))

    app.gamma_var.set(2.0)
    app._render_heightmap_preview()

    assert seen["gamma"] == pytest.approx(2.0)


@requires_display
def test_aspect_lock_derives_size_y_from_the_image(app, tmp_path):
    app.load_image_file(make_png(tmp_path, height=20, width=10))

    app.size_x_var.set("50")

    assert float(app.size_y_var.get()) == pytest.approx(100.0)


@requires_display
def test_unlocked_aspect_leaves_size_y_alone(app, tmp_path):
    app.load_image_file(make_png(tmp_path, height=20, width=10))
    app.size_y_var.set("")

    app.lock_aspect_var.set(False)
    app.size_x_var.set("50")

    assert app.size_y_var.get() == ""


@requires_display
def test_invalid_parameters_show_a_dialog(app, tmp_path, monkeypatch):
    shown = {}
    monkeypatch.setattr(
        gui_module.messagebox, "showerror",
        lambda title, message, **kwargs: shown.update(title=title, message=message),
    )
    app.load_image_file(make_png(tmp_path))
    app.thickness_var.set("nope")

    app._on_generate()

    assert shown["title"] == "Invalid parameters"
    assert "thickness" in shown["message"]


@requires_display
def test_generate_converts_and_updates_the_result_panel(app, tmp_path, monkeypatch):
    app.load_image_file(make_png(tmp_path))
    output = tmp_path / "out.stl"
    app.output_path_var.set(str(output))
    app.size_x_var.set("10")
    app.size_y_var.set("10")
    app.dot_size_var.set("2")
    app.max_layers_var.set(12)

    app._on_generate()

    assert pump(app.winfo_toplevel(), lambda: not app._busy)
    assert output.is_file()
    assert app.stats_vars["triangles"].get() != "—"
    assert app.stats_vars["file_size"].get().endswith("bytes")
    assert "Wrote" in app.status_var.get()
    assert str(app.show_button.cget("state")) == "normal"


@requires_display
def test_existing_output_requires_confirmation(app, tmp_path, monkeypatch):
    app.load_image_file(make_png(tmp_path))
    output = tmp_path / "out.stl"
    output.write_bytes(b"precious")
    app.output_path_var.set(str(output))
    app.size_x_var.set("10")
    app.size_y_var.set("10")
    app.dot_size_var.set("2")
    asked = {}
    monkeypatch.setattr(
        gui_module.messagebox, "askyesno",
        lambda title, message, **kwargs: asked.setdefault("asked", True) and False,
    )

    app._on_generate()

    assert asked["asked"]
    assert output.read_bytes() == b"precious"
    assert "cancelled" in app.status_var.get()
    assert not app._busy


@requires_display
def test_confirmed_overwrite_replaces_the_file(app, tmp_path, monkeypatch):
    app.load_image_file(make_png(tmp_path))
    output = tmp_path / "out.stl"
    output.write_bytes(b"precious")
    app.output_path_var.set(str(output))
    app.size_x_var.set("10")
    app.size_y_var.set("10")
    app.dot_size_var.set("2")
    monkeypatch.setattr(
        gui_module.messagebox, "askyesno", lambda *args, **kwargs: True
    )

    app._on_generate()

    assert pump(app.winfo_toplevel(), lambda: not app._busy)
    assert output.read_bytes()[:8] != b"precious"


@requires_display
def test_conversion_failure_shows_an_error_dialog(app, tmp_path, monkeypatch):
    shown = {}
    monkeypatch.setattr(
        gui_module.messagebox, "showerror",
        lambda title, message, **kwargs: shown.update(title=title, message=message),
    )
    app.load_image_file(make_png(tmp_path))
    app.output_path_var.set(str(tmp_path / "out.stl"))

    def boom(*args, **kwargs):
        raise RuntimeError("engine failure")

    monkeypatch.setattr(gui_module, "convert", boom)

    app._on_generate()

    assert pump(app.winfo_toplevel(), lambda: not app._busy)
    assert shown["title"] == "Conversion failed"
    assert "engine failure" in shown["message"]
    assert not (tmp_path / "out.stl").exists()


@requires_display
def test_busy_state_disables_the_generate_button(app):
    app._set_busy(True)
    assert str(app.generate_button.cget("state")) == "disabled"

    app._set_busy(False)
    assert str(app.generate_button.cget("state")) == "normal"


@requires_display
def test_preview_panel_has_one_zoomable_preview_and_no_toggle(app):
    assert hasattr(app, "preview_canvas")
    assert hasattr(app, "zoom_scale")
    assert hasattr(app, "fit_button")
    assert not hasattr(app, "preview_label")
    assert not hasattr(app, "source_label")
    assert not hasattr(app, "invert_preview_var")
    assert not hasattr(app, "invert_preview_check")
    assert app._preview_image is None

    texts = [
        app.preview_canvas.itemcget(item, "text")
        for item in app.preview_canvas.find_all()
    ]
    assert any("Open an image" in text for text in texts)


@requires_display
def test_preview_shows_bright_areas_light_by_default(app, tmp_path):
    # Left half white, right half black: the left half is thin and transmits light.
    pixels = np.zeros((10, 20, 3), dtype=np.uint8)
    pixels[:, :10] = 255
    path = tmp_path / "halves.png"
    Image.fromarray(pixels).save(path)

    app.load_image_file(path)
    app._render_heightmap_preview()

    preview = np.asarray(app._preview_image)
    assert preview[:, : preview.shape[1] // 2].mean() > preview[:, preview.shape[1] // 2 :].mean()
    assert "base" in app.preview_info_var.get()


@requires_display
def test_histogram_scales_with_the_canvas_width(app, tmp_path, monkeypatch):
    app.load_image_file(make_png(tmp_path))
    monkeypatch.setattr(app.histogram_canvas, "winfo_width", lambda: 640)
    monkeypatch.setattr(app.histogram_canvas, "winfo_height", lambda: 120)

    app._draw_histogram()

    canvas = app.histogram_canvas
    polygons = [item for item in canvas.find_all() if canvas.type(item) == "polygon"]
    lines = [item for item in canvas.find_all() if canvas.type(item) == "line"]
    bbox = canvas.bbox(polygons[0])
    assert bbox[2] - bbox[0] >= 639
    marker_xs = {canvas.coords(item)[0] for item in lines}
    assert 0.0 in marker_xs
    assert 640.0 in marker_xs  # the white point marker follows the canvas width


@requires_display
def test_histogram_redraws_after_a_resize_event(app, tmp_path):
    app.load_image_file(make_png(tmp_path))
    app.histogram_canvas.delete("all")

    app._on_histogram_configure()

    assert app._histogram_job is not None
    assert pump(app.winfo_toplevel(), lambda: app._histogram_job is None, timeout=2)
    assert app.histogram_canvas.find_all()


@requires_display
def test_zoom_defaults_to_fit(app, tmp_path):
    app.load_image_file(make_png(tmp_path))

    assert app.zoom_var.get() == pytest.approx(ZOOM_MIN)
    assert app.zoom_label_var.get() == "Fit"
    fit_width = app._preview_image.width

    app.zoom_var.set(2.0)

    assert app._preview_image.width == pytest.approx(fit_width * 2, abs=2)
    assert app.zoom_label_var.get() == "2.0×"


@requires_display
def test_mouse_wheel_zooms_the_preview(app, tmp_path):
    app.load_image_file(make_png(tmp_path))
    before = app._preview_image.width

    app._on_preview_wheel(SimpleNamespace(delta=120, x=10, y=10))

    assert app.zoom_var.get() == pytest.approx(1.1)
    assert app._preview_image.width > before

    app._on_preview_wheel(SimpleNamespace(delta=-120, x=10, y=10))

    assert app.zoom_var.get() == pytest.approx(1.0)


@requires_display
def test_zoom_is_clamped_to_the_supported_range(app, tmp_path):
    app.load_image_file(make_png(tmp_path))

    for _ in range(80):
        app._on_preview_wheel(SimpleNamespace(delta=120, x=0, y=0))
    assert app.zoom_var.get() == pytest.approx(ZOOM_MAX)

    for _ in range(200):
        app._on_preview_wheel(SimpleNamespace(delta=-120, x=0, y=0))
    assert app.zoom_var.get() == pytest.approx(ZOOM_MIN)
    assert app.zoom_label_var.get() == "Fit"


@requires_display
def test_fit_button_restores_the_whole_view(app, tmp_path):
    app.load_image_file(make_png(tmp_path))
    app.zoom_var.set(3.0)

    app.fit_button.invoke()

    assert app.zoom_var.get() == pytest.approx(ZOOM_MIN)
    assert app.zoom_label_var.get() == "Fit"


@requires_display
def test_zooming_makes_the_preview_scrollable(app, tmp_path, monkeypatch):
    monkeypatch.setattr(app.preview_canvas, "winfo_width", lambda: 200)
    monkeypatch.setattr(app.preview_canvas, "winfo_height", lambda: 200)
    app.load_image_file(make_png(tmp_path))
    app._render_preview_canvas()

    app.zoom_var.set(4.0)

    region = [float(value) for value in app.preview_canvas.cget("scrollregion").split()]
    assert region[2] > 200
    assert region[3] > 200


@requires_display
def test_dragging_the_preview_pans(app, tmp_path, monkeypatch):
    app.load_image_file(make_png(tmp_path))
    app.zoom_var.set(4.0)
    marks = []
    monkeypatch.setattr(app.preview_canvas, "scan_mark", lambda x, y: marks.append(("mark", x, y)))
    monkeypatch.setattr(app.preview_canvas, "scan_dragto", lambda x, y, gain: marks.append(("drag", x, y)))

    app._start_pan(SimpleNamespace(x=5, y=6))
    app._drag_pan(SimpleNamespace(x=15, y=16))
    app._end_pan()

    assert marks == [("mark", 5, 6), ("drag", 15, 16)]
    assert app.preview_canvas.cget("cursor") == ""


@requires_display
def test_show_in_folder_is_disabled_until_a_result_exists(app):
    assert str(app.show_button.cget("state")) == "disabled"


@requires_display
def test_missing_image_is_reported_in_a_dialog(app, tmp_path, monkeypatch):
    shown = {}
    monkeypatch.setattr(
        gui_module.messagebox, "showerror",
        lambda title, message, **kwargs: shown.update(title=title),
    )

    app.load_image_file(tmp_path / "nope.png")

    assert shown["title"] == "Cannot open image"
    assert app.image_path_var.get() == ""


@requires_display
def test_browse_image_uses_the_dialog_choice(app, tmp_path, monkeypatch):
    path = make_png(tmp_path)
    monkeypatch.setattr(
        gui_module.filedialog, "askopenfilename", lambda **kwargs: str(path)
    )

    app._browse_image()

    assert app.image_path_var.get() == str(path)


@requires_display
def test_browse_output_uses_the_dialog_choice(app, tmp_path, monkeypatch):
    chosen = tmp_path / "chosen.stl"
    monkeypatch.setattr(
        gui_module.filedialog, "asksaveasfilename", lambda **kwargs: str(chosen)
    )

    app._browse_output()

    assert app.output_path_var.get() == str(chosen)


@requires_display
def test_generate_without_an_output_path_asks_for_one(app, tmp_path, monkeypatch):
    app.load_image_file(make_png(tmp_path))
    app.output_path_var.set("")
    app.size_x_var.set("10")
    app.size_y_var.set("10")
    app.dot_size_var.set("2")
    chosen = tmp_path / "asked.stl"
    monkeypatch.setattr(
        gui_module.filedialog, "asksaveasfilename", lambda **kwargs: str(chosen)
    )

    app._on_generate()

    assert pump(app.winfo_toplevel(), lambda: not app._busy)
    assert chosen.is_file()
    assert app.output_path_var.get() == str(chosen)


@requires_display
def test_generate_with_a_cancelled_save_dialog_does_nothing(app, tmp_path, monkeypatch):
    app.load_image_file(make_png(tmp_path))
    app.output_path_var.set("")
    monkeypatch.setattr(gui_module.filedialog, "asksaveasfilename", lambda **kwargs: "")

    app._on_generate()

    assert not app._busy
    assert app.output_path_var.get() == ""


@requires_display
def test_show_in_folder_opens_the_result(app, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        gui_module, "_open_in_file_manager", lambda path: calls.append(path)
    )
    app._last_output = tmp_path / "out.stl"

    app._show_in_folder()

    assert calls == [tmp_path / "out.stl"]


@requires_display
def test_preview_reports_invalid_parameters_instead_of_crashing(app, tmp_path):
    app.load_image_file(make_png(tmp_path))
    app.thickness_var.set("nope")

    app._render_heightmap_preview()

    assert app.preview_info_var.get().startswith("preview unavailable")


@requires_display
def test_loading_another_image_keeps_a_chosen_output_path(app, tmp_path):
    app.load_image_file(make_png(tmp_path, name="first.png"))
    app.output_path_var.set("custom.stl")

    app.load_image_file(make_png(tmp_path, name="second.png"))

    assert app.output_path_var.get() == "custom.stl"
