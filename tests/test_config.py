"""Tests for the parameter contract (TODO.md, LM-2)."""

from __future__ import annotations

import math
import re
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest

from lithomaker.config import (
    ASPECT_TOLERANCE,
    MAX_GRID_VERTICES,
    ConfigError,
    Footprint,
    LevelsConfig,
    LithoConfig,
)

# 3:2 landscape image used by the geometry tests below (aspect ratio 2/3).
IMG_W, IMG_H = 300, 200


def make_config(**overrides) -> LithoConfig:
    kwargs = {"input": "photo.png"}
    kwargs.update(overrides)
    return LithoConfig(**kwargs)


# --------------------------------------------------------------------------
# defaults
# --------------------------------------------------------------------------
def test_defaults_match_cli_spec():
    cfg = make_config()
    assert cfg.min_layers == 1
    assert cfg.max_layers == 13
    assert cfg.thickness == pytest.approx(0.2)
    assert cfg.size_x == pytest.approx(120.0)
    assert cfg.size_y is None
    assert cfg.dot_size == pytest.approx(0.4)
    assert cfg.base_thickness == pytest.approx(0.4)
    assert cfg.ascii is False
    assert cfg.levels.is_identity


def test_paths_are_coerced_to_path_objects():
    cfg = LithoConfig(input="photo.png", output="out.stl")
    assert isinstance(cfg.input, Path)
    assert isinstance(cfg.output, Path)
    assert cfg.input == Path("photo.png")


def test_pathlib_input_is_preserved():
    cfg = LithoConfig(input=Path("/tmp/photo.png"))
    assert cfg.input == Path("/tmp/photo.png")


def test_config_is_immutable():
    cfg = make_config()
    with pytest.raises(FrozenInstanceError):
        cfg.thickness = 0.3  # type: ignore[misc]


# --------------------------------------------------------------------------
# LevelsConfig validation
# --------------------------------------------------------------------------
def test_levels_defaults_are_identity():
    levels = LevelsConfig()
    assert levels.is_identity
    assert levels.describe() == "levels black=0 white=1 gamma=1"


@pytest.mark.parametrize("black", [-0.1, 1.0, 1.5, math.nan, math.inf])
def test_levels_black_out_of_range(black):
    with pytest.raises(ConfigError):
        LevelsConfig(black=black)


@pytest.mark.parametrize("white", [-0.5, 1.01, math.nan])
def test_levels_white_out_of_range(white):
    with pytest.raises(ConfigError):
        LevelsConfig(white=white)


def test_levels_black_must_be_below_white():
    with pytest.raises(ConfigError, match="must be lower than white point"):
        LevelsConfig(black=0.8, white=0.2)
    # Equal endpoints would divide by zero in the levels remap.
    with pytest.raises(ConfigError, match="must be lower than white point"):
        LevelsConfig(black=0.5, white=0.5)


@pytest.mark.parametrize("gamma", [0.0, -1.0, math.nan])
def test_levels_gamma_must_be_positive(gamma):
    with pytest.raises(ConfigError, match="gamma"):
        LevelsConfig(gamma=gamma)


@pytest.mark.parametrize("value", ["0.1", None, [0.1], True])
def test_levels_rejects_non_numeric(value):
    with pytest.raises(ConfigError):
        LevelsConfig(black=value)


def test_levels_auto_is_coerced_to_bool():
    assert LevelsConfig(auto=1).auto is True
    assert LevelsConfig(auto=0).auto is False


def test_levels_from_mapping_is_accepted():
    cfg = make_config(levels={"black": 0.1, "white": 0.9, "gamma": 1.2})
    assert isinstance(cfg.levels, LevelsConfig)
    assert cfg.levels.black == pytest.approx(0.1)
    assert not cfg.levels.is_identity


def test_levels_none_falls_back_to_default():
    assert make_config(levels=None).levels.is_identity


def test_levels_wrong_type_is_rejected():
    with pytest.raises(ConfigError, match="LevelsConfig or mapping"):
        make_config(levels=42)


# --------------------------------------------------------------------------
# LithoConfig validation
# --------------------------------------------------------------------------
@pytest.mark.parametrize("bad_input", ["", "   ", None, 42])
def test_invalid_input_path(bad_input):
    with pytest.raises(ConfigError):
        LithoConfig(input=bad_input)


@pytest.mark.parametrize("bad_output", ["", 42, object()])
def test_invalid_output_path(bad_output):
    with pytest.raises(ConfigError):
        LithoConfig(input="photo.png", output=bad_output)


@pytest.mark.parametrize("min_layers", [0, -1])
def test_min_layers_must_be_at_least_one(min_layers):
    with pytest.raises(ConfigError, match="min layers"):
        make_config(min_layers=min_layers)


@pytest.mark.parametrize("max_layers", [0, -5])
def test_max_layers_must_be_at_least_one(max_layers):
    with pytest.raises(ConfigError, match="max layers"):
        make_config(max_layers=max_layers)


def test_max_layers_must_not_be_below_min_layers():
    with pytest.raises(ConfigError, match="greater than or equal to min layers"):
        make_config(min_layers=10, max_layers=9)
    # Equal values are legitimate: a uniform-thickness panel.
    assert make_config(min_layers=10, max_layers=10).z_max == pytest.approx(2.0)


@pytest.mark.parametrize("value", [10.0, 10])
def test_integral_floats_accepted_for_layer_counts(value):
    assert make_config(max_layers=value).max_layers == 10


@pytest.mark.parametrize("value", [10.5, "10", None, True])
def test_non_integral_layer_counts_rejected(value):
    with pytest.raises(ConfigError, match="max layers"):
        make_config(max_layers=value)


@pytest.mark.parametrize("thickness", [0.0, -0.2, math.nan, math.inf])
def test_thickness_must_be_positive_and_finite(thickness):
    with pytest.raises(ConfigError, match="layer thickness"):
        make_config(thickness=thickness)


@pytest.mark.parametrize("size_x", [0.0, -100.0, math.nan])
def test_size_x_must_be_positive(size_x):
    with pytest.raises(ConfigError, match="output size X"):
        make_config(size_x=size_x)


@pytest.mark.parametrize("size_y", [0.0, -10.0, math.inf])
def test_size_y_must_be_positive_when_given(size_y):
    with pytest.raises(ConfigError, match="output size Y"):
        make_config(size_y=size_y)


@pytest.mark.parametrize("dot_size", [0.0, -0.4, math.nan])
def test_dot_size_must_be_positive(dot_size):
    with pytest.raises(ConfigError, match="dot size"):
        make_config(dot_size=dot_size)


def test_base_thickness_may_be_zero_but_not_negative():
    assert make_config(base_thickness=0).base_thickness == 0.0
    with pytest.raises(ConfigError, match="base thickness"):
        make_config(base_thickness=-1.0)


# --------------------------------------------------------------------------
# derived Z geometry (decision D10)
# --------------------------------------------------------------------------
def test_z_range_derivation():
    cfg = make_config(min_layers=4, max_layers=32, thickness=0.2, base_thickness=1.0)
    assert cfg.z_min == pytest.approx(0.8)
    assert cfg.z_max == pytest.approx(6.4)
    assert cfg.total_height == pytest.approx(7.4)


def test_z_range_with_zero_base():
    cfg = make_config(min_layers=1, max_layers=5, thickness=0.1, base_thickness=0.0)
    assert cfg.z_min == pytest.approx(0.1)
    assert cfg.z_max == pytest.approx(0.5)
    assert cfg.total_height == pytest.approx(0.5)


def test_height_to_z_orientation():
    """Bright source pixels are thin; dark pixels are thick (decision D1)."""
    cfg = make_config(min_layers=2, max_layers=10, thickness=0.2, base_thickness=1.0)
    assert cfg.height_to_z(0.0) == pytest.approx(1.0 + 2.0)  # dark  -> max thickness
    assert cfg.height_to_z(1.0) == pytest.approx(1.0 + 0.4)  # bright -> min thickness
    assert cfg.height_to_z(0.5) == pytest.approx(1.0 + 1.2)


def test_height_to_z_is_monotonic_decreasing():
    cfg = make_config()
    values = [cfg.height_to_z(h) for h in np.linspace(0.0, 1.0, 11)]
    assert all(a >= b for a, b in zip(values, values[1:]))


def test_height_to_z_works_on_arrays():
    cfg = make_config(min_layers=2, max_layers=12, thickness=0.1, base_thickness=0.5)
    heights = np.array([0.0, 0.5, 1.0], dtype=np.float32)
    z = cfg.height_to_z(heights)
    assert isinstance(z, np.ndarray)
    assert z.shape == heights.shape
    np.testing.assert_allclose(z, [0.5 + 1.2, 0.5 + 0.7, 0.5 + 0.2], rtol=1e-6)


# --------------------------------------------------------------------------
# resolved_size (decision D2)
# --------------------------------------------------------------------------
def test_size_y_defaults_to_aspect_correct():
    cfg = make_config(size_x=120.0)
    footprint = cfg.resolved_size(IMG_W, IMG_H)
    assert isinstance(footprint, Footprint)
    assert footprint.x == pytest.approx(120.0)
    assert footprint.y == pytest.approx(80.0)
    assert footprint.aspect_warning is None


def test_resolved_size_unpacks_as_tuple():
    """Core code may consume the footprint positionally or by name."""
    x, y, warning = make_config(size_x=120.0).resolved_size(IMG_W, IMG_H)
    assert x == pytest.approx(120.0)
    assert y == pytest.approx(80.0)
    assert warning is None


def test_explicit_size_y_within_tolerance_is_silent():
    # 100 x 100.9 mm against a 1:1 image deviates by 0.9% — under the 1% tolerance.
    cfg = make_config(size_x=100.0, size_y=100.9)
    footprint = cfg.resolved_size(1000, 1000)
    assert footprint.y == pytest.approx(100.9)
    assert footprint.aspect_warning is None


def test_explicit_size_y_outside_tolerance_warns_but_stretches():
    cfg = make_config(size_x=100.0, size_y=110.0)
    footprint = cfg.resolved_size(1000, 1000)
    assert footprint.aspect_warning is not None
    assert "stretched" in footprint.aspect_warning
    assert "--size-y" in footprint.aspect_warning
    # Stretching still honours what the user asked for.
    assert (footprint.x, footprint.y) == (100.0, 110.0)


def test_landscape_image_derives_short_height():
    cfg = make_config(size_x=200.0)
    assert cfg.resolved_size(4000, 1000).y == pytest.approx(50.0)


def test_portrait_image_derives_tall_height():
    cfg = make_config(size_x=100.0)
    assert cfg.resolved_size(1000, 2000).y == pytest.approx(200.0)


@pytest.mark.parametrize("bad", [(0, 100), (100, 0), (-5, 10)])
def test_resolved_size_rejects_bad_image_dimensions(bad):
    with pytest.raises(ConfigError, match="image"):
        make_config().resolved_size(*bad)


def test_aspect_tolerance_constant_is_one_percent():
    assert ASPECT_TOLERANCE == pytest.approx(0.01)


# --------------------------------------------------------------------------
# grid_shape (decisions D2 + D3)
# --------------------------------------------------------------------------
def test_grid_shape_matches_dot_pitch():
    # 120 x 80 mm at 0.4 mm per dot -> 300 x 200 dots, i.e. 301 x 201 vertices.
    assert make_config(size_x=120.0).grid_shape(IMG_W, IMG_H) == (301, 201)


def test_grid_shape_uses_explicit_size_y():
    cfg = make_config(size_x=100.0, size_y=50.0)
    assert cfg.grid_shape(IMG_W, IMG_H) == (251, 126)


def test_grid_shape_never_goes_below_two_vertices():
    cfg = make_config(size_x=10.0, size_y=10.0, dot_size=100.0)
    assert cfg.grid_shape(585, 585) == (2, 2)


def test_grid_shape_scales_with_dot_size():
    fine = make_config(size_x=100.0, dot_size=0.1).grid_shape(IMG_W, IMG_H)
    coarse = make_config(size_x=100.0, dot_size=1.0).grid_shape(IMG_W, IMG_H)
    assert fine[0] > coarse[0] and fine[1] > coarse[1]


def test_grid_shape_rejects_runaway_mesh_sizes():
    cfg = make_config(size_x=120.0, dot_size=0.01)
    with pytest.raises(ConfigError) as excinfo:
        cfg.grid_shape(IMG_W, IMG_H)

    message = str(excinfo.value)
    assert "mesh too large" in message
    assert f"{MAX_GRID_VERTICES:,}-vertex limit" in message
    assert "dot-size" in message


def test_runaway_mesh_error_suggests_a_dot_size_that_actually_fits():
    cfg = make_config(size_x=120.0, dot_size=0.01)
    with pytest.raises(ConfigError) as excinfo:
        cfg.grid_shape(IMG_W, IMG_H)

    match = re.search(r"--dot-size ([\d.]+) mm", str(excinfo.value))
    assert match, f"no dot size suggestion in: {excinfo.value}"
    suggested = float(match.group(1))

    assert suggested > cfg.dot_size
    nx, ny = make_config(size_x=120.0, dot_size=suggested).grid_shape(IMG_W, IMG_H)
    assert nx * ny <= MAX_GRID_VERTICES


def test_grid_shape_accepts_a_mesh_exactly_at_the_limit():
    # A deliberately coarse request must never trip the guard.
    cfg = make_config(size_x=50.0, size_y=50.0, dot_size=1.0)
    nx, ny = cfg.grid_shape(500, 500)
    assert nx * ny < MAX_GRID_VERTICES


# --------------------------------------------------------------------------
# resolved_output (decision D5)
# --------------------------------------------------------------------------
def test_output_defaults_to_input_stem():
    assert make_config(input="photos/dog.png").resolved_output == Path("photos/dog.stl")


def test_output_defaults_when_input_has_no_suffix():
    assert make_config(input="render").resolved_output == Path("render.stl")


def test_explicit_output_wins():
    cfg = LithoConfig(input="dog.png", output="build/litho.stl")
    assert cfg.resolved_output == Path("build/litho.stl")
