"""Tests for the heightmap pipeline and end-to-end orchestration (LM-13 / LM-17)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from lithomaker.config import ConfigError, LevelsConfig, LithoConfig
from lithomaker.core import (
    ConversionResult,
    Heightmap,
    build_heightmap,
    build_mesh,
    convert,
)
from lithomaker.image_io import ImageLoadError
from lithomaker.mesh import read_stl


def make_config(**overrides) -> LithoConfig:
    """Small, fast geometry: a 21x21 grid at 20 mm wide with a 1 mm dot pitch."""
    kwargs = {
        "input": "photo.png",
        "size_x": 20.0,
        "dot_size": 1.0,
        "min_layers": 4,
        "max_layers": 20,
        "thickness": 0.2,
        "base_thickness": 1.0,
    }
    kwargs.update(overrides)
    return LithoConfig(**kwargs)


# --------------------------------------------------------------------------
# build_heightmap — shape and value contract (LM-13)
# --------------------------------------------------------------------------
def test_heightmap_matches_the_configured_grid(ramp_rgb):
    config = make_config()

    heightmap = build_heightmap(ramp_rgb, config)

    assert isinstance(heightmap, Heightmap)
    assert heightmap.grid_shape == config.grid_shape(16, 16)
    assert heightmap.values.shape == heightmap.grid_shape
    assert heightmap.values.dtype == np.float32


def test_heightmap_values_stay_normalized(ramp_rgb):
    values = build_heightmap(ramp_rgb, make_config()).values

    assert values.min() >= 0.0
    assert values.max() <= 1.0


def test_heightmap_axis_zero_is_x(ramp_rgb):
    # ramp_rgb brightens left to right, so brightness must increase along axis 0.
    values = build_heightmap(ramp_rgb, make_config()).values
    row_means = values.mean(axis=1)

    assert np.all(np.diff(row_means) > 0)
    assert np.allclose(values, row_means[:, np.newaxis], atol=1e-5)


def test_heightmap_carries_the_configured_z_range():
    config = make_config(min_layers=3, max_layers=11, thickness=0.15)

    heightmap = build_heightmap(np.full((4, 4, 3), 0.5, dtype=np.float32), config)

    assert heightmap.z_min == pytest.approx(0.45)
    assert heightmap.z_max == pytest.approx(1.65)


def test_heightmap_resolved_size_is_aspect_correct():
    config = make_config(size_x=40.0, size_y=None)
    image = np.zeros((50, 100, 3), dtype=np.float32)  # 2:1 landscape

    heightmap = build_heightmap(image, config)

    assert heightmap.resolved_size == pytest.approx((40.0, 20.0))


def test_heightmap_accepts_a_path(example_png):
    heightmap = build_heightmap(example_png, make_config(size_x=10.0, dot_size=1.0))

    assert heightmap.values.shape == (11, 11)
    assert 0.0 <= heightmap.values.min() <= heightmap.values.max() <= 1.0


def test_heightmap_accepts_uint8_arrays():
    image = np.full((6, 6, 3), 128, dtype=np.uint8)

    values = build_heightmap(image, make_config()).values

    assert values == pytest.approx(128 / 255.0, abs=1e-3)


def test_heightmap_rejects_non_rgb_arrays():
    with pytest.raises(ValueError, match=r"\(H, W, 3\)"):
        build_heightmap(np.zeros((8, 8), dtype=np.float32), make_config())


def test_heightmap_propagates_image_errors(tmp_path):
    with pytest.raises(ImageLoadError):
        build_heightmap(tmp_path / "missing.png", make_config())


def test_heightmap_propagates_the_mesh_size_guard():
    config = make_config(size_x=400.0, dot_size=0.05)

    with pytest.raises(ConfigError, match="mesh too large"):
        build_heightmap(np.zeros((8, 8, 3), dtype=np.float32), config)


# --------------------------------------------------------------------------
# build_heightmap — levels and warnings (LM-13, D1/D2/D7)
# --------------------------------------------------------------------------
def test_white_image_maps_to_the_thinnest_material(solid_white):
    config = make_config()

    heightmap = build_heightmap(solid_white, config)
    top_z = config.height_to_z(heightmap.values)

    assert np.allclose(top_z, config.base_thickness + config.z_min)


def test_black_image_maps_to_the_thickest_material(solid_black):
    config = make_config()

    heightmap = build_heightmap(solid_black, config)
    top_z = config.height_to_z(heightmap.values)

    assert np.allclose(top_z, config.base_thickness + config.z_max)


def test_brightness_to_height_is_monotone(ramp_rgb):
    config = make_config()
    heightmap = build_heightmap(ramp_rgb, config)

    z_row = config.height_to_z(heightmap.values).mean(axis=1)

    assert np.all(np.diff(z_row) < 0)  # brighter -> thinner


def test_levels_are_applied_before_meshing(ramp_rgb):
    plain = build_heightmap(ramp_rgb, make_config())
    lifted = build_heightmap(ramp_rgb, make_config(levels=LevelsConfig(black=0.2, white=0.8)))

    # A tighter window clips more pixels to the extremes of the [0, 1] range.
    assert np.mean(lifted.values == 0.0) > np.mean(plain.values == 0.0)
    assert np.mean(lifted.values == 1.0) > np.mean(plain.values == 1.0)


def test_auto_levels_stretches_a_low_contrast_image(low_contrast_rgb):
    config = make_config(levels=LevelsConfig(auto=True), max_layers=20, min_layers=4)

    values = build_heightmap(low_contrast_rgb, config).values

    assert values.min() == pytest.approx(0.0, abs=1e-5)
    assert values.max() == pytest.approx(1.0, abs=1e-5)


def test_auto_levels_warns_when_manual_levels_are_ignored(ramp_rgb):
    config = make_config(levels=LevelsConfig(black=0.1, auto=True))

    warnings = build_heightmap(ramp_rgb, config).warnings

    assert any("auto levels" in warning for warning in warnings)


def test_auto_levels_with_identity_levels_does_not_warn(ramp_rgb):
    config = make_config(levels=LevelsConfig(auto=True))

    warnings = build_heightmap(ramp_rgb, config).warnings

    assert warnings == []


def test_aspect_warning_propagates_onto_the_heightmap(ramp_rgb):
    config = make_config(size_x=20.0, size_y=40.0)  # square image, stretched target

    warnings = build_heightmap(ramp_rgb, config).warnings

    assert len(warnings) == 1
    assert "stretched" in warnings[0]


def test_no_warnings_for_a_matching_footprint(ramp_rgb):
    assert build_heightmap(ramp_rgb, make_config()).warnings == []


# --------------------------------------------------------------------------
# build_mesh (LM-13 -> LM-15 integration)
# --------------------------------------------------------------------------
def test_build_mesh_is_closed_and_positive(ramp_rgb):
    config = make_config()
    mesh = build_mesh(build_heightmap(ramp_rgb, config), config)

    assert mesh.is_closed() is True
    assert mesh.volume() > 0


def test_build_mesh_footprint_matches_the_resolved_size(ramp_rgb):
    config = make_config(size_x=30.0, size_y=None)
    heightmap = build_heightmap(ramp_rgb, config)

    (x0, y0, _), (x1, y1, _) = build_mesh(heightmap, config).bounds()

    assert (x0, y0) == pytest.approx((0.0, 0.0))
    assert (x1, y1) == pytest.approx(heightmap.resolved_size)


def test_build_mesh_reaches_the_configured_total_height(solid_black):
    config = make_config()

    z_top = build_mesh(build_heightmap(solid_black, config), config).bounds()[1][2]

    assert z_top == pytest.approx(config.total_height)


def test_build_mesh_respects_the_base_plate(solid_white):
    config = make_config(base_thickness=1.5)

    z_top = build_mesh(build_heightmap(solid_white, config), config).bounds()[1][2]

    assert z_top == pytest.approx(1.5 + config.z_min)


# --------------------------------------------------------------------------
# convert — end to end (LM-17)
# --------------------------------------------------------------------------
def test_convert_writes_a_readable_stl(example_png, tmp_path):
    output = tmp_path / "out.stl"

    result = convert(example_png, output, make_config())

    assert isinstance(result, ConversionResult)
    assert result.output_path == output
    assert output.is_file()


def test_convert_result_matches_the_written_mesh(example_png, tmp_path):
    output = tmp_path / "out.stl"

    result = convert(example_png, output, make_config())
    reloaded = read_stl(output)

    assert result.triangle_count == reloaded.triangle_count
    assert result.vertex_count == reloaded.vertex_count
    assert reloaded.vertex_count == 2 * result.grid_shape[0] * result.grid_shape[1]
    assert result.file_size == output.stat().st_size
    assert result.file_size == 84 + 50 * reloaded.triangle_count
    assert reloaded.is_closed() is True
    assert reloaded.volume() > 0


def test_convert_reports_geometry_and_z_range(example_png, tmp_path):
    config = make_config(size_x=12.0, min_layers=2, max_layers=8, thickness=0.25)

    result = convert(example_png, tmp_path / "out.stl", config)

    assert result.resolved_size == pytest.approx((12.0, 12.0))
    assert result.grid_shape == config.grid_shape(256, 256)
    assert result.z_min == pytest.approx(0.5)
    assert result.z_max == pytest.approx(2.0)
    assert result.total_height == pytest.approx(config.base_thickness + 2.0)


def test_convert_defaults_the_output_to_the_input_stem(example_png, tmp_path):
    image = tmp_path / "portrait.png"
    image.write_bytes(example_png.read_bytes())

    result = convert(image, config=make_config())

    assert result.output_path == image.with_suffix(".stl")
    assert result.output_path.is_file()


def test_convert_creates_missing_parent_directories(example_png, tmp_path):
    output = tmp_path / "a" / "b" / "out.stl"

    convert(example_png, output, make_config())

    assert output.is_file()


def test_convert_refuses_to_overwrite_without_permission(example_png, tmp_path):
    output = tmp_path / "out.stl"
    output.write_bytes(b"precious")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        convert(example_png, output, make_config())

    assert output.read_bytes() == b"precious"


def test_convert_overwrites_when_allowed(example_png, tmp_path):
    output = tmp_path / "out.stl"
    output.write_bytes(b"stale")

    result = convert(example_png, output, make_config(), overwrite=True)

    assert output.read_bytes()[:5] != b"stale"
    assert result.file_size == output.stat().st_size


def test_convert_honours_the_ascii_flag(example_png, tmp_path):
    output = tmp_path / "out.stl"

    convert(example_png, output, make_config(ascii=True))

    assert output.read_text(encoding="ascii").startswith("solid ")


def test_convert_emits_monotonic_progress(example_png, tmp_path):
    events = []

    convert(example_png, tmp_path / "out.stl", make_config(), progress=lambda m, f: events.append((m, f)))

    fractions = [fraction for _, fraction in events]
    assert fractions == sorted(fractions)
    assert fractions[0] < 0.1
    assert fractions[-1] == pytest.approx(1.0)
    assert events[-1][0] == "done"


def test_convert_works_without_a_progress_callback(example_png, tmp_path):
    assert convert(example_png, tmp_path / "out.stl", make_config()).output_path.is_file()


def test_convert_propagates_warnings(example_png, tmp_path):
    config = make_config(size_x=10.0, size_y=10.0)  # square target, square sample

    result = convert(example_png, tmp_path / "out.stl", config)

    assert result.warnings == []

    stretched = convert(
        example_png,
        tmp_path / "stretched.stl",
        make_config(size_x=10.0, size_y=20.0),
    )
    assert any("stretched" in warning for warning in stretched.warnings)


def test_convert_positional_output_wins_over_config_output(example_png, tmp_path):
    config = make_config(output=tmp_path / "from_config.stl")

    result = convert(example_png, tmp_path / "explicit.stl", config)

    assert result.output_path == tmp_path / "explicit.stl"
    assert not (tmp_path / "from_config.stl").exists()


def test_convert_builds_a_config_when_none_is_given(example_png, tmp_path):
    image = tmp_path / "photo.png"
    image.write_bytes(example_png.read_bytes())

    result = convert(image)

    assert result.output_path == Path(image).with_suffix(".stl")
    assert result.grid_shape == (301, 301)
    assert result.warnings == []


def test_convert_accepts_string_paths(example_png, tmp_path):
    result = convert(str(example_png), str(tmp_path / "out.stl"), make_config())

    assert result.output_path == tmp_path / "out.stl"


def test_convert_leaves_no_temporary_files(example_png, tmp_path):
    convert(example_png, tmp_path / "out.stl", make_config())

    assert [path.name for path in tmp_path.iterdir()] == ["out.stl"]


def test_convert_missing_input_raises_image_error(tmp_path):
    with pytest.raises(ImageLoadError):
        convert(tmp_path / "nope.png", tmp_path / "out.stl", make_config())


def test_convert_survives_a_one_pixel_wide_image(column_rgb, tmp_path, make_image_file):
    image = make_image_file(column_rgb, "column.png")

    result = convert(image, tmp_path / "column.stl", make_config())

    assert result.grid_shape[1] >= 2
    assert result.triangle_count > 0


# --------------------------------------------------------------------------
# Public API surface (D12)
# --------------------------------------------------------------------------
def test_package_reexports_the_public_api():
    import lithomaker

    assert lithomaker.convert is convert
    assert lithomaker.build_mesh is build_mesh
    assert lithomaker.build_heightmap is build_heightmap


def test_conversion_result_is_a_plain_dataclass():
    result = ConversionResult(
        output_path=Path("out.stl"),
        vertex_count=1,
        triangle_count=1,
        file_size=134,
        z_min=0.8,
        z_max=4.0,
        total_height=5.0,
        resolved_size=(10.0, 10.0),
        grid_shape=(2, 2),
    )

    assert result.warnings == []
