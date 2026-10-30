"""Tests for image loading, luminance/levels and resampling (LM-10/11/12)."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from lithomaker.config import LevelsConfig
from lithomaker.image_io import (
    SUPPORTED_EXTENSIONS,
    ImageLoadError,
    apply_levels,
    auto_levels,
    histogram,
    load_image,
    resample,
    to_luminance,
)


# --------------------------------------------------------------------------
# load_image — happy path & formats (LM-10)
# --------------------------------------------------------------------------
def test_load_image_returns_float32_rgb(ramp_rgb, make_image_file):
    path = make_image_file(ramp_rgb, "ramp.png")

    result = load_image(path)

    assert result.shape == (16, 16, 3)
    assert result.dtype == np.float32
    assert result.min() >= 0.0 and result.max() <= 1.0


def test_load_image_matches_written_pixels(tiny_rgb, make_image_file):
    path = make_image_file(tiny_rgb, "tiny.png")

    result = load_image(path)

    np.testing.assert_allclose(result, tiny_rgb, atol=1.0 / 255.0)


@pytest.mark.parametrize("name", ["a.png", "a.jpg", "a.bmp", "a.tiff", "a.webp"])
def test_load_image_accepts_every_allowlisted_format(ramp_rgb, make_image_file, name):
    path = make_image_file(ramp_rgb, name)

    assert load_image(path).shape == (16, 16, 3)


def test_supported_extensions_cover_plan_formats():
    assert {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"} == set(
        SUPPORTED_EXTENSIONS
    )


def test_load_image_accepts_uppercase_extension(ramp_rgb, make_image_file):
    path = make_image_file(ramp_rgb, "SHOUTY.PNG")

    assert load_image(path).shape == (16, 16, 3)


def test_load_image_accepts_string_paths(ramp_rgb, make_image_file):
    path = make_image_file(ramp_rgb, "ramp.png")

    assert load_image(str(path)).shape == (16, 16, 3)


def test_load_image_is_contiguous(ramp_rgb, make_image_file):
    result = load_image(make_image_file(ramp_rgb, "ramp.png"))

    assert result.flags["C_CONTIGUOUS"]


# --------------------------------------------------------------------------
# load_image — failure modes (LM-10)
# --------------------------------------------------------------------------
def test_image_load_error_is_a_value_error():
    assert issubclass(ImageLoadError, ValueError)


def test_load_image_missing_file(tmp_path):
    with pytest.raises(ImageLoadError, match="not found"):
        load_image(tmp_path / "nope.png")


def test_load_image_rejects_directory(tmp_path):
    directory = tmp_path / "pictures"
    directory.mkdir()

    with pytest.raises(ImageLoadError, match="directory"):
        load_image(directory)


def test_load_image_rejects_unsupported_extension(ramp_rgb, make_image_file):
    path = make_image_file(ramp_rgb, "anim.gif")

    with pytest.raises(ImageLoadError) as excinfo:
        load_image(path)

    message = str(excinfo.value)
    assert "unsupported image format" in message
    assert ".png" in message  # lists what *is* supported


def test_load_image_rejects_missing_extension(ramp_rgb, make_image_file, tmp_path):
    source = make_image_file(ramp_rgb, "source.png")
    path = tmp_path / "noextension"
    path.write_bytes(source.read_bytes())

    with pytest.raises(ImageLoadError, match="unsupported image format"):
        load_image(path)


def test_load_image_rejects_corrupt_data(tmp_path):
    path = tmp_path / "broken.png"
    path.write_bytes(b"this is definitely not a PNG")

    with pytest.raises(ImageLoadError, match="could not read image"):
        load_image(path)


def test_load_image_rejects_truncated_png(ramp_rgb, make_image_file):
    path = make_image_file(ramp_rgb, "cut.png")
    path.write_bytes(path.read_bytes()[:40])  # valid signature, missing payload

    with pytest.raises(ImageLoadError, match="could not read image"):
        load_image(path)


# --------------------------------------------------------------------------
# load_image — EXIF, modes, ICC (LM-10)
# --------------------------------------------------------------------------
def test_load_image_applies_exif_orientation(tmp_path):
    # Saved as 8x4 (landscape) with orientation 6, so it must come back 4x8.
    pixels = np.zeros((4, 8, 3), dtype=np.uint8)
    pixels[:, :4] = (255, 0, 0)
    pixels[:, 4:] = (0, 0, 255)
    exif = Image.Exif()
    exif[0x0112] = 6
    path = tmp_path / "rotated.jpg"
    Image.fromarray(pixels, "RGB").save(path, exif=exif.tobytes(), quality=95)

    result = load_image(path)

    assert result.shape == (8, 4, 3)


@pytest.mark.parametrize("mode", ["L", "LA", "P", "RGBA", "CMYK"])
def test_load_image_normalises_stock_modes(save_image, mode):
    color = {"L": 100, "LA": (100, 255), "P": 100, "RGBA": (10, 20, 30, 255)}.get(
        mode, (0, 255, 0, 0)
    )
    suffix = ".tiff" if mode == "CMYK" else ".png"
    path = save_image(Image.new(mode, (5, 3), color), f"mode_{mode}{suffix}")

    result = load_image(path)

    assert result.shape == (3, 5, 3)
    assert result.dtype == np.float32


def test_load_image_handles_16bit_grayscale(save_image):
    pixels = (np.arange(48, dtype=np.uint16) * 1000).reshape(6, 8)
    path = save_image(Image.fromarray(pixels), "deep.tiff")

    result = load_image(path)

    assert result.shape == (6, 8, 3)
    assert result.dtype == np.float32


def test_load_image_survives_broken_icc_profile(ramp_rgb, make_image_file):
    path = make_image_file(ramp_rgb, "icc.png", icc_profile=b"not an icc profile")

    assert load_image(path).shape == (16, 16, 3)


def test_load_image_applies_valid_icc_profile(ramp_rgb, make_image_file):
    from PIL import ImageCms

    profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    path = make_image_file(ramp_rgb, "srgb.png", icc_profile=profile)

    result = load_image(path)

    assert result.shape == (16, 16, 3)
    assert result.min() >= 0.0 and result.max() <= 1.0


def test_load_image_releases_the_file_handle(ramp_rgb, make_image_file):
    path = make_image_file(ramp_rgb, "ramp.png")
    load_image(path)

    path.unlink()  # fails on Windows if the handle leaked

    assert not path.exists()


def test_load_image_reads_the_project_sample(example_png):
    result = load_image(example_png)

    assert result.shape == (256, 256, 3)
    assert 0.0 <= result.min() < result.max() <= 1.0


# --------------------------------------------------------------------------
# to_luminance (LM-11)
# --------------------------------------------------------------------------
def test_to_luminance_uses_rec709_weights():
    pixel = np.array([[[1.0, 0.0, 0.0]]], dtype=np.float32)
    assert to_luminance(pixel)[0, 0] == pytest.approx(0.2126)

    pixel = np.array([[[0.0, 1.0, 0.0]]], dtype=np.float32)
    assert to_luminance(pixel)[0, 0] == pytest.approx(0.7152)

    pixel = np.array([[[0.0, 0.0, 1.0]]], dtype=np.float32)
    assert to_luminance(pixel)[0, 0] == pytest.approx(0.0722)


def test_to_luminance_of_grey_is_the_grey_value():
    pixel = np.array([[[0.4, 0.4, 0.4]]], dtype=np.float32)

    assert to_luminance(pixel)[0, 0] == pytest.approx(0.4, abs=1e-6)


def test_to_luminance_shrinks_axes(ramp_rgb):
    result = to_luminance(ramp_rgb)

    assert result.shape == (16, 16)
    assert result.dtype == np.float32


def test_to_luminance_returns_2d_input_unchanged(ramp_rgb):
    grey = to_luminance(ramp_rgb)

    result = to_luminance(grey)

    np.testing.assert_array_equal(result, grey)


def test_to_luminance_returns_a_copy():
    grey = np.full((2, 2), 0.5, dtype=np.float32)

    result = to_luminance(grey)
    result[0, 0] = 1.0

    assert grey[0, 0] == pytest.approx(0.5)


def test_to_luminance_squeezes_single_channel():
    data = np.array([[[0.25]], [[0.75]]], dtype=np.float32)

    assert to_luminance(data).shape == (2, 1)


def test_to_luminance_rejects_non_image_input():
    with pytest.raises(ValueError, match="expected an RGB image"):
        to_luminance(np.zeros((2, 2, 2), dtype=np.float32))


# --------------------------------------------------------------------------
# apply_levels (LM-11)
# --------------------------------------------------------------------------
def test_identity_levels_are_a_noop(ramp_rgb):
    grey = to_luminance(ramp_rgb)

    result = apply_levels(grey, LevelsConfig())

    np.testing.assert_allclose(result, grey, atol=1e-7)


def test_black_white_points_remap_the_range():
    grey = np.array([0.0, 0.5, 1.0], dtype=np.float32)

    result = apply_levels(grey, LevelsConfig(black=0.25, white=0.75))

    np.testing.assert_allclose(result, [0.0, 0.5, 1.0], atol=1e-6)


def test_apply_levels_clamps_outside_the_window():
    grey = np.array([0.1, 0.2, 0.8, 0.9], dtype=np.float32)

    result = apply_levels(grey, LevelsConfig(black=0.25, white=0.75))

    np.testing.assert_array_equal(result, [0.0, 0.0, 1.0, 1.0])


def test_gamma_above_one_darkens_midtones():
    grey = np.array([0.5], dtype=np.float32)

    assert apply_levels(grey, LevelsConfig(gamma=2.0))[0] == pytest.approx(0.25)


def test_gamma_below_one_brightens_midtones():
    grey = np.array([0.25], dtype=np.float32)

    assert apply_levels(grey, LevelsConfig(gamma=0.5))[0] == pytest.approx(0.5)


def test_apply_levels_preserves_dtype_and_range(ramp_rgb):
    result = apply_levels(to_luminance(ramp_rgb), LevelsConfig(black=0.1, gamma=1.7))

    assert result.dtype == np.float32
    assert result.min() >= 0.0 and result.max() <= 1.0


def test_apply_levels_does_not_mutate_input(ramp_rgb):
    grey = to_luminance(ramp_rgb)
    original = grey.copy()

    apply_levels(grey, LevelsConfig(black=0.2, white=0.8, gamma=2.0))

    np.testing.assert_array_equal(grey, original)


# --------------------------------------------------------------------------
# auto_levels (LM-11, D7)
# --------------------------------------------------------------------------
def test_auto_levels_stretches_low_contrast_to_full_range(low_contrast_rgb):
    grey = to_luminance(low_contrast_rgb)

    levels = auto_levels(grey)
    stretched = apply_levels(grey, levels)

    assert levels.black == pytest.approx(grey.min(), abs=1e-6)
    assert levels.white == pytest.approx(grey.max(), abs=1e-6)
    assert levels.gamma == pytest.approx(1.0)
    assert stretched.min() == pytest.approx(0.0, abs=1e-6)
    assert stretched.max() == pytest.approx(1.0, abs=1e-6)


def test_auto_levels_keeps_black_below_white_for_flat_images():
    flat = np.full((4, 4), 0.5, dtype=np.float32)

    levels = auto_levels(flat)

    assert 0.0 <= levels.black < levels.white <= 1.0


def test_auto_levels_handles_pure_white_image():
    levels = auto_levels(np.ones((3, 3), dtype=np.float32))

    assert levels.black < levels.white


def test_auto_levels_handles_pure_black_image():
    levels = auto_levels(np.zeros((3, 3), dtype=np.float32))

    assert levels.black < levels.white


def test_auto_levels_handles_empty_input():
    assert auto_levels(np.zeros((0, 0), dtype=np.float32)).is_identity


def test_auto_levels_ignores_stray_extremes():
    grey = np.full((10, 10), 0.5, dtype=np.float32)
    grey[0, 0] = 0.0
    grey[0, 1] = 1.0

    levels = auto_levels(grey, clip_percent=0.5)

    assert levels.black > 0.0
    assert levels.white < 1.0


# --------------------------------------------------------------------------
# histogram (LM-11)
# --------------------------------------------------------------------------
def test_histogram_counts_every_pixel(ramp_rgb):
    grey = to_luminance(ramp_rgb)

    counts = histogram(grey)

    assert counts.shape == (256,)
    assert counts.sum() == grey.size


def test_histogram_honours_bin_count(ramp_rgb):
    grey = to_luminance(ramp_rgb)

    assert histogram(grey, bins=16).shape == (16,)


def test_histogram_places_extremes_in_edge_bins():
    grey = np.array([0.0, 1.0], dtype=np.float32)

    counts = histogram(grey, bins=4)

    assert counts[0] == 1
    assert counts[-1] == 1


def test_histogram_of_flat_mid_grey(ramp_rgb):
    grey = np.full((4, 4), 0.5, dtype=np.float32)

    counts = histogram(grey, bins=256)

    assert counts.sum() == 16
    assert counts.argmax() in (127, 128)


# --------------------------------------------------------------------------
# resample (LM-12)
# --------------------------------------------------------------------------
@pytest.mark.parametrize("grid_shape", [(5, 7), (64, 32), (17, 9), (2, 2)])
def test_resample_output_shape_is_exact(grid_shape, ramp_rgb):
    assert resample(to_luminance(ramp_rgb), grid_shape).shape == grid_shape


def test_resample_axis_zero_is_x(ramp_rgb):
    # The fixture ramps along image width, so after transposition it must ramp along
    # axis 0 (X) and stay constant along axis 1 (Y).
    result = resample(to_luminance(ramp_rgb), (8, 4))

    row_means = result.mean(axis=1)
    assert np.all(np.diff(row_means) > 0)
    assert np.allclose(result, row_means[:, np.newaxis], atol=1e-5)


def test_resample_preserves_a_constant_image():
    grey = np.full((32, 32), 0.25, dtype=np.float32)

    result = resample(grey, (6, 6))

    np.testing.assert_allclose(result, 0.25, atol=1e-5)


def test_resample_downsample_of_checkerboard_is_mid_grey():
    rows, cols = np.indices((64, 64))
    checker = ((rows + cols) % 2).astype(np.float32)

    result = resample(checker, (3, 3))

    np.testing.assert_allclose(result, 0.5, atol=0.01)


def test_resample_upsamples_without_error():
    grey = np.linspace(0.0, 1.0, 4, dtype=np.float32)[np.newaxis, :].repeat(4, axis=0)

    result = resample(grey, (40, 40))

    assert result.shape == (40, 40)
    assert 0.0 <= result.min() <= result.max() <= 1.0


def test_resample_returns_float32(ramp_rgb):
    assert resample(to_luminance(ramp_rgb), (5, 5)).dtype == np.float32


def test_resample_rejects_non_2d_input(ramp_rgb):
    with pytest.raises(ValueError, match="2D luminance"):
        resample(ramp_rgb, (4, 4))


@pytest.mark.parametrize("grid_shape", [(0, 4), (4, 0), (-1, 3)])
def test_resample_rejects_non_positive_grid(grid_shape, ramp_rgb):
    with pytest.raises(ValueError, match="grid_shape must be positive"):
        resample(to_luminance(ramp_rgb), grid_shape)


def test_resample_handles_single_row_source(column_rgb):
    result = resample(to_luminance(column_rgb), (6, 6))

    assert result.shape == (6, 6)
