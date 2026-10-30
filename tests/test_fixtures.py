"""Smoke tests for the shared fixtures in ``conftest.py`` (TODO.md, LM-3).

These do not test product code; they protect the fixtures themselves, so that a broken
fixture fails here instead of producing confusing failures in the Phase B pipeline tests.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image


def test_ramp_rgb_shape_and_direction(ramp_rgb):
    assert ramp_rgb.shape == (16, 16, 3)
    assert ramp_rgb.dtype == np.float32
    assert ramp_rgb.min() == 0.0 and ramp_rgb.max() == 1.0
    # Ramps left -> right, and is constant down each column.
    assert ramp_rgb[0, 0, 0] < ramp_rgb[0, -1, 0]
    np.testing.assert_allclose(ramp_rgb[0], ramp_rgb[-1])


def test_tiny_rgb_has_four_distinct_levels(tiny_rgb):
    assert tiny_rgb.shape == (2, 2, 3)
    levels = sorted(tiny_rgb[:, :, 0].ravel().tolist())
    assert levels == [0.0, 0.25, 0.75, 1.0]


def test_column_rgb_is_a_single_row(column_rgb):
    assert column_rgb.shape == (1, 8, 3)


def test_solid_images(solid_black, solid_white):
    assert solid_black.shape == (8, 8, 3)
    assert solid_black.max() == 0.0
    assert solid_white.min() == 1.0


def test_low_contrast_is_narrow(low_contrast_rgb):
    assert low_contrast_rgb.min() == pytest.approx(0.4)
    assert low_contrast_rgb.max() == pytest.approx(0.6)


# --------------------------------------------------------------------------
# file factories
# --------------------------------------------------------------------------
def test_make_image_file_writes_readable_png(make_image_file, ramp_rgb):
    path = make_image_file(ramp_rgb, "ramp.png")
    assert path.exists() and path.suffix == ".png"

    with Image.open(path) as image:
        assert image.mode == "RGB"
        assert image.size == (16, 16)
        pixels = np.asarray(image)
    # 0.0 -> 0 and 1.0 -> 255 survive quantization unchanged.
    assert pixels[0, 0, 0] == 0
    assert pixels[0, -1, 0] == 255


def test_make_image_file_quantizes_floats(make_image_file, tiny_rgb):
    path = make_image_file(tiny_rgb, "tiny.png")
    with Image.open(path) as image:
        pixels = np.asarray(image)
    assert pixels[0, 1, 0] == round(0.25 * 255)
    assert pixels[1, 0, 0] == round(0.75 * 255)


def test_make_image_file_honours_extension(make_image_file, ramp_rgb):
    path = make_image_file(ramp_rgb, "photo.jpg", quality=95)
    with Image.open(path) as image:
        assert image.format == "JPEG"


def test_make_image_file_grayscale_mode(make_image_file, ramp_rgb):
    path = make_image_file(ramp_rgb[:, :, 0], "gray.png", mode="L")
    with Image.open(path) as image:
        assert image.mode == "L"


def test_make_image_file_creates_subdirectories(make_image_file, ramp_rgb):
    path = make_image_file(ramp_rgb, "nested/dir/ramp.png")
    assert path.exists()


def test_save_image_handles_arbitrary_modes(save_image):
    palette = Image.new("P", (4, 4))
    path = save_image(palette, "palette.png")
    with Image.open(path) as image:
        assert image.mode == "P"


# --------------------------------------------------------------------------
# committed example
# --------------------------------------------------------------------------
def test_example_png_is_generated_and_full_range(example_png):
    assert example_png.exists()
    assert example_png.name == "sample.png"

    with Image.open(example_png) as image:
        assert image.size == (256, 256)
        assert image.mode == "RGB"
        gray = np.asarray(image.convert("L"))

    # The reference patches guarantee the histogram spans the whole range, which is
    # what makes the sample useful for exercising levels adjustments.
    assert gray.min() == 0
    assert gray.max() == 255
