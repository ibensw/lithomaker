"""Shared test fixtures for Lithomaker.

Images are built in memory (or into ``tmp_path``) rather than committed as binaries, so
the repository stays small and every test controls its own pixel values exactly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pytest
from PIL import Image

EXAMPLE_SAMPLE = Path(__file__).resolve().parent.parent / "examples" / "sample.png"


def _to_uint8(array: np.ndarray) -> np.ndarray:
    """Quantize a float [0, 1] or int array to uint8 with rounding."""
    if array.dtype == np.uint8:
        return array
    if np.issubdtype(array.dtype, np.floating):
        return np.clip(np.round(array * 255.0), 0, 255).astype(np.uint8)
    return np.clip(array, 0, 255).astype(np.uint8)


@pytest.fixture
def make_image_file(tmp_path: Path) -> Callable[..., Path]:
    """Factory writing an array to an image file and returning its path.

    ``array`` may be ``(H, W)`` (grayscale) or ``(H, W, 3)`` (RGB); float arrays in
    [0, 1] are quantized to uint8. ``mode`` overrides the inferred Pillow mode.
    """

    def _make(
        array: np.ndarray,
        name: str = "image.png",
        mode: str | None = None,
        **save_kwargs,
    ) -> Path:
        data = _to_uint8(np.asarray(array))
        if mode is None:
            mode = "L" if data.ndim == 2 else "RGB"
        image = Image.fromarray(data, mode=mode)
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        image.save(path, **save_kwargs)
        return path

    return _make


@pytest.fixture
def save_image(tmp_path: Path) -> Callable[..., Path]:
    """Factory writing an existing PIL image (any mode) and returning its path."""

    def _save(image: Image.Image, name: str = "image.png", **save_kwargs) -> Path:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        image.save(path, **save_kwargs)
        return path

    return _save


@pytest.fixture
def ramp_rgb() -> np.ndarray:
    """16x16 RGB image whose luminance ramps 0 -> 1 left to right."""
    ramp = np.linspace(0.0, 1.0, 16, dtype=np.float32)
    return np.repeat(ramp[np.newaxis, :, np.newaxis], 16, axis=0).repeat(3, axis=2)


@pytest.fixture
def tiny_rgb() -> np.ndarray:
    """2x2 RGB image with four known, distinct luminance levels."""
    levels = np.array(
        [
            [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
            [[0.75, 0.75, 0.75], [1.0, 1.0, 1.0]],
        ],
        dtype=np.float32,
    )
    return levels


@pytest.fixture
def column_rgb() -> np.ndarray:
    """1x8 RGB image — an axis with a single row, for degenerate shape tests."""
    ramp = np.linspace(0.0, 1.0, 8, dtype=np.float32)[np.newaxis, :, np.newaxis]
    return ramp.repeat(1, axis=0).repeat(3, axis=2)


@pytest.fixture
def solid_black() -> np.ndarray:
    """8x8 pure-black RGB image (should yield the maximum material thickness)."""
    return np.zeros((8, 8, 3), dtype=np.float32)


@pytest.fixture
def solid_white() -> np.ndarray:
    """8x8 pure-white RGB image (should yield the minimum material thickness)."""
    return np.ones((8, 8, 3), dtype=np.float32)


@pytest.fixture
def low_contrast_rgb() -> np.ndarray:
    """8x8 image squeezed into the 0.4..0.6 range, for auto-levels tests."""
    ramp = np.linspace(0.4, 0.6, 8, dtype=np.float32)
    return np.repeat(ramp[np.newaxis, :, np.newaxis], 8, axis=0).repeat(3, axis=2)


@pytest.fixture
def example_png() -> Path:
    """Path to ``examples/sample.png``, generated on demand if it is missing."""
    if not EXAMPLE_SAMPLE.exists():
        import importlib.util

        script = EXAMPLE_SAMPLE.parent.parent / "tools" / "make_sample.py"
        spec = importlib.util.spec_from_file_location("_make_sample", script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.main([])
    return EXAMPLE_SAMPLE
