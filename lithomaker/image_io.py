"""Image loading, levels adjustment and resampling (TODO.md LM-10/11/12).

Everything here works on plain NumPy arrays and Pillow:

* :func:`load_image` decodes any supported raster into ``float32`` RGB in ``[0, 1]``.
* :func:`to_luminance`, :func:`apply_levels`, :func:`auto_levels` and
  :func:`histogram` operate on the luminance channel (photo-editor "levels").
* :func:`resample` fits luminance onto the mesh grid.

No file here allocates a mesh or writes STL; that is :mod:`lithomaker.mesh`.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from .config import LevelsConfig

__all__ = [
    "ImageLoadError",
    "SUPPORTED_EXTENSIONS",
    "apply_levels",
    "auto_levels",
    "histogram",
    "load_image",
    "resample",
    "to_luminance",
]

PathLike = Union[str, Path]

#: File extensions :func:`load_image` will attempt to decode.
SUPPORTED_EXTENSIONS = frozenset(
    {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}
)

#: Resampling filter used by :func:`resample`. ``Resampling`` only exists on Pillow
#: >= 9.1, so fall back to the module-level alias on older versions (requires-python
#: floor is Pillow >= 9).
_LANCZOS = getattr(Image, "Resampling", Image).LANCZOS


class ImageLoadError(ValueError):
    """Raised when an image cannot be opened, decoded or validated.

    Subclasses :class:`ValueError` so the CLI can catch it alongside configuration
    errors and print the message without a traceback.
    """


def _apply_icc_profile(image: Image.Image) -> Image.Image:
    """Best-effort conversion of an embedded ICC profile to sRGB.

    If anything goes wrong (missing LittleCMS, exotic profile, unusual mode) the image
    is returned untouched — colour management is a nicety, not a correctness
    requirement for a lithophane.
    """
    icc_bytes = image.info.get("icc_profile")
    if not icc_bytes:
        return image
    try:  # pragma: no cover - depends on optional LittleCMS availability
        from PIL import ImageCms

        src_profile = ImageCms.ImageCmsProfile(io.BytesIO(icc_bytes))
        dst_profile = ImageCms.createProfile("sRGB")
        # ``profileToProfile`` needs a concrete mode; RGB is what we ultimately want.
        working = image if image.mode in ("RGB", "RGBA") else image.convert("RGB")
        return ImageCms.profileToProfile(
            working, src_profile, dst_profile, outputMode="RGB"
        )
    except Exception:
        return image


def load_image(path: PathLike) -> np.ndarray:
    """Load ``path`` and return ``float32`` RGB in ``[0, 1]`` with shape ``(H, W, 3)``.

    The extension is validated against :data:`SUPPORTED_EXTENSIONS` *and* the bytes
    are actually decoded, so a mislabelled file is rejected rather than silently
    producing garbage. EXIF orientation is applied and every input mode
    (``L``/``LA``/``P``/``RGBA``/``CMYK``/16-bit) is normalised to RGB.
    """
    path = Path(path)
    if not path.exists():
        raise ImageLoadError(f"image not found: {path}")
    if path.is_dir():
        raise ImageLoadError(f"expected an image file but got a directory: {path}")

    extension = path.suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ImageLoadError(
            f"unsupported image format {extension or '(none)'!r}: {path}\n"
            f"supported formats: {supported}"
        )

    try:
        with Image.open(path) as image:
            image.load()  # force a full decode now, inside the ``with`` block
            transposed = ImageOps.exif_transpose(image)
            oriented = transposed if transposed is not None else image
            oriented = _apply_icc_profile(oriented)
            rgb = oriented.convert("RGB")
            array = np.asarray(rgb, dtype=np.float32) / 255.0
    except ImageLoadError:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ImageLoadError(f"could not read image {path}: {exc}") from exc

    # ``convert("RGB")`` always yields (H, W, 3); assert it for clarity.
    if array.ndim != 3 or array.shape[2] != 3:
        raise ImageLoadError(f"unexpected pixel layout after decode: {array.shape}")
    return np.ascontiguousarray(array, dtype=np.float32)


def to_luminance(rgb: np.ndarray) -> np.ndarray:
    """Relative luminance using Rec. 709 weights, ``float32`` in ``[0, 1]``.

    Accepts ``(H, W, 3)`` RGB, ``(H, W, 1)`` or already-2D luminance arrays (returned
    unchanged apart from dtype).
    """
    array = np.asarray(rgb, dtype=np.float32)
    if array.ndim == 2:
        return array.copy()
    if array.ndim == 3 and array.shape[2] == 1:
        return array[:, :, 0].copy()
    if array.ndim != 3 or array.shape[2] < 3:
        raise ValueError(f"expected an RGB image, got shape {array.shape}")
    red, green, blue = array[:, :, 0], array[:, :, 1], array[:, :, 2]
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    return luminance.astype(np.float32)


def apply_levels(lum: np.ndarray, levels: LevelsConfig) -> np.ndarray:
    """Apply a photo-editor levels adjustment to ``lum``.

    Remaps ``[black, white]`` to ``[0, 1]``, clamps, then applies ``gamma``. With the
    default identity levels this is a clamped no-op.
    """
    array = np.asarray(lum, dtype=np.float32)
    black, white, gamma = levels.black, levels.white, levels.gamma
    scaled = (array - black) / (white - black)
    np.clip(scaled, 0.0, 1.0, out=scaled)
    if gamma != 1.0:
        scaled = np.power(scaled, gamma)
    return scaled.astype(np.float32)


def auto_levels(lum: np.ndarray, clip_percent: float = 0.5) -> LevelsConfig:
    """Derive black/white points from the ``clip_percent`` / ``100-clip`` percentiles.

    Clipping a fraction of a percent at each end ignores stray pure-black/white pixels
    (lens caps, specular highlights) so a low-contrast photo is stretched to fill the
    range. ``gamma`` is left at 1.0 (decision D7). Degenerate inputs (constant images,
    or percentiles that collapse) are nudged to keep ``black < white``.
    """
    array = np.asarray(lum, dtype=np.float32).ravel()
    if array.size == 0:
        return LevelsConfig()

    low = float(np.percentile(array, clip_percent))
    high = float(np.percentile(array, 100.0 - clip_percent))

    black = min(max(low, 0.0), 1.0)
    white = min(max(high, 0.0), 1.0)

    if white <= black:
        # Expand by a hair rather than raising: a flat image should still convert.
        if black < 1.0:
            white = min(1.0, black + 1e-4)
        else:
            black = white - 1e-4
    return LevelsConfig(black=black, white=white, gamma=1.0)


def histogram(lum: np.ndarray, bins: int = 256) -> np.ndarray:
    """Count luminance values into ``bins`` buckets spanning ``[0, 1]``."""
    array = np.asarray(lum, dtype=np.float32)
    counts, _ = np.histogram(array, bins=bins, range=(0.0, 1.0))
    return counts


def resample(lum: np.ndarray, grid_shape: tuple) -> np.ndarray:
    """Resize ``lum`` (shape ``(H, W)``) to exactly ``grid_shape`` = ``(nx, ny)``.

    Pillow's float-mode Lanczos filter is used so there is no SciPy dependency and the
    result is area-averaged rather than point-sampled: downsampling a fine
    black/white checkerboard collapses to mid-grey instead of aliasing to extremes.

    The returned array is transposed relative to image convention so that **axis 0 is
    X** (width, ``nx``) and **axis 1 is Y** (height, ``ny``) — matching the mesh grid,
    where ``values[i, j]`` sits at ``(x_i, y_j)``.
    """
    nx, ny = int(grid_shape[0]), int(grid_shape[1])
    if nx < 1 or ny < 1:
        raise ValueError(f"grid_shape must be positive, got {grid_shape!r}")

    array = np.asarray(lum, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"expected a 2D luminance array, got shape {array.shape}")

    image = Image.fromarray(np.ascontiguousarray(array), mode="F")
    resized = image.resize((nx, ny), resample=_LANCZOS)  # PIL size = (width, height)
    # PIL gives (ny, nx); transpose so axis 0 is X to match the mesh contract.
    result = np.asarray(resized, dtype=np.float32).T
    # Lanczos rings around hard edges and can overshoot [0, 1]; clamp so downstream
    # height mapping never leaves the [z_min, z_max] band.
    return np.ascontiguousarray(np.clip(result, 0.0, 1.0))
