"""Shared processing core: image -> heightmap -> mesh -> STL (LM-13 / LM-17).

Both frontends call exactly one function, :func:`convert` (TODO.md decision D12), so the
CLI and the GUI cannot drift apart::

    result = convert("photo.png", "lithophane.stl", LithoConfig(input="photo.png"))

Pipeline order follows plan §4: load → luminance → levels/auto → resample onto the mesh
grid → Z mapping. Decision D1 (bright pixels become *thin* material) is applied by
:meth:`lithomaker.config.LithoConfig.height_to_z`, so :class:`Heightmap` carries
**normalized brightness** (``1.0`` = brightest source pixel = thinnest material) and
never a pre-inverted thickness. That keeps a single convention across the package.

All lengths are millimetres (decision D4).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Union

import numpy as np

from .config import LithoConfig
from .image_io import apply_levels, auto_levels, load_image, resample, to_luminance
from .mesh import Mesh, build_grid_mesh, write_stl

__all__ = [
    "ConversionResult",
    "Heightmap",
    "build_heightmap",
    "build_mesh",
    "convert",
]

PathLike = Union[str, Path]
ProgressCallback = Callable[[str, float], None]


@dataclass
class Heightmap:
    """Normalized brightness on the mesh grid, plus the geometry it maps onto.

    ``values`` is ``(nx, ny)`` ``float32`` in ``[0, 1]`` with **axis 0 = X** and
    ``values[i, j]`` sitting at ``(x_i, y_j)``. ``1.0`` means "bright source pixel",
    which :meth:`LithoConfig.height_to_z` turns into the *minimum* material thickness
    (decision D1). The inversion therefore lives in the Z mapping, not in this array.
    """

    values: np.ndarray
    grid_shape: tuple
    resolved_size: tuple
    z_min: float
    z_max: float
    warnings: list = field(default_factory=list)


@dataclass
class ConversionResult:
    """Everything a frontend needs to report success (LM-17)."""

    output_path: Path
    vertex_count: int
    triangle_count: int
    file_size: int
    z_min: float
    z_max: float
    total_height: float
    resolved_size: tuple
    grid_shape: tuple
    warnings: list = field(default_factory=list)


def _report(progress: ProgressCallback | None, message: str, fraction: float) -> None:
    """Emit one progress event when a callback was supplied."""
    if progress is not None:
        progress(message, fraction)


def _as_rgb(image: PathLike | np.ndarray) -> np.ndarray:
    """Accept a path or an in-memory image and return ``(H, W, 3)`` float32 RGB."""
    if isinstance(image, np.ndarray):
        array = np.asarray(image)
        if array.ndim != 3 or array.shape[2] != 3:
            raise ValueError(f"expected an (H, W, 3) RGB image, got shape {array.shape}")
        if np.issubdtype(array.dtype, np.integer):
            array = array.astype(np.float32) / 255.0
        array = array.astype(np.float32, copy=False)
        np.clip(array, 0.0, 1.0, out=array)
        return np.ascontiguousarray(array)
    return load_image(image)


def build_heightmap(
    image: PathLike | np.ndarray, config: LithoConfig
) -> Heightmap:
    """Turn ``image`` into a :class:`Heightmap` sized to the mesh grid (LM-13).

    ``image`` may be a file path (decoded with :func:`lithomaker.image_io.load_image`)
    or an in-memory ``(H, W, 3)`` array. Warnings derived from the parameters — the D2
    aspect-ratio stretch and the D7 "auto levels ignores black/white/gamma" notice —
    are attached to the result instead of printed, so the caller can decide.
    """
    rgb = _as_rgb(image)
    image_height, image_width = rgb.shape[:2]

    footprint = config.resolved_size(image_width, image_height)
    grid_shape = config.grid_shape(image_width, image_height)

    warnings = []
    if footprint.aspect_warning:
        warnings.append(footprint.aspect_warning)

    luminance = to_luminance(rgb)
    levels = config.levels
    if levels.auto:
        if not levels.is_identity:
            warnings.append(
                "auto levels is enabled: black point, white point and gamma are ignored"
            )
        levels = auto_levels(luminance)
    adjusted = apply_levels(luminance, levels)

    # Normalized brightness (1.0 = bright). D1's inversion happens in height_to_z.
    values = resample(adjusted, grid_shape)
    np.clip(values, 0.0, 1.0, out=values)

    return Heightmap(
        values=values,
        grid_shape=(int(grid_shape[0]), int(grid_shape[1])),
        resolved_size=(footprint.x, footprint.y),
        z_min=config.z_min,
        z_max=config.z_max,
        warnings=warnings,
    )


def build_mesh(heightmap: Heightmap, config: LithoConfig) -> Mesh:
    """Displace the heightmap into a closed, outward-wound :class:`Mesh` (LM-15)."""
    z_grid = config.height_to_z(heightmap.values)
    size_x, size_y = heightmap.resolved_size
    return build_grid_mesh(z_grid, size_x, size_y)


def _resolve_config(
    image_path: PathLike, out_path: PathLike | None, config: LithoConfig | None
) -> LithoConfig:
    """Reconcile the explicit paths with the parameter object.

    The positional arguments win: a caller that passes ``out_path`` gets that file even
    if ``config.output`` says otherwise, and the loaded image is always ``image_path``.
    """
    image_path = Path(image_path)
    target = None if out_path is None else Path(out_path)

    if config is None:
        return LithoConfig(input=image_path, output=target)

    if config.input != image_path or (target is not None and config.output != target):
        return replace(
            config,
            input=image_path,
            output=target if target is not None else config.output,
        )
    return config


def convert(
    image_path: PathLike,
    out_path: PathLike | None = None,
    config: LithoConfig | None = None,
    *,
    overwrite: bool = False,
    progress: ProgressCallback | None = None,
) -> ConversionResult:
    """Convert ``image_path`` to an STL and return its statistics (decision D12).

    ``out_path`` defaults to ``<image_stem>.stl`` (D5). Existing files are never
    clobbered unless ``overwrite`` is true — the CLI maps that to ``--force`` and the
    GUI to a confirmation dialog. ``progress(message, fraction)`` is called with a
    monotonically increasing fraction in ``[0, 1]``; CLI callers may pass ``None``.
    """
    settings = _resolve_config(image_path, out_path, config)
    destination = settings.resolved_output

    if destination.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing file: {destination}")

    _report(progress, "loading image", 0.05)
    heightmap = build_heightmap(settings.input, settings)

    _report(progress, "building mesh", 0.55)
    mesh = build_mesh(heightmap, settings)

    _report(progress, "writing STL", 0.85)
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_stl(mesh, destination, binary=not settings.ascii)

    _report(progress, "done", 1.0)
    return ConversionResult(
        output_path=destination,
        vertex_count=mesh.vertex_count,
        triangle_count=mesh.triangle_count,
        file_size=destination.stat().st_size,
        z_min=heightmap.z_min,
        z_max=heightmap.z_max,
        total_height=settings.total_height,
        resolved_size=heightmap.resolved_size,
        grid_shape=heightmap.grid_shape,
        warnings=heightmap.warnings,
    )
