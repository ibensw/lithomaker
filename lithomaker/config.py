"""Parameter contract shared by the CLI and the GUI.

This module is the single source of truth for *every* user-tunable parameter. Both
frontends build one :class:`LithoConfig` and hand it to :func:`lithomaker.core.convert`;
neither frontend is allowed to do its own validation. That is what makes plan §7's
"CLI/GUI parity" requirement enforceable by a test (see TODO.md, LM-40).

All lengths are in millimetres (TODO.md, decision D4). STL itself is unitless; we
document mm and every slicer interprets it that way.

Errors are raised as :class:`ConfigError` (a ``ValueError`` subclass) with a message
that is safe to show verbatim in a terminal *or* a message box.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NamedTuple, Union

__all__ = [
    "ASPECT_TOLERANCE",
    "MAX_GRID_VERTICES",
    "ConfigError",
    "Footprint",
    "LevelsConfig",
    "LithoConfig",
]

PathLike = Union[str, Path]

#: Hard ceiling on ``nx * ny`` grid vertices (decision D3). Reaching this in practice
#: means the user asked for sub-nozzle detail on a large print, which produces an
#: unusable file and a very slow conversion. v1 fails fast instead of decimating.
MAX_GRID_VERTICES = 4_000_000

#: Relative deviation from the image aspect ratio that is tolerated before the user
#: is warned about stretching (decision D2).
ASPECT_TOLERANCE = 0.01


class ConfigError(ValueError):
    """Invalid or contradictory parameters.

    Subclasses :class:`ValueError` so callers that don't care can catch the builtin
    type, while the CLI can catch this narrow class to produce a usage error.
    """


class Footprint(NamedTuple):
    """Effective physical X/Y size in mm, plus an optional aspect-ratio warning."""

    x: float
    y: float
    aspect_warning: str | None = None


def _as_float(name: str, value: Any) -> float:
    """Coerce to a finite float, rejecting bools and non-numeric input."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{name} must be a number, got {type(value).__name__}: {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise ConfigError(f"{name} must be a finite number, got {value!r}")
    return result


def _as_int(name: str, value: Any) -> int:
    """Coerce to an int. Integral floats (e.g. ``32.0`` from JSON) are accepted."""
    if isinstance(value, bool):
        raise ConfigError(f"{name} must be an integer, got a boolean: {value!r}")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    raise ConfigError(f"{name} must be an integer, got {type(value).__name__}: {value!r}")


def _as_path(name: str, value: Any) -> Path:
    if isinstance(value, Path):
        path = value
    elif isinstance(value, str):
        # Check the raw string first: Path("") silently normalizes to Path(".") and
        # would otherwise sail through as a "valid" path.
        if not value.strip():
            raise ConfigError(f"{name} must not be empty")
        path = Path(value)
    else:
        raise ConfigError(f"{name} must be a file path, got {type(value).__name__}: {value!r}")
    return path


def _axis_count(size_mm: float, dot_size_mm: float) -> int:
    """Number of mesh vertices along one axis for a given size and dot pitch.

    ``+1`` because N dots span N+1 vertices; at least 2 so a mesh is always possible.
    """
    return max(2, int(round(size_mm / dot_size_mm)) + 1)


def _min_dot_size(size_x: float, size_y: float, max_vertices: int) -> float:
    """Smallest dot size whose grid fits within ``max_vertices``, rounded up to 0.01 mm.

    Only used to build an actionable error message, so a short search is fine.
    """
    dot = math.sqrt(size_x * size_y / max(max_vertices, 1))
    for _ in range(200):
        if _axis_count(size_x, dot) * _axis_count(size_y, dot) <= max_vertices:
            break
        dot *= 1.02
    else:  # pragma: no cover - unreachable for finite inputs
        return dot
    # Round up so the suggestion is guaranteed to satisfy the limit.
    return math.ceil(dot * 100.0) / 100.0


@dataclass(frozen=True)
class LevelsConfig:
    """Photo-editor "levels" adjustment applied to luminance before meshing.

    * ``black`` — input luminance mapped to 0 (clamped below).
    * ``white`` — input luminance mapped to 1 (clamped above).
    * ``gamma`` — midtone curve, applied after the black/white remap.
    * ``auto`` — derive black/white from the image histogram, ignoring the other three
      (decision D7).
    """

    black: float = 0.0
    white: float = 1.0
    gamma: float = 1.0
    auto: bool = False

    def __post_init__(self) -> None:
        black = _as_float("levels black point", self.black)
        white = _as_float("levels white point", self.white)
        gamma = _as_float("levels gamma", self.gamma)

        if not 0.0 <= black <= 1.0:
            raise ConfigError(f"levels black point must be within 0..1, got {black:g}")
        if not 0.0 <= white <= 1.0:
            raise ConfigError(f"levels white point must be within 0..1, got {white:g}")
        if black >= white:
            raise ConfigError(
                f"levels black point ({black:g}) must be lower than white point ({white:g})"
            )
        if gamma <= 0.0:
            raise ConfigError(f"levels gamma must be greater than 0, got {gamma:g}")

        object.__setattr__(self, "black", black)
        object.__setattr__(self, "white", white)
        object.__setattr__(self, "gamma", gamma)
        object.__setattr__(self, "auto", bool(self.auto))

    @property
    def is_identity(self) -> bool:
        """True when the adjustment is a no-op (ignoring ``auto``)."""
        return self.black == 0.0 and self.white == 1.0 and self.gamma == 1.0

    def describe(self) -> str:
        if self.auto:
            return "auto levels"
        return f"levels black={self.black:g} white={self.white:g} gamma={self.gamma:g}"


@dataclass(frozen=True)
class LithoConfig:
    """Everything needed to turn one image into one STL file.

    ``output`` may be ``None``, in which case :attr:`resolved_output` derives
    ``<input_stem>.stl`` per decision D5. Input file *existence* is deliberately not
    checked here — that is :mod:`lithomaker.image_io`'s job, so this class stays pure
    and cheap to build on every GUI slider move.
    """

    input: PathLike
    output: PathLike | None = None
    levels: LevelsConfig | dict | None = field(default_factory=LevelsConfig)
    min_layers: int = 4
    max_layers: int = 32
    thickness: float = 0.2
    size_x: float = 120.0
    size_y: float | None = None
    dot_size: float = 0.4
    base_thickness: float = 1.0
    ascii: bool = False

    # -- validation ---------------------------------------------------------
    def __post_init__(self) -> None:
        object.__setattr__(self, "input", _as_path("input image path", self.input))
        object.__setattr__(
            self,
            "output",
            None if self.output is None else _as_path("output path", self.output),
        )

        levels = self.levels
        if levels is None:
            levels = LevelsConfig()
        elif isinstance(levels, dict):
            levels = LevelsConfig(**levels)
        elif not isinstance(levels, LevelsConfig):
            raise ConfigError(
                f"levels must be a LevelsConfig or mapping, got {type(levels).__name__}"
            )
        object.__setattr__(self, "levels", levels)

        min_layers = _as_int("min layers", self.min_layers)
        max_layers = _as_int("max layers", self.max_layers)
        if min_layers < 1:
            raise ConfigError(f"min layers must be at least 1, got {min_layers}")
        if max_layers < 1:
            raise ConfigError(f"max layers must be at least 1, got {max_layers}")
        if max_layers < min_layers:
            raise ConfigError(
                f"max layers ({max_layers}) must be greater than or equal to min layers ({min_layers})"
            )

        thickness = _as_float("layer thickness", self.thickness)
        if thickness <= 0.0:
            raise ConfigError(f"layer thickness must be greater than 0 mm, got {thickness:g}")

        size_x = _as_float("output size X", self.size_x)
        if size_x <= 0.0:
            raise ConfigError(f"output size X must be greater than 0 mm, got {size_x:g}")

        size_y = self.size_y
        if size_y is not None:
            size_y = _as_float("output size Y", size_y)
            if size_y <= 0.0:
                raise ConfigError(f"output size Y must be greater than 0 mm, got {size_y:g}")

        dot_size = _as_float("dot size", self.dot_size)
        if dot_size <= 0.0:
            raise ConfigError(f"dot size must be greater than 0 mm, got {dot_size:g}")

        base_thickness = _as_float("base thickness", self.base_thickness)
        if base_thickness < 0.0:
            raise ConfigError(
                f"base thickness must be 0 mm or greater, got {base_thickness:g}"
            )

        object.__setattr__(self, "min_layers", min_layers)
        object.__setattr__(self, "max_layers", max_layers)
        object.__setattr__(self, "thickness", thickness)
        object.__setattr__(self, "size_x", size_x)
        object.__setattr__(self, "size_y", size_y)
        object.__setattr__(self, "dot_size", dot_size)
        object.__setattr__(self, "base_thickness", base_thickness)
        object.__setattr__(self, "ascii", bool(self.ascii))

    # -- derived Z geometry (decision D10) ----------------------------------
    @property
    def z_min(self) -> float:
        """Material thickness above the base at the brightest pixel, in mm."""
        return self.min_layers * self.thickness

    @property
    def z_max(self) -> float:
        """Material thickness above the base at the darkest pixel, in mm."""
        return self.max_layers * self.thickness

    @property
    def total_height(self) -> float:
        """Total model height from the bed (``z = 0``) to the tallest point, in mm."""
        return self.base_thickness + self.z_max

    def height_to_z(self, normalized: Any) -> Any:
        """Map inverted brightness ``h`` in [0, 1] to absolute Z in mm from the bed.

        ``h = 0`` (dark source pixel) → ``z_max``; ``h = 1`` (bright source pixel) →
        ``z_min``. Keeps the standard transmissive convention of decision D1: bright
        areas are thin and let light through.

        Implemented with plain arithmetic so it works on scalars and on NumPy arrays
        without importing NumPy here.
        """
        return self.base_thickness + self.z_min + (1.0 - normalized) * (self.z_max - self.z_min)

    # -- derived XY geometry -------------------------------------------------
    def _check_image_size(self, image_width: int, image_height: int) -> tuple:
        width = _as_int("image width", image_width)
        height = _as_int("image height", image_height)
        if width < 1 or height < 1:
            raise ConfigError(f"image dimensions must be positive, got {width}x{height}")
        return width, height

    def resolved_size(self, image_width: int, image_height: int) -> Footprint:
        """Effective footprint in mm, honouring the image aspect ratio (decision D2).

        When ``size_y`` is unset the height is derived from the image, so the result is
        always aspect-correct. When both axes are given we stretch rather than fail, but
        attach a warning the caller can surface.
        """
        width, height = self._check_image_size(image_width, image_height)
        image_aspect = height / width

        if self.size_y is None:
            return Footprint(self.size_x, self.size_x * image_aspect)

        requested_aspect = self.size_y / self.size_x
        deviation = abs(requested_aspect - image_aspect) / image_aspect
        if deviation <= ASPECT_TOLERANCE:
            return Footprint(self.size_x, self.size_y)

        warning = (
            f"output size {self.size_x:g}x{self.size_y:g} mm does not match the image "
            f"aspect ratio ({width}x{height} px = {image_aspect:.3f}); the image will be "
            f"stretched. Omit --size-y to keep the aspect ratio automatically."
        )
        return Footprint(self.size_x, self.size_y, warning)

    def grid_shape(self, image_width: int, image_height: int) -> tuple:
        """Vertex counts ``(nx, ny)`` for the mesh grid.

        Derived from the resolved footprint and ``dot_size``, and guarded against
        runaway mesh sizes (decision D3) *before* anything is allocated.
        """
        footprint = self.resolved_size(image_width, image_height)
        nx = _axis_count(footprint.x, self.dot_size)
        ny = _axis_count(footprint.y, self.dot_size)

        if nx * ny > MAX_GRID_VERTICES:
            suggestion = _min_dot_size(footprint.x, footprint.y, MAX_GRID_VERTICES)
            raise ConfigError(
                f"mesh too large: {nx}x{ny} = {nx * ny:,} vertices exceeds the "
                f"{MAX_GRID_VERTICES:,}-vertex limit at dot size {self.dot_size:g} mm. "
                f"Use --dot-size {suggestion:g} mm or larger, or reduce --size-x/--size-y."
            )
        return nx, ny

    # -- output path (decision D5) -------------------------------------------
    @property
    def resolved_output(self) -> Path:
        """Output STL path, defaulting to ``<input_stem>.stl`` next to the working dir."""
        if self.output is not None:
            return Path(self.output)
        return Path(self.input).with_suffix(".stl")
