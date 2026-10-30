"""Lithomaker — convert a 2D image into a 3D-printable lithophane STL.

Public API (see TODO.md, decision D12)::

    from lithomaker import LithoConfig, LevelsConfig, convert

    result = convert("photo.png", "out.stl", LithoConfig(input="photo.png"))

``convert``, ``build_heightmap`` and ``build_mesh`` are resolved lazily (PEP 562) so
that ``import lithomaker`` stays cheap and does not pull in NumPy/Pillow. This also
lets the package import cleanly while the processing core is still being written.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .config import LevelsConfig, LithoConfig

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "LevelsConfig",
    "LithoConfig",
    "build_heightmap",
    "build_mesh",
    "convert",
]

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .core import build_heightmap, build_mesh, convert

_LAZY_CORE_NAMES = frozenset({"build_heightmap", "build_mesh", "convert"})


def __getattr__(name: str) -> Any:
    if name in _LAZY_CORE_NAMES:
        from . import core

        value = getattr(core, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
