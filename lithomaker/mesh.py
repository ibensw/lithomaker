"""Mesh data structure, grid mesh generation and STL export (LM-14/15/16).

A lithophane is a "height field in a box": a displaced top surface, four vertical side
walls and a flat bottom at ``z = 0``. :func:`build_grid_mesh` turns an ``(nx, ny)`` grid
of absolute top-surface Z values into a closed, outward-wound :class:`Mesh`; the STL
writers serialise it.

Coordinate conventions (see TODO.md decision D10):

* ``x`` runs from ``0`` to ``size_x`` mm, ``y`` from ``0`` to ``size_y`` mm.
* ``z = 0`` is the bottom face on the bed; the lit surface rises in ``+Z``.
* ``z_grid[i, j]`` is the top-surface height at ``(x_i, y_j)`` — axis 0 is X.
"""

from __future__ import annotations

import os
import struct
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np

__all__ = ["Mesh", "build_grid_mesh", "read_stl", "write_stl"]

PathLike = Union[str, Path]

#: Triangles with a cross-product area below this (mm^2) are dropped (decision D9).
_AREA_EPSILON = 1e-12


@dataclass
class Mesh:
    """Triangle soup with shared vertices.

    ``vertices`` is ``(N, 3)`` float64, ``triangles`` is ``(M, 3)`` int64 indexing into
    ``vertices`` with counter-clockwise winding (outward normals for a closed solid).
    """

    vertices: np.ndarray
    triangles: np.ndarray

    def __post_init__(self) -> None:
        self.vertices = np.ascontiguousarray(self.vertices, dtype=np.float64)
        self.triangles = np.ascontiguousarray(self.triangles, dtype=np.int64)
        if self.vertices.ndim != 2 or self.vertices.shape[1] != 3:
            raise ValueError(f"vertices must be (N, 3), got {self.vertices.shape}")
        if self.triangles.ndim != 2 or self.triangles.shape[1] != 3:
            raise ValueError(f"triangles must be (M, 3), got {self.triangles.shape}")

    # -- counts -------------------------------------------------------------
    @property
    def vertex_count(self) -> int:
        return int(self.vertices.shape[0])

    @property
    def triangle_count(self) -> int:
        return int(self.triangles.shape[0])

    # -- geometry -----------------------------------------------------------
    def bounds(self) -> tuple:
        """``((x0, y0, z0), (x1, y1, z1))`` axis-aligned bounding box."""
        if self.vertex_count == 0:
            return ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        lo = self.vertices.min(axis=0)
        hi = self.vertices.max(axis=0)
        return (tuple(lo.tolist()), tuple(hi.tolist()))

    def is_closed(self) -> bool:
        """True when every undirected edge is shared by exactly two triangles."""
        if self.triangle_count == 0:
            return False
        t = self.triangles
        edges = np.concatenate(
            [t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]], axis=0
        )
        edges = np.sort(edges, axis=1)
        _, counts = np.unique(edges, axis=0, return_counts=True)
        return bool(np.all(counts == 2))

    def volume(self) -> float:
        """Signed volume via the tetrahedron sum; positive for outward normals."""
        v = self.vertices
        t = self.triangles
        a, b, c = v[t[:, 0]], v[t[:, 1]], v[t[:, 2]]
        signed = np.einsum("ij,ij->i", a, np.cross(b, c))
        return float(signed.sum() / 6.0)


def _drop_degenerate(vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """Remove zero-area triangles (decision D9)."""
    if triangles.shape[0] == 0:
        return triangles
    a, b, c = vertices[triangles[:, 0]], vertices[triangles[:, 1]], vertices[triangles[:, 2]]
    area2 = np.linalg.norm(np.cross(b - a, c - a), axis=1)
    keep = area2 > _AREA_EPSILON
    return triangles[keep]


def build_grid_mesh(z_grid: np.ndarray, size_x: float, size_y: float) -> Mesh:
    """Build a closed height-field mesh from an ``(nx, ny)`` grid of top-surface Z.

    ``z_grid[i, j]`` is the absolute top height (mm) at ``(x_i, y_j)``. Produces a
    displaced top surface (CCW seen from ``+Z``), four side skirts down to ``z = 0``
    and a flat bottom (normal ``-Z``); all outward-wound and sharing vertices so the
    result is watertight. Degenerate faces are dropped (D9). Fully vectorised.
    """
    z = np.asarray(z_grid, dtype=np.float64)
    if z.ndim != 2:
        raise ValueError(f"z_grid must be 2D, got shape {z.shape}")
    nx, ny = z.shape
    if nx < 2 or ny < 2:
        raise ValueError(f"grid must be at least 2x2 vertices, got {nx}x{ny}")
    if size_x <= 0 or size_y <= 0:
        raise ValueError(f"size must be positive, got {size_x}x{size_y}")

    xs = np.linspace(0.0, float(size_x), nx)
    ys = np.linspace(0.0, float(size_y), ny)

    # Vertex layout: top surface first [0, nx*ny), then the flat bottom [nx*ny, 2*nx*ny).
    # Index of grid node (i, j) for the top is ``i*ny + j`` (matches C-order of z).
    xx, yy = np.meshgrid(xs, ys, indexing="ij")  # (nx, ny)
    top_xy = np.stack([xx, yy], axis=-1).reshape(-1, 2)
    top_vertices = np.concatenate([top_xy, z.reshape(-1, 1)], axis=1)
    base = nx * ny
    bottom_vertices = np.concatenate([top_xy, np.zeros((base, 1))], axis=1)
    vertices = np.concatenate([top_vertices, bottom_vertices], axis=0)

    def top(i, j):
        return i * ny + j

    def bottom(i, j):
        return base + i * ny + j

    # -- top surface: two CCW triangles per cell (normal +Z) ----------------
    ci, cj = np.meshgrid(np.arange(nx - 1), np.arange(ny - 1), indexing="ij")
    v00, v10 = top(ci, cj), top(ci + 1, cj)
    v01, v11 = top(ci, cj + 1), top(ci + 1, cj + 1)
    top_tris = np.concatenate(
        [
            np.stack([v00, v10, v11], axis=-1).reshape(-1, 3),
            np.stack([v00, v11, v01], axis=-1).reshape(-1, 3),
        ],
        axis=0,
    )

    # -- bottom face: same cells, reversed winding (normal -Z) --------------
    b00, b10 = bottom(ci, cj), bottom(ci + 1, cj)
    b01, b11 = bottom(ci, cj + 1), bottom(ci + 1, cj + 1)
    bottom_tris = np.concatenate(
        [
            np.stack([b00, b11, b10], axis=-1).reshape(-1, 3),
            np.stack([b00, b01, b11], axis=-1).reshape(-1, 3),
        ],
        axis=0,
    )

    # -- side walls ---------------------------------------------------------
    # Each wall is a strip of quads; winding is chosen so normals point outward.
    i = np.arange(nx - 1)
    # y = 0 edge -> normal -Y
    ti, ti1 = top(i, 0), top(i + 1, 0)
    bi, bi1 = bottom(i, 0), bottom(i + 1, 0)
    wall_front = np.concatenate(
        [
            np.stack([ti, bi, bi1], axis=-1).reshape(-1, 3),
            np.stack([ti, bi1, ti1], axis=-1).reshape(-1, 3),
        ],
        axis=0,
    )
    # y = max edge -> normal +Y (reversed winding of the front wall)
    tj, tj1 = top(i, ny - 1), top(i + 1, ny - 1)
    bj, bj1 = bottom(i, ny - 1), bottom(i + 1, ny - 1)
    wall_back = np.concatenate(
        [
            np.stack([tj, bj1, bj], axis=-1).reshape(-1, 3),
            np.stack([tj, tj1, bj1], axis=-1).reshape(-1, 3),
        ],
        axis=0,
    )

    j = np.arange(ny - 1)
    # x = 0 edge -> normal -X
    tk, tk1 = top(0, j), top(0, j + 1)
    bk, bk1 = bottom(0, j), bottom(0, j + 1)
    wall_left = np.concatenate(
        [
            np.stack([tk, bk1, bk], axis=-1).reshape(-1, 3),
            np.stack([tk, tk1, bk1], axis=-1).reshape(-1, 3),
        ],
        axis=0,
    )
    # x = max edge -> normal +X
    tm, tm1 = top(nx - 1, j), top(nx - 1, j + 1)
    bm, bm1 = bottom(nx - 1, j), bottom(nx - 1, j + 1)
    wall_right = np.concatenate(
        [
            np.stack([tm, bm, bm1], axis=-1).reshape(-1, 3),
            np.stack([tm, bm1, tm1], axis=-1).reshape(-1, 3),
        ],
        axis=0,
    )

    triangles = np.concatenate(
        [top_tris, bottom_tris, wall_front, wall_back, wall_left, wall_right],
        axis=0,
    )
    triangles = _drop_degenerate(vertices, triangles)
    return Mesh(vertices=vertices, triangles=triangles)


def _triangle_normals(vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """Unit outward normals per triangle; degenerate faces map to ``(0, 0, 0)``."""
    a, b, c = vertices[triangles[:, 0]], vertices[triangles[:, 1]], vertices[triangles[:, 2]]
    normals = np.cross(b - a, c - a)
    lengths = np.linalg.norm(normals, axis=1)
    nonzero = lengths > 0
    normals[nonzero] /= lengths[nonzero, np.newaxis]
    normals[~nonzero] = 0.0
    return normals


# Binary STL record: normal(3*f4) + 3 vertices(9*f4) + uint16 attribute = 50 bytes.
_BINARY_RECORD = np.dtype(
    [("normal", "<f4", 3), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")]
)


def _write_binary_stl(mesh: Mesh, handle) -> None:
    header = b"Lithomaker binary STL".ljust(80, b" ")
    # Never begin with "solid" or some ASCII auto-detectors misread the file.
    if header[:5].lower() == b"solid":
        header = b"_" + header[1:]
    handle.write(header)

    triangles = mesh.triangles
    handle.write(struct.pack("<I", mesh.triangle_count))

    records = np.zeros(mesh.triangle_count, dtype=_BINARY_RECORD)
    records["normal"] = _triangle_normals(mesh.vertices, triangles)
    records["vertices"] = mesh.vertices[triangles]
    records["attribute"] = 0
    handle.write(records.tobytes())


def _write_ascii_stl(mesh: Mesh, handle, name: str = "lithomaker") -> None:
    normals = _triangle_normals(mesh.vertices, mesh.triangles)
    verts = mesh.vertices
    lines = [f"solid {name}"]
    for tri_index, tri in enumerate(mesh.triangles):
        nx, ny, nz = normals[tri_index]
        lines.append(f"  facet normal {nx:.6g} {ny:.6g} {nz:.6g}")
        lines.append("    outer loop")
        for vertex_index in tri:
            x, y, z = verts[vertex_index]
            lines.append(f"      vertex {x:.6g} {y:.6g} {z:.6g}")
        lines.append("    endloop")
        lines.append("  endfacet")
    lines.append(f"endsolid {name}")
    handle.write(("\n".join(lines) + "\n").encode("ascii"))


def _atomic_write(path: Path, writer) -> None:
    """Write via a temp file in the same directory, then ``os.replace`` atomically."""
    path = Path(path)
    directory = path.parent if str(path.parent) else Path(".")
    directory.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".litho-", suffix=".stl", dir=directory)
    try:
        with os.fdopen(fd, "wb") as handle:
            writer(handle)
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def write_stl(mesh: Mesh, path: PathLike, binary: bool = True) -> None:
    """Write ``mesh`` to ``path`` as binary (default) or ASCII STL, atomically."""
    if binary:
        _atomic_write(Path(path), lambda handle: _write_binary_stl(mesh, handle))
    else:
        _atomic_write(Path(path), lambda handle: _write_ascii_stl(mesh, handle))


def read_stl(path: PathLike) -> Mesh:
    """Read a **binary** STL back into a :class:`Mesh` (for round-trip checks).

    STL stores each triangle's three vertices, but our writer emits identical float32
    bytes for shared corners, so the soup is welded back with ``np.unique`` (exact
    comparison). That restores the top/bottom/edge topology, which lets a reloaded
    file answer :meth:`Mesh.is_closed` and :meth:`Mesh.volume` just like the original.
    """
    data = Path(path).read_bytes()
    if len(data) < 84:
        raise ValueError(f"file too small to be a binary STL: {path}")
    count = struct.unpack("<I", data[80:84])[0]
    expected_size = 84 + _BINARY_RECORD.itemsize * count
    if len(data) < expected_size:
        raise ValueError(
            f"truncated binary STL: {path} declares {count} triangles "
            f"({expected_size} bytes) but is only {len(data)} bytes"
        )
    body = np.frombuffer(data, dtype=_BINARY_RECORD, count=count, offset=84)
    soup = body["vertices"].reshape(-1, 3).astype(np.float64)
    vertices, inverse = np.unique(soup, axis=0, return_inverse=True)
    triangles = inverse.reshape(-1, 3)
    return Mesh(vertices=vertices, triangles=triangles)
