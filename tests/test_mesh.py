"""Tests for the mesh data structure, grid generation and STL I/O (LM-14/15/16)."""

from __future__ import annotations

import struct
import time

import numpy as np
import pytest

import lithomaker.mesh as mesh_module
from lithomaker.mesh import Mesh, build_grid_mesh, read_stl, write_stl

# A 4x4 grid spanning 10 x 6 mm with a constant 2 mm top surface is the workhorse
# fixture: 2 * 3 * 3 = 18 top triangles, 18 bottom, 4 * 3 * 2 = 24 wall triangles.
NX, NY = 4, 4
SIZE_X, SIZE_Y = 10.0, 6.0
FLAT_Z = 2.0

EXPECTED_TRIANGLES = 2 * (NX - 1) * (NY - 1) * 2 + 4 * (NX - 1) * 2


def flat_grid(value: float = FLAT_Z, shape: tuple = (NX, NY)) -> np.ndarray:
    return np.full(shape, value, dtype=np.float64)


@pytest.fixture
def flat_mesh() -> Mesh:
    return build_grid_mesh(flat_grid(), SIZE_X, SIZE_Y)


@pytest.fixture
def ramp_mesh() -> Mesh:
    ramp = np.linspace(1.0, 5.0, NX * NY, dtype=np.float64).reshape(NX, NY)
    return build_grid_mesh(ramp, SIZE_X, SIZE_Y)


def triangle_normals(mesh: Mesh) -> np.ndarray:
    v = mesh.vertices[mesh.triangles]
    return np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])


# --------------------------------------------------------------------------
# Mesh data structure (LM-14)
# --------------------------------------------------------------------------
def test_mesh_reports_counts():
    mesh = Mesh(vertices=np.zeros((3, 3)), triangles=np.array([[0, 1, 2]]))

    assert mesh.vertex_count == 3
    assert mesh.triangle_count == 1


def test_mesh_coerces_dtypes_and_layout():
    mesh = Mesh(
        vertices=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32),
        triangles=np.array([[0, 1, 2]], dtype=np.int32),
    )

    assert mesh.vertices.dtype == np.float64
    assert mesh.triangles.dtype == np.int64
    assert mesh.vertices.flags["C_CONTIGUOUS"]


def test_mesh_rejects_bad_vertex_shape():
    with pytest.raises(ValueError, match=r"vertices must be \(N, 3\)"):
        Mesh(vertices=np.zeros((3, 2)), triangles=np.array([[0, 1, 2]]))


def test_mesh_rejects_bad_triangle_shape():
    with pytest.raises(ValueError, match=r"triangles must be \(M, 3\)"):
        Mesh(vertices=np.zeros((3, 3)), triangles=np.array([[0, 1]]))


def test_bounds_of_empty_mesh_are_zero():
    mesh = Mesh(vertices=np.zeros((0, 3)), triangles=np.zeros((0, 3), dtype=np.int64))

    assert mesh.bounds() == ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))


def test_bounds_are_axis_aligned_min_max():
    vertices = np.array([[1.0, 2.0, 3.0], [-1.0, 5.0, 0.0], [4.0, 0.0, -2.0]])
    mesh = Mesh(vertices=vertices, triangles=np.array([[0, 1, 2]]))

    assert mesh.bounds() == ((-1.0, 0.0, -2.0), (4.0, 5.0, 3.0))


def test_is_closed_is_false_for_empty_mesh():
    empty = Mesh(vertices=np.zeros((0, 3)), triangles=np.zeros((0, 3), dtype=np.int64))

    assert empty.is_closed() is False


def test_is_closed_is_false_for_a_single_triangle():
    mesh = Mesh(
        vertices=np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]]),
        triangles=np.array([[0, 1, 2]]),
    )

    assert mesh.is_closed() is False


def test_volume_of_unit_tetrahedron_is_one_sixth():
    mesh = Mesh(
        vertices=np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        triangles=np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]]),
    )

    assert mesh.volume() == pytest.approx(1.0 / 6.0)


def test_volume_sign_follows_winding():
    vertices = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    outward = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])

    positive = Mesh(vertices=vertices, triangles=outward).volume()
    negative = Mesh(vertices=vertices, triangles=outward[:, ::-1]).volume()

    assert positive == pytest.approx(-negative)


# --------------------------------------------------------------------------
# build_grid_mesh (LM-15)
# --------------------------------------------------------------------------
def test_build_grid_mesh_rejects_non_2d_grid():
    with pytest.raises(ValueError, match="must be 2D"):
        build_grid_mesh(np.zeros(4), SIZE_X, SIZE_Y)


@pytest.mark.parametrize("shape", [(1, 4), (4, 1), (1, 1)])
def test_build_grid_mesh_requires_two_vertices_per_axis(shape):
    with pytest.raises(ValueError, match="at least 2x2"):
        build_grid_mesh(np.ones(shape), SIZE_X, SIZE_Y)


@pytest.mark.parametrize("size", [(0.0, 5.0), (5.0, 0.0), (-1.0, 5.0)])
def test_build_grid_mesh_rejects_non_positive_size(size):
    with pytest.raises(ValueError, match="size must be positive"):
        build_grid_mesh(flat_grid(), *size)


def test_build_grid_mesh_footprint_matches_request(flat_mesh):
    (x0, y0, z0), (x1, y1, z1) = flat_mesh.bounds()

    assert (x0, y0, z0) == pytest.approx((0.0, 0.0, 0.0))
    assert (x1, y1) == pytest.approx((SIZE_X, SIZE_Y))
    assert z1 == pytest.approx(FLAT_Z)


def test_build_grid_mesh_keeps_top_and_bottom_vertices_separate(flat_mesh):
    assert flat_mesh.vertex_count == 2 * NX * NY


def test_build_grid_mesh_triangle_count_for_flat_top(flat_mesh):
    assert flat_mesh.triangle_count == EXPECTED_TRIANGLES


def test_build_grid_mesh_triangle_count_for_smallest_grid():
    mesh = build_grid_mesh(np.ones((2, 2)), 4.0, 4.0)

    # 2 top + 2 bottom + 4 walls, 2 triangles per wall.
    assert mesh.triangle_count == 2 + 2 + 8


def test_build_grid_mesh_drops_degenerate_walls_at_zero_height():
    mesh = build_grid_mesh(flat_grid(0.0), SIZE_X, SIZE_Y)

    # Walls have zero area when the top surface sits on the bed; only top and bottom stay.
    assert mesh.triangle_count == 2 * (NX - 1) * (NY - 1) * 2
    assert mesh.bounds()[1][2] == pytest.approx(0.0)


def test_build_grid_mesh_is_closed(ramp_mesh):
    assert ramp_mesh.is_closed() is True


def test_build_grid_mesh_has_positive_volume(flat_mesh):
    assert flat_mesh.volume() == pytest.approx(SIZE_X * SIZE_Y * FLAT_Z)
    assert flat_mesh.volume() > 0


def test_build_grid_mesh_volume_matches_prism_for_ramp(ramp_mesh):
    mean_height = np.linspace(1.0, 5.0, NX * NY).mean()

    assert ramp_mesh.volume() == pytest.approx(SIZE_X * SIZE_Y * mean_height, rel=1e-9)


def test_build_grid_mesh_never_dips_below_the_bed(ramp_mesh):
    assert ramp_mesh.vertices[:, 2].min() == pytest.approx(0.0)


def test_build_grid_mesh_keeps_every_face_of_a_ramp(ramp_mesh):
    # A monotone ramp has no flat walls or coincident triangles, so nothing is dropped.
    assert ramp_mesh.triangle_count == EXPECTED_TRIANGLES
    assert np.all(np.linalg.norm(triangle_normals(ramp_mesh), axis=1) > 0)


def test_top_surface_winding_faces_plus_z(flat_mesh):
    top_count = 2 * (NX - 1) * (NY - 1)

    assert np.all(triangle_normals(flat_mesh)[:top_count, 2] > 0)


def test_bottom_surface_winding_faces_minus_z(flat_mesh):
    top_count = 2 * (NX - 1) * (NY - 1)
    bottom_count = top_count

    assert np.all(triangle_normals(flat_mesh)[top_count : top_count + bottom_count, 2] < 0)


def test_side_walls_face_away_from_the_centre(flat_mesh):
    top_count = 2 * (NX - 1) * (NY - 1) * 2
    walls = flat_mesh.triangles[top_count:]
    centroids = flat_mesh.vertices[walls].mean(axis=1)
    normals = triangle_normals(flat_mesh)[top_count:]
    outward = centroids[:, :2] - np.array([SIZE_X / 2.0, SIZE_Y / 2.0])

    # Wall normals have no Z component and point away from the footprint centre.
    assert np.allclose(normals[:, 2], 0.0)
    assert np.all(np.einsum("ij,ij->i", normals[:, :2], outward) >= 0.0)


def test_build_grid_mesh_is_watertight_for_a_noisy_surface():
    rng = np.random.default_rng(1234)
    grid = rng.uniform(0.1, 3.0, size=(9, 7))

    mesh = build_grid_mesh(grid, 12.0, 9.0)

    assert mesh.is_closed() is True
    assert mesh.volume() > 0


def test_build_grid_mesh_handles_a_single_valley():
    grid = np.full((6, 6), 3.0)
    grid[2:4, 2:4] = 1.0

    mesh = build_grid_mesh(grid, 6.0, 6.0)

    assert mesh.is_closed() is True
    assert mesh.volume() > 0


def test_build_grid_mesh_is_deterministic():
    grid = np.linspace(0.5, 3.0, 25).reshape(5, 5)

    first = build_grid_mesh(grid, 5.0, 5.0)
    second = build_grid_mesh(grid, 5.0, 5.0)

    np.testing.assert_array_equal(first.vertices, second.vertices)
    np.testing.assert_array_equal(first.triangles, second.triangles)


def test_build_grid_mesh_is_fast_enough_for_300x300():
    grid = np.random.default_rng(0).uniform(0.5, 4.0, size=(300, 300))

    start = time.perf_counter()
    mesh = build_grid_mesh(grid, 120.0, 120.0)
    elapsed = time.perf_counter() - start

    assert mesh.is_closed() is True
    assert elapsed < 5.0, f"300x300 grid took {elapsed:.2f}s"


# --------------------------------------------------------------------------
# binary STL export (LM-16)
# --------------------------------------------------------------------------
def test_binary_stl_has_the_spec_layout(flat_mesh, tmp_path):
    path = tmp_path / "flat.stl"

    write_stl(flat_mesh, path)

    data = path.read_bytes()
    assert len(data) == 84 + 50 * flat_mesh.triangle_count
    assert len(data[:80]) == 80
    assert not data.lower().startswith(b"solid")
    assert struct.unpack("<I", data[80:84])[0] == flat_mesh.triangle_count


def test_binary_stl_records_are_50_bytes(tmp_path):
    path = tmp_path / "one.stl"
    mesh = Mesh(
        vertices=np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]]),
        triangles=np.array([[0, 1, 2]]),
    )

    write_stl(mesh, path)

    assert path.stat().st_size == 84 + 50


def test_binary_stl_single_triangle_matches_hand_built_reference(tmp_path):
    vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    mesh = Mesh(vertices=vertices, triangles=np.array([[0, 1, 2]]))
    path = tmp_path / "reference.stl"

    write_stl(mesh, path)

    header = b"Lithomaker binary STL".ljust(80, b" ")
    record = struct.pack(
        "<12fH",
        0.0, 0.0, 1.0,
        0.0, 0.0, 0.0,
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0,
    )
    expected = header + struct.pack("<I", 1) + record
    assert path.read_bytes() == expected


def test_binary_stl_normals_are_unit_length(ramp_mesh, tmp_path):
    path = tmp_path / "ramp.stl"
    write_stl(ramp_mesh, path)

    data = path.read_bytes()
    records = np.frombuffer(data, dtype=mesh_module._BINARY_RECORD, offset=84)
    lengths = np.linalg.norm(records["normal"].astype(np.float64), axis=1)

    np.testing.assert_allclose(lengths, 1.0, atol=1e-5)


def test_binary_stl_writes_zero_normal_for_degenerate_triangle(tmp_path):
    mesh = Mesh(
        vertices=np.array([[0.0, 0, 0], [1, 0, 0], [2, 0, 0]]),
        triangles=np.array([[0, 1, 2]]),
    )
    path = tmp_path / "degenerate.stl"

    write_stl(mesh, path)

    data = path.read_bytes()
    records = np.frombuffer(data, dtype=mesh_module._BINARY_RECORD, offset=84)
    np.testing.assert_array_equal(records["normal"][0], np.zeros(3, dtype=np.float32))


def test_binary_stl_round_trip_preserves_counts_and_bounds(ramp_mesh, tmp_path):
    path = tmp_path / "round.stl"

    write_stl(ramp_mesh, path)
    reloaded = read_stl(path)

    assert reloaded.triangle_count == ramp_mesh.triangle_count
    assert reloaded.vertex_count == ramp_mesh.vertex_count
    assert reloaded.is_closed() is True
    assert reloaded.volume() == pytest.approx(ramp_mesh.volume(), rel=1e-5)
    np.testing.assert_allclose(reloaded.bounds()[0], ramp_mesh.bounds()[0], atol=1e-5)
    np.testing.assert_allclose(reloaded.bounds()[1], ramp_mesh.bounds()[1], atol=1e-5)


def test_write_stl_creates_parent_directories(flat_mesh, tmp_path):
    path = tmp_path / "nested" / "deeper" / "out.stl"

    write_stl(flat_mesh, path)

    assert path.is_file()


def test_write_stl_overwrites_atomically(flat_mesh, tmp_path):
    path = tmp_path / "out.stl"
    path.write_bytes(b"stale")

    write_stl(flat_mesh, path)

    assert path.stat().st_size == 84 + 50 * flat_mesh.triangle_count
    assert not list(tmp_path.glob(".litho-*"))


def test_write_stl_cleans_up_temp_file_on_failure(flat_mesh, tmp_path, monkeypatch):
    def boom(mesh, handle):
        handle.write(b"partial")
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(mesh_module, "_write_binary_stl", boom)
    path = tmp_path / "out.stl"

    with pytest.raises(RuntimeError, match="disk on fire"):
        write_stl(flat_mesh, path)

    assert not path.exists()
    assert not list(tmp_path.glob(".litho-*"))


def test_read_stl_rejects_tiny_files(tmp_path):
    path = tmp_path / "tiny.stl"
    path.write_bytes(b"nope")

    with pytest.raises(ValueError, match="too small"):
        read_stl(path)


def test_read_stl_rejects_truncated_body(tmp_path):
    path = tmp_path / "truncated.stl"
    path.write_bytes(b"\0" * 80 + struct.pack("<I", 4) + b"\0" * 50)

    with pytest.raises(ValueError, match="truncated"):
        read_stl(path)


# --------------------------------------------------------------------------
# ASCII STL export (LM-16)
# --------------------------------------------------------------------------
def parse_ascii_stl(text: str) -> Mesh:
    normals, vertices, triangles = [], [], []
    current = []
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "facet":
            normals.append([float(value) for value in parts[2:5]])
        elif parts[0] == "vertex":
            current.append([float(value) for value in parts[1:4]])
            if len(current) == 3:
                triangles.append([len(vertices) + 0, len(vertices) + 1, len(vertices) + 2])
                vertices.extend(current)
                current = []
    return Mesh(vertices=np.array(vertices), triangles=np.array(triangles))


def test_ascii_stl_has_the_expected_tokens(flat_mesh, tmp_path):
    path = tmp_path / "flat_ascii.stl"

    write_stl(flat_mesh, path, binary=False)

    text = path.read_text(encoding="ascii")
    assert text.startswith("solid ")
    assert text.rstrip().endswith("endsolid lithomaker")
    assert text.count("facet normal") == flat_mesh.triangle_count
    assert text.count("outer loop") == flat_mesh.triangle_count
    assert text.count("endloop") == flat_mesh.triangle_count
    assert text.count("endfacet") == flat_mesh.triangle_count
    assert text.count("\n      vertex") == 3 * flat_mesh.triangle_count


def test_ascii_stl_parses_back_to_the_same_geometry(ramp_mesh, tmp_path):
    path = tmp_path / "ramp_ascii.stl"

    write_stl(ramp_mesh, path, binary=False)
    parsed = parse_ascii_stl(path.read_text(encoding="ascii"))

    assert parsed.triangle_count == ramp_mesh.triangle_count
    np.testing.assert_allclose(parsed.bounds()[0], ramp_mesh.bounds()[0], atol=1e-4)
    np.testing.assert_allclose(parsed.bounds()[1], ramp_mesh.bounds()[1], atol=1e-4)


def test_default_writer_is_binary(flat_mesh, tmp_path):
    path = tmp_path / "default.stl"

    write_stl(flat_mesh, path)

    assert not path.read_text(encoding="latin-1").lower().startswith("solid")
    assert path.stat().st_size == 84 + 50 * flat_mesh.triangle_count
