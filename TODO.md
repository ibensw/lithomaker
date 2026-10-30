# Lithomaker — Implementation TODO

Derived from `plan.md`. Tasks are numbered `LM-<n>`; reference them in commits/PRs
(e.g. `feat(core): heightmap generation (LM-14)`).

**How to use this file:** pick the lowest-numbered task whose *Depends on* entries are
all `[x]`. Work only on the files listed in the task's **Files** line to avoid conflicts
with other agents. Check the box and add a one-line note when done.

---

## 0. Resolved decisions (do not re-litigate; see §11 of plan.md)

These settle the plan's open questions so parallel agents stay consistent.

| # | Question | Decision |
|---|----------|----------|
| D1 | Inversion convention | **Bright pixel → thin material** (standard transmissive lithophane). `z = z_max - brightness * (z_max - z_min)` where brightness is inverted luminance. Provide `--invert/--no-invert` later (v2); default fixed for v1. |
| D2 | Aspect ratio | `--size-y` defaults to `size_x * image_h / image_w` (aspect-correct). If **both** `--size-x` and `--size-y` are given and deviate from image aspect by > 1%, **stretch** and emit a warning (do not error). |
| D3 | Mesh size guard | Hard guard: `(nx * ny) > 4_000_000` vertices → fail with a clear error suggesting a larger `--dot-size`. No decimation in v1. |
| D4 | Units | Document millimetres everywhere (STL is unitless). All public functions use mm. |
| D5 | `INPUT_IMAGE` / `OUTPUT` | **Positional** args are canonical per plan §6. `-i/--input` and `-o/--output` are accepted as aliases; supplying both positional and flag for the same value is an error. `OUTPUT` is optional and defaults to `Path(input).with_suffix(".stl")` in the CWD. |
| D6 | `--min-layers` | The plan's input spec requires min layers but the CLI table omits it. **Add `--min-layers` (int, default 4)** for CLI/GUI parity. |
| D7 | Auto levels | Exposed as `--levels auto` **and** as the boolean flag pair `--auto-levels/--no-auto-levels`. When auto is on, `--black/--white/--gamma` are ignored with a warning. |
| D8 | STL format | Binary STL is the default writer; `--ascii` writes ASCII STL (`--format {binary,ascii}` also accepted). |
| D9 | Zero-area face policy | Heightmaps are flat-shaded per triangle; degenerate (zero-area) triangles at perfectly flat regions are **skipped** rather than written, so slicers don't choke. |
| D10 | Bed orientation | `z = 0` is the base plate bottom face; lit face rises in `+Z`. Top surface `z_max = base_thickness + max_layers * thickness` (relative to base bottom). |
| D11 | Python floor | Python ≥ 3.9. Runtime deps: `Pillow`, `numpy`, `click`. GUI uses **tkinter** (stdlib, zero extra dep). |
| D12 | Public API | `lithomaker.core.convert(image_path, out_path, config) -> ConversionResult`; `lithomaker.core.build_mesh(image, config) -> Mesh`; `lithomaker.core.build_heightmap(image, config) -> np.ndarray`. Everything else is internal. |

---

## 1. Phase A — Scaffolding & config contract

### [ ] LM-1 · Project skeleton
**Files:** `pyproject.toml`, `README.md`, `lithomaker/__init__.py`, `tests/__init__.py`, `.gitignore`, `examples/`
Set up a modern `pyproject.toml` (PEP 621, `build-backend = "setuptools.build_meta"`):
- `name = "lithomaker"`, `version = "0.1.0"`, `requires-python = ">=3.9"`
- deps: `Pillow>=9`, `numpy>=1.21`, `click>=8`
- extras: `dev = ["pytest", "pytest-cov", "ruff"]`
- console script: `lithomaker = lithomaker.cli:main`
- `lithomaker/__init__.py` exposes `__version__` and re-exports `convert`, `build_mesh`, `build_heightmap`, `LithoConfig`
**Done when:** `pip install -e .[dev]` succeeds and `python -c "import lithomaker"` works.

**Depends on:** —

### [ ] LM-2 · `config.py` — parameter dataclasses
**Files:** `lithomaker/config.py`, `tests/test_config.py`
Single source of truth for every parameter (CLI and GUI both build one of these).
- `LevelsConfig(black=0.0, white=1.0, gamma=1.0, auto=False)` with `__post_init__` validation:
  `0 <= black < white <= 1`, `gamma > 0`.
- `LithoConfig(input, output=None, levels, min_layers=4, max_layers=32, thickness=0.2,
  size_x=120.0, size_y=None, dot_size=0.4, base_thickness=1.0, ascii=False)`.
  Validate: `max_layers >= min_layers >= 1`, `thickness > 0`, `size_x > 0`, `size_y > 0 | None`,
  `dot_size > 0`, `base_thickness >= 0`.
- Derived properties (per plan §3, D10):
  - `z_min = min_layers * thickness`
  - `z_max = max_layers * thickness`
  - `total_height = base_thickness + z_max`
  - `resolved_size(image_w, image_h) -> (size_x, size_y)` implementing **D2**; returns the
    effective sizes plus an `aspect_warning: str | None`.
  - `grid_shape(image_w, image_h) -> (nx, ny)` = `round(effective_size / dot_size) + 1`, clamped to ≥ 2, and
    enforcing **D3**.
- Raise `ValueError` with human-readable messages (the CLI and GUI both surface these verbatim).
**Done when:** unit tests cover every validation rule and every derived property.

**Depends on:** LM-1

### [ ] LM-3 · Example fixture image
**Files:** `examples/sample.png`, `tools/make_sample.py`
Scripted 256×256 PNG generation (Pillow, no binary committed by hand) containing a vertical
grayscale ramp **plus** a radial gradient so levels/gamma changes are visually obvious.
`tools/make_sample.py` is re-runnable and idempotent. Also add `tests/conftest.py` with
fixtures that build tiny in-memory images (2×2, 1×N, 16×16) — no large binaries in the repo.
**Done when:** `python tools/make_sample.py` regenerates the PNG; `pytest` collects the fixtures.

**Depends on:** LM-1

---

## 2. Phase B — Core pipeline (CLI-only milestone 1)

### [ ] LM-10 · `image_io.py` — load & validate
**Files:** `lithomaker/image_io.py`, `tests/test_image_io.py`
- `load_image(path) -> np.ndarray` returning `float32` RGB in `[0, 1]`, shape `(H, W, 3)`.
- Accept PNG/JPG/JPEG/BMP/TIFF/TIF/WEBP (extension allowlist **and** a real decode attempt).
- Handle: missing file, directory, unsupported extension, corrupt data, EXIF-rotated JPEG
  (apply `ImageOps.exif_transpose`), palette/`L`/`LA`/`RGBA`/`CMYK`/16-bit modes via
  `convert("RGB")`.
- Apply ICC profile → sRGB if present (best-effort; ignore failure).
- `Image.open` in a `with` block so file handles are released (Windows).
- Errors: raise a dedicated `ImageLoadError(ValueError)` so the CLI can print it cleanly.
**Done when:** tests cover each failure mode and each mode conversion.

**Depends on:** LM-1

### [ ] LM-11 · `image_io.py` — luminance + levels + histogram
**Files:** `lithomaker/image_io.py`, `tests/test_image_io.py`
- `to_luminance(rgb) -> np.ndarray` — Rec. 709 (`0.2126R + 0.7152G + 0.0722B`), `float32 [0,1]`.
- `apply_levels(lum, levels) -> np.ndarray` — `((lum - black) / (white - black))`, clamp to `[0,1]`,
  then `** gamma`. Vectorized NumPy only.
- `auto_levels(lum, clip_percent=0.5) -> LevelsConfig` — black/white from the 0.5th/99.5th
  percentile, gamma 1.0 (D7).
- `histogram(lum, bins=256) -> np.ndarray` — used by CLI `--stats` and the GUI preview.
**Done when:** identity check (`black=0, white=1, gamma=1` is a no-op within float eps),
clamping and gamma direction are tested, auto levels on a low-contrast image expand to `[0,1]`.

**Depends on:** LM-2, LM-10

### [ ] LM-12 · `image_io.py` — resample to mesh grid
**Files:** `lithomaker/image_io.py`, `tests/test_image_io.py`
`resample(lum, grid_shape) -> np.ndarray` using Pillow `Image.resize` in `F`/float mode
(no `scipy` dependency), Lanczos by default. Must be **area-preserving-ish and strictly
shape-correct**: output shape equals `grid_shape` exactly. Guard against the "downsampling a
4096px image to 30 vertices" aliasing case (test that a black/white checkerboard resamples
to a mid-gray, not to extreme values).
**Done when:** output shape is exact for up-, down- and non-integer-scale cases; aliasing test passes.

**Depends on:** LM-11

### [ ] LM-13 · `core.py` — heightmap generation
**Files:** `lithomaker/core.py`, `tests/test_core.py`
`build_heightmap(image, config) -> Heightmap` where `Heightmap` is a small dataclass
(`values: np.ndarray[float32]` in `[0,1]`, `grid_shape`, `resolved_size`, `z_min`, `z_max`,
`warnings: list[str]`).
Pipeline order per plan §4: load → luminance → levels/auto → **invert (D1)** →
resample → normalize.
`height_to_z(h, config)` returns absolute Z in mm from the *base bottom* (D10):
`base_thickness + z_min + (1 - h) * (z_max - z_min)` — with `h` the *inverted* brightness;
state in code comments that bright → low Z.
**Done when:** a pure-white image yields the minimum Z and a pure-black image the maximum;
monotonicity is tested; warnings from D2 propagate onto the result.

**Depends on:** LM-12

### [ ] LM-14 · `mesh.py` — grid mesh data structure
**Files:** `lithomaker/mesh.py`, `tests/test_mesh.py`
`@dataclass Mesh: vertices: np.ndarray (N,3) float64 or float32, triangles: np.ndarray (M,3) int64`.
Helpers: `vertex_count`, `triangle_count`, `bounds() -> ((x0,y0,z0),(x1,y1,z1))`,
`is_closed()` (every undirected edge appears exactly twice — needed for watertightness tests),
`volume()` (signed tetrahedron sum, used to prove outward normals).
No I/O in this file; keep it pure data + geometry.

**Depends on:** LM-1

### [ ] LM-15 · `mesh.py` — mesh generation
**Files:** `lithomaker/mesh.py`, `tests/test_mesh.py`
Build the `nx × ny` grid per plan §4 step 6:
- **Top surface:** heightmap-displaced grid, `2 * (nx-1) * (ny-1)` triangles, consistent CCW winding
  when viewed from `+Z`.
- **Side walls:** 4 skirts from the top edge down to `z = 0`, quads split into triangles.
- **Bottom face:** flat plane at `z = 0`, wound so its normal points `-Z`.
- Skip degenerate triangles (**D9**).
- Vectorize with NumPy (no Python loops over vertices); a 300×300 grid must build in < 2 s.
- Origin at `(0, 0, 0)` corner of the base; footprint exactly `size_x × size_y`.
**Done when:** `is_closed()` is `True`; `volume() > 0` (outward normals); bounds match the
requested footprint and `total_height`; a flat heightmap produces exactly the expected
triangle count after degenerate removal.

**Depends on:** LM-13, LM-14

### [ ] LM-16 · `mesh.py` — binary + ASCII STL export
**Files:** `lithomaker/mesh.py`, `tests/test_mesh.py`
- Binary writer per STL spec: 80-byte header (must **not** begin with `solid`), `uint32` triangle
  count, 50 bytes/triangle (float32 normal + 3 × float32 vertices + `uint16` attr, zeroed).
- Compute normals from vertex winding; normalize; zero normal → written as `(0,0,0)` and let the
  slicer recompute (or skip if degenerate per D9 — document the choice).
- ASCII writer with `solid <name>` / `facet normal` / `outer loop` / `vertex` … `endloop` / `endsolid`.
- Write atomically (temp file + `os.replace`) so a crash can't leave a half-written STL.
- `read_stl(path) -> Mesh` for round-trip tests (binary only is enough).
**Done when:** write→read round-trip preserves triangle count and bounds; a binary file written
from a known 1-triangle mesh is byte-identical to a hand-built reference; ASCII parses.

**Depends on:** LM-14

### [ ] LM-17 · `core.py` — `convert()` orchestration + result object
**Files:** `lithomaker/core.py`, `tests/test_core.py`
`convert(image_path, out_path=None, config=...) -> ConversionResult` implementing **D12**.
`ConversionResult`: `output_path`, `vertex_count`, `triangle_count`, `file_size`,
`z_min`, `z_max`, `total_height`, `resolved_size`, `grid_shape`, `warnings`.
- Resolve `out_path` per **D5**; create parent dirs; refuse to overwrite without
  `overwrite=True` (CLI maps this to `--force`).
- Emit progress via an optional `progress: Callable[[str, float], None]` callback
  (GUI uses it; CLI ignores it or prints to stderr under `--verbose`).
- Single code path — the GUI must call exactly this function (plan §7 parity).
**Done when:** an end-to-end test converts the sample PNG, asserts the STL re-reads as
closed & positive-volume, and asserts result stats match the mesh.

**Depends on:** LM-15, LM-16

### [ ] LM-18 · Mesh size guard & clear failures
**Files:** `lithomaker/config.py`, `tests/test_config.py`
Implement **D3** fully: compute `nx * ny` in `grid_shape()` *before* allocating anything,
raise `ValueError` naming the offending numbers and the minimum `dot_size` that would fit.
Add a `--max-vertices` escape hatch? **No** — v1 keeps the fixed 4M cap; just make the message
actionable.
**Done when:** a test asserts the error message contains both the computed vertex count and a
suggested `dot_size`.

**Depends on:** LM-2

---

## 3. Phase C — CLI (milestone 2)

### [ ] LM-20 · `cli.py` — Click command `convert`
**Files:** `lithomaker/cli.py`, `tests/test_cli.py`
Implement plan §6 exactly, with the corrections in **D5–D8**:
```
lithomaker convert [OPTIONS] [INPUT_IMAGE] [OUTPUT.STL]
```
Options: `-i/--input`, `-o/--output`, `-b/--black` (0.0), `-w/--white` (1.0), `-g/--gamma` (1.0),
`--auto-levels/--no-auto-levels`, `--min-layers` (4), `--max-layers` (32), `--thickness` (0.2),
`--size-x` (120.0), `--size-y` (None), `--dot-size` (0.4), `--base-thickness` (1.0),
`--ascii/--binary`, `--force`, `-v/--verbose`, `--stats/--no-stats`.
- Build a `LithoConfig` in the command body; **all** validation lives in `config.py`, the CLI
  only translates `ValueError` → `click.ClickException` (exit code 2 for usage, 1 for runtime).
- `--stats` prints a compact table (vertices, triangles, Z range, footprint, file size) to stdout.
- Nothing but the STL path + stats on stdout by default, so the tool is pipeline-friendly.
**Done when:** `CliRunner` tests cover the happy path, each validation failure, `--force`,
and positional/flag conflict per D5.

**Depends on:** LM-17, LM-18

### [ ] LM-21 · `cli.py` — `--version`, `--help`, error hygiene
**Files:** `lithomaker/cli.py`, `tests/test_cli.py`
`lithomaker --version` from `lithomaker.__version__`; ensure `--help` at group *and* subcommand
level is readable; add an `info` subcommand exposing library versions (`pillow`, `numpy`,
`python`) for bug reports. Failures must never print a raw traceback unless `-v`.
**Depends on:** LM-20

### [ ] LM-22 · CLI docs & runnable examples
**Files:** `README.md`, `docs/cli.md`
- `README.md`: what a lithophane is, install, the plan §6 example, a parameters table identical to
  §6 (+ the D5–D8 additions), a note that STL is unitless-but-mm (**D4**), and an FAQ
  (why is my model inverted / why is it so big / slicer orientation).
- `docs/cli.md`: full `--help` capture, exit codes, three worked examples
  (small test print, full-size 150 mm portrait, CI one-liner).
**Depends on:** LM-21

---

## 4. Phase D — GUI (milestone 3)

### [ ] LM-30 · `gui.py` — application shell & layout
**Files:** `lithomaker/gui.py`
tkinter, plan §7 layout: left = preview canvas, right = parameters (the same widgets grouped as
"Image / Levels / Geometry"), bottom = action bar (Generate STL, output path) + status line.
- **No business logic in the GUI.** It owns widgets and calls `core.*`/`config.*` only.
- Exportable as `python -m lithomaker.gui` and `lithomaker gui` (add the subcommand in LM-20's
  CLI group in the same PR as a one-line `cli.py` hook).
- Thread the conversion with `threading.Thread` + a `queue.Queue` polled by `root.after(50, ...)`
  so the UI never freezes; disable the button during work; progress via the `convert(progress=…)` callback.
**Done when:** launching shows the window and the widgets are hooked to a `LithoConfig` builder that
prints the config on "Generate" (stubbed conversion).

**Depends on:** LM-17

### [ ] LM-31 · GUI — image loading & levels preview
**Files:** `lithomaker/gui.py`
File picker with a format filter matching LM-10; on load show the grayscale image (Pillow →
`ImageTk.PhotoImage`, scaled to fit, aspect preserved) and a 256-bin histogram canvas.
Levels sliders (black/white/gamma) + "Auto" checkbox; live re-render on change, debounced
(`after(150, ...)`) so dragging stays smooth.
**Depends on:** LM-30, LM-11

### [ ] LM-32 · GUI — 2D heightmap preview
**Files:** `lithomaker/gui.py`
Render the (downsampled) heightmap with a grayscale/heat colormap into a fixed-size preview
(e.g. 320 px) using `build_heightmap` on a *coarse* `dot_size` so it stays interactive even for
large images. Show resolved footprint (mm), Z range and grid size as labels.
**Depends on:** LM-31, LM-13

### [ ] LM-33 · GUI — parameters, generate & stats
**Files:** `lithomaker/gui.py`
All CLI parameters as widgets (**parity table in a docstring/comment mapping CLI flag → widget**,
that mapping is what the parity test in LM-40 checks): spinboxes for min/max layers, size X/Y
(with an "lock aspect" checkbox bound to **D2**), thickness, dot size, base thickness, binary/ASCII
radio, `--force` overwrite confirm dialog. On generate: `filedialog.asksaveasfilename`, then
`core.convert(...)`, then a stats panel (vertices, triangles, Z range, file size) + a "show in
folder" convenience. Errors → `messagebox.showerror` with the `ValueError` text (never a traceback).
**Depends on:** LM-32, LM-20

---

## 5. Phase E — Packaging, tests, CI (milestone 4)

### [ ] LM-40 · Test suite consolidation
**Files:** `tests/`, `pyproject.toml`
- Fill remaining gaps so `pytest --cov=lithomaker --cov-fail-under=85` passes.
- Golden-image tests: assert the mapping is monotone for a ramp; assert a synthetic
  4×4 pattern produces a known, hand-computed Z at each vertex.
- **Parity test:** introspect the Click command's params and the GUI's declared parameter map
  (exposed as a module-level dict in `gui.py`, importable without a display) and assert the two
  sets of names are identical → this is how plan §7's "parity" requirement gets enforced in CI.
- Tolerance for floats: `pytest.approx`, and compare STL bytes only for the tiny reference case.
**Depends on:** LM-17, LM-20, LM-33

### [ ] LM-41 · CI workflow
**Files:** `.github/workflows/ci.yml`
Matrix `ubuntu-latest` (+ `windows-latest`) × Python 3.9–3.12: install, `ruff check`,
`pytest --cov`. GUI module must import cleanly on headless Linux (`import tkinter` at module top
is fine, but no `Tk()` at import time — verify with a `--help`-style import test; use
`xvfb-run` only if a real widget test is added, otherwise skip GUI widget tests when
`DISPLAY` is unset).
**Depends on:** LM-40

### [ ] LM-42 · Release / usage polish
**Files:** `README.md`, `CHANGELOG.md`, `pyproject.toml`
`CHANGELOG.md` with `0.1.0` entries; verify `pip install lithomaker` (sdist+wheel via `python -m build`)
and that the console script works from a clean venv; document `pip install .[gui]`-free
tkinter caveat on Debian/Ubuntu (`apt install python3-tk`).
**Depends on:** LM-41

---

## 6. Out of scope for v1 (tracked, do not implement)

- Decimation / adaptive mesh resolution (v2).
- `--invert` toggle for reflective lithophanes (v2).
- Curved/cylindrical lithophanes, lampshade mode.
- Multi-material / color output, 3MF or OBJ export.
- Slicer integration, 3D preview widget.

---

## 7. Suggested parallelization

| Wave | Tasks (safe to run concurrently — disjoint files) |
|------|---------------------------------------------------|
| 1 | LM-1 → then LM-2, LM-3, LM-14 (independent) |
| 2 | LM-10, LM-12* , LM-15† |
| 3 | LM-11, LM-16, LM-18 |
| 4 | LM-13 → LM-17 |
| 5 | LM-20 → LM-21, LM-30 (CLI and GUI in parallel) |
| 6 | LM-22, LM-31 → LM-32 → LM-33 |
| 7 | LM-40 → LM-41 → LM-42 |

\* LM-12 and LM-11 both touch `image_io.py` — either serialize them or have one agent own the file.
† LM-15 needs the heightmap *shape* contract only, so it can start from a synthetic array before LM-13 lands.

**Blocker watch:** LM-17 (core orchestration) blocks every frontend task. Keep it small; land it early
with the pipeline stubbed and fill in as LM-13/15/16 merge.
