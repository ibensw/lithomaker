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
| D6 | `--min-layers` | The plan's input spec requires min layers but the CLI table omits it. **Add `--min-layers` (int, default 1)** for CLI/GUI parity. |
| D7 | Auto levels | Exposed as `--levels auto` **and** as the boolean flag pair `--auto-levels/--no-auto-levels`. When auto is on, `--black/--white/--gamma` are ignored with a warning. |
| D8 | STL format | Binary STL is the default writer; `--ascii` writes ASCII STL (`--format {binary,ascii}` also accepted). |
| D9 | Zero-area face policy | Heightmaps are flat-shaded per triangle; degenerate (zero-area) triangles at perfectly flat regions are **skipped** rather than written, so slicers don't choke. |
| D10 | Bed orientation | `z = 0` is the base plate bottom face; lit face rises in `+Z`. Top surface `z_max = base_thickness + max_layers * thickness` (relative to base bottom). |
| D11 | Python floor | Python ≥ 3.9. Runtime deps: `Pillow`, `numpy`, `click`. GUI uses **tkinter** (stdlib, zero extra dep). |
| D12 | Public API | `lithomaker.core.convert(image_path, out_path, config) -> ConversionResult`; `lithomaker.core.build_mesh(image, config) -> Mesh`; `lithomaker.core.build_heightmap(image, config) -> np.ndarray`. Everything else is internal. |
| D13 | v1 defaults | `min_layers=1`, `max_layers=13`, `base_thickness=0.4` mm (quick test prints out of the box). GUI, CLI and `LithoConfig` share these. |

---

## 1. Phase A — Scaffolding & config contract

### [x] LM-1 · Project skeleton
> Done: `pip install -e .[dev]` succeeds and `import lithomaker` works (`__version__` = 0.1.0); `convert`/`build_mesh`/`build_heightmap` are lazily re-exported (PEP 562) and resolve once `core.py` lands in Phase B.
**Files:** `pyproject.toml`, `README.md`, `lithomaker/__init__.py`, `tests/__init__.py`, `.gitignore`, `examples/`
Set up a modern `pyproject.toml` (PEP 621, `build-backend = "setuptools.build_meta"`):
- `name = "lithomaker"`, `version = "0.1.0"`, `requires-python = ">=3.9"`
- deps: `Pillow>=9`, `numpy>=1.21`, `click>=8`
- extras: `dev = ["pytest", "pytest-cov", "ruff"]`
- console script: `lithomaker = lithomaker.cli:main`
- `lithomaker/__init__.py` exposes `__version__` and re-exports `convert`, `build_mesh`, `build_heightmap`, `LithoConfig`
**Done when:** `pip install -e .[dev]` succeeds and `python -c "import lithomaker"` works.

**Depends on:** —

### [x] LM-2 · `config.py` — parameter dataclasses
> Done: `LevelsConfig`/`LithoConfig` with full `__post_init__` validation, derived `z_min`/`z_max`/`total_height`, `resolved_size` (D2, returns a `Footprint(x, y, aspect_warning)` named tuple), `grid_shape` (D3, 4M-vertex guard with an actionable `--dot-size` suggestion), `resolved_output` (D5). 82 tests in `tests/test_config.py` pass (94 in the suite at that point).
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

### [x] LM-3 · Example fixture image
> Done: `tools/make_sample.py` regenerates a deterministic 256x256 `examples/sample.png` byte-identically (verified by sha256); `.gitignore` now ignores `examples/*.stl` instead of the whole directory, so the sample is versioned as plan §9 expects; `tests/conftest.py` provides in-memory 2x2/1xN/16x16 and solid fixtures (covered by `tests/test_fixtures.py`).
**Files:** `examples/sample.png`, `tools/make_sample.py`
Scripted 256×256 PNG generation (Pillow, no binary committed by hand) containing a vertical
grayscale ramp **plus** a radial gradient so levels/gamma changes are visually obvious.
`tools/make_sample.py` is re-runnable and idempotent. Also add `tests/conftest.py` with
fixtures that build tiny in-memory images (2×2, 1×N, 16×16) — no large binaries in the repo.
**Done when:** `python tools/make_sample.py` regenerates the PNG; `pytest` collects the fixtures.

**Depends on:** LM-1

---

## 2. Phase B — Core pipeline (CLI-only milestone 1)

### [x] LM-10 · `image_io.py` — load & validate
> Done: `load_image` returns `float32` RGB in `[0,1]`; extension allowlist plus a real decode; EXIF transpose, ICC→sRGB best effort, `with Image.open` + `image.load()` so handles are released; `ImageLoadError(ValueError)` with verbatim-safe messages. 27 tests cover every failure mode, every stock mode (L/LA/P/RGBA/CMYK/16-bit), EXIF rotation and the sample PNG (67 in `tests/test_image_io.py` overall).
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

### [x] LM-11 · `image_io.py` — luminance + levels + histogram
> Done: Rec. 709 `to_luminance` (accepts RGB, (H,W,1) or 2D), vectorized `apply_levels` (remap → clamp → gamma), percentile `auto_levels` with flat-image and empty-input guards, 256-bin `histogram`. Tests cover Rec. 709 weights, identity no-op, clamping, gamma direction, low-contrast stretch, stray-extreme clipping and histogram edge bins.
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

### [x] LM-12 · `image_io.py` — resample to mesh grid
> Done: Pillow `F`-mode Lanczos `resample`, output shape equals `grid_shape` exactly, transposed so **axis 0 is X** to match the mesh contract. Lanczos ringing is clamped back to `[0,1]` so heights can never leave `[z_min, z_max]`. Tests cover exact shape for up/down/non-integer scales, the axis convention, constant images, the 64→3 checkerboard aliasing case, and invalid input.
**Files:** `lithomaker/image_io.py`, `tests/test_image_io.py`
`resample(lum, grid_shape) -> np.ndarray` using Pillow `Image.resize` in `F`/float mode
(no `scipy` dependency), Lanczos by default. Must be **area-preserving-ish and strictly
shape-correct**: output shape equals `grid_shape` exactly. Guard against the "downsampling a
4096px image to 30 vertices" aliasing case (test that a black/white checkerboard resamples
to a mid-gray, not to extreme values).
**Done when:** output shape is exact for up-, down- and non-integer-scale cases; aliasing test passes.

**Depends on:** LM-11

### [x] LM-13 · `core.py` — heightmap generation
> Done: `build_heightmap(image, config) -> Heightmap` (path or in-memory RGB array) runs load → luminance → levels/auto → resample and carries `grid_shape`, `resolved_size`, `z_min`/`z_max` and `warnings`. `Heightmap.values` stores **normalized brightness** (1.0 = brightest) because `LithoConfig.height_to_z` owns D1's inversion, keeping one convention package-wide; white image → `base+z_min`, black → `base+z_max`, both tested. D2 aspect and D7 "auto ignores manual levels" warnings travel on the result. 26 tests in `tests/test_core.py`.
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

### [x] LM-14 · `mesh.py` — grid mesh data structure
> Done: `@dataclass Mesh` (float64 `(N,3)` / int64 `(M,3)`, contiguous, shape-validated) with `vertex_count`, `triangle_count`, `bounds()`, `is_closed()` (undirected edge count == 2) and signed `volume()`. No I/O in the file's data section; `tests/test_mesh.py` covers coercion, invalid shapes, empty meshes, closedness and volume sign (48 tests in the file overall).
**Files:** `lithomaker/mesh.py`, `tests/test_mesh.py`
`@dataclass Mesh: vertices: np.ndarray (N,3) float64 or float32, triangles: np.ndarray (M,3) int64`.
Helpers: `vertex_count`, `triangle_count`, `bounds() -> ((x0,y0,z0),(x1,y1,z1))`,
`is_closed()` (every undirected edge appears exactly twice — needed for watertightness tests),
`volume()` (signed tetrahedron sum, used to prove outward normals).
No I/O in this file; keep it pure data + geometry.

**Depends on:** LM-1

### [x] LM-15 · `mesh.py` — mesh generation
> Done: `build_grid_mesh(z_grid, size_x, size_y)` builds top surface + 4 skirts + bottom from a grid where axis 0 is X; fully vectorized, degenerate faces dropped per D9, footprint anchored at `(0,0,0)`. Tests assert exact triangle counts (60 for 4×4, 12 for 2×2, 36 when zero-height walls drop), `is_closed()`, positive volume equal to `size_x·size_y·mean_z`, bounds, winding per surface, and a 300×300 build under 5 s.
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

### [x] LM-16 · `mesh.py` — binary + ASCII STL export
> Done: `write_stl(mesh, path, binary=True)`, atomic temp-file + `os.replace`, 80-byte header that never starts with `solid`, normals from winding (zero for degenerate), ASCII writer, `read_stl` for binary round-trip with explicit too-small/truncated errors. Tests include a byte-identical hand-built single-triangle reference, 50-byte record layout, unit normals, round-trip bounds, parent-dir creation, overwrite, temp cleanup on failure, and an ASCII parse-back.
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

### [x] LM-17 · `core.py` — `convert()` orchestration + result object
> Done: `convert(image_path, out_path=None, config=None, *, overwrite=False, progress=None)` is the single code path both frontends must call; positional paths override `config.input`/`config.output` via `dataclasses.replace`, output defaults to `<stem>.stl` per D5, parent dirs are created, existing files raise `FileExistsError` unless `overwrite=True`, and progress is reported as monotonic `(message, fraction)` events ending in `"done"`. `ConversionResult` carries output path, vertex/triangle counts, file size, Z range, total height, resolved size, grid shape and warnings. `read_stl` now welds shared vertices so the round-trip proves `is_closed()` and volume, per this task's Done-when.
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

### [x] LM-18 · Mesh size guard & clear failures
> Done: implemented in `LithoConfig.grid_shape()` (`lithomaker/config.py:313`) before any allocation, naming `nx`, `ny`, the total, the limit and a `--dot-size` suggestion from `_min_dot_size`; asserted in `tests/test_config.py` (message contents plus a "suggested dot size actually fits" check).
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

### [x] LM-20 · `cli.py` — Click command `convert`
> Done: `lithomaker convert [OPTIONS] [INPUT_IMAGE] [OUTPUT.STL]` with the full plan §6 option set plus D5–D8: positional arguments with `-i/-o` aliases (both spellings at once is a usage error), `--min-layers` (D6), `--levels auto` **and** `--auto-levels/--no-auto-levels` (D7, contradictory spellings rejected), `--ascii/--binary` **and** `--format {binary,ascii}` (D8). Everything is delegated to `LithoConfig` + `core.convert()`; `ConfigError` → `UsageError` (exit 2), I/O failures → `ClickException` (exit 1). stdout is the written path then the stats table (`--no-stats` leaves the path alone); warnings and `-v` progress go to stderr. 46 tests in `tests/test_cli.py` cover the happy path, aliases, every validation failure, `--force`, conflicts and the entry point.
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

### [x] LM-21 · `cli.py` — `--version`, `--help`, error hygiene
> Done: `--version` reads `lithomaker.__version__`; group and `convert` help are tested for readable option coverage (`-h` alias included); `info` reports lithomaker/python/pillow/numpy/click via `importlib.metadata`. Every failure is a one-line `Error: …`; a raw traceback only appears with `-v`, which also re-raises unexpected exceptions (both paths tested). `main()` provides the `lithomaker` console script declared in `pyproject.toml`.
**Files:** `lithomaker/cli.py`, `tests/test_cli.py`
`lithomaker --version` from `lithomaker.__version__`; ensure `--help` at group *and* subcommand
level is readable; add an `info` subcommand exposing library versions (`pillow`, `numpy`,
`python`) for bug reports. Failures must never print a raw traceback unless `-v`.
**Depends on:** LM-20

### [x] LM-22 · CLI docs & runnable examples
> Done: `README.md` now carries the plan §6 parameter table plus the D5–D8 additions, the stdout/stderr contract, the mm-units note (D4) and a FAQ (inverted model, oversized STL/`mesh too large`, slicer orientation, faceting, aspect warning). `docs/cli.md` has the verbatim `--help` capture for the group and `convert`, argument/exit-code tables, three worked examples (40 mm proof print, 150 mm portrait, CI one-liner) and a troubleshooting table; the example numbers were verified against real runs (68×68 grid / 18,492 triangles / 0.9 MB and 376×376 / 565,500 / 28 MB).
**Files:** `README.md`, `docs/cli.md`
- `README.md`: what a lithophane is, install, the plan §6 example, a parameters table identical to
  §6 (+ the D5–D8 additions), a note that STL is unitless-but-mm (**D4**), and an FAQ
  (why is my model inverted / why is it so big / slicer orientation).
- `docs/cli.md`: full `--help` capture, exit codes, three worked examples
  (small test print, full-size 150 mm portrait, CI one-liner).
**Depends on:** LM-21

---

## 4. Phase D — GUI (milestone 3)

### [x] LM-30 · `gui.py` — application shell & layout
> Done: `LithomakerApp(ttk.Frame)` with plan §7 layout (left = grayscale source + heightmap preview + histogram, right = Image/Levels/Geometry/Format groups, bottom = action bar + status + progress) and no business logic — it only builds `LithoConfig` and calls `core.convert()`. Launchable via `lithomaker gui`, `python -m lithomaker.gui` and the `lithomaker-gui` gui-script; conversion runs on a `threading.Thread` feeding a `queue.Queue` drained by `after(50, …)`, with the button disabled while busy. Importing the module creates no `Tk()` root (tested in a subprocess with `DISPLAY` removed).
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

### [x] LM-31 · GUI — image loading & levels preview
> Done: `filedialog` picker filtered to the LM-10 formats; a 128-bin luminance histogram is drawn on a `tk.Canvas` with dashed black/white markers and redraws on `<Configure>` so it tracks the panel width (the separate grayscale source thumbnail was removed on request — the single Preview panel serves that role). Black/white/gamma scales plus an "Auto levels" check button; every change debounces a preview rebuild via `after(150, …)` and updates the numeric labels.
**Files:** `lithomaker/gui.py`
File picker with a format filter matching LM-10; on load show the grayscale image (Pillow →
`ImageTk.PhotoImage`, scaled to fit, aspect preserved) and a 256-bin histogram canvas.
Levels sliders (black/white/gamma) + "Auto" checkbox; live re-render on change, debounced
(`after(150, ...)`) so dragging stays smooth.
**Depends on:** LM-30, LM-11

### [x] LM-32 · GUI — 2D heightmap preview
> Done: `preview_config()` clones the config with a coarser `dot_size` (target ≤160 vertices on the long axis) so dragging sliders stays interactive, then `build_heightmap` renders the preview. A label reports the preview grid, the real resolved footprint in mm and the Z range; invalid intermediate values show "preview unavailable: …" instead of raising. The preview is a transmission estimate, not a plain thickness ramp: total material is `base + z_min + (1 - brightness) * (z_max - z_min)` (the same absolute Z the mesh uses) attenuated by Beer–Lambert with `LIGHT_ABSORPTION_PER_MM = 0.35`, so the base plate is part of the picture and raising it visibly dims the preview; the info line reports layers + base = total thickness. The **Preview** is a scrollable canvas: a 1×–8× zoom slider relative to fit, mouse-wheel zoom anchored under the pointer (`<MouseWheel>` plus Button-4/5 for X11), a Fit button and left-drag panning; it redraws on `<Configure>` and keeps the histogram markers lined up. The single **Preview** panel shows transmitted light (white = bright) by default; the earlier source-grayscale thumbnail and the preview-polarity toggle were removed on request, so `heightmap_image()` has no `invert` parameter and always renders the transmission estimate.
**Files:** `lithomaker/gui.py`
Render the (downsampled) heightmap with a grayscale/heat colormap into a fixed-size preview
(e.g. 320 px) using `build_heightmap` on a *coarse* `dot_size` so it stays interactive even for
large images. Show resolved footprint (mm), Z range and grid size as labels.
**Depends on:** LM-31, LM-13

### [x] LM-33 · GUI — parameters, generate & stats
> Done: every processing parameter has a widget (`PARAMETER_MAP` plus the CLI-flag table in the module docstring), spinboxes for min/max layers, editable X/Y with a "Lock aspect ratio" checkbox implementing D2, binary/ASCII radios, and `--force` surfaced as an overwrite confirmation dialog. "Generate STL" opens a save dialog only when the output field is blank, then calls `core.convert()` and fills the Result strip (vertices, triangles, Z range, footprint, file size) with an enabled "Show in folder" button. Errors go through `messagebox.showerror`, never a traceback. Widget tests run under a real display and skip when `DISPLAY` is unset.
**Files:** `lithomaker/gui.py`
All CLI parameters as widgets (**parity table in a docstring/comment mapping CLI flag → widget**,
that mapping is what the parity test in LM-40 checks): spinboxes for min/max layers, size X/Y
(with an "lock aspect" checkbox bound to **D2**), thickness, dot size, base thickness, binary/ASCII
radio, `--force` overwrite confirm dialog. On generate: `filedialog.asksaveasfilename`, then
`core.convert(...)`, then a stats panel (vertices, triangles, Z range, file size) + a "show in
folder" convenience. Errors → `messagebox.showerror` with the `ValueError` text (never a traceback).
**Depends on:** LM-32, LM-20

### [x] LM-34 · GUI visual refresh (flat card theme)
> Done: `_configure_styles()` now forces `clam` on every platform (native `vista`/`aqua` ignore colour options), applies a light token palette (`gui.PALETTE`) and defines the previously missing `Accent.TButton` (it used to fall back to a stock button). Layout moved from `LabelFrame` group boxes to flat cards with uppercase section captions, a header with the primary Generate action, a metric strip for results and a thin accent progress bar; the preview/histogram sit on a white card with a dark canvas. Typography, spacing (4/8/12/16) and hover/focus states are centralised in the stylesheet. Tests assert the theme, accent style and palette tokens. Browse/save pickers now prefer the desktop's native dialog — `zenity` then `kdialog` on Linux, Tk's native dialog on Windows/macOS — with a Tk fallback when no helper is available, and fall back cleanly if the helper cannot start.

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
