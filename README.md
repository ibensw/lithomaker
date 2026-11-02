# Lithomaker

Convert a 2D picture into a 3D-printable **lithophane** STL.

A lithophane is a thin panel whose *thickness* encodes an image: thin areas transmit
light and look bright, thick areas block it and look dark. Lithomaker turns a picture
into exactly that heightmap and exports a watertight, flat-bottomed STL in millimetres.

> **Status: Phase D complete (config, image I/O, core pipeline, mesh/STL, CLI and GUI).**
> Packaging, CI and release polish (Phase E) remain — see [`plan.md`](plan.md) for the
> design and [`TODO.md`](TODO.md) for progress. Full CLI reference: [`docs/cli.md`](docs/cli.md).

## Install (development)

```bash
python -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

Requires Python 3.9+. Runtime dependencies: Pillow, NumPy, Click. The GUI uses
`tkinter` from the standard library (on Debian/Ubuntu: `apt install python3-tk`).

## Usage

```bash
lithomaker convert photo.png lithophane.stl \
  --black 0.05 --white 0.95 --gamma 1.2 \
  --max-layers 20 --thickness 0.2 --size-x 100 \
  --dot-size 0.4 --base-thickness 1.2
```

`INPUT_IMAGE` and `OUTPUT.STL` are positional. `-i/--input` and `-o/--output` are
accepted as aliases — but not at the same time as the positional forms. `OUTPUT.STL` is
optional and defaults to the input name with an `.stl` suffix, written in the current
directory.

The first line on stdout is always the written path, followed by a stats table
(`--no-stats` keeps it to just the path); warnings and `-v` progress go to stderr, so the
tool composes cleanly in pipelines.

### Parameters

| Option | Type | Default | Meaning |
|--------|------|---------|---------|
| `INPUT_IMAGE` / `-i, --input` | path | required | Source picture (PNG, JPG, BMP, TIFF, WebP) |
| `OUTPUT.STL` / `-o, --output` | path | `<input>.stl` | Output STL path |
| `-b, --black` | float | `0.0` | Levels black point (0–1) |
| `-w, --white` | float | `1.0` | Levels white point (0–1) |
| `-g, --gamma` | float | `1.0` | Levels gamma (> 0; above 1 darkens midtones) |
| `--levels` | `auto`\|`manual` | `manual` | Derive the levels window from the image histogram |
| `--auto-levels/--no-auto-levels` | flag | off | Boolean spelling of `--levels auto`; when on, black/white/gamma are ignored with a warning |
| `--min-layers` | int | `1` | Minimum Z layers above the base |
| `--max-layers` | int | `13` | Maximum Z layers above the base |
| `--thickness` | float | `0.2` | Layer thickness in mm |
| `--size-x` | float | `120.0` | Output width in mm |
| `--size-y` | float | aspect-correct | Output height in mm; defaults to `size_x × image_h / image_w` |
| `--dot-size` | float | `0.4` | XY dot/pixel pitch in mm — detail versus triangle count |
| `--base-thickness` | float | `0.4` | Flat base plate thickness in mm |
| `--ascii/--binary` | flag | binary | STL flavour |
| `--format` | `binary`\|`ascii` | — | Long-form spelling of the flag above |
| `--force` | flag | off | Overwrite an existing output file |
| `-v, --verbose` | flag | off | Progress on stderr, tracebacks for unexpected errors |
| `--stats/--no-stats` | flag | stats | Print the summary table on stdout |

Two behaviours are worth calling out:

* **Aspect ratio** — omit `--size-y` and the image aspect is preserved. Pass both axes
  and a mismatch of more than 1 % is *stretched*, with a warning on stderr.
* **Mesh guard** — a grid above 4,000,000 vertices is refused before anything is
  allocated, with a suggested `--dot-size` that would fit. Prefer a bigger dot size over
  a smaller one: triangles scale with `1 / dot_size²`.

## GUI

```bash
lithomaker gui              # console script
python -m lithomaker.gui    # module entry point
lithomaker-gui              # gui-script entry point
```

The window is a thin wrapper over the same `core.convert()` the CLI uses, so every
processing parameter has a widget (the mapping is machine-checked by the tests). It uses a
flat card theme built on `ttk`'s `clam` engine: a header with the app title and the primary
**Generate STL** action, the live **Preview** and luminance histogram on a white card, the
parameters grouped as Image / Levels / Geometry on the right, and a metric strip
(vertices, triangles, Z range, footprint, file size) above the status bar.

The preview estimates how the part looks backlit: it uses total material thickness — base
plate plus lithophane layer — with Beer–Lambert attenuation (μ = 0.35/mm), so a thicker
base visibly dims the image. White means light passing through (bright), which is the
inverted view of material thickness. It is display-only and never changes the STL.

The preview zooms like a photo editor: use the **Zoom** slider (1×–8× relative to fit),
the mouse wheel over the preview (anchored under the pointer), or **Fit** to go back to
the whole heightmap; drag with the left mouse button to pan when zoomed in. The luminance
histogram redraws to the full panel width as the window is resized.
"Generate STL" runs the conversion on a worker thread, so progress, the status bar and the
Result strip stay live and the window never freezes; failures appear as message boxes,
never tracebacks.

File pickers use the desktop's own dialog: GTK's picker through `zenity`, then
`kdialog`, on Linux; the native dialog Tk already provides on Windows and macOS. If
neither helper is installed it falls back to Tk's own dialog, so nothing extra is
required.

`tkinter` ships with CPython (on Debian/Ubuntu: `apt install python3-tk`). The GUI needs a
display, but importing `lithomaker.gui` headless is safe — CI does exactly that.

## How it works

1. Load and validate the image.
2. Apply "levels" (black point, white point, gamma) to the luminance channel.
3. Invert: bright pixels become thin, dark pixels become thick.
4. Map brightness into the Z range `min_layers × thickness … max_layers × thickness`.
5. Add a flat base plate so the part sits on the bed (`z = 0` is the bottom face).
6. Displace a regular grid into a closed mesh: top surface, four side walls, flat bottom.
7. Write binary STL.

## FAQ

**Why is my model inverted?**
Bright pixels become *thin* material (standard transmissive lithophane), so backlit dark
areas read as thick and opaque. If you are printing a reflective (front-lit) panel or you
want the opposite mapping, invert the source image before converting; an `--invert` flag
is planned for v2.

**Why is my STL so large, or why do I get "mesh too large"?**
Triangle count grows with `(size_x / dot_size) × (size_y / dot_size)`. Halving `--dot-size`
roughly quadruples it. Details smaller than your nozzle cannot print anyway, so start at
`--dot-size 0.4` (0.4 mm nozzle) and go up, or reduce `--size-x/--size-y`.

**Do I need to rotate the model in my slicer?**
No. `z = 0` is the flat base plate and the image rises in +Z, which is exactly how
slicers expect a part on the bed. Just make sure the slicer reads the file as
millimetres.

**My flat areas look faceted or the file is huge.**
That is the `--dot-size` trade-off, not a bug: the mesh is a regular grid, one vertex per
dot. Increase `--dot-size` for smaller files.

**The conversion warns that the image will be stretched.**
Both `--size-x` and `--size-y` were given and do not match the image aspect. Drop
`--size-y` to keep the aspect ratio automatically.

## Repository layout

```
lithomaker/       package: config, image_io, core, mesh, cli, gui
tests/            pytest suite; test images are generated in-memory
tools/            developer scripts (make_sample.py regenerates examples/sample.png)
examples/         sample.png — deterministic test pattern
docs/cli.md       full CLI reference
plan.md           design document
TODO.md           task breakdown, decisions and dependencies
```

## Units

STL is unitless. **All Lithomaker output is in millimetres** — load the STL into your
slicer with mm units (the default in Cura, PrusaSlicer and friends). See `plan.md` §11.
