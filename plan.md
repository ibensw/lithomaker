
# Lithomaker — Project Plan


## 1. Overview
**Lithomaker** is a Python application that converts a 2D picture into a 3D-printable lithophane STL file.


A lithophane works by varying material thickness: thinner areas let more light through (appear bright), thicker areas block light (appear dark). The application must therefore convert image brightness into an inverse heightmap.








The application has two frontends:
* **CLI** — full-featured, scriptable, suitable for automation and CI.
* **GUI** — desktop app for interactive use.


**Core requirement:** The CLI must be able to do everything the GUI does. The GUI should be a thin wrapper around the same processing core.

## 2. Goals & Non-Goals

### Goals
* Load common raster image formats (PNG, JPG, BMP, TIFF, WebP).
* Apply photo-editor-style "levels" adjustments to the source image before meshing.
* Generate a valid STL mesh with a flat bottom surface (base plate).
* Support user-defined physical output size (X/Y dimensions in mm).
* Support user-defined Z resolution (layer thickness) and layer count bounds.
* Export binary STL (millimeters) suitable for slicers.
* CLI and GUI parity.

### Non-Goals (v1)
* Direct slicer integration (Cura, PrusaSlicer, etc.).
* 3D preview/rendering in the GUI (beyond a simple 2D heightmap preview).
* Multi-material or color lithophanes.
* Mesh repair beyond basic manifold guarantees.

## 3. Input Specification

| Parameter | Type | Meaning | Notes |
|-----------|------|---------|-------|
| **Input image** | File path | Source picture | PNG/JPG recommended. Must exist and be readable. |
| **Levels** | 3 values (black point, white point, gamma) or "auto" | Photo-editor level adjustment applied to grayscale image before heightmap generation | Controls contrast and brightness mapping. Default: auto-stretch or neutral. |
| **Min layers** | Integer ≥ 1 | Minimum number of Z layers (excluding base) | Used to compute minimum feature thickness. |
| **Max layers** | Integer ≥ min layers | Maximum number of Z layers (excluding base) | Caps total Z height. |
| **Layer thickness** | Float > 0 (mm) | Thickness of one Z step | Combined with layer count to determine total part height. |
| **Output size** | Float or W×H (mm) | Physical X/Y dimensions of the flat lithophane | Aspect ratio should match image unless user explicitly overrides. |
| **Dot size** | Float > 0 (mm) | XY resolution / effective pixel pitch | Determines how many mesh vertices per mm; related to nozzle size or desired detail. |

### Derived values
The core should compute:
* **Image heightmap:** grayscale value → normalized thickness.
* **Z range:** `min_layers × layer_thickness` to `max_layers × layer_thickness` (plus base thickness).
* **Mesh resolution:** `output_size / dot_size` vertices per axis.

## 4. Processing Pipeline

1. **Load & validate image** — decode pixels, check dimensions, convert to RGB if needed.
2. **Levels adjustment** — apply black/white point and gamma to the luminance channel. Clamp to [0, 1].
3. **Grayscale + inversion** — convert to single-channel brightness; invert so bright image pixels → thin lithophane regions, dark pixels → thick regions.
4. **Heightmap generation** — map normalized brightness to Z thickness within `[min_layers × thickness, max_layers × thickness]`.
5. **Base plate** — add a flat bottom layer of configurable thickness so the part has a stable printing surface. The flat surface is the bottom of the STL.
6. **Mesh generation** — create a regular grid (X × Y), displace vertices by heightmap, triangulate into faces.
7. **STL export** — write binary STL in millimetres. Ensure normals point outward and the mesh is manifold (watertight) where possible.

## 5. Output Specification

* **Format:** STL (binary preferred; ASCII optional).
* **Units:** Millimetres.
* **Orientation:** Flat surface on the bottom (Z = 0 is the base plate; lithophane rises in +Z).
* **File naming:** Default `<input_stem>.stl` in output directory; overridable via CLI/GUI.

## 6. CLI Specification

### Command structure
```
lithomaker convert [OPTIONS] INPUT_IMAGE OUTPUT.STL
lithomaker --version
lithomaker --help
```

### Options (all GUI-exposed parameters must have CLI equivalents)

| Short | Long | Type | Default | Description |
|-------|------|------|---------|-------------|
| `-i` | `--input` | path | required | Input image file |
| `-o` | `--output` | path | required | Output STL path |
| `-b` | `--black` | float | 0.0 | Levels black point (0–1) |
| `-w` | `--white` | float | 1.0 | Levels white point (0–1) |
| `-g` | `--gamma` | float | 1.0 | Levels gamma |
| `--max-layers` | int | 32 | Maximum Z layers (excl. base) |
| `--thickness` | float | 0.2 | Layer thickness in mm |
| `--size-x` | float | 120.0 | Output width in mm |
| `--size-y` | float | — | Output height in mm (defaults to aspect-correct) |
| `--dot-size` | float | 0.4 | XY dot/pixel size in mm |
| `--base-thickness` | float | 1.0 | Base plate thickness in mm |

### Example
```bash
lithomaker convert photo.png lithophane.stl \
  --black 0.05 --white 0.95 --gamma 1.2 \
  --max-layers 20 --thickness 0.2 --size-x 100 \
  --dot-size 0.4 --base-thickness 1.2
```

## 7. GUI Specification

* **Parity:** Every CLI parameter must be adjustable in the GUI; the GUI must invoke the same core processing function.
* **Workflow:**
  1. Open image (file picker).
  2. Preview original grayscale + levels histogram.
  3. Adjust levels (sliders), see real-time 2D heightmap preview.
  4. Set physical parameters (size, thickness, layers, dot size).
  5. Click "Generate STL" → choose output path → export.
  6. Show success/error message and optional STL stats (vertex count, file size, Z range).
* **Layout:** Left panel = image/preview; right panel = parameters; bottom = action bar and status.

## 8. Suggested Tech Stack (Python)

| Concern | Library | Notes |
|---------|---------|-------|
| Image loading | Pillow (PIL) | Supports PNG, JPG, BMP, TIFF, WebP |
| Array math | NumPy | Heightmap and mesh vertex manipulation |
| CLI | Click or Typer | Clean command structure |
| GUI | tkinter (stdlib) or PyQt6 | tkinter = zero extra dependency; PyQt = richer widgets |
| STL I/O | numpy-stl or trimesh | Binary STL writing, basic mesh validation |
| Packaging | setuptools / pyproject.toml | Console entry point for `lithomaker` command |

## 9. Suggested File Structure

```
lithomaker/
├── pyproject.toml
├── README.md
├── plan.md
├── lithomaker/
│   ├── __init__.py
│   ├── cli.py          # CLI entry point (Click/Typer)
│   ├── gui.py          # GUI application
│   ├── core.py         # Shared processing: load → levels → heightmap → mesh → STL
│   ├── image_io.py     # Image loading and levels adjustment
│   ├── mesh.py         # Grid mesh generation and STL export
│   └── config.py       # Dataclasses for input parameters
├── tests/
│   ├── test_core.py
│   ├── test_image_io.py
│   └── test_mesh.py
└── examples/
    └── sample.png
```

## 10. Milestones

1. **Core library:** image loading, levels, heightmap, mesh generation, STL export (CLI-only).
2. **CLI:** full parameter set, error handling, examples.
3. **GUI:** parameter controls, preview, generate button, parity with CLI.
4. **Packaging & tests:** unit tests for core, installable package, CI check.

## 11. Open Questions / Risks

* **Inversion convention:** Confirm whether bright pixels should be thin (standard transmissive lithophane) or thick (reflective). Default: bright = thin.
* **Aspect ratio:** If user sets both `--size-x` and `--size-y` differently from image aspect, should we stretch, letterbox, or error? Recommendation: stretch with a warning, or default `--size-y` to preserve aspect.
* **Mesh size:** High dot size + large output can create millions of triangles. Add a max vertex guard or decimation option in v2.
* **STL units:** STL is unitless; we must document that output is in millimetres and slicers interpret it as such.
