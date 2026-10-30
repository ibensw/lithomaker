# Lithomaker

Convert a 2D picture into a 3D-printable **lithophane** STL.

A lithophane is a thin panel whose *thickness* encodes an image: thin areas transmit
light and look bright, thick areas block it and look dark. Lithomaker turns a picture
into exactly that heightmap and exports a watertight, flat-bottomed STL in millimetres.

> **Status: Phase B complete (image I/O, heightmap, mesh and STL core).**
> `lithomaker.core.convert()` is the single entry point; the CLI (Phase C) and GUI
> (Phase D) are not yet written — see [`plan.md`](plan.md) for the design and
> [`TODO.md`](TODO.md) for progress.

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

Landing in Phase C (LM-20). See [`plan.md` §6](plan.md) for the full option list.

## How it works

1. Load and validate the image.
2. Apply "levels" (black point, white point, gamma) to the luminance channel.
3. Invert: bright pixels become thin, dark pixels become thick.
4. Map brightness into the Z range `min_layers × thickness … max_layers × thickness`.
5. Add a flat base plate so the part sits on the bed (`z = 0` is the bottom face).
6. Displace a regular grid into a closed mesh: top surface, four side walls, flat bottom.
7. Write binary STL.

## Repository layout

```
lithomaker/       package (config.py is the shared parameter contract)
tests/            pytest suite; images are generated in-memory, not committed
tools/            developer scripts (make_sample.py regenerates examples/sample.png)
examples/         sample.png — deterministic test pattern
plan.md           design document
TODO.md           task breakdown, decisions and dependencies
```

## Units

STL is unitless. **All Lithomaker output is in millimetres** — load the STL into your
slicer with mm units (the default in Cura, PrusaSlicer and friends). See `plan.md` §11.
