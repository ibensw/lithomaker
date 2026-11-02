# Lithomaker CLI reference

`lithomaker` is the scriptable frontend of the same processing core the GUI uses:
every GUI parameter has a CLI equivalent (plan §7), and both call
`lithomaker.core.convert()`.

The examples below use the `lithomaker` command, which is available after
`pip install -e .`. Without an editable install, use `python -m lithomaker.cli` instead
of `lithomaker` — the arguments are identical.

## Synopsis

```
lithomaker [OPTIONS] COMMAND [ARGS]...
lithomaker convert [OPTIONS] [INPUT_IMAGE] [OUTPUT.STL]
lithomaker gui
lithomaker info
lithomaker --version
```

```
Usage: lithomaker [OPTIONS] COMMAND [ARGS]...

  Convert a 2D picture into a 3D-printable lithophane STL.

  Examples:
    lithomaker convert photo.png
    lithomaker convert photo.png lithophane.stl --size-x 100 --max-layers 20
    lithomaker convert -i photo.png -o out.stl --ascii --stats

Options:
  --version   Show the version and exit.
  -h, --help  Show this message and exit.

Commands:
  convert  Convert INPUT_IMAGE (or -i) to OUTPUT.STL (or -o).
  gui      Launch the graphical interface (requires tkinter).
  info     Print versions useful in bug reports.
```

## `lithomaker convert`

```
Usage: lithomaker convert [OPTIONS] [INPUT_IMAGE] [OUTPUT.STL]

  Convert INPUT_IMAGE (or -i) to OUTPUT.STL (or -o).

  OUTPUT is optional and defaults to the input name with a .stl suffix.
  The first line on stdout is always the written path; use --no-stats to
  keep it the only line.

Options:
  -i, --input FILE                Input image (alias of the INPUT_IMAGE
                                  positional argument).
  -o, --output FILE               Output STL (alias of the OUTPUT positional
                                  argument).
  -b, --black FLOAT               Levels black point, 0..1.  [default: 0.0]
  -w, --white FLOAT               Levels white point, 0..1.  [default: 1.0]
  -g, --gamma FLOAT               Levels gamma, > 0 (values above 1 darken
                                  midtones).  [default: 1.0]
  --levels [auto|manual]          Derive the levels window from the histogram
                                  of the image.  [default: manual]
  --auto-levels / --no-auto-levels
                                  Boolean spelling of --levels auto; ignores
                                  --black/--white/--gamma.
  --min-layers INTEGER            Minimum Z layers above the base.  [default:
                                  1]
  --max-layers INTEGER            Maximum Z layers above the base.  [default:
                                  13]
  --thickness FLOAT               Layer thickness in mm.  [default: 0.2]
  --size-x FLOAT                  Output width in mm.  [default: 120.0]
  --size-y FLOAT                  Output height in mm (defaults to aspect-
                                  correct).
  --dot-size FLOAT                XY dot/pixel pitch in mm; smaller means more
                                  detail and more triangles.  [default: 0.4]
  --base-thickness FLOAT          Flat base plate thickness in mm.  [default:
                                  0.4]
  --ascii / --binary              STL flavour (default: binary).
  --format [binary|ascii]         STL flavour, long-form spelling of
                                  --ascii/--binary.
  --force                         Overwrite the output file if it already
                                  exists.
  -v, --verbose                   Print progress to stderr and show tracebacks
                                  for unexpected errors.
  --stats / --no-stats            Print a summary table after conversion.
                                  [default: stats]
  -h, --help                      Show this message and exit.
```

### Argument handling (decision D5)

* `INPUT_IMAGE` and `OUTPUT.STL` are positional and canonical.
* `-i/--input` and `-o/--output` are aliases. Supplying the positional form **and** the
  flag for the same value is a usage error, not a silent override.
* Omitting `OUTPUT.STL` writes `<input_stem>.stl` in the current directory.
* A positional output always wins over `config.output` when the library is used
  directly; on the command line there is only one way to specify each path.

### stdout, stderr and exit codes

| Stream | Content |
|--------|---------|
| stdout | The written STL path on the first line, then the stats table (`--stats`, default) |
| stderr | Warnings (`warning: …`), `-v` progress lines, and errors (`Error: …`) |

| Exit code | Meaning | Typical cause |
|-----------|---------|---------------|
| `0` | Success | STL written |
| `1` | Runtime failure | Missing/corrupt image, output exists without `--force`, disk error |
| `2` | Usage failure | Unknown option, invalid parameter, positional/flag conflict, mesh-size guard |

Failures are always a single `Error: …` line. A traceback appears only with `-v`, which is
intended for bug reports.

### Levels (decision D7)

Two spellings are available and they are equivalent:

```bash
lithomaker convert photo.png --levels auto
lithomaker convert photo.png --auto-levels
```

Auto levels derive the black/white points from the 0.5th/99.5th luminance percentiles.
While it is on, `--black`, `--white` and `--gamma` are ignored and a warning is printed
to stderr; silence it by not passing them.

### STL flavour (decision D8)

Binary STL is the default. `--ascii` selects text output, and `--format ascii` is the
long-form spelling. Combining them inconsistently (`--ascii --format binary`) is a usage
error.

## `lithomaker gui`

Opens the tkinter frontend. It exposes the same processing parameters as `convert` and
calls the same `core.convert()`, so anything scripted with the CLI can be reproduced
interactively. Needs a display and a working `tkinter` (Debian/Ubuntu:
`apt install python3-tk`); `python -m lithomaker.gui` and the `lithomaker-gui` entry
point are equivalent.

## `lithomaker info`

Prints the versions that matter in a bug report:

```
$ lithomaker info
lithomaker  0.1.0
python      3.12.4 (linux)
pillow      11.0.0
numpy       2.1.1
click       8.1.7
```

## Worked examples

### 1. Small proof print (40 mm, fast)

Cheap to print, ~2 minutes on an FDM machine, good for checking orientation and
brightness:

```bash
lithomaker convert portrait.jpg test.stl \
  --size-x 40 \
  --dot-size 0.6 \
  --min-layers 3 --max-layers 10 \
  --thickness 0.2 \
  --base-thickness 1.0 \
  --stats
```

`--dot-size 0.6` keeps the grid coarse (68 × 68 grid, 9,782 triangles, about 0.5 MB),
so the file stays under a megabyte instead of tens of megabytes.

### 2. Full-size 150 mm portrait

```bash
lithomaker convert portrait.jpg portrait.stl \
  --size-x 150 \
  --dot-size 0.4 \
  --min-layers 4 --max-layers 32 \
  --thickness 0.15 \
  --base-thickness 1.5 \
  --black 0.05 --white 0.95 --gamma 1.1
```

Omit `--size-y` so the height follows the image aspect ratio (a warning is printed if you
pass both axes and they disagree by more than 1 %). Expect a large file: on a square source
this example produces a 376 × 376 grid, 285,750 triangles and a ~14 MB binary STL.

### 3. CI one-liner

Headless, quiet, idempotent, and fails the build on a bad parameter:

```bash
lithomaker convert assets/logo.png build/logo.stl \
  --size-x 80 --dot-size 0.5 --force --no-stats
```

stdout is exactly one line — the output path — so it can be fed straight into the next
step:

```bash
stl=$(lithomaker convert assets/logo.png build/logo.stl --force --no-stats)
echo "wrote $stl"
```

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `mesh too large: … Use --dot-size X mm` | Grid exceeds 4M vertices | Use the suggested dot size, or shrink `--size-x/--size-y` |
| `image not found` | Wrong path, or `-i` and the positional given together | Fix the path; pass the image once |
| `could not read image …` | Corrupt or mislabelled file | Re-export as PNG/JPG |
| `refusing to overwrite existing file` | Output already exists | Add `--force` |
| Model looks inverted | Bright source pixels are intentionally thin | Invert the image first; `--invert` is v2 |
| Slicer asks about units | STL is unitless | Import as millimetres |
