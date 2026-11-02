"""tkinter frontend (LM-30 … LM-33).

Plan §7 requires the GUI to be a *thin* wrapper: it owns widgets only and routes every
parameter through :class:`lithomaker.config.LithoConfig` and every conversion through
:func:`lithomaker.core.convert`. There is no geometry, I/O or validation logic in this
module.

Running it::

    lithomaker gui          # console script
    python -m lithomaker.gui

Importing this module is safe on a headless machine: ``tkinter`` and ``PIL.ImageTk`` are
imported, but no ``Tk()`` root is created until :func:`main` runs. The parity map and the
pure helpers below are importable and testable without a display (LM-40 builds on this).

CLI flag → widget mapping (the table LM-40 asserts against ``--help``)::

    INPUT_IMAGE / -i --input   image_path_var   "Image" entry
    OUTPUT.STL / -o --output   output_path_var  "Output" entry
    -b --black                 black_var        Levels scale
    -w --white                 white_var        Levels scale
    -g --gamma                 gamma_var        Levels scale
    --levels / --auto-levels   auto_levels_var  "Auto levels" check button
    --min-layers               min_layers_var   Geometry spinbox
    --max-layers               max_layers_var   Geometry spinbox
    --thickness                thickness_var    Geometry entry
    --size-x                   size_x_var       Geometry entry
    --size-y                   size_y_var       Geometry entry (+ "lock aspect")
    --dot-size                 dot_size_var     Geometry entry
    --base-thickness           base_thickness_var  Geometry entry
    --ascii / --binary         ascii_var        Format radio buttons
    --force                    force_var        overwrite confirmation dialog

Presentation-only CLI switches (``--verbose``, ``--stats``, ``--format``) have no
processing effect and are deliberately absent.
"""

from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any

import numpy as np
from PIL import Image, ImageTk

from . import __version__
from .config import ConfigError, LevelsConfig, LithoConfig
from .core import build_heightmap, convert
from .image_io import ImageLoadError, load_image, to_luminance

__all__ = [
    "PARAMETER_MAP",
    "LithomakerApp",
    "config_from_parameters",
    "heightmap_image",
    "histogram_points",
    "main",
    "preview_config",
]

#: Canonical processing parameter → GUI variable attribute (see the module docstring).
PARAMETER_MAP = {
    "input": "image_path_var",
    "output": "output_path_var",
    "black": "black_var",
    "white": "white_var",
    "gamma": "gamma_var",
    "auto_levels": "auto_levels_var",
    "min_layers": "min_layers_var",
    "max_layers": "max_layers_var",
    "thickness": "thickness_var",
    "size_x": "size_x_var",
    "size_y": "size_y_var",
    "dot_size": "dot_size_var",
    "base_thickness": "base_thickness_var",
    "ascii": "ascii_var",
    "force": "force_var",
}

IMAGE_FILE_TYPES = [
    ("Images", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"),
    ("All files", "*.*"),
]

#: Heightmap section caption; swaps when "Invert preview" is ticked.
HEIGHTMAP_CAPTION = "Heightmap (transmission + base, white = thick)"
HEIGHTMAP_CAPTION_INVERTED = "Heightmap (transmission + base, white = bright)"

#: Rough Beer–Lambert absorption of PLA-ish filament at visible wavelengths, per mm of
#: material. The preview is an estimate — uniform absorption, no scattering — but it is
#: *absolute*: the base plate and the layer thickness both attenuate the light, so
#: "why does a thicker base look darker?" has a visible answer.
LIGHT_ABSORPTION_PER_MM = 0.35

PREVIEW_WIDTH = 420
PREVIEW_HEIGHT = 320
SOURCE_PREVIEW_HEIGHT = 140
HISTOGRAM_HEIGHT = 110
#: Target vertex count along the long axis for the interactive heightmap preview.
PREVIEW_MAX_VERTICES = 160


# --------------------------------------------------------------------------
# Pure helpers (no Tk objects, safe on a headless machine)
# --------------------------------------------------------------------------
def _to_number(raw: Any, label: str, cast=float):
    """Coerce a form value, turning blanks and garbage into a readable ConfigError."""
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        raise ConfigError(f"{label} must not be empty")
    try:
        return cast(raw)
    except (TypeError, ValueError) as exc:
        kind = "a number" if cast is float else "an integer"
        raise ConfigError(f"{label} must be {kind}, got {raw!r}") from exc


def _optional_number(raw: Any, label: str, cast=float):
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return None
    return _to_number(raw, label, cast)


def config_from_parameters(values: Mapping[str, Any]) -> LithoConfig:
    """Build a validated :class:`LithoConfig` from the widget values.

    ``values`` uses the canonical keys of :data:`PARAMETER_MAP`; strings from ``Entry``
    widgets are accepted. All range/relationship checks stay in ``config.py``.
    """
    image = values.get("input")
    if image is None or (isinstance(image, str) and not image.strip()):
        raise ConfigError("choose an input image first")

    output = values.get("output")
    if isinstance(output, str):
        output = output.strip() or None

    return LithoConfig(
        input=image,
        output=output,
        levels=LevelsConfig(
            black=_to_number(values.get("black", 0.0), "black point"),
            white=_to_number(values.get("white", 1.0), "white point"),
            gamma=_to_number(values.get("gamma", 1.0), "gamma"),
            auto=bool(values.get("auto_levels", False)),
        ),
        min_layers=_to_number(values.get("min_layers", 4), "min layers", int),
        max_layers=_to_number(values.get("max_layers", 32), "max layers", int),
        thickness=_to_number(values.get("thickness", 0.2), "thickness"),
        size_x=_to_number(values.get("size_x", 120.0), "size X"),
        size_y=_optional_number(values.get("size_y"), "size Y"),
        dot_size=_to_number(values.get("dot_size", 0.4), "dot size"),
        base_thickness=_to_number(values.get("base_thickness", 1.0), "base thickness"),
        ascii=bool(values.get("ascii", False)),
    )


def preview_config(config: LithoConfig, max_vertices: int = PREVIEW_MAX_VERTICES) -> LithoConfig:
    """Coarse copy of ``config`` so the 2D preview stays interactive (LM-32).

    Large images at the real dot size would build millions of vertices per slider drag;
    the preview only needs the shape of the height field, not its final resolution.
    """
    coarse = max(config.dot_size, config.size_x / float(max_vertices), 0.01)
    return replace(config, dot_size=coarse)


def transmission(heightmap, base_thickness: float = 0.0) -> np.ndarray:
    """Estimated fraction of light passing through, per vertex (Beer–Lambert).

    Total material along a ray is the base plate plus the lithophane layer, i.e. the
    same absolute Z the mesh uses: ``base + z_min + (1 - brightness) * (z_max - z_min)``
    in mm. Transmission is ``exp(-mu * thickness)`` with
    :data:`LIGHT_ABSORPTION_PER_MM`; because the coefficient is absolute rather than
    renormalised per image, changing ``--base-thickness`` really does dim the preview.
    """
    brightness = np.clip(np.asarray(heightmap.values, dtype=np.float32), 0.0, 1.0)
    thickness_mm = (
        float(base_thickness)
        + heightmap.z_min
        + (1.0 - brightness) * (heightmap.z_max - heightmap.z_min)
    )
    return np.exp(-LIGHT_ABSORPTION_PER_MM * thickness_mm)


def heightmap_image(heightmap, invert: bool = False, base_thickness: float = 0.0):
    """Render a :class:`lithomaker.core.Heightmap` as a grayscale preview.

    The preview estimates how the printed part looks backlit: thin areas transmit more
    light than thick ones, and the **base plate is included** in the thickness (see
    :func:`transmission`). By default the transmitted light is inverted, so white means
    thick/opaque; ``invert=True`` shows the transmission itself, so white means
    bright/thin. Both are display-only — the generated STL never changes. The array is
    transposed because ``values`` is indexed ``[x, y]`` while PIL rows are Y.
    """
    transmitted = transmission(heightmap, base_thickness)
    shown = transmitted if invert else 1.0 - transmitted
    pixels = np.round(np.clip(shown, 0.0, 1.0).T * 255.0).astype(np.uint8)
    return Image.fromarray(pixels)


def histogram_points(counts: np.ndarray, width: int, height: int) -> list:
    """Polygon outline for a luminance histogram, in canvas coordinates."""
    counts = np.asarray(counts, dtype=np.float64)
    if counts.size == 0 or counts.max() <= 0:
        return [(0.0, float(height)), (float(width), float(height))]
    xs = np.linspace(0.0, float(width), counts.size)
    ys = height - (counts / counts.max()) * height
    return [(0.0, float(height)), *zip(xs, ys), (float(width), float(height))]


def _open_in_file_manager(path: Path) -> None:
    """Best-effort "show in folder" across platforms."""
    directory = str(path.parent if path.suffix else path)
    if sys.platform.startswith("win"):
        os.startfile(directory)  # type: ignore[attr-defined]  # pragma: no cover
    elif sys.platform == "darwin":  # pragma: no cover - macOS only
        subprocess.Popen(["open", directory])
    else:
        subprocess.Popen(["xdg-open", directory])


# --------------------------------------------------------------------------
# Application
# --------------------------------------------------------------------------
class LithomakerApp(ttk.Frame):
    """Main window contents: preview on the left, parameters on the right."""

    def __init__(self, master: tk.Misc) -> None:
        super().__init__(master, padding=10)
        self._rgb = None
        self._image_size = None
        self._luminance = None
        self._source_photo = None
        self._preview_photo = None
        self._preview_image = None
        self._last_heightmap = None
        self._preview_job = None
        self._busy = False
        self._queue: queue.Queue | None = None
        self._last_output: Path | None = None
        self._updating_aspect = False

        self._create_variables()
        self._build_layout()
        self._wire_events()

    # -- construction -------------------------------------------------------
    def _create_variables(self) -> None:
        self.image_path_var = tk.StringVar()
        self.output_path_var = tk.StringVar()
        self.black_var = tk.DoubleVar(value=0.0)
        self.white_var = tk.DoubleVar(value=1.0)
        self.gamma_var = tk.DoubleVar(value=1.0)
        self.auto_levels_var = tk.BooleanVar(value=False)
        self.min_layers_var = tk.IntVar(value=4)
        self.max_layers_var = tk.IntVar(value=32)
        self.thickness_var = tk.StringVar(value="0.2")
        self.size_x_var = tk.StringVar(value="120")
        self.size_y_var = tk.StringVar(value="")
        self.dot_size_var = tk.StringVar(value="0.4")
        self.base_thickness_var = tk.StringVar(value="1.0")
        self.lock_aspect_var = tk.BooleanVar(value=True)
        self.ascii_var = tk.BooleanVar(value=False)
        self.force_var = tk.BooleanVar(value=False)
        self.status_var = tk.StringVar(value="Open an image to begin.")
        self.progress_var = tk.DoubleVar(value=0.0)
        self.preview_info_var = tk.StringVar(value="")
        self.invert_preview_var = tk.BooleanVar(value=False)
        self.heightmap_caption_var = tk.StringVar(value=HEIGHTMAP_CAPTION)
        self.black_label_var = tk.StringVar(value="0.00")
        self.white_label_var = tk.StringVar(value="1.00")
        self.gamma_label_var = tk.StringVar(value="1.00")
        self.stats_vars = {
            key: tk.StringVar(value="—")
            for key in ("vertices", "triangles", "z_range", "footprint", "file_size")
        }

    def _build_layout(self) -> None:
        self.columnconfigure(0, weight=3)
        self.columnconfigure(1, weight=2)
        self.rowconfigure(0, weight=1)

        self._build_preview(self._panel(side="left"))
        self._build_parameters(self._panel(side="right"))
        self._build_results()
        self._build_status_bar()

    def _panel(self, side: str) -> ttk.Frame:
        frame = ttk.Frame(self)
        if side == "left":
            frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        else:
            frame.grid(row=0, column=1, sticky="nsew")
        frame.columnconfigure(0, weight=1)
        return frame

    def _build_preview(self, parent: ttk.Frame) -> None:
        parent.rowconfigure(3, weight=1)

        ttk.Label(parent, text="Source (grayscale)", style="Heading.TLabel").grid(
            row=0, column=0, sticky="w", pady=(0, 4)
        )
        self.source_label = ttk.Label(
            parent, text="No image loaded", anchor="center", relief="sunken", padding=4
        )
        self.source_label.grid(row=1, column=0, sticky="ew")

        header = ttk.Frame(parent)
        header.grid(row=2, column=0, sticky="ew", pady=(8, 4))
        header.columnconfigure(0, weight=1)
        ttk.Label(
            header, textvariable=self.heightmap_caption_var, style="Heading.TLabel"
        ).grid(row=0, column=0, sticky="w")
        self.invert_preview_check = ttk.Checkbutton(
            header,
            text="Invert preview",
            variable=self.invert_preview_var,
            command=self._on_invert_preview,
        )
        self.invert_preview_check.grid(row=0, column=1, sticky="e")
        self.heightmap_label = ttk.Label(
            parent, text="Adjust the parameters to see the heightmap", anchor="center",
            relief="sunken", padding=4,
        )
        self.heightmap_label.grid(row=3, column=0, sticky="nsew")

        ttk.Label(parent, text="Luminance histogram (black/white points marked)").grid(
            row=4, column=0, sticky="w", pady=(8, 2)
        )
        self.histogram_canvas = tk.Canvas(
            parent, height=HISTOGRAM_HEIGHT, background="white", highlightthickness=1,
            highlightbackground="#c0c0c0",
        )
        self.histogram_canvas.grid(row=5, column=0, sticky="ew")

        ttk.Label(
            parent, textvariable=self.preview_info_var, style="Muted.TLabel",
            wraplength=PREVIEW_WIDTH, justify="left",
        ).grid(row=6, column=0, sticky="w", pady=(6, 0))

    def _build_parameters(self, parent: ttk.Frame) -> None:
        row = 0
        row = self._build_image_group(parent, row)
        row = self._build_levels_group(parent, row)
        row = self._build_geometry_group(parent, row)
        row = self._build_format_group(parent, row)
        row = self._build_action_group(parent, row)
        parent.rowconfigure(row, weight=1)

    def _group(self, parent: ttk.Frame, title: str, row: int) -> ttk.LabelFrame:
        frame = ttk.LabelFrame(parent, text=title, padding=8)
        frame.grid(row=row, column=0, sticky="ew", pady=(0, 8))
        frame.columnconfigure(1, weight=1)
        return frame

    @staticmethod
    def _entry_row(frame, row, label, variable, width=16):
        ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=2)
        entry = ttk.Entry(frame, textvariable=variable, width=width)
        entry.grid(row=row, column=1, sticky="ew", pady=2)
        return entry

    def _build_image_group(self, parent, row):
        frame = self._group(parent, "Image", row)
        self.image_entry = self._entry_row(frame, 0, "Image", self.image_path_var)
        ttk.Button(frame, text="Browse…", command=self._browse_image).grid(
            row=0, column=2, padx=(6, 0)
        )
        self.output_entry = self._entry_row(frame, 1, "Output STL", self.output_path_var)
        ttk.Button(frame, text="Browse…", command=self._browse_output).grid(
            row=1, column=2, padx=(6, 0)
        )
        return row + 1

    def _scale_row(self, frame, row, label, variable, label_var, to=1.0):
        ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=2)
        scale = ttk.Scale(frame, from_=0.0, to=to, variable=variable)
        scale.grid(row=row, column=1, sticky="ew", pady=2)
        ttk.Label(frame, textvariable=label_var, width=5, anchor="e").grid(
            row=row, column=2, padx=(6, 0)
        )
        return scale

    def _build_levels_group(self, parent, row):
        frame = self._group(parent, "Levels", row)
        self.black_scale = self._scale_row(
            frame, 0, "Black point", self.black_var, self.black_label_var
        )
        self.white_scale = self._scale_row(
            frame, 1, "White point", self.white_var, self.white_label_var
        )
        self.gamma_scale = self._scale_row(
            frame, 2, "Gamma", self.gamma_var, self.gamma_label_var, to=3.0
        )
        ttk.Checkbutton(
            frame, text="Auto levels (ignore black/white/gamma)", variable=self.auto_levels_var
        ).grid(row=3, column=0, columnspan=3, sticky="w", pady=(4, 0))
        return row + 1

    def _spin_row(self, frame, row, label, variable, from_, to):
        ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=2)
        spin = ttk.Spinbox(frame, from_=from_, to=to, textvariable=variable, width=14)
        spin.grid(row=row, column=1, sticky="ew", pady=2)
        return spin

    def _build_geometry_group(self, parent, row):
        frame = self._group(parent, "Geometry (mm)", row)
        frame.columnconfigure(2, weight=1)
        self.min_layers_spin = self._spin_row(frame, 0, "Min layers", self.min_layers_var, 1, 10000)
        self.max_layers_spin = self._spin_row(frame, 1, "Max layers", self.max_layers_var, 1, 10000)
        self.thickness_entry = self._entry_row(frame, 2, "Layer thickness", self.thickness_var)
        self.size_x_entry = self._entry_row(frame, 3, "Width X", self.size_x_var)
        self.size_y_entry = self._entry_row(frame, 4, "Height Y (blank = aspect)", self.size_y_var)
        ttk.Checkbutton(
            frame, text="Lock aspect ratio", variable=self.lock_aspect_var
        ).grid(row=5, column=1, sticky="w")
        self.dot_size_entry = self._entry_row(frame, 6, "Dot size", self.dot_size_var)
        self.base_entry = self._entry_row(frame, 7, "Base thickness", self.base_thickness_var)
        return row + 1

    def _build_format_group(self, parent, row):
        frame = self._group(parent, "STL format", row)
        ttk.Radiobutton(
            frame, text="Binary (default)", value=False, variable=self.ascii_var
        ).grid(row=0, column=0, sticky="w", padx=(0, 12))
        ttk.Radiobutton(frame, text="ASCII", value=True, variable=self.ascii_var).grid(
            row=0, column=1, sticky="w"
        )
        return row + 1

    def _build_action_group(self, parent, row):
        frame = ttk.Frame(parent)
        frame.grid(row=row, column=0, sticky="ew")
        frame.columnconfigure(0, weight=1)
        self.generate_button = ttk.Button(
            frame, text="Generate STL", command=self._on_generate, style="Accent.TButton"
        )
        self.generate_button.grid(row=0, column=0, sticky="ew")
        return row + 1

    def _build_results(self) -> None:
        frame = ttk.LabelFrame(self, text="Result", padding=8)
        frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        for index, (key, label) in enumerate(
            (
                ("vertices", "Vertices"),
                ("triangles", "Triangles"),
                ("z_range", "Z range"),
                ("footprint", "Footprint"),
                ("file_size", "File size"),
            )
        ):
            ttk.Label(frame, text=label).grid(row=0, column=index * 2, sticky="w", padx=(0, 4))
            ttk.Label(frame, textvariable=self.stats_vars[key]).grid(
                row=0, column=index * 2 + 1, sticky="w", padx=(0, 14)
            )
        self.show_button = ttk.Button(
            frame, text="Show in folder", command=self._show_in_folder, state="disabled"
        )
        self.show_button.grid(row=0, column=10, sticky="e")

    def _build_status_bar(self) -> None:
        frame = ttk.Frame(self)
        frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        frame.columnconfigure(0, weight=1)
        ttk.Label(frame, textvariable=self.status_var, anchor="w").grid(
            row=0, column=0, sticky="ew"
        )
        self.progress = ttk.Progressbar(
            frame, orient="horizontal", length=180, mode="determinate",
            maximum=100.0, variable=self.progress_var,
        )
        self.progress.grid(row=0, column=1, sticky="e")

    def _wire_events(self) -> None:
        for variable in (self.black_var, self.white_var, self.gamma_var, self.auto_levels_var):
            variable.trace_add("write", self._on_levels_changed)
        for variable in (
            self.thickness_var,
            self.size_y_var,
            self.dot_size_var,
            self.base_thickness_var,
            self.min_layers_var,
            self.max_layers_var,
        ):
            variable.trace_add("write", self._schedule_preview)
        self.size_x_var.trace_add("write", self._on_size_x_changed)

    # -- parameter plumbing -------------------------------------------------
    def read_parameters(self) -> dict:
        """Snapshot every widget into the canonical parameter names."""
        return {
            "input": self.image_path_var.get(),
            "output": self.output_path_var.get(),
            "black": self.black_var.get(),
            "white": self.white_var.get(),
            "gamma": self.gamma_var.get(),
            "auto_levels": self.auto_levels_var.get(),
            "min_layers": self.min_layers_var.get(),
            "max_layers": self.max_layers_var.get(),
            "thickness": self.thickness_var.get(),
            "size_x": self.size_x_var.get(),
            "size_y": self.size_y_var.get(),
            "dot_size": self.dot_size_var.get(),
            "base_thickness": self.base_thickness_var.get(),
            "ascii": self.ascii_var.get(),
            "force": self.force_var.get(),
        }

    def current_config(self) -> LithoConfig:
        """Validated config for the current widgets (raises ``ConfigError``)."""
        return config_from_parameters(self.read_parameters())

    # -- image and preview --------------------------------------------------
    def _browse_image(self) -> None:
        chosen = filedialog.askopenfilename(
            title="Open image", filetypes=IMAGE_FILE_TYPES,
            initialfile=self.image_path_var.get() or None,
        )
        if chosen:
            self.load_image_file(chosen)

    def _browse_output(self) -> None:
        chosen = filedialog.asksaveasfilename(
            title="Save STL as", defaultextension=".stl",
            filetypes=[("STL", "*.stl"), ("All files", "*.*")],
            initialfile=self.output_path_var.get() or None,
        )
        if chosen:
            self.output_path_var.set(chosen)

    def load_image_file(self, path: str | Path) -> None:
        """Load an image into the preview (dialog-free, so tests can call it)."""
        try:
            rgb = load_image(path)
        except ImageLoadError as exc:
            messagebox.showerror("Cannot open image", str(exc), parent=self)
            return

        self._rgb = rgb
        self._last_heightmap = None
        self._image_size = (rgb.shape[1], rgb.shape[0])
        self._luminance = to_luminance(rgb)
        self.image_path_var.set(str(path))
        if not self.output_path_var.get().strip():
            self.output_path_var.set(str(Path(path).with_suffix(".stl")))
        if self.lock_aspect_var.get():
            self._apply_aspect_lock()
        self._render_source_image()
        self._draw_histogram()
        self.status_var.set("Image loaded. Adjust levels, then generate.")
        self._schedule_preview()

    def _render_source_image(self) -> None:
        """Show the source in grayscale, the channel the levels actually operate on."""
        if self._luminance is None:
            return
        pixels = np.round(np.clip(self._luminance, 0.0, 1.0) * 255.0).astype(np.uint8)
        image = Image.fromarray(pixels)
        image.thumbnail((PREVIEW_WIDTH, SOURCE_PREVIEW_HEIGHT))
        self._source_photo = ImageTk.PhotoImage(image)
        self.source_label.configure(image=self._source_photo, text="")

    def _on_levels_changed(self, *_args) -> None:
        self.black_label_var.set(f"{self.black_var.get():.2f}")
        self.white_label_var.set(f"{self.white_var.get():.2f}")
        self.gamma_label_var.set(f"{self.gamma_var.get():.2f}")
        self._draw_histogram()
        self._schedule_preview()

    def _draw_histogram(self) -> None:
        canvas = self.histogram_canvas
        canvas.delete("all")
        if self._luminance is None:
            return
        width = max(canvas.winfo_width(), PREVIEW_WIDTH)
        height = HISTOGRAM_HEIGHT
        counts, _edges = np.histogram(self._luminance, bins=128, range=(0.0, 1.0))
        canvas.create_polygon(
            histogram_points(counts, width, height), fill="#9ec7e8", outline="#3c78a8"
        )
        for value, colour in ((self.black_var.get(), "black"), (self.white_var.get(), "black")):
            x = float(np.clip(value, 0.0, 1.0)) * width
            canvas.create_line(x, 0, x, height, fill=colour, dash=(3, 3))

    def _on_size_x_changed(self, *_args) -> None:
        if self.lock_aspect_var.get():
            self._apply_aspect_lock()
        self._schedule_preview()

    def _apply_aspect_lock(self) -> None:
        if self._image_size is None or self._updating_aspect:
            return
        width, height = self._image_size
        try:
            size_x = float(self.size_x_var.get())
        except ValueError:
            return
        self._updating_aspect = True
        try:
            self.size_y_var.set(f"{size_x * height / width:.3f}")
        finally:
            self._updating_aspect = False

    def _schedule_preview(self, *_args) -> None:
        if self._preview_job is not None:
            self.after_cancel(self._preview_job)
        self._preview_job = self.after(150, self._render_heightmap_preview)

    def _render_heightmap_preview(self) -> None:
        self._preview_job = None
        if self._rgb is None:
            return
        try:
            config = preview_config(self.current_config())
            heightmap = build_heightmap(self._rgb, config)
        except (ConfigError, ValueError) as exc:
            self.preview_info_var.set(f"preview unavailable: {exc}")
            return

        self._last_heightmap = heightmap
        self._last_base_thickness = config.base_thickness
        self._apply_heightmap_image(heightmap)
        self._update_preview_info(heightmap)

    def _update_preview_info(self, heightmap) -> None:
        try:
            footprint = self.current_config().resolved_size(*self._image_size)
            mm = f"{footprint.x:.1f} × {footprint.y:.1f} mm"
        except (ConfigError, ValueError):
            mm = "size pending"
        nx, ny = heightmap.grid_shape
        base = self._last_base_thickness
        total_low = base + heightmap.z_min
        total_high = base + heightmap.z_max
        self.preview_info_var.set(
            f"heightmap {nx} × {ny} · {mm}\n"
            f"layers {heightmap.z_min:.2f}–{heightmap.z_max:.2f} + {base:g} mm base "
            f"= {total_low:.2f}–{total_high:.2f} mm through the part "
            f"(μ {LIGHT_ABSORPTION_PER_MM:g}/mm)"
        )

    def _apply_heightmap_image(self, heightmap) -> None:
        """Draw the cached heightmap in the current preview polarity."""
        image = heightmap_image(
            heightmap,
            invert=self.invert_preview_var.get(),
            base_thickness=self._last_base_thickness,
        )
        image.thumbnail((PREVIEW_WIDTH, PREVIEW_HEIGHT), Image.Resampling.NEAREST)
        self._preview_image = image
        self._preview_photo = ImageTk.PhotoImage(image)
        self.heightmap_label.configure(image=self._preview_photo, text="")

    def _on_invert_preview(self) -> None:
        """Flip the preview polarity; the STL is unaffected (display-only setting)."""
        inverted = self.invert_preview_var.get()
        self.heightmap_caption_var.set(
            HEIGHTMAP_CAPTION_INVERTED if inverted else HEIGHTMAP_CAPTION
        )
        if self._last_heightmap is not None:
            self._apply_heightmap_image(self._last_heightmap)
            self._update_preview_info(self._last_heightmap)

    # -- conversion ---------------------------------------------------------
    def _on_generate(self) -> None:
        try:
            config = self.current_config()
        except ConfigError as exc:
            messagebox.showerror("Invalid parameters", str(exc), parent=self)
            return

        destination = config.resolved_output
        if not self.output_path_var.get().strip():
            chosen = filedialog.asksaveasfilename(
                title="Save STL as", defaultextension=".stl",
                filetypes=[("STL", "*.stl"), ("All files", "*.*")],
                initialfile=str(destination),
            )
            if not chosen:
                return
            destination = Path(chosen)
            self.output_path_var.set(str(destination))

        overwrite = False
        if destination.exists():
            confirm = messagebox.askyesno(
                "Overwrite?",
                f"{destination} already exists.\nReplace it?",
                parent=self,
            )
            if not confirm:
                self.status_var.set("Conversion cancelled; the existing file was kept.")
                return
            overwrite = True
            self.force_var.set(True)

        self.start_conversion(config, destination, overwrite=overwrite)

    def start_conversion(
        self,
        config: LithoConfig,
        destination: Path,
        *,
        overwrite: bool = False,
    ) -> None:
        """Run :func:`lithomaker.core.convert` on a worker thread (LM-30)."""
        if self._busy:
            return
        self._set_busy(True)
        self.status_var.set("Converting…")
        self._queue = queue.Queue()

        def worker() -> None:
            try:
                result = convert(
                    config.input,
                    destination,
                    config,
                    overwrite=overwrite,
                    progress=lambda message, fraction: self._queue.put(
                        ("progress", message, fraction)
                    ),
                )
            except BaseException as exc:  # noqa: BLE001 - surfaced in the UI, never raised
                self._queue.put(("error", exc))
            else:
                self._queue.put(("done", result))

        threading.Thread(target=worker, daemon=True, name="lithomaker-convert").start()
        self._poll_queue()

    def _poll_queue(self) -> None:
        if self._queue is None:
            return
        try:
            message = self._queue.get_nowait()
        except queue.Empty:
            if self._busy:
                self.after(50, self._poll_queue)
            return

        kind = message[0]
        if kind == "progress":
            _, text, fraction = message
            self.status_var.set(text)
            self.progress_var.set(float(fraction) * 100.0)
            self.after(50, self._poll_queue)
        elif kind == "error":
            self._set_busy(False)
            self.status_var.set("Conversion failed.")
            messagebox.showerror("Conversion failed", str(message[1]), parent=self)
        else:
            self._set_busy(False)
            self._show_result(message[1])

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        self.generate_button.configure(state="disabled" if busy else "normal")
        if not busy:
            self.progress_var.set(0.0)

    def _show_result(self, result) -> None:
        self._last_output = Path(result.output_path)
        self.stats_vars["vertices"].set(f"{result.vertex_count:,}")
        self.stats_vars["triangles"].set(f"{result.triangle_count:,}")
        self.stats_vars["z_range"].set(f"{result.z_min:g}–{result.z_max:g} mm")
        self.stats_vars["footprint"].set(
            f"{result.resolved_size[0]:g} × {result.resolved_size[1]:g} mm"
        )
        self.stats_vars["file_size"].set(f"{result.file_size:,} bytes")
        self.show_button.configure(state="normal")
        self.status_var.set(f"Wrote {result.output_path}")
        for warning in result.warnings:
            self.status_var.set(f"Wrote {result.output_path} — warning: {warning}")

    def _show_in_folder(self) -> None:
        if self._last_output is None:
            return
        try:
            _open_in_file_manager(self._last_output)
        except OSError as exc:  # pragma: no cover - platform dependent
            messagebox.showerror("Cannot open folder", str(exc), parent=self)


# --------------------------------------------------------------------------
# Entry points
# --------------------------------------------------------------------------
def _configure_styles(root: tk.Misc) -> None:
    style = ttk.Style(root)
    for candidate in ("clam", "vista", "default"):
        if candidate in style.theme_names():
            style.theme_use(candidate)
            break
    style.configure("Heading.TLabel", font=("TkDefaultFont", 12, "bold"))
    style.configure("Muted.TLabel", foreground="#555555")


def main() -> None:
    """Open the GUI window (``lithomaker gui`` / ``python -m lithomaker.gui``)."""
    root = tk.Tk()
    root.title(f"Lithomaker {__version__}")
    root.minsize(900, 620)
    _configure_styles(root)
    app = LithomakerApp(root)
    app.pack(fill="both", expand=True)
    root.mainloop()


if __name__ == "__main__":  # pragma: no cover - interactive entry point
    main()
