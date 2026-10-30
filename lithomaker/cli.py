"""Command-line interface (LM-20 / LM-21).

The CLI is a thin translator: every parameter is collected here and handed to
:class:`lithomaker.config.LithoConfig`, every conversion goes through
:func:`lithomaker.core.convert`. No validation, geometry or file handling lives in this
module — that keeps plan §7's CLI/GUI parity meaningful (both frontends are wrappers
around the same core).

Exit codes:

* ``0`` — success.
* ``1`` — runtime failure (missing/corrupt image, refusing to overwrite, disk error).
* ``2`` — usage failure (bad or contradictory parameters), Click's convention.

``--verbose`` streams progress to stderr and lets unexpected exceptions escape with a
traceback; without it every failure is a one-line ``Error: …`` message.
"""

from __future__ import annotations

import platform
import sys
from pathlib import Path

import click

from . import __version__
from .config import ConfigError, LevelsConfig, LithoConfig
from .core import convert
from .image_io import ImageLoadError

__all__ = ["cli", "main"]

CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"]}

def _echo_progress(message: str, fraction: float) -> None:
    click.echo(f"[{fraction:4.0%}] {message}", err=True)


def _pick(primary, alias, primary_name: str, alias_name: str):
    """Resolve a canonical value against its ``-i``/``-o`` alias (decision D5).

    Two spellings for the same parameter are accepted, but not at the same time: that
    is almost always a shell-history accident, and silently preferring one would hide
    it.
    """
    if primary is not None and alias is not None:
        raise click.UsageError(
            f"{primary_name} and {alias_name} were both given; pass the value once"
        )
    value = primary if primary is not None else alias
    if value is None:
        raise click.UsageError(f"missing required argument: {primary_name}")
    return value


def _resolve_auto_levels(levels_choice: str, auto_flag: bool | None) -> bool:
    """Combine ``--levels auto`` with the ``--auto-levels/--no-auto-levels`` pair (D7)."""
    if auto_flag is not None:
        if levels_choice != "manual" and (levels_choice == "auto") != auto_flag:
            raise click.UsageError("--levels and --auto-levels/--no-auto-levels disagree")
        return auto_flag
    return levels_choice == "auto"


def _resolve_format(ascii_flag: bool | None, format_choice: str | None) -> bool:
    """Combine ``--ascii/--binary`` with ``--format`` (decision D8)."""
    if ascii_flag is not None and format_choice is not None:
        if ascii_flag != (format_choice == "ascii"):
            raise click.UsageError("--ascii/--binary and --format disagree")
        return ascii_flag
    if format_choice is not None:
        return format_choice == "ascii"
    return bool(ascii_flag)


def _build_config(
    *,
    image: Path,
    output: Path | None,
    black: float,
    white: float,
    gamma: float,
    auto_levels: bool,
    min_layers: int,
    max_layers: int,
    thickness: float,
    size_x: float,
    size_y: float | None,
    dot_size: float,
    base_thickness: float,
    ascii_mode: bool,
) -> LithoConfig:
    """Assemble the shared parameter object; every rule lives in ``config.py``."""
    try:
        return LithoConfig(
            input=image,
            output=output,
            levels=LevelsConfig(black=black, white=white, gamma=gamma, auto=auto_levels),
            min_layers=min_layers,
            max_layers=max_layers,
            thickness=thickness,
            size_x=size_x,
            size_y=size_y,
            dot_size=dot_size,
            base_thickness=base_thickness,
            ascii=ascii_mode,
        )
    except ConfigError as exc:
        raise click.UsageError(str(exc)) from exc


def _format_stats(result, ascii_mode: bool) -> str:
    """Compact human-readable stats table (fixed-width labels, one value per line)."""
    rows = [
        ("vertices", f"{result.vertex_count:,}"),
        ("triangles", f"{result.triangle_count:,}"),
        ("z range", f"{result.z_min:g} .. {result.z_max:g} mm"),
        ("total height", f"{result.total_height:g} mm"),
        ("footprint", f"{result.resolved_size[0]:g} x {result.resolved_size[1]:g} mm"),
        ("grid", f"{result.grid_shape[0]} x {result.grid_shape[1]}"),
        ("file size", f"{result.file_size:,} bytes"),
        ("format", "ASCII STL" if ascii_mode else "binary STL"),
    ]
    width = max(len(label) for label, _ in rows)
    return "\n".join(f"{label:<{width}}  {value}" for label, value in rows)


@click.group(context_settings=CONTEXT_SETTINGS)
@click.version_option(__version__, "--version", prog_name="lithomaker")
def cli() -> None:
    """Convert a 2D picture into a 3D-printable lithophane STL.

    \b
    Examples:
      lithomaker convert photo.png
      lithomaker convert photo.png lithophane.stl --size-x 100 --max-layers 20
      lithomaker convert -i photo.png -o out.stl --ascii --stats
    """
    pass


@cli.command(name="convert", context_settings=CONTEXT_SETTINGS)
@click.argument("input_image", required=False, metavar="[INPUT_IMAGE]",
                 type=click.Path(dir_okay=False))
@click.argument("output_arg", required=False, metavar="[OUTPUT.STL]",
                 type=click.Path(dir_okay=False))
@click.option(
    "-i", "--input", "input_flag", type=click.Path(dir_okay=False),
    help="Input image (alias of the INPUT_IMAGE positional argument).",
)
@click.option(
    "-o", "--output", "output_flag", type=click.Path(dir_okay=False),
    help="Output STL (alias of the OUTPUT positional argument).",
)
@click.option("-b", "--black", type=float, default=0.0, show_default=True,
              help="Levels black point, 0..1.")
@click.option("-w", "--white", type=float, default=1.0, show_default=True,
              help="Levels white point, 0..1.")
@click.option("-g", "--gamma", type=float, default=1.0, show_default=True,
              help="Levels gamma, > 0 (values above 1 darken midtones).")
@click.option("--levels", "levels_choice", type=click.Choice(["auto", "manual"]),
              default="manual", show_default=True,
              help="Derive the levels window from the histogram of the image.")
@click.option("--auto-levels/--no-auto-levels", "auto_flag", default=None,
              help="Boolean spelling of --levels auto; ignores --black/--white/--gamma.")
@click.option("--min-layers", type=int, default=4, show_default=True,
              help="Minimum Z layers above the base.")
@click.option("--max-layers", type=int, default=32, show_default=True,
              help="Maximum Z layers above the base.")
@click.option("--thickness", type=float, default=0.2, show_default=True,
              help="Layer thickness in mm.")
@click.option("--size-x", type=float, default=120.0, show_default=True,
              help="Output width in mm.")
@click.option("--size-y", type=float, default=None,
              help="Output height in mm (defaults to aspect-correct).")
@click.option("--dot-size", type=float, default=0.4, show_default=True,
              help="XY dot/pixel pitch in mm; smaller means more detail and more triangles.")
@click.option("--base-thickness", type=float, default=1.0, show_default=True,
              help="Flat base plate thickness in mm.")
@click.option("--ascii/--binary", "ascii_flag", default=None,
              help="STL flavour (default: binary).")
@click.option("--format", "format_choice", type=click.Choice(["binary", "ascii"]),
              default=None,
              help="STL flavour, long-form spelling of --ascii/--binary.")
@click.option("--force", is_flag=True,
              help="Overwrite the output file if it already exists.")
@click.option("-v", "--verbose", is_flag=True,
              help="Print progress to stderr and show tracebacks for unexpected errors.")
@click.option("--stats/--no-stats", default=True, show_default=True,
              help="Print a summary table after conversion.")
def convert_command(
    input_image,
    output_arg,
    input_flag,
    output_flag,
    black,
    white,
    gamma,
    levels_choice,
    auto_flag,
    min_layers,
    max_layers,
    thickness,
    size_x,
    size_y,
    dot_size,
    base_thickness,
    ascii_flag,
    format_choice,
    force,
    verbose,
    stats,
) -> None:
    """Convert INPUT_IMAGE (or -i) to OUTPUT.STL (or -o).

    \b
    OUTPUT is optional and defaults to the input name with a .stl suffix.
    The first line on stdout is always the written path; use --no-stats to
    keep it the only line.
    """
    image = Path(_pick(input_image, input_flag, "INPUT_IMAGE", "-i/--input"))
    if output_arg is not None and output_flag is not None:
        raise click.UsageError(
            "OUTPUT.STL and -o/--output were both given; pass the value once"
        )
    output = None
    if output_arg is not None:
        output = Path(output_arg)
    elif output_flag is not None:
        output = Path(output_flag)
    ascii_mode = _resolve_format(ascii_flag, format_choice)
    auto_levels = _resolve_auto_levels(levels_choice, auto_flag)

    config = _build_config(
        image=image,
        output=output,
        black=black,
        white=white,
        gamma=gamma,
        auto_levels=auto_levels,
        min_layers=min_layers,
        max_layers=max_layers,
        thickness=thickness,
        size_x=size_x,
        size_y=size_y,
        dot_size=dot_size,
        base_thickness=base_thickness,
        ascii_mode=ascii_mode,
    )

    if verbose:
        click.echo(
            f"image     {image}\n"
            f"output    {config.resolved_output}\n"
            f"levels    {config.levels.describe()}\n"
            f"layers    {config.min_layers}..{config.max_layers} x {config.thickness:g} mm\n"
            f"size      {config.size_x:g} x "
            f"{'auto' if config.size_y is None else f'{config.size_y:g}'} mm\n"
            f"dot size  {config.dot_size:g} mm",
            err=True,
        )

    try:
        result = convert(
            image,
            output,
            config,
            overwrite=force,
            progress=_echo_progress if verbose else None,
        )
    except ImageLoadError as exc:
        raise click.ClickException(str(exc)) from exc
    except ConfigError as exc:
        raise click.UsageError(str(exc)) from exc
    except FileExistsError as exc:
        raise click.ClickException(f"{exc} (use --force to overwrite)") from exc
    except OSError as exc:
        raise click.ClickException(str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive: keep stdout clean
        if verbose:
            raise
        raise click.ClickException(f"{type(exc).__name__}: {exc}") from exc

    for warning in result.warnings:
        click.echo(f"warning: {warning}", err=True)

    click.echo(result.output_path)
    if stats:
        click.echo(_format_stats(result, ascii_mode))


@cli.command()
def info() -> None:
    """Print versions useful in bug reports."""
    from importlib.metadata import PackageNotFoundError, version

    def _version(distribution: str) -> str:
        try:
            return version(distribution)
        except PackageNotFoundError:  # pragma: no cover - only in a broken install
            return "not installed"

    click.echo(f"lithomaker  {__version__}")
    click.echo(f"python      {platform.python_version()} ({sys.platform})")
    click.echo(f"pillow      {_version('Pillow')}")
    click.echo(f"numpy       {_version('numpy')}")
    click.echo(f"click       {_version('click')}")


def main() -> None:
    """Console-script entry point (``lithomaker = lithomaker.cli:main``)."""
    cli(prog_name="lithomaker")


if __name__ == "__main__":  # pragma: no cover - exercised via the console script
    main()
