#!/usr/bin/env python3
"""Regenerate ``examples/sample.png``.

The sample is a deterministic 256x256 test pattern (no randomness, so re-running this
script reproduces byte-identical pixels):

* a vertical luminance ramp (black at the top, white at the bottom),
* a radial falloff brightening the centre,
* a flat mid-grey border, plus black and white reference patches that pin the
  histogram to exactly 0 and 255.

It carries its own orientation cues on purpose: the black patch in the top-left makes
it obvious in the generated STL whether the image was flipped, mirrored or stretched,
and the black/white patches make the bright=thin inversion obvious by eye.

Usage::

    python tools/make_sample.py [--size 256] [--output examples/sample.png]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

DEFAULT_SIZE = 256
DEFAULT_OUTPUT = Path(__file__).resolve().parent.parent / "examples" / "sample.png"


def build_pattern(size: int = DEFAULT_SIZE) -> np.ndarray:
    """Return a ``(size, size)`` float grayscale array in [0, 1]."""
    if size < 8:
        raise ValueError(f"size must be at least 8 pixels, got {size}")

    y, x = np.mgrid[0:size, 0:size]
    last = size - 1

    # Vertical ramp: 0.0 at the top row, 1.0 at the bottom row.
    ramp = y / last

    # Radial falloff: 1.0 at the centre, 0.0 at the inscribed-circle edge and beyond.
    centre = last / 2.0
    radius = np.hypot(x - centre, y - centre) / centre
    radial = 1.0 - np.clip(radius, 0.0, 1.0)

    value = 0.7 * ramp + 0.3 * radial

    # Spread the midtones across the full range. Without this the two terms are
    # anti-correlated (the ramp peaks along the bottom edge, where the radial term has
    # already decayed to zero) and the pattern only spans about 0..0.7.
    low, high = float(value.min()), float(value.max())
    value = (value - low) / (high - low)

    # A flat mid-grey border frames the pattern.
    frame = max(1, size // 64)
    value[:frame, :] = 0.5
    value[-frame:, :] = 0.5
    value[:, :frame] = 0.5
    value[:, -frame:] = 0.5

    # Reference patches, deliberately *inside* the frame so the border cannot clobber
    # them: a black square at the origin (top-left) and a white square at the bottom
    # right. Besides pinning min/max to exactly 0 and 255, they make the orientation
    # and the bright=thin inversion convention easy to check by eye in the STL.
    patch = max(2, size // 8)
    value[frame : frame + patch, frame : frame + patch] = 0.0
    value[-frame - patch : -frame, -frame - patch : -frame] = 1.0

    return np.clip(value, 0.0, 1.0)


def build_image(size: int = DEFAULT_SIZE) -> Image.Image:
    """Build the sample as an 8-bit RGB PIL image."""
    gray = np.round(build_pattern(size) * 255.0).astype(np.uint8)
    rgb = np.repeat(gray[:, :, np.newaxis], 3, axis=2)
    return Image.fromarray(rgb, mode="RGB")


def main(argv: list = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE, help="edge length in pixels")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="destination PNG path (default: examples/sample.png)",
    )
    args = parser.parse_args(argv)

    image = build_image(args.size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output, format="PNG", optimize=True)

    gray = np.asarray(image.convert("L"), dtype=np.uint8)
    print(f"wrote {args.output} ({args.size}x{args.size}, {args.output.stat().st_size:,} bytes)")
    print(
        f"luminance: min={gray.min()} max={gray.max()} "
        f"mean={gray.mean():.1f} unique={len(np.unique(gray))}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
