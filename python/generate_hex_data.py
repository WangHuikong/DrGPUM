#!/usr/bin/env python3
"""Generate sequential lowercase hex lines to a file."""

from __future__ import annotations

import argparse
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate sequential lowercase hex lines."
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("hex_data.txt"),
        help="Output file path (default: hex_data.txt).",
    )
    parser.add_argument(
        "-n",
        "--lines",
        type=int,
        default=262144,
        help="Number of lines to generate (default: 262144).",
    )
    parser.add_argument(
        "-w",
        "--width",
        type=int,
        default=8,
        help="Hex width with zero padding (default: 8).",
    )
    parser.add_argument(
        "-s",
        "--start",
        type=int,
        default=0,
        help="Start value (default: 0).",
    )
    return parser.parse_args()


def generate_hex_data(output: Path, lines: int, width: int, start: int) -> None:
    if lines < 0:
        raise ValueError("lines must be non-negative")
    if width < 1:
        raise ValueError("width must be at least 1")
    if start < 0:
        raise ValueError("start must be non-negative")

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="ascii", newline="\n") as file:
        for i in range(lines):
            file.write(f"{start + i:0{width}x}\n")


def main() -> None:
    args = _parse_args()
    generate_hex_data(args.output, args.lines, args.width, args.start)


if __name__ == "__main__":
    main()
