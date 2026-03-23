#!/usr/bin/env python3
"""Generate packed 32-bit hex from incrementing fp16 values."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate 32-bit hex values composed of two incrementing fp16 numbers."
        )
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("fp16_hex.txt"),
        help="Output file path (default: fp16_hex.txt).",
    )
    parser.add_argument(
        "-n",
        "--lines",
        type=int,
        default=262144,
        help="Number of 32-bit values to generate (default: 262144).",
    )
    parser.add_argument(
        "-s",
        "--start",
        type=int,
        default=0,
        help="Starting integer value (default: 0).",
    )
    parser.add_argument(
        "-w",
        "--wrap",
        type=int,
        default=65536,
        help="Wrap value for integers (default: 65536, 0 to disable).",
    )
    parser.add_argument(
        "-e",
        "--endian",
        choices=("big", "little"),
        default="big",
        help="Byte order when packing fp16 values (default: big).",
    )
    return parser.parse_args()


def _fp16_bits(value: int) -> int:
    return np.asarray(value, dtype=np.float16).view(np.uint16).item()


def generate_fp16_hex(
    output: Path, lines: int, start: int, wrap: int, endian: str
) -> None:
    if lines < 0:
        raise ValueError("lines must be non-negative")
    if start < 0:
        raise ValueError("start must be non-negative")
    if wrap < 0:
        raise ValueError("wrap must be non-negative")

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="ascii", newline="\n") as file:
        for line_idx in range(lines):
            base = start + line_idx * 2
            values = [base, base + 1]
            if wrap:
                values = [value % wrap for value in values]

            words = [_fp16_bits(value) for value in values]
            if endian == "big":
                packed = (words[0] << 16) | words[1]
            else:
                packed = (words[1] << 16) | words[0]

            file.write(f"{packed:08x}\n")


def main() -> None:
    args = _parse_args()
    generate_fp16_hex(args.output, args.lines, args.start, args.wrap, args.endian)


if __name__ == "__main__":
    main()
