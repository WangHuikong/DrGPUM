#!/usr/bin/env python3
"""Generate 32-bit hex values from 8-bit incrementing integers."""

from __future__ import annotations

import argparse
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate 32-bit hex values composed of 4 incrementing 8-bit integers."
        )
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("hex_words.txt"),
        help="Output file path (default: hex_words.txt).",
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
        help="Starting 8-bit integer value (default: 0).",
    )
    parser.add_argument(
        "-e",
        "--endian",
        choices=("big", "little"),
        default="big",
        help="Byte order when composing hex (default: big).",
    )
    return parser.parse_args()


def generate_hex_words(output: Path, lines: int, start: int, endian: str) -> None:
    if lines < 0:
        raise ValueError("lines must be non-negative")
    if start < 0:
        raise ValueError("start must be non-negative")

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="ascii", newline="\n") as file:
        for line_idx in range(lines):
            base = (start + line_idx * 4) % 256
            bytes_seq = [(base + offset) % 256 for offset in range(4)]
            if endian == "little":
                bytes_seq.reverse()
            file.write("".join(f"{byte:02x}" for byte in bytes_seq))
            file.write("\n")


def main() -> None:
    args = _parse_args()
    generate_hex_words(args.output, args.lines, args.start, args.endian)


if __name__ == "__main__":
    main()
