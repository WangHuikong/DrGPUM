"""Command-line tool to convert 32-character Windows hex commands into
formatted 128-bit binary field breakdowns.

Usage example:
    python windows_hex_converter.py 05F821E300020000000C000000000000
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from typing import Iterable, List


BIT_FIELDS: List[tuple[int, int]] = [
    (127, 110),
    (109, 105),
    (104, 96),
    (95, 92),
    (91, 80),
    (79, 64),
    (63, 48),
    (47, 32),
    (31, 16),
    (15, 0),
]


@dataclass(frozen=True)
class BitField:
    label: str
    start: int
    end: int
    bit_length: int
    binary: str
    value: int

    @property
    def hex(self) -> str:
        """Return the hexadecimal representation of the bit field."""

        hex_digits = max(1, (self.bit_length + 3) // 4)
        return f"0x{self.value:0{hex_digits}X}"


HEX_PATTERN = re.compile(r"^[0-9a-fA-F]{32}$")


def normalize_hex(input_hex: str) -> str:
    """Normalize and validate the incoming hex string."""

    cleaned = input_hex.strip().replace(" ", "").replace("0x", "").replace("0X", "")
    if not HEX_PATTERN.fullmatch(cleaned):
        raise ValueError(
            "Input must be a 32-character hexadecimal string (128 bits)."
        )
    return cleaned.upper()


def split_bit_fields(binary: str) -> List[BitField]:
    """Split the 128-bit binary string into the configured bit fields."""

    if len(binary) != 128:
        raise ValueError("Binary string must be exactly 128 bits long.")

    fields: List[BitField] = []
    for start, end in BIT_FIELDS:
        bit_length = start - end + 1
        slice_start = 127 - start
        slice_end = 127 - end
        bit_slice = binary[slice_start : slice_end + 1]
        fields.append(
            BitField(
                label=f"{start}-{end}",
                start=start,
                end=end,
                bit_length=bit_length,
                binary=bit_slice,
                value=int(bit_slice, 2),
            )
        )
    return fields


def group_binary(binary: str, group_size: int = 4) -> str:
    """Group the binary string into chunks to improve readability."""

    return " ".join(binary[i : i + group_size] for i in range(0, len(binary), group_size))


def convert_hex_to_bitfields(input_hex: str) -> tuple[str, List[BitField]]:
    """Convert the input hex string into its 128-bit binary and bit fields."""

    normalized = normalize_hex(input_hex)
    as_int = int(normalized, 16)
    binary = f"{as_int:0128b}"
    fields = split_bit_fields(binary)
    return binary, fields


def format_fields(fields: Iterable[BitField]) -> str:
    """Format bit fields into a human-readable table."""

    header = f"{'Field':>9} {'Bits':>4} {'Binary':>50} {'Hex':>10} {'Dec':>20}"
    lines = [header, "-" * len(header)]
    for field in fields:
        lines.append(
            f"{field.label:>9} {field.bit_length:>4} {field.binary:>50} {field.hex:>10} {field.value:>20}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a 32-character Windows hexadecimal command into a 128-bit "
            "binary representation segmented by predefined bit fields."
        )
    )
    parser.add_argument(
        "hex_value",
        help="32-character hexadecimal string to convert (with or without 0x prefix).",
    )
    args = parser.parse_args()

    try:
        binary, fields = convert_hex_to_bitfields(args.hex_value)
    except ValueError as exc:
        parser.error(str(exc))

    print(f"Input Hex: 0x{normalize_hex(args.hex_value)}")
    print("128-bit Binary:")
    print(group_binary(binary))
    print()
    print(format_fields(fields))


if __name__ == "__main__":
    main()
