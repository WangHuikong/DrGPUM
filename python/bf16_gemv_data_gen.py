#!/usr/bin/env python3
"""
Generate BF16 GEMV test data:
  A: (m, k)
  B: (k, n)
  C: (m, n), where C = A @ B

Default shape is:
  A: (1, 4096), B: (4096, 4096), C: (1, 4096)

Data generation rule:
  - A is filled with BF16 value 1.0.
  - B can be generated in two modes:
    1) range: repeat integers in [b_min, b_max]
    2) ones:  all B values are BF16 1.0
  - Default B mode is range with cycle [0, 1, ..., 10].

Hex output rule:
  - Pack 2 BF16 values into one 32-bit word per line.
  - Default packing is "little" (example: [0.0, 1.0] -> 3F800000).
  - Output one 8-hex-digit word per line (uppercase).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def float32_to_bf16_bits(values: np.ndarray) -> np.ndarray:
    """Convert float32 array to BF16 bit-patterns (uint16) with RNE rounding."""
    values = np.asarray(values, dtype=np.float32)
    u32 = values.view(np.uint32)
    rounding_bias = np.uint32(0x7FFF) + ((u32 >> np.uint32(16)) & np.uint32(1))
    return ((u32 + rounding_bias) >> np.uint32(16)).astype(np.uint16)


def bf16_bits_to_float32(bits: np.ndarray) -> np.ndarray:
    """Convert BF16 bit-patterns (uint16) to float32 values."""
    bits = np.asarray(bits, dtype=np.uint16)
    return (bits.astype(np.uint32) << np.uint32(16)).view(np.float32)


def pack_two_bf16_to_u32(bits: np.ndarray, endianness: str) -> np.ndarray:
    """Pack BF16 words by 2 into uint32."""
    bits = np.asarray(bits, dtype=np.uint16).reshape(-1)
    if bits.size % 2 != 0:
        bits = np.pad(bits, (0, 1), mode="constant")

    pairs = bits.reshape(-1, 2).astype(np.uint32)
    if endianness == "big":
        # Keep textual/visual order intuitive:
        # [v0, v1] -> 0xVVVVWWWW (v0 in high 16 bits).
        return (pairs[:, 0] << np.uint32(16)) | pairs[:, 1]
    return pairs[:, 0] | (pairs[:, 1] << np.uint32(16))


def write_bf16_hex_file(bits: np.ndarray, out_path: Path, endianness: str, with_0x: bool) -> None:
    """Write a BF16 array in packed-2 format (one uint32 hex per line)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    packed = pack_two_bf16_to_u32(bits, endianness)
    fmt = "0x%08X" if with_0x else "%08X"
    np.savetxt(out_path, packed, fmt=fmt)


def build_b_cycle_range(b_min: int, b_max: int) -> tuple[np.ndarray, np.ndarray]:
    """Build one B cycle from integer source b_min..b_max."""
    src = np.arange(b_min, b_max + 1, dtype=np.float32)
    cycle_bits = float32_to_bf16_bits(src)
    cycle_f32 = bf16_bits_to_float32(cycle_bits)

    if cycle_bits.size == 0:
        raise ValueError("B cycle is empty; check b_min/b_max")
    return cycle_bits.astype(np.uint16), cycle_f32.astype(np.float32)


def build_b_cycle_ones() -> tuple[np.ndarray, np.ndarray]:
    """Build B cycle where every element is BF16 1.0."""
    cycle_bits = np.array([np.uint16(0x3F80)], dtype=np.uint16)
    cycle_f32 = bf16_bits_to_float32(cycle_bits)
    return cycle_bits, cycle_f32.astype(np.float32)


def build_b_cycle(mode: str, b_min: int, b_max: int) -> tuple[np.ndarray, np.ndarray]:
    """Dispatch B cycle builder by mode."""
    if mode == "ones":
        return build_b_cycle_ones()
    return build_b_cycle_range(b_min=b_min, b_max=b_max)


def generate_b_and_compute_c(
    a_bf16_f32: np.ndarray,
    k: int,
    n: int,
    b_cycle_bits: np.ndarray,
    b_cycle_f32: np.ndarray,
    b_out_path: Path,
    endianness: str,
    with_0x: bool,
    progress_every: int,
) -> np.ndarray:
    """
    Stream-generate B, write B hex, and accumulate C in float32:
      C += A[:, i] * B[i, :]
    """
    m = a_bf16_f32.shape[0]
    c_acc = np.zeros((m, n), dtype=np.float32)

    cycle_len = int(b_cycle_bits.shape[0])
    col_base = np.arange(n, dtype=np.int64)
    fmt = "0x%08X" if with_0x else "%08X"

    b_out_path.parent.mkdir(parents=True, exist_ok=True)
    with b_out_path.open("w", encoding="ascii") as fp:
        for row in range(k):
            row_offset = row * n
            row_idx = (col_base + row_offset) % cycle_len
            row_bits = b_cycle_bits[row_idx]
            row_bf16_f32 = b_cycle_f32[row_idx]

            packed = pack_two_bf16_to_u32(row_bits, endianness)
            np.savetxt(fp, packed, fmt=fmt)

            c_acc += a_bf16_f32[:, row : row + 1] * row_bf16_f32[None, :]

            if progress_every > 0 and ((row + 1) % progress_every == 0 or row + 1 == k):
                print(f"[progress] processed B row {row + 1}/{k}")

    return c_acc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate BF16 GEMV A/B/C test data in packed hex format."
    )
    parser.add_argument("--m", type=int, default=1, help="A rows")
    parser.add_argument("--k", type=int, default=4096, help="A cols / B rows")
    parser.add_argument("--n", type=int, default=4096, help="B cols / C cols")
    parser.add_argument(
        "--b-mode",
        choices=["range", "ones"],
        default="range",
        help="B generation mode: range repeats integers [b_min,b_max], ones uses constant 1.0",
    )
    parser.add_argument("--b-min", type=int, default=0, help="B cycle integer minimum")
    parser.add_argument("--b-max", type=int, default=10, help="B cycle integer maximum")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("bf16_gemv_data"),
        help="Output directory",
    )
    parser.add_argument("--a-file", default="A_hex.txt", help="A output filename")
    parser.add_argument("--b-file", default="B_hex.txt", help="B output filename")
    parser.add_argument("--c-file", default="C_hex.txt", help="C output filename")
    parser.add_argument(
        "--endianness",
        choices=["little", "big"],
        default="little",
        help="Packing order for two BF16 words into one uint32 (default: [v0,v1] -> v1v0 hex)",
    )
    parser.add_argument(
        "--with-0x",
        action="store_true",
        help="Prefix each hex line with 0x",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=256,
        help="Print progress every N rows of B (0 to disable)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.m <= 0 or args.k <= 0 or args.n <= 0:
        raise ValueError("m, k, n must be positive integers")
    if args.b_mode == "range" and args.b_min > args.b_max:
        raise ValueError("b_min must be <= b_max")

    out_dir = args.output_dir
    a_path = out_dir / args.a_file
    b_path = out_dir / args.b_file
    c_path = out_dir / args.c_file

    print(f"[info] generating A({args.m}, {args.k}), B({args.k}, {args.n})")
    if args.b_mode == "ones":
        print(f"[info] B mode=ones (all values are 1.0), endianness={args.endianness}")
    else:
        print(f"[info] B mode=range, integer cycle=[{args.b_min}..{args.b_max}], endianness={args.endianness}")
    print(f"[info] output dir: {out_dir.resolve()}")

    # A is constant BF16 1.0 for all elements.
    a_bits = np.full((args.m, args.k), np.uint16(0x3F80), dtype=np.uint16)
    a_bf16_f32 = bf16_bits_to_float32(a_bits)
    write_bf16_hex_file(a_bits.reshape(-1), a_path, args.endianness, args.with_0x)
    print(f"[info] wrote A hex -> {a_path}")

    b_cycle_bits, b_cycle_f32 = build_b_cycle(mode=args.b_mode, b_min=args.b_min, b_max=args.b_max)
    print(f"[info] B cycle length={b_cycle_bits.size}")

    # Stream B generation and C accumulation in float32.
    c_acc = generate_b_and_compute_c(
        a_bf16_f32=a_bf16_f32,
        k=args.k,
        n=args.n,
        b_cycle_bits=b_cycle_bits,
        b_cycle_f32=b_cycle_f32,
        b_out_path=b_path,
        endianness=args.endianness,
        with_0x=args.with_0x,
        progress_every=args.progress_every,
    )
    print(f"[info] wrote B hex -> {b_path}")

    # C is BF16 output.
    c_bits = float32_to_bf16_bits(c_acc.reshape(-1))
    write_bf16_hex_file(c_bits, c_path, args.endianness, args.with_0x)
    print(f"[info] wrote C hex -> {c_path}")
    print("[done] all files generated.")


if __name__ == "__main__":
    main()
