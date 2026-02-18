#!/usr/bin/env python3
"""
Generate BF16 GEMV test data:
  A: (m, k)
  B: (k, n)
  C: (m, n), where C = A @ B

Default shape is:
  A: (1, 4096), B: (4096, 4096), C: (1, 4096)

Data generation rule:
  - A and B start from 0 and increment by 1.
  - When value exceeds wrap_value, it wraps to 0.
  - Default wrap_value is 65504 (max finite FP16 value).

Hex output rule:
  - Pack 2 BF16 values into one 32-bit word per line.
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
    if endianness == "little":
        return pairs[:, 0] | (pairs[:, 1] << np.uint32(16))
    return (pairs[:, 0] << np.uint32(16)) | pairs[:, 1]


def write_bf16_hex_file(bits: np.ndarray, out_path: Path, endianness: str, with_0x: bool) -> None:
    """Write a BF16 array in packed-2 format (one uint32 hex per line)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    packed = pack_two_bf16_to_u32(bits, endianness)
    fmt = "0x%08X" if with_0x else "%08X"
    np.savetxt(out_path, packed, fmt=fmt)


def make_wrapped_sequence(count: int, wrap_value: int) -> np.ndarray:
    """Create float32 sequence: 0,1,2,...,wrap_value,0,1,2,..."""
    if wrap_value < 0:
        raise ValueError("wrap_value must be non-negative")
    modulo = np.uint32(wrap_value + 1)
    seq = np.arange(count, dtype=np.uint32) % modulo
    return seq.astype(np.float32)


def generate_b_and_compute_c(
    a_bf16_f32: np.ndarray,
    k: int,
    n: int,
    wrap_value: int,
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

    modulo = np.uint32(wrap_value + 1)
    col_base = np.arange(n, dtype=np.uint32)
    fmt = "0x%08X" if with_0x else "%08X"

    b_out_path.parent.mkdir(parents=True, exist_ok=True)
    with b_out_path.open("w", encoding="ascii") as fp:
        for row in range(k):
            row_offset = np.uint32(row * n)
            row_vals = (col_base + row_offset) % modulo
            row_f32 = row_vals.astype(np.float32)
            row_bits = float32_to_bf16_bits(row_f32)

            packed = pack_two_bf16_to_u32(row_bits, endianness)
            np.savetxt(fp, packed, fmt=fmt)

            row_bf16_f32 = bf16_bits_to_float32(row_bits)
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
        "--wrap-value",
        type=int,
        default=65504,
        help="Sequence wraps after this value (default: FP16 max finite 65504)",
    )
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
        help="Packing order for two BF16 words into one uint32",
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
    if args.wrap_value < 0:
        raise ValueError("wrap_value must be non-negative")

    out_dir = args.output_dir
    a_path = out_dir / args.a_file
    b_path = out_dir / args.b_file
    c_path = out_dir / args.c_file

    print(f"[info] generating A({args.m}, {args.k}), B({args.k}, {args.n})")
    print(f"[info] wrap_value={args.wrap_value}, endianness={args.endianness}")
    print(f"[info] output dir: {out_dir.resolve()}")

    # A sequence: 0..wrap_value looping, then quantize to BF16.
    a_seq = make_wrapped_sequence(args.m * args.k, args.wrap_value).reshape(args.m, args.k)
    a_bits = float32_to_bf16_bits(a_seq.reshape(-1)).reshape(args.m, args.k)
    a_bf16_f32 = bf16_bits_to_float32(a_bits)
    write_bf16_hex_file(a_bits.reshape(-1), a_path, args.endianness, args.with_0x)
    print(f"[info] wrote A hex -> {a_path}")

    # Stream B generation and C accumulation in float32.
    c_acc = generate_b_and_compute_c(
        a_bf16_f32=a_bf16_f32,
        k=args.k,
        n=args.n,
        wrap_value=args.wrap_value,
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
