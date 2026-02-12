#!/usr/bin/env python3
"""Generate bf16 GEMV vectors/matrix and export packed-hex text files.

Output format:
- One line is one 32-bit word in hex (8 hex chars).
- Each 32-bit word packs two bf16 values:
  low 16 bits  = element[i]
  high 16 bits = element[i + 1]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def float32_to_bf16_bits(x: np.ndarray) -> np.ndarray:
    """Convert float32 array to bf16 bit-patterns (round-to-nearest-even)."""
    x32 = np.asarray(x, dtype=np.float32)
    bits = x32.view(np.uint32)
    rounding_bias = np.uint32(0x7FFF) + ((bits >> 16) & np.uint32(1))
    return ((bits + rounding_bias) >> 16).astype(np.uint16)


def bf16_bits_to_float32(bits: np.ndarray) -> np.ndarray:
    """Convert bf16 bit-patterns to float32 values."""
    b16 = np.asarray(bits, dtype=np.uint16)
    return (b16.astype(np.uint32) << 16).view(np.float32)


def pack_two_bf16_per_word(bits: np.ndarray) -> np.ndarray:
    """Pack every 2 bf16 values into one uint32 word."""
    flat = np.asarray(bits, dtype=np.uint16).reshape(-1)
    if flat.size % 2 != 0:
        raise ValueError(f"bf16 element count must be even, got {flat.size}")
    pairs = flat.reshape(-1, 2).astype(np.uint32)
    return pairs[:, 0] | (pairs[:, 1] << 16)


def write_packed_hex(path: Path, bf16_bits: np.ndarray) -> None:
    """Write packed uint32 hex words, one per line."""
    words = pack_two_bf16_per_word(bf16_bits)
    chunk_words = 262_144
    with path.open("w", encoding="ascii") as f:
        for start in range(0, words.size, chunk_words):
            chunk = words[start : start + chunk_words]
            f.writelines(f"{int(word):08X}\n" for word in chunk)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate bf16 GEMV data: A(1,K), B(K,N), C(1,N)=A@B."
    )
    parser.add_argument("--k", type=int, default=4096, help="K dimension")
    parser.add_argument("--n", type=int, default=4096, help="N dimension")
    parser.add_argument("--seed", type=int, default=20260212, help="RNG seed")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("python"),
        help="Output directory for generated hex files",
    )
    parser.add_argument(
        "--low",
        type=float,
        default=-1.0,
        help="Lower bound of random uniform input",
    )
    parser.add_argument(
        "--high",
        type=float,
        default=1.0,
        help="Upper bound of random uniform input",
    )
    args = parser.parse_args()

    if args.high <= args.low:
        raise ValueError("--high must be greater than --low")

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    a_f32 = rng.uniform(args.low, args.high, size=(1, args.k)).astype(np.float32)
    b_f32 = rng.uniform(args.low, args.high, size=(args.k, args.n)).astype(np.float32)

    # Quantize inputs to bf16 first to model bf16 operands.
    a_bf16_bits = float32_to_bf16_bits(a_f32)
    b_bf16_bits = float32_to_bf16_bits(b_f32)
    a_bf16_f32 = bf16_bits_to_float32(a_bf16_bits)
    b_bf16_f32 = bf16_bits_to_float32(b_bf16_bits)

    c_f32 = (a_bf16_f32 @ b_bf16_f32).astype(np.float32)
    c_bf16_bits = float32_to_bf16_bits(c_f32)

    a_path = out_dir / f"bf16_gemv_A_1x{args.k}.hex"
    b_path = out_dir / f"bf16_gemv_B_{args.k}x{args.n}.hex"
    c_path = out_dir / f"bf16_gemv_C_1x{args.n}.hex"
    meta_path = out_dir / f"bf16_gemv_meta_1x{args.k}_{args.k}x{args.n}.txt"

    write_packed_hex(a_path, a_bf16_bits.reshape(-1))
    write_packed_hex(b_path, b_bf16_bits.reshape(-1))
    write_packed_hex(c_path, c_bf16_bits.reshape(-1))

    with meta_path.open("w", encoding="ascii") as f:
        f.write(f"seed={args.seed}\n")
        f.write(f"A_shape=1x{args.k}\n")
        f.write(f"B_shape={args.k}x{args.n}\n")
        f.write(f"C_shape=1x{args.n}\n")
        f.write("layout=row-major flatten before packing\n")
        f.write("pack=uint32_word: low16=A[i], high16=A[i+1]\n")
        f.write("hex_format=one_uint32_per_line\n")
        f.write(f"input_uniform_range=[{args.low}, {args.high})\n")

    print(f"Generated: {a_path}")
    print(f"Generated: {b_path}")
    print(f"Generated: {c_path}")
    print(f"Generated: {meta_path}")


if __name__ == "__main__":
    main()
