#!/usr/bin/env python3
"""
Split a JSONL dataset into train / val / test files.

Usage:
    python split_jsonl.py --input data/rapid_1000_1100_small.jsonl --out-dir data/splits_1000_1100
"""

import json
import random
from pathlib import Path
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input JSONL file.")
    parser.add_argument("--out-dir", required=True, help="Output directory.")
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--test-frac", type=float, default=0.1)
    args = parser.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with in_path.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    print(f"Loaded {len(lines)} samples from {in_path}")

    random.shuffle(lines)

    n = len(lines)
    n_val = int(n * args.val_frac)
    n_test = int(n * args.test_frac)
    n_train = n - n_val - n_test

    train_lines = lines[:n_train]
    val_lines = lines[n_train:n_train + n_val]
    test_lines = lines[n_train + n_val:]

    (out_dir / "train.jsonl").write_text("".join(train_lines), encoding="utf-8")
    (out_dir / "val.jsonl").write_text("".join(val_lines), encoding="utf-8")
    (out_dir / "test.jsonl").write_text("".join(test_lines), encoding="utf-8")

    print(f"Train: {n_train}, Val: {n_val}, Test: {n_test}")
    print(f"Wrote files to {out_dir}")


if __name__ == "__main__":
    main()