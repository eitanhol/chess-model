#!/usr/bin/env python3

"""
Create ONLY validation and test splits from a large JSONL file
without creating a huge train.jsonl.

Keeps:
  - val.jsonl  (10%)
  - test.jsonl (10%)
"""

import json
import random
from pathlib import Path
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--test-frac", type=float, default=0.1)
    args = parser.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading lines...")
    with in_path.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    print(f"Loaded {len(lines)} samples")

    random.shuffle(lines)

    n = len(lines)
    n_val = int(n * args.val_frac)
    n_test = int(n * args.test_frac)

    val_lines = lines[:n_val]
    test_lines = lines[n_val:n_val + n_test]

    (out_dir / "val.jsonl").write_text("".join(val_lines), encoding="utf-8")
    (out_dir / "test.jsonl").write_text("".join(test_lines), encoding="utf-8")

    print(f"Validation samples: {n_val}")
    print(f"Test samples:       {n_test}")
    print("Done.")


if __name__ == "__main__":
    main()
