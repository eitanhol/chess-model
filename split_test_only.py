#!/usr/bin/env python3

"""
Create ONLY a test split from a large JSONL file
without creating train or validation files.

Keeps:
  - test.jsonl (test-frac, default 0.2)

The rest of the data is NOT written out (saves disk + time).
"""
#!/usr/bin/env python3

# Create ONLY a test split from a large JSONL file
# without creating train or validation files.

import random
from pathlib import Path
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True,
                        help="Full dataset JSONL")
    parser.add_argument("--out-dir", required=True,
                        help="Directory where test.jsonl will be saved")
    parser.add_argument("--test-frac", type=float, default=0.2,
                        help="Fraction of samples to use for test set")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading lines...")
    with in_path.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    n = len(lines)
    print(f"Loaded {n} samples total")

    random.shuffle(lines)

    n_test = int(n * args.test_frac)
    test_lines = lines[:n_test]

    # Save only the test set
    test_path = out_dir / "test.jsonl"
    with test_path.open("w", encoding="utf-8") as f:
        f.writelines(test_lines)

    print(f"Test samples: {n_test}")
    print("Done.")


if __name__ == "__main__":
    main()

import json
import random
from pathlib import Path
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True,
                        help="Full dataset JSONL")
    parser.add_argument("--out-dir", required=True,
                        help="Directory where test.jsonl will be saved")
    parser.add_argument("--test-frac", type=float, default=0.2,
                        help="Fraction of samples to use for test set")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading lines...")
    with in_path.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    n = len(lines)
    print(f"Loaded {n} samples total")

    random.shuffle(lines)

    n_test = int(n * args.test_frac)
    test_lines = lines[:n_test]

    # Save only the test set
    (out_dir / "test.jsonl").write_text("".join(test_lines), encoding="utf-8")

    print(f"Test samples: {n_test}")
    print("Done - only test.jsonl was created. No training/validation files written.")


if __name__ == "__main__":
    main()
