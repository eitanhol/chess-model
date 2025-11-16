# train.py

import argparse
from model_and_dataset import train_with_val


def main():
    parser = argparse.ArgumentParser(description="Train rating-conditioned chess move model.")
    parser.add_argument("--data-path", required=True, help="Path to JSONL samples.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument(
        "--save-path",
        type=str,
        default="checkpoints/rapid_1000_1100_small.pt",
        help="Where to save best checkpoint.",
    )
    args = parser.parse_args()

    train_with_val(
        jsonl_path=args.data_path,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        val_frac=args.val_frac,
        lr=args.lr,
        save_path=args.save_path,
    )


if __name__ == "__main__":
    main()
