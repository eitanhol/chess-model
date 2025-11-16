#!/usr/bin/env python3
"""
Evaluate a saved checkpoint on a JSONL dataset (e.g. test split).

Usage:
    python eval_model.py --checkpoint checkpoints/rapid_1000_1100_small.pt \
                         --data-path data/splits_1000_1100/test.jsonl
"""

import argparse

import torch
from torch.utils.data import DataLoader

from model_and_dataset import ChessMoveDataset
from inference import load_model_and_vocab


def evaluate(checkpoint_path: str, data_path: str, batch_size: int = 512):
    model, move2id, id2move, device = load_model_and_vocab(checkpoint_path)

    dataset = ChessMoveDataset(data_path)
    # Make sure dataset uses the same move vocab as the checkpoint
    dataset.move2id = move2id
    dataset.id2move = id2move

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    correct_top1 = 0
    correct_top3 = 0
    total = 0

    with torch.no_grad():
        for boards, ratings, move_ids in loader:
            boards = boards.to(device)
            ratings = ratings.to(device)
            move_ids = move_ids.to(device)

            logits = model(boards, ratings)
            probs = torch.softmax(logits, dim=1)

            # Top-1
            top1 = torch.argmax(probs, dim=1)
            correct_top1 += (top1 == move_ids).sum().item()

            # Top-3
            k = min(3, probs.size(1))
            _, topk_idx = torch.topk(probs, k=k, dim=1)
            correct_top3 += (topk_idx == move_ids.unsqueeze(1)).any(dim=1).sum().item()

            total += move_ids.size(0)

    top1_acc = correct_top1 / total
    top3_acc = correct_top3 / total
    print(f"Top-1 accuracy: {top1_acc:.3f}")
    print(f"Top-3 accuracy: {top3_acc:.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-path", required=True,
                        help="JSONL for evaluation (e.g., a test set).")
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()

    evaluate(args.checkpoint, args.data_path, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
