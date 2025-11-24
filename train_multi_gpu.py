# train_multi_gpu.py
"""
Multi-GPU training for the rating-conditioned chess move model
using torch.nn.DataParallel (single process, N GPUs).

Does NOT modify existing code. Uses ChessMoveDataset and SimpleChessNet
from model_and_dataset.py and keeps the same CLI as train.py.
"""

import argparse
import os
from typing import Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

from model_and_dataset import ChessMoveDataset, SimpleChessNet  # :contentReference[oaicite:0]{index=0}


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch_idx: int,
    log_every: int = 1000,
) -> float:
    model.train()
    total_loss = 0.0
    num_samples = 0

    for batch_idx, (boards, ratings, move_ids) in enumerate(loader):
        boards = boards.to(device, non_blocking=True)
        ratings = ratings.to(device, non_blocking=True)
        move_ids = move_ids.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(boards, ratings)
        loss = criterion(logits, move_ids)
        loss.backward()
        optimizer.step()

        batch_size = boards.size(0)
        total_loss += loss.item() * batch_size
        num_samples += batch_size

        if log_every and (batch_idx % log_every == 0):
            print(
                f"[Epoch {epoch_idx}] "
                f"Batch {batch_idx}/{len(loader)} - "
                f"loss: {loss.item():.4f}",
                flush=True,
            )

    avg_loss = total_loss / max(1, num_samples)
    return avg_loss


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_moves: int,
) -> Tuple[float, float]:
    model.eval()
    correct_top1 = 0
    correct_top3 = 0
    total = 0

    for boards, ratings, move_ids in loader:
        boards = boards.to(device, non_blocking=True)
        ratings = ratings.to(device, non_blocking=True)
        move_ids = move_ids.to(device, non_blocking=True)

        logits = model(boards, ratings)              # (B, num_moves)
        probs = torch.softmax(logits, dim=1)

        # Top-1
        top1 = torch.argmax(probs, dim=1)
        correct_top1 += (top1 == move_ids).sum().item()

        # Top-3
        k = min(3, num_moves)
        _, topk_idx = torch.topk(probs, k=k, dim=1)
        correct_top3 += (
            topk_idx == move_ids.unsqueeze(1)
        ).any(dim=1).sum().item()

        total += move_ids.size(0)

    top1_acc = correct_top1 / max(1, total)
    top3_acc = correct_top3 / max(1, total)
    return top1_acc, top3_acc


def main():
    parser = argparse.ArgumentParser(
        description="Multi-GPU training for rating-conditioned chess move model."
    )
    parser.add_argument("--data-path", required=True, help="Path to JSONL samples.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument(
        "--save-path",
        type=str,
        default="checkpoints/rapid_elo1000plus_multi4.pt",
        help="Where to save best checkpoint.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader workers per process.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=1000,
        help="Print training loss every N batches.",
    )
    args = parser.parse_args()

    # ----- Dataset -----
    dataset = ChessMoveDataset(args.data_path)  # loads JSONL once  :contentReference[oaicite:1]{index=1}
    num_moves = len(dataset.move2id)
    print(f"Loaded {len(dataset)} samples, {num_moves} unique moves.")

    val_size = max(1, int(len(dataset) * args.val_frac))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    print(f"Train size: {train_size}, Val size: {val_size}")

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    # ----- Model + multi-GPU -----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleChessNet(num_moves)  # same model as single-GPU  :contentReference[oaicite:2]{index=2}

    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            print(f"[Multi-GPU] Using {num_gpus} GPUs with DataParallel", flush=True)
            model = nn.DataParallel(model)
        else:
            print("[Single-GPU] Using 1 GPU", flush=True)
    else:
        print("[CPU] CUDA not available; running on CPU", flush=True)

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_top1 = 0.0
    best_state_dict = None

    # ----- Training loop -----
    for epoch in range(1, args.epochs + 1):
        avg_train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            epoch_idx=epoch,
            log_every=args.log_every,
        )

        top1, top3 = evaluate(model, val_loader, device, num_moves)

        print(
            f"Epoch {epoch}/{args.epochs} - "
            f"train loss: {avg_train_loss:.4f}, "
            f"val top1: {top1:.3f}, val top3: {top3:.3f}",
            flush=True,
        )

        if top1 > best_top1:
            best_top1 = top1
            # If wrapped in DataParallel, save the underlying module’s weights
            state_dict = (
                model.module.state_dict()
                if isinstance(model, nn.DataParallel)
                else model.state_dict()
            )
            best_state_dict = {
                "model_state_dict": state_dict,
                "move2id": dataset.move2id,
                "id2move": dataset.id2move,
                "num_moves": num_moves,
            }

    if args.save_path is not None and best_state_dict is not None:
        ckpt_dir = os.path.dirname(args.save_path)
        if ckpt_dir:
            os.makedirs(ckpt_dir, exist_ok=True)
        torch.save(best_state_dict, args.save_path)
        print(
            f"[?] Saved best checkpoint (val top1={best_top1:.3f}) "
            f"to {args.save_path}",
            flush=True,
        )


if __name__ == "__main__":
    main()
