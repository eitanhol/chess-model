"""
Minimal PyTorch dataset + model for your chess move predictor.

Assumes JSONL from extract_samples.py:
    {"fen": "...", "move_uci": "e2e4", "rating": 1830}
"""

import json
from pathlib import Path
from typing import List, Dict

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

import chess
from torch.utils.data import random_split
import os

PIECE_TYPES = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]


def encode_board(fen: str) -> np.ndarray:
    """
    Encode a board FEN as a (C, 8, 8) float32 array.

    Channels:
      0-5:  white pawn/knight/bishop/rook/queen/king
      6-11: black pawn/knight/bishop/rook/queen/king
      12:   side to move (all ones if white to move, zeros if black)
      13:   simple "any castling rights remaining" plane
    """
    board = chess.Board(fen)
    planes = np.zeros((14, 8, 8), dtype=np.float32)

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is None:
            continue
        pt = piece.piece_type
        color = piece.color  # True for white, False for black

        pt_index = PIECE_TYPES.index(pt)
        channel = pt_index if color == chess.WHITE else 6 + pt_index

        rank = chess.square_rank(square)
        file = chess.square_file(square)

        # Put white at bottom of plane
        planes[channel, 7 - rank, file] = 1.0

    # Side-to-move plane
    if board.turn == chess.WHITE:
        planes[12, :, :] = 1.0

    # Any castling rights plane
    if board.castling_rights:
        planes[13, :, :] = 1.0

    return planes


class ChessMoveDataset(Dataset):
    def __init__(self, jsonl_path: str, move_dict: Dict[str, int] = None):
        self.jsonl_path = Path(jsonl_path)
        self.samples: List[Dict] = []

        with self.jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                self.samples.append(json.loads(line))

        # Build or reuse move dictionary (move_uci -> class index)
        if move_dict is None:
            unique_moves = sorted({s["move_uci"] for s in self.samples})
            self.move2id = {m: i for i, m in enumerate(unique_moves)}
        else:
            self.move2id = move_dict

        self.id2move = {i: m for m, i in self.move2id.items()}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        board_tensor = encode_board(s["fen"])  # (C,8,8)
        move_id = self.move2id[s["move_uci"]]
        rating = float(s["rating"])

        board_tensor = torch.from_numpy(board_tensor)           # float32
        move_id = torch.tensor(move_id, dtype=torch.long)
        rating = torch.tensor(rating, dtype=torch.float32)

        return board_tensor, rating, move_id


class SimpleChessNet(nn.Module):
    """
    Very small CNN + MLP that takes:
      - board tensor (B, 14, 8, 8)
      - rating (B,)
    and outputs logits over |V| possible moves.
    """

    def __init__(self, num_moves: int, board_channels: int = 14):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(board_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        conv_output_size = 64 * 8 * 8  # 64 channels, 8x8 board

        self.fc = nn.Sequential(
            nn.Linear(conv_output_size + 1, 256),
            nn.ReLU(),
            nn.Linear(256, num_moves),
        )

        # No softmax here: CrossEntropyLoss expects raw logits

    def forward(self, boards: torch.Tensor, ratings: torch.Tensor) -> torch.Tensor:
        """
        boards:  (B, 14, 8, 8)
        ratings: (B,) scalar Elo
        """
        x = self.conv(boards)  # (B, conv_output_size)

        # Scale rating to ~[0, 1] range
        r = (ratings / 3000.0).unsqueeze(1)
        x = torch.cat([x, r], dim=1)

        logits = self.fc(x)  # (B, num_moves)
        return logits


def tiny_train_example(jsonl_path: str, batch_size: int = 64, num_epochs: int = 2):
    """
    Quick sanity-check training loop on a small JSONL file.
    Just to prove the pipeline works on your laptop before going to HPCC.
    """
    dataset = ChessMoveDataset(jsonl_path)
    num_moves = len(dataset.move2id)
    print(f"Loaded {len(dataset)} samples, {num_moves} unique moves.")

    model = SimpleChessNet(num_moves)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for boards, ratings, move_ids in dataloader:
            boards = boards.to(device)
            ratings = ratings.to(device)
            move_ids = move_ids.to(device)

            optimizer.zero_grad()
            logits = model(boards, ratings)  # (B, num_moves)
            loss = criterion(logits, move_ids)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * boards.size(0)

        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch+1}/{num_epochs} - loss: {avg_loss:.4f}")

    return model, dataset

def train_with_val(
    jsonl_path: str,
    batch_size: int = 64,
    num_epochs: int = 5,
    val_frac: float = 0.2,
    lr: float = 1e-3,
    save_path: str | None = None,
):
    """
    Train SimpleChessNet with a train/val split, report top-1/top-3,
    and optionally save the best checkpoint (by val top-1 accuracy).

    Checkpoint includes:
      - model_state_dict
      - move2id / id2move
      - num_moves
    """
    dataset = ChessMoveDataset(jsonl_path)
    num_moves = len(dataset.move2id)
    print(f"Loaded {len(dataset)} samples, {num_moves} unique moves.")

    # Train/val split
    val_size = max(1, int(len(dataset) * val_frac))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    print(f"Train size: {train_size}, Val size: {val_size}")

    model = SimpleChessNet(num_moves)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_top1 = 0.0
    best_checkpoint = None

    for epoch in range(num_epochs):
        # ---- train ----
        model.train()
        total_loss = 0.0
        for boards, ratings, move_ids in train_loader:
            boards = boards.to(device)
            ratings = ratings.to(device)
            move_ids = move_ids.to(device)

            optimizer.zero_grad()
            logits = model(boards, ratings)
            loss = criterion(logits, move_ids)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * boards.size(0)

        avg_train_loss = total_loss / train_size

        # ---- validate ----
        model.eval()
        correct_top1 = 0
        correct_top3 = 0
        total = 0

        with torch.no_grad():
            for boards, ratings, move_ids in val_loader:
                boards = boards.to(device)
                ratings = ratings.to(device)
                move_ids = move_ids.to(device)

                logits = model(boards, ratings)        # (B, num_moves)
                probs = torch.softmax(logits, dim=1)   # (B, num_moves)

                # Top-1
                top1 = torch.argmax(probs, dim=1)
                correct_top1 += (top1 == move_ids).sum().item()

                # Top-3
                k = min(3, num_moves)
                _, topk_idx = torch.topk(probs, k=k, dim=1)
                correct_top3 += (topk_idx == move_ids.unsqueeze(1)).any(dim=1).sum().item()

                total += move_ids.size(0)

        top1_acc = correct_top1 / total
        top3_acc = correct_top3 / total

        print(
            f"Epoch {epoch+1}/{num_epochs} - "
            f"train loss: {avg_train_loss:.4f}, "
            f"val top1: {top1_acc:.3f}, val top3: {top3_acc:.3f}"
        )

        # Track best checkpoint
        if top1_acc > best_val_top1:
            best_val_top1 = top1_acc
            best_checkpoint = {
                "model_state_dict": model.state_dict(),
                "move2id": dataset.move2id,
                "id2move": dataset.id2move,
                "num_moves": num_moves,
            }

    # Save best model if requested
    if save_path is not None and best_checkpoint is not None:
        ckpt_dir = os.path.dirname(save_path)
        if ckpt_dir:
            os.makedirs(ckpt_dir, exist_ok=True)
        torch.save(best_checkpoint, save_path)
        print(f"[✓] Saved best checkpoint (val top1={best_val_top1:.3f}) to {save_path}")

    return model, (train_ds, val_ds)

if __name__ == "__main__":
    # Quick manual test
    train_with_val(
        "data/rapid_1000_1100_small.jsonl",
        num_epochs=3,
        save_path="checkpoints/debug_1000_1100.pt",
    )