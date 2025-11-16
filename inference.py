# inference.py

import torch
import chess

from model_and_dataset import SimpleChessNet, encode_board


def load_model_and_vocab(checkpoint_path: str, device: str | None = None):
    """
    Load SimpleChessNet and move vocab from a checkpoint created by train_with_val().
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint = torch.load(checkpoint_path, map_location=device)

    num_moves = checkpoint["num_moves"]
    move2id = checkpoint["move2id"]
    id2move = checkpoint["id2move"]

    model = SimpleChessNet(num_moves).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, move2id, id2move, device


def predict_top_k(
    model,
    id2move,
    fen: str,
    rating: int,
    k: int = 5,
    device: str | torch.device = "cpu",
):
    """
    Given a model + vocab, FEN, and rating, return top-k predicted moves.

    Returns:
        List[{"uci": str, "san": str, "prob": float}]
    """
    # Board tensor
    board_np = encode_board(fen)  # (14, 8, 8) np.float32
    board_tensor = torch.from_numpy(board_np).unsqueeze(0).to(device)  # (1, 14, 8, 8)

    # Rating should be 1D: shape (B,) = (1,)
    rating_tensor = torch.tensor([float(rating)], dtype=torch.float32, device=device)

    with torch.no_grad():
        logits = model(board_tensor, rating_tensor)  # (1, num_moves)
        probs = torch.softmax(logits, dim=-1).squeeze(0)  # (num_moves,)

    topk = torch.topk(probs, k)
    indices = topk.indices.cpu().tolist()
    values = topk.values.cpu().tolist()

    board = chess.Board(fen)
    results = []
    for idx, p in zip(indices, values):
        uci = id2move[idx]
        move = chess.Move.from_uci(uci)
        san = board.san(move)
        results.append(
            {
                "uci": uci,
                "san": san,
                "prob": float(p),
            }
        )

    return results

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--fen", default=chess.STARTING_FEN)
    parser.add_argument("--rating", type=int, default=1050)
    parser.add_argument("-k", type=int, default=5)
    args = parser.parse_args()

    model, move2id, id2move, device = load_model_and_vocab(args.checkpoint)
    preds = predict_top_k(model, id2move, args.fen, args.rating, k=args.k, device=device)

    print(f"Position FEN: {args.fen}")
    print(f"Rating: {args.rating}")
    print()
    for i, p in enumerate(preds, start=1):
        print(f"{i}. {p['san']:>6} ({p['uci']})   p={p['prob']:.3f}")
