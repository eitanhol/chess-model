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
    Given a model + vocab, FEN, and rating, return top-k *legal* predicted moves.

    Returns:
        List[{"uci": str, "san": str, "prob": float}]
    """
    # 1. Prepare Board Data
    board_np = encode_board(fen)  # (14, 8, 8) np.float32
    board_tensor = torch.from_numpy(board_np).unsqueeze(0).to(device)  # (1, 14, 8, 8)

    # Rating should be 1D: shape (B,) = (1,)
    rating_tensor = torch.tensor([float(rating)], dtype=torch.float32, device=device)

    # 2. Run Inference
    with torch.no_grad():
        logits = model(board_tensor, rating_tensor)  # (1, num_moves)
        probs = torch.softmax(logits, dim=-1).squeeze(0)  # (num_moves,)

    # 3. Filter for Legal Moves
    # Instead of just taking topk (which might be illegal), we sort all probs
    # and iterate until we find k *legal* moves.
    
    # Sort indices by probability descending
    sorted_indices = torch.argsort(probs, descending=True)
    
    board = chess.Board(fen)
    legal_moves = set(board.legal_moves) # fast lookup
    results = []

    # Iterate through predictions (highest prob first)
    for idx_tensor in sorted_indices:
        idx = idx_tensor.item()
        
        # Decode move
        if idx not in id2move:
            continue
            
        uci = id2move[idx]
        move = chess.Move.from_uci(uci)

        # CHECK LEGALITY BEFORE ADDING
        if move in legal_moves:
            san = board.san(move) # This is safe now because we checked legality
            results.append({
                "uci": uci,
                "san": san,
                "prob": float(probs[idx]),
            })

        # Stop once we have found 'k' legal moves
        if len(results) >= k:
            break

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
    
    print(f"Position FEN: {args.fen}")
    print(f"Rating: {args.rating}")
    
    try:
        preds = predict_top_k(model, id2move, args.fen, args.rating, k=args.k, device=device)
        print()
        if not preds:
            print("No legal moves found (Checkmate or Stalemate?)")
        for i, p in enumerate(preds, start=1):
            print(f"{i}. {p['san']:>6} ({p['uci']})   p={p['prob']:.3f}")
    except Exception as e:
        print(f"\nError: {e}")