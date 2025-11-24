#!/usr/bin/env python3

"""
Evaluate a checkpoint and report accuracy per Elo band (for example, 100 point buckets),
and optionally save results to JSON and CSV.

Adds progress logging so it is easy to monitor in Slurm logs.
"""

import argparse
from collections import defaultdict
import json
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from model_and_dataset import ChessMoveDataset
from inference import load_model_and_vocab


def evaluate_by_elo(
    checkpoint_path,
    data_path,
    batch_size,
    elo_min,
    elo_max,
    band_size,
    out_json=None,
    out_csv=None,
    log_every_samples=50000,
):
    # Load model and vocab
    model, move2id, id2move, device = load_model_and_vocab(checkpoint_path)

    dataset = ChessMoveDataset(data_path)
    dataset.move2id = move2id
    dataset.id2move = id2move

    total_samples = len(dataset)
    print(f"[info] Loaded dataset: {data_path} with {total_samples} samples", flush=True)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model.eval()

    global_top1 = 0
    global_top3 = 0
    global_total = 0

    # band_start -> stats dict
    bands = defaultdict(lambda: {"top1": 0, "top3": 0, "total": 0})

    processed = 0
    next_log = log_every_samples if log_every_samples and log_every_samples > 0 else None

    with torch.no_grad():
        for batch_idx, (boards, ratings, move_ids) in enumerate(loader):
            boards = boards.to(device)
            ratings = ratings.to(device)
            move_ids = move_ids.to(device)

            logits = model(boards, ratings)
            probs = torch.softmax(logits, dim=1)

            top1 = torch.argmax(probs, dim=1)
            k = min(3, probs.size(1))
            _, topk = torch.topk(probs, k=k, dim=1)

            batch_n = move_ids.size(0)

            # Global stats
            match1 = (top1 == move_ids).sum().item()
            match3 = (topk == move_ids.unsqueeze(1)).any(dim=1).sum().item()

            global_top1 += match1
            global_top3 += match3
            global_total += batch_n

            # Per sample for Elo bands
            for r, p1, p3_list, tgt in zip(
                ratings.cpu().tolist(),
                top1.cpu().tolist(),
                topk.cpu().tolist(),
                move_ids.cpu().tolist(),
            ):
                r_int = int(round(r))
                if r_int < elo_min or r_int > elo_max:
                    continue

                band_start = (r_int // band_size) * band_size
                stats = bands[band_start]
                stats["total"] += 1
                if p1 == tgt:
                    stats["top1"] += 1
                if tgt in p3_list:
                    stats["top3"] += 1

            # Progress logging
            processed += batch_n
            if next_log is not None and processed >= next_log:
                pct = 100.0 * processed / total_samples
                print(
                    f"[progress] Processed {processed}/{total_samples} samples "
                    f"({pct:.2f} percent)",
                    flush=True,
                )
                next_log += log_every_samples

    # Print global results
    print("\nGLOBAL ACCURACY", flush=True)
    print("----------------", flush=True)

    if global_total == 0:
        print("No samples found in the given Elo range.", flush=True)
    else:
        global_top1_acc = global_top1 / global_total
        global_top3_acc = global_top3 / global_total
        print(f"Total samples: {global_total}", flush=True)
        print(f"Top-1: {global_top1_acc:.4f}", flush=True)
        print(f"Top-3: {global_top3_acc:.4f}", flush=True)

    # Build per band results
    results = []
    for band_start in sorted(bands.keys()):
        stats = bands[band_start]
        if stats["total"] == 0:
            continue
        band_end = band_start + band_size - 1
        total = stats["total"]
        top1_acc = stats["top1"] / total
        top3_acc = stats["top3"] / total

        results.append(
            {
                "elo_start": band_start,
                "elo_end": band_end,
                "total": total,
                "correct_top1": stats["top1"],
                "correct_top3": stats["top3"],
                "top1_acc": top1_acc,
                "top3_acc": top3_acc,
            }
        )

    # Print per band summary
    print("\nPER-ELO BUCKET ACCURACY", flush=True)
    print("------------------------", flush=True)
    for row in results:
        print(
            f"{row['elo_start']:4d}-{row['elo_end']:4d}: "
            f"n={row['total']:7d} | "
            f"top1={row['top1_acc']:.4f} | "
            f"top3={row['top3_acc']:.4f}",
            flush=True,
        )

    # Save JSON
    if out_json is not None:
        out_path = Path(out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "checkpoint": str(checkpoint_path),
            "data_path": str(data_path),
            "elo_min": elo_min,
            "elo_max": elo_max,
            "band_size": band_size,
            "global": {
                "total": global_total,
                "correct_top1": global_top1,
                "correct_top3": global_top3,
                "top1_acc": global_top1 / global_total if global_total else None,
                "top3_acc": global_top3 / global_total if global_total else None,
            },
            "bands": results,
        }
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"[info] Saved JSON results to {out_path}", flush=True)

    # Save CSV
    if out_csv is not None:
        out_path = Path(out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "elo_start",
                    "elo_end",
                    "total",
                    "correct_top1",
                    "correct_top3",
                    "top1_acc",
                    "top3_acc",
                ],
            )
            writer.writeheader()
            for row in results:
                writer.writerow(row)
        print(f"[info] Saved CSV results to {out_path}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True,
                        help="Path to model checkpoint (.pt)")
    parser.add_argument("--data-path", required=True,
                        help="JSONL dataset to evaluate on")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--elo-min", type=int, default=1000)
    parser.add_argument("--elo-max", type=int, default=3200)
    parser.add_argument("--band-size", type=int, default=100)
    parser.add_argument("--out-json", type=str, default=None,
                        help="Path to save detailed results as JSON")
    parser.add_argument("--out-csv", type=str, default=None,
                        help="Path to save per-band results as CSV")
    parser.add_argument(
        "--log-every",
        type=int,
        default=50000,
        help="Log progress every N samples (0 disables periodic logs).",
    )
    args = parser.parse_args()

    evaluate_by_elo(
        checkpoint_path=args.checkpoint,
        data_path=args.data_path,
        batch_size=args.batch_size,
        elo_min=args.elo_min,
        elo_max=args.elo_max,
        band_size=args.band_size,
        out_json=args.out_json,
        out_csv=args.out_csv,
        log_every_samples=args.log_every,
    )


if __name__ == "__main__":
    main()
