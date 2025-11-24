#!/usr/bin/env python3
"""
Stream-filter a Lichess .zst dump to get a PGN slice with games where:

  - At least one player has Elo >= MIN_RATING
  - The game is RAPID (based on an approximate time-control heuristic)
  - We only read up to MAX_INPUT_GAMES from the dump to limit runtime

Outputs:
    - A filtered PGN file with only the games we keep.
    - A JSON file with counts of how many kept games fall into each
      100-Elo bucket (1000–1099, 1100–1199, etc.), based on the *higher*
      Elo of the two players in the game.
"""

from __future__ import annotations

from pathlib import Path
from collections import Counter
from typing import Optional
import io
import json

import chess.pgn
import zstandard as zstd


# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------

# Path to your .zst file on the cluster.
ZST_PATH = Path("/home/eholdema/chess-model/chess-pgn/lichess_db_standard_rated_2024-12.pgn.zst")

# Where to write the filtered PGN.
OUT_PGN = Path("/home/eholdema/chess-model/pgn/filtered_rapid_elo1000plus_2024-12_full.pgn")

# Rating range. MAX_RATING = None means "no upper cap" (1000+).
MIN_RATING = 1000
MAX_RATING: Optional[int] = None

# LIMITS:
# We will stop after reading at most this many games from the .zst,
# regardless of how many we keep.
MAX_INPUT_GAMES: Optional[int] = 10_000_000  # <-- your "10 million" cap
# If you ever want to stop after keeping N games, set MAX_KEPT_GAMES instead.
MAX_KEPT_GAMES: Optional[int] = None

# JSON to store per-bucket counts.
BUCKET_COUNTS_JSON = Path("/home/eholdema/chess-model/logs/rating_buckets_elo1000plus.json")


# ------------------------------------------------------------------------------
# Time control helpers
# ------------------------------------------------------------------------------

def estimate_timecontrol_seconds(tc: Optional[str]) -> Optional[int]:
    """
    Estimate the total time for a Lichess TimeControl string.

    Typical formats:
        "600+0"    -> base + increment per move
        "300+2"
        "60"       -> just base
        "0+1"      -> pure increment (we treat base=0)
        "-" / "?"  -> unknown, return None

    We approximate using 40 moves:
        est = base + 40 * increment
    """
    if not tc or tc in ("-", "?", "null"):
        return None

    lower = tc.lower()
    if lower in ("∞", "infinite", "correspondence"):
        return None

    # Some lichess formats: "600+0,600+0" (multiple phases). Take first.
    main_part = tc.split(",")[0]

    if "+" in main_part:
        base_str, inc_str = main_part.split("+", 1)
    else:
        base_str, inc_str = main_part, "0"

    try:
        base = int(base_str)
        inc = int(inc_str)
    except ValueError:
        return None

    approx_moves = 40
    return base + inc * approx_moves


def is_rapid_time_control(tc: Optional[str]) -> bool:
    """
    Decide if the game is RAPID based on estimated total time.

    Approximate lichess thresholds (seconds):
        bullet   < 180
        blitz    180–479
        rapid    480–1499
        classical >=1500

    We treat RAPID as 480 <= est < 1500.
    """
    est = estimate_timecontrol_seconds(tc)
    if est is None:
        return False
    return 480 <= est < 1500


# ------------------------------------------------------------------------------
# Rating-bucket helper
# ------------------------------------------------------------------------------

def get_rating_bucket(
    rating: int,
    bucket_size: int = 100,
    min_rating: int = 1000,
) -> Optional[str]:
    """
    Map a numeric Elo to a bucket label.

    Example (min_rating=1000, bucket_size=100):
        1000–1099 -> "1000-1099"
        1100–1199 -> "1100-1199"
        etc.

    Returns None if rating < min_rating.
    """
    if rating < min_rating:
        return None
    bucket_index = (rating - min_rating) // bucket_size
    start = min_rating + bucket_index * bucket_size
    end = start + bucket_size - 1
    return f"{start}-{end}"


# ------------------------------------------------------------------------------
# Core logic
# ------------------------------------------------------------------------------

def stream_filter_from_zst(
    zst_path: Path,
    out_path: Path,
    min_rating: int,
    max_rating: Optional[int],
    max_input_games: Optional[int],
    max_kept_games: Optional[int],
    bucket_counts_out: Optional[Path],
):
    print(f"[+] Reading: {zst_path}")
    print(f"[+] Writing filtered PGN to: {out_path}")

    if max_rating is None:
        print(f"[+] Rating filter: at least one player Elo >= {min_rating}")
    else:
        print(f"[+] Rating filter: at least one player in [{min_rating}, {max_rating}]")

    if max_input_games is None:
        print("[+] Game read limit: no limit (read entire dump)")
    else:
        print(f"[+] Game read limit: {max_input_games} games from the dump")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    dctx = zstd.ZstdDecompressor()
    bucket_counts = Counter()

    game_count = 0
    kept_count = 0

    with zst_path.open("rb") as fh, out_path.open("w", encoding="utf-8") as out_f:
        with dctx.stream_reader(fh) as reader:
            text_stream = io.TextIOWrapper(reader, encoding="utf-8")

            while True:
                if max_input_games is not None and game_count >= max_input_games:
                    print(f"[!] Reached MAX_INPUT_GAMES = {max_input_games}, stopping.")
                    break

                game = chess.pgn.read_game(text_stream)
                if game is None:
                    print("[+] Reached end of PGN stream.")
                    break

                game_count += 1
                if game_count % 100_000 == 0:
                    print(f"[+] Processed {game_count} games... (kept {kept_count})")

                # Rapid filter
                tc = game.headers.get("TimeControl")
                if not is_rapid_time_control(tc):
                    continue

                # Elo filter
                try:
                    w = int(game.headers.get("WhiteElo", 0))
                    b = int(game.headers.get("BlackElo", 0))
                except ValueError:
                    continue

                if max_rating is None:
                    in_range = (w >= min_rating) or (b >= min_rating)
                else:
                    in_range = (min_rating <= w <= max_rating) or (min_rating <= b <= max_rating)

                if not in_range:
                    continue

                # Bucket count uses *max* Elo in game
                highest = max(w, b)
                bucket_label = get_rating_bucket(highest)
                if bucket_label:
                    bucket_counts[bucket_label] += 1

                kept_count += 1
                print(game, file=out_f, end="\n\n")

                if max_kept_games is not None and kept_count >= max_kept_games:
                    print(f"[!] Reached MAX_KEPT_GAMES = {max_kept_games}, stopping.")
                    break

    print(f"[✓] Done. Total games read: {game_count}, Kept: {kept_count}")

    if bucket_counts_out:
        bucket_counts_out.parent.mkdir(parents=True, exist_ok=True)
        with bucket_counts_out.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "min_rating": min_rating,
                    "max_rating": max_rating,
                    "bucket_size": 100,
                    "total_games_read": game_count,
                    "total_kept": kept_count,
                    "buckets": dict(sorted(bucket_counts.items())),
                },
                f,
                indent=2,
            )
        print(f"[+] Bucket counts saved to {bucket_counts_out}")


# ------------------------------------------------------------------------------
def main():
    stream_filter_from_zst(
        zst_path=ZST_PATH,
        out_path=OUT_PGN,
        min_rating=MIN_RATING,
        max_rating=MAX_RATING,
        max_input_games=MAX_INPUT_GAMES,
        max_kept_games=MAX_KEPT_GAMES,
        bucket_counts_out=BUCKET_COUNTS_JSON,
    )


if __name__ == "__main__":
    main()
