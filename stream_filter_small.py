#!/usr/bin/env python3
"""
Stream-filter a Lichess .zst dump to get a SMALL PGN slice
with games where:

  - At least one player is in [MIN_RATING, MAX_RATING]
  - The game is RAPID (by Lichess-style time classification)

We do this directly from the .zst file (no huge .pgn on disk).

Output:
    pgn/filtered_rapid_1000_1100_small.pgn
"""

from pathlib import Path
import io

import zstandard as zstd
import chess.pgn


# ------------- CONFIG (tweak here) -----------------

ZST_PATH = Path("pgn/lichess_db_standard_rated_2024-12.pgn.zst")

OUT_PGN = Path("pgn/filtered_rapid_1000_1100_double.pgn")

MIN_RATING = 1000
MAX_RATING = 1100

# Read at most this many games from the dump
MAX_INPUT_GAMES = 400_000      # feel free to bump later

# Keep at most this many matching games
MAX_KEPT_GAMES = 10_000         # what you asked for

# --------------------------------------------------


def estimate_timecontrol_seconds(tc: str | None) -> int | None:
    """
    Approximate total game time in seconds from a Lichess 'TimeControl' string.

    Lichess classifies time controls using:
        estimated_time = initial + 40 * increment

    We'll implement a simple version of that:

        - '600+0' -> 600 + 40*0  = 600
        - '180+2' -> 180 + 40*2 = 260
        - '300'   -> 300 + 40*0 = 300

    If parsing fails, return None.
    """
    if not tc or tc == "-":
        return None

    # Some rare forms may exist, but most are "base+inc" or "base"
    if "+" in tc:
        base_str, inc_str = tc.split("+", 1)
    else:
        base_str, inc_str = tc, "0"

    try:
        base = int(base_str)
        inc = int(inc_str)
    except ValueError:
        return None

    return base + 40 * inc


def is_rapid_time_control(tc: str | None) -> bool:
    """
    Decide whether a game is RAPID using the estimated time.

    Lichess thresholds (seconds) are roughly:
        bullet   < 180
        blitz    180–479
        rapid    480–1499
        classical >=1500

    We'll treat RAPID as 480 <= estimated < 1500.
    """
    est = estimate_timecontrol_seconds(tc)
    if est is None:
        return False
    return 480 <= est < 1500


def stream_filter_from_zst(
    zst_path: Path,
    out_path: Path,
    min_rating: int,
    max_rating: int,
    max_input_games: int,
    max_kept_games: int,
):
    if not zst_path.exists():
        raise FileNotFoundError(f"ZST file not found: {zst_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[+] Opening compressed file: {zst_path}")
    print(f"[+] Writing filtered games to: {out_path}")
    print(f"[+] Rating range: {min_rating}–{max_rating}")
    print(f"[+] Rapid only (based on TimeControl header).")
    print(f"[+] Will read up to {max_input_games} games, keep up to {max_kept_games} games.")

    dctx = zstd.ZstdDecompressor()

    game_counter = 0
    kept_counter = 0

    with zst_path.open("rb") as f_in, dctx.stream_reader(f_in) as reader:
        text_stream = io.TextIOWrapper(reader, encoding="utf-8", errors="ignore")

        with out_path.open("w", encoding="utf-8") as f_out:
            while True:
                game = chess.pgn.read_game(text_stream)
                if game is None:
                    break  # end of stream

                game_counter += 1
                if game_counter % 50_000 == 0:
                    print(f"  Processed {game_counter} games, kept {kept_counter} so far...")

                # Stop if we've read enough games
                if game_counter > max_input_games:
                    print("[!] Reached MAX_INPUT_GAMES limit; stopping.")
                    break

                # --- RAPID filter ---
                tc = game.headers.get("TimeControl")
                if not is_rapid_time_control(tc):
                    continue

                # --- Rating filter ---
                try:
                    w_elo = int(game.headers.get("WhiteElo", 0))
                    b_elo = int(game.headers.get("BlackElo", 0))
                except ValueError:
                    continue

                if not (
                    (min_rating <= w_elo <= max_rating)
                    or (min_rating <= b_elo <= max_rating)
                ):
                    continue

                # If we get here, game passes both filters
                f_out.write(str(game) + "\n\n")
                kept_counter += 1

                if kept_counter >= max_kept_games:
                    print("[!] Reached MAX_KEPT_GAMES limit; stopping.")
                    break

    print(
        f"[✓] Done. Processed {game_counter} total games, "
        f"kept {kept_counter} rapid games in rating range."
    )


def main():
    stream_filter_from_zst(
        zst_path=ZST_PATH,
        out_path=OUT_PGN,
        min_rating=MIN_RATING,
        max_rating=MAX_RATING,
        max_input_games=MAX_INPUT_GAMES,
        max_kept_games=MAX_KEPT_GAMES,
    )


if __name__ == "__main__":
    main()
