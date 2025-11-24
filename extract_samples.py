#!/usr/bin/env python3
"""
Minimal PGN → JSONL extractor for your chess move model.

Usage:
    python extract_samples.py input.pgn output.jsonl [min_rating max_rating]
"""

import sys
import json
import re

import chess
import chess.pgn

CLK_RE = re.compile(r'%clk\s+(\d+:\d+:\d+)')


def parse_clock_seconds(comment: str):
    """Return remaining clock time in *seconds* from a PGN comment like '{ [%clk 0:09:32] }'."""
    if not comment:
        return None
    m = CLK_RE.search(comment)
    if not m:
        return None
    h, m_, s = map(int, m.group(1).split(':'))
    return h * 3600 + m_ * 60 + s


def parse_timecontrol(tc: str):
    """
    Parse a TimeControl string like '600+0' into (base_seconds, increment_seconds).
    Returns (None, None) if unknown.
    """
    if not tc or tc == "-":
        return None, None
    if '+' in tc:
        base, inc = tc.split('+', 1)
    else:
        base, inc = tc, '0'
    try:
        return int(base), int(inc)
    except ValueError:
        return None, None


def samples_from_game(
    game,
    min_clock_secs=60,
    enforce_rapid=False,
    rapid_min_base=600,
    min_rating=None,
    max_rating=None,
):
    """
    Given a python-chess Game, extract training samples:

    Each sample is a dict:
        {
            "fen": <board before move>,
            "move_uci": <move in UCI>,
            "rating": <player Elo>,
        }
    """
    # Grab ratings; if missing, skip the game
    try:
        white_elo = int(game.headers.get("WhiteElo"))
        black_elo = int(game.headers.get("BlackElo"))
    except (TypeError, ValueError):
        return []

    # Basic rapid filter (you can tighten this later if you want)
    base, inc = parse_timecontrol(game.headers.get("TimeControl"))
    if enforce_rapid and base is not None and base < rapid_min_base:
        return []

    no_increment = (inc == 0)

    board = game.board()
    samples = []

    last_white_clk = None
    last_black_clk = None
    both_below = False

    for node in game.mainline():
        move = node.move
        comment = node.comment or ""
        clk = parse_clock_seconds(comment)

        # If we don't have clock info for this move, skip it
        if clk is None:
            board.push(move)
            continue

        # Player to move *before* the move is the one whose clock is shown after the move
        to_move_is_white = board.turn

        if to_move_is_white:
            last_white_clk = clk
            rating = white_elo
        else:
            last_black_clk = clk
            rating = black_elo

        # Optional per-move rating band filter
        if (
            min_rating is not None
            and max_rating is not None
            and not (min_rating <= rating <= max_rating)
        ):
            # Still advance the board, but don't create a sample
            board.push(move)
            continue

        # Detect "both players under 60s" in no-increment games
        if no_increment and last_white_clk is not None and last_black_clk is not None:
            if last_white_clk < min_clock_secs and last_black_clk < min_clock_secs:
                both_below = True

        # Apply filters:
        #  - this move's player must have ≥ min_clock_secs remaining
        #  - and in no-increment games, we ignore the rest of the game once both are low
        if clk >= min_clock_secs and not (no_increment and both_below):
            samples.append(
                {
                    "fen": board.fen(),      # position *before* the move
                    "move_uci": move.uci(),  # the move that was actually played
                    "rating": rating,
                }
            )

        # Advance board
        board.push(move)

    return samples


def process_pgn_file(
    input_path,
    output_path,
    min_clock_secs=60,
    enforce_rapid=False,
    rapid_min_base=600,
    min_rating=None,
    max_rating=None,
):
    """Stream through a PGN file and write one JSON object per line to output_path."""
    num_games = 0
    num_samples = 0

    with open(input_path, "r", encoding="utf-8") as f_in, open(
        output_path, "w", encoding="utf-8"
    ) as f_out:
        while True:
            game = chess.pgn.read_game(f_in)
            if game is None:
                break
            num_games += 1
            samples = samples_from_game(
                game,
                min_clock_secs=min_clock_secs,
                enforce_rapid=enforce_rapid,
                rapid_min_base=rapid_min_base,
                min_rating=min_rating,
                max_rating=max_rating,
            )
            for s in samples:
                f_out.write(json.dumps(s) + "\n")
                num_samples += 1

            if num_games % 100 == 0:
                print(
                    f"Processed {num_games} games, {num_samples} samples so far...",
                    file=sys.stderr,
                )

    print(
        f"Done. Games: {num_games}, samples written: {num_samples}",
        file=sys.stderr,
    )


def main(argv=None):
    argv = argv or sys.argv[1:]
    if len(argv) not in (2, 4):
        print(
            "Usage: python extract_samples.py input.pgn output.jsonl [min_rating max_rating]",
            file=sys.stderr,
        )
        raise SystemExit(1)

    input_path, output_path = argv[0], argv[1]

    min_rating = None
    max_rating = None
    if len(argv) == 4:
        try:
            min_rating = int(argv[2])
            max_rating = int(argv[3])
        except ValueError:
            print(
                "min_rating and max_rating must be integers",
                file=sys.stderr,
            )
            raise SystemExit(1)

    process_pgn_file(
        input_path,
        output_path,
        min_clock_secs=60,
        enforce_rapid=False,
        min_rating=min_rating,
        max_rating=max_rating,
    )


if __name__ == "__main__":
    main()
