#!/usr/bin/env python3
"""
Download Lichess monthly PGN dumps and filter games
for player ratings in the 1000–1100 range.

Output:
    pgn/filtered_1000_1100.pgn
"""

import os
import requests
import zstandard as zstd
import chess.pgn
from pathlib import Path

DOWNLOAD_URL = "https://database.lichess.org/standard/lichess_db_standard_rated_{year}-{month}.pgn.zst"

OUT_DIR = Path("pgn")
OUT_DIR.mkdir(exist_ok=True)

def download_month(year: int, month: int) -> Path:
    url = DOWNLOAD_URL.format(year=year, month=f"{month:02d}")
    out_path = OUT_DIR / f"{year}-{month:02d}.pgn.zst"

    if out_path.exists():
        print(f"[+] Already downloaded: {out_path}")
        return out_path

    print(f"[+] Downloading {url}")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

    print(f"[+] Done.")
    return out_path


def decompress_zst(zst_path: Path) -> Path:
    out_path = zst_path.with_suffix("")  # strip .zst → .pgn

    if out_path.exists():
        print(f"[+] Already decompressed: {out_path}")
        return out_path

    print(f"[+] Decompressing {zst_path} ...")
    dctx = zstd.ZstdDecompressor()
    with open(zst_path, "rb") as f_in, open(out_path, "wb") as f_out:
        dctx.copy_stream(f_in, f_out)

    print(f"[+] Done decompressing to {out_path}")
    return out_path


def filter_pgn_for_rating(pgn_path: Path, min_rating=1000, max_rating=1100):
    out_path = OUT_DIR / f"filtered_{min_rating}_{max_rating}.pgn"

    print(f"[+] Filtering {pgn_path} → {out_path}")

    count = 0
    kept = 0

    with open(pgn_path, "r", encoding="utf-8", errors="ignore") as f_in, \
         open(out_path, "a", encoding="utf-8") as f_out:

        while True:
            game = chess.pgn.read_game(f_in)
            if game is None:
                break

            count += 1
            try:
                w = int(game.headers.get("WhiteElo", 0))
                b = int(game.headers.get("BlackElo", 0))
            except ValueError:
                continue

            # Keep if either player is in the range
            if min_rating <= w <= max_rating or min_rating <= b <= max_rating:
                f_out.write(str(game) + "\n\n")
                kept += 1

            if count % 50000 == 0:
                print(f"  Processed {count} games, kept {kept}...")

    print(f"[✓] Done. Total games: {count}, kept: {kept}")
    return out_path


def main():
    year = 2024
    month = 12

    # Step 1: download & decompress monthly dump
    zst_file = download_month(year, month)
    pgn_file = decompress_zst(zst_file)

    # Step 2: filter for Elo range
    filter_pgn_for_rating(pgn_file, 1000, 1100)


if __name__ == "__main__":
    main()
