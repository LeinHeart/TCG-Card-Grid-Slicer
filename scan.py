r"""Scan grid sheets of TCG cards, cut every card out, save one PNG per card.

    put every scanned sheet into  scan_grids\
    double-click scan.bat  (or: python scan.py)

Every sheet becomes one grid:
    scan_grids\sheet1.png  ->  cards\Card_grid1_card1.png ... Card_grid1_card6.png
    scan_grids\sheet2.png  ->  cards\Card_grid2_card1.png ... Card_grid2_card6.png
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np

from cardscan import (
    GridWriter,
    crop_card,
    detect_cards,
    grid_quads,
    order_grid,
    pad_card,
    trim_paper_edges,
)
from cardscan.detect import CARD_RATIO, CARD_W_MM, Detection, best_layout
from cardscan.naming import _imread as read_image
from cardscan.naming import grid_hint

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

DEFAULT_IN = "scan_grids"
DEFAULT_OUT = "cards"
PREFIX = "Card_"

BANNER = r"""
 _         _       ____  _
| |    ___(_)_ __ |  _ \(_)_ __
| |   / _ \ | '_ \| |_) | | '_ \
| |__|  __/ | | | |  _ <| | | | |
|_____\___|_|_| |_|_| \_\_|_| |_|
"""


def show_banner() -> None:
    print(BANNER)
    print("  TCG grid sheet splitter  |  6 cards per sheet  |  scan_grids -> cards")
    print()


# --------------------------------------------------------------------------- util


def parse_size(text: str | None) -> tuple[int, int] | None:
    if not text:
        return None
    try:
        w, h = text.lower().split("x")
        w, h = int(w), int(h)
    except ValueError:
        raise SystemExit(f"bad --size '{text}', expected WxH like 750x1050")
    if w < 50 or h < 50:
        raise SystemExit("--size too small, use at least 50x50")
    return w, h


def collect_inputs(target: Path, recursive: bool) -> list[Path]:
    if target.is_file():
        return [target]
    if target.is_dir():
        it = target.rglob("*") if recursive else target.glob("*")
        return sorted(
            (p for p in it if p.is_file() and p.suffix.lower() in IMAGE_EXTS),
            key=lambda p: p.name.lower(),
        )
    raise SystemExit(f"input not found: {target}")


def annotate(img: np.ndarray, dets: list[Detection], grid: int) -> np.ndarray:
    vis = img.copy()
    for i, d in enumerate(dets, start=1):
        pts = d.points.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(vis, [pts], True, (0, 255, 0), 3)
        cx, cy = (int(v) for v in d.center)
        cv2.putText(vis, f"{grid}.{i}", (cx - 20, cy + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 255), 4)
    return vis


def save_png(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".png", img, [cv2.IMWRITE_PNG_COMPRESSION, 6])
    if not ok:
        raise OSError(f"encode failed: {path}")
    buf.tofile(str(path))


def resolve_margin(args: argparse.Namespace, card_w: float) -> int:
    """Card margin in pixels, from --margin-cm / --margin-mm / --margin-px."""
    if args.margin_px is not None:
        return int(args.margin_px)
    mm = args.margin_mm if args.margin_mm is not None else args.margin_cm * 10.0
    return max(0, int(round(mm * card_w / CARD_W_MM)))


# --------------------------------------------------------------------------- core


def process_frame(
    img: np.ndarray,
    args: argparse.Namespace,
    writer: GridWriter,
    tag: str = "sheet",
    hint: int | None = None,
) -> int:
    """Detect, crop, pad, save one sheet. Returns the number of cards written."""
    dets, diag = detect_cards(img, expect=args.expect, cols=args.cols)
    layout = f"{diag['cols']}x{diag['rows']}" if diag.get("cols") else "?"
    print(f"  {diag['kept']} card(s) via {diag['method']} ({layout}) "
          f"- {diag['unique']} unique / {diag['candidates']} raw candidates")

    cols = args.cols or diag.get("cols") or 3

    if len(dets) != args.expect:
        if not args.fallback:
            print(f"  ! expected {args.expect} cards, detected {len(dets)} - saving what was found")
        else:
            fcols, frows = ((args.cols, args.rows) if (args.cols and args.rows)
                            else best_layout(img.shape, args.expect))
            print(f"  ! expected {args.expect} cards, detected {len(dets)} - "
                  f"falling back to a {fcols}x{frows} grid split")
            dets = [
                Detection(points=q, score=0.0, area=float(cv2.contourArea(q)), source="grid")
                for q in grid_quads(img.shape, cols=fcols, rows=frows, inset=args.grid_inset)
            ]
            cols = fcols

    ordered = order_grid(dets, cols=cols)
    if not ordered:
        return 0

    # One output size for the whole sheet: median measured height, true card ratio.
    card_size = args.size
    if card_size is None and args.uniform:
        heights = []
        for d in ordered:
            w, h = d.size
            heights.append(h if h >= w else w / CARD_RATIO)
        out_h = int(np.clip(np.median(heights), 400, 2000))
        card_size = (max(1, int(round(out_h * CARD_RATIO))), out_h)

    grid = writer.begin(hint)
    if args.debug_dir:
        save_png(Path(args.debug_dir) / f"{tag}__debug.png", annotate(img, dets, grid))

    written = 0
    for card_no, det in enumerate(ordered, start=1):
        card = crop_card(img, det.points, size=card_size, expand=args.expand)
        card = trim_paper_edges(card)
        if card_size is not None and (card.shape[1], card.shape[0]) != card_size:
            card = cv2.resize(card, card_size, interpolation=cv2.INTER_AREA)

        pad = resolve_margin(args, card.shape[1])
        hash_img = card
        if pad:
            card = pad_card(card, pad, pad, transparent=args.margin_transparent)

        path, duplicate = writer.save(card, card_no, hash_img=hash_img)
        if duplicate:
            print(f"  = card {card_no} matches an already saved card, skipped")
            continue
        assert path is not None
        note = f"  [card {card.shape[1] - 2 * pad}x{card.shape[0] - 2 * pad} + {pad}px margin]" if pad else ""
        print(f"  + {path.name}  {card.shape[1]}x{card.shape[0]}{note}  <- {det.source or 'detect'}")
        written += 1

    if written == 0:
        writer.abort()  # nothing saved: give the grid number back
    return written


def run_files(args: argparse.Namespace, writer: GridWriter) -> int:
    targets = collect_inputs(Path(args.input), args.recursive)
    if not targets:
        print(f"Nothing to scan: no images in {Path(args.input).resolve()}")
        print("Put your scanned sheets there and run again.")
        return 2

    print(f"{len(targets)} sheet(s) to scan\n")
    total = 0
    failed = 0
    moved = 0
    for i, path in enumerate(targets, 1):
        print(f"[{i}/{len(targets)}] {path.name}")
        img = read_image(path)
        if img is None:
            print("  ! could not read image")
            failed += 1
            print()
            continue

        total += process_frame(img, args, writer, tag=path.stem, hint=grid_hint(path))

        if args.move_done:
            dest_dir = Path(args.move_done)
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / path.name
            k = 1
            while dest.exists():
                dest = dest_dir / f"{path.stem}_{k}{path.suffix}"
                k += 1
            try:
                shutil.move(str(path), str(dest))
                moved += 1
            except OSError as exc:
                print(f"  ! could not move {path.name}: {exc}")
        print()

    print(f"Done. {total} card image(s) saved, {failed} sheet(s) failed.", end="")
    print(f" {moved} sheet(s) moved to {args.move_done}." if moved else "")
    return 0 if failed == 0 else 1


# ---------------------------------------------------------------------------- cli


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="scan.py",
        description="Cut a grid sheet of TCG cards into one PNG per card.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("input", nargs="?", default=DEFAULT_IN,
                   help="folder of scanned sheets, or a single image")
    p.add_argument("-o", "--out", default=DEFAULT_OUT, help="output folder")
    p.add_argument("-r", "--recursive", action="store_true", help="scan folders recursively")

    p.add_argument("--expect", type=int, default=6, help="cards per sheet")
    p.add_argument("--cols", type=int, default=None, help="force grid columns (auto by default)")
    p.add_argument("--rows", type=int, default=None, help="force grid rows (auto by default)")
    p.add_argument("--no-fallback", dest="fallback", action="store_false",
                   help="do not grid-split when detection finds the wrong count")
    p.add_argument("--grid-inset", type=float, default=0.02,
                   help="fraction trimmed from each grid cell in fallback mode")

    p.add_argument("--margin-cm", type=float, default=0.5,
                   help="blank margin around each card, in cm")
    p.add_argument("--margin-mm", type=float, default=None,
                   help="blank margin in mm (overrides --margin-cm)")
    p.add_argument("--margin-px", type=int, default=None,
                   help="blank margin in pixels (overrides both)")
    p.add_argument("--margin-transparent", action="store_true",
                   help="transparent margin (RGBA) instead of white")
    p.add_argument("--expand", type=float, default=0.01,
                   help="grow the sample box so a small detection cannot clip the card")
    p.add_argument("--size", default=None, help="force card size (margin excluded), e.g. 750x1050")
    p.add_argument("--no-uniform", dest="uniform", action="store_false",
                   help="keep each card's own measured resolution")

    p.add_argument("--start-grid", type=int, default=None,
                   help="force the grid number of the next sheet")
    p.add_argument("--dedupe", action="store_true",
                   help="skip cards that look identical to already saved ones")

    p.add_argument("--debug", dest="debug_dir", default=None, help="save annotated debug images here")
    p.add_argument("--move-done", dest="move_done", default=None,
                   help="move each processed sheet into this folder")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.size = parse_size(args.size)
    if args.cols and args.rows and args.cols * args.rows < args.expect:
        raise SystemExit("--cols * --rows must be >= --expect")

    show_banner()
    writer = GridWriter(args.out, prefix=PREFIX, start_grid=args.start_grid, dedupe=args.dedupe)
    print(f"Input : {Path(args.input).resolve()}")
    print(f"Output: {writer.out_dir.resolve()} (next grid {writer.next_grid})\n")
    return run_files(args, writer)


if __name__ == "__main__":
    raise SystemExit(main())
