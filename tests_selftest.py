"""Synthetic self-test: builds fake 3x2 sheets and asserts the split is correct.

    python tests_selftest.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cardscan import (  # noqa: E402
    GridWriter,
    crop_card,
    detect_cards,
    grid_hint,
    grid_quads,
    order_grid,
)

CARD_W, CARD_H = 630, 880  # 63 x 88 mm * 10
COLORS = [
    (40, 40, 220), (40, 200, 40), (220, 150, 40),
    (200, 40, 200), (40, 200, 220), (220, 220, 40),
]


def make_card(color, label: str, border: int = 26) -> np.ndarray:
    card = np.full((CARD_H, CARD_W, 3), 255, np.uint8)
    card[border:-border, border:-border] = color
    cv2.putText(card, label, (border + 30, CARD_H // 2), cv2.FONT_HERSHEY_SIMPLEX, 4.0, (255, 255, 255), 10)
    return card


def paste(canvas, card, center, angle=0.0):
    h, w = card.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    m[0, 2] += nw / 2 - w / 2
    m[1, 2] += nh / 2 - h / 2
    rot = cv2.warpAffine(card, m, (nw, nh), borderValue=(255, 255, 255))
    x = int(center[0] - nw / 2)
    y = int(center[1] - nh / 2)
    canvas[y : y + nh, x : x + nw] = rot
    return canvas


def build_sheet(angles=None, gap=(70, 90), pad=(120, 140)):
    angles = angles or [0.0] * 6
    gx, gy = gap
    px, py = pad
    sheet_w = px * 2 + CARD_W * 3 + gx * 2
    sheet_h = py * 2 + CARD_H * 2 + gy
    sheet = np.full((sheet_h, sheet_w, 3), 55, np.uint8)
    centers = []
    for r in range(2):
        for c in range(3):
            cx = px + CARD_W / 2 + c * (CARD_W + gx)
            cy = py + CARD_H / 2 + r * (CARD_H + gy)
            centers.append((cx, cy))
    for i, (cx, cy) in enumerate(centers):
        paste(sheet, make_card(COLORS[i], str(i + 1)), (cx, cy), angles[i])
    return sheet, centers


def center_color(img):
    h, w = img.shape[:2]
    patch = img[int(h * 0.45) : int(h * 0.55), int(w * 0.45) : int(w * 0.55)]
    return patch.reshape(-1, 3).mean(axis=0)


def check(name, img, expect_match=True):
    dets, diag = detect_cards(img, expect=6, cols=3)
    print(f"\n[{name}] kept={diag['kept']} unique={diag['unique']} raw={diag['candidates']}")
    assert diag["kept"] == 6, f"{name}: expected 6 detections, got {diag['kept']}"

    ordered = order_grid(dets, cols=3)
    ok = 0
    for i, det in enumerate(ordered):
        card = crop_card(img, det.points)
        got = center_color(card)
        want = np.array(COLORS[i], float)
        dist = float(np.linalg.norm(got - want))
        status = "OK " if dist < 60 else "BAD"
        if dist < 60:
            ok += 1
        print(f"  card {i + 1}: {card.shape[1]}x{card.shape[0]} center={got.round(0)} "
              f"want={want.round(0)} dist={dist:.1f} {status}")
    if expect_match:
        assert ok == 6, f"{name}: only {ok}/6 crops landed on the right card"
    return dets


def check_fallback():
    flat = np.full((1200, 1800, 3), 90, np.uint8)  # nothing to detect
    dets, diag = detect_cards(flat, expect=6, cols=3)
    print(f"\n[fallback] detections={diag['kept']} (expected 0)")
    assert diag["kept"] == 0, "flat image should not produce detections"
    quads = grid_quads(flat.shape, cols=3, rows=2, inset=0.02)
    assert len(quads) == 6
    for q in quads:
        card = crop_card(flat, q)
        assert card.shape[0] > 100 and card.shape[1] > 100
    print("  grid fallback produced 6 crops")


def check_padding():
    from cardscan import pad_card
    from cardscan.detect import CARD_W_MM

    sheet, _ = build_sheet()
    dets, _ = detect_cards(sheet, expect=6)
    assert len(dets) == 6
    card = crop_card(sheet, order_grid(dets, 3)[0].points, size=(493, 689))
    px_per_mm = card.shape[1] / CARD_W_MM
    pad = int(round(5.0 * px_per_mm))  # 0.5 cm
    padded = pad_card(card, pad, pad)
    h, w = padded.shape[:2]
    assert (w, h) == (493 + 2 * pad, 689 + 2 * pad), padded.shape
    # margin is pure white, card area is untouched
    assert padded[:pad].min() == 255 and padded[-pad:].min() == 255
    assert padded[:, :pad].min() == 255 and padded[:, -pad:].min() == 255
    assert np.array_equal(padded[pad : pad + 689, pad : pad + 493], card)

    rgba = pad_card(card, pad, pad, transparent=True)
    assert rgba.shape[2] == 4
    assert rgba[0, 0, 3] == 0 and rgba[pad + 10, pad + 10, 3] == 255
    print(f"\n[padding] 0.5cm margin -> {w}x{h} (card 493x689, margin {pad}px/side), transparent OK")


def check_rotated_sheet(angle: float = 3.0):
    sheet, _ = build_sheet()
    h, w = sheet.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    rotated = cv2.warpAffine(sheet, m, (w, h), borderValue=(55, 55, 55))
    check(f"sheet rotated {angle}deg", rotated)


def check_grid_naming():
    with tempfile.TemporaryDirectory() as tmp:
        img = np.zeros((100, 80, 3), np.uint8)

        w = GridWriter(tmp)
        assert w.begin(None) == 1
        p, _ = w.save(img, 1)
        assert p.name == "Card_grid1_card1.png", p.name
        p, _ = w.save(img + 1, 2)
        assert p.name == "Card_grid1_card2.png", p.name

        assert w.begin(None) == 2  # next grid after the used one
        p, _ = w.save(img + 2, 1)
        assert p.name == "Card_grid2_card1.png", p.name

        # a filename hint claims that grid number when it is still free
        assert w.begin(5) == 5
        p, _ = w.save(img + 3, 3)
        assert p.name == "Card_grid5_card3.png", p.name

        # a hint that is already taken falls back to the next free number
        assert w.begin(1) == 3

        # nothing written -> the number is released again
        w.abort()
        assert w.begin(None) == 3, "aborted grid number must be reusable"

        # resume across runs
        w2 = GridWriter(tmp)
        assert w2.next_grid == 6, w2.next_grid
        assert w2.begin(None) == 6

        # --start-grid can only move forward
        w3 = GridWriter(tmp, start_grid=20)
        assert w3.next_grid == 20, w3.next_grid

        # hints lifted from file names
        assert grid_hint("card_scan_grid_12.png") == 12
        assert grid_hint("sheet.png") is None
        assert grid_hint("grid_99999.png") is None
        print("\n[grid naming] hint + resume + abort + no-overwrite all OK")


def main() -> int:
    sheet, _ = build_sheet()
    check("clean grid", sheet)

    rng = np.random.default_rng(7)
    tilted, _ = build_sheet(angles=list(rng.uniform(-5, 5, 6)))
    check("tilted cards", tilted)

    # slight perspective: squeeze the top of the sheet
    h, w = sheet.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[w * 0.06, h * 0.02], [w * 0.94, 0], [w, h], [0, h * 0.99]])
    persp = cv2.warpPerspective(sheet, cv2.getPerspectiveTransform(src, dst), (w, h),
                                borderValue=(55, 55, 55))
    check("perspective", persp)

    check_fallback()
    check_padding()
    check_rotated_sheet(3.0)
    check_rotated_sheet(-4.5)
    check_grid_naming()
    print("\nALL TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
