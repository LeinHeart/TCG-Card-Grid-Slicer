"""Perspective-corrected card extraction."""

from __future__ import annotations

import cv2
import numpy as np

from .detect import CARD_RATIO, _dist, order_points

MAX_OUT_H = 2000
MIN_OUT_H = 400


def crop_card(
    img: np.ndarray,
    points: np.ndarray,
    size: tuple[int, int] | None = None,
    expand: float = 0.0,
) -> np.ndarray:
    """Warp the quad at `points` into a straight card image.

    size  : optional exact (width, height). Default keeps the measured card
            resolution (clamped) and forces the true 63:88 aspect ratio.
    expand: fraction the source quad is grown by before sampling. Absorbs a
            slightly-too-small detection so the card edge is never cut.
    """
    p = order_points(points)
    if expand > 0:
        centre = p.mean(axis=0)
        p = (centre + (p - centre) * (1.0 + expand)).astype(np.float32)

    if size is None:
        w = (_dist(p[0], p[1]) + _dist(p[3], p[2])) * 0.5
        h = (_dist(p[1], p[2]) + _dist(p[0], p[3])) * 0.5
        if w < h:  # portrait card: trust the longer side
            out_h = int(round(h))
        else:
            out_h = int(round(w / CARD_RATIO))
        out_h = int(np.clip(out_h, MIN_OUT_H, MAX_OUT_H))
        out_w = max(1, int(round(out_h * CARD_RATIO)))
    else:
        out_w, out_h = int(size[0]), int(size[1])

    dst = np.array(
        [[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(p, dst)
    return cv2.warpPerspective(
        img, matrix, (out_w, out_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def trim_paper_edges(
    img: np.ndarray,
    max_frac: float = 0.03,
    max_std: float = 2.0,
) -> np.ndarray:
    """Cut flat, featureless strips off the borders of a crop.

    Paper (even in shadow) has almost no variation along a row/column, while a
    card border always carries some structure, so a low standard deviation is a
    reliable "this is not the card" signal.
    """
    h, w = img.shape[:2]
    if h < 20 or w < 20:
        return img

    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) if img.ndim == 3 else img.astype(np.float32)
    max_y = max(1, int(max_frac * h))
    max_x = max(1, int(max_frac * w))

    def flat_row(y: int) -> bool:
        return float(g[y].std()) < max_std

    def flat_col(x: int) -> bool:
        return float(g[:, x].std()) < max_std

    top = bottom = left = right = 0
    while top < max_y and flat_row(top):
        top += 1
    while bottom < max_y and flat_row(h - 1 - bottom):
        bottom += 1
    while left < max_x and flat_col(left):
        left += 1
    while right < max_x and flat_col(w - 1 - right):
        right += 1

    if top + bottom + left + right == 0:
        return img
    return img[top : h - bottom, left : w - right]


def pad_card(
    img: np.ndarray,
    pad_x: int,
    pad_y: int,
    color: tuple[int, int, int] = (255, 255, 255),
    transparent: bool = False,
) -> np.ndarray:
    """Centre the card on a larger canvas, adding pad_x / pad_y per side."""
    pad_x = max(0, int(pad_x))
    pad_y = max(0, int(pad_y))
    h, w = img.shape[:2]
    if pad_x == 0 and pad_y == 0 and not transparent:
        return img

    if transparent:
        src = img if img.shape[2] == 4 else cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        canvas = np.zeros((h + 2 * pad_y, w + 2 * pad_x, 4), np.uint8)
        canvas[pad_y : pad_y + h, pad_x : pad_x + w] = src
        if src is img:  # already had alpha (unlikely) - keep the card fully opaque
            canvas[pad_y : pad_y + h, pad_x : pad_x + w, 3] = 255
        return canvas

    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    canvas = np.full((h + 2 * pad_y, w + 2 * pad_x, 3), color, np.uint8)
    canvas[pad_y : pad_y + h, pad_x : pad_x + w] = img
    return canvas
