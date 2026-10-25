"""Card detection.

Pipeline: several binary masks (Otsu both ways, adaptive threshold, closed Canny
edges, paper-relative split) -> contours -> quadrilateral per contour -> quality
filter -> dedupe by rotated-rect overlap -> pick the N candidates whose area is
closest to the median -> fit them into the cell lattice -> reading order.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# A real trading card is 63 x 88 mm.
CARD_W_MM = 63.0
CARD_H_MM = 88.0
CARD_RATIO = CARD_W_MM / CARD_H_MM

DETECT_MAX_SIDE = 1600  # detection runs on a downscaled copy for speed
MIN_AREA_FRAC = 0.004  # of the frame
MAX_AREA_FRAC = 0.60
MIN_CARD_SIDE = 90  # px on the detection copy
ASPECT_MIN = 0.45  # min(side)/max(side) - cards are ~0.72
ASPECT_MAX = 0.96  # reject near-squares
MIN_RECTANGULARITY = 0.75  # contour area / quad area
DEDUPE_OVERLAP = 0.45


@dataclass
class Detection:
    points: np.ndarray  # (4, 2) float32, ordered tl, tr, br, bl (source coords)
    score: float
    area: float
    source: str = ""

    @property
    def center(self) -> np.ndarray:
        return self.points.mean(axis=0)

    @property
    def size(self) -> tuple[float, float]:
        """(width, height) as the mean of opposite sides."""
        p = self.points
        w = (_dist(p[0], p[1]) + _dist(p[3], p[2])) * 0.5
        h = (_dist(p[1], p[2]) + _dist(p[0], p[3])) * 0.5
        return float(w), float(h)


def _dist(a, b) -> float:
    return float(np.hypot(*(np.asarray(a, float) - np.asarray(b, float))))


def _cross2(a, b) -> float:
    """2D cross product (NumPy's np.cross dropped 2D support)."""
    return float(a[0]) * float(b[1]) - float(a[1]) * float(b[0])


def order_points(pts) -> np.ndarray:
    """Return corners as tl, tr, br, bl (robust for rotated quads)."""
    p = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
    if len(p) != 4:
        raise ValueError("need exactly 4 points")
    c = p.mean(axis=0)
    ang = np.arctan2(p[:, 1] - c[1], p[:, 0] - c[0])
    p = p[np.argsort(ang)]
    p = np.roll(p, -int(np.argmin(p.sum(axis=1))), axis=0)
    # cross > 0 in image coords means tl -> tr -> br -> bl
    if _cross2(p[1] - p[0], p[2] - p[0]) < 0:
        p = p[[0, 3, 2, 1]]
    return p.astype(np.float32)


def _masks(gray: np.ndarray) -> dict[str, np.ndarray]:
    """Several complementary binarisations; every one is a separate vote."""
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)
    k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    k5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    k9 = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    out: dict[str, np.ndarray] = {}

    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    out["otsu"] = otsu
    out["otsu_inv"] = cv2.bitwise_not(otsu)

    out["adaptive"] = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 41, 8
    )

    # Cards can be nearly white on white paper: their outline is a faint edge,
    # so work on a contrast-equalised copy with low Canny thresholds.
    smooth = cv2.GaussianBlur(clahe, (5, 5), 0)
    for lo, hi in ((10, 30), (30, 90)):
        edges = cv2.Canny(smooth, lo, hi)
        out[f"canny{lo}"] = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k5, iterations=2)

    # Anything measurably darker than the paper counts as card.
    paper = float(np.percentile(gray, 97))
    if paper >= 180:
        _, split = cv2.threshold(gray, max(0.0, paper - 4.0), 255, cv2.THRESH_BINARY_INV)
        split = cv2.morphologyEx(split, cv2.MORPH_OPEN, k3)
        out["paper_split"] = cv2.morphologyEx(split, cv2.MORPH_CLOSE, k9)
    return out


def _score_quad(pts: np.ndarray, contour_area: float) -> Detection | None:
    p = order_points(pts)
    w = (_dist(p[0], p[1]) + _dist(p[3], p[2])) * 0.5
    h = (_dist(p[1], p[2]) + _dist(p[0], p[3])) * 0.5
    if w <= 1 or h <= 1 or min(w, h) < MIN_CARD_SIDE:
        return None

    aspect = min(w, h) / max(w, h)
    if not (ASPECT_MIN <= aspect <= ASPECT_MAX):
        return None

    quad_area = float(cv2.contourArea(p))
    if quad_area <= 0:
        return None
    rectangularity = min(1.0, contour_area / quad_area)
    if rectangularity < MIN_RECTANGULARITY:
        return None

    aspect_score = float(np.exp(-(((aspect - CARD_RATIO) / 0.13) ** 2)))
    score = aspect_score * rectangularity * float(np.sqrt(quad_area))
    return Detection(points=p, score=score, area=quad_area)


def _quads_from_mask(mask, name: str, min_area: float, max_area: float) -> list[Detection]:
    # RETR_LIST, not RETR_EXTERNAL: a scan often has a faint frame/border that
    # connects every card into one outer contour.
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    found: list[Detection] = []
    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area < min_area or area > max_area:
            continue
        peri = float(cv2.arcLength(cnt, True))
        if peri <= 0:
            continue
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            pts = approx.reshape(4, 2).astype(np.float32)
        else:
            pts = cv2.boxPoints(cv2.minAreaRect(cnt)).astype(np.float32)
        det = _score_quad(pts, area)
        if det is not None:
            det.source = name
            found.append(det)
    return found


def _rot_rect(det: Detection):
    return cv2.minAreaRect(det.points.reshape(-1, 1, 2))


def _intersect_area(a: Detection, b: Detection) -> float:
    ra, rb = _rot_rect(a), _rot_rect(b)
    try:
        _, inter = cv2.rotatedRectangleIntersection(ra, rb)
    except cv2.error:
        return 0.0
    if inter is None or len(inter) < 3:
        return 0.0
    return abs(float(cv2.contourArea(inter.astype(np.float32))))


def _overlap(a: Detection, b: Detection) -> float:
    """Intersection over the smaller rect: 1.0 means fully contained."""
    ra, rb = _rot_rect(a), _rot_rect(b)
    smaller = min(ra[1][0] * ra[1][1], rb[1][0] * rb[1][1])
    return _intersect_area(a, b) / smaller if smaller > 0 else 0.0


def _dedupe(dets: list[Detection]) -> list[Detection]:
    """Collapse duplicates, and drop merged blobs that swallow several cards."""
    ordered = sorted(dets, key=lambda d: d.area)

    # A big candidate that fully contains >= 2 smaller candidates is a merged
    # blob (two touching cards on one mask). The pieces win.
    children: dict[int, int] = {}
    for i, small in enumerate(ordered):
        for j in range(i + 1, len(ordered)):
            big = ordered[j]
            if big.area < small.area * 1.35:
                continue
            if _overlap(small, big) > 0.75:
                children[j] = children.get(j, 0) + 1
    merged_blobs = {j for j, n in children.items() if n >= 2}

    kept: list[Detection] = []
    for d in sorted(
        (d for i, d in enumerate(ordered) if i not in merged_blobs),
        key=lambda x: x.score,
        reverse=True,
    ):
        if all(_overlap(d, k) < DEDUPE_OVERLAP for k in kept):
            kept.append(d)
    return kept


def _pick_best(dets: list[Detection], expect: int) -> list[Detection]:
    """Keep `expect` candidates: prefer the ones close to the median card area."""
    if len(dets) <= expect:
        return dets
    areas = np.array([d.area for d in dets], dtype=np.float64)
    med = float(np.median(areas)) or 1.0
    ranked = sorted(dets, key=lambda d: (abs(d.area - med) / med, -d.score))
    return ranked[:expect]


def order_grid(dets: list[Detection], cols: int = 3) -> list[Detection]:
    """Sort detections in reading order (rows top->bottom, each row left->right)."""
    if not dets:
        return []
    cols = max(1, cols)
    by_y = sorted(dets, key=lambda d: float(d.center[1]))
    rows = max(1, int(np.ceil(len(by_y) / cols)))
    out: list[Detection] = []
    for r in range(rows):
        chunk = by_y[r * cols : (r + 1) * cols]
        out.extend(sorted(chunk, key=lambda d: float(d.center[0])))
    return out


def _cluster_1d(values: list[float], tol: float) -> list[list[float]]:
    vals = sorted(values)
    groups: list[list[float]] = [[vals[0]]]
    for v in vals[1:]:
        if v - groups[-1][-1] <= tol:
            groups[-1].append(v)
        else:
            groups.append([v])
    return groups


def infer_layout(dets: list[Detection]) -> tuple[int, int]:
    """(cols, rows) from where the detected cards actually sit."""
    if not dets:
        return (0, 0)
    med_w = float(np.median([d.size[0] for d in dets]))
    med_h = float(np.median([d.size[1] for d in dets]))
    cols = len(_cluster_1d([float(d.center[0]) for d in dets], 0.6 * med_w))
    rows = len(_cluster_1d([float(d.center[1]) for d in dets], 0.6 * med_h))
    return cols, rows


def best_layout(shape, expect: int = 6) -> tuple[int, int]:
    """Guess (cols, rows) for a blind grid split: the cells closest to card shape."""
    h, w = shape[:2]
    best = (expect, 1)
    best_delta = float("inf")
    for cols in range(1, expect + 1):
        if expect % cols:
            continue
        rows = expect // cols
        cw, ch = w / cols, h / rows
        ratio = min(cw, ch) / max(cw, ch)
        delta = abs(ratio - CARD_RATIO)
        if delta < best_delta:
            best, best_delta = (cols, rows), delta
    return best


def grid_quads(shape, cols: int = 3, rows: int = 2, inset: float = 0.0) -> list[np.ndarray]:
    """Fallback: slice the frame into a cols x rows grid. `inset` trims each cell."""
    h, w = shape[:2]
    cw, ch = w / cols, h / rows
    quads = []
    for r in range(rows):
        for c in range(cols):
            x0 = c * cw + inset * cw
            x1 = (c + 1) * cw - inset * cw
            y0 = r * ch + inset * ch
            y1 = (r + 1) * ch - inset * ch
            quads.append(
                np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)
            )
    return quads


def _rect_points(cx: float, cy: float, w: float, h: float) -> np.ndarray:
    return np.array(
        [
            [cx - w / 2, cy - h / 2],
            [cx + w / 2, cy - h / 2],
            [cx + w / 2, cy + h / 2],
            [cx - w / 2, cy + h / 2],
        ],
        dtype=np.float32,
    )


def _integral(mag: np.ndarray) -> np.ndarray:
    return cv2.integral(mag)  # (H+1, W+1) float64


def _rect_sum(ii: np.ndarray, x0: float, y0: float, x1: float, y1: float) -> float:
    h, w = ii.shape[0] - 1, ii.shape[1] - 1
    x0 = int(np.clip(round(x0), 0, w))
    x1 = int(np.clip(round(x1), 0, w))
    y0 = int(np.clip(round(y0), 0, h))
    y1 = int(np.clip(round(y1), 0, h))
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return float(ii[y1, x1] - ii[y0, x1] - ii[y1, x0] + ii[y0, x0])


def _align_box(
    gray: np.ndarray,
    cx: float,
    cy: float,
    w: float,
    h: float,
    search: float = 0.12,
    band: float = 2.0,
) -> tuple[float, float, float]:
    """Slide a w x h box around (cx, cy) to land its 4 edges on the card border.

    Scores the mean gradient inside a thin band along all four sides, with a
    penalty for moving away from the predicted centre and a small scale search.
    Returns (cx, cy, confidence) where confidence is the winning edge response
    minus the response at the predicted position (0 = no better than the guess).
    """
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    peak = float(mag.max())
    if peak <= 0:
        return cx, cy, 0.0
    mag = mag * (255.0 / peak)
    ii = _integral(mag)

    def edge_score(x0, y0, x1, y1) -> float:
        band_px = max(1.0, band)
        top = _rect_sum(ii, x0, y0 - band_px, x1, y0 + band_px)
        bot = _rect_sum(ii, x0, y1 - band_px, x1, y1 + band_px)
        left = _rect_sum(ii, x0 - band_px, y0, x0 + band_px, y1)
        right = _rect_sum(ii, x1 - band_px, y0, x1 + band_px, y1)
        area = (x1 - x0) * band_px * 2 * 2 + (y1 - y0) * band_px * 2 * 2
        return (top + bot + left + right) / max(1.0, area)

    def total(x0, y0, x1, y1) -> float:
        dx = (x0 + x1) / 2.0 - cx
        dy = (y0 + y1) / 2.0 - cy
        # stay near the prediction unless the edges clearly say otherwise
        penalty = 34.0 * ((dx / max(1.0, search * w)) ** 2 + (dy / max(1.0, search * h)) ** 2)
        return edge_score(x0, y0, x1, y1) - penalty

    base = total(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
    step = max(2, int(round(min(w, h) * 0.01)))
    rx = int(round(search * w))
    ry = int(round(search * h))

    best = (base, 0.0, 0.0, 1.0)
    for scale in (1.0, 0.97, 1.03):
        bw, bh = w * scale, h * scale
        for dx in range(-rx, rx + 1, step):
            for dy in range(-ry, ry + 1, step):
                ncx, ncy = cx + dx, cy + dy
                s = total(ncx - bw / 2, ncy - bh / 2, ncx + bw / 2, ncy + bh / 2)
                if s > best[0]:
                    best = (s, float(dx), float(dy), scale)
    gain, dx, dy, scale = best
    return cx + dx, cy + dy, float(gain - base)


def _synth_cell(
    gray: np.ndarray, cx: float, cy: float, w: float, h: float
) -> tuple[np.ndarray, float]:
    """Build a cell from the grid, then align it to the card border if visible."""
    nx, ny, conf = _align_box(gray, cx, cy, w, h)
    if conf < 1.0:  # no usable edge evidence: trust the lattice
        nx, ny = cx, cy
    return _rect_points(nx, ny, w, h), conf


def _snap_to_grid(
    candidates: list[Detection],
    picked: list[Detection],
    cols: int,
    rows: int,
    gray: np.ndarray,
) -> list[Detection]:
    """Use the confident detections as a lattice; fill cells that failed.

    A pale card on white paper often has no detectable outline. If at least
    `cols*rows - 1` cards were found, their centres define the grid, and the
    missing cell is rebuilt from the lattice (then edge-aligned), so the sheet
    still yields the full set.
    """
    if cols < 1 or rows < 1 or len(picked) < cols * rows - 1:
        return picked

    med_w = float(np.median([d.size[0] for d in picked]))
    med_h = float(np.median([d.size[1] for d in picked]))
    col_groups = _cluster_1d([float(d.center[0]) for d in picked], 0.6 * med_w)
    row_groups = _cluster_1d([float(d.center[1]) for d in picked], 0.6 * med_h)
    if len(col_groups) != cols or len(row_groups) != rows:
        return picked

    col_x = [float(np.median(g)) for g in col_groups]
    row_y = [float(np.median(g)) for g in row_groups]

    used: set[int] = set()
    out: list[Detection] = []
    for cy in row_y:
        for cx in col_x:
            best_i, best_key = -1, None
            for i, d in enumerate(candidates):
                if i in used:
                    continue
                if abs(d.center[0] - cx) > 0.30 * med_w or abs(d.center[1] - cy) > 0.30 * med_h:
                    continue
                if not (0.75 * med_w <= d.size[0] <= 1.40 * med_w):
                    continue
                if not (0.75 * med_h <= d.size[1] <= 1.40 * med_h):
                    continue
                key = (
                    abs(d.center[0] - cx) / med_w + abs(d.center[1] - cy) / med_h,
                    -d.score,
                )
                if best_key is None or key < best_key:
                    best_i, best_key = i, key

            if best_i >= 0:
                used.add(best_i)
                out.append(candidates[best_i])
                continue

            pts, conf = _synth_cell(gray, cx, cy, med_w, med_h)
            out.append(Detection(points=pts, score=conf, area=med_w * med_h, source="grid"))
    return out


def _content_mask(gray: np.ndarray) -> np.ndarray:
    """255 where pixels differ from the sheet background (paper or mat)."""
    ring = np.concatenate(
        [
            gray[0:8, :].ravel(),
            gray[-8:, :].ravel(),
            gray[:, 0:8].ravel(),
            gray[:, -8:].ravel(),
        ]
    )
    bg = float(np.median(ring)) if ring.size else float(np.median(gray))
    diff = np.abs(gray.astype(np.int16) - int(round(bg)))
    return (diff > 8).astype(np.uint8) * 255


def _profile_bands(
    profile: np.ndarray, min_frac: float = 0.02, min_len_frac: float = 0.05
) -> list[tuple[int, int]]:
    """Contiguous bands where content exists, ignoring hair-thin artefacts."""
    n = profile.size
    active = profile > min_frac
    runs: list[list[int]] = []
    start = -1
    for i in range(n):
        if active[i]:
            if start < 0:
                start = i
        elif start >= 0:
            runs.append([start, i - 1])
            start = -1
    if start >= 0:
        runs.append([start, n - 1])
    if not runs:
        return []

    merge_gap = max(2, int(0.01 * n))
    merged = [runs[0]]
    for r in runs[1:]:
        if r[0] - merged[-1][1] <= merge_gap:
            merged[-1][1] = r[1]
        else:
            merged.append(r)

    min_len = max(8, int(min_len_frac * n))
    return [(a, b) for a, b in merged if b - a + 1 >= min_len]


def projection_cells(
    gray: np.ndarray, expect: int = 6
) -> tuple[list[np.ndarray], int, int]:
    """Split the sheet by content bands: (quads in reading order, cols, rows).

    A row/column of paper between cards shows up as an empty band in the
    content profile, which is far more reliable than contour finding when the
    cards are pale, touching, or low contrast.
    """
    mask = _content_mask(gray)
    col_bands = _profile_bands(mask.mean(axis=0) / 255.0)
    row_bands = _profile_bands(mask.mean(axis=1) / 255.0)
    cols, rows = len(col_bands), len(row_bands)
    if cols < 1 or rows < 1 or cols * rows != expect:
        return [], cols, rows

    quads: list[np.ndarray] = []
    for y0, y1 in row_bands:
        for x0, x1 in col_bands:
            quads.append(
                np.array(
                    [[x0, y0], [x1 + 1, y0], [x1 + 1, y1 + 1], [x0, y1 + 1]],
                    dtype=np.float32,
                )
            )
    return quads, cols, rows


def _grow_to_cell(quad: np.ndarray, cell_quad: np.ndarray) -> np.ndarray:
    """Canny tends to lock onto the inner print line; push the quad back out to
    the cell boundary when the two clearly describe the same card."""
    qb = cv2.boundingRect(np.round(quad).astype(np.int32))
    cb = cv2.boundingRect(np.round(cell_quad).astype(np.int32))
    qx, qy, qw, qh = qb
    cx, cy, cw, ch = cb
    if qw <= 0 or qh <= 0 or cw <= 0 or ch <= 0:
        return quad
    if cw / qw > 1.2 or ch / qh > 1.2:
        return quad
    if abs((qx + qw / 2) - (cx + cw / 2)) > 0.03 * cw:
        return quad
    if abs((qy + qh / 2) - (cy + ch / 2)) > 0.03 * ch:
        return quad
    kx = min(1.08, cw / qw)
    ky = min(1.08, ch / qh)
    if kx <= 1.001 and ky <= 1.001:
        return quad
    centre = quad.mean(axis=0)
    out = np.empty_like(quad)
    out[:, 0] = centre[0] + (quad[:, 0] - centre[0]) * kx
    out[:, 1] = centre[1] + (quad[:, 1] - centre[1]) * ky
    return out.astype(np.float32)


def _fit_cells(cells: list[np.ndarray], candidates: list[Detection]) -> list[Detection]:
    """Prefer a real contour inside each cell, else use the cell rectangle."""
    out: list[Detection] = []
    used: set[int] = set()
    for quad in cells:
        cell = Detection(points=quad, score=0.0, area=float(cv2.contourArea(quad)))
        best_i, best_score = -1, 0.0
        for i, d in enumerate(candidates):
            if i in used:
                continue
            if not (0.75 * cell.area <= d.area <= 1.30 * cell.area):
                continue
            if _intersect_area(d, cell) < 0.70 * cell.area:
                continue
            if d.score > best_score:
                best_i, best_score = i, d.score
        if best_i >= 0:
            used.add(best_i)
            cand = candidates[best_i]
            out.append(
                Detection(
                    points=_grow_to_cell(cand.points, quad),
                    score=cand.score,
                    area=cand.area,
                    source=cand.source,
                )
            )
        else:
            out.append(
                Detection(points=quad, score=0.0, area=cell.area, source="cell")
            )
    return out


def detect_cards(
    img: np.ndarray,
    expect: int = 6,
    cols: int | None = None,
) -> tuple[list[Detection], dict]:
    """Detect up to `expect` cards. Returns (detections in reading order, diagnostics)."""
    h, w = img.shape[:2]
    scale = min(1.0, DETECT_MAX_SIDE / float(max(h, w)))
    if scale < 1.0:
        small = cv2.resize(
            img, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        small = img

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if small.ndim == 3 else small
    img_area = float(gray.shape[0] * gray.shape[1])
    min_area = img_area * MIN_AREA_FRAC
    max_area = img_area * MAX_AREA_FRAC

    cands: list[Detection] = []
    for name, mask in _masks(gray).items():
        cands.extend(_quads_from_mask(mask, name, min_area, max_area))

    unique = _dedupe(cands)
    picked = _pick_best(unique, expect)

    rows = 0
    method = "contour"
    cells, proj_cols, proj_rows = projection_cells(gray, expect)
    if cells:
        method = "projection"
        cols, rows = proj_cols, proj_rows
        final = _fit_cells(cells, unique)
    else:
        if cols is None:
            cols, rows = infer_layout(picked) if picked else (0, 0)
        else:
            rows = max(1, int(np.ceil(expect / max(1, cols))))

        final = picked
        if cols * rows == expect and len(picked) >= expect - 1:
            method = "contour+grid"
            final = _snap_to_grid(unique, picked, cols, rows, gray)

    if scale < 1.0:
        for d in final:
            d.points = d.points / scale
            d.area = d.area / (scale * scale)
    ordered = order_grid(final, cols=cols or 3)

    diag = {
        "candidates": len(cands),
        "unique": len(unique),
        "kept": len(ordered),
        "cols": cols,
        "rows": rows,
        "method": method,
        "sources": sorted({d.source for d in cands}),
    }
    return ordered, diag
