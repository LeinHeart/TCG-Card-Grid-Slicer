# Documentation — Auto Grid Scan (TCG Card Sheets)

Full technical reference: what each library does, what each function does, and how
a scanned sheet becomes six PNG files.

- [1. What the app does](#1-what-the-app-does)
- [2. Requirements and install](#2-requirements-and-install)
- [3. The workflow](#3-the-workflow)
- [4. Libraries and what they are used for](#4-libraries-and-what-they-are-used-for)
- [5. Module by module](#5-module-by-module)
- [6. The detection pipeline in detail](#6-the-detection-pipeline-in-detail)
- [7. Geometry and the 0.5 cm margin](#7-geometry-and-the-05-cm-margin)
- [8. Naming rules](#8-naming-rules)
- [9. Command line reference](#9-command-line-reference)
- [10. Tuning constants](#10-tuning-constants)
- [11. Failure modes and fallbacks](#11-failure-modes-and-fallbacks)
- [12. Extending the app](#12-extending-the-app)
- [13. Testing](#13-testing)
- [14. File map](#14-file-map)

---

## 1. What the app does

Input: a scanned (or photographed) sheet containing a grid of trading cards —
typically 6 cards, 3 columns x 2 rows or 2 columns x 3 rows.

Output: one PNG per card, each card perspective-corrected, upscaled/downscaled to
one uniform size per sheet, and centred on a white canvas with a blank margin
(0.5 cm by default) around it.

The app is deliberately forgiving:

- the sheet does **not** need to be perfectly aligned — each card is warped with
  its own four corners, so rotation and perspective are corrected per card;
- the grid layout is **detected**, not assumed (2x3 and 3x2 both work);
- pale cards with no visible outline are rebuilt from the lattice of the cards
  that *were* detected;
- if detection fails completely, a blind grid split still produces output.

The whole thing is a batch tool: drop sheets in a folder, double-click
`scan.bat`, walk away.

---

## 2. Requirements and install

```powershell
pip install -r requirements.txt
```

| Package | Version | Role |
|---|---|---|
| `numpy` | >= 1.24 | all pixel data, array math, medians, clipping |
| `opencv-python` | >= 4.8 | image I/O, thresholding, contours, warps |

Tested on Python 3.15 with `opencv-python 5.0.0` and `numpy 2.5.4`.

No other third-party dependencies. Everything else is standard library
(`argparse`, `pathlib`, `shutil`, `re`, `dataclasses`, `tempfile`).

---

## 3. The workflow

### 3.1 Batch flow (the normal one)

```
  scanner
     |
     v
 scan_grids\           <- you drop every scanned sheet here
     |
     |  double-click scan.bat
     v
 +------------------------------+
 | scan.bat                     |
 |  python scan.py scan_grids   |
 |         -o cards             |
 |         --move-done scanned  |
 +------------------------------+
     |
     v
 cardscan\detect.py   ->  where are the cards?      (quads in pixel coords)
     |
     v
 cardscan\warp.py     ->  straighten + trim + pad   (one image per card)
     |
     v
 cardscan\naming.py   ->  which file name?          (Card_gridN_cardM.png)
     |
     +--> cards\Card_grid1_card1.png ... Card_grid1_card6.png
     |    cards\Card_grid2_card1.png ... Card_grid2_card6.png
     |
     +--> scanned\   (the processed sheets, moved out of the way)
```

Running it a second time does not reprocess anything: the used sheets were moved
to `scanned\`, and the next sheet gets the next grid number.

### 3.2 Per-sheet flow

```
 image file
     |
     v
 read (unicode-safe imdecode)
     |
     v
 detect_cards()
     |   downscale to <=1600 px
     |   6 masks -> contours -> quads -> filter -> dedupe -> pick 6
     |   projection-profile cell split (preferred)
     |   or contour+grid / blind grid fallback
     v
 6 quads in reading order
     |
     v
 for each card:
     crop_card()         perspective warp to 63:88, uniform size
       |
       v
     trim_paper_edges()  cut flat shadow/paper strips (<=3% per side)
       |
       v
     resize to exact sheet-wide size (if trimmed)
       |
       v
     pad_card()          add 0.5 cm blank margin on all four sides
       |
       v
     GridWriter.save()   Card_grid{N}_card{M}.png, never overwrite
     |
     v
 6 PNG files
```

### 3.3 One scan, step by step, on the real sheet

The reference sheet `card_scan_grid_1.png` is 1792x2400 px, 2 columns x 3 rows,
cards about 493x689 px (≈200 DPI):

| Step | Result |
|---|---|
| detect | 6 cards, method `projection`, layout 2x3 |
| warp | each card → 493x689 px (true 63:88) |
| margin | `493 / 63 * 5 = 39` px per side |
| output | 571x767 px, card centred at (39,39)-(531,727) |
| names | `Card_grid1_card1.png` … `Card_grid1_card6.png` |

---

## 4. Libraries and what they are used for

### 4.1 NumPy — pixel storage and math

| Used for | Where |
|---|---|
| image buffers (`np.ndarray`, uint8 BGR / BGRA) | everywhere |
| `np.median` — median card height, background level, cluster centres | `detect.py`, `scan.py` |
| `np.clip` — clamp output size, search boxes to image bounds | `warp.py`, `detect.py` |
| `np.hypot` — distance between two quad corners | `detect.py::_dist` |
| `np.arctan2`, `np.argsort`, `np.roll` — sort quad corners by angle | `detect.py::order_points` |
| `np.percentile` — estimate paper brightness (97th percentile) | `detect.py::_masks` |
| `np.count_nonzero` — Hamming distance between card hashes | `naming.py::_duplicate` |
| `np.fromfile` / `buf.tofile` — unicode-safe file I/O | `naming.py::_imread/_imwrite` |
| `np.float32` corner arrays | `detect.py`, `warp.py` |

### 4.2 OpenCV (`cv2`) — everything image related

| Function | What it does here | Where |
|---|---|---|
| `imdecode` / `imencode` | read/write PNG through a byte buffer, so non-ASCII Windows paths work | `naming.py` |
| `cvtColor` | BGR→GRAY for detection, BGR↔BGRA for transparent margins | `detect.py`, `warp.py`, `naming.py` |
| `GaussianBlur` | remove scan noise before thresholding / Canny | `detect.py::_masks` |
| `createCLAHE().apply` | local contrast equalisation — makes pale white-on-white borders visible | `detect.py::_masks` |
| `threshold` + `THRESH_OTSU` | automatic global binarisation, both polarities | `detect.py::_masks` |
| `adaptiveThreshold` | local binarisation, survives uneven scanner light | `detect.py::_masks` |
| `Canny` | edge map at two sensitivity levels | `detect.py::_masks` |
| `morphologyEx` (`CLOSE`/`OPEN`) | join broken card outlines / remove speckle | `detect.py::_masks` |
| `findContours` (`RETR_LIST`) | candidate outlines; `RETR_LIST` because a faint frame often encloses all cards | `detect.py::_quads_from_mask` |
| `contourArea`, `arcLength` | size + perimeter filters | `detect.py` |
| `approxPolyDP` | reduce a contour to 4 corners when it is already rectangular | `detect.py::_quads_from_mask` |
| `isContourConvex` | reject non-convex approximations | `detect.py::_quads_from_mask` |
| `minAreaRect` / `boxPoints` | minimum rotated rectangle for blobby contours | `detect.py` |
| `rotatedRectangleIntersection` | overlap test between two card candidates | `detect.py::_intersect_area` |
| `Sobel` + `magnitude` | gradient magnitude for edge-snapping | `detect.py::_align_box` |
| `integral` | summed-area table so edge scores are O(1) | `detect.py::_integral` |
| `getPerspectiveTransform` + `warpPerspective` | the actual card straightening | `warp.py::crop_card` |
| `resize` | downscale for detection, restore exact output size | `detect.py`, `scan.py`, `warp.py` |
| `polylines`, `putText` | annotated `--debug` images | `scan.py::annotate` |
| `warpAffine`, `getRotationMatrix2D` | rotate synthetic sheets in the self-test | `tests_selftest.py` |

### 4.3 Standard library

| Module | Role |
|---|---|
| `argparse` | CLI, `--help`, defaults |
| `pathlib` | folder creation, extension checks, file names |
| `shutil` | move processed sheets into `scanned\` |
| `re` | output-name pattern matching (`Card_gridN_cardM.png`) |
| `dataclasses` | the `Detection` record |
| `tempfile` | throwaway folders in the self-test |

---

## 5. Module by module

### 5.1 `scan.py` — CLI and orchestration

| Function | Line | Responsibility |
|---|---|---|
| `show_banner()` | 48 | prints the ASCII banner |
| `parse_size(text)` | 57 | `"750x1050"` → `(750, 1050)`, validates minimums |
| `collect_inputs(target, recursive)` | 70 | file or folder → sorted list of image paths |
| `annotate(img, dets, grid)` | 82 | draws the detected quads + `grid.card` labels for `--debug` |
| `save_png(path, img)` | 93 | unicode-safe PNG write for debug output |
| `resolve_margin(args, card_w)` | 101 | cm/mm/px → margin in pixels (DPI-independent) |
| `process_frame(img, args, writer, tag, hint)` | 112 | **the core per-sheet routine**: detect → fallback → size → warp → trim → pad → save |
| `run_files(args, writer)` | 185 | iterates the folder, per-sheet messages, moves sheets to `--move-done` |
| `build_parser()` | 230 | every CLI flag and its default |
| `main(argv)` | 274 | wires args → `GridWriter` → `run_files` |

`process_frame` is the only place that knows the order of operations. Everything
else in the app is a pure function it calls.

### 5.2 `cardscan/detect.py` — find the cards

| Symbol | Line | Responsibility |
|---|---|---|
| `Detection` | 32 | dataclass: `points` (4x2 float32, tl→tr→br→bl), `score`, `area`, `source` |
| `Detection.center` / `.size` | 39 / 43 | mean of corners / mean of opposite side lengths |
| `_dist(a, b)` | 51 | Euclidean distance between two points |
| `_cross2(a, b)` | 55 | 2D cross product (NumPy removed 2D `cross`) |
| `order_points(pts)` | 60 | sorts 4 corners into tl, tr, br, bl robustly for rotated quads |
| `_masks(gray)` | 75 | builds the 6 binary/edge masks that vote for a card |
| `_score_quad(pts, contour_area)` | 108 | rejects by size/aspect/rectangularity, scores by aspect closeness x rectangularity x √area |
| `_quads_from_mask(...)` | 131 | contours → quads → `Detection`s for one mask |
| `_rot_rect` / `_intersect_area` / `_overlap` | 155-170 | rotated-rectangle overlap helpers |
| `_dedupe(dets)` | 177 | merges duplicate candidates, drops blobs that swallowed 2+ cards |
| `_pick_best(dets, expect)` | 204 | keeps the N candidates whose area is closest to the median |
| `order_grid(dets, cols)` | 214 | reading order: rows top→bottom, each row left→right |
| `_cluster_1d(values, tol)` | 228 | groups 1-D coordinates into columns/rows |
| `infer_layout(dets)` | 239 | how many columns/rows the detections actually form |
| `best_layout(shape, expect)` | 250 | blind-split guess: which cols x rows gives the most card-like cells |
| `grid_quads(shape, cols, rows, inset)` | 267 | slices the frame into a grid (fallback crop) |
| `_rect_points` | 284 | axis-aligned quad from centre + size |
| `_integral` / `_rect_sum` | 296 / 300 | summed-area table for O(1) box sums |
| `_align_box(gray, cx, cy, w, h)` | 311 | slides a box over a gradient map to land its edges on the card border |
| `_synth_cell(gray, cx, cy, w, h)` | 370 | build a missing cell from the lattice, then edge-align it |
| `_snap_to_grid(...)` | 380 | uses confident detections as a lattice to rebuild missing cells |
| `_content_mask(gray)` | 438 | 255 where pixels differ from the sheet background |
| `_profile_bands(profile, ...)` | 453 | contiguous bands of content in a 1-D projection |
| `projection_cells(gray, expect)` | 485 | splits the sheet by content bands → (quads, cols, rows) |
| `_grow_to_cell(quad, cell_quad)` | 513 | pushes an inner-print contour back out to the cell boundary |
| `_fit_cells(cells, candidates)` | 539 | best contour per cell, else the cell rectangle |
| `detect_cards(img, expect, cols)` | 573 | the public entry point, returns `(detections, diagnostics)` |

### 5.3 `cardscan/warp.py` — turn a quad into a card image

| Function | Line | Responsibility |
|---|---|---|
| `crop_card(img, points, size, expand)` | 14 | perspective warp to a straight card with the true 63:88 ratio |
| `trim_paper_edges(img, max_frac, max_std)` | 56 | removes flat (featureless) strips at the borders — shadow/paper |
| `pad_card(img, pad_x, pad_y, color, transparent)` | 96 | centres the card on a larger canvas (white, or transparent RGBA) |

### 5.4 `cardscan/naming.py` — output names, no overwrites

| Symbol | Line | Responsibility |
|---|---|---|
| `dhash(img, size)` | 19 | 64-bit difference hash used for the optional `--dedupe` |
| `grid_hint(path)` | 27 | last number in a filename (`sheet_grid_12.png` → 12), else `None` |
| `GridWriter` | 38 | allocates grid numbers and writes `Card_grid{N}_card{M}.png` |
| `.next_grid` | 71 | the number the next sheet will get |
| `.begin(hint)` | 74 | claims the grid number for the sheet being processed |
| `.abort()` | 87 | releases it again when nothing was written |
| `.path_for(card, grid)` | 97 | builds a path without writing |
| `.save(img, card_index, hash_img)` | 103 | dedupe check, no-overwrite bump, write → `(path, duplicate)` |
| `_imread` / `_imwrite` | 151 / 162 | unicode-safe PNG I/O |

### 5.5 `cardscan/__init__.py`

Re-exports the public API so `scan.py` and the tests import from one place:

```python
Detection, detect_cards, grid_quads, order_grid,
crop_card, trim_paper_edges, pad_card,
GridWriter, grid_hint
```

---

## 6. The detection pipeline in detail

### 6.1 Downscale

Detection runs on a copy capped at `DETECT_MAX_SIDE = 1600` px on the long side.
Contours are found on the small copy; the resulting corners are scaled back by
`1/scale` at the end. This keeps a 4000 px scan fast without losing accuracy
(card edges are many pixels wide even at 1600 px).

### 6.2 Six masks vote for "this is a card"

| Mask | Why it exists |
|---|---|
| `otsu` | cards darker than the sheet, global threshold |
| `otsu_inv` | cards lighter than the sheet (white cards on grey mat) |
| `adaptive` | uneven scanner lighting across the sheet |
| `canny10` | faint low-contrast borders (CLAHE-equalised, low thresholds) |
| `canny30` | the same, stricter — fewer false positives |
| `paper_split` | everything measurably darker than the paper (97th percentile − 4) |

Each mask produces its own candidate list; duplicates between masks are merged
later, so a card found by three masks counts once.

### 6.3 Contour → quad → quality gate

For every contour (`RETR_LIST`, area between 0.4% and 60% of the frame):

1. if `approxPolyDP` yields a **convex 4-gon**, use it (preserves perspective);
2. otherwise use the **minimum rotated rectangle** (robust for noisy blobs);
3. reject if `min(side) < 90 px`, if `min/max` aspect is outside `0.45 … 0.96`,
   or if `contour area / quad area < 0.75` (i.e. not rectangular enough);
4. score = `exp(-((aspect - 0.716)/0.13)^2) * rectangularity * sqrt(area)`.

A real card is 63:88 = **0.716**, so the score peaks at the true card shape.

### 6.4 Dedupe and pick

`_dedupe` does two things:

- **merged blobs** — a candidate that contains ≥ 2 much smaller candidates
  (≥ 1.35x area, > 0.75 overlap) is a blob where two touching cards merged; it is
  dropped and the pieces survive;
- **duplicates** — remaining candidates are taken greedily by score, rejecting
  anything overlapping a kept one by more than 45% of the smaller rect.

`_pick_best` then keeps the 6 candidates whose area is closest to the median card
area — background junk (shadows, the scanner lid, the sheet edge) is usually much
larger or much smaller than a card.

### 6.5 Preferred path: projection profile (`method = projection`)

This is what runs on real scans, and it is the most reliable mode.

1. `_content_mask` marks pixels that differ from the sheet background by more
   than 8 grey levels (background = median of an 8 px border ring).
2. Column profile = fraction of content pixels per column; row profile likewise.
3. `_profile_bands` finds contiguous bands of content, merging gaps < 1% and
   dropping bands shorter than 5% — those bands *are* the card columns/rows,
   because the paper gaps between cards read as empty.
4. If `bands_x * bands_y == expect` (6), the intersection rectangles are the
   cells. Otherwise this path is abandoned and the contour path runs instead.
5. `_fit_cells` gives every cell the best real contour inside it (area 0.75–1.30x
   the cell, at least 70% inside), so tilted cards keep their perspective; cells
   with no usable contour keep the plain rectangle with `source="cell"`.
6. `_grow_to_cell` expands a contour that locked onto the *inner print line*
   instead of the card edge, but only when both are clearly the same card
   (same centre within 3%, size within 1.2x) and never by more than 8%.

### 6.6 Second path: contour + grid lattice (`method = contour+grid`)

If the projection split is impossible but `cols * rows == expect` and at least 5
cards were detected:

1. the detected centres are clustered into columns and rows;
2. every cell is filled with the candidate closest to it, or
3. **rebuilt** by `_synth_cell`: the missing card is placed from the lattice, then
   `_align_box` searches ±12% around that spot (step 1% of the card's small side,
   scales 0.97 / 1.00 / 1.03) for the position whose four edges sit on the highest
   average gradient, with a penalty for moving too far from the prediction;
4. if the edge evidence is weak (gain < 1.0), the lattice position is trusted as-is.

This is what saves a pale card whose outline is invisible: 5 cards find the
lattice, the 6th is rebuilt.

### 6.7 Final path: blind grid split (`source = grid`)

If fewer than 5 cards were detected, `scan.py` calls `grid_quads` and slices the
whole image into `cols x rows` rectangles (`best_layout` picks the arrangement
whose cells are closest to 63:88, `--grid-inset` trims 2% per cell). You always
get 6 images, they are just not individually aligned. Disable with `--no-fallback`.

---

## 7. Geometry and the 0.5 cm margin

A real card is **63 x 88 mm**, so `CARD_RATIO = 63/88 = 0.7159`.

### Card size

`crop_card` measures the detected quad and produces a card with the true ratio:

```
out_h = clip(measured height, 400, 2000)
out_w = round(out_h * 0.7159)
```

`--size WxH` overrides this. By default the whole sheet shares one size (median
height of all six cards) so the output files are uniform; `--no-uniform` keeps
each card's own measured resolution instead.

### Margin

The margin is specified in real-world units and converted with the card's own
pixel width, so it is correct at any DPI:

```
px_per_mm = card_width_px / 63
margin_px = round(5 mm * px_per_mm)           # 0.5 cm default
```

Example at ≈200 DPI: card 493 px wide → `493/63 = 7.83 px/mm` → 0.5 cm → **39 px**
per side → final canvas `493 + 2*39 = 571` x `689 + 2*39 = 767`.

| Want | Flag |
|---|---|
| 0.5 cm (default) | – |
| 1 cm | `--margin-cm 1` |
| 3 cm | `--margin-cm 3` |
| 15 mm | `--margin-mm 15` |
| exact pixels | `--margin-px 200` |
| no margin | `--margin-px 0` |
| transparent margin | `--margin-transparent` (RGBA PNG, alpha 0 outside the card) |

### Why the margin makes misalignment harmless

`--expand 0.01` grows the sampling quad by 1% per side before warping. If the
detector lands a few pixels *inside* the card edge, the extra strip comes from the
surrounding paper instead of being cut off the card — and that slack disappears
into the margin, so the card itself is never clipped. `--expand 0.02` gives even
more room at the cost of a sliver of background along the edges.

---

## 8. Naming rules

Pattern: `Card_grid{grid}_card{card}.png`

| Part | Meaning | Range |
|---|---|---|
| `grid` | which scanned sheet | 1, 2, 3, … |
| `card` | position on the sheet | 1–6, reading order |

### How the grid number is chosen

1. If the sheet's filename contains a number (the last one — `shot_grid_12.png`
   → 12) **and** that grid number is not used yet → use it.
2. Otherwise → the next unused number after the highest existing grid.

```
cards\ has: grid1..grid4
scan_grids\sheet.png          -> grid 5   (no number in the name)
scan_grids\card_scan_grid_2.png -> grid 5 (2 is taken, so next free)
scan_grids\card_scan_grid_7.png -> grid 7 (free, used)
```

`--start-grid N` forces the next sheet to at least N.

### Never destructive

- `GridWriter.save` bumps the card index until it finds a free filename, so an
  existing `Card_grid1_card1.png` is never overwritten.
- If a sheet produces **zero** cards, `writer.abort()` releases the number, so the
  next sheet reuses it instead of leaving a hole.

### Reading order

`order_grid` sorts by vertical centre into rows, then each row left→right, so
`card1..card3` is the top row and `card4..card6` the bottom row in a 3x2 layout
(and `card1..card2`, `card3..card4`, `card5..card6` for 2x3).

---

## 9. Command line reference

```
python scan.py [input] [options]
```

| Flag | Default | Meaning |
|---|---|---|
| `input` | `scan_grids` | folder of sheets, or a single image |
| `-o, --out` | `cards` | output folder (created if missing) |
| `-r, --recursive` | off | also scan sub-folders |
| `--expect` | `6` | cards per sheet |
| `--cols` / `--rows` | auto | force the grid layout |
| `--no-fallback` | off | do not blind-split when the count is wrong |
| `--grid-inset` | `0.02` | trim fraction per cell in blind-split mode |
| `--margin-cm` | `0.5` | blank margin per side, in cm |
| `--margin-mm` | – | same, in mm (overrides `--margin-cm`) |
| `--margin-px` | – | same, in pixels (overrides both) |
| `--margin-transparent` | off | RGBA margin instead of white |
| `--expand` | `0.01` | grow the sample box so the card cannot be clipped |
| `--size WxH` | auto | force the card size (margin excluded) |
| `--no-uniform` | off | per-card measured size instead of one per sheet |
| `--start-grid` | auto | force the next grid number |
| `--dedupe` | off | skip cards that look identical to saved ones |
| `--debug DIR` | off | write annotated detection images |
| `--move-done DIR` | off | move each processed sheet into DIR |

Exit codes: `0` success, `1` at least one sheet failed to read/write, `2` nothing
to scan (used by `scan.bat` to print the "drop your scans in scan_grids" hint).

---

## 10. Tuning constants

`cardscan/detect.py` — top of the file:

| Constant | Value | Meaning |
|---|---|---|
| `CARD_W_MM` / `CARD_H_MM` | 63 / 88 | real card size, drives `CARD_RATIO` |
| `DETECT_MAX_SIDE` | 1600 | detection runs on a copy capped at this |
| `MIN_AREA_FRAC` | 0.004 | a card is at least 0.4% of the frame |
| `MAX_AREA_FRAC` | 0.60 | …and at most 60% |
| `MIN_CARD_SIDE` | 90 | px, on the detection copy |
| `ASPECT_MIN` / `ASPECT_MAX` | 0.45 / 0.96 | plausible min/max side ratio |
| `MIN_RECTANGULARITY` | 0.75 | contour area / quad area |
| `DEDUPE_OVERLAP` | 0.45 | candidates overlapping more than this are duplicates |

`cardscan/warp.py`:

| Constant | Value | Meaning |
|---|---|---|
| `MIN_OUT_H` / `MAX_OUT_H` | 400 / 2000 | clamp for auto output height |

`cardscan/naming.py`:

| Constant | Value | Meaning |
|---|---|---|
| `HASH_LOOKBACK` | 500 | how many existing cards `--dedupe` compares against |
| `dedupe_bits` | 4 | Hamming distance under which two cards are "the same" |

`scan.py`:

| Constant | Value | Meaning |
|---|---|---|
| `DEFAULT_IN` / `DEFAULT_OUT` | `scan_grids` / `cards` | default folders |
| `PREFIX` | `Card_` | output filename prefix |
| `IMAGE_EXTS` | jpg jpeg png bmp tif tiff webp | what counts as a sheet |

---

## 11. Failure modes and fallbacks

| Situation | What happens |
|---|---|
| Sheet perfectly aligned | `projection`, 6 cards, `source=canny10` etc. |
| One card pale, no outline | lattice rebuild, `source=cell` or `source=grid` |
| Cards touching | merged blob dropped by `_dedupe`, real cards survive |
| Faint frame around the whole block | handled by `RETR_LIST` + `paper_split` |
| Whole sheet rotated a few degrees | each card warped with its own corners |
| Detection finds 3 cards | blind grid split, 6 images, `source=grid` |
| Image unreadable | that sheet is skipped, counted in `failed`, exit code 1 |
| No images at all | exit code 2, `scan.bat` prints the "drop scans" hint |
| Output name already exists | index bumped, nothing overwritten |
| Sheet produced 0 cards | grid number released via `abort()` |

Diagnostic line printed per sheet:

```
  6 card(s) via projection (2x3) - 7 unique / 31 raw candidates
  ^kept  ^method      ^layout   ^after dedupe / ^raw contours from 6 masks
```

Use `--debug DIR` to get the sheet with the detected quads and `grid.card` labels
drawn on it — the fastest way to see *why* a card was missed.

---

## 12. Extending the app

| Goal | Where to change |
|---|---|
| Different card size (e.g. mini cards) | `CARD_W_MM` / `CARD_H_MM` in `detect.py` — margin follows automatically |
| More cards per sheet | `--expect 8` (layout is detected) |
| Different output naming | `GridWriter.path_for` in `naming.py` |
| Adaptive margin (e.g. 10% of card) | `scan.resolve_margin` |
| Save JPEG/WebP | `_imwrite` in `naming.py` |
| Pre-processing (deskew the whole sheet first) | start of `process_frame` in `scan.py` |
| Parallel processing | `run_files` — `concurrent.futures` around `process_frame` |
| Different detection strategy | add a mask in `_masks`, or a new branch in `detect_cards` |

The code is organised so that `detect.py` returns geometry, `warp.py` turns
geometry into pixels, and `naming.py` decides file names. Those three never
import each other's siblings except for shared constants, so any of them can be
replaced without touching the others.

---

## 13. Testing

```powershell
python tests_selftest.py
```

The self-test builds synthetic sheets with OpenCV (`make_card`, `paste`,
`build_sheet`) — coloured rectangles with white borders on a dark background,
which exercises the same code paths as a real scan.

| Test | Checks |
|---|---|
| `clean grid` | 6 detections, each crop lands on the right card (colour distance < 60) |
| `tilted cards` | random ±5° per-card rotation still crops correctly |
| `perspective` | sheet warped with a perspective transform still crops correctly |
| `fallback` | a featureless image produces 0 detections, and `grid_quads` yields 6 crops |
| `padding` | 0.5 cm margin → exact canvas size, margin pure white, card area untouched, RGBA variant transparent |
| `sheet rotated ±3° / -4.5°` | whole-sheet rotation handled |
| `grid naming` | hint, resume across runs, abort/reuse, `--start-grid` floor, `grid_hint` parsing |

Pass criteria: every crop must contain the card it was cut from. A wrong card in
a crop is a hard failure, not a warning.

---

## 14. File map

```
Auto Grid Scan TCG Card 3x2\
├── scan.py                CLI + per-sheet orchestration
├── scan.bat               double-click wrapper (folder batch / drag & drop)
├── requirements.txt       numpy, opencv-python
├── README.md              short user guide
├── DOCS.md                this document
├── tests_selftest.py      synthetic-sheet regression test
│
├── cardscan\
│   ├── __init__.py        public API re-exports
│   ├── detect.py          masks, contours, grid lattice, reading order
│   ├── warp.py            perspective warp, trimming, margin padding
│   └── naming.py          grid numbering, unicode-safe I/O, dedupe hash
│
├── scan_grids\            input:  drop scanned sheets here
├── cards\                 output: Card_grid{N}_card{M}.png
└── scanned\               processed sheets, moved out of the way (created on first run)
```
