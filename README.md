# Auto Grid Scan — TCG Card Sheets

Drop every scanned sheet into **`scan_grids\`**, double-click **`scan.bat`**, and
each sheet is cut into 6 individual cards in **`cards\`**:

```
scan_grids\sheet A.png   ->  cards\Card_grid1_card1.png ... Card_grid1_card6.png
scan_grids\sheet B.png   ->  cards\Card_grid2_card1.png ... Card_grid2_card6.png
```

`grid` = which scanned sheet, `card` = which card on that sheet (1–6, left→right,
top→bottom). Each card is straightened (perspective + rotation corrected) and
saved on a canvas with a **0.5 cm blank margin** on every side.

Grid numbers continue across runs, so nothing is ever overwritten.

Full technical documentation — libraries, module/function breakdown, the detection
pipeline and its fallbacks — is in **[DOCS.md](DOCS.md)**.

## Install

```powershell
pip install -r requirements.txt
```

Requires Python 3.9+ (tested on 3.15) with `opencv-python` and `numpy`.

## Use

Every run starts with the banner, then one line per sheet:

```
 _         _       ____  _
| |    ___(_)_ __ |  _ \(_)_ __
| |   / _ \ | '_ \| |_) | | '_ \
| |__|  __/ | | | |  _ <| | | | |
|_____\___|_|_| |_|_| \_\_|_| |_|

  TCG grid sheet splitter  |  6 cards per sheet  |  scan_grids -> cards
```

```powershell
# the normal flow: every sheet in scan_grids\ -> cards\
python scan.py

# one sheet, grid number taken from its filename ("...grid_7.png" -> grid 7)
python scan.py sheet.png

# any folder of sheets
python scan.py scans_sheet\ -o cards

# debug what the detector sees
python scan.py sheet.png --debug debug
```

Each saved image is the card **plus a 0.5 cm blank margin on every side**
(`--margin-cm`). The margin is computed from the real card size, so it is 0.5 cm
no matter what DPI the sheet was scanned at. A small safety expansion
(`--expand`) means a detection that lands a few pixels inside the card edge can
never clip the card — that slack ends up in the margin.

### Drag & drop / double click

- **Double-click `scan.bat`** → scans every image in `.\scan_grids`, writes
  `Card_grid{N}_card{M}.png` into `.\cards`, and moves the used sheets into
  `.\scanned` so a second double-click does not redo them.
- **Drag a sheet onto `scan.bat`** → scans just that file (it stays where it is).

## How it works

1. **Layout** — the sheet is found from a content profile: empty bands of
   background between cards become the row/column separators. This also gives
   the layout (a 2x3 portrait sheet and a 3x2 landscape sheet both work).
2. **Detect** — five masks (Otsu both polarities, adaptive threshold, CLAHE-based
   Canny at two scales, paper-relative split) produce contour candidates. Each is
   turned into a quadrilateral and filtered by area, card aspect ratio
   (63:88 ≈ 0.716) and rectangularity.
3. **Clean up** — duplicates between masks are merged by rotated-rect overlap;
   merged blobs that swallow two cards are dropped in favour of the real cards;
   the candidates closest to the median card area win.
4. **Fit** — each grid cell takes the best contour inside it (keeps perspective
   correction for tilted cards), falls back to the cell rectangle for pale cards
   whose outline is invisible. Rebuilt cells are edge-aligned against the lattice
   of the cards that were found.
5. **Warp** — perspective transform to a straight card, true 63:88 ratio, one
   uniform size per sheet. Flat paper strips left by a shadow are trimmed.
6. **Margin** — the card is centred on a canvas with a blank border (0.5 cm default,
   scaled from the real card width so the DPI does not matter).
7. **Save** — `Card_grid{N}_card{M}.png`, one grid number per sheet. The number
   comes from the sheet's filename when it contains one (and it is still free),
   otherwise from the next unused number. Nothing is overwritten.

Grid numbers are allocated per sheet, so a re-run of the same folder continues
after the last grid instead of clobbering it. If a sheet produces no card at
all, its number is released again.

If the layout or the detection cannot produce the expected number of cards, the
script falls back to a plain grid split sized to the frame (use `--no-fallback`
to disable).

## Options

| Flag | Default | Meaning |
|---|---|---|
| `-o, --out` | `cards` | output folder |
| `--expect` | `6` | cards per sheet |
| `--cols` / `--rows` | auto | force the grid layout (otherwise inferred) |
| `--margin-cm N` | `0.5` | blank margin around each card, in cm |
| `--margin-mm N` | – | same, in mm (overrides `--margin-cm`) |
| `--margin-px N` | – | same, in pixels (overrides both); `0` disables the margin |
| `--margin-transparent` | off | transparent margin (RGBA) instead of white |
| `--expand` | `0.01` | grow the sample box so a small detection cannot clip the card |
| `--size WxH` | auto | force the card size, margin excluded |
| `--no-uniform` | off | keep per-card measured resolution instead of one size per sheet |
| `--grid-inset` | `0.02` | trim fraction per cell in fallback mode |
| `--no-fallback` | off | save only what was detected |
| `--start-grid N` | auto | force the grid number of the next sheet |
| `--dedupe` | off | skip cards that look identical to already saved ones |
| `--debug DIR` | off | write annotated detection images |
| `--move-done DIR` | off | move each processed sheet into DIR |
| `-r, --recursive` | off | scan folders recursively |

Sizing example: a 200 DPI A4 scan gives cards of about 493x689 px, so a 0.5 cm
margin is 39 px per side and each PNG comes out 571x767 px.

## Self-test

```powershell
python tests_selftest.py
```

Generates synthetic 3x2 sheets (clean, tilted, whole-sheet rotated ±3–4.5°,
perspective-warped), runs the full detection + crop + margin pipeline, and
asserts every crop contains the right card, that the margin is exactly the
requested size, and that grid numbering (hint, resume, abort) behaves.

## Troubleshooting

- **Wrong count / grid fallback every time** — shoot straight down, keep the whole
  sheet inside the frame, leave a visible gap between cards and around the block,
  and use a background that contrasts with the card borders.
- **A card is slightly shifted / rotated** — handled: each card is warped with its
  own four corners and the 0.5 cm margin absorbs any leftover offset. If a card
  still looks cut, raise `--expand` (e.g. `0.02`).
- **Crops include a sliver of background** — that is the deliberate `--expand`
  slack; lower it to `0` for a tight cut.
- **Need more white space** — `--margin-cm 1` or `--margin-cm 3`.
- **Need no white space at all** — `--margin-px 0`.
- **Non-ASCII paths** — handled (uses `imdecode`/`tofile`, not `cv2.imread`).
