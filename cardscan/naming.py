"""Output naming: Card_grid{grid}_card{card}.png

One scanned sheet = one grid. Grid numbers continue from what is already on
disk, so nothing is ever overwritten.
"""

from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np

HASH_LOOKBACK = 500  # how many existing cards to compare against when deduping
GRID_RE = re.compile(r"(\d+)(?!.*\d)")  # last number in a file name


def dhash(img: np.ndarray, size: int = 8) -> np.ndarray:
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    g = cv2.resize(g, (size + 1, size), interpolation=cv2.INTER_AREA)
    return (g[:, 1:] > g[:, :-1]).astype(np.uint8).ravel()


def grid_hint(path: str | Path | None) -> int | None:
    """Grid number suggested by a sheet's file name (last number in it)."""
    if path is None:
        return None
    m = GRID_RE.search(Path(path).stem)
    if not m:
        return None
    value = int(m.group(1))
    return value if 1 <= value <= 9999 else None


class GridWriter:
    """Card_grid{grid}_card{card}.png, grid numbers continue across runs."""

    def __init__(
        self,
        out_dir: str | Path,
        prefix: str = "Card_",
        start_grid: int | None = None,
        dedupe: bool = False,
        dedupe_bits: int = 4,
    ) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix
        self.dedupe = dedupe
        self.dedupe_bits = dedupe_bits
        self._re = re.compile(rf"^{re.escape(prefix)}grid(\d+)_card(\d+)\.png$", re.IGNORECASE)

        used: set[int] = set()
        for p in self._existing():
            m = self._re.match(p.name)
            if m:
                used.add(int(m.group(1)))
        self._used = used
        self._next = (max(used) + 1) if used else 1
        if start_grid:
            self._next = max(self._next, start_grid)
        self._grid: int | None = None
        self._hashes: list[np.ndarray] = self._load_hashes() if dedupe else []

    # ------------------------------------------------------------------ state

    @property
    def next_grid(self) -> int:
        return self._next

    def begin(self, hint: int | None = None) -> int:
        """Claim the grid number for the sheet about to be saved."""
        if hint is not None and hint >= 1 and hint not in self._used:
            grid = hint
        else:
            grid = self._next
            while grid in self._used:
                grid += 1
            self._next = grid + 1
        self._used.add(grid)
        self._grid = grid
        return grid

    def abort(self) -> None:
        """Release the current grid number (nothing was written)."""
        if self._grid is not None:
            self._used.discard(self._grid)
            if self._grid < self._next:
                self._next = self._grid
            self._grid = None

    # ------------------------------------------------------------------- save

    def path_for(self, card: int, grid: int | None = None) -> Path:
        grid = self._grid if grid is None else grid
        if grid is None:
            raise RuntimeError("begin() a grid before saving")
        return self.out_dir / f"{self.prefix}grid{grid}_card{card}.png"

    def save(
        self,
        img: np.ndarray,
        card_index: int,
        hash_img: np.ndarray | None = None,
    ) -> tuple[Path | None, bool]:
        """Save one card. Returns (path, skipped_as_duplicate)."""
        if self._grid is None:
            raise RuntimeError("begin() a grid before saving")
        if self._duplicate(img if hash_img is None else hash_img):
            return None, True

        path = self.path_for(card_index)
        guard = 0
        while path.exists():  # never overwrite
            card_index += 1
            path = self.path_for(card_index)
            guard += 1
            if guard > 999:
                raise OSError(f"no free card slot for grid {self._grid}")
        if not _imwrite(path, img):
            raise OSError(f"failed to write {path}")
        return path, False

    # -------------------------------------------------------------- internals

    def _existing(self) -> list[Path]:
        return [p for p in self.out_dir.iterdir() if p.is_file() and self._re.match(p.name)]

    def _load_hashes(self) -> list[np.ndarray]:
        paths = sorted(self._existing(), key=lambda p: p.stat().st_mtime, reverse=True)
        out: list[np.ndarray] = []
        for p in paths[:HASH_LOOKBACK]:
            img = _imread(p)
            if img is not None:
                out.append(dhash(img))
        return out

    def _duplicate(self, hash_img: np.ndarray) -> bool:
        if not self.dedupe:
            return False
        h = dhash(hash_img)
        if any(int(np.count_nonzero(h != old)) <= self.dedupe_bits for old in self._hashes):
            return True
        self._hashes.append(h)
        return False


def _imread(path: str | Path) -> np.ndarray | None:
    """Unicode-safe imread (cv2.imread breaks on non-ASCII Windows paths)."""
    try:
        buf = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if buf.size == 0:
        return None
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def _imwrite(path: str | Path, img: np.ndarray) -> bool:
    ok, buf = cv2.imencode(".png", img, [cv2.IMWRITE_PNG_COMPRESSION, 6])
    if not ok:
        return False
    try:
        buf.tofile(str(path))
    except OSError:
        return False
    return True
