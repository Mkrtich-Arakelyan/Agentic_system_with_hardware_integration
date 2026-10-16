"""Fake scans of a trial sheet, only for tests and fake-printer dry runs (never for real scans).

fake_scan() renders each page of a trial sheet PDF in grayscale, as if it had been printed and
scanned, with an optional small rotation, Gaussian noise and darkening of the whole page (paper
included, like darker paper or a dim scan). It saves one PNG per page that read_scan /
evaluate_scan accept.
"""
import math
from pathlib import Path

import numpy as np
import pymupdf
from PIL import Image

MAX_ROTATION_DEG = 5.0             # a scan lying more crooked than this is not a useful stand-in


def fake_scan(pdf_path, out_dir, *, dpi: int = 300, rotate_deg: float = 0.0, noise: float = 0.0,
              darken: float = 0.0, seed: int = 0) -> list[str]:
    """Save scan_page_<n>.png (n from 1) for every page of pdf_path in out_dir; return the paths in page order.

    dpi: render resolution, also written into each PNG. rotate_deg: rotation in degrees (white fill).
    noise: standard deviation of Gaussian noise on the 0..1 gray scale. darken: pulls every gray of
    the whole page, blank paper included, toward black by this fraction (0 <= darken < 1), like
    darker paper or a dim scan. seed: makes the noise repeatable.
    Raises ValueError for bad arguments, an unreadable PDF or a PNG that already exists.
    """
    if isinstance(dpi, bool) or not isinstance(dpi, (int, float)) or not dpi > 0:
        raise ValueError(f"dpi must be a positive number, got {dpi!r}")
    for name, value in (("rotate_deg", rotate_deg), ("noise", noise), ("darken", darken)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{name} must be a finite number, got {value!r}")
    if not abs(rotate_deg) <= MAX_ROTATION_DEG:
        raise ValueError(f"rotate_deg must be within ±{MAX_ROTATION_DEG} degrees, got {rotate_deg!r}")
    if not noise >= 0:
        raise ValueError(f"noise must be >= 0, got {noise!r}")
    if not 0 <= darken < 1:
        raise ValueError(f"darken must be in [0, 1), got {darken!r}")
    try:
        document = pymupdf.open(pdf_path)
    except Exception as error:
        raise ValueError(f"cannot open the trial sheet PDF {pdf_path}: {error}") from error

    out_dir = Path(out_dir)
    rng = np.random.default_rng(seed)
    with document:
        paths = [out_dir / f"scan_page_{number}.png" for number in range(1, document.page_count + 1)]
        existing = [path.name for path in paths if path.exists()]
        if existing:
            raise ValueError(f"{out_dir} already holds {existing}; fake_scan never overwrites, use another folder")
        out_dir.mkdir(parents=True, exist_ok=True)
        for page, path in zip(document, paths):
            pixmap = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY, alpha=False)
            image = Image.frombytes("L", (pixmap.width, pixmap.height), pixmap.samples, "raw", "L", pixmap.stride)
            if rotate_deg:
                image = image.rotate(rotate_deg, resample=Image.BICUBIC, fillcolor=255)
            gray = np.asarray(image, dtype=np.float64) / 255 * (1 - darken)
            if noise:
                gray = gray + rng.normal(0.0, noise, gray.shape)
            pixels = np.clip(np.round(gray * 255), 0, 255).astype(np.uint8)
            Image.fromarray(pixels, "L").save(path, dpi=(dpi, dpi))
    return [str(path) for path in paths]
