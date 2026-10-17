"""Render all PDF pages for visual review, without altering the source artifact."""
import argparse
from pathlib import Path
import pymupdf


def render(pdf, out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    with pymupdf.open(pdf) as document:
        for index, page in enumerate(document):
            path = out / f"page_{index + 1}.png"
            if path.exists(): raise ValueError("Preview exists; choose a new directory")
            page.get_pixmap(dpi=140, alpha=False).save(path)
            sizes = [span["size"] for block in page.get_text("dict")["blocks"]
                     for line in block.get("lines", []) for span in line["spans"]]
            print(f"Page {index + 1}: {page.rect.width:.1f} x {page.rect.height:.1f} pt; minimum text {min(sizes):.2f} pt; {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, required=True); parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(); render(args.pdf, args.out)
