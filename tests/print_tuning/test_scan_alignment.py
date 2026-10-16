"""Representative scanner geometry and bilevel-marker regressions; no private scans."""

from copy import deepcopy
import json

import numpy as np
from PIL import Image, ImageDraw
import pytest

from app.print_tuning import scan_evaluation as scan

PPMM = 6
DPI = PPMM * 25.4


@pytest.fixture
def page():
    # The trial producer's real 8 mm markers and six-bit page-code geometry.
    markers = [{"corner": c, "box": b} for c, b in zip(scan.IMAGE_CORNERS,
        ([10, 10, 8, 8], [192, 10, 8, 8], [10, 279, 8, 8], [192, 279, 8, 8]))]
    bits = [0, 0, 1, 0, 0, 1]
    code = [{"bit": bit, "box": [20 + 4 * i, 12.5, 3, 3]} for i, bit in enumerate(bits)]
    regions = {"markers": markers, "page_code": code, "plots": [{"box": [30, 60, 20, 20]}]}
    manifest = [{"page_number": 1, "page_code": bits, "regions": regions}]
    pixels = np.full((297 * PPMM, 210 * PPMM), 255, np.uint8)
    for marker in markers:
        x, y, w, h = (int(v * PPMM) for v in marker["box"])
        # The actual failing PDFs are bilevel scans whose lower markers average
        # about .51-.53 gray. A checkerboard reproduces the .35 absolute-gray failure.
        yy, xx = np.indices((h, w))
        pixels[y:y+h, x:x+w] = ((xx + yy) % 2 * 255).astype(np.uint8)
    for square in code:
        x, y, w, h = (int(v * PPMM) for v in square["box"])
        if square["bit"]:
            pixels[y:y+h, x:x+w] = 0
    # Give the crop an asymmetric visible feature to verify rotation/translation.
    pixels[62 * PPMM:68 * PPMM, 32 * PPMM:36 * PPMM] = 80
    return Image.fromarray(pixels), manifest


def align(image, manifest):
    gray, rgb = scan._gray_and_rgb(image)
    return scan._align_page(gray, rgb, DPI, manifest, 0)


def test_halftoned_markers_pass_shape_contrast_and_code_checks(page):
    image, manifest = page
    result = align(image, manifest)
    assert result["page_code_read"] == manifest[0]["page_code"]
    assert result["fit_error_mm"] < .15
    assert all(scan._region(result["image"], marker["box"], 1).mean() > .35
               for marker in manifest[0]["regions"]["markers"])
    assert scan._region(result["image"], [32, 62, 4, 6], .5).mean() == pytest.approx(80/255, abs=.02)


@pytest.mark.parametrize("angle", [0, 90, 180, 270])
def test_quarter_turns_use_page_code_and_preserve_manifest_crop(page, angle):
    image, manifest = page
    result = align(image.rotate(-angle, expand=True), manifest)
    assert result["orientation"] == angle
    assert scan._region(result["image"], [32, 62, 4, 6], .5).mean() == pytest.approx(80/255, abs=.02)
    assert scan._estimate_dpi(np.zeros((image.width, image.height))) == pytest.approx(DPI)


@pytest.mark.parametrize("scale,angle,offset", [(.85, 1.0, (60, 75)), (1.08, -1.5, (30, 20))])
def test_scanner_margins_translation_small_rotation_and_scale(page, scale, angle, offset):
    image, manifest = page
    resized = image.resize((round(image.width * scale), round(image.height * scale)))
    canvas = Image.new("L", (resized.width + 120, resized.height + 120), 255)
    canvas.paste(resized, offset)
    result = align(canvas.rotate(angle, resample=Image.Resampling.BICUBIC, fillcolor=255), manifest)
    assert result["fit_error_mm"] < .5
    assert result["page_code_read"] == manifest[0]["page_code"]
    assert scan._region(result["image"], [32, 62, 4, 6], 1).mean() == pytest.approx(80/255, abs=.03)


@pytest.mark.parametrize("problem", ["missing", "outline", "rectangle", "clipped", "faint", "misplaced"])
def test_invalid_marker_evidence_is_rejected(page, problem):
    image, manifest = page
    draw = ImageDraw.Draw(image)
    draw.rectangle((60, 60, 107, 107), fill=255)
    if problem == "outline":
        draw.rectangle((60, 60, 107, 107), outline=0, width=3)
    elif problem == "rectangle":
        draw.rectangle((60, 60, 83, 107), fill=0)
    elif problem == "clipped":
        draw.rectangle((0, 60, 35, 107), fill=0)
    elif problem == "faint":
        draw.rectangle((60, 60, 107, 107), fill=235)
    elif problem == "misplaced":
        draw.rectangle((100, 60, 147, 107), fill=0)
    with pytest.raises(scan.ScanError) as error:
        align(image, manifest)
    assert error.value.code in ("markers_not_found", "page_mismatch")


def test_wrong_page_code_is_not_accepted_even_with_four_valid_markers(page):
    image, manifest = page
    wrong = deepcopy(manifest)
    wrong[0]["page_code"] = [1, 1, 1, 1, 1, 1]
    with pytest.raises(scan.ScanError) as error:
        align(image, wrong)
    assert error.value.code == "page_mismatch"


def test_displaced_fourth_square_cannot_bypass_geometric_fit_with_valid_page_code(page):
    image, manifest = page
    draw = ImageDraw.Draw(image)
    draw.rectangle((192 * PPMM, 279 * PPMM, 200 * PPMM - 1, 287 * PPMM - 1), fill=255)
    draw.rectangle((188 * PPMM, 279 * PPMM, 196 * PPMM - 1, 287 * PPMM - 1), fill=0)
    with pytest.raises(scan.ScanError, match="corner markers fit only within") as error:
        align(image, manifest)
    assert error.value.code == "markers_not_found"


def test_ambiguous_corner_squares_are_not_silently_selected(page):
    image, _ = page
    draw = ImageDraw.Draw(image)
    draw.rectangle((60, 60, 107, 107), fill=255)
    for x, y in ((10, 23), (23, 10)):
        draw.rectangle((x * PPMM, y * PPMM, (x + 8) * PPMM - 1, (y + 8) * PPMM - 1), fill=0)
    gray, _ = scan._gray_and_rgb(image)
    with pytest.raises(scan.ScanError, match="ambiguous"):
        scan._find_marker(gray, "top_left", PPMM)


def test_diagnostic_overlay_records_rejected_candidates_and_never_changes_sources(page, tmp_path):
    image, pages = page
    original = tmp_path / "scan.png"
    image.save(original)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"trial_id": "diagnostic-trial", "pages": pages,
                                   "config": {"scan": {"accepted_formats": ["png"]}}}))
    before = original.read_bytes(), manifest.read_bytes()
    report = scan.diagnose_scan_alignment(original, manifest, tmp_path / "diagnostics")
    record = report["pages"][0]
    assert record["status"] == "ok"
    assert all(any(c.get("selected") for c in values) for values in record["candidates"].values())
    assert any(c["reason"].startswith("rejected") for values in record["candidates"].values() for c in values)
    assert Image.open(record["overlay_path"]).width > image.width
    assert (original.read_bytes(), manifest.read_bytes()) == before
