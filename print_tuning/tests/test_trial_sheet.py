"""Tests for print_tuning/trial_sheet.py.

Run with `python -m pytest print_tuning/tests`, or without pytest as
`python print_tuning/tests/test_trial_sheet.py`.
"""
import hashlib
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import pymupdf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # repo root, for running as a script

from print_tuning.grayscale_render import default_mapping, load_config
from print_tuning.trial_sheet import TrialSheetError, create_trial_sheet

DPI = 150
PANEL_NAMES = ("Initial", "Final", "Difference")
GAMMA_TESTS = (0.4, 0.6, 1.5, 2.2, 3.0)
GRAY_TESTS = ((0.0, 0.9), (0.05, 1.0), (0.1, 0.9), (0.05, 0.95), (0.15, 0.85))


class raises:
    """Like pytest.raises, so the tests also run without pytest."""

    def __init__(self, error):
        self.error = error

    def __enter__(self):
        return self

    def __exit__(self, kind, value, traceback):
        if kind is None:
            raise AssertionError(f"{self.error.__name__} was not raised")
        return issubclass(kind, self.error)


def make_selection(folder, panel_counts, seed=0):
    """One item per entry of panel_counts, each panel a random 32 x 32 array with a focus box mark."""
    rng = np.random.default_rng(seed)
    items = []
    for index, count in enumerate(panel_counts):
        panels = []
        for name in PANEL_NAMES[:count]:
            path = Path(folder) / f"item{index}_{name}.npy"
            np.save(path, rng.normal(100.0, 20.0, size=(32, 32)))
            panels.append({"name": name, "kind": "heatmap", "data_path": str(path),
                           "marks": [{"type": "box", "row": 16, "col": 16, "size": 6}]})
        items.append({"item_id": f"exp_{index}", "label": f"Experiment {index}", "panels": panels})
    return {"run_id": "run_test", "selection_id": "sel_test", "selection_version": 1, "items": items}


def round1_variations(selection, config):
    base = default_mapping(config)
    return {item["item_id"]: [{"variation_id": "ref", "role": "reference", "mapping": base, "note": "best"}] +
            [{"variation_id": f"g{i}", "role": "test", "mapping": base | {"gamma": gamma}, "note": ""}
             for i, gamma in enumerate(GAMMA_TESTS)]
            for item in selection["items"]}


def round2_variations(selection, config, gray_tests=GRAY_TESTS):
    base = default_mapping(config) | {"gamma": 0.6}
    return {item["item_id"]: [{"variation_id": "ref", "role": "reference", "mapping": base, "note": "best"}] +
            [{"variation_id": f"l{i}", "role": "test", "note": "",
              "mapping": base | {"dark_level": dark, "light_level": light}}
             for i, (dark, light) in enumerate(gray_tests)]
            for item in selection["items"]}


def render_pages(pdf_path):
    images = []
    with pymupdf.open(pdf_path) as document:
        for page in document:
            pixmap = page.get_pixmap(dpi=DPI, colorspace=pymupdf.csGRAY, alpha=False)
            samples = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(pixmap.height, pixmap.stride)
            images.append(samples[:, :pixmap.width] / 255.0)
    return images


def region(image, box, inset):
    """Pixels of image inside an mm box, shrunk by inset mm on every side."""
    x, y, w, h = box
    px = DPI / 25.4
    return image[int(round((y + inset) * px)):int(round((y + h - inset) * px)),
                 int(round((x + inset) * px)):int(round((x + w - inset) * px))]


def overlaps(a, b):
    return a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]


def test_four_items_round1_pages_regions_and_hash():
    config = load_config()
    with tempfile.TemporaryDirectory() as folder:
        selection = make_selection(folder, [2, 2, 2, 2])
        record = create_trial_sheet(selection, round1_variations(selection, config), 1, folder, "cal_1")
        assert record["schema_version"] == "trial/1" and record["tested_parameter"] == "gamma"
        assert record["pdf"]["pages"] == 2 and len(record["pages"]) == 2
        for page in record["pages"]:
            assert len(page["regions"]["tiles"]) == 12 and len(page["regions"]["plots"]) == 24
            assert len(page["regions"]["markers"]) == 4 and len(page["regions"]["gray_patches"]) == 11
        assert [page["page_code"] for page in record["pages"]] == [[0, 0, 1, 0, 0, 1], [0, 1, 0, 0, 0, 1]]
        pdf = Path(folder) / "trial_r1.pdf"
        assert hashlib.sha256(pdf.read_bytes()).hexdigest() == record["pdf"]["sha256"]
        saved = json.loads((Path(folder) / "trial_r1_manifest.json").read_text(encoding="utf-8"))
        assert saved["trial_id"] == record["trial_id"]
        with pymupdf.open(pdf) as document:
            assert document.page_count == 2
            assert abs(document[0].rect.width - 210 / 25.4 * 72) < 0.5
            assert abs(document[0].rect.height - 297 / 25.4 * 72) < 0.5
        with np.load(record["cells_path"]) as cells:
            assert "exp_0/data/Initial" in cells and "exp_3/g4/Final" in cells


def test_one_and_three_panels_work():
    config = load_config()
    with tempfile.TemporaryDirectory() as folder:
        selection = make_selection(folder, [1, 3])
        record = create_trial_sheet(selection, round1_variations(selection, config), 1, folder, "cal_1")
        assert record["pdf"]["pages"] == 1
        assert len(record["pages"][0]["regions"]["plots"]) == 6 * 1 + 6 * 3
        check_printed_page(record)


def check_printed_page(record):
    """Measure the rendered PDF at the manifest positions."""
    items = {item["item_id"]: item for item in record["items"]}
    with np.load(record["cells_path"]) as cells:
        for image, page in zip(render_pages(record["pdf"]["path"]), record["pages"]):
            regions = page["regions"]
            assert region(image, regions["black_patch"]["box"], 1.0).mean() < 0.15
            for marker in regions["markers"]:
                assert region(image, marker["box"], 1.0).mean() < 0.15
            for paper in regions["paper_refs"]:
                assert region(image, paper["box"], 1.0).mean() > 0.95
            for patch in regions["gray_patches"]:
                assert abs(region(image, patch["box"], 2.0).mean() - patch["level"]) <= 0.06, patch
            for plot in regions["plots"]:
                grays = cells[f"{plot['item_id']}/{plot['variation_id']}/{plot['panel']}"]
                marks = next(p["marks"] for p in items[plot["item_id"]]["panels"] if p["name"] == plot["panel"])
                printed, expected = plot_pixels(image, plot, grays, marks)
                assert abs(printed.mean() - expected.mean()) <= 0.06, plot

            # nothing written over plots, scale bars, patches, paper areas or markers
            keep_clear = ([r["box"] for key in ("plots", "scale_bars", "paper_refs", "gray_patches", "markers",
                                                "page_code") for r in regions[key]] +
                          [regions["black_patch"]["box"]])
            for label in regions["labels"]:
                for box in keep_clear:
                    assert not overlaps(label["box"], box), (label, box)


def plot_pixels(image, plot, grays, marks):
    """Printed pixels inside the plot, away from its edge and box marks, and the cell value under each."""
    x, y, w, h = plot["box"]
    rows, cols = grays.shape
    px = DPI / 25.4
    ys = (np.arange(image.shape[0]) + 0.5) / px
    xs = (np.arange(image.shape[1]) + 0.5) / px
    inside_y = np.where((ys > y + 0.5) & (ys < y + h - 0.5))[0]
    inside_x = np.where((xs > x + 0.5) & (xs < x + w - 0.5))[0]
    cell_r = ((ys[inside_y] - y) / h * rows).astype(int)
    cell_c = ((xs[inside_x] - x) / w * cols).astype(int)
    keep = np.ones((rows, cols), dtype=bool)
    for mark in marks:   # drop the box outline and two cells around it
        half = mark["size"] / 2 + 2
        r0, r1 = int(mark["row"] + 0.5 - half), int(np.ceil(mark["row"] + 0.5 + half))
        c0, c1 = int(mark["col"] + 0.5 - half), int(np.ceil(mark["col"] + 0.5 + half))
        keep[max(r0, 0):r1, max(c0, 0):c1] = False
    rr, cc = np.meshgrid(cell_r, cell_c, indexing="ij")
    selected = keep[rr, cc]
    printed = image[np.ix_(inside_y, inside_x)][selected]
    return printed, grays[rr, cc][selected]


def test_printed_page_matches_manifest():
    config = load_config()
    with tempfile.TemporaryDirectory() as folder:
        selection = make_selection(folder, [2, 2, 2, 2])
        record = create_trial_sheet(selection, round1_variations(selection, config), 1, folder, "cal_1")
        check_printed_page(record)


def test_bad_inputs_raise_trial_sheet_error():
    config = load_config()
    with tempfile.TemporaryDirectory() as folder:
        selection = make_selection(folder, [2, 2])
        good1 = round1_variations(selection, config)

        short = {key: value[:5] for key, value in good1.items()}
        with raises(TrialSheetError):
            create_trial_sheet(selection, short, 1, folder, "cal_1")

        two_refs = round1_variations(selection, config)
        two_refs["exp_0"][1]["role"] = "reference"
        with raises(TrialSheetError):
            create_trial_sheet(selection, two_refs, 1, folder, "cal_1")

        changes_gamma = round2_variations(selection, config)
        changes_gamma["exp_1"][2]["mapping"] = changes_gamma["exp_1"][2]["mapping"] | {"gamma": 1.5}
        with raises(TrialSheetError):
            create_trial_sheet(selection, changes_gamma, 2, folder, "cal_1")

        different = round2_variations(selection, config)
        other = round2_variations(selection, config, GRAY_TESTS[:-1] + ((0.2, 0.8),))
        different["exp_1"] = other["exp_1"]
        with raises(TrialSheetError):
            create_trial_sheet(selection, different, 2, folder, "cal_1")

        with raises(TrialSheetError):
            create_trial_sheet(selection, good1, 9, folder, "cal_1")

        np.save(selection["items"][0]["panels"][0]["data_path"], np.full((32, 32), np.nan))
        with raises(TrialSheetError):
            create_trial_sheet(selection, good1, 1, folder, "cal_1")
        assert not (Path(folder) / "trial_r1.pdf").exists()


def test_returned_limits_are_kept_in_next_round():
    config = load_config()
    with tempfile.TemporaryDirectory() as folder:
        selection = make_selection(folder, [2, 2, 1])
        first = create_trial_sheet(selection, round1_variations(selection, config), 1, folder, "cal_1")
        limits = {item["item_id"]: item["limits"] for item in first["items"]}

        path = selection["items"][0]["panels"][0]["data_path"]   # data changes must not move the limits
        np.save(path, np.load(path) * 3.0)
        second = create_trial_sheet(selection, round2_variations(selection, config), 2, folder, "cal_1",
                                    limits=limits)
        assert {item["item_id"]: item["limits"] for item in second["items"]} == limits
        assert second["tested_parameter"] == "gray_levels"


if __name__ == "__main__":
    tests = [(name, test) for name, test in list(globals().items()) if name.startswith("test_")]
    failed = 0
    for name, test in tests:
        try:
            test()
            print(f"PASS {name}")
        except Exception as error:
            failed += 1
            print(f"FAIL {name}: {type(error).__name__}: {error}")
    print(f"{len(tests) - failed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
