import csv, json, os, hashlib
from pathlib import Path
import fitz
from PIL import Image

OUT = Path('/Users/razmik/Documents/Agentic_system_with_hardware_integration/runs/20261017_205142_877726')
REPORT = OUT / 'report.pdf'
CSV_PATH = OUT / 'experiments.csv'
FIGURE = OUT / 'print_tuning/final_heatmaps_467c6601.png'
SELECTED = [
 ('experiment_18848d544a654115a84d698c48e80b3a', OUT / 'experiment_18848d544a654115a84d698c48e80b3a/results.json'),
 ('experiment_0a53b0d22dca48bbb95c5cd1eea54236', OUT / 'experiment_0a53b0d22dca48bbb95c5cd1eea54236/results.json'),
 ('experiment_92f251efb4d4474ebbfd2e37a03be6fc', OUT / 'experiment_92f251efb4d4474ebbfd2e37a03be6fc/results.json'),
 ('experiment_2f5f3e6a9c104597979ec1175375ede4', OUT / 'experiment_2f5f3e6a9c104597979ec1175375ede4/results.json'),
]

# Load the shared experiment table and the four frozen, scientist-selected records.
with CSV_PATH.open(newline='', encoding='utf-8-sig') as f:
    csv_rows = list(csv.DictReader(f))
if not csv_rows:
    raise RuntimeError('experiments.csv has no recorded rows')
raw_csv = CSV_PATH.read_text(encoding='utf-8-sig')
records = []
for expected_id, result_path in SELECTED:
    if expected_id not in raw_csv:
        raise RuntimeError(f'{expected_id} is absent from experiments.csv')
    result = json.loads(result_path.read_text(encoding='utf-8'))
    if result.get('experiment_id') != expected_id:
        raise RuntimeError(f'results.json ID mismatch for {expected_id}')
    settings = result.get('settings', {})
    if settings.get('macropixel_size') is None or result.get('final_enhancement') is None:
        raise RuntimeError(f'Missing recorded value for {expected_id}')
    records.append({
        'id': expected_id,
        'size': int(settings['macropixel_size']),
        'enhancement': float(result['final_enhancement']),
        'iterations': settings.get('iterations'),
        'shots': settings.get('shots_per_measurement'),
        'validation': settings.get('validation_shots'),
    })

im = Image.open(FIGURE)
im.verify()
im = Image.open(FIGURE)
img_w, img_h = im.size
if not (img_w > 0 and img_h > 0):
    raise RuntimeError('Invalid supplied final figure')

# A4 portrait in points. Use the supplied PNG directly as a PDF image XObject.
page_w, page_h = 595.276, 841.890
doc = fitz.open()
page = doc.new_page(width=page_w, height=page_h)
page.draw_rect(page.rect, color=(1,1,1), fill=(1,1,1), overlay=True)
black = (0.08, 0.08, 0.08)
gray = (0.38, 0.38, 0.38)
rule = (0.72, 0.72, 0.72)

# Text helper: fail rather than silently clipping any block.
def textbox(rect, text, size=10, font='tiro', color=black, align=0, lineheight=1.15):
    spare = page.insert_textbox(fitz.Rect(*rect), text, fontname=font, fontsize=size,
                                color=color, align=align, lineheight=lineheight, overlay=True)
    if spare < -0.5:
        raise RuntimeError(f'Text overflow ({spare:.1f} pt): {text[:80]}')
    return spare

L, R = 37, page_w-37
textbox((L, 25, R, 52), 'Macropixel scale and focusing performance', 18, 'tibo', black, lineheight=1.0)
textbox((L, 54, R, 72), 'Research question  |  How does control-block size affect validated focusing enhancement?', 10, 'tiro', gray, lineheight=1.0)
findings = ('On this fixed simulated bench, validated enhancement declined as macropixel size increased from 1 to 8. '
            'All runs used identical focus/background regions and 1,000 iterations, while coarser blocks provide fewer controllable regions. '
            'Enhancement is the focus mean divided by the equally weighted mean of four background regions.')
textbox((L, 77, R, 132), findings, 10.1, 'tiro', black, lineheight=1.17)
page.draw_line(fitz.Point(L, 138), fitz.Point(R, 138), color=rule, width=0.7)

# Preserve native PNG pixel content and aspect ratio; only its placement scale changes.
max_w, max_h = 360.0, 400.0
scale = min(max_w / img_w, max_h / img_h)
figure_w, figure_h = img_w * scale, img_h * scale
fx = (page_w - figure_w) / 2
fy = 145.0
fig_rect = fitz.Rect(fx, fy, fx + figure_w, fy + figure_h)
page.insert_image(fig_rect, filename=str(FIGURE), keep_proportion=True, overlay=True)
cap_y = fy + figure_h + 3
textbox((L, cap_y, R, cap_y + 15),
        'Initial / Final camera heatmaps; experiments ordered by macropixel size 1, 2, 4, 8. Grayscale tuning changes presentation only, not the underlying measurements.',
        8.7, 'tiro', gray, lineheight=1.0)

# Four-row comparison table, preserving the frozen selection order.
table_top = cap_y + 21
textbox((L, table_top, R, table_top + 15), 'COMPARISON  |  common search settings: seed 1; 3 × 3 focus; 9 × 9 background', 10, 'tibo', black, lineheight=1.0)
y0 = table_top + 19
col_x = [L, L+83, L+185, L+294, R]
headers = ['Macropixel', 'Final enhancement', 'Search budget', 'Captures']
header_y = y0
for i, label in enumerate(headers):
    textbox((col_x[i]+3, header_y, col_x[i+1]-3, header_y+15), label, 8.7, 'tibo', gray, lineheight=1.0)
page.draw_line(fitz.Point(L, header_y+16), fitz.Point(R, header_y+16), color=rule, width=0.65)
row_h = 17
for idx, rec in enumerate(records):
    ry = header_y + 18 + idx*row_h
    vals = [f"{rec['size']} × {rec['size']}", f"{rec['enhancement']:.3f}",
            f"{rec['iterations']} iterations" if rec['iterations'] is not None else 'not recorded',
            (f"{rec['shots']} shots/measurement; {rec['validation']} validation" if rec['shots'] is not None and rec['validation'] is not None else 'not recorded')]
    for i, val in enumerate(vals):
        textbox((col_x[i]+3, ry, col_x[i+1]-3, ry+14), val, 8.8, 'tiro', black, lineheight=1.0)
    if idx < len(records)-1:
        page.draw_line(fitz.Point(L, ry+15), fitz.Point(R, ry+15), color=(0.88,0.88,0.88), width=0.4)
last_row_bottom = header_y + 18 + len(records)*row_h
page.draw_line(fitz.Point(L, last_row_bottom), fitz.Point(R, last_row_bottom), color=rule, width=0.65)

verdict_y = last_row_bottom + 8
textbox((L, verdict_y, R, verdict_y+15), 'VERDICT AND NEXT STEP', 10.2, 'tibo', black, lineheight=1.0)
verdict = ('The comparison is fair at equal absolute iteration and capture settings, but not per control degree of freedom: sizes 1 and 2 receive about 3.9 and 15.6 iterations per block, versus 62.5 and 250 for sizes 4 and 8. '
           'Size 1 gave the highest observed validated enhancement (28.268), and size 8 the lowest (1.160); this single search seed does not establish robustness. '
           'Recorded cost is 1,000 iterations and 3 shots per search measurement plus 5 validation shots in every run; wall time and total camera-shot counts were not recorded. '
           'Next, repeat size 1 with search seed 2 and all other settings fixed to test trajectory sensitivity.')
textbox((L, verdict_y+17, R, 805), verdict, 9.5, 'tiro', black, lineheight=1.12)
page.draw_line(fitz.Point(L, 814), fitz.Point(R, 814), color=rule, width=0.5)
textbox((L, 817, R, 833), 'Simulation results; relative intensity units. The printed figure is a tuned grayscale rendering, not a new measurement.', 8.2, 'tiro', gray, lineheight=1.0)

REPORT.parent.mkdir(parents=True, exist_ok=True)
doc.set_metadata({'title':'Macropixel scale and focusing performance','author':'Scientific brief'})
doc.save(REPORT, garbage=4, deflate=True)
doc.close()

# Validate A4, one page, and that the exact supplied figure is present as an image.
check = fitz.open(REPORT)
if check.page_count != 1:
    raise RuntimeError(f'Report must be exactly one page, got {check.page_count}')
p = check[0]
if abs(p.rect.width-page_w)>1 or abs(p.rect.height-page_h)>1:
    raise RuntimeError('Report page is not A4 portrait')
imgs = p.get_images(full=True)
if not imgs:
    raise RuntimeError('No embedded figure image found')
# Ensure the original image dimensions, and pixel payload, are preserved in the embedded image.
matched = False
for info in imgs:
    xref = info[0]
    extracted = check.extract_image(xref)
    if (extracted.get('width'), extracted.get('height')) == (img_w, img_h):
        try:
            import io
            embedded = Image.open(io.BytesIO(extracted['image'])).convert('RGBA')
            original = Image.open(FIGURE).convert('RGBA')
            if embedded.size == original.size and list(embedded.getdata()) == list(original.getdata()):
                matched = True
                break
        except Exception:
            pass
if not matched:
    raise RuntimeError('Could not verify original figure pixels in the PDF')
print(json.dumps({'report':str(REPORT),'pages':check.page_count,'figure_pixels_verified':matched,
                  'figure_pixels': [img_w,img_h], 'bytes':REPORT.stat().st_size,
                  'sha256':hashlib.sha256(REPORT.read_bytes()).hexdigest()}))
