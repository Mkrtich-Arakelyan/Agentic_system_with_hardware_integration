import os, json, csv, math
from pathlib import Path
import fitz
from PIL import Image

ROOT = Path('/Users/razmik/Documents/Agentic_system_with_hardware_integration/runs/20261017_221127_850186')
REPORT = ROOT / 'report.pdf'
CSV_PATH = ROOT / 'experiments.csv'
FIGURE = ROOT / 'print_tuning/final_heatmaps_df29ddee.png'
IDS = [
    'experiment_1f77972a2eca40c2b75fc9d9fb143331',
    'experiment_6f6aebea38fb4813873582cd5b237ce4',
    'experiment_9bf696e7feb643cd8a555bcf80cc06e6',
    'experiment_a47ef024e7d14b3a83d81bedda24255d',
]
RESULT_PATHS = [ROOT / eid / 'results.json' for eid in IDS]
# Read and check the recorded run index as well as the four frozen result files.
with CSV_PATH.open(newline='', encoding='utf-8') as f:
    csv_rows = list(csv.DictReader(f))
if not csv_rows:
    raise RuntimeError('experiments.csv is empty')
csv_text = CSV_PATH.read_text(encoding='utf-8')
results = []
for eid, path in zip(IDS, RESULT_PATHS):
    if eid not in csv_text:
        raise RuntimeError(f'{eid} is absent from experiments.csv')
    data = json.loads(path.read_text(encoding='utf-8'))
    if data.get('experiment_id') != eid:
        raise RuntimeError(f'results.json experiment_id mismatch for {eid}')
    results.append(data)
if not FIGURE.is_file():
    raise FileNotFoundError(FIGURE)

# Sort/order is intentionally the frozen selection order; verify recorded results.
settings = [r['settings'] for r in results]
if [s['iterations'] for s in settings] != sorted(s['iterations'] for s in settings):
    raise RuntimeError('Frozen selection order is not ascending iteration budget')
vals = [float(r['final_enhancement']) for r in results]
budgets = [int(s['iterations']) for s in settings]
shots = [int(s['shots_per_measurement']) for s in settings]
validation = [int(s['validation_shots']) for s in settings]
if len(set(shots)) != 1 or len(set(validation)) != 1:
    raise RuntimeError('Shot protocol differs across selected runs')

# A4 portrait, points. Conservative margins; all text is placed in explicit boxes.
PAGE_W, PAGE_H = 595.276, 841.890
M = 34.0  # ~12 mm
W = PAGE_W - 2*M
page = fitz.open().new_page(width=PAGE_W, height=PAGE_H)

def text_box(text, x, y, w, h, size, font='tiro', color=(0.08,0.08,0.08), align=0, lineheight=1.18):
    rect = fitz.Rect(x, y, x+w, y+h)
    rc = page.insert_textbox(rect, text, fontsize=size, fontname=font, color=color,
                             align=align, lineheight=lineheight)
    if rc < -0.1:
        raise RuntimeError(f'Text does not fit: {text[:70]!r}; remaining={rc}')
    return rect

# Use built-in Times family, a portable serif equivalent.
# Title and research question.
y = 31
text_box('Iteration budget and simulated optical focusing', M, y, W, 29, 19, font='tibo')
y += 31
question = 'Research question  How does increasing the search iteration budget affect validated focusing when all other settings are held fixed?'
text_box(question, M, y, W, 27, 10.5)
y += 29
# Main scientific finding, using only saved values and selected settings.
inc_100_300 = vals[1]-vals[0]
inc_300_600 = vals[2]-vals[1]
inc_600_1000 = vals[3]-vals[2]
finding = (f'With 2×2 macropixels, fixed focus/background regions and search seed, the four-run comparison is fair for the iteration-budget question; each result is validated from {validation[0]} fresh captures. '
           f'Enhancement rose from {vals[0]:.2f} at {budgets[0]} iterations to {vals[-1]:.2f} at {budgets[-1]}, with the largest gain ({inc_100_300:.2f}) from {budgets[0]} to {budgets[1]}; later gains were smaller ({inc_300_600:.2f} and {inc_600_1000:.2f}).')
text_box(finding, M, y, W, 48, 10.2, lineheight=1.17)
y += 50
# Place the exact supplied raster (unmodified), preserve aspect ratio and use no resampling.
with Image.open(FIGURE) as im:
    pix_w, pix_h = im.size
ratio = pix_w / pix_h
img_w = min(W, 310.0)  # ~109 mm wide; image itself is 180x200 mm, maintain its ratio
img_h = img_w / ratio
# Dominant image placed centered. Caption follows immediately.
img_x = (PAGE_W-img_w)/2
page.insert_image(fitz.Rect(img_x, y, img_x+img_w, y+img_h), filename=str(FIGURE), keep_proportion=True)
y += img_h + 4
caption = 'Initial (left) and Final (right) heatmaps; rows follow the frozen 100, 300, 600 and 1,000 iteration selection. Grayscale tuning changes presentation, not the underlying measurements.'
text_box(caption, M, y, W, 24, 8.7, color=(0.23,0.23,0.23), lineheight=1.12)
y += 25
# Compact comparison table immediately below the figure.
text_box('Recorded comparison', M, y, W, 15, 11, font='tibo')
y += 17
col_widths = [105, 145, 133, W-383]
headers = ['Budget (iterations)', 'Validated enhancement', 'Search captures / measurement', 'Validation captures']
rows = [[str(budgets[i]), f'{vals[i]:.2f}', str(shots[i]), str(validation[i])] for i in range(4)]
row_h = 15.3
x0 = M
# Thin rules and serif text; no heavy fill/boxes.
page.draw_line((M,y), (M+W,y), color=(0.42,0.42,0.42), width=0.65)
for i, (label, cw) in enumerate(zip(headers, col_widths)):
    text_box(label, x0+3, y+2, cw-6, 21, 8.2, font='tibo', lineheight=1.08)
    x0 += cw
y += 22
page.draw_line((M,y), (M+W,y), color=(0.72,0.72,0.72), width=0.45)
for row in rows:
    x0 = M
    for val, cw in zip(row, col_widths):
        text_box(val, x0+3, y+2, cw-6, row_h-2, 9.0, lineheight=1.05)
        x0 += cw
    y += row_h
    page.draw_line((M,y), (M+W,y), color=(0.83,0.83,0.83), width=0.35)
y += 6
# Concise verdict, cost context, limitation and next experiment.
verdict = (f'Verdict  The highest result was {vals[-1]:.2f} at {budgets[-1]} iterations, but returns diminished after {budgets[1]}; cost is recorded as iterations and {shots[0]} camera captures per search measurement ({validation[0]} validation captures), with total shots and runtime not recorded. '
           f'All four budgets remain below the 1,280-iteration guideline; a single fixed seed does not establish robustness. Next, repeat the {budgets[-1]}-iteration setting with search_seed=43, changing only the seed. Enhancement is focus mean divided by the equally weighted mean of four background-region means. Simulation; physical print quality was not validated.')
text_box(verdict, M, y, W, 50, 9.3, lineheight=1.12)
# Restrained footer rule and brief image-label limitation statement if space remains.
y += 52
if y < PAGE_H-M:
    page.draw_line((M,y), (PAGE_W-M,y), color=(0.72,0.72,0.72), width=0.45)
    text_box('Four selected experiments · common simulated bench and measurement regions', M, y+5, W, 12, 8, color=(0.38,0.38,0.38))

doc = page.parent
doc.set_metadata({'title':'Iteration budget and simulated optical focusing', 'author':'Scientific brief'})
doc.save(REPORT, garbage=4, deflate=True)
doc.close()
# Confirm exactly one A4 page and that the embedded image remains the supplied source.
check = fitz.open(REPORT)
if check.page_count != 1:
    raise RuntimeError(f'Report must be one page, got {check.page_count}')
if abs(check[0].rect.width-PAGE_W) > 1 or abs(check[0].rect.height-PAGE_H) > 1:
    raise RuntimeError('Unexpected PDF page dimensions')
print(json.dumps({'report':str(REPORT),'pages':check.page_count,'image_pixels':[pix_w,pix_h],'image_placement_mm':[round(img_w*25.4/72,1),round(img_h*25.4/72,1)],'ending_y_pt':round(y,1),'result_values':vals}, indent=2))
