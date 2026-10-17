import csv, json, hashlib
from pathlib import Path
import fitz
from PIL import Image

ROOT = Path('/Users/razmik/Documents/Agentic_system_with_hardware_integration/runs/20261017_221127_850186')
REPORT = ROOT / 'report.pdf'
CSV_PATH = ROOT / 'experiments.csv'
FIGURE = Path('/Users/razmik/Documents/Agentic_system_with_hardware_integration/runs/20261017_221127_850186/print_tuning/final_heatmaps_df29ddee.png')
# Frozen workflow selection and order; scientific values are read from recorded files.
IDS = [
 'experiment_1f77972a2eca40c2b75fc9d9fb143331',
 'experiment_6f6aebea38fb4813873582cd5b237ce4',
 'experiment_9bf696e7feb643cd8a555bcf80cc06e6',
 'experiment_a47ef024e7d14b3a83d81bedda24255d',
]
with CSV_PATH.open(newline='', encoding='utf-8') as f:
    csv_rows = list(csv.DictReader(f))
if not csv_rows:
    raise RuntimeError('experiments.csv is empty')
csv_text = CSV_PATH.read_text(encoding='utf-8')
results = []
for eid in IDS:
    if eid not in csv_text:
        raise RuntimeError(f'Frozen selected ID is absent from experiments.csv: {eid}')
    data = json.loads((ROOT / eid / 'results.json').read_text(encoding='utf-8'))
    if data.get('experiment_id') != eid:
        raise RuntimeError(f'results.json experiment_id mismatch: {eid}')
    results.append(data)
if not FIGURE.is_file():
    raise FileNotFoundError(FIGURE)
if hashlib.sha256(FIGURE.read_bytes()).hexdigest() != '811cac04aa2f8c4319ad9189c21b6e97c7c25c146136a81e6c68c6cee7dea49a':
    raise RuntimeError('Supplied final tuned figure differs from workflow metadata')
settings = [r['settings'] for r in results]
budgets = [int(s['iterations']) for s in settings]
values = [float(r['final_enhancement']) for r in results]
search_shots = [int(s['shots_per_measurement']) for s in settings]
validation_shots = [int(s['validation_shots']) for s in settings]
if len(results) != 4 or budgets != sorted(budgets):
    raise RuntimeError('Unexpected frozen experiment count or order')
if len(set(search_shots)) != 1 or len(set(validation_shots)) != 1:
    raise RuntimeError('Capture protocol varies across selected experiments')

# A4 portrait, points; white page, Times-family serif throughout.
PW, PH, M = 595.276, 841.890, 36.0
W = PW - 2*M
pdf = fitz.open()
page = pdf.new_page(width=PW, height=PH)
black = (0.08, 0.08, 0.08)
gray = (0.38, 0.38, 0.38)

def wrap(text, font, size, width):
    words = text.split()
    lines, current = [], ''
    for word in words:
        candidate = word if not current else current + ' ' + word
        if fitz.get_text_length(candidate, fontname=font, fontsize=size) <= width:
            current = candidate
        else:
            if not current:
                raise RuntimeError('Unbreakable text is wider than its box')
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines

def paragraph(text, x, top, width, size=9.2, font='tiro', leading=None, color=black):
    leading = leading or size * 1.22
    lines = wrap(text, font, size, width)
    baseline = top + size
    for line in lines:
        page.insert_text((x, baseline), line, fontname=font, fontsize=size, color=color)
        baseline += leading
    return baseline

# Compact heading and research question.
y = 27.0
title = 'Iteration budget and optical focusing'
page.insert_text((M, y+19), title, fontname='tibo', fontsize=18.5, color=black)
y += 31
question = 'Research question: How does search iteration budget affect validated focusing when other settings are fixed?'
y = paragraph(question, M, y, W, size=10.0, leading=12.2)
y += 7
# Fairness, validated results, diminishing returns, and metric in concise prose.
finding = (
    f'This is a fair budget comparison: the simulated bench, seed, 2×2 macropixels and measurement regions are fixed; only iterations change. '
    f'Validated enhancement rose from {values[0]:.2f} to {values[-1]:.2f}. The gain was {values[1]-values[0]:.2f} from {budgets[0]} to {budgets[1]} iterations, '
    f'then {values[2]-values[1]:.2f} and {values[3]-values[2]:.2f}. Enhancement is focus mean divided by the equally weighted mean of four background-region means.'
)
y = paragraph(finding, M, y, W, size=9.15, leading=11.0)
y += 8

# Embed the exact supplied PNG, unchanged, directly as the dominant visual.
with Image.open(FIGURE) as im:
    iw, ih = im.size
img_w = 420.0
img_h = img_w * ih / iw
img_x = (PW-img_w)/2
img_y = y
if img_w < PW*0.25 or img_h < PH*0.25 or img_x < M:
    raise RuntimeError('Figure would not meet size/margin requirements')
page.insert_image(fitz.Rect(img_x, img_y, img_x+img_w, img_y+img_h), filename=str(FIGURE), keep_proportion=True)
y = img_y + img_h + 4
caption = ('Initial (left) and Final (right) heatmaps; rows follow the frozen budget order shown below. '
           'Grayscale tuning changes presentation, not the underlying measurements. The original raster labels may be small at print size.')
y = paragraph(caption, M, y, W, size=8.2, leading=9.5, color=gray)
y += 5

# Four-row comparison table immediately below the figure, in frozen order.
page.insert_text((M, y+10), 'Recorded comparison', fontname='tibo', fontsize=10.5, color=black)
y += 15
col_widths = [112.0, 112.0, 190.0, W-414.0]
headers = ['Budget (iterations)', 'Final enhancement', 'Search cost: camera shots / measurement', 'Validation captures']
row_height = 17.5
page.draw_line((M, y), (M+W, y), color=(0.48,0.48,0.48), width=0.55)
x = M
for heading, cw in zip(headers, col_widths):
    lines = wrap(heading, 'tibo', 7.5, cw-6)
    base = y + 8.5
    for line in lines:
        page.insert_text((x+3, base), line, fontname='tibo', fontsize=7.5, color=black)
        base += 8.1
    x += cw
y += row_height
page.draw_line((M,y), (M+W,y), color=(0.72,0.72,0.72), width=0.4)
for i in range(len(results)):
    cells = [f'{budgets[i]}', f'{values[i]:.2f}', f'{search_shots[i]} shots', f'{validation_shots[i]} fresh']
    x = M
    for value, cw in zip(cells, col_widths):
        page.insert_text((x+3, y+12), value, fontname='tiro', fontsize=8.6, color=black)
        x += cw
    y += row_height
    page.draw_line((M,y), (M+W,y), color=(0.84,0.84,0.84), width=0.35)
y += 6

# Verdict and cost/comparison caveats, with next experiment.
verdict = (
    f'Verdict: {budgets[-1]} iterations gave the highest recorded enhancement, but returns diminished after {budgets[1]}. '
    f'Cost is reported as iterations and {search_shots[0]} camera shots per search measurement ({validation_shots[0]} fresh validation captures); total shots and runtime were not recorded. '
    f'These four runs share one search seed, so they do not establish robustness. Next, repeat the highest-budget setting with search_seed=43, changing only the seed. '
    f'This was a simulation; physical print quality was not validated.'
)
y = paragraph(verdict, M, y, W, size=8.65, leading=10.3)
if y > PH-M:
    raise RuntimeError(f'Report content extends beyond bottom margin: {y:.1f} pt')

pdf.set_metadata({'title': title, 'author': 'Scientific brief'})
pdf.save(REPORT, garbage=4, deflate=True)
pdf.close()
check = fitz.open(REPORT)
if check.page_count != 1 or abs(check[0].rect.width-PW)>1 or abs(check[0].rect.height-PH)>1:
    raise RuntimeError('Report must be exactly one A4 portrait page')
images = check[0].get_images(full=True)
if not images:
    raise RuntimeError('Final tuned figure is not embedded')
print(json.dumps({'report':str(REPORT), 'pages':check.page_count, 'figure_pixels':[iw,ih],
                  'figure_size_mm':[round(img_w*25.4/72,1),round(img_h*25.4/72,1)],
                  'content_bottom_pt':round(y,1), 'values_from_results':values,
                  'embedded_image_count':len(images)}, indent=2))