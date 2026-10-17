import csv, json, hashlib
from pathlib import Path
import fitz
from PIL import Image

ROOT = Path('/Users/razmik/Documents/Agentic_system_with_hardware_integration/runs/20261017_221127_850186')
REPORT = ROOT / 'report.pdf'
CSV_PATH = ROOT / 'experiments.csv'
FIGURE = Path('/Users/razmik/Documents/Agentic_system_with_hardware_integration/runs/20261017_221127_850186/print_tuning/final_heatmaps_df29ddee.png')
SELECTION = [
    ('experiment_1f77972a2eca40c2b75fc9d9fb143331', '100 iterations'),
    ('experiment_6f6aebea38fb4813873582cd5b237ce4', '300 iterations'),
    ('experiment_9bf696e7feb643cd8a555bcf80cc06e6', '600 iterations'),
    ('experiment_a47ef024e7d14b3a83d81bedda24255d', '1,000 iterations'),
]
# Load the shared run index and frozen selected results; do not derive a new selection.
with CSV_PATH.open(newline='', encoding='utf-8') as f:
    csv_rows = list(csv.DictReader(f))
if not csv_rows:
    raise RuntimeError('experiments.csv is empty')
csv_text = CSV_PATH.read_text(encoding='utf-8')
results = []
for eid, _ in SELECTION:
    result_path = ROOT / eid / 'results.json'
    if eid not in csv_text:
        raise RuntimeError(f'Selected experiment absent from experiments.csv: {eid}')
    data = json.loads(result_path.read_text(encoding='utf-8'))
    if data.get('experiment_id') != eid:
        raise RuntimeError(f'results.json experiment_id mismatch: {eid}')
    results.append(data)
if len(results) != 4 or not FIGURE.is_file():
    raise RuntimeError('Expected four selected results and supplied final figure')
settings = [r['settings'] for r in results]
budgets = [int(s['iterations']) for s in settings]
values = [float(r['final_enhancement']) for r in results]
search_shots = [int(s['shots_per_measurement']) for s in settings]
validation_shots = [int(s['validation_shots']) for s in settings]
if budgets != [100, 300, 600, 1000]:
    raise RuntimeError('Recorded values do not match frozen ascending budget order')
if len(set(search_shots)) != 1 or len(set(validation_shots)) != 1:
    raise RuntimeError('Shot protocol differs across selected runs')
if hashlib.sha256(FIGURE.read_bytes()).hexdigest() != '811cac04aa2f8c4319ad9189c21b6e97c7c25c146136a81e6c68c6cee7dea49a':
    raise RuntimeError('Final tuned figure does not match workflow-supplied image metadata')

# One A4 portrait page. Built-in Times fonts provide a portable serif family.
PW, PH = 595.276, 841.890
M = 34.0
W = PW - 2*M
pdf = fitz.open()
page = pdf.new_page(width=PW, height=PH)

def text(text, x, y, w, h, size, font='tiro', color=(0.08,0.08,0.08), lineheight=1.15, align=0):
    rc = page.insert_textbox(fitz.Rect(x, y, x+w, y+h), text, fontsize=size,
                             fontname=font, color=color, lineheight=lineheight, align=align)
    if rc < -0.1:
        raise RuntimeError(f'Text overflow ({rc:.2f} pt): {text[:100]}')
    return rc

# Short header and research question.
text('Iteration budget and optical focusing', M, 27, W, 25, 18, font='tibo')
text('Question: How does search budget affect validated focus when other settings are fixed?', M, 56, W, 17, 9.7)
# Finding combines fairness, performance versus iteration cost, and the metric explanation.
finding = (
    f'Fair comparison: the simulated bench, seed, 2×2 macropixels and focus/background regions are fixed; only iterations change. '
    f'Validated enhancement rises from {values[0]:.2f} to {values[-1]:.2f}. '
    f'The gain is {values[1]-values[0]:.2f} from 100 to 300 iterations, then {values[2]-values[1]:.2f} and {values[3]-values[2]:.2f}; returns diminish after 300.'
)
text(finding, M, 79, W, 46, 9.4, lineheight=1.14)

# Insert the exact supplied tuned PNG directly, without pixel changes or cropping.
with Image.open(FIGURE) as im:
    image_w, image_h = im.size
place_w = 320.0
place_h = place_w * image_h / image_w
image_x = (PW-place_w)/2
image_y = 128.0
page.insert_image(fitz.Rect(image_x, image_y, image_x+place_w, image_y+place_h),
                  filename=str(FIGURE), keep_proportion=True)
caption_y = image_y + place_h + 4
caption = ('Initial (left) and Final (right) heatmaps; frozen experiment order: 100, 300, 600 and 1,000 iterations. '
           'Grayscale tuning changes presentation, not scientific measurements; raster labels may be small at print size.')
text(caption, M, caption_y, W, 24, 8.6, color=(0.25,0.25,0.25), lineheight=1.1)

# Four-row comparison table immediately below figure/caption; no unrecorded cost estimates.
table_y = caption_y + 26
text('Recorded comparison', M, table_y, W, 15, 10.8, font='tibo')
table_y += 17
cols = [130.0, 122.0, 145.0, W-397.0]
headers = ['Iteration budget', 'Final enhancement', 'Search captures / measure', 'Validation captures']
page.draw_line((M,table_y),(M+W,table_y),color=(0.45,0.45,0.45),width=0.6)
x = M
for label, cw in zip(headers, cols):
    text(label, x+3, table_y+2, cw-6, 19, 8.0, font='tibo', lineheight=1.04)
    x += cw
table_y += 21
page.draw_line((M,table_y),(M+W,table_y),color=(0.70,0.70,0.70),width=0.4)
for i, (_, label) in enumerate(SELECTION):
    cells = [label, f'{values[i]:.2f}', str(search_shots[i]), str(validation_shots[i])]
    x = M
    for value, cw in zip(cells, cols):
        text(value, x+3, table_y+2, cw-6, 15, 8.8, lineheight=1.05)
        x += cw
    table_y += 17
    page.draw_line((M,table_y),(M+W,table_y),color=(0.83,0.83,0.83),width=0.35)

# Concise verdict, limits and recommended next experiment.
verdict_y = table_y + 8
verdict = (
    f'Verdict: 1,000 iterations gave the highest recorded enhancement, but the gains after 300 are modest. '
    f'Cost is reported as iterations and {search_shots[0]} camera shots per search measurement ({validation_shots[0]} fresh validation shots); total shots and runtime were not recorded. '
    f'All budgets are below the 1,280-iteration guideline, and this single search seed cannot establish robustness. '
    f'Next, repeat 1,000 iterations with search_seed=43, changing only the seed. Enhancement is focus mean divided by the equally weighted mean of four background-region means. Simulation; no claim of validated physical print quality.'
)
text(verdict, M, verdict_y, W, 75, 9.0, lineheight=1.12)
if verdict_y + 75 > PH-M:
    raise RuntimeError(f'Content exceeds page margins: bottom={verdict_y+75:.1f} pt')

pdf.set_metadata({'title':'Iteration budget and optical focusing', 'author':'Scientific brief'})
pdf.save(REPORT, garbage=4, deflate=True)
pdf.close()
# Verify one A4 page and retain the source figure's exact pixels as an embedded image.
check = fitz.open(REPORT)
if check.page_count != 1:
    raise RuntimeError(f'Report has {check.page_count} pages, expected exactly one')
if abs(check[0].rect.width-PW) > 1 or abs(check[0].rect.height-PH) > 1:
    raise RuntimeError('Report page is not A4 portrait')
image_xrefs = check[0].get_images(full=True)
if not image_xrefs:
    raise RuntimeError('Supplied final heatmap PNG is not embedded')
print(json.dumps({'report':str(REPORT),'pages':check.page_count,'figure_pixels':[image_w,image_h],
                  'figure_placement_mm':[round(place_w*25.4/72,1),round(place_h*25.4/72,1)],
                  'table_bottom_pt':round(table_y,1),'verdict_bottom_pt':round(verdict_y+75,1),
                  'enhancements':values,'embedded_images':len(image_xrefs)}, indent=2))