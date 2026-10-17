import csv, json, hashlib
from pathlib import Path
import fitz
from PIL import Image

ROOT = Path('/Users/razmik/Documents/Agentic_system_with_hardware_integration/runs/20261017_221127_850186')
REPORT = ROOT / 'report.pdf'
CSV_PATH = ROOT / 'experiments.csv'
FIGURE = Path('/Users/razmik/Documents/Agentic_system_with_hardware_integration/runs/20261017_221127_850186/print_tuning/final_heatmaps_df29ddee.png')
IDS = ['experiment_1f77972a2eca40c2b75fc9d9fb143331','experiment_6f6aebea38fb4813873582cd5b237ce4','experiment_9bf696e7feb643cd8a555bcf80cc06e6','experiment_a47ef024e7d14b3a83d81bedda24255d']
paths = [ROOT / eid / 'results.json' for eid in IDS]
# Read the shared experiment index and selected results; preserve the frozen selection order.
with CSV_PATH.open(newline='', encoding='utf-8') as f:
    records = list(csv.DictReader(f))
if not records:
    raise RuntimeError('experiments.csv is empty')
index_text = CSV_PATH.read_text(encoding='utf-8')
results = []
for eid, path in zip(IDS, paths):
    if eid not in index_text:
        raise RuntimeError(f'Selected ID absent from experiments.csv: {eid}')
    r = json.loads(path.read_text(encoding='utf-8'))
    if r.get('experiment_id') != eid:
        raise RuntimeError(f'results.json experiment_id does not match frozen selection: {eid}')
    results.append(r)
if not FIGURE.is_file():
    raise FileNotFoundError(FIGURE)
if hashlib.sha256(FIGURE.read_bytes()).hexdigest() != '811cac04aa2f8c4319ad9189c21b6e97c7c25c146136a81e6c68c6cee7dea49a':
    raise RuntimeError('Supplied final tuned figure hash does not match workflow metadata')
settings = [r['settings'] for r in results]
budgets = [int(s['iterations']) for s in settings]
if budgets != sorted(budgets) or len(results) != 4:
    raise RuntimeError('Unexpected selected experiment order or count')
vals = [float(r['final_enhancement']) for r in results]
shots = [int(s['shots_per_measurement']) for s in settings]
validations = [int(s['validation_shots']) for s in settings]
if len(set(shots)) != 1 or len(set(validations)) != 1:
    raise RuntimeError('Capture settings differ across selected experiments')

# A4 portrait, dimensions in PDF points. Times is a portable serif equivalent.
PW, PH = 595.276, 841.890
M = 34.0
W = PW - 2*M
page = fitz.open().new_page(width=PW, height=PH)

def put(text, x, y, w, h, size, font='tiro', color=(0.08,0.08,0.08), lineheight=1.15, align=0):
    rc = page.insert_textbox(fitz.Rect(x,y,x+w,y+h), text, fontsize=size, fontname=font,
                             color=color, lineheight=lineheight, align=align)
    if rc < -0.15:
        raise RuntimeError(f'Text overflow ({rc:.2f} pt): {text[:95]}')
    return rc

# Compact header and research question.
put('Iteration budget and simulated optical focusing', M, 31, W, 25, 18.5, font='tibo')
put('Research question  How does search iteration budget affect validated focusing when other settings are fixed?', M, 60, W, 17, 10.2)
# Fairness and main recorded result; concise, evidence-linked explanation.
finding = (f'All four runs are a fair budget comparison: the simulated bench, seed, 2×2 macropixels and focus/background regions are fixed; only iterations change. '
           f'Validated enhancement (focus mean divided by the equally weighted mean of four background-region means) rises from {vals[0]:.2f} at {budgets[0]} iterations to {vals[-1]:.2f} at {budgets[-1]}. '
           f'The gain is largest from {budgets[0]} to {budgets[1]} ({vals[1]-vals[0]:.2f}); subsequent gains are smaller.')
put(finding, M, 83, W, 43, 9.5, lineheight=1.15)

# Embed the supplied tuned PNG directly, unchanged, preserving its aspect ratio.
with Image.open(FIGURE) as im:
    iw, ih = im.size
max_h = 410.0
img_h = max_h
img_w = img_h * iw / ih
if img_w > W:
    img_w = W
    img_h = img_w * ih / iw
img_x = (PW-img_w)/2
img_y = 132.0
page.insert_image(fitz.Rect(img_x, img_y, img_x+img_w, img_y+img_h), filename=str(FIGURE), keep_proportion=True)
caption_y = img_y + img_h + 5
caption = ('Initial and Final camera-intensity panels; experiment order follows the table (100, 300, 600, 1,000 iterations). '
           'Grayscale tuning changes presentation, not measurements; the raster labels may be small at page scale.')
put(caption, M, caption_y, W, 22, 8.6, color=(0.24,0.24,0.24), lineheight=1.1)

# Compact four-row table directly below the figure caption.
table_y = caption_y + 24
put('Four-run comparison', M, table_y, W, 15, 10.8, font='tibo')
table_y += 17
cols = [139.0, 112.0, 139.0, W-390.0]
headers = ['Budget (iterations)', 'Final enhancement', 'Search cost', 'Validation']
page.draw_line((M,table_y),(M+W,table_y),color=(0.48,0.48,0.48),width=0.6)
x = M
for label,cw in zip(headers,cols):
    put(label,x+3,table_y+2,cw-6,20,8.5,font='tibo',lineheight=1.05)
    x += cw
table_y += 22
page.draw_line((M,table_y),(M+W,table_y),color=(0.72,0.72,0.72),width=0.4)
for i in range(4):
    cells = [str(budgets[i]), f'{vals[i]:.2f}', f'{budgets[i]} iterations; {shots[i]} shots/measurement', f'{validations[i]} fresh captures']
    x = M
    for cell,cw in zip(cells,cols):
        put(cell,x+3,table_y+2,cw-6,16,8.7,lineheight=1.05)
        x += cw
    table_y += 18
    page.draw_line((M,table_y),(M+W,table_y),color=(0.84,0.84,0.84),width=0.35)

# Verdict includes the cost limitation, comparison caveat and recommended next test.
verdict_y = table_y + 9
verdict = (f'Verdict  The 1,000-iteration run is highest, but returns diminish substantially after 300. Cost is recorded as iterations and {shots[0]} camera shots per search measurement; total shots and runtime were not recorded. '
           f'All budgets are below the 1,280-iteration guideline, and one fixed search seed cannot establish robustness. Next, repeat 1,000 iterations with search_seed=43, changing only the seed. This is a simulation; physical print quality was not validated.')
put(verdict, M, verdict_y, W, 67, 9.1, lineheight=1.12)

# Enforce the page bounds and write exactly one A4 page.
if verdict_y + 67 > PH-M:
    raise RuntimeError(f'Content exceeds bottom margin: {verdict_y+67:.1f} pt')
doc = page.parent
doc.set_metadata({'title':'Iteration budget and simulated optical focusing','author':'Scientific brief'})
doc.save(REPORT, garbage=4, deflate=True)
doc.close()
check = fitz.open(REPORT)
if check.page_count != 1 or abs(check[0].rect.width-PW)>1 or abs(check[0].rect.height-PH)>1:
    raise RuntimeError('Output is not exactly one A4 portrait page')
print(json.dumps({'report':str(REPORT),'pages':check.page_count,'page_points':[check[0].rect.width,check[0].rect.height], 'figure_pixels':[iw,ih], 'figure_placement_mm':[round(img_w*25.4/72,1),round(img_h*25.4/72,1)],'caption_y':round(caption_y,1),'table_end_y':round(table_y,1),'verdict_y':round(verdict_y,1),'enhancements':vals},indent=2))