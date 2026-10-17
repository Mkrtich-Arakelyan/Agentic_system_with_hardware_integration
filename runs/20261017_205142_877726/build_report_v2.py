import csv, json, hashlib, io
from pathlib import Path
import fitz
from PIL import Image

OUT = Path('/Users/razmik/Documents/Agentic_system_with_hardware_integration/runs/20261017_205142_877726')
REPORT = OUT / 'report.pdf'
CSV_PATH = OUT / 'experiments.csv'
FIGURE = Path('/Users/razmik/Documents/Agentic_system_with_hardware_integration/runs/20261017_205142_877726/print_tuning/final_heatmaps_467c6601.png')
SELECTED = [
 ('experiment_18848d544a654115a84d698c48e80b3a', OUT / 'experiment_18848d544a654115a84d698c48e80b3a/results.json'),
 ('experiment_0a53b0d22dca48bbb95c5cd1eea54236', OUT / 'experiment_0a53b0d22dca48bbb95c5cd1eea54236/results.json'),
 ('experiment_92f251efb4d4474ebbfd2e37a03be6fc', OUT / 'experiment_92f251efb4d4474ebbfd2e37a03be6fc/results.json'),
 ('experiment_2f5f3e6a9c104597979ec1175375ede4', OUT / 'experiment_2f5f3e6a9c104597979ec1175375ede4/results.json'),
]

# Read the shared table and validate each frozen selected result JSON.
with CSV_PATH.open(newline='', encoding='utf-8-sig') as f:
    csv_rows = list(csv.DictReader(f))
if not csv_rows:
    raise RuntimeError('experiments.csv contains no records')
records = []
for expected_id, path in SELECTED:
    result = json.loads(path.read_text(encoding='utf-8'))
    if result.get('experiment_id') != expected_id:
        raise RuntimeError(f'results.json experiment_id mismatch: {expected_id}')
    if not any(expected_id in str(row) for row in csv_rows):
        raise RuntimeError(f'{expected_id} is absent from experiments.csv')
    settings = result['settings']
    records.append({'id': expected_id,
                    'macro': int(settings['macropixel_size']),
                    'enh': float(result['final_enhancement']),
                    'iterations': settings.get('iterations'),
                    'shots': settings.get('shots_per_measurement'),
                    'validation': settings.get('validation_shots'),
                    'dmd_shape': settings.get('dmd_shape'),
                    'focus': settings.get('focus_size'),
                    'background': settings.get('background_size'),
                    'seed': settings.get('search_seed')})
if [r['macro'] for r in records] != sorted(r['macro'] for r in records):
    raise RuntimeError('Selected results are not in frozen macropixel order')

im = Image.open(FIGURE)
im.verify()
im = Image.open(FIGURE).convert('RGBA')
img_w, img_h = im.size
if (img_w, img_h) != (900, 1000):
    # Validate from file itself rather than altering/recreating the supplied figure.
    if img_w <= 0 or img_h <= 0:
        raise RuntimeError('Invalid final figure dimensions')

# Single A4 portrait page; the supplied PNG is inserted directly and unchanged.
PW, PH = 595.276, 841.890
pdf = fitz.open()
p = pdf.new_page(width=PW, height=PH)
p.draw_rect(p.rect, color=(1,1,1), fill=(1,1,1), overlay=True)
ink = (0.08, 0.08, 0.08)
mid = (0.36, 0.36, 0.36)
line = (0.72, 0.72, 0.72)
L, R = 37, PW-37

def text(rect, value, size=10, font='tiro', color=ink, lineheight=1.12):
    spare = p.insert_textbox(fitz.Rect(*rect), value, fontname=font, fontsize=size,
                             color=color, lineheight=lineheight, overlay=True)
    if spare < -0.6:
        raise RuntimeError(f'Text overflow ({spare:.1f} pt): {value[:100]}')
    return spare

text((L,24,R,51),'Macropixel scale and focusing performance',19,'tibo',ink,1.0)
question = 'Research question  |  How does control-block size affect validated focusing enhancement?'
text((L,53,R,69),question,10,'tiro',mid,1.0)
# Two concise scientific finding sentences, using measured values and settings read above.
first, last = records[0], records[-1]
summary = (f"On the fixed simulated bench, validated enhancement fell from {first['enh']:.3g} at macropixel size {first['macro']} "
           f"to {last['enh']:.3g} at size {last['macro']} in this four-run comparison. "
           f"Enhancement is focus mean divided by the equally weighted mean of four background-region means; all runs held seed, regions and absolute iteration budget fixed.")
text((L,74,R,112),summary,10,'tiro',ink,1.14)
p.draw_line(fitz.Point(L,119),fitz.Point(R,119),color=line,width=0.7)

# Fit the supplied 180 x 200 mm-class PNG at dominant size, preserving its full aspect ratio.
max_w, max_h = 360.0, 375.0
scale = min(max_w/img_w, max_h/img_h)
fw, fh = img_w*scale, img_h*scale
fx, fy = (PW-fw)/2, 124.0
figbox = fitz.Rect(fx,fy,fx+fw,fy+fh)
p.insert_image(figbox, filename=str(FIGURE), keep_proportion=True, overlay=True)
cap_y = fy + fh + 3
caption = 'Initial / Final heatmaps; macropixel sizes 1, 2, 4, 8. Grayscale tuning changes appearance, not measurements.'
text((L,cap_y,R,cap_y+20),caption,8.8,'tiro',mid,1.0)

# Compact table follows image and preserves frozen selection order.
top = cap_y + 24
text((L,top,R,top+14),'COMPARISON  |  common settings: fixed regions; search seed and iteration budget held constant',9.8,'tibo',ink,1.0)
y = top+17
cols = [L, L+86, L+198, L+310, R]
heads = ['Macropixel size','Final enhancement','Iterations','Camera shots']
for j,h in enumerate(heads):
    text((cols[j]+3,y,cols[j+1]-3,y+13),h,8.5,'tibo',mid,1.0)
p.draw_line(fitz.Point(L,y+14),fitz.Point(R,y+14),color=line,width=0.6)
row_h=16
for i,r in enumerate(records):
    yy=y+15+i*row_h
    vals=[str(r['macro']),f"{r['enh']:.3f}",
          f"{r['iterations']} iterations" if r['iterations'] is not None else 'not recorded',
          (f"{r['shots']} / search; {r['validation']} validation" if r['shots'] is not None and r['validation'] is not None else 'not recorded')]
    for j,v in enumerate(vals):
        text((cols[j]+3,yy,cols[j+1]-3,yy+14),v,8.5,'tiro',ink,1.0)
    if i < len(records)-1:
        p.draw_line(fitz.Point(L,yy+14.5),fitz.Point(R,yy+14.5),color=(0.87,0.87,0.87),width=0.35)
table_end=y+15+len(records)*row_h
p.draw_line(fitz.Point(L,table_end),fitz.Point(R,table_end),color=line,width=0.6)

# Verdict combines fairness caveat, performance/cost, metric definition and the suggested follow-up.
# Controllable-block counts and nominal iterations per block are calculated from saved settings.
def per_block(r):
    shape = r['dmd_shape']
    nblocks = (shape[0]//r['macro'])*(shape[1]//r['macro'])
    return r['iterations']/nblocks
low_a, low_b = records[:2]
perf = (f"Validated enhancement decreased across the selected runs, from {records[0]['enh']:.3f} (size {records[0]['macro']}) "
        f"to {records[-1]['enh']:.3f} (size {records[-1]['macro']}); each used {records[0]['iterations']} iterations, "
        f"{records[0]['shots']} camera shots per search measurement and {records[0]['validation']} fresh validation shots.")
fair = (f"This is fair at equal absolute budget and fixed measurement regions, but not per control block: sizes {low_a['macro']} and {low_b['macro']} average "
        f"{per_block(low_a):.1f} and {per_block(low_b):.1f} iterations per block, respectively. Wall time and total camera-shot counts were not recorded; the fine-grid outcome is budget-limited, and one seed cannot establish robustness.")
next_exp = (f"Next: repeat macropixel size {records[0]['macro']} with search seed {records[0]['seed']+1}, holding other settings fixed, to test search-trajectory sensitivity.")
body = fair+' '+perf+' '+next_exp
vy=table_end+7
text((L,vy,R,vy+14),'VERDICT AND NEXT EXPERIMENT',10,'tibo',ink,1.0)
text((L,vy+16,R,815),body,9.0,'tiro',ink,1.12)
p.draw_line(fitz.Point(L,817),fitz.Point(R,817),color=line,width=0.5)
text((L,820,R,835),'Simulation; relative intensity units. Grayscale tuning is presentational and does not change the measurements.',8,'tiro',mid,1.0)

pdf.set_metadata({'title':'Macropixel scale and focusing performance','author':'Scientific brief'})
pdf.save(REPORT,garbage=4,deflate=True)
pdf.close()

# Enforce the page dimensions and exact inclusion of the supplied source pixels.
check=fitz.open(REPORT)
if check.page_count != 1:
    raise RuntimeError(f'Report must be exactly one page; found {check.page_count}')
page=check[0]
if abs(page.rect.width-PW)>1 or abs(page.rect.height-PH)>1:
    raise RuntimeError('PDF is not A4 portrait')
# Verify embedded figure XObject dimensions and decoded pixel equality.
matched=False
for entry in page.get_images(full=True):
    extracted=check.extract_image(entry[0])
    if (extracted.get('width'),extracted.get('height')) == (img_w,img_h):
        embedded=Image.open(io.BytesIO(extracted['image'])).convert('RGBA')
        if embedded.size == im.size and embedded.tobytes() == im.tobytes():
            matched=True
            break
if not matched:
    raise RuntimeError('Exact original final-figure pixels were not verified in PDF')
# Figure placement must satisfy minimum quarter-page dimensions and stay within margins.
images=page.get_image_rects(entry[0], transform=False) if page.get_images(full=True) else []
placed=[rect for rect in images if rect.width >= PW*.25 and rect.height >= PH*.25]
if not placed:
    raise RuntimeError('Supplied figure placement is below the required minimum size')
print(json.dumps({'report':str(REPORT),'pages':check.page_count,'figure_pixels_verified':matched,
                  'figure_source_pixels':[img_w,img_h],'placed_size_pt':[placed[0].width,placed[0].height],
                  'sha256':hashlib.sha256(REPORT.read_bytes()).hexdigest(),'bytes':REPORT.stat().st_size}))