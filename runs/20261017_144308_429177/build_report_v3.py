import os, csv, json, math, hashlib
import fitz
from PIL import Image

OUT='/Users/razmik/Documents/Agentic_system_with_hardware_integration/runs/20261017_144308_429177'
REPORT=os.path.join(OUT,'report.pdf')
CSV_PATH=os.path.join(OUT,'experiments.csv')
FIG='/Users/razmik/Documents/Agentic_system_with_hardware_integration/runs/20261017_144308_429177/print_tuning/final_heatmaps_315d5679.png'
selected=[
 ('experiment_dbff339b85ce42699275093e1e0babcd','/Users/razmik/Documents/Agentic_system_with_hardware_integration/runs/20261017_144308_429177/experiment_dbff339b85ce42699275093e1e0babcd/results.json'),
 ('experiment_0bf9ddf950094096a83ef6e54da60334','/Users/razmik/Documents/Agentic_system_with_hardware_integration/runs/20261017_144308_429177/experiment_0bf9ddf950094096a83ef6e54da60334/results.json'),
 ('experiment_d4c01d44524c4b6d8fdf4d3f2d780f8a','/Users/razmik/Documents/Agentic_system_with_hardware_integration/runs/20261017_144308_429177/experiment_d4c01d44524c4b6d8fdf4d3f2d780f8a/results.json'),
 ('experiment_e309e0568d494bce9e80704fbc901f1c','/Users/razmik/Documents/Agentic_system_with_hardware_integration/runs/20261017_144308_429177/experiment_e309e0568d494bce9e80704fbc901f1c/results.json')]
assert os.path.isfile(CSV_PATH) and os.path.isfile(FIG)
with open(CSV_PATH,newline='',encoding='utf-8') as f:
    csv_rows=list(csv.DictReader(f))
assert csv_rows, 'experiments.csv is empty'
csv_blob=' '.join(str(v) for row in csv_rows for v in row.values())
records=[]
for eid,path in selected:
    assert eid in csv_blob, 'Frozen selected experiment is absent from experiments.csv: '+eid
    with open(path,encoding='utf-8') as f: r=json.load(f)
    assert r.get('experiment_id')==eid, 'results.json experiment_id mismatch: '+eid
    records.append(r)

# Derive reported findings and table entries only from recorded results.
first=records[0]['settings']; last=records[-1]['settings']
low=min(records,key=lambda r:float(r['final_enhancement']))
high=max(records,key=lambda r:float(r['final_enhancement']))
monotonic=all(float(records[i]['final_enhancement'])>float(records[i+1]['final_enhancement']) for i in range(len(records)-1))
common_iter=all(r['settings'].get('iterations')==first.get('iterations') for r in records)
common_seed=all(r['settings'].get('search_seed')==first.get('search_seed') for r in records)
common_regions=all((r['settings'].get('focus_size'),r['settings'].get('background_size'))==(first.get('focus_size'),first.get('background_size')) for r in records)
all_shots=all('shots_per_measurement' in r['settings'] and 'validation_shots' in r['settings'] for r in records)
any_wall=any('wall_time_seconds' in r for r in records)
any_total=any('camera_shots' in r for r in records)

# A4 portrait, white background, Times serif. All coordinates are PDF points.
W,H=fitz.paper_size('a4')
doc=fitz.open()
page=doc.new_page(width=W,height=H)
page.draw_rect(page.rect,color=None,fill=(1,1,1),overlay=False)
font='tiro' # PDF built-in Times Roman, consistent throughout
black=(0.08,0.08,0.08); gray=(0.42,0.42,0.42); rule=(0.72,0.72,0.72)
margin=37.5
usable=W-2*margin

def put(text, rect, size, color=black, align=0, lineheight=1.18):
    result=page.insert_textbox(rect,text,fontname=font,fontsize=size,color=color,align=align,lineheight=lineheight,overlay=True)
    if result < -0.1:
        raise RuntimeError(f'Text clipped ({result:.2f}pt): {text[:100]}')
    return result

def ruleline(y, x0=margin, x1=W-margin, color=rule, width=0.55):
    page.draw_line((x0,y),(x1,y),color=color,width=width,overlay=True)

# Compact title, question, and the scientific finding.
y=36
put('Macropixel size and validated focusing enhancement',fitz.Rect(margin,y,W-margin,y+26),18.0)
y+=29
ruleline(y,color=gray,width=0.7); y+=9
put('Research question  How does DMD macropixel size affect validated focusing when regions and search budget are held fixed?',
    fitz.Rect(margin,y,W-margin,y+28),10.1,lineheight=1.2)
y+=31
if monotonic:
    finding=(f"In this simulation, validated enhancement declined at each tested increase in macropixel size: "
             f"{int(high['settings']['macropixel_size'])} px was highest ({float(high['final_enhancement']):.2f}) and "
             f"{int(low['settings']['macropixel_size'])} px lowest ({float(low['final_enhancement']):.2f}).")
else:
    finding=(f"In this simulation, validated enhancement ranged from {float(low['final_enhancement']):.2f} to "
             f"{float(high['final_enhancement']):.2f} across the selected macropixel sizes.")
if common_iter and common_seed and common_regions:
    finding += (f" Only macropixel size varied; focus/background regions ({first['focus_size']}/{first['background_size']}), "
                f"search seed and {first['iterations']}-iteration budget were shared.")
else:
    finding += ' Settings were not identical across all selected runs; interpret the comparison cautiously.'
finding += ' The fixed budget limits fine-grid search coverage, and fresh measurement noise differs between runs.'
put(finding,fitz.Rect(margin,y,W-margin,y+48),9.5,lineheight=1.22)
y+=48

caption_y=y+1
put('Tuned grayscale heatmaps · Initial and Final panels, in frozen experiment order (macropixel sizes 1, 2, 4, 8).',
    fitz.Rect(margin,caption_y,W-margin,caption_y+13),8.4,color=gray)
y=caption_y+16

# Insert the exact supplied PNG, unchanged, preserving its native aspect ratio.
with Image.open(FIG) as im:
    pixel_w,pixel_h=im.size
ratio=pixel_w/pixel_h
image_w=min(usable*0.76,320.0)
image_h=image_w/ratio
# Require a substantial readable placement while reserving the table and text below.
image_top=y
image_rect=fitz.Rect((W-image_w)/2,image_top,(W+image_w)/2,image_top+image_h)
assert image_w >= W*0.25 and image_h >= H*0.25
page.insert_image(image_rect,filename=FIG,keep_proportion=False,overlay=True)
y=image_rect.y1+8

# Four-row table directly below the figure, in frozen selection order.
ruleline(y,color=gray,width=0.6)
header_top=y+4
header_h=18
col_widths=[0.21,0.25,0.20,0.34]
col_x=[margin]
for frac in col_widths[:-1]: col_x.append(col_x[-1]+usable*frac)
headers=['Macropixel size (px)','Validated enhancement','Iterations (count)','Camera shots / measurement (count)']
for i,hdr in enumerate(headers):
    put(hdr,fitz.Rect(col_x[i]+3,header_top,col_x[i]+usable*col_widths[i]-2,header_top+header_h),8.0,color=gray,lineheight=1.05)
y=header_top+header_h
ruleline(y,width=0.45)
row_h=16.5
for r in records:
    s=r['settings']
    vals=[str(s['macropixel_size']),f"{float(r['final_enhancement']):.2f}",str(s['iterations']),
          f"{s.get('shots_per_measurement','not recorded')} search; {s.get('validation_shots','not recorded')} validation"]
    for i,val in enumerate(vals):
        put(val,fitz.Rect(col_x[i]+3,y+2,col_x[i]+usable*col_widths[i]-2,y+row_h),8.7,lineheight=1.0)
    y+=row_h
    ruleline(y,width=0.35)

# Concise interpretation: fairness, cost, limitations, recommendation and tuning note.
y+=8
put('Verdict and next experiment',fitz.Rect(margin,y,W-margin,y+15),10.6)
y+=17
metric=('Enhancement is the validated focus mean divided by the equally weighted mean of the four background-region means. '
        f"All runs used {first['iterations']} iterations; recorded shot cost was {first.get('shots_per_measurement','not recorded')} "
        f"camera shots per search measurement and {first.get('validation_shots','not recorded')} validation shots. "
        f"Wall time and total camera shots were {'recorded' if any_wall or any_total else 'not recorded'}.")
put(metric,fitz.Rect(margin,y,W-margin,y+38),8.9,lineheight=1.18)
y+=39
next_seed=int(first['search_seed'])+1
next_exp=(f"Verdict: finer macropixels gave higher enhancement in this fixed-budget run, but noise and the fine-grid iteration cap limit generalization. "
          f"Next, repeat size {first['macropixel_size']} px with focus/background sizes {first['focus_size']}/{first['background_size']}, "
          f"{first['iterations']} iterations and search seed {next_seed} to test seed sensitivity. Grayscale tuning changes presentation, not measurements; "
          'physical print quality was not validated.')
put(next_exp,fitz.Rect(margin,y,W-margin,y+48),8.9,lineheight=1.18)
y+=48
assert y <= H-margin, f'Content exceeds bottom margin: {y:.1f} > {H-margin:.1f}'

# Save then verify single A4 page and exact supplied PNG embedded as a PDF image.
tmp=REPORT+'.tmp'
doc.set_metadata({'title':'Macropixel size and validated focusing enhancement','author':'Scientific brief'})
doc.save(tmp,garbage=0,deflate=False)
doc.close()
os.replace(tmp,REPORT)
check=fitz.open(REPORT)
assert len(check)==1
p=check[0]
assert abs(p.rect.width-W)<0.2 and abs(p.rect.height-H)<0.2
images=p.get_images(full=True)
assert images, 'No raster figure embedded'
# Inserted PNG should be embedded without pixel alteration; inspect extracted image bytes.
source_hash=hashlib.sha256(open(FIG,'rb').read()).hexdigest()
matched=False
for info in images:
    extracted=check.extract_image(info[0])
    if extracted.get('ext')=='png' and hashlib.sha256(extracted['image']).hexdigest()==source_hash:
        matched=True
        break
# Some PDF writers normalize PNG encoding while retaining the identical decoded pixels;
# verify decoded pixel arrays when byte streams differ.
if not matched:
    import io
    with Image.open(FIG) as src:
        src_rgba=src.convert('RGBA').tobytes(); src_size=src.size
    for info in images:
        ex=check.extract_image(info[0])
        try:
            with Image.open(io.BytesIO(ex['image'])) as embedded:
                if embedded.size==src_size and embedded.convert('RGBA').tobytes()==src_rgba:
                    matched=True; break
        except Exception:
            pass
assert matched, 'Could not confirm the supplied PNG is embedded pixel-identically'
check.close()
print(json.dumps({'report':REPORT,'pages':1,'page_points':[W,H],
                  'figure_points':[image_rect.width,image_rect.height],
                  'figure_source_sha256':source_hash,'figure_pixel_identical':matched,
                  'bottom_content_pt':y,'size_bytes':os.path.getsize(REPORT)},indent=2))