import os, csv, json, hashlib
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
csv_dump=' '.join(str(v) for row in csv_rows for v in row.values())
records=[]
for eid,path in selected:
    assert eid in csv_dump, f'Selected ID absent from experiments.csv: {eid}'
    with open(path,encoding='utf-8') as f: result=json.load(f)
    assert result.get('experiment_id')==eid, f'results.json ID mismatch for {eid}'
    records.append(result)

# Assemble findings and comparisons from recorded result files only.
settings=[r['settings'] for r in records]
values=[float(r['final_enhancement']) for r in records]
strictly_declining=all(values[i]>values[i+1] for i in range(len(values)-1))
common_iterations=len({s.get('iterations') for s in settings})==1
common_seed=len({s.get('search_seed') for s in settings})==1
common_regions=len({(s.get('focus_size'),s.get('background_size')) for s in settings})==1
has_shots=all('shots_per_measurement' in s and 'validation_shots' in s for s in settings)
wall_recorded=any('wall_time_seconds' in r for r in records)
total_recorded=any('camera_shots' in r for r in records)
first=settings[0]

W,H=fitz.paper_size('a4')
page_doc=fitz.open()
page=page_doc.new_page(width=W,height=H)
font='tiro'  # Times Roman serif, used consistently
black=(0.08,0.08,0.08); gray=(0.40,0.40,0.40); rule=(0.72,0.72,0.72)
margin=38.0
usable=W-2*margin

def text(text, x, y, w, h, size, color=black, lineheight=1.16, align=0):
    rect=fitz.Rect(x,y,x+w,y+h)
    remain=page.insert_textbox(rect,text,fontname=font,fontsize=size,color=color,
                               lineheight=lineheight,align=align,overlay=True)
    if remain < -0.05:
        raise RuntimeError(f'Text box overflow ({remain:.2f}pt): {text[:100]}')
    return remain

def hline(y, shade=rule, thickness=0.5):
    page.draw_line((margin,y),(W-margin,y),color=shade,width=thickness,overlay=True)

# 1) Title and research question.
text('Macropixel size and validated focusing enhancement',margin,31,usable,25,17.0)
hline(61,gray,0.65)
text('Research question: How does DMD macropixel size affect validated focusing when the regions and search budget are fixed?',
     margin,70,usable,27,10.0,lineheight=1.18)

# 2) Main finding, fairness and comparison limitations.
if strictly_declining:
    finding=('In this simulation, validated enhancement decreased at every tested increase in macropixel size, from '
             f"{values[0]:.2f} at size {settings[0]['macropixel_size']} to {values[-1]:.2f} at size {settings[-1]['macropixel_size']}. ")
else:
    finding=(f'Validated enhancement ranged from {min(values):.2f} to {max(values):.2f} across the selected settings. ')
if common_iterations and common_seed and common_regions:
    finding+=(f"This is a controlled fixed-budget comparison: only macropixel size varied; focus/background sizes "
              f"({first['focus_size']}/{first['background_size']}), search seed and {first['iterations']}-iteration budget were shared. ")
else:
    finding+='The selected settings do not hold every other parameter fixed, so comparison is limited. '
finding+='The 1,000-iteration cap constrains fine-grid coverage, and fresh measurement noise differs between runs.'
text(finding,margin,102,usable,51,9.3,lineheight=1.18)

# 3) Supplied calibrated/tuned figure is inserted directly, byte-for-byte source PNG,
# with aspect ratio preserved; no redraw, crop, recoloring or resampling.
caption='Tuned grayscale heatmaps: Initial and Final panels for the four selected experiments, in frozen order.'
text(caption,margin,157,usable,14,8.6,color=gray)
with Image.open(FIG) as im:
    ratio=im.width/im.height
# Dominant figure. Source is 180 x 200 mm; render as large as remaining layout permits.
image_x=105.0
image_width=min(usable,385.0)
image_height=image_width/ratio
image_y=174.0
image_rect=fitz.Rect(image_x,image_y,image_x+image_width,image_y+image_height)
assert image_width>=W*0.25 and image_height>=H*0.25
page.insert_image(image_rect,filename=FIG,keep_proportion=False,overlay=True)

# 4) Compact comparison table immediately below figure; frozen order, varied setting,
# validated enhancement and recorded computational cost only.
table_y=image_rect.y1+8
hline(table_y,gray,0.55)
col_fracs=[0.21,0.25,0.17,0.37]
col_x=[margin]
for fraction in col_fracs[:-1]: col_x.append(col_x[-1]+usable*fraction)
headers=['Macropixel size (px)','Final enhancement','Iterations (count)','Camera shots / measurement']
header_y=table_y+4
for i,label in enumerate(headers):
    text(label,col_x[i]+3,header_y,usable*col_fracs[i]-6,20,8.0,color=gray,lineheight=1.0)
row_y=header_y+21
hline(row_y,rule,0.45)
row_height=17.0
for rec in records:
    st=rec['settings']
    shots=(f"{st['shots_per_measurement']} search; {st['validation_shots']} validation"
           if 'shots_per_measurement' in st and 'validation_shots' in st else 'not recorded')
    cells=[str(st['macropixel_size']),f"{float(rec['final_enhancement']):.2f}",str(st.get('iterations','not recorded')),shots]
    for i,value in enumerate(cells):
        text(value,col_x[i]+3,row_y+2,usable*col_fracs[i]-6,row_height-2,8.6,lineheight=1.0)
    row_y+=row_height
    hline(row_y,rule,0.35)

# 5) Verdict with metric definition, cost comparison, limitations and next experiment.
footer_y=row_y+7
text('Verdict and next experiment',margin,footer_y,usable,16,10.2)
footer_y+=18
cost=(f"All runs used {first.get('iterations','not recorded')} iterations; each search measurement used "
      f"{first.get('shots_per_measurement','not recorded')} camera shots and validation used "
      f"{first.get('validation_shots','not recorded')} shots. Wall time and total camera-shot counts were "
      f"{'recorded' if wall_recorded or total_recorded else 'not recorded'}. Enhancement is the validated focus mean divided by the equally weighted mean of four background-region means.")
text(cost,margin,footer_y,usable,37,8.6,lineheight=1.15)
footer_y+=38
next_seed=int(first.get('search_seed',0))+1
verdict=(f"Finer macropixels gave higher enhancement in this fixed-budget run, but the fine-grid cap and fresh noise limit generalization. "
         f"Next, repeat size {first['macropixel_size']} with focus/background sizes {first['focus_size']}/{first['background_size']}, "
         f"{first['iterations']} iterations and search seed {next_seed} to test seed sensitivity. Grayscale tuning changes presentation, not scientific measurements; physical print quality was not validated.")
text(verdict,margin,footer_y,usable,39,8.6,lineheight=1.15)
assert footer_y+39 <= H-margin, f'Footer exceeds bottom margin: {footer_y+39:.1f}pt'
assert image_rect.x0>=margin and image_rect.x1<=W-margin and image_rect.y1<table_y

page_doc.set_metadata({'title':'Macropixel size and validated focusing enhancement','author':'Scientific brief'})
page_doc.save(REPORT,garbage=0,deflate=False)
page_doc.close()

# Validate exact one-page A4 and ensure the PNG is embedded unchanged by PDF image xref.
check=fitz.open(REPORT)
assert check.page_count==1
pg=check[0]
assert abs(pg.rect.width-W)<0.2 and abs(pg.rect.height-H)<0.2
source_sha=hashlib.sha256(open(FIG,'rb').read()).hexdigest()
image_refs=pg.get_images(full=True)
assert image_refs, 'No image present in report'
with Image.open(FIG) as source:
    source_pixels=source.convert('RGBA').tobytes(); source_size=source.size
pixel_match=False
for info in image_refs:
    embedded=check.extract_image(info[0])
    import io
    try:
        with Image.open(io.BytesIO(embedded['image'])) as candidate:
            if candidate.size==source_size and candidate.convert('RGBA').tobytes()==source_pixels:
                pixel_match=True
                break
    except Exception:
        pass
assert pixel_match, 'Supplied tuned image did not survive embedding pixel-identically'
check.close()
print(json.dumps({'report':REPORT,'pages':1,'page_points':[W,H],
                  'figure_rect_points':[image_rect.x0,image_rect.y0,image_rect.width,image_rect.height],
                  'figure_source_sha256':source_sha,'figure_pixels_unchanged':pixel_match,
                  'file_bytes':os.path.getsize(REPORT)},indent=2))