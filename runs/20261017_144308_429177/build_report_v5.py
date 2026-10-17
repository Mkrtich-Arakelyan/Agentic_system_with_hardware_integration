import os, csv, json, hashlib, io
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
csv_content=' '.join(str(v) for row in csv_rows for v in row.values())
records=[]
for eid,path in selected:
    assert eid in csv_content, f'Selected ID missing in experiments.csv: {eid}'
    with open(path,encoding='utf-8') as f:
        rec=json.load(f)
    assert rec.get('experiment_id')==eid, f'Results ID mismatch: {eid}'
    records.append(rec)

settings=[r['settings'] for r in records]
vals=[float(r['final_enhancement']) for r in records]
assert len(records)==4
s0=settings[0]
common=(len({s.get('iterations') for s in settings})==1 and
        len({s.get('search_seed') for s in settings})==1 and
        len({(s.get('focus_size'),s.get('background_size')) for s in settings})==1)
monotonic=all(vals[i]>vals[i+1] for i in range(3))
W,H=fitz.paper_size('a4')
margin=38.0
usable=W-2*margin
pdf=fitz.open()
page=pdf.new_page(width=W,height=H)
font='tiro'
black=(0.08,0.08,0.08); gray=(0.40,0.40,0.40); rule=(0.72,0.72,0.72)

def put(txt,x,y,w,h,size,color=black,leading=1.16):
    remain=page.insert_textbox(fitz.Rect(x,y,x+w,y+h),txt,fontname=font,fontsize=size,
                               color=color,lineheight=leading,overlay=True)
    if remain < -0.15:
        raise RuntimeError(f'Text clipped ({remain:.2f} pt): {txt[:90]}')

def line(y,color=rule,width=0.5):
    page.draw_line((margin,y),(W-margin,y),color=color,width=width,overlay=True)

# Title, question, and concise finding.
put('Macropixel size and validated focusing enhancement',margin,31,usable,25,17.0)
line(61,gray,0.65)
put('Research question: How does DMD macropixel size affect validated focusing when the regions and search budget are fixed?',
    margin,70,usable,27,10.0,leading=1.16)
if monotonic:
    finding=(f"In this simulation, validated enhancement decreased at each tested increase in macropixel size: "
             f"{vals[0]:.2f} at size {settings[0]['macropixel_size']} to {vals[-1]:.2f} at size {settings[-1]['macropixel_size']}. ")
else:
    finding=f'Validated enhancement ranged from {min(vals):.2f} to {max(vals):.2f} across the selected settings. '
if common:
    finding+=(f"Only macropixel size varied; focus/background sizes ({s0['focus_size']}/{s0['background_size']}), "
              f"search seed and {s0['iterations']}-iteration budget were shared. ")
else:
    finding+='Other settings also varied, so comparison is limited. '
finding+='The fixed iteration cap constrains fine-grid coverage, and fresh measurement noise differs between runs.'
put(finding,margin,102,usable,51,9.3,leading=1.17)
put('Final tuned heatmaps: Initial and Final panels, experiments in frozen order (macropixel sizes 1, 2, 4, 8).',
    margin,157,usable,14,8.5,color=gray)

# Insert supplied PNG directly and unchanged, preserving its complete native aspect ratio.
with Image.open(FIG) as im:
    px_w,px_h=im.size
ratio=px_w/px_h
img_w=365.0
img_h=img_w/ratio
img_x=(W-img_w)/2
img_y=174.0
img_rect=fitz.Rect(img_x,img_y,img_x+img_w,img_y+img_h)
assert img_w>=W*0.25 and img_h>=H*0.25
page.insert_image(img_rect,filename=FIG,keep_proportion=False,overlay=True)

# Four-row table immediately beneath figure; rows follow frozen selection order.
table_y=img_rect.y1+8
line(table_y,gray,0.55)
fracs=[0.21,0.25,0.17,0.37]
xs=[margin]
for f in fracs[:-1]: xs.append(xs[-1]+usable*f)
headers=['Macropixel size (px)','Final enhancement','Iterations (count)','Camera shots / measurement']
head_y=table_y+4
for i,label in enumerate(headers):
    put(label,xs[i]+3,head_y,usable*fracs[i]-6,20,8.0,color=gray,leading=1.0)
row_y=head_y+21
line(row_y,rule,0.45)
row_h=17.0
for rec in records:
    st=rec['settings']
    shots=(f"{st['shots_per_measurement']} search; {st['validation_shots']} validation"
           if 'shots_per_measurement' in st and 'validation_shots' in st else 'not recorded')
    cells=[str(st['macropixel_size']),f"{float(rec['final_enhancement']):.2f}",
           str(st.get('iterations','not recorded')),shots]
    for i,val in enumerate(cells):
        put(val,xs[i]+3,row_y+2,usable*fracs[i]-6,row_h-2,8.6,leading=1.0)
    row_y+=row_h
    line(row_y,rule,0.35)

# Verdict, fair-comparison limits, cost and next experiment.
y=row_y+7
put('Verdict and next experiment',margin,y,usable,16,10.2)
y+=18
shots_cost=(f"All runs used {s0['iterations']} iterations, {s0.get('shots_per_measurement','not recorded')} camera shots per search measurement "
            f"and {s0.get('validation_shots','not recorded')} validation shots; wall time and total camera shots were not recorded. ")
metric='Enhancement is the validated focus mean divided by the equally weighted mean of four background-region means. '
put(shots_cost+metric,margin,y,usable,34,8.35,leading=1.12)
y+=35
seed=int(s0['search_seed'])+1
verdict=(f"Finer macropixels gave higher enhancement in this fixed-budget run, but the fine-grid cap and fresh noise limit generalization. "
         f"Repeat size {s0['macropixel_size']} with focus/background sizes {s0['focus_size']}/{s0['background_size']}, "
         f"{s0['iterations']} iterations and search seed {seed} to test seed sensitivity. Grayscale tuning changes presentation, not measurements; "
         'physical print quality was not validated.')
put(verdict,margin,y,usable,37,8.35,leading=1.12)
y+=37
assert y<=H-margin, f'Page content below bottom margin ({y:.1f} > {H-margin:.1f})'
assert img_rect.x0>=margin and img_rect.x1<=W-margin and img_rect.y1<table_y

pdf.set_metadata({'title':'Macropixel size and validated focusing enhancement','author':'Scientific brief'})
pdf.save(REPORT,garbage=0,deflate=False)
pdf.close()

# Verify one A4 page and pixel-identical embedded source image.
check=fitz.open(REPORT)
assert check.page_count==1
p=check[0]
assert abs(p.rect.width-W)<0.2 and abs(p.rect.height-H)<0.2
with Image.open(FIG) as im:
    source_size=im.size
    source_pixels=im.convert('RGBA').tobytes()
matched=False
for info in p.get_images(full=True):
    ext=check.extract_image(info[0])
    try:
        with Image.open(io.BytesIO(ext['image'])) as embedded:
            if embedded.size==source_size and embedded.convert('RGBA').tobytes()==source_pixels:
                matched=True
                break
    except Exception:
        pass
assert matched, 'Supplied tuned figure not pixel-identical in PDF'
check.close()
print(json.dumps({'report':REPORT,'pages':1,'A4_points':[W,H],
                  'figure_rect_points':[img_rect.x0,img_rect.y0,img_rect.width,img_rect.height],
                  'figure_sha256':hashlib.sha256(open(FIG,'rb').read()).hexdigest(),
                  'figure_pixels_unchanged':matched,'content_bottom_pt':y,
                  'bytes':os.path.getsize(REPORT)},indent=2))