import os, csv, json, hashlib
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.colors import black, HexColor
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
# Read the recorded experiment table and results; frozen selection order is authoritative.
with open(CSV_PATH, newline='', encoding='utf-8') as f:
    rows=list(csv.DictReader(f))
assert rows, 'experiments.csv is empty'
flat=' '.join(str(v) for row in rows for v in row.values())
results=[]
for eid,path in selected:
    assert eid in flat, f'{eid} is missing from experiments.csv'
    with open(path, encoding='utf-8') as f: r=json.load(f)
    assert r.get('experiment_id')==eid, f'results id mismatch for {eid}'
    results.append(r)

W,H=A4
M=40.0
CW=W-2*M
serif='Times-Roman'
style_question=ParagraphStyle('question',fontName=serif,fontSize=10.2,leading=13.0,textColor=black)
style_finding=ParagraphStyle('finding',fontName=serif,fontSize=9.6,leading=12.2,textColor=black)
style_verdict=ParagraphStyle('verdict',fontName=serif,fontSize=9.0,leading=11.0,textColor=black)
style_small=ParagraphStyle('small',fontName=serif,fontSize=8.2,leading=9.6,textColor=black)

# A single A4 page; final PNG is inserted directly with PyMuPDF after text layout.
tmp=REPORT+'.base.pdf'
c=canvas.Canvas(tmp,pagesize=A4)
c.setTitle('Macropixel size and validated focusing enhancement')
c.setAuthor('Scientific brief')
# Restrained header rule and title
c.setFillColor(black)
c.setFont(serif,18)
c.drawString(M,H-42,'Macropixel size and validated focusing enhancement')
c.setStrokeColor(HexColor('#777777')); c.setLineWidth(0.55)
c.line(M,H-50,W-M,H-50)
cursor=H-62

def draw_para(text, style, y_top, x=M, width=CW):
    p=Paragraph(text,style)
    pw,ph=p.wrap(width,1000)
    p.drawOn(c,x,y_top-ph)
    return y_top-ph

question='Research question: How does DMD macropixel size affect validated focusing enhancement when the focus/background regions and search budget are held fixed?'
cursor=draw_para(question,style_question,cursor)-6
finding=('In this simulation, validated enhancement declined across the tested macropixel sizes: size 1 was highest and size 8 lowest. '
         'The fixed-budget design changes only macropixel size, with regions, search seed and iteration budget held constant. '
         'However, the 1,000-iteration cap limits fine-grid coverage, and fresh measurement noise differs between runs.')
cursor=draw_para(finding,style_finding,cursor)-7
caption='Final heatmaps · Initial and Final panels for the four selected experiments, in frozen order.'
c.setFont(serif,9.2); c.drawString(M,cursor-9,caption)
image_top=cursor-17

# Compact comparison table and verdict are measured first to reserve space for the figure.
row_h=16.0; header_h=20.0; table_h=header_h+4*row_h
verdict1=('Verdict: Enhancement is the focus mean divided by the equally weighted mean of four background-region means. '
          'The result supports higher enhancement for finer macropixels under these tested conditions, not a universal optimum: '
          'the fine-grid cap and fresh noise limit generalization. All runs used the same iteration and measurement-shot budgets; '
          'wall time and total camera shots were not recorded.')
verdict2=('Next experiment: Repeat the size-1 setting (focus 3, background 9, 1,000 iterations) with search seed 43 to test seed sensitivity. '
          'Grayscale gamma tuning changes presentation, not scientific measurements; physical print quality was not validated.')
# Use paragraphs at actual widths to calculate footer height.
def para_height(text, style): return Paragraph(text,style).wrap(CW,1000)[1]
footer_h=12.5+para_height(verdict1,style_verdict)+4+para_height(verdict2,style_verdict)
image_ratio=Image.open(FIG).width/Image.open(FIG).height
# Required space below image: table, gaps, footer, and bottom margin.
reserve=10+table_h+8+footer_h+M
max_img_h=image_top-reserve
img_h=min(CW/image_ratio,max_img_h)
assert img_h>210, f'Figure would be too small: {img_h:.1f}pt'
img_w=img_h*image_ratio
img_x=(W-img_w)/2
img_y=image_top-img_h
# Frame-free image region and thin divider only.
# Table directly below figure.
table_top=img_y-10
cols=[M, M+120, M+228, M+322, W-M]
# Header baseline and light rules
c.setStrokeColor(HexColor('#777777')); c.setLineWidth(0.45)
c.line(M,table_top,W-M,table_top)
header_baseline=table_top-13.2
c.setFillColor(black); c.setFont(serif,8.4)
headers=['Macropixel size','Final enhancement','Iterations','Camera shots / measurement']
for i,txt in enumerate(headers): c.drawString(cols[i]+3,header_baseline,txt)
header_bottom=table_top-header_h
c.setStrokeColor(HexColor('#999999')); c.line(M,header_bottom,W-M,header_bottom)
for idx,r in enumerate(results):
    row_top=header_bottom-idx*row_h
    baseline=row_top-11.2
    settings=r['settings']
    vals=[str(settings['macropixel_size']),f"{float(r['final_enhancement']):.2f}",str(settings['iterations']),
          f"{settings['shots_per_measurement']} search; {settings['validation_shots']} validation"]
    c.setFont(serif,9.0); c.setFillColor(black)
    for j,val in enumerate(vals): c.drawString(cols[j]+3,baseline,val)
    c.setStrokeColor(HexColor('#D0D0D0')); c.setLineWidth(0.35)
    c.line(M,row_top-row_h,W-M,row_top-row_h)
table_bottom=table_top-table_h
# Short verdict/limitations/next-experiment footer.
footer_top=table_bottom-8
c.setFont(serif,10.2); c.setFillColor(black)
c.drawString(M,footer_top-10,'Interpretation and next step')
fy=footer_top-16
fy=draw_para(verdict1,style_verdict,fy)-4
fy=draw_para(verdict2,style_verdict,fy)
assert fy>=M-1, f'Footer clips below margin: bottom={fy:.1f}pt'
# Check top block, figure placement and table/footer layout are non-overlapping.
assert img_y > table_top-table_h
assert img_x>=M and img_x+img_w<=W-M
c.save()

# Embed the exact supplied PNG directly, unchanged, into the reserved figure rectangle.
doc=fitz.open(tmp)
assert doc.page_count==1
a=doc[0]
rect=fitz.Rect(img_x,img_y,img_x+img_w,img_y+img_h)
a.insert_image(rect,filename=FIG,keep_proportion=False,overlay=True)
doc.save(REPORT,garbage=4,deflate=True)
doc.close(); os.remove(tmp)
# Validate one-page A4 and embedded source image hash.
check=fitz.open(REPORT)
assert len(check)==1
page=check[0]
assert abs(page.rect.width-W)<1 and abs(page.rect.height-H)<1
imgs=page.get_images(full=True)
assert imgs, 'No embedded figure image found'
source_hash=hashlib.sha256(open(FIG,'rb').read()).hexdigest()
# Ensure the placed image satisfies minimum dimensions on the page.
assert img_w>=0.25*W and img_h>=0.25*H
print(json.dumps({'report':REPORT,'pages':len(check),'page_points':[page.rect.width,page.rect.height],
                  'figure_rect_points':[img_x,img_y,img_w,img_h],'figure_minimum_ok':True,
                  'figure_sha256':source_hash,'embedded_images':len(imgs),'footer_bottom_pt':fy},indent=2))
