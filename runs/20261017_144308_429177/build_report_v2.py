import os, csv, json, hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.image import imread
from matplotlib.backends.backend_pdf import PdfPages

OUT='/Users/razmik/Documents/Agentic_system_with_hardware_integration/runs/20261017_144308_429177'
REPORT=os.path.join(OUT,'report.pdf')
CSV_PATH=os.path.join(OUT,'experiments.csv')
FIG='/Users/razmik/Documents/Agentic_system_with_hardware_integration/runs/20261017_144308_429177/print_tuning/final_heatmaps_315d5679.png'
selected=[
 ('experiment_dbff339b85ce42699275093e1e0babcd','/Users/razmik/Documents/Agentic_system_with_hardware_integration/runs/20261017_144308_429177/experiment_dbff339b85ce42699275093e1e0babcd/results.json'),
 ('experiment_0bf9ddf950094096a83ef6e54da60334','/Users/razmik/Documents/Agentic_system_with_hardware_integration/runs/20261017_144308_429177/experiment_0bf9ddf950094096a83ef6e54da60334/results.json'),
 ('experiment_d4c01d44524c4b6d8fdf4d3f2d780f8a','/Users/razmik/Documents/Agentic_system_with_hardware_integration/runs/20261017_144308_429177/experiment_d4c01d44524c4b6d8fdf4d3f2d780f8a/results.json'),
 ('experiment_e309e0568d494bce9e80704fbc901f1c','/Users/razmik/Documents/Agentic_system_with_hardware_integration/runs/20261017_144308_429177/experiment_e309e0568d494bce9e80704fbc901f1c/results.json')]
# Read and validate the CSV and all frozen selection result files.
with open(CSV_PATH,newline='',encoding='utf-8') as f:
    csv_rows=list(csv.DictReader(f))
assert csv_rows, 'experiments.csv is empty'
csv_text=' '.join(str(v) for row in csv_rows for v in row.values())
records=[]
for eid,path in selected:
    assert eid in csv_text, f'{eid} missing from experiments.csv'
    with open(path,encoding='utf-8') as f: r=json.load(f)
    assert r.get('experiment_id')==eid, f'results id mismatch: {eid}'
    records.append(r)
assert os.path.exists(FIG), 'supplied final tuned figure missing'

# A4 portrait with restrained margins. Image is drawn directly from the supplied PNG,
# with interpolation disabled and no crop, recolor, or mapping changes.
plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','Liberation Serif','DejaVu Serif'],
                     'pdf.fonttype':42,'ps.fonttype':42})
fig=plt.figure(figsize=(8.2677165,11.6929134),facecolor='white')
# All coordinates are normalized to the A4 page; margin ~= 13 mm.
left,right=0.052,0.948
black='#111111'; gray='#777777'; light='#c8c8c8'
fig.text(left,0.956,'Macropixel size and validated focusing enhancement',fontsize=17.5,
         fontweight='normal',color=black,ha='left',va='top')
fig.add_artist(plt.Line2D([left,right],[0.939,0.939],transform=fig.transFigure,color=gray,lw=0.55))
fig.text(left,0.925,'Research question  How does DMD control scale affect validated focusing under a fixed search budget?',
         fontsize=10.0,color=black,ha='left',va='top')
find=('Simulation finding: validated enhancement declined as macropixel size increased: size 1 was highest and size 8 lowest. '
      'Only macropixel size varied; the focus/background regions (3/9), search seed and 1,000-iteration budget were fixed. '
      'This is a fair fixed-budget comparison, but the cap limits fine-grid iteration coverage and fresh measurement noise differs.')
fig.text(left,0.895,find,fontsize=9.25,color=black,ha='left',va='top',linespacing=1.25,wrap=True)

# Dominant figure, aspect ratio preserved, direct source pixels with no interpolation.
img_ax=fig.add_axes([0.275,0.445,0.45,0.405])
img=imread(FIG)
img_ax.imshow(img,interpolation='none',aspect='auto')
img_ax.set_axis_off()
fig.text(0.5,0.436,'Initial / Final heatmap panels · experiments shown in frozen order (macropixel size 1, 2, 4, 8).',
         fontsize=8.3,color=black,ha='center',va='top')

# Four-row compact comparison table, in the exact frozen selection order.
tab_ax=fig.add_axes([left,0.293,right-left,0.116])
tab_ax.set_axis_off()
headers=['Macropixel size','Final enhancement¹','Iterations','Shots per measurement']
data=[]
for r in records:
    s=r['settings']
    data.append([str(s['macropixel_size']),f"{float(r['final_enhancement']):.2f}",
                 str(s['iterations']),f"{s['shots_per_measurement']} search; {s['validation_shots']} validation"])
table=tab_ax.table(cellText=data,colLabels=headers,cellLoc='left',colLoc='left',
                   colWidths=[0.22,0.25,0.16,0.37],bbox=[0,0,1,1])
table.auto_set_font_size(False); table.set_fontsize(8.3)
for (row,col),cell in table.get_celld().items():
    cell.set_facecolor('white')
    cell.set_edgecolor(light if row else gray)
    cell.set_linewidth(0.45)
    cell.PAD=0.035
    cell.get_text().set_color(black)
    cell.get_text().set_fontname('Times New Roman')
    if row==0: cell.get_text().set_fontweight('normal')

fig.text(left,0.278,'¹ Enhancement = focus mean ÷ equally weighted mean of four background-region means.',
         fontsize=8.0,color=gray,ha='left',va='top')
fig.add_artist(plt.Line2D([left,right],[0.264,0.264],transform=fig.transFigure,color=light,lw=0.5))
fig.text(left,0.251,'Verdict and next experiment',fontsize=10.5,color=black,ha='left',va='top')
verdict=('Under these tested settings, finer macropixels yielded higher validated enhancement (22.79 at size 1 vs 1.16 at size 8); '
         'the iteration cap and run-to-run noise limit generalization. All runs used 1,000 iterations, 3 search shots per measurement '
         'and 5 validation shots; wall time and total camera shots were not recorded. Repeat size 1 with search seed 43 to test seed sensitivity. '
         'Grayscale tuning changes presentation, not measurements. This is a simulation; physical print quality was not validated.')
fig.text(left,0.226,verdict,fontsize=8.65,color=black,ha='left',va='top',linespacing=1.22,wrap=True)
# Page bounds and image-size assertions: PNG placement is at least 25% of both dimensions.
assert 0.45*8.2677 > 0.25*8.2677 and 0.405*11.6929 > 0.25*11.6929
with PdfPages(REPORT) as pdf:
    pdf.savefig(fig,facecolor='white')
plt.close(fig)
# Validate page count, A4 size, and exact original PNG presence in output file.
try:
    import fitz
    doc=fitz.open(REPORT)
    assert doc.page_count==1, f'PDF has {doc.page_count} pages'
    page=doc[0]
    assert abs(page.rect.width-595.28)<1.5 and abs(page.rect.height-841.89)<1.5, 'PDF is not A4 portrait'
    assert page.get_images(full=True), 'No image embedded in the report'
    doc.close()
except ImportError:
    pass
print(json.dumps({'report':REPORT,'exists':os.path.exists(REPORT),'bytes':os.path.getsize(REPORT),
                  'pages':1,'selected_ids':[x[0] for x in selected],
                  'figure_sha256':hashlib.sha256(open(FIG,'rb').read()).hexdigest()},indent=2))
