import json, numpy as np, OpenEXR, sys
from PIL import Image
sys.path.insert(0,'/tmp/claude-0/-root-autodl-tmp/ac32074c-db8a-4e0f-87bd-8cdb40db961a/scratchpad/build')
from split import tiles
O='/root/autodl-tmp/OPPO-depth-guided-video-bokeh/outputs/'
T='/tmp/claude-0/-root-autodl-tmp/ac32074c-db8a-4e0f-87bd-8cdb40db961a/scratchpad/build/tiles/'
def grab(path, r, c, name, inset=3):
    t=tiles(O+path); r0,r1,c0,c1=t[r][c]
    im=Image.open(O+path).convert('RGB').crop((c0+inset,r0+inset,c1-inset,r1-inset))
    im.save(T+name+'.png'); return im.size
jobs=[
 # S1b in-focus f001 F1.4: cols Cycles0 LDR1 O3_SHARP2 SHARP_full3 A_color4
 ('route3-sharp-s1b-2026-11-11-01/evaluation-native/B2-high-f001-N1.4-F1.4-in_focus.png', {'gt':0,'o3':2,'sharp':3,'anchor':4}, 'p9_inf'),
 ('route3-sharp-s1b-2026-11-11-01/evaluation-native/B2-high-f030-N1.4-F5-band.png', {'gt':0,'o3':2,'sharp':3,'anchor':4}, 'p9_band'),
 # S1 in-focus: Cycles0 O3_DA3 2 O3_SHARP3 VG_SHARP4 SHARP_full6
 ('route3-sharp-s1-2026-11-11-01/evaluation-native/B2-high-f001-N1.4-F1.4-in_focus.png', {'gt':0,'vg':4,'sharpfull':6}, 'p4_s1inf'),
 ('route3-sharp-s1-2026-11-11-01/zero-evaluation/f001-native.png', None, 'p4_zero'),
 ('route3-c1-2026-11-10-01/source-diagnostics-panel.png', None, 'p4_movies'),
 ('route3-b2only-2026-11-10-01/evaluation/visualizations/B2-high-N1.4-F5-patches.png', None,'p7_patch'),
 ('route3-upper-bound-2026-11-11-01/U0-evaluation/B2-high-f030-N1.4-F5-U0.png', None,'p2_u0'),
]
for path,cols,name in jobs[:3]:
    for k,c in cols.items():
        for r in (0,1): print(name,k,r,grab(path,r,c,f'{name}_{k}_r{r}'))
grab(jobs[3][0],0,0,'p4_zero_input'); grab(jobs[3][0],0,2,'p4_zero_sharpfull'); grab(jobs[3][0],1,2,'p4_zero_err5'); grab(jobs[3][0],1,0,'p4_zero_vg')
grab(jobs[4][0],1,0,'p4_movies_input'); grab(jobs[4][0],1,2,'p4_movies_m1')
for c,k in [(0,'gt'),(1,'o3'),(2,'o5'),(4,'o6rec')]:
    grab(jobs[5][0],0,c,f'p7_patch_{k}')
for c,k in [(0,'cycles'),(2,'u0')]:
    for r in (0,1): grab(jobs[6][0],r,c,f'p2_u0_{k}_r{r}')
# subviews
plan=json.load(open(O+'route3-upper-bound-2026-11-11-01/view-plan.json'))
g=plan['evaluation_groups']['B2-high-f030-N1.4-F5']
xs=[(v['xy_m'][0],v['xy_m'][1],v['key']) for v in g]
pick=[min(xs,key=lambda t:t[0]), max(xs,key=lambda t:t[0]), max(xs,key=lambda t:t[1])]
def srgb(l):
    l=np.clip(l,0,1); return np.where(l<=0.0031308,12.92*l,1.055*np.power(l,1/2.4)-0.055)
for i,(x,y,key) in enumerate(pick):
    with OpenEXR.File(O+f'route3-upper-bound-2026-11-11-01/U0-subviews/{key}.exr') as e:
        a=e.channels()['RGB'].pixels
    Image.fromarray((srgb(a)*255+0.5).astype(np.uint8)).save(T+f'p2_sub{i}.png'); print('sub',key,x,y)
