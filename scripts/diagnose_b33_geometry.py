"""Input-only cross-rope profiles and paired focus/disparity decomposition for B33."""
from pathlib import Path
import json,sys
import numpy as np
import cv2
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from prepare_atb_b33_b36_diagnostics import band_pixels
OUT=ROOT/'outputs/any2bokeh-failure-analysis-20261009/B33-geometry-diagnosis-20261015'
REC=ROOT/'03_实验记录/records/any2bokeh-B33-geometry-diagnosis-20261015.json'
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def main():
 OUT.mkdir(exist_ok=True,parents=True);cv2.setNumThreads(1)
 loc=json.loads((ROOT/'03_实验记录/records/any2bokeh-B33-B36-diagnostics-20261014-localization.json').read_text())['rows'];base=ROOT/'data/processed/davis2016-v2/eval/car-turn';profiles=[];paired={};frames=[1,6,12,18,22]
 for t in frames:
  rgb=cv2.cvtColor(cv2.imread(str(base/'rgb'/f'{t:05d}.jpg')),cv2.COLOR_BGR2RGB);gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY).astype(float)
  dp=next((base/'disparity').glob(f'{t:05d}_zf_*.png'));d=np.asarray(Image.open(dp),dtype=float)/255;focus=float(dp.stem.split('_zf_')[1])
  for r in loc:
   if r['frame_id']!=t or not r['is_rope'] or not r['accepted']:continue
   cid=r['correspondence_id'];center=r['native_center'];tangent=r['tangent'];offsets=list(range(-10,11));rows=[]
   for off in offsets:
    xx,yy,_=band_pixels(center,tangent,off,21,1)
    rows.append(dict(offset_native=off,pixel_count=len(xx),RGB_luma_mean=float(gray[yy,xx].mean()),d_mean=float(d[yy,xx].mean()),d_median=float(np.median(d[yy,xx])),nominal_mean=float((16*abs(d[yy,xx]-focus)).mean())))
   # RGB-only local central-line choice, fixed +/-3 normal pixels, no disparity-based recentering.
   peak=max([q for q in rows if abs(q['offset_native'])<=3],key=lambda q:q['RGB_luma_mean']);ix=offsets.index(peak['offset_native']);background=[rows[0],rows[-1]]
   xx,yy,_=band_pixels(center,tangent,0,21,3);v=d[yy,xx].copy()
   if t in [1,22]:paired[(cid,t)]=(v,focus)
   profiles.append(dict(frame=t,correspondence_id=cid,focus=focus,center_native=center,profile=rows,RGB_peak_candidate=peak,side_d_mean=float(np.mean([q['d_mean'] for q in background])),peak_minus_sides_d=peak['d_mean']-float(np.mean([q['d_mean'] for q in background])),limits='RGB peak is a 1-native-pixel candidate, not pure physical rope segmentation; fixed slope/context registration retained.'))
   cx,cy=np.round(center).astype(int);crop=rgb[cy-38:cy+38,cx-45:cx+45];q=d[cy-38:cy+38,cx-45:cx+45];color=cv2.cvtColor(cv2.applyColorMap(np.round(q*255).astype(np.uint8),cv2.COLORMAP_VIRIDIS),cv2.COLOR_BGR2RGB)
   canvas=Image.new('RGB',(540,248),'white');canvas.paste(Image.fromarray(crop).resize((270,228),Image.Resampling.NEAREST),(0,20));canvas.paste(Image.fromarray(color).resize((270,228),Image.Resampling.NEAREST),(270,20));dr=ImageDraw.Draw(canvas);dr.text((3,3),f'{cid} frame{t} RGB | native VDA fixed0-1',fill='black');canvas.save(OUT/f'{cid}-{t:02d}-input.png')
 decomposition=[]
 for cid in ['R1','R2','R3','R4']:
  if (cid,1) not in paired or (cid,22) not in paired:decomposition.append(dict(correspondence_id=cid,status='missing paired location; no interpolation'));continue
  q1,f1=paired[cid,1];q2,f2=paired[cid,22];C=lambda q,f:float((16*abs(q-f)).mean());a,b,c,d=C(q1,f1),C(q1,f2),C(q2,f1),C(q2,f2);F=.5*((b-a)+(d-c));D=.5*((c-a)+(d-b));assert abs(F+D-(d-a))<1e-10
  decomposition.append(dict(correspondence_id=cid,native_mean_nominal_frame1=a,native_mean_nominal_frame22=d,hold_frame1_disparity_change_focus=b,hold_frame1_focus_change_disparity=c,symmetric_focus_contribution=F,symmetric_disparity_contribution=D,total_change=d-a,limits='Algebraic decomposition of declared control at tracked candidate bands, not causal truth of physical geometry.'))
 result=dict(scene='car-turn',frames=frames,profiles=profiles,endpoint_decomposition=decomposition,selection='Reuse frozen RGB-only localization; all valid R1–R4, no discarded contradictions. RGB normal profiles and peak search fixed before reading results.',limits=['R3 frame22 unavailable','One source, relative predicted geometry only','Profiles do not verify real depth or output quality','Focus drift can dominate temporal change even if the static rope depth itself is biased.'])
 save(REC,result)
 for r in profiles:
  if r['frame'] in [1,22]:print(r['frame'],r['correspondence_id'],'peak offset',r['RGB_peak_candidate']['offset_native'],'peak d',round(r['RGB_peak_candidate']['d_mean'],4),'sides',round(r['side_d_mean'],4),'delta',round(r['peak_minus_sides_d'],4))
 print(json.dumps(decomposition,indent=2))
if __name__=='__main__':main()
