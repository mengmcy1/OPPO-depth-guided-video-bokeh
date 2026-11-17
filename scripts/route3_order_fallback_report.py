"""Fixed T1 comparisons and one-time order/fallback selection; never gates G3."""
import csv,json,sys,math
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from route3_order_fallback import OUT,REC,PREFIX,METHODS,EXPLICIT,write
from route3_upper_bound_evaluate import pool

def table(tag,rows):
 if not rows:return
 keys=list(dict.fromkeys(k for r in rows for k in r))
 with (REC/(PREFIX+'-'+tag+'.csv')).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=keys,lineterminator='\n');w.writeheader();w.writerows(rows)
def zpool(rr):return -10*math.log10(max(sum(r['zero_sse'] for r in rr)/(3*sum(r['zero_pixels'] for r in rr)),1e-30))
def main():
 rows=json.loads((OUT/'scores.json').read_text())['rows'];assert len(list(OUT.glob('*/f*/P*/*/render.json')))==72
 keys=['scene','rule','nominal_focus_m','fstop','method','region'];groups={}
 for r in rows:groups.setdefault(tuple(r[k] for k in keys),[]).append(r)
 pooled=[dict(zip(keys,k),**pool(rr),zero_PSNR=zpool(rr),zero_sse=sum(r['zero_sse'] for r in rr),zero_pixels=sum(r['zero_pixels'] for r in rr)) for k,rr in groups.items()]
 for rule in ['P2','P1']:
  table(rule+'-cells',[r for r in rows if r['rule']==rule]);table(rule+'-pooled',[r for r in pooled if r['rule']==rule])
 pri=[r for r in rows if r['scene'].startswith('metric-') and r['rule']=='P2' and r['nominal_focus_m']==5 and r['fstop']==1.4 and r['region']=='recoverable'];az=zpool([r for r in pri if r['method']=='A']);ab=pool([r for r in pri if r['method']=='A']);candidates=[];winners=[]
 for fall in ['F1','F2']:
  choices=[]
  for order in ['O1','O2']:
   name=f'D1_{order}_{fall}';rr=[r for r in pri if r['method']==name];pp=pool(rr);zero=zpool(rr);v=dict(order=order,fallback=fall,method=name,zero_PSNR=zero,zero_vs_A_dB=zero-az,zero_guard_pass=zero-az>=-.1,recoverable_PSNR=pp['psnr'],recoverable_vs_A_dB=pp['psnr']-ab['psnr'],patch_LPIPS=pp['patch_lpips']);choices.append(v);candidates.append(v)
  winners.append(max(choices,key=lambda r:(r['zero_guard_pass'],r['recoverable_PSNR'])))
 difference=abs(winners[0]['recoverable_PSNR']-winners[1]['recoverable_PSNR']);selected=winners[1] if difference<.1 else max(winners,key=lambda r:r['recoverable_PSNR']);result=dict(selected_order=selected['order'],selected_fallback=selected['fallback'],selected=selected,within_fallback_winners=winners,candidates=candidates,fallback_PSNR_gap_dB=difference,scope='twoG2 only,P2 far f1.4,6cells,SSE pooled;GT Oracle reference excluded',algorithm='within each F:zero<=0.1dB loss relative A prioritized,then recPSNR;compare the two winning F,PSNR gap<0.1dB choosesF2',selection_frozen=True,T2_authorized_regardless_of_T1_results=True,A_zero_PSNR=az,A_recoverable_PSNR=ab['psnr'],order_margin=1.03,O2_background_filter_radius_px=24,source_commit='5411f35')
 write(OUT/'selection.json',result);write(REC/(PREFIX+'-selection.json'),result);table('selection-candidates',candidates)
 guards=[];contrasts=[]
 for level,data in [('frame',rows),('pooled',pooled)]:
  kk=['scene','rule','nominal_focus_m','fstop','region']+(['frame'] if level=='frame' else []);gg={}
  for r in data:gg.setdefault(tuple(r[k] for k in kk),{})[r['method']]=r
  for key,look in gg.items():
   base=look['A'];common=dict(level=level,**dict(zip(kk,key)))
   for m in METHODS[1:]:
    r=look[m];d0=look['D0_'+m.rsplit('_',1)[-1]] if m.startswith('D1_') else base
    contrasts.append(dict(**common,method=m,pixels=r['pixels'],vs_A_dB=r['psnr']-base['psnr'] if r['psnr'] is not None else None,vs_A_LPIPS=r['patch_lpips']-base['patch_lpips'] if r['patch_lpips'] is not None else None,vs_same_F_D0_dB=r['psnr']-d0['psnr'] if r['psnr'] is not None else None))
    if common['region']=='in_focus':guards.append(dict(**common,method=m,zero_PSNR=r['zero_PSNR'],zero_vs_A_dB=r['zero_PSNR']-base['zero_PSNR'],zero_drop_flag=r['zero_PSNR']-base['zero_PSNR']<-.1,zero_absolute_ge50=r['zero_PSNR']>=50,infocus_vs_A_dB=r['psnr']-base['psnr'],P2_infocus_drop_flag=common['rule']=='P2' and r['psnr']-base['psnr']<-.1))
 table('contrasts',contrasts);table('guardrails',guards)
 focus=[];orders=[];gtrefs=[]
 for p in OUT.glob('*/f*/P*/*/render.json'):
  d=json.loads(p.read_text());common={k:d[k] for k in ['scene','frame','id','rule','nominal_focus_m','fstop','focus_m','GT_focus_m']};focus.append(common)
  orders.extend(dict(**common,order_rule=r.get('rule'),**{k:v for k,v in r.items() if k!='rule'}) for r in d['order'].values())
  for label in ['DGT0','DGT1']:
   new=json.loads((p.parent/(label+'_F1-scores.json')).read_text());old=np.load(Path(d['GT_old_A_focus_reference'])/(label+'.npz'))['prediction_srgb'];oldscore=json.loads((Path(d['GT_old_A_focus_reference'])/(label+'-scores.json')).read_text())
   for region in ['full_image','in_focus','band','recoverable']:
    aa=new['dof'][region];bb=oldscore['dof'][region];gtrefs.append(dict(**common,method=label,region=region,new_GT_focus_PSNR=aa['psnr'],old_A_focus_PSNR=bb['psnr'],PSNR_change=aa['psnr']-bb['psnr'],new_GT_focus_LPIPS=aa['patch_lpips'],old_A_focus_LPIPS=bb['patch_lpips'],new_focus_zero_PSNR=new['zero']['psnr'],old_A_focus_prediction=str(Path(d['GT_old_A_focus_reference'])/(label+'.npz')),new_GT_focus_prediction=str(p.parent/(label+'_F1.npz')),old_new_prediction_display_RMSE=float(np.sqrt(np.mean((old-np.load(p.parent/(label+'_F1.npz'))['prediction_srgb'])**2)))))
 table('focus',focus);table('order',orders);table('GT-focus-reference',gtrefs)
 summary=dict(status='T1_COMPLETE_PROCEED_T2_WITHOUT_RESULT_GATE',selection=result,cases=72,groups=METHODS,rows=len(rows),full_fixed_K=64,explicit_parameters=dict(subdivisions=5,sigma_pixels=.7,opacity=.999),no_training_in_T1=True,old_outputs_unchanged=True,output=str(OUT))
 write(OUT/'summary.json',summary);write(REC/(PREFIX+'-summary.json'),summary);print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
