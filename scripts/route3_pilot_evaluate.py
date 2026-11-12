"""Development-only four-group scoring and LDR references; no method tuning."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from route3_pilot_fields import OUT,RECORDS,PREFIX,read,write
from route3_upper_bound_evaluate import display,score,pool,items
from route3_ldr_u1r import model,evaluation_masks
from src.dual_layer import depth_band


def pilot_cases(label):
    return [dict(**c,scene=label,frame_id=e['frame']) for e in json.loads((OUT/label/'prepared.json').read_text())['rows'] for c in e['cases']]


def masks(label,c):
    g=np.load(OUT/label/'geometry'/f'f{c["frame"]:03d}-center.npz');a=np.load(OUT/label/'geometry'/(c['id']+'-K16.npz'));z=np.where(g['center_visible'],g['depth_axial_m'],1e4);band,_,coc=depth_band(z,g['K'][0,0],c['nominal_focus_m'],.05/(2*c['fstop']))
    return dict(full_image=np.ones((512,512),bool),in_focus=g['center_visible']&(coc<=1),out_of_focus=g['center_visible']&(coc>=2),band=band&~a['residual_pixels'],recoverable=band&a['mask_recoverable_total'])


def references():
    rows=json.loads((OUT/'ldr-subviews-jobs-results.json').read_text());lookup={j['id']:j for j in rows};network=model();records=[];noises=[]
    for label in ['pilot-thin','pilot-multilayer']:
        dest=OUT/label/'LDR-reference';dest.mkdir(exist_ok=True)
        for case in pilot_cases(label):
            chosen=[j for j in rows if j['scene']==label and j['frame']==case['frame'] and j['focus']==case['nominal_focus_m'] and j['fstop']==case['fstop']];group={}
            for tag in ['main','noise']:
                subset=[j for j in chosen if j['reference_tag']==tag]
                if not subset:continue
                assert len(subset)==64
                linear=sum(read(j['output'])[...,:3].astype(np.float64).clip(0,1) for j in subset)/64;rgb=display(linear);path=dest/(case['id']+'-'+tag+'.npz');np.savez_compressed(path,reference_srgb=rgb,rgb_linear=linear.astype(np.float32));group[tag]=rgb
            if 'noise' in group:noises.append(dict(scene=label,id=case['id'],metrics=score(group['main'],group['noise'],masks(label,case),network)))
            records.append(dict(scene=label,id=case['id'],path=str(dest/(case['id']+'-main.npz'))));print('PILOT_LDR_REFERENCE',label,case['id'],flush=True)
    write(RECORDS/(PREFIX+'-LDR-references.json'),dict(rows=records,noise=noises,K=64,spp_per_subview=64,formula='standard_sRGB(mean64 clip(scene_linearSubview,0,1))',GT_only_scoring=True))


def recovery(method,visible,oracle):
    if oracle['psnr']-visible['psnr']<.1:return None
    base=visible['sse']/(3*visible['pixels']);true=oracle['sse']/(3*oracle['pixels']);estimated=method['sse']/(3*method['pixels']);den=base-true
    return float((base-estimated)/den) if den>=.01*base else None


def evaluate(reference):
    torch.set_num_threads(6);net=model();metrics={};paths={};pooled={};gaps=[];figures=[]
    old=json.loads((RECORDS/'route3-sharp-s1c-2026-11-10-evaluation.json').read_text());s1d=json.loads((RECORDS/'route3-sharp-s1d-2026-11-11-evaluation.json').read_text());b2criteria=json.loads((RECORDS/'route3-sharp-s1d-criteria-2026-11-11.json').read_text());refs=json.loads((ROOT/'outputs/route3-ldr-u1r-2026-11-10-01/references.json').read_text())['rows'];refmap={r['id']:r['path'] for r in refs}
    import matplotlib
    matplotlib.use('Agg');import matplotlib.pyplot as plt
    for label in ['pilot-thin','pilot-multilayer','B2-high']:
        metrics[label]={};paths[label]={};rows=json.loads((OUT/label/'predictions.json').read_text())['rows'];cases=pilot_cases(label) if label!='B2-high' else [dict(id=x['id'],frame=x['frame_id'],nominal_focus_m=x['focus_distance_m'],fstop=x['fstop'],original=x) for x in items()]
        for c in cases:
            cid=c['id'];cm={};cp={}
            if label=='B2-high':
                ldr=np.load(refmap[cid])['reference_srgb'];gt,region,_=evaluation_masks(c['original'],b2criteria,ldr);region={k:region[k] for k in ['full_image','in_focus','out_of_focus','band','recoverable']};truth=gt if reference=='Cycles' else ldr
                for name,key in [('A_color','cal_E1__A_color'),('GTrec','cal_E1__A_color_O6_GT_rec')]:cm[name]=old['per_case_metrics'][reference][cid][key];cp[name]=old['predictions'][cid][key]
                cm['R1_R2']=s1d['per_case_metrics'][reference][cid]['R2'];cp['R1_R2']=s1d['predictions'][cid]['R2']
            else:
                truth=display(read(OUT/label/'dof'/(cid+'.exr'))[...,:3]) if reference=='Cycles' else np.load(OUT/label/'LDR-reference'/(cid+'-main.npz'))['reference_srgb'];region=masks(label,c)
            for row in [r for r in rows if r['id']==cid]:
                method=row['method'];p=np.load(row['path'])['prediction_srgb'];cm[method]=score(p,truth,region,net);cp[method]=row['path']
            assert set(cm)=={'A_color','R1_R2','C2_ProPainter','GTrec'};metrics[label][cid]=cm;paths[label][cid]=cp
            if reference=='Cycles':
                ims=[truth]+[np.load(cp[n])['prediction_srgb'] for n in ['A_color','R1_R2','C2_ProPainter','GTrec']];y,x=np.nonzero(region['recoverable']);cy=int(np.clip(np.median(y)-64,0,384));cx=int(np.clip(np.median(x)-64,0,384));fig,axes=plt.subplots(2,5,figsize=(15,6))
                for col,(im,n) in enumerate(zip(ims,['Cycles','A-color','R1+R2','C2 ProPainter','GT-rec reference'])):
                    axes[0,col].imshow(im);axes[0,col].set_title(n);axes[1,col].imshow(im[cy:cy+128,cx:cx+128]);axes[0,col].axis('off');axes[1,col].axis('off')
                fig.suptitle(label+' '+cid+';GT-only display crop');fig.tight_layout();dest=OUT/'evaluation';dest.mkdir(exist_ok=True);p=dest/(label+'-'+cid+'.png');fig.savefig(p,dpi=100);plt.close(fig);figures.append(str(p))
            print('PILOT_SCORE',reference,label,cid,flush=True)
        pooled[label]={}
        for focus in [1.4,5.]:
            ids=[c['id'] for c in cases if c['nominal_focus_m']==focus];table={name:{reg:pool([metrics[label][cid][name][reg] for cid in ids]) for reg in ['full_image','in_focus','out_of_focus','band','recoverable']} for name in ['A_color','R1_R2','C2_ProPainter','GTrec']};pooled[label][str(focus)]=table
            for name,regions in table.items():
                for reg,value in regions.items():gaps.append(dict(scene=label,focus_m=focus,method=name,region=reg,PSNR_dB=value['psnr'],patch_LPIPS=value['patch_lpips'],pixels=value['pixels'],PSNR_gain_vs_visible=value['psnr']-table['A_color'][reg]['psnr'],MSE_gap_recovery=recovery(value,table['A_color'][reg],table['GTrec'][reg])))
    write(RECORDS/(PREFIX+'-baseline-'+reference+'.json'),dict(completed=True,reference=reference,per_case=metrics,pooled=pooled,table=gaps,predictions=paths,figures=figures,development_descriptive_not_new_formal_gate=True,B2_old_metrics_reused_exactly=True,GTrec_is_not_perfect_optical_upper_bound=True))
    import csv
    with (RECORDS/(PREFIX+'-baseline-'+reference+'.csv')).open('w') as f:w=csv.DictWriter(f,fieldnames=list(gaps[0]),lineterminator='\n');w.writeheader();w.writerows(gaps)


def aligned(reference):
    """A4 label placement sanity check; does not replace original four groups."""
    torch.set_num_threads(6);network=model();baseline=json.loads((RECORDS/(PREFIX+'-baseline-'+reference+'.json')).read_text());rows=[];case_metrics={};figures=[]
    for label in ['pilot-thin','pilot-multilayer']:
        preds=json.loads((OUT/label/'aligned-reference-predictions.json').read_text())['rows'];case_metrics[label]={}
        for c in pilot_cases(label):
            row=next(x for x in preds if x['id']==c['id']);pred=np.load(row['path'])['prediction_srgb'];truth=display(read(OUT/label/'dof'/(c['id']+'.exr'))[...,:3]) if reference=='Cycles' else np.load(OUT/label/'LDR-reference'/(c['id']+'-main.npz'))['reference_srgb'];case_metrics[label][c['id']]=score(pred,truth,masks(label,c),network)
        for focus in [1.4,5.]:
            selected=[c['id'] for c in pilot_cases(label) if c['nominal_focus_m']==focus];old=baseline['pooled'][label][str(focus)]
            for region in ['full_image','in_focus','band','recoverable']:
                oracle=pool([case_metrics[label][cid][region] for cid in selected]);rows.append(dict(scene=label,focus_m=focus,region=region,aligned_GTrec=oracle,aligned_vs_visible_dB=oracle['psnr']-old['A_color'][region]['psnr'],original_GTrec_vs_visible_dB=old['GTrec'][region]['psnr']-old['A_color'][region]['psnr'],R1_R2_gap_recovery_vs_aligned=recovery(old['R1_R2'][region],old['A_color'][region],oracle),C2_gap_recovery_vs_aligned=recovery(old['C2_ProPainter'][region],old['A_color'][region],oracle)))
    write(RECORDS/(PREFIX+'-A4-placement-render-'+reference+'.json'),dict(rows=rows,per_case=case_metrics,additional_Oracle_only=True,original_four_groups_and_prior_results_unchanged=True,local_fit_unsupported_hidden_parents_excluded=True))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=['references','Cycles','LDR','aligned_Cycles','aligned_LDR'],required=True);a=p.parse_args()
    if a.mode=='references':references()
    elif a.mode.startswith('aligned_'):aligned(a.mode.split('_',1)[1])
    else:evaluate(a.mode)


if __name__=='__main__':main()
