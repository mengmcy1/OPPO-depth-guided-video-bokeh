"""Stage2 frozen old references/regions; >=1dB per-cell recovery reporting.

This entry is called only after training. GT controls display crops and
scoring; it never changes method outputs, split or training choices.
"""
import argparse,csv,json,os,sys
from pathlib import Path
os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR','1')
import cv2,numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from route3_s2a_screen import OUT,RECORDS,write
from route3_s2a_proxy import display,read
from route3_upper_bound_evaluate import score,pool,items
from route3_ldr_u1r import model,evaluation_masks
from src.dual_layer import depth_band
from route3_s2a_data import descriptor,calibration_applied
from route3_b2_evaluate import gt_supported_crop
REGIONS=['recoverable','band','in_focus','full_image']
PILOT=ROOT/'outputs/route3-network-pilot-2026-11-12-01'

def masks(s,c):
    g=np.load(s/'geometry'/f'f{c["frame"]:03d}-center.npz');a=np.load(s/'geometry'/(c['id']+'-K16.npz'));z=np.where(g['center_visible'],g['depth_axial_m'],1e4);band,_,coc=depth_band(z,g['K'][0,0],c['nominal_focus_m'],.05/(2*c['fstop']))
    return dict(full_image=np.ones((512,512),bool),in_focus=g['center_visible']&(coc<=1),out_of_focus=g['center_visible']&(coc>=2),band=band&~a['residual_pixels'],recoverable=band&a['mask_recoverable_total'])

def figure(name,c,truth,region,images,special=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    y,x,selection=gt_supported_crop(region['recoverable'])
    if special is not None:y,x=special;selection=dict(historical_crop_reused_exactly=True,crop_GT_pixels=int(region['recoverable'][y:y+64,x:x+64].sum()))
    dest=OUT/'visualizations';dest.mkdir(exist_ok=True);names=['Cycles']+list(images);ims=[truth]+list(images.values());fig,axes=plt.subplots(4,len(ims),figsize=(3*len(ims),11),squeeze=False)
    for col,(im,label) in enumerate(zip(ims,names)):
        axes[0,col].imshow(im);axes[0,col].set_title(label);axes[1,col].imshow(im[y:y+64,x:x+64],interpolation='nearest');err=np.sqrt(np.mean((im-truth)**2,axis=-1));axes[2,col].imshow(err,vmin=0,vmax=.25,cmap='magma',interpolation='nearest');last=axes[3,col].imshow(err[y:y+64,x:x+64],vmin=0,vmax=.25,cmap='magma',interpolation='nearest')
        for row in range(4):axes[row,col].axis('off')
    fig.colorbar(last,ax=axes[3,:].tolist(),fraction=.02,pad=.02,label='display RGB RMSE;shared scale0-.25');fig.suptitle(name+' '+c['id']+(';estimated E1 geometry;historical PPT comparison used GT geometry' if special is not None else ';crop selected from GT support only'));path=dest/(name+'-'+c['id']+('-PPT-same-location' if special is not None else '')+'.png');fig.savefig(path,dpi=120);plt.close(fig)
    fig,ax=plt.subplots(figsize=(5,5));ax.imshow(truth)
    if region['recoverable'].any():ax.contour(region['recoverable'].astype(np.uint8),levels=[.5],colors=['lime'],linewidths=.5)
    ax.add_patch(Rectangle((x-.5,y-.5),64,64,fill=False,edgecolor='yellow',linewidth=1.5));ax.set_title(name+' '+c['id']+f' crop=({y},{x})');ax.axis('off');locator=path.with_name(path.stem+'-locator.png');fig.savefig(locator,dpi=120);plt.close(fig);return dict(scene=name,id=c['id'],path=str(path),locator=str(locator),crop_yx=[y,x],crop_size=64,interpolation='nearest',shared_error_scale=[0,.25],**selection)

def evaluate(reference):
    torch.set_num_threads(6);net=model();split=json.loads((OUT/'split.json').read_text());names=split['heldout']+['B2-high','pilot-thin','pilot-multilayer'];metrics={};paths={};cells=[];figs=[];cohort=[]
    previous=json.loads((RECORDS/'route3-network-pilot-2026-11-12-baseline-'+reference+'.json').read_text());ldr_b2={r['id']:r['path'] for r in json.loads((ROOT/'outputs/route3-ldr-u1r-2026-11-10-01/references.json').read_text())['rows']};b2items={x['id']:x for x in items()};criteria=json.loads((RECORDS/'route3-sharp-s1d-criteria-2026-11-11.json').read_text());historic=json.loads((RECORDS/'route3-c1-b2-2026-11-10-visualizations.json').read_text());historic_rows=historic['rows'] if 'rows' in historic else historic.get('plots',historic.get('figures',[]))
    # Historical figure list has a stable scene/fstop/focus and per-frame crop.
    if not historic_rows:
        for v in historic.values():
            if isinstance(v,list) and v and isinstance(v[0],dict) and 'cases' in v[0]:historic_rows=v;break
    special={c['id']:c['crop_yx'] for g in historic_rows if g.get('scene')=='B2-high' and g.get('fstop')==1.4 and g.get('focus_distance_m')==5. for c in g['cases']}
    for name in names:
        desc=descriptor(name);folder=desc['source_root'];preds=json.loads((OUT/'inference'/name/'predictions.json').read_text())['rows'];metrics[name]={};paths[name]={}
        for e in desc['entries']:
            for c in e['cases']:
                cid=c['id'];original=b2items[cid] if name=='B2-high' else None
                if name=='B2-high':
                    ldr=np.load(ldr_b2[cid])['reference_srgb'];cycles,allregions,_=evaluation_masks(original,criteria,ldr);regions={k:allregions[k] for k in ['full_image','in_focus','out_of_focus','band','recoverable']}
                else:
                    cycles=display(read(folder/'dof'/(cid+'.exr')));ldr=np.load(folder/'LDR-reference'/(cid+'-main.npz'))['reference_srgb'];regions=masks(folder,c)
                truth=cycles if reference=='Cycles' else ldr;pp={}
                if name in ['B2-high','pilot-thin','pilot-multilayer']:pp=previous['predictions'][name][cid].copy()
                else:
                    pp={method:str(folder/'predictions'/(cid+'-'+method+'.npz')) for method in ['A_color','R1_R2','GTrec','C2_ProPainter']}
                pp['LightFusion']=next(p['path'] for p in preds if p['id']==cid);mm={method:score(np.load(path)['prediction_srgb'],truth,regions,net) for method,path in pp.items()};metrics[name][cid]=mm;paths[name][cid]=pp
                cal=True
                if folder is not None:cal=calibration_applied(next(x for x in json.loads((folder/'models/calibration/summary.json').read_text())['rows'] if x['frame']==e['frame']))
                for reg in REGIONS:
                    vis=mm['A_color'][reg];oracle=mm['GTrec'][reg];qualified=oracle['psnr'] is not None and vis['psnr'] is not None and oracle['psnr']-vis['psnr']>=1.0 and min(oracle['pixels'],vis['pixels'])>=1000
                    for method,value in mm.items():
                        v=value[reg];ratio=None
                        if qualified:
                            base=vis['sse']/(3*vis['pixels']);target=oracle['sse']/(3*oracle['pixels']);actual=v['sse']/(3*v['pixels']);ratio=(base-actual)/(base-target)
                        cells.append(dict(scene=name,split='heldout' if name in split['heldout'] else 'fixed-development',id=cid,frame=e['frame'],focus_m=c['nominal_focus_m'],fstop=c['fstop'],region=reg,method=method,PSNR_dB=v['psnr'],patch_LPIPS=v['patch_lpips'],pixels=v['pixels'],valid_pixels_ge1000=v['pixels']>=1000,calibration='accepted' if cal else 'rejected-native-retained',GTrec_minus_visible_dB=oracle['psnr']-vis['psnr'] if vis['pixels'] else None,qualified_for_gap_ratio=qualified,MSE_gap_recovery=ratio,PSNR_vs_R1R2_dB=v['psnr']-mm['R1_R2'][reg]['psnr'] if v['pixels'] else None,PSNR_vs_visible_dB=v['psnr']-vis['psnr'] if v['pixels'] else None,focus_guardrail_pass=(v['psnr']-vis['psnr']>=-.2) if reg=='in_focus' and v['pixels'] else None))
                if reference=='Cycles':
                    ims={n:np.load(pp[n])['prediction_srgb'] for n in ['A_color','R1_R2','LightFusion','GTrec','C2_ProPainter']};figs.append(figure(name,c,truth,regions,ims))
                    if cid in special:figs.append(figure(name,c,truth,regions,{n:ims[n] for n in ['A_color','R1_R2','LightFusion','GTrec']},special[cid]))
                print('S2A_SCORE',reference,name,cid,flush=True)
    pooled={};summary=[]
    for name in names:
        pooled[name]={}
        for focus in [1.4,5.]:
            ids=[c['id'] for e in descriptor(name)['entries'] for c in e['cases'] if c['nominal_focus_m']==focus];group={}
            for method in metrics[name][ids[0]]:
                group[method]={reg:pool([metrics[name][cid][method][reg] for cid in ids]) for reg in REGIONS}
                for reg in REGIONS:
                    chosen=[r for r in cells if r['scene']==name and r['focus_m']==focus and r['method']==method and r['region']==reg];valid=[r for r in chosen if r['MSE_gap_recovery'] is not None];all_delta=[r['PSNR_vs_R1R2_dB'] for r in chosen if r['PSNR_vs_R1R2_dB'] is not None];v=group[method][reg];summary.append(dict(scene=name,focus_m=focus,method=method,region=reg,pooled_PSNR_dB=v['psnr'],pooled_patch_LPIPS=v['patch_lpips'],pixels=v['pixels'],qualified_cells=len(valid),total_cells=len(chosen),mean_gap_recovery_qualified=float(np.mean([r['MSE_gap_recovery'] for r in valid])) if valid else None,mean_dB_vs_R1R2_all_cells=float(np.mean(all_delta)) if all_delta else None))
            pooled[name][str(focus)]=group
    for label in ['accepted','rejected-native-retained']:
        for reg in REGIONS:
            subset=[r for r in cells if r['calibration']==label and r['region']==reg and r['method']=='LightFusion'];cohort.append(dict(calibration=label,region=reg,cells=len(subset),mean_dB_vs_rules=float(np.mean([r['PSNR_vs_R1R2_dB'] for r in subset if r['PSNR_vs_R1R2_dB'] is not None])) if subset else None))
    record=dict(completed=True,reference=reference,cells=cells,summary=summary,pooled=pooled,per_case=metrics,predictions=paths,figures=figs,calibration_strata=cohort,ratio_definition='region-specific per-cell GTrec-visible>=1dB and>=1000pixels;MSE recovery;otherwiseNA;no clipping of ratios',model_training_resolution=256,model_training_context=5,evaluation_resolution=512,evaluation_context=13,E1_GT_poses=True,B2_and_pilots_never_used_for_training=True)
    write(RECORDS/('route3-network-s2a-2026-11-12-evaluation-'+reference+'.json'),record)
    for tag,rows in [('cells',cells),('summary',summary)]:
        with (RECORDS/f'route3-network-s2a-2026-11-12-{reference}-{tag}.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--reference',choices=['Cycles','LDR'],required=True);a=p.parse_args();evaluate(a.reference)
