"""Appendix10 frozen-rule tables, matched contrasts, guardrails and decision."""
import csv
import json
import math
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from route3_local_replacement import OUT,REC,PREFIX,SCENES,ROWS,COLS,REGIONS,desc_for,frame_dir,case_dir,load,write,reference
from route3_upper_bound_evaluate import pool

def table(path,rows):
    if not rows:return
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=keys,lineterminator='\n');writer.writeheader();writer.writerows(rows)

def zero_pool(rr,prefix):
    n=sum(r[prefix+'_pixels'] for r in rr);s=sum(r[prefix+'_sse'] for r in rr)
    return -10*math.log10(max(s/max(3*n,1),1e-30))

def summarize_scores():
    rows=json.loads((OUT/'scores.json').read_text())['rows'];groups={}
    keys=['scene','nominal_focus_m','fstop','rule','row','column','region']
    for r in rows:groups.setdefault(tuple(r[k] for k in keys),[]).append(r)
    pooled=[dict(zip(keys,key),**pool(rr),zero_full_PSNR=zero_pool(rr,'zero_full'),zero_infocus_PSNR=zero_pool(rr,'zero_infocus')) for key,rr in groups.items()]
    for rule in ['P2','P1']:
        table(REC/(PREFIX+'-'+rule+'-cells.csv'),[r for r in rows if r['rule']==rule]);table(REC/(PREFIX+'-'+rule+'-pooled.csv'),[r for r in pooled if r['rule']==rule])
    return rows,pooled

def contrasts(rows,pooled):
    result=[];guards=[]
    for level,data in [('frame',rows),('pooled',pooled)]:
        keynames=['scene','nominal_focus_m','fstop','rule','region']+(['frame'] if level=='frame' else [])
        groups={}
        for r in data:groups.setdefault(tuple(r[k] for k in keynames),{})[(r['row'],r['column'])]=r
        for key,lookup in groups.items():
            base=lookup[('R0','none')];common=dict(level=level,**dict(zip(keynames,key)))
            def delta(a,b,field):return a[field]-b[field] if a[field] is not None and b[field] is not None else None
            align=delta(lookup['R0','GTaligned'],lookup['R0','GTraw'],'psnr');align_lp=delta(lookup['R0','GTaligned'],lookup['R0','GTraw'],'patch_lpips');refgain=delta(lookup['R0','GTaligned'],base,'psnr');ref_lp=delta(lookup['R0','GTaligned'],base,'patch_lpips')
            for row in ROWS:
                a=lookup[row,'GTaligned'];none=lookup[row,'none'];method=lookup[row,'retrieval'];replace=delta(a,none,'psnr');replace_lp=delta(a,none,'patch_lpips')
                result.append(dict(**common,row=row,H_dB=delta(a,base,'psnr'),H_delta_LPIPS=delta(a,base,'patch_lpips'),delta_align_dB=align,delta_align_LPIPS=align_lp,delta_replace_dB=replace,delta_replace_LPIPS=replace_lp,interaction_dB=replace-refgain if replace is not None else None,interaction_LPIPS=replace_lp-ref_lp if replace_lp is not None else None,suppression_cost_dB=delta(none,base,'psnr'),suppression_cost_LPIPS=delta(none,base,'patch_lpips'),M_dB=delta(method,base,'psnr'),M_delta_LPIPS=delta(method,base,'patch_lpips'),pixels=base['pixels']))
            if common['region']=='in_focus':
                for (row,column),r in lookup.items():
                    dz=r['zero_full_PSNR']-base['zero_full_PSNR'];df=r['psnr']-base['psnr'];guards.append(dict(**common,row=row,column=column,zero_full_change_dB=dz,zero_infocus_change_dB=r['zero_infocus_PSNR']-base['zero_infocus_PSNR'],infocus_change_dB=df,zero_full_flag=dz<-.1,P2_infocus_flag=common['rule']=='P2' and df<-.1,guard_pass=bool(dz>=-.1 and (common['rule']!='P2' or df>=-.1))))
    table(REC/(PREFIX+'-contrasts.csv'),result);table(REC/(PREFIX+'-guardrails.csv'),guards);return result,guards

def metadata():
    support=[];suppression=[];focus=[];coc=[];old=[];placements=[]
    for p in sorted(OUT.glob('*/f*/P*/*/render.json')):
        d=json.loads(p.read_text());common={k:d[k] for k in ['scene','frame','id','rule','nominal_focus_m','fstop','focus_m']}
        support.append(dict(**common,recoverable_pixels=d['recoverable_pixels'],supported_recoverable_pixels=d['oracle_support_recoverable_pixels'],supported_fraction=d['oracle_support_recoverable_fraction'],**d['parent_counts']))
        suppression.extend(dict(**common,**r) for r in d['suppression'])
        c=dict(**common,valid_infocus_pixels=d['CoC']['valid_infocus_pixels'])
        for name,stat in d['CoC'].items():
            if isinstance(stat,dict):
                for k in ['mean','median','fraction_gt_half']:c[name+'_'+k]=stat.get(k)
        coc.append(c)
        old.extend(r['old_result_max_linear_error'] for r in d['rows'] if r['old_result_max_linear_error'] is not None)
    for p in sorted(OUT.glob('*/f*/prepared.json')):
        d=json.loads(p.read_text());focus.extend(dict(scene=d['scene'],frame=d['frame'],**r) for r in d['focus'])
        desc=desc_for(d['scene']);parents=load(desc['parents'](d['frame']));alignment=load(p.parent/'alignment.npz');own=np.load(p.parent/'visible-depth.npz')['depth'];uv=parents['uv'];x=np.clip(uv[:,0].astype(int),0,511);y=np.clip(uv[:,1].astype(int),0,511);valid=alignment['parent_valid'];vis=own[y,x];mapped=alignment['parent_depth_aligned'];rel=(mapped[valid]-vis[valid])/vis[valid]
        placements.append(dict(scene=d['scene'],frame=d['frame'],valid_parents=int(valid.sum()),aligned_in_front_fraction=float((rel<0).mean()) if len(rel) else None,aligned_within5percent_visible_fraction=float((abs(rel)<=.05).mean()) if len(rel) else None,aligned_relative_depth_median=float(np.median(rel)) if len(rel) else None,raw_relative_depth_median=float(np.median((parents['depth'][valid]-vis[valid])/vis[valid])) if len(rel) else None,**d['local_alignment']))
    table(REC/(PREFIX+'-support.csv'),support);table(REC/(PREFIX+'-suppression.csv'),suppression);table(REC/(PREFIX+'-CoC.csv'),coc)
    write(REC/(PREFIX+'-preparation.json'),dict(rows=[json.loads(p.read_text()) for p in sorted(OUT.glob('*/f*/prepared.json'))],focus=focus,old_result_reproduction_max_linear_error=max(old) if old else None))
    table(REC/(PREFIX+'-placement.csv'),placements)
    return support,suppression,focus,max(old) if old else None

def method_controls(main_rows,pooled):
    auxiliary=json.loads((OUT/'method-control-scores.json').read_text())['rows'];keys=['scene','nominal_focus_m','fstop','rule','row','region'];groups={}
    for r in auxiliary:groups.setdefault(tuple(r[k] for k in keys),[]).append(r)
    pp=[dict(zip(keys,key),**pool(rr)) for key,rr in groups.items()];result=[]
    for level,rows,reference in [('frame',auxiliary,main_rows),('pooled',pp,pooled)]:
        matchkeys=['scene','nominal_focus_m','fstop','rule','region']+(['frame'] if level=='frame' else [])
        lookup={tuple(r[k] for k in matchkeys)+(r['row'],r['column']):r for r in reference}
        for r in rows:
            key=tuple(r[k] for k in matchkeys);base=lookup[key+('R0','none')];method=lookup[key+(r['row'],'retrieval')]
            if r['pixels']:
                result.append(dict(level=level,**{k:r[k] for k in matchkeys},row=r['row'],pixels=r['pixels'],matched_method_suppression_cost_dB=r['psnr']-base['psnr'],retrieval_replacement_vs_matched_suppression_dB=method['psnr']-r['psnr'],M_dB=method['psnr']-base['psnr']))
    table(REC/(PREFIX+'-method-support-controls.csv'),result)

def transmittance():
    cells=[];pooled=[];order=['visible_layer0_le5pct','visible_layer0_5to20pct','visible_layer0_gt20pct','visible_layer1_le5pct','visible_layer1_5to20pct','visible_layer1_gt20pct','replacement_le5pct','replacement_5to20pct','replacement_gt20pct']
    def statistic(v,h):
        total=v.sum(-1)+h.sum(-1);tv=np.exp(-v.sum(-1));tt=np.exp(-total);den=max(float(total.sum()),1e-30);vd=max(float(v.sum()),1e-30)
        r=dict(rays=len(total),T_median=float(np.median(tt)) if len(total) else None,T_mean=float(tt.mean()) if len(total) else None,T_gt_half=float((tt>.5).mean()) if len(total) else None,visible_only_T_median=float(np.median(tv)) if len(tv) else None,visible_only_T_gt_half=float((tv>.5).mean()) if len(tv) else None,visible_margin_le5pct_attenuation_fraction=float(v[:,[0,3]].sum()/vd),visible_margin_5to20pct_attenuation_fraction=float(v[:,[1,4]].sum()/vd),visible_margin_gt20pct_attenuation_fraction=float(v[:,[2,5]].sum()/vd))
        for i,label in enumerate(order):r[label+'_attenuation_fraction']=float((v[:,i].sum() if i<6 else h[:,i-6].sum())/den)
        return r
    # Group at fixed scene/lens/rule; only three frames resident at a time.
    for scene in SCENES:
      desc=desc_for(scene)
      for rule in ['P2','P1']:
       for nominal in [1.4,5.]:
        for stop in [1.4,2.]:
          frame_data=[]
          for e in desc['entries']:
            c=next(c for c in e['cases'] if c['nominal_focus_m']==nominal and c['fstop']==stop);folder=case_dir(scene,e['frame'],c,rule);meta=json.loads((folder/'transmittance.json').read_text());frame_data.append((e['frame'],load(meta['path'])))
          for plane in ['aligned','raw']:
           for kind in ['hidden','control']:
            for row in ROWS:
             for column in COLS:
              state='full' if row=='R0' else ('method_' if column=='retrieval' else 'oracle_')+row;vs=[];hs=[]
              common=dict(scene=scene,rule=rule,nominal_focus_m=nominal,fstop=stop,reference_plane=plane,ray_group=kind,row=row,column=column)
              for frame,a in frame_data:
                v=a[plane+'__'+state+'__'+kind];h=a[plane+'__'+column+'__'+kind] if column!='none' else np.zeros((len(v),3),np.float32);vs.append(v);hs.append(h);cells.append(dict(**common,frame=frame,**statistic(v,h)))
              pooled.append(dict(**common,**statistic(np.concatenate(vs),np.concatenate(hs))))
          del frame_data
    table(REC/(PREFIX+'-T-cells.csv'),cells);table(REC/(PREFIX+'-T-pooled.csv'),pooled)
    write(REC/(PREFIX+'-T-definition.json'),dict(category_order=order,primary='aligned placed-parent surface,continuous actual projected rays;support-limited and missing counts explicit',auxiliary='raw GT placed-parent plane',visible_only='prefix from retained visible SHARP Gaussians',merged='visible prefix plus other replacement Gaussians in front ofreference plane;not the post-native-earlystop residual alpha',cutoff='sameAA eps2d.0015,alpha>=1/255,cap.999;allforwardproduct,no earlytermination',controls='actual nominal foreground-blocked ray,limit median hidden plane fromotherK16rays',parent_matching='sameobject trueworld nearestGTparent<=3GTpixelfootprints+0.1mm;unmatched notassumedtransparent'))
    return pooled

def decision(rows,contrasts,guards):
    selected=lambda row,scene,column:[r for r in rows if r['scene']==scene and r['row']==row and r['column']==column and r['rule']=='P2' and r['nominal_focus_m']==5 and r['fstop']==1.4 and r['region']=='recoverable']
    reports=[]
    for row in ROWS:
        pilots=SCENES[:2];a=pool([r for s in pilots for r in selected(row,s,'GTaligned')]);b=pool([r for s in pilots for r in selected('R0',s,'none')]);h=a['psnr']-b['psnr'];per=[]
        for s in pilots:
            cc=[r for r in contrasts if r['level']=='frame' and r['scene']==s and r['row']==row and r['rule']=='P2' and r['nominal_focus_m']==5 and r['fstop']==1.4 and r['region']=='recoverable'];per.append(dict(scene=s,frame_H=[dict(frame=r['frame'],H_dB=r['H_dB'],pixels=r['pixels']) for r in cc],frames_ge1dB=sum(r['H_dB']>=1 for r in cc)))
        anchor=pool(selected(row,'B2-high','GTaligned'))['psnr']-pool(selected('R0','B2-high','none'))['psnr'];gg=[r for r in guards if r['level']=='frame' and r['row']==row and r['column']=='GTaligned' and r['rule']=='P2' and r['fstop']==1.4];allrow=[r for r in guards if r['level']=='frame' and r['row']==row and r['rule']=='P2' and r['fstop']==1.4];m=pool([r for s in pilots for r in selected(row,s,'retrieval')])['psnr']-b['psnr'];guard_pass=all(r['guard_pass'] for r in gg)
        report=dict(row=row,G2_far_H_pooled_dB=h,per_pilot=per,B2high_far_H_dB=anchor,aligned_guardrails_pass=guard_pass,aligned_guardrail_failed_cells=sum(not r['guard_pass'] for r in gg),all_row_columns_guardrails_pass=all(r['guard_pass'] for r in allrow),all_row_columns_failed_cells=sum(not r['guard_pass'] for r in allrow),method_M_G2_pooled_dB=m,passed=bool(h>=1 and all(r['frames_ge1dB']>=2 for r in per) and anchor>=2.54 and guard_pass))
        reports.append(report)
    passing=[r['row'] for r in reports if r['row']!='R0' and r['passed']]
    status='STOP_CONFIRM_LOCAL_REPLACEMENT' if passing else ('CONTINUE_G3_S2A_R0_ONLY' if reports[0]['passed'] else 'STOP_REPORT_DIAGNOSTIC')
    return dict(status=status,rows=reports,passing_suppression_rows=passing,guardrail_scope='actual aligned-GT candidate cells,all5scenes,bothfocus,3frames,P2f1.4;allrow diagnostic flags separately;methodvariant guards separatelyreported',no_automatic_rule_or_threshold_changes=True)

def figures():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from route3_b2_evaluate import gt_supported_crop
    dest=OUT/'visualizations';dest.mkdir(exist_ok=True);rows=[]
    for scene in SCENES:
      desc=desc_for(scene);e=next(e for e in desc['entries'] if e['frame']==30)
      for focus in [1.4,5.]:
        c=next(c for c in e['cases'] if c['nominal_focus_m']==focus and c['fstop']==1.4);folder=case_dir(scene,30,c,'P2');truth,regions=reference(scene,desc,c,30);y,x,info=gt_supported_crop(regions['recoverable']);labels=['Cycles','R0-none','R0-GTaligned','R1-GTaligned','R2-GTaligned','R1-retrieval','R2-retrieval'];ims=[truth]+[np.load(folder/(name+'.npz'))['prediction_srgb'] for name in labels[1:]];fig,axes=plt.subplots(3,len(ims),figsize=(21,9))
        for col,(im,name) in enumerate(zip(ims,labels)):
            axes[0,col].imshow(im);axes[0,col].set_title(name);axes[0,col].add_patch(Rectangle((x-.5,y-.5),64,64,edgecolor='yellow',fill=False))
            if col==0 and regions['recoverable'].any():axes[0,col].contour(regions['recoverable'].astype(np.uint8),levels=[.5],colors=['lime'],linewidths=.5)
            axes[1,col].imshow(im[y:y+64,x:x+64],interpolation='nearest');err=np.sqrt(((im-truth)**2).mean(-1));axes[2,col].imshow(err[y:y+64,x:x+64],cmap='magma',vmin=0,vmax=.15,interpolation='nearest')
            for rr in range(3):axes[rr,col].axis('off')
        fig.suptitle(scene+' '+c['id']+' P2;GT-supported64pxcrop;shared error0-.15');path=dest/(scene+'-F'+str(focus)+'.png');fig.savefig(path,dpi=110);plt.close(fig);rows.append(dict(scene=scene,focus_m=focus,path=str(path),crop_yx=[y,x],**info))
    return rows

def main():
    rows,pooled=summarize_scores();deltas,guards=contrasts(rows,pooled);support,suppression,focus,old=metadata();method_controls(rows,pooled);T=transmittance();d=decision(rows,deltas,guards);plots=figures();summary=dict(protocol='appendix10 seventh revision,7e5779b',**d,method_images=len(rows)//5,old_result_reproduction_max_linear_error=old,output=str(OUT),main_table=PREFIX+'-P2-pooled.csv',auxiliary_table=PREFIX+'-P1-pooled.csv',contrasts=PREFIX+'-contrasts.csv',T=PREFIX+'-T-pooled.csv',support=PREFIX+'-support.csv',suppression=PREFIX+'-suppression.csv',guardrails=PREFIX+'-guardrails.csv',CoC=PREFIX+'-CoC.csv',figures=plots,boundary='five controlled existing scenes,E1GTposes;Oraclealignment is diagnostic placement,not physical upperbound;no learning or new scenes',old_outputs_unchanged=True,next_step='stop and user decides unlessR0aligned alonepasses explicitcontinuation branch')
    write(REC/(PREFIX+'-summary.json'),summary);write(OUT/'summary.json',summary);print(json.dumps(summary,indent=2,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
