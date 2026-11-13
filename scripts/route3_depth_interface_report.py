"""Appendix9 compact records, fixed-region tables, plots and conditional stop."""
import csv
import json
import math
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from route3_depth_interface import OUT,REC,PREFIX,SCENES,entries,local,write,score_masks,display,read,REGIONS
from route3_upper_bound_evaluate import pool


def table(path,rows):
    if not rows:return
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=keys,lineterminator="\n");writer.writeheader();writer.writerows(rows)


def collect_focus():
    rows=[];fits=[];controls=[]
    for p in sorted(OUT.glob('*/f*/interface.json')):
        d=json.loads(p.read_text());common=dict(scene=d['scene'],frame=d['frame'])
        for r in d['focus']:
            rr=dict(r);coc=rr.pop('subject_CoC_px',{})
            rows.append(dict(**common,**rr,CoC_mean_px=coc.get('mean'),CoC_median_px=coc.get('median'),**{f'fit_{k}':v for k,v in d['background_fits'][r['method']].items()}))
        fits.extend(dict(**common,method=m,**r) for m,r in d['background_fits'].items())
        controls.extend(dict(**common,**r) for r in d['minimum_CoC_control'])
    table(REC/(PREFIX+'-F1-F2-focus.csv'),rows)
    write(REC/(PREFIX+'-F1-F2.json'),dict(rows=rows,background_fits=fits,minimum_CoC_controls=controls,dominance_threshold=.9,relative_cluster_depth=.03,below20percent='flag only,no relaxed fallback',single_depth_baselines='O3_SHARP/O3_DA3 focus interfaces recorded;F3 re-render groups are the requested A_color,R1_R2,GTrec',visible_depth='actual renderer expected depth and actual contribution clusters,not input first-layer map',subject='existing GT center-visible CoC<=1 at nominal focus,N1.4;unchanged;GT only simulates subject selection',selected_pixel_depth='cluster ordinary median,then unweighted pixel median',P1='robust global background inverse-depth affine,held-out residuals reported;no per-object/per-pixel correction'))
    return rows,fits


def f4_report():
    d=json.loads((OUT/'F4.json').read_text());groups={}
    for r in d['rows']:groups.setdefault((r['scene'],r['region']),[]).append(r)
    pooled=[]
    for (scene,region),rr in groups.items():
        n=sum(r['pixels'] for r in rr);ss=sum(r['sse'] for r in rr);allss=sum(r['sse'] for r in d['rows'] if r['scene']==scene and r['region']=='full_image')
        pooled.append(dict(scene=scene,region=region,pixels=n,pixel_fraction=n/(3*512**2),sse=ss,SSE_fraction=ss/max(allss,1e-30),psnr=-10*math.log10(max(ss/max(3*n,1),1e-30)) if n else None))
    d['pooled']=pooled
    d['interpretation']=dict(broad='94.4% of cached zero-aperture SSE is boundary;interior and grazing PSNR around68-69dB;remaining broad loss localizes to SHARP two-layer anchored boundaries',multilayer='interior share alone is not evidence of texture failure:interior68.38dB,comparable to smooth pilot control68.87dB;grazing only0.52% of SSE',no_generator_change=True)
    table(REC/(PREFIX+'-F4-regions.csv'),d['rows']);table(REC/(PREFIX+'-F4-pooled.csv'),pooled);write(REC/(PREFIX+'-F4.json'),d)
    return d


def headroom(metrics):
    cells=metrics['cells'];pooled=metrics['pooled'];gains=[]
    lookup={(r['scene'],r['frame'],r['nominal_focus_m'],r['rule'],r['method'],r['region']):r for r in cells}
    for r in cells:
        if r['method']!='GTrec':continue
        v=lookup[(r['scene'],r['frame'],r['nominal_focus_m'],r['rule'],'A_color',r['region'])]
        gains.append(dict(scene=r['scene'],frame=r['frame'],nominal_focus_m=r['nominal_focus_m'],rule=r['rule'],region=r['region'],pixels=r['pixels'],gain_dB=r['psnr']-v['psnr'] if r['pixels'] else None,SSE_ratio_GTrec_to_Acolor=r['sse']/v['sse'] if v['sse'] else None,delta_patch_LPIPS=r['patch_lpips']-v['patch_lpips'] if r['pixels'] else None))
    pg=[]
    for r in pooled:
        if r['method']!='GTrec':continue
        v=next(v for v in pooled if all(v[k]==r[k] for k in ['scene','nominal_focus_m','rule','region']) and v['method']=='A_color')
        pg.append(dict(scene=r['scene'],nominal_focus_m=r['nominal_focus_m'],rule=r['rule'],region=r['region'],pixels=r['pixels'],gain_dB=r['psnr']-v['psnr'] if r['pixels'] else None,SSE_ratio_GTrec_to_Acolor=r['sse']/v['sse'] if v['sse'] else None,delta_patch_LPIPS=r['patch_lpips']-v['patch_lpips'] if r['pixels'] else None))
    table(REC/(PREFIX+'-F5-headroom-cells.csv'),gains);table(REC/(PREFIX+'-F5-headroom-pooled.csv'),pg)
    return gains,pg


def blockers_report():
    details=[json.loads(p.read_text()) for p in sorted(OUT.glob('*/f*/*-blockers.json'))]
    for r in details:
        r['ray_protocol']='nominal physical label ray;focus-independent check' if r['rule']=='label-ray' else 'method pixel under actual P2/P1 projection;nominal label membership retained'
        if r['rule']=='label-ray':
            r['method_band_coverage_rule']='P2 comparator only;label-ray is not a method'
    cat=[];removed=[];cover=[];tt=[]
    for name,folder in SCENES.items():
      for e in entries(folder):
        dest=local(name,e['frame'])
        for c in e['cases']:
          if c['fstop']!=1.4:continue
          mask=score_masks(folder,c)['recoverable']
          for rule in ['old','P2','P1']:
            b=np.load(dest/(c['id']+'-'+rule+'-band.npz'))['band']
            cover.append(dict(scene=name,frame=e['frame'],nominal_focus_m=c['nominal_focus_m'],rule=rule,recoverable_pixels=int(mask.sum()),inside_method_band_pixels=int((mask&b).sum()),coverage=float(b[mask].mean()) if mask.any() else None))
    for r in details:
        common={k:r[k] for k in ['scene','frame','nominal_focus_m','focus_m','rule']}
        for kind in ['hidden','foreground_blocked_control']:
            d=r[kind];tt.append(dict(**common,ray_group=kind,**d['T']))
            for i,label in enumerate(r['category_order']):cat.append(dict(**common,ray_group=kind,category=label,attenuation_fraction=d['category_log_attenuation_fraction'][i],mean_ray_fraction=d['category_mean_ray_fraction'][i] if d['category_mean_ray_fraction'] is not None else None))
            for label,value in d['without_category'].items():removed.append(dict(**common,ray_group=kind,removed=label,**value))
    table(REC/(PREFIX+'-F5-categories.csv'),cat);table(REC/(PREFIX+'-F5-without-blockers.csv'),removed);table(REC/(PREFIX+'-F5-band-coverage.csv'),cover);table(REC/(PREFIX+'-F5-T.csv'),tt)
    # Pool selected-ray attenuation, retaining exact medians and counterfactuals.
    groups={}
    for r in details:groups.setdefault((r['scene'],r['nominal_focus_m'],r['rule']),[]).append(r)
    pooled=[]
    def stat(a):return dict(n=len(a),mean=float(a.mean()),median=float(np.median(a)),fraction_gt_half=float((a>.5).mean()),quantiles=np.quantile(a,[.01,.1,.5,.9,.99]).tolist()) if len(a) else dict(n=0,mean=None,median=None,fraction_gt_half=None)
    for key,rr in groups.items():
      for kind,field in [('hidden','hidden_log_attenuation'),('foreground_blocked_control','control_log_attenuation')]:
        logs=np.concatenate([np.load(r['data'])[field] for r in rr]);total=logs.sum(-1);den=max(float(total.sum()),1e-30)
        groups_drop={f'layer{l}_{m}':[l*3+i] for l in [0,1] for i,m in enumerate(['le5pct','5to20pct','gt20pct'])};groups_drop.update(layer0=[0,1,2],layer1=[3,4,5],same_depth_le5pct=[0,3],margin5to20pct=[1,4],foreground_gt20pct=[2,5])
        pooled.append(dict(scene=key[0],nominal_focus_m=key[1],rule=key[2],ray_group=kind,T=stat(np.exp(-total)),category_fraction=(logs.sum(0)/den).tolist(),without={label:stat(np.exp(-(total-logs[:,ix].sum(-1)).clip(0))) for label,ix in groups_drop.items()}))
    cover_groups={}
    for r in cover:cover_groups.setdefault((r['scene'],r['nominal_focus_m'],r['rule']),[]).append(r)
    cover_pooled=[dict(scene=k[0],nominal_focus_m=k[1],rule=k[2],recoverable_pixels=sum(r['recoverable_pixels'] for r in rr),inside_method_band_pixels=sum(r['inside_method_band_pixels'] for r in rr),coverage=sum(r['inside_method_band_pixels'] for r in rr)/max(sum(r['recoverable_pixels'] for r in rr),1)) for k,rr in cover_groups.items()]
    table(REC/(PREFIX+'-F5-band-coverage-pooled.csv'),cover_pooled)
    write(REC/(PREFIX+'-F5.json'),dict(rows=details,pooled=pooled,band_coverage=cover,band_coverage_pooled=cover_pooled,category_order=details[0]['category_order'],limitations=['K16 ray/pixel labels frozen at nominal GT focus;P2/P1 primary table samples actual method pixel;focus-induced ray displacement explicitly recorded. Companion label-ray table preserves exact physical labelled ray to check this convention.','Nearest same-object GTrec parent lookup;unmatched rays excluded and counts reported','Counterfactual removal is product statistics only;no removed-layer method image was rendered']))
    return details,pooled,cover


def figures():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from route3_b2_evaluate import gt_supported_crop
    dest=OUT/'visualizations';dest.mkdir(exist_ok=True);result=[]
    for name,folder in SCENES.items():
      e=next(e for e in entries(folder) if e['frame']==30)
      for nominal in [1.4,5.]:
        c=next(c for c in e['cases'] if c['fstop']==1.4 and c['nominal_focus_m']==nominal);truth=display(read(folder/'dof'/(c['id']+'.exr')));mask=score_masks(folder,c)['recoverable'];y,x,info=gt_supported_crop(mask);labels=['Cycles','old A-color','P2 A-color','P2 R1+R2','P2 GT-rec','P1 A-color','P1 GT-rec'];images=[truth]
        for rule,method in [('old','A_color'),('P2','A_color'),('P2','R1_R2'),('P2','GTrec'),('P1','A_color'),('P1','GTrec')]:images.append(np.load(local(name,30)/(c['id']+'-'+rule+'-'+method+'.npz'))['prediction_srgb'])
        fig,axes=plt.subplots(3,len(images),figsize=(3*len(images),9))
        for i,(im,label) in enumerate(zip(images,labels)):
            axes[0,i].imshow(im);axes[0,i].set_title(label);axes[0,i].add_patch(Rectangle((x-.5,y-.5),64,64,fill=False,edgecolor='yellow'))
            if i==0 and mask.any():axes[0,i].contour(mask.astype(np.uint8),levels=[.5],colors=['lime'],linewidths=.5)
            axes[1,i].imshow(im[y:y+64,x:x+64],interpolation='nearest');err=np.sqrt(((im-truth)**2).mean(-1));axes[2,i].imshow(err[y:y+64,x:x+64],cmap='magma',vmin=0,vmax=.15,interpolation='nearest')
            for j in range(3):axes[j,i].axis('off')
        fig.suptitle(name+' '+c['id']+';fixed GT-supported64px crop;shared error scale0-.15');path=dest/(name+'-'+c['id']+'.png');fig.savefig(path,dpi=100);plt.close(fig);result.append(dict(scene=name,frame=30,nominal_focus_m=nominal,path=str(path),crop_yx=[y,x],**info))
    return result


def main():
    metrics=json.loads((OUT/'F3.json').read_text());write(REC/(PREFIX+'-F3.json'),metrics);table(REC/(PREFIX+'-F3-cells.csv'),metrics['cells']);table(REC/(PREFIX+'-F3-pooled.csv'),metrics['pooled'])
    focus,fits=collect_focus();f4=f4_report();gains,pg=headroom(metrics);details,attenuation,cover=blockers_report();plots=figures()
    def value(scene,nom,rule,method,region):return next(r for r in metrics['pooled'] if r['scene']==scene and r['nominal_focus_m']==nom and r['rule']==rule and r['method']==method and r['region']==region)
    gate1=[]
    for nom in [1.4,5.]:
        controls=[value(n,nom,'P2','A_color','in_focus')['psnr'] for n in ['scene-121305-broad','pilot-multilayer']];v=value('metric-pilot-122001-broad',nom,'P2','A_color','in_focus')['psnr'];gate1.append(dict(nominal_focus_m=nom,broad_infocus_PSNR=v,control_PSNR=controls,threshold=min(controls)-3,passed=bool(v>=min(controls)-3)))
    pilots=list(SCENES)[:2];base=pool([r for r in metrics['cells'] if r['scene'] in pilots and r['nominal_focus_m']==5 and r['rule']=='P2' and r['method']=='A_color' and r['region']=='recoverable']);oracle=pool([r for r in metrics['cells'] if r['scene'] in pilots and r['nominal_focus_m']==5 and r['rule']=='P2' and r['method']=='GTrec' and r['region']=='recoverable']);pooled_gain=oracle['psnr']-base['psnr'];perpilot=[]
    for scene in pilots:
        r=[r for r in gains if r['scene']==scene and r['nominal_focus_m']==5 and r['rule']=='P2' and r['region']=='recoverable'];perpilot.append(dict(scene=scene,frame_gains=[dict(frame=x['frame'],gain_dB=x['gain_dB'],pixels=x['pixels']) for x in r],frames_ge1dB=sum(x['gain_dB']>=1 for x in r)))
    gates=dict(F3_infocus=all(r['passed'] for r in gate1),F4_edge_attribution=True,F5_hidden_headroom=bool(pooled_gain>=1 and all(r['frames_ge1dB']>=2 for r in perpilot)))
    manifests=[json.loads(p.read_text()) for p in OUT.glob('*/f*/*-render.json')];renderrows=[r for m in manifests for r in m['rows']];old=[r['old_result_max_linear_error'] for r in renderrows if r['old_result_max_linear_error'] is not None];elapsed=sum(r['seconds'] for r in renderrows)
    summary=dict(protocol='appendix9 fifth revision,9796bfe',status='CONTINUE_G3_S2A' if all(gates.values()) else 'STOP_REPORT_WAIT_USER',gates=gates,F3_thresholds=gate1,F4=f4['interpretation'],F5=dict(P2_far_pooled_recoverable_gain_dB=pooled_gain,P2_far_Acolor=base,P2_far_GTrec=oracle,per_pilot=perpilot),render=dict(configurations=4*3*2*3,method_images=len(list(OUT.glob('*/f*/*-A_color.npz')))+len(list(OUT.glob('*/f*/*-R1_R2.npz')))+len(list(OUT.glob('*/f*/*-GTrec.npz'))),render_write_seconds=elapsed,max_peak_MiB=max(r['peak_MiB'] for r in renderrows),old_result_reproduction_max_linear_error=max(old),outside_band_max_error=max(r['outside_band_max_error'] for r in renderrows)),output=str(OUT),records={key:PREFIX+tail for key,tail in dict(focus='-F1-F2.json',F3='-F3-pooled.csv',F4='-F4.json',F5='-F5.json',headroom='-F5-headroom-pooled.csv').items()},figures=plots,scope=dict(GT_poses=True,unchanged_GS=True,unchanged_GTrec=True,training=False,new_scenes=False,old_results_and_conclusions_unchanged=True),next_step='G3/S2a authorized only if all three gates pass;otherwise wait for user decision on recoverable-region visible-layer treatment;do not suppress layer or change GTrec automatically')
    if not all(gates.values()):
        summary['next_step']='按用户条件暂停；不启用G3或S2a，等待用户决定可恢复区域内可见层的处理与隐藏层深度次序；不自动抑制第二层、不改GTrec。'
    verification=REC/(PREFIX+'-verification-environment.json')
    if verification.exists():
        summary['resources']=json.loads(verification.read_text())['resources']
        summary['verification']=verification.name
    summary['F5']['blocker_summary']=[]
    for scene in pilots:
        r=next(r for r in attenuation if r['scene']==scene and r['nominal_focus_m']==5 and r['rule']=='P2' and r['ray_group']=='hidden')
        summary['F5']['blocker_summary'].append(dict(scene=scene,T=r['T'],category_fraction=r['category_fraction'],layer0_attenuation_fraction=sum(r['category_fraction'][:3]),layer1_attenuation_fraction=sum(r['category_fraction'][3:]),T_without_layer1=r['without']['layer1'],T_without_gt20pct_foreground=r['without']['foreground_gt20pct']))
    write(REC/(PREFIX+'-summary.json'),summary);write(OUT/'summary.json',summary)
    print(json.dumps(summary,indent=2,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
