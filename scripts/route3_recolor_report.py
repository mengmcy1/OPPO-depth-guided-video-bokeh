"""Appendix11 P0 validation-gated stop and P1 full-scope result tables."""
import argparse,csv,json,sys,math
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from route3_recolor import OUT,REC,PREFIX,C,PREV,SCENES,METHODS,REGIONS,desc_for,fd,cd,load,write,weighted_stats
from route3_step2_diagnostic import numpy_stat
from route3_upper_bound_evaluate import pool
from route3_s2a_proxy import display

def table(path,rows):
    if not rows:return
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=keys,lineterminator='\n');w.writeheader();w.writerows(rows)

def quality_pool(rows):
    n=sum(r['pixels'] for r in rows);ss=sum(r['sse'] for r in rows)
    return dict(pixels=n,sse=ss,psnr=-10*math.log10(max(ss/max(3*n,1),1e-30)) if n else None)

def p0_report():
    cases=[];matching=[];prepared=[];errors={}
    for name in SCENES:
      desc=desc_for(name)
      for e in desc['entries']:
        f=e['frame'];identity=load(fd(name,f)/'identity-colors.npz');path=desc['source_root']/'visible'/f'f{f:03d}.pt' if desc['source_root'] is not None else ROOT/'outputs/route3-sharp-s1c-2026-11-10-01/anchor'/e['id']/'A_color/attributes-final.pt';colors=torch.load(path,map_location='cpu',weights_only=True)['colors'].numpy();errors[name,f]=(abs(colors-identity['oracle_color']).mean(-1),abs(display(colors)-display(identity['oracle_color'])).mean(-1))
        prepared.append(json.loads((fd(name,f)/'identity-colors.json').read_text()))
        for c in e['cases']:
          matching.append(json.loads((fd(name,f)/'K16'/c['id']/'matching.json').read_text()))
          for rule in ['P2','P1']:
            p=json.loads((cd(name,f,c,rule)/'P0.json').read_text());common={k:p[k] for k in ['scene','frame','id','nominal_focus_m','fstop','rule']}
            for method,regions,zero in [('A_color',p['baseline_regions'],p['baseline_zero_full'])]+[(r['method'],r['regions'],r['zero_full']) for r in p['rows']]:
                rr=next((r for r in p['rows'] if r['method']==method),None)
                for region,v in regions.items():cases.append(dict(**common,method=method,region=region,**v,zero_psnr=zero['psnr'],zero_sse=zero['sse'],zero_pixels=zero['pixels'],outside_MAE=rr['outside_change']['MAE'] if rr else 0.,outside_fraction_gt1_255=rr['outside_change']['fraction_pixels_max_change_gt_1_255'] if rr else 0.,zero_outside_MAE=rr['zero_outside_change']['MAE'] if rr else 0.))
    keys=['scene','nominal_focus_m','fstop','rule','method','region'];groups={}
    for r in cases:groups.setdefault(tuple(r[k] for k in keys),[]).append(r)
    pooled=[]
    for key,rr in groups.items():
        zz=quality_pool([dict(pixels=r['zero_pixels'],sse=r['zero_sse']) for r in rr]);pooled.append(dict(zip(keys,key),**quality_pool(rr),zero_psnr=zz['psnr']))
    lookup={tuple(r[k] for k in ['scene','nominal_focus_m','fstop','rule','region']):r for r in pooled if r['method']=='A_color'}
    for r in pooled:
        b=lookup[tuple(r[k] for k in ['scene','nominal_focus_m','fstop','rule','region'])];r['gain_dB']=r['psnr']-b['psnr'];r['zero_change_dB']=r['zero_psnr']-b['zero_psnr']
    matchgroups={}
    for r in matching:matchgroups.setdefault((r['scene'],r['nominal_focus_m'],r['fstop']),[]).append(r)
    optical=[]
    for (name,nominal,stop),rr in matchgroups.items():
        rays=[];em={label:{space:[] for space in ['linear','display']} for label in ['main','aux']};wm={label:[] for label in ['main','aux']}
        for row in rr:
            v=load(row['path']);rays.append(v['ray_statistics']);lin,disp=errors[name,row['frame']]
            for label in ['main','aux']:
                w=v['gaussian_optical_weight'];mask=v[label+'_mask']&(w>0);wm[label].append(w[mask]);em[label]['linear'].append(lin[mask]);em[label]['display'].append(disp[mask])
        a=np.concatenate(rays);den=np.maximum(a[:,0],1e-30);r=dict(scene=name,nominal_focus_m=nominal,fstop=stop,rays=len(a),top8_match_fraction=float(a[:,7].sum()/max(a[:,6].sum(),1)),top8_any_match_fraction=float((a[:,7]>0).mean()),top8_depth_close_wrong_identity_raw_fraction=float(a[:,11].sum()/max(a[:,6].sum(),1)),top8_depth_close_wrong_identity_aligned_fraction=float(a[:,10].sum()/max(a[:,6].sum(),1)))
        for label,column in [('all_match',1),('color_available',2),('main',3),('close_wrong_raw',4),('close_wrong_aligned',5)]:
            stat=numpy_stat(a[:,column]/den);r[label+'_optical_mean']=stat['mean'];r[label+'_optical_median']=stat['median'];r[label+'_optical_q01_q10_q50_q90_q99']=stat['quantiles']
        for label in ['main','aux']:
            for space in ['linear','display']:
                stat=weighted_stats(np.concatenate(em[label][space]),np.concatenate(wm[label]));r[label+'_'+space+'_error_mean']=stat['mean'];r[label+'_'+space+'_error_quantiles']=stat['quantiles']
        optical.append(r)
    table(REC/(PREFIX+'-P0-cells.csv'),cases);table(REC/(PREFIX+'-P0-pooled.csv'),pooled);table(REC/(PREFIX+'-P0-optical.csv'),optical);write(REC/(PREFIX+'-P0-identities.json'),dict(rows=prepared,matching=matching,pooled=optical,GT_assignment='sourcepixel identity Oracle,not a geometric accuracy claim for displaced learned centers',color='same LDRinputlinearRGB,other12frames only;GT-consistent source interpolation mayretain originalpixel antialiasing'))
    def gain(scenes,method):
        def select(m):return [r for r in cases if r['scene'] in scenes and r['method']==m and r['rule']=='P2' and r['nominal_focus_m']==5 and r['fstop']==1.4 and r['region']=='recoverable']
        return quality_pool(select(method))['psnr']-quality_pool(select('A_color'))['psnr']
    check=json.loads((OUT/'P0-single-frame-check.json').read_text());anchor=gain(['B2-high'],'Oracle_main');gains={m:gain(SCENES[:2],m) for m in ['Oracle_main','Oracle_aux']};trusted=check['passed'] and anchor>=2;early=trusted and all(v<.5 for v in gains.values());decision=dict(single_frame_check=check,B2high_P2_far_main_expected_gain_dB=anchor,B2high_anchor_pass=anchor>=2,P0_trusted_for_early_stop=bool(trusted),G2_P2_far_expected_gains_dB=gains,early_stop=bool(early),next_step='stop P0,do notrenderP1' if early else 'P1 required;failedanchororcheckdoesnotpermitearlystop',prediction_method='K64fixedsampleweights times globallyfixedperGaussian RGBdelta;K16onlyselects/reportsidentities',guardrail_unchanged=.1)
    write(OUT/'P0-decision.json',decision);write(REC/(PREFIX+'-P0-decision.json'),decision);print(json.dumps(decision,ensure_ascii=False,indent=2),flush=True);return decision

def p1_report():
    rows=json.loads((OUT/'P1-scores.json').read_text())['rows'];groups={};keys=['scene','nominal_focus_m','fstop','rule','method','region']
    for r in rows:groups.setdefault(tuple(r[k] for k in keys),[]).append(r)
    pooled=[]
    for key,rr in groups.items():
        z=quality_pool([dict(pixels=r['zero_pixels'],sse=r['zero_sse']) for r in rr]);pooled.append(dict(zip(keys,key),**pool(rr),zero_PSNR=z['psnr'],worst_frame_zero_PSNR=min(r['zero_PSNR'] for r in rr)))
    guard=[];differences=[]
    for level,data in [('frame',rows),('pooled',pooled)]:
        kk=['scene','nominal_focus_m','fstop','rule','region']+(['frame'] if level=='frame' else []);lookup={tuple(r[k] for k in kk):r for r in data if r['method']=='A_color'}
        for r in data:
            base=lookup[tuple(r[k] for k in kk)];common=dict(level=level,**{k:r[k] for k in kk},method=r['method']);dz=r['zero_PSNR']-base['zero_PSNR'];dd=r['psnr']-base['psnr'];differences.append(dict(**common,gain_dB=dd,delta_LPIPS=r['patch_lpips']-base['patch_lpips'],zero_change_dB=dz))
            if r['region']=='in_focus':guard.append(dict(**common,zero_PSNR=r['zero_PSNR'],zero_change_dB=dz,zero_drop_flag=dz<-.1,zero_absolute_ge50=r['zero_PSNR']>=50,P2_infocus_change_dB=dd if r['rule']=='P2' else None,P2_infocus_drop_flag=r['rule']=='P2' and dd<-.1))
    for rule in ['P2','P1']:
        table(REC/(PREFIX+'-'+rule+'-cells.csv'),[r for r in rows if r['rule']==rule]);table(REC/(PREFIX+'-'+rule+'-pooled.csv'),[r for r in pooled if r['rule']==rule])
    table(REC/(PREFIX+'-gains.csv'),differences);table(REC/(PREFIX+'-guardrails.csv'),guard)
    worst=[]
    for method in METHODS:
        for scene in SCENES:
            for f in [1,30,60]:
                rr=[r for r in rows if r['method']==method and r['scene']==scene and r['frame']==f and r['region']=='in_focus'];worst.append(dict(method=method,scene=scene,frame=f,worst_zero_PSNR=min(r['zero_PSNR'] for r in rr),worst_zero_change_dB=min(r['zero_change_dB'] for r in differences if r['level']=='frame' and r['method']==method and r['scene']==scene and r['frame']==f),worst_P2_infocus_gain_dB=min(r['gain_dB'] for r in differences if r['level']=='frame' and r['method']==method and r['scene']==scene and r['frame']==f and r['rule']=='P2' and r['region']=='in_focus')))
    table(REC/(PREFIX+'-worst-frame.csv'),worst)
    retrieval=[];outside=[]
    for p in sorted(OUT.glob('*/f*/P*/*/P1.json')):
        d=json.loads(p.read_text());common={k:d[k] for k in ['scene','frame','id','nominal_focus_m','fstop','rule']};retrieval.append(dict(**common,**{k:v for k,v in d['retrieval'].items() if not isinstance(v,dict)},confidence_mean=d['retrieval']['confidence']['mean'],confidence_median=d['retrieval']['confidence']['median'],confidence_quantiles=d['retrieval']['confidence']['quantiles'],source_count_mean=d['retrieval']['source_count']['mean'],source_count_quantiles=d['retrieval']['source_count']['quantiles']))
        for method in METHODS:
            rr=json.loads((p.parent/(method+'-scores.json')).read_text());outside.append(dict(**common,method=method,**{'dof_'+k:v for k,v in rr['outside_change'].items()},**{'zero_'+k:v for k,v in rr['zero_outside_change'].items()}))
    table(REC/(PREFIX+'-retrieval.csv'),retrieval)
    table(REC/(PREFIX+'-outside-changes.csv'),outside)
    def select(method,scenes,focus=5):return [r for r in rows if r['method']==method and r['scene'] in scenes and r['rule']=='P2' and r['nominal_focus_m']==focus and r['fstop']==1.4 and r['region']=='recoverable']
    def gain(method,scenes,focus=5):return pool(select(method,scenes,focus))['psnr']-pool(select('A_color',scenes,focus))['psnr']
    reports=[]
    for method in METHODS[1:]:
        per=[]
        for scene in SCENES[:2]:
            rr=[r for r in differences if r['level']=='frame' and r['scene']==scene and r['method']==method and r['rule']=='P2' and r['nominal_focus_m']==5 and r['fstop']==1.4 and r['region']=='recoverable'];per.append(dict(scene=scene,frame_gains=[dict(frame=r['frame'],gain_dB=r['gain_dB']) for r in rr],frames_ge1dB=sum(r['gain_dB']>=1 for r in rr)))
        gg=[r for r in guard if r['level']=='frame' and r['method']==method and r['rule']=='P2' and r['fstop']==1.4];h=gain(method,SCENES[:2]);reports.append(dict(method=method,G2_far_gain_dB=h,G2_near_gain_dB=gain(method,SCENES[:2],1.4),B2high_far_gain_dB=gain(method,['B2-high']),per_pilot=per,gain_gate=bool(h>=1 and all(r['frames_ge1dB']>=2 for r in per)),guardrails_pass=all(not r['zero_drop_flag'] and not r['P2_infocus_drop_flag'] for r in gg),zero_flag_cells=sum(r['zero_drop_flag'] for r in gg),infocus_flag_cells=sum(r['P2_infocus_drop_flag'] for r in gg),worst_zero_PSNR=min(r['zero_PSNR'] for r in gg)))
    m=next(r for r in reports if r['method']=='Oracle_main');a=next(r for r in reports if r['method']=='Oracle_aux');r=next(r for r in reports if r['method']=='Retrieval');control=next(r for r in reports if r['method']=='Control')
    if m['gain_gate'] and m['guardrails_pass'] and not control['gain_gate']:status='STOP_USER_CONFIRM_COLOR_NETWORK' if r['G2_far_gain_dB']>0 else 'STOP_USER_DECIDE_SELECTION_OR_COLOR_TRAINING'
    elif a['gain_gate'] and not m['gain_gate']:status='STOP_AUX_ONLY_REPORT_GUARD_COST'
    elif m['G2_far_gain_dB']<=0 and a['G2_far_gain_dB']<=0:status='STOP_COLOR_ROUTE_STRATEGIC_DECISION'
    else:status='STOP_BELOW_GATE_OR_GUARDRAIL_USER_DECISION'
    actual_diff=[abs(r['actual_minus_expected_PSNR']) for r in rows if r['actual_minus_expected_PSNR'] is not None];zero_diff=[abs(r['zero_actual_minus_expected_PSNR']) for r in rows if r.get('zero_actual_minus_expected_PSNR') is not None];decision=dict(status=status,methods=reports,control_equivalent_definition='sameG2gain_gate;alsoallnumericgainsreported,no newmargintuned',max_actual_vs_P0_PSNR_difference_dB=max(actual_diff),max_zero_actual_vs_P0_PSNR_difference_dB=max(zero_diff) if zero_diff else None,max_alpha_change=max(r['alpha_max_change'] for r in rows),main_vs_control_G2_far_dB=m['G2_far_gain_dB']-control['G2_far_gain_dB'],selection_gap_G2_far_dB=m['G2_far_gain_dB']-next(x['G2_far_gain_dB'] for x in reports if x['method']=='Oracle_retrieval_set'),color_gap_G2_far_dB=next(x['G2_far_gain_dB'] for x in reports if x['method']=='Oracle_retrieval_set')-r['G2_far_gain_dB'],not_training=True,geometry_opacity_count_unchanged=True)
    write(OUT/'P1-decision.json',decision);write(REC/(PREFIX+'-P1-decision.json'),decision);print(json.dumps(decision,ensure_ascii=False,indent=2),flush=True);return decision

def figures():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from route3_local_replacement import reference
    from route3_b2_evaluate import gt_supported_crop
    dest=OUT/'visualizations';dest.mkdir(exist_ok=True);plots=[]
    for name in SCENES:
      desc=desc_for(name);entry=next(e for e in desc['entries'] if e['frame']==30)
      for nominal in [1.4,5.]:
        case=next(c for c in entry['cases'] if c['nominal_focus_m']==nominal and c['fstop']==1.4);folder=cd(name,30,case,'P2');truth,regions=reference(name,desc,case,30);y,x,choice=gt_supported_crop(regions['recoverable']);images=[truth]+[np.load(folder/(m+'.npz'))['prediction_srgb'] for m in METHODS];labels=['Cycles','A-color','Oracle-main','Oracle-aux','Oracle@retrieval','Retrieval','Control'];fig,ax=plt.subplots(3,len(images),figsize=(21,9))
        for col,(im,label) in enumerate(zip(images,labels)):
            ax[0,col].imshow(im);ax[0,col].set_title(label);ax[0,col].add_patch(Rectangle((x-.5,y-.5),64,64,edgecolor='yellow',fill=False))
            if col==0 and regions['recoverable'].any():ax[0,col].contour(regions['recoverable'].astype(np.uint8),levels=[.5],colors=['lime'],linewidths=.5)
            ax[1,col].imshow(im[y:y+64,x:x+64],interpolation='nearest');err=np.sqrt(((im-truth)**2).mean(-1));ax[2,col].imshow(err[y:y+64,x:x+64],cmap='magma',vmin=0,vmax=.15,interpolation='nearest')
            for row in range(3):ax[row,col].axis('off')
        fig.suptitle(name+' '+case['id']+' P2;GT-supported64pxcrop;shared error0-.15');path=dest/(name+'-F'+str(nominal)+'.png');fig.savefig(path,dpi=110);plt.close(fig);plots.append(dict(scene=name,nominal_focus_m=nominal,path=str(path),crop_yx=[y,x],**choice))
        if nominal==5:
            match=load(fd(name,30)/'K16'/case['id']/'matching.npz');ray=match['ray_statistics'];pixel=match['ray_pixel'];den=np.bincount(pixel,weights=ray[:,0],minlength=512**2);allmatch=np.bincount(pixel,weights=ray[:,1],minlength=512**2)/np.maximum(den,1e-30);main=np.bincount(pixel,weights=ray[:,3],minlength=512**2)/np.maximum(den,1e-30);top=np.bincount(pixel,weights=ray[:,7],minlength=512**2)/np.maximum(np.bincount(pixel,weights=ray[:,6],minlength=512**2),1)
            fig,ax=plt.subplots(1,4,figsize=(16,4));ax[0].imshow(truth);ax[0].set_title('Cycles / GT support')
            if regions['recoverable'].any():ax[0].contour(regions['recoverable'].astype(np.uint8),levels=[.5],colors=['lime'],linewidths=.6)
            for j,(a,title) in enumerate([(allmatch,'GT identity matched weight / alpha'),(main,'Main matched weight / alpha'),(top,'Top8 identity match fraction')],1):
                masked=np.ma.masked_where(den.reshape(512,512)==0,a.reshape(512,512));im=ax[j].imshow(masked,vmin=0,vmax=1,cmap='viridis',interpolation='nearest');ax[j].set_title(title)
            for a in ax:a.axis('off')
            fig.colorbar(im,ax=ax[1:].tolist(),fraction=.02);fig.suptitle(name+' f30,F5,N1.4;P0physicalK16labels;undefined outside GT hidden-ray domain');mp=dest/(name+'-P0-optical.png');fig.savefig(mp,dpi=120);plt.close(fig);plots.append(dict(scene=name,type='P0-optical',path=str(mp)))
    return plots

def finalize():
    p0=json.loads((OUT/'P0-decision.json').read_text());p1=json.loads((OUT/'P1-decision.json').read_text()) if (OUT/'P1-decision.json').exists() else None
    plots=figures() if p1 else [];previous_errors=[]
    for p in OUT.glob('*/f*/P*/*/P1.json'):
        d=json.loads(p.read_text());previous_errors.extend(r['Acolor_previous_max_linear_error'] for r in d['rows'] if r['Acolor_previous_max_linear_error'] is not None)
    summary=dict(protocol='appendix11 ninthrevision,9028497',P0=p0,P1=p1,status='STOP_P0' if p0['early_stop'] else p1['status'],outputs=str(OUT),matching_source='GTidentities assigned to768sourcepixels:layer0visible,layer1firstpeeled;not actuallearnedgeometrytruth',optics='unchanged appendix9/10R0P2/P1focus andmethodband,K64fixed;K16onlyidentitymatching',color='other12inputs,existingLDRlinearRGB;no hidden targetcolor inretrieval',statistics='top8 sortedbyactualcompositedweight;GTmatch3targetGTpixelfootprints;rawandlocallyaligneddepthclose5%diagnostic',no_training=True,geometry_opacity_count_unchanged=True,old_outputs_and_decisions_unchanged=True,Acolor_previous_max_linear_error=max(previous_errors) if previous_errors else None,records=dict(P0='-P0-pooled.csv',optical='-P0-optical.csv',P2='-P2-pooled.csv',P1='-P1-pooled.csv',gains='-gains.csv',retrieval='-retrieval.csv',guardrails='-guardrails.csv',worst='-worst-frame.csv'),figures=plots,next_step='按判定暂停，等待用户决定；不自动训练或扩充场景。',scope='five existingcontrolledscenes,E1givenGTposes;sourcepixelGTidentity andobservedimagecolors areOracleproxy,not perfectopticalupperbound')
    summary['records']={k:PREFIX+v for k,v in summary['records'].items()};write(OUT/'summary.json',summary);write(REC/(PREFIX+'-summary.json'),summary);print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['p0','p1','finalize']);a=p.parse_args();dict(p0=p0_report,p1=p1_report,finalize=finalize)[a.mode]()
