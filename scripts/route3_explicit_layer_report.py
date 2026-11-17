"""Appendix12 fixed-interface tables, guardrails, figures and decision."""
import csv
import json
import math
import shutil
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from route3_explicit_layer import OUT,REC,PREFIX,SCENES,METHODS,fd,cd,load,write,desc_for,reference
from route3_upper_bound_evaluate import pool


def table(suffix,rows):
    if not rows:
        return
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with (REC/(PREFIX+'-'+suffix+'.csv')).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=keys,lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def zp(rr):
    return float(-10*np.log10(max(sum(r['zero_sse'] for r in rr)/(3*sum(r['zero_pixels'] for r in rr)),1e-30)))


def scores():
    rows=json.loads((OUT/'scores.json').read_text())['rows']
    keys=['scene','rule','nominal_focus_m','fstop','method','region']
    groups={}
    for r in rows:
        groups.setdefault(tuple(r[k] for k in keys),[]).append(r)
    pooled=[]
    for key,rr in groups.items():
        n=sum(r['pixels'] for r in rr)
        rs=sum(r['region_zero_sse'] for r in rr)
        pooled.append(dict(zip(keys,key),**pool(rr),zero_PSNR=zp(rr),zero_sse=sum(r['zero_sse'] for r in rr),zero_pixels=sum(r['zero_pixels'] for r in rr),region_zero_sse=rs,region_zero_PSNR=float(-10*np.log10(max(rs/(3*n),1e-30))) if n else None))
    for rule in ['P2','P1']:
        table(rule+'-cells',[r for r in rows if r['rule']==rule])
        table(rule+'-pooled',[r for r in pooled if r['rule']==rule])
    return rows,pooled


CONTRASTS=dict(H_GT=('DGT1','DGT0'),H_C=('C1','A'),H_D=('D1','A'),delta_C=('C1','C0'),delta_D=('D1','D0'),C1_minus_B=('C1','B'),D1_minus_B=('D1','B'),DGT1_minus_D1=('DGT1','D1'),DGT1_minus_A=('DGT1','A'),B_minus_A=('B','A'),Bprime_minus_A=('Bprime','A'))


def contrasts(rows,pooled):
    result=[]
    guards=[]
    for level,data in [('frame',rows),('pooled',pooled)]:
        keys=['scene','rule','nominal_focus_m','fstop','region']+(['frame'] if level=='frame' else [])
        groups={}
        for r in data:
            groups.setdefault(tuple(r[k] for k in keys),{})[r['method']]=r
        for key,lookup in groups.items():
            common=dict(level=level,**dict(zip(keys,key)))
            base=lookup['A']
            for name,(a,b) in CONTRASTS.items():
                aa,bb=lookup[a],lookup[b]
                result.append(dict(**common,contrast=name,pixels=aa['pixels'],gain_dB=aa['psnr']-bb['psnr'] if aa['psnr'] is not None and bb['psnr'] is not None else None,delta_patch_LPIPS=aa['patch_lpips']-bb['patch_lpips'] if aa['patch_lpips'] is not None and bb['patch_lpips'] is not None else None,SSE_ratio=aa['sse']/bb['sse'] if bb['sse'] else None))
            if common['region']=='in_focus':
                for method,r in lookup.items():
                    dz=r['zero_PSNR']-base['zero_PSNR']
                    di=r['psnr']-base['psnr'] if r['psnr'] is not None and base['psnr'] is not None else None
                    guards.append(dict(**common,method=method,diagnostic_only=method.endswith('-black'),zero_PSNR=r['zero_PSNR'],zero_change_dB=dz,zero_drop_flag=dz<-.1,zero_absolute_ge50=r['zero_PSNR']>=50,infocus_change_dB=di,P2_infocus_drop_flag=common['rule']=='P2' and di is not None and di<-.1))
    table('contrasts',result)
    table('guardrails',guards)
    worst=[]
    for method in sorted({r['method'] for r in rows}):
        for field in ['zero_PSNR','psnr']:
            rr=[r for r in rows if r['method']==method and r['rule']=='P2' and r['region']=='in_focus' and r[field] is not None]
            r=min(rr,key=lambda r:r[field])
            worst.append(dict(method=method,field=field,**{k:r[k] for k in ['scene','frame','id','nominal_focus_m','fstop']},worst_value=r[field]))
    table('worst-frame',worst)
    table('seam',[r for r in rows if r['region'] in ['seam','seam_infocus']])
    table('invalid-geometry',[r for r in rows if r['region'] in ['invalid_geometry','invalid_infocus']])
    table('black-diagnostic',[r for r in rows if r['method'].endswith('-black')])
    return result,guards


def metadata():
    prep=[]
    foc=[]
    replacements=[]
    support=[]
    trans=[]
    timing=[]
    for name in SCENES:
      desc=desc_for(name)
      for entry in desc['entries']:
        f=entry['frame']
        p=json.loads((fd(name,f)/'prepared.json').read_text())
        prep.append({k:v for k,v in p.items() if k!='focus'})
        foc.extend(dict(scene=name,frame=f,**r) for r in p['focus'])
        for path in fd(name,f).glob('P*/replacement-*.json'):
            replacements.append(json.loads(path.read_text()))
        for rule in ['P2','P1']:
          for case in entry['cases']:
            folder=cd(name,f,case,rule)
            meta=json.loads((folder/'render.json').read_text())
            common={k:meta[k] for k in ['scene','frame','id','rule','nominal_focus_m','fstop','focus_m']}
            _,regions=reference(name,desc,case,f)
            fill_mask=load(fd(name,f)/'fill-input.npz')['fill_mask']
            recoverable=regions['recoverable']
            outside=recoverable&~fill_mask
            support.append(dict(**common,**{k:meta[k] for k in ['aligned_parents','raw_band_parents','raw_all_recoverable_parents','support_recoverable_fraction','support_recoverable_pixels','recoverable_pixels','aligned_parent_not_behind_explicit_center_fraction']},recoverable_outside_LaMa_mask_pixels=int(outside.sum()),recoverable_outside_LaMa_mask_fraction=float(outside.sum()/max(int(recoverable.sum()),1))))
            timing.extend(dict(**common,**r) for r in meta['rows'])
            t=json.loads((folder/'transmittance.json').read_text())
            for r in t['rows']:
                stat={k:v for k,v in r.items() if k!='quantiles'}
                for qi,q in zip(['q01','q10','q50','q90','q99'],r.get('quantiles') or []):
                    stat[qi]=q
                trans.append(dict(**common,**stat))
    table('preparation',prep)
    table('focus',foc)
    table('replacement',replacements)
    table('support',support)
    table('transmittance',trans)
    table('timing',timing)
    return prep,replacements,support,trans


def pooled_transmittance():
    from route3_step2_diagnostic import numpy_stat
    mapping=dict(A=('A',None),B=('A','aligned'),Bprime=('A','raw'),C0=('C',None),C1=('C','aligned'),D0=('D',None),D1=('D','aligned'),DGT0=('DGT',None),DGT1=('DGT','allraw'))
    rows=[]
    for name in SCENES:
      desc=desc_for(name)
      for rule in ['P2','P1']:
       for nominal in [1.4,5.]:
        for stop in [1.4,2.]:
            parts=[]
            for entry in desc['entries']:
                case=next(c for c in entry['cases'] if c['nominal_focus_m']==nominal and c['fstop']==stop)
                parts.append(load(cd(name,entry['frame'],case,rule)/'prefix-log-components.npz'))
            arrays={key:np.concatenate([p[key] for p in parts]) for key in parts[0]}
            for plane in ['aligned','raw']:
             for domain in ['hidden','control']:
              for method,(vv,hh) in mapping.items():
                logs=arrays[plane+'__visible_'+vv+'__'+domain].astype(np.float64)
                if hh is not None:
                    logs+=arrays[plane+'__hidden_'+hh+'__'+domain]
                stat=numpy_stat(np.exp(-np.minimum(logs,745)))
                rows.append(dict(scene=name,rule=rule,nominal_focus_m=nominal,fstop=stop,method=method,plane=plane,domain=domain,preferred_plane='raw' if method in ['Bprime','DGT0','DGT1'] else 'aligned',**stat))
            del arrays,parts
    table('transmittance-pooled',rows)
    return rows


def upper_absolute(rows,pooled,support):
    """User steering: distinguish H_GT from the DGT0 residual-fill baseline."""
    result=[]
    for level,data in [('frame',rows),('pooled',pooled)]:
        keys=['scene','rule','nominal_focus_m','fstop']+(['frame'] if level=='frame' else [])
        grouped={}
        for r in data:
            if r['region']=='recoverable' and r['method'] in ['A','DGT0','DGT1']:
                grouped.setdefault(tuple(r[k] for k in keys),{})[r['method']]=r
        for key,lookup in grouped.items():
            common=dict(zip(keys,key))
            ss=[r for r in support if all(r[k]==v for k,v in common.items())]
            n=sum(r['recoverable_pixels'] for r in ss)
            outside=sum(r['recoverable_outside_LaMa_mask_pixels'] for r in ss)
            a,b,c=[lookup[k] for k in ['A','DGT0','DGT1']]
            result.append(dict(level=level,**common,pixels=a['pixels'],A_PSNR=a['psnr'],DGT0_PSNR=b['psnr'],DGT1_PSNR=c['psnr'],H_GT_dB=c['psnr']-b['psnr'],DGT1_minus_A_dB=c['psnr']-a['psnr'],A_patch_LPIPS=a['patch_lpips'],DGT0_patch_LPIPS=b['patch_lpips'],DGT1_patch_LPIPS=c['patch_lpips'],recoverable_outside_LaMa_mask_pixels=outside,recoverable_outside_LaMa_mask_fraction=outside/max(n,1)))
    table('upper-absolute',result)
    return result


def zero_absolute(rows,pooled):
    result=[]
    for level,data in [('frame',rows),('pooled',pooled)]:
        keys=['scene','rule','nominal_focus_m','fstop']+(['frame'] if level=='frame' else [])
        groups={}
        for r in data:
            if r['region']=='full_image':
                groups.setdefault(tuple(r[k] for k in keys),{})[r['method']]=r
        for key,lookup in groups.items():
            common=dict(level=level,**dict(zip(keys,key)))
            values={m+'_zero_PSNR':lookup[m]['zero_PSNR'] for m in ['A','D0','D1','C0','C1','DGT0','DGT1']}
            for m in ['D0','D1','C0','C1','DGT0','DGT1']:
                values[m+'_minus_A_zero_dB']=lookup[m]['zero_PSNR']-lookup['A']['zero_PSNR']
            for a,b in [('C1','C0'),('D1','D0'),('DGT1','DGT0')]:
                values[a+'_minus_'+b+'_zero_dB']=lookup[a]['zero_PSNR']-lookup[b]['zero_PSNR']
            result.append(dict(**common,**values))
    table('zero-absolute',result)
    return result


def zero_diagnostics():
    spatial=[]
    depth=[]
    for path in sorted(OUT.glob('*/f*/P*/*/zero-diagnostics.json')):
        d=json.loads(path.read_text())
        common={k:d[k] for k in ['scene','frame','id','rule','nominal_focus_m','fstop']}
        for r in d['rows']:
            for row in r['spatial']:
                spatial.append(dict(level='frame',**common,pair=r['pair'],total_net_SSE_delta=r['net_SSE_delta'],total_positive_SSE_delta=r['positive_SSE_delta'],**row))
            for row in r['depth']:
                out=dict(level='frame',**common,pair=r['pair'],**{k:v for k,v in row.items() if not isinstance(v,dict)})
                for label in ['front','within3percent','behind']:
                    out.update({label+'_'+k:v for k,v in row[label].items()})
                for label in ['relative_depth','weight_mean_relative_depth']:
                    out[label+'_mean']=row[label].get('mean')
                    out[label+'_median']=row[label].get('median')
                    out[label+'_quantiles']=row[label].get('quantiles')
                depth.append(out)
    keys=['scene','rule','nominal_focus_m','fstop','pair','category']
    groups={}
    for r in spatial:
        groups.setdefault(tuple(r[k] for k in keys),[]).append(r)
    pp=[]
    for key,rr in groups.items():
        sums={k:sum(r[k] for r in rr) for k in ['pixels','positive_pixels','old_SSE','new_SSE','net_SSE_delta','positive_SSE_delta','negative_SSE_delta','total_net_SSE_delta','total_positive_SSE_delta']}
        pp.append(dict(level='pooled',**dict(zip(keys,key)),**sums,positive_increment_fraction=sums['positive_SSE_delta']/sums['total_positive_SSE_delta'] if sums['total_positive_SSE_delta'] else None,signed_net_fraction=sums['net_SSE_delta']/sums['total_net_SSE_delta'] if abs(sums['total_net_SSE_delta'])>1e-12 else None))
    table('zero-spatial',spatial+pp)
    groups={}
    for r in depth:
        groups.setdefault(tuple(r[k] for k in keys),[]).append(r)
    dp=[]
    for key,rr in groups.items():
        sums={k:sum(r[k] for r in rr) for k in ['positive_pixels','valid_hidden_and_explicit_reference_pixels','no_hidden_contribution_pixels','outside_explicit_coverage_pixels','hidden_optical_total_weight','valid_positive_SSE_total']}
        out=dict(level='pooled',**dict(zip(keys,key)),**sums)
        for label in ['front','within3percent','behind']:
            count=sum(r[label+'_pixel_count'] for r in rr)
            out[label+'_pixel_count']=count
            out[label+'_pixel_fraction']=count/max(sums['valid_hidden_and_explicit_reference_pixels'],1)
            out[label+'_positive_SSE_fraction']=sum(r[label+'_positive_SSE_fraction']*r['valid_positive_SSE_total'] for r in rr)/max(sums['valid_positive_SSE_total'],1e-30)
            out[label+'_hidden_optical_weight_fraction']=sum(r[label+'_hidden_optical_weight_fraction']*r['hidden_optical_total_weight'] for r in rr)/max(sums['hidden_optical_total_weight'],1e-30)
        dp.append(out)
    table('zero-hidden-depth',depth+dp)
    return pp,dp


def primary_zero_summary(spatial,depth):
    result=[]
    for pair in ['C1-C0','D1-D0','DGT1-DGT0']:
        ss=[r for r in spatial if r['scene'] in SCENES[:2] and r['rule']=='P2' and r['nominal_focus_m']==5 and r['fstop']==1.4 and r['pair']==pair]
        dd=[r for r in depth if r['scene'] in SCENES[:2] and r['rule']=='P2' and r['nominal_focus_m']==5 and r['fstop']==1.4 and r['pair']==pair and r['category']=='all']
        positive=sum(r['positive_SSE_delta'] for r in ss)
        valid=sum(r['valid_positive_SSE_total'] for r in dd)
        n=sum(r['valid_hidden_and_explicit_reference_pixels'] for r in dd)
        spatial_share={category:sum(r['positive_SSE_delta'] for r in ss if r['category']==category)/max(positive,1e-30) for category in sorted({r['category'] for r in ss})}
        bins={}
        for label in ['front','within3percent','behind']:
            bins[label]=dict(pixel_fraction=sum(r[label+'_pixel_count'] for r in dd)/max(n,1),positive_SSE_fraction=sum(r[label+'_positive_SSE_fraction']*r['valid_positive_SSE_total'] for r in dd)/max(valid,1e-30),hidden_optical_weight_fraction=sum(r[label+'_hidden_optical_weight_fraction']*r['hidden_optical_total_weight'] for r in dd)/max(sum(r['hidden_optical_total_weight'] for r in dd),1e-30))
        result.append(dict(pair=pair,scope='two G2 pilots pooled,3frames,P2 far f1.4;zero aperture input-reconstruction error',net_SSE_delta=sum(r['net_SSE_delta'] for r in ss),positive_SSE_delta=positive,spatial_positive_SSE_fraction=spatial_share,depth_diagnostic_positive_SSE_coverage=valid/max(positive,1e-30),depth=bins))
    write(REC/(PREFIX+'-zero-primary-summary.json'),dict(rows=result,settings=PREFIX+'-zero-diagnostic-settings.json'))
    return result


def decision(rows,guards):
    primary=[r for r in rows if r['scene'] in SCENES[:2] and r['rule']=='P2' and r['nominal_focus_m']==5 and r['fstop']==1.4 and r['region']=='recoverable']
    lookup={(r['scene'],r['frame'],r['method']):r for r in primary}
    gates=[]
    for name in ['H_GT','H_C','H_D','delta_C','delta_D']:
        a,b=CONTRASTS[name]
        aa=pool([r for r in primary if r['method']==a])
        bb=pool([r for r in primary if r['method']==b])
        gain=aa['psnr']-bb['psnr']
        ff=[]
        counts={}
        for scene in SCENES[:2]:
            for f in [1,30,60]:
                ff.append(dict(scene=scene,frame=f,gain_dB=lookup[scene,f,a]['psnr']-lookup[scene,f,b]['psnr']))
            counts[scene]=sum(r['gain_dB']>=1 for r in ff if r['scene']==scene)
        gates.append(dict(contrast=name,pooled_gain_dB=gain,pooled_delta_LPIPS=aa['patch_lpips']-bb['patch_lpips'],frame_gains=ff,frames_ge1dB=counts,gain_gate=gain>=1 and all(v>=2 for v in counts.values())))
    guardstatus=[]
    for method in ['C1','D1']:
        gg=[r for r in guards if r['level']=='frame' and r['method']==method]
        b2=[r for r in rows if r['scene']=='B2-high' and r['rule']=='P2' and r['region'] in ['full_image','in_focus','recoverable'] and r['method'] in [method,'A']]
        bb={(r['frame'],r['nominal_focus_m'],r['fstop'],r['region']):r for r in b2 if r['method']=='A'}
        changes=[dict(frame=r['frame'],nominal_focus_m=r['nominal_focus_m'],fstop=r['fstop'],region=r['region'],gain_dB=r['psnr']-bb[r['frame'],r['nominal_focus_m'],r['fstop'],r['region']]['psnr']) for r in b2 if r['method']==method]
        bad=[r for r in changes if r['gain_dB']<-.1]
        guardstatus.append(dict(method=method,zero_flag_cells=sum(r['zero_drop_flag'] for r in gg),P2_infocus_flag_cells=sum(r['P2_infocus_drop_flag'] for r in gg),worst_zero_PSNR=min(r['zero_PSNR'] for r in gg),B2_high_changes=changes,B2_high_drop_flag_cells=len(bad),guardrails_pass=not any(r['zero_drop_flag'] or r['P2_infocus_drop_flag'] for r in gg) and not bad))
    by={r['contrast']:r for r in gates}
    gs={r['method']:r for r in guardstatus}
    if not by['H_GT']['gain_gate']:
        status='STOP_HGT_INSUFFICIENT_USER_SCENE_DECISION'
    elif by['H_C']['gain_gate'] and gs['C1']['guardrails_pass']:
        status='STOP_HC_PASS_WAIT_USER_INTERFACE_CONFIRMATION'
    elif by['H_D']['gain_gate'] and not by['H_C']['gain_gate']:
        status='STOP_HD_ONLY_REPORT_SEAMS_AND_GUARDS_USER_DECISION'
    elif by['delta_C']['gain_gate'] or by['delta_D']['gain_gate']:
        status='STOP_HIDDEN_GAIN_BUT_VISIBLE_OR_FILL_LIMITS_USER_DECISION'
    else:
        status='STOP_ESTIMATED_GEOMETRY_EXPLICIT_INTERFACE_LIMITED_DIRECTION_DECISION'
    result=dict(status=status,gates=gates,guards=guardstatus,not_training=True,not_enter_G3_S2a=True,gate_scope='P2 G2 far f1.4 recoverable,pooled SSE and each scene>=2/3 frames>=1dB;guards all reported configs,zero both protocols,P2infocus;B2High report all frame/regional drops',B2_guard_definition='same0.1dB reference magnitude;no separately relaxed visible-drop tolerance')
    write(OUT/'decision.json',result)
    write(REC/(PREFIX+'-decision.json'),result)
    return result


def figures():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from route3_b2_evaluate import gt_supported_crop
    out=OUT/'visualizations'
    out.mkdir(exist_ok=True)
    result=[]
    methods=['A','B','C0','C1','D0','D1','DGT1']
    for name in SCENES:
      desc=desc_for(name)
      entry=next(r for r in desc['entries'] if r['frame']==30)
      for nominal in [1.4,5.]:
        case=next(c for c in entry['cases'] if c['nominal_focus_m']==nominal and c['fstop']==1.4)
        folder=cd(name,30,case,'P2')
        truth,regions=reference(name,desc,case,30)
        y,x,audit=gt_supported_crop(regions['recoverable'])
        images=[truth]+[load(folder/(m+'.npz'))['prediction_srgb'] for m in methods]
        labels=['Cycles']+methods
        fig,axes=plt.subplots(3,len(images),figsize=(22,8),squeeze=False)
        vmax=max(float(np.max(abs(v[y:y+64,x:x+64]-truth[y:y+64,x:x+64]).mean(-1))) for v in images)
        for j,(v,label) in enumerate(zip(images,labels)):
            axes[0,j].imshow(v)
            axes[0,j].add_patch(Rectangle((x,y),64,64,fill=False,edgecolor='yellow',linewidth=1))
            axes[0,j].contour(regions['recoverable'],levels=[.5],colors='cyan',linewidths=.3)
            axes[0,j].set_title(label)
            axes[1,j].imshow(v[y:y+64,x:x+64],interpolation='nearest')
            axes[2,j].imshow(abs(v[y:y+64,x:x+64]-truth[y:y+64,x:x+64]).mean(-1),cmap='magma',vmin=0,vmax=max(vmax,1e-6),interpolation='nearest')
        for ax in axes.ravel():
            ax.axis('off')
        fig.suptitle(f'{name}, frame30, P2 N1.4 nominal F{nominal:g}; GT-selected crop ({y},{x}); error shared [0,{vmax:.3f}]')
        fig.tight_layout()
        path=out/f'{name}-F{nominal:g}.png'
        fig.savefig(path,dpi=140)
        plt.close(fig)
        result.append(dict(scene=name,frame=30,nominal_focus_m=nominal,path=str(path),crop_yx=[y,x],**audit))
        maps=load(folder/'zero-error-maps.npz')
        fig,axes=plt.subplots(2,4,figsize=(16,8),squeeze=False)
        maximum=max(float(np.maximum(maps[k+'_SSE_delta'],0).max()) for k in ['C','D','DGT'])
        for j,label in enumerate(['C','D','DGT']):
            positive=np.maximum(maps[label+'_SSE_delta'],1e-12)
            im=axes[0,j].imshow(np.log10(positive),cmap='magma',vmin=-12,vmax=max(math.log10(max(maximum,1e-12)),-11.9))
            axes[0,j].set_title(label+'1-'+label+'0: log10 positive SSE',fontsize=11)
            dep=axes[1,j].imshow(maps[label+'_hidden_dominant_relative_depth'],cmap='coolwarm',vmin=-.2,vmax=.2)
            axes[1,j].set_title('Dominant hidden Δz/z (clip ±0.2)',fontsize=11)
        axes[0,3].imshow(maps['categories'],cmap=plt.get_cmap('tab10',4),vmin=0,vmax=3)
        axes[0,3].set_title('Categories 0 / 1 / 2 / 3',fontsize=11)
        axes[1,3].imshow(load(entry['input_npz'])['rgb_srgb'])
        axes[1,3].set_title('Input pinhole RGB')
        for ax in axes.ravel():
            ax.axis('off')
        fig.colorbar(im,ax=list(axes[0,:3]),fraction=.02,pad=.01)
        fig.colorbar(dep,ax=list(axes[1,:3]),fraction=.02,pad=.01)
        fig.suptitle(f'{name} frame30 P2 N1.4 F{nominal:g}: zero-aperture diagnostics only')
        fig.text(.5,.025,'0: foreground boundary ±2px; 1: invalid geometry; 2: large occluder interior; 3: other. Depth: strongest composited hidden Gaussian.',ha='center',fontsize=10)
        path=out/f'{name}-F{nominal:g}-zero-diagnostic.png'
        fig.savefig(path,dpi=140)
        plt.close(fig)
        result.append(dict(scene=name,frame=30,nominal_focus_m=nominal,path=str(path),type='zero-SSE spatial and hidden-depth statistics;no new render group'))
    return result


def main():
    rows,pooled=scores()
    cc,guards=contrasts(rows,pooled)
    prep,replacements,support,trans=metadata()
    pooled_transmittance()
    absolute=upper_absolute(rows,pooled,support)
    zero_abs=zero_absolute(rows,pooled)
    zero_spatial,zero_depth=zero_diagnostics()
    combined_zero=primary_zero_summary(zero_spatial,zero_depth)
    dec=decision(rows,guards)
    plots=figures()
    usage=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())
    result=dict(protocol='appendix12 tenth revision,d6175d3',environment_verification=PREFIX+'-environment-verification.json',status=dec['status'],validation=json.loads((OUT/'validation-E2.json').read_text()),upper_gate=json.loads((OUT/'upper-gate.json').read_text()),decision=dec,preparation=prep,figures=plots,combined_G2_zero_primary=combined_zero,zero_absolute_far_primary=[r for r in zero_abs if r['level']=='pooled' and r['rule']=='P2' and r['nominal_focus_m']==5 and r['fstop']==1.4],zero_spatial_far_primary=[r for r in zero_spatial if r['rule']=='P2' and r['nominal_focus_m']==5 and r['fstop']==1.4],zero_hidden_depth_far_primary=[r for r in zero_depth if r['rule']=='P2' and r['nominal_focus_m']==5 and r['fstop']==1.4 and r['category']=='all'],upper_absolute_table=PREFIX+'-upper-absolute.csv',upper_absolute_far_primary=[r for r in absolute if r['level']=='pooled' and r['rule']=='P2' and r['nominal_focus_m']==5 and r['fstop']==1.4],output=str(OUT),rows=len(rows),rendered_cases=72,groups=METHODS,black_groups=['C0','C1','D0','D1'],reference='Cycles,PSNR and fixed64px AlexNetLPIPS0.1;GT score regions unchanged',zero_and_infocus_measured=True,old_results_untouched=True,SHARP_weights_unchanged=True,no_training=True,no_new_scenes_or_Blender_GT=True,resources=dict(data_free_GiB=shutil.disk_usage(ROOT).free/2**30,new_outputs_GiB=usage/2**30),limits=['Three existing controlled scenes,E1 geometry with given GT poses;not real-video robustness or HiddenQuery recovery evidence.','Explicit depths come from A dominant contribution clusters;invalid pixels retain largest cluster and remain separately reported.','All groups,including DGT,share A scalar focus and band,method-only LaMa fill and estimated background depth.','Aligned hidden placement uses fixed appendix10 fit/support;no GT depth fallback,clamp or post-render image pasteback.','Footprints use deterministic pixel-center3sigma ellipses plus Euclidean CoC dilation on padded canvas;overlap/seam errors explicitly reported.','Black diagnostic exactly subtracts per-ray residual-fill contribution from the same accumulated image;not a separate method.','Transmittance is diagnostic at continuous projected placed-parent centers using frozen K16 label correspondences,not proof of image quality.'],next_step='Pause for user decision under appendix12;do not automatically generate G3,train S2a,alter geometry/opacity or tune compositor')
    write(OUT/'summary.json',result)
    write(REC/(PREFIX+'-summary.json'),result)
    print(json.dumps(dec,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
