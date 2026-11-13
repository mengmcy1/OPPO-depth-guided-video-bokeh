"""Existing-output D report: exact pooled T quantiles and GT-selected crops."""
import argparse,json,csv,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from route3_step2_diagnostic import C,OUT,REC,PREFIX,descriptor,references,original,write,numpy_stat

def main(evaluation_mask=False,stats_only=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from route3_b2_evaluate import gt_supported_crop
    from matplotlib.patches import Rectangle
    render=json.loads((OUT/'render-relative.json').read_text());metrics=json.loads((REC/(PREFIX+'-D-metrics-relative.json')).read_text());dest=OUT/'D-figures';dest.mkdir(exist_ok=True);pooled=[];figures=[];perframe=[];region_cache={};tfig,axes=plt.subplots(5,2,figsize=(12,17),squeeze=False)
    for ri,name in enumerate(C['scenes']):
      desc=descriptor(name);fg_ids=np.load(desc['scene']/'scene-geometry.npz')['foreground_object_ids']
      for ci,focus in enumerate([1.4,5.]):
        ax=axes[ri,ci]
        for geometry,color in [('focus-only','tab:blue'),('foreground','tab:orange'),('full-GT','tab:green')]:
            values={k:[] for k in ['hidden','foreground_blocked_control']}
            for e in desc['entries']:
                f=e['frame'];a=np.load(desc['geometry']/f'f{f:03d}-N1.4-F{focus:g}-K16.npz');row=next(r for r in render['transmittance'] if r['scene']==name and r['frame']==f and r['geometry']==geometry and r['nominal_focus_m']==focus);T=np.load(row['path'])['transmittance'];hidden=a['hidden']&a['recoverable']&~a['residual_pixels'][None]
                if evaluation_mask:
                    key=(name,f,focus)
                    if key not in region_cache:
                        case=next(c for c in e['cases'] if c['fstop']==1.4 and c['nominal_focus_m']==focus);region_cache[key]=references(name,case,desc)[1]['recoverable']
                    hidden&=region_cache[key][None]
                control=np.isin(a['object_id'],fg_ids)&~a['hidden']&hidden.any(0)[None];values['hidden'].append(T[hidden]);values['foreground_blocked_control'].append(T[control]);perframe.append(dict(scene=name,frame=f,focus_m=focus,geometry=geometry,hidden=numpy_stat(T[hidden]),foreground_blocked_control=numpy_stat(T[control]),mask_scope='exact frozen scoring recoverable region' if evaluation_mask else 'all mesh recoverable hidden rays'))
            for group,parts in values.items():
                v=np.concatenate(parts);pooled.append(dict(scene=name,focus_m=focus,geometry=geometry,group=group,**numpy_stat(v)));hist,edges=np.histogram(v,bins=np.linspace(0,1,101));cdf=np.cumsum(hist)/max(len(v),1);ax.plot(edges[1:],cdf,color=color,ls='-' if group=='hidden' else '--',label=geometry+(' hidden' if group=='hidden' else ' blocked'))
        ax.set(title=f'{name}; F={focus:g}',xlabel='T before hidden surface',ylabel='CDF',xlim=(0,1),ylim=(0,1));ax.legend(fontsize=7)
        if stats_only:continue
        e=next(x for x in desc['entries'] if x['frame']==30);case=next(c for c in e['cases'] if c['fstop']==1.4 and c['nominal_focus_m']==focus);refs,regions=references(name,case,desc);truth=refs['Cycles'];y,x,selection=gt_supported_crop(regions['recoverable']);images={'Cycles':truth,'original A':np.load(original(name,case)['A_color'])['prediction_srgb']}
        for geo in ['focus-only','foreground','full-GT']:
            row=next(r for r in render['rows'] if r['scene']==name and r['id']==case['id'] and r['geometry']==geo and r['method']=='A_color');images[geo+' A']=np.load(row['path'])['prediction_srgb']
        row=next(r for r in render['rows'] if r['scene']==name and r['id']==case['id'] and r['geometry']=='full-GT' and r['method']=='GTrec');images['full-GT + hidden']=np.load(row['path'])['prediction_srgb'];fig,ax2=plt.subplots(2,len(images),figsize=(3*len(images),6),squeeze=False)
        for col,(label,im) in enumerate(images.items()):ax2[0,col].imshow(im[y:y+64,x:x+64],interpolation='nearest');ax2[0,col].set_title(label);err=np.sqrt(np.mean((im-truth)**2,-1));ax2[1,col].imshow(err[y:y+64,x:x+64],vmin=0,vmax=.25,cmap='magma');ax2[0,col].axis('off');ax2[1,col].axis('off')
        fig.suptitle(name+' '+case['id']+';fixed colors;relative layers;GT-selected crop;shared RMSE0-.25');fig.tight_layout();path=dest/(name+'-'+case['id']+'.png');fig.savefig(path,dpi=120);plt.close(fig);fig,ax3=plt.subplots(figsize=(5,5));ax3.imshow(truth);ax3.contour(regions['recoverable'].astype(int),levels=[.5],colors=['lime'],linewidths=.5);ax3.add_patch(Rectangle((x-.5,y-.5),64,64,fill=False,edgecolor='yellow'));ax3.axis('off');loc=path.with_name(path.stem+'-locator.png');fig.savefig(loc,dpi=110);plt.close(fig);figures.append(dict(path=str(path),locator=str(loc),scene=name,focus_m=focus,crop_yx=[y,x],**selection))
    suffix='-evaluation-region' if evaluation_mask else '';tfig.tight_layout();tpath=dest/('prefix-transmittance-CDF'+suffix+'.png');tfig.savefig(tpath,dpi=110);plt.close(tfig)
    for tag,rows in [('T-pooled'+suffix,pooled)]:
        with (REC/(PREFIX+'-'+tag+'.csv')).open('w') as h:
            w=csv.DictWriter(h,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    write(REC/(PREFIX+'-T-pooled'+suffix+'.json'),dict(rows=pooled,per_frame=perframe,plot=str(tpath),definition='exact pooled quantiles from corresponding K16 rays and3frames;CDF is binned100bins;blocked control plane defined in D2 record',mask_scope='exact frozen scoring recoverable region' if evaluation_mask else 'all mesh recoverable hidden rays'))
    if stats_only:print('D_T_STATS',len(pooled),flush=True);return
    # Compact report of primary N1.4 pooled in-focus and recoverable effects.
    brief=[];data=metrics['pooled']
    for name in C['scenes']:
      for focus in [1.4,5.]:
       for geometry in ['original','focus-only','foreground','full-GT']:
        for region in ['full_image','in_focus','band','recoverable']:
            get=lambda method,sigma:next(x for x in data if x['scene']==name and x['focus_m']==focus and x['geometry']==geometry and x['region']==region and x['reference']=='Cycles' and x['sigma_px']==sigma and x['method']==method)
            a=get('A_color',0);g=get('GTrec',0);b=get('A_color',2);brief.append(dict(scene=name,focus_m=focus,geometry=geometry,region=region,Acolor_PSNR_dB=a['psnr'],Acolor_patch_LPIPS=a['patch_lpips'],GTrec_PSNR_dB=g['psnr'],GTrec_patch_LPIPS=g['patch_lpips'],GTrec_gain_dB=g['psnr']-a['psnr'] if a['pixels'] else None,sigma2_Acolor_PSNR_dB=b['psnr'],sigma2_error_fraction=b['sse']/a['sse'] if a['sse'] else None))
    with (REC/(PREFIX+'-D-brief.csv')).open('w') as h:w=csv.DictWriter(h,fieldnames=list(brief[0]),lineterminator='\n');w.writeheader();w.writerows(brief)
    write(REC/(PREFIX+'-D-figures.json'),dict(figures=figures,transmittance_CDF=str(tpath),initial_collapsed_geometry_not_used=True,relative_depth_edit_still_changes_pinhole_sorting=True));print('D_REPORT',len(brief),len(pooled),len(figures),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--evaluation-mask',action='store_true');p.add_argument('--stats-only',action='store_true');a=p.parse_args();main(a.evaluation_mask,a.stats_only)
