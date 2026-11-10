"""Analyze sparse B1 samples; never launches full-sequence rendering."""
import argparse
import csv
import json
import os
from pathlib import Path
import time

os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_exr(path):
    value=cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if value is None:raise ValueError(f'Cannot load EXR: {path}')
    value=value[...,:3][...,::-1].astype(np.float64)
    if not np.isfinite(value).all():raise ValueError(f'Nonfinite EXR: {path}')
    return value


def luma(value):
    return value@np.array([.2126,.7152,.0722])


def metrics(image, reference, roi_reference=None):
    delta=image-reference; lum=luma(reference if roi_reference is None else roi_reference)
    gradient=np.hypot(cv2.Sobel(lum,cv2.CV_64F,1,0,ksize=3)/8,
                      cv2.Sobel(lum,cv2.CV_64F,0,1,ksize=3)/8)
    edge=cv2.dilate((gradient>=.05).astype(np.uint8),np.ones((7,7),np.uint8))>0
    highlight=cv2.dilate((lum>=1.).astype(np.uint8),np.ones((13,13),np.uint8))>0
    yy,xx=np.mgrid[:lum.shape[0],:lum.shape[1]]
    a=luma(image)*highlight;b=luma(reference)*highlight
    mass_a=float(a.sum());mass_b=float(b.sum())
    ca=np.array([(a*xx).sum(),(a*yy).sum()])/max(mass_a,1e-12)
    cb=np.array([(b*xx).sum(),(b*yy).sum()])/max(mass_b,1e-12)
    return dict(linear_rgb_rmse=float(np.sqrt(np.mean(delta**2))),
                boundary_linear_rgb_rmse=float(np.sqrt(np.mean(delta[edge]**2))) if edge.any() else None,
                boundary_pixels=int(edge.sum()),highlight_pixels=int(highlight.sum()),
                highlight_integrated_luminance_relative_error=abs(mass_a-mass_b)/max(mass_b,1e-12) if highlight.any() else None,
                highlight_luminance_centroid_displacement_px=float(np.linalg.norm(ca-cb)) if highlight.any() else None)


def passes(m):
    return (m['linear_rgb_rmse']<=.005 and m['boundary_linear_rgb_rmse'] is not None
            and m['boundary_linear_rgb_rmse']<=.01
            and m['highlight_integrated_luminance_relative_error'] is not None
            and m['highlight_integrated_luminance_relative_error']<=.02
            and m['highlight_luminance_centroid_displacement_px']<=.25)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path);p.add_argument('--record',type=Path,required=True)
    a=p.parse_args();a.output=a.output.resolve();rows=json.loads((a.output/'timings.json').read_text())
    # Default seed establishes a nested-sample convergence curve. An independent
    # terminal seed is required before accepting 1024 as sufficiently stable.
    mainrows=[r for r in rows if r['resolution']==[512,512] and r['device']=='GPU' and not r.get('pinhole',False) and r.get('seed',20261110)==20261110]
    groups=sorted({(r['frame'],r['fstop'],r['focus_m']) for r in mainrows});results=[];checks=[];recommended={}
    fig,axes=plt.subplots(1,2,figsize=(11,4.3),constrained_layout=True)
    for frame,fstop,focus in groups:
        group=sorted([r for r in mainrows if (r['frame'],r['fstop'],r['focus_m'])==(frame,fstop,focus)],key=lambda r:r['samples'])
        refrow=next((r for r in group if r['samples']==1024),None)
        if refrow is None:continue
        reference=load_exr(a.output/(refrow['key']+'.exr'))
        independent=next((r for r in rows if r['resolution']==[512,512] and r['device']=='GPU' and not r.get('pinhole',False) and r.get('seed')==20261111 and (r['frame'],r['fstop'],r['focus_m'])==(frame,fstop,focus) and r['samples']==1024),None)
        terminal=metrics(load_exr(a.output/(independent['key']+'.exr')),reference) if independent else None
        check=dict(frame=frame,fstop=fstop,focus_m=focus,reference_samples=1024,independent_seed=20261111,
                   independent_1024_metrics=terminal,independent_terminal_passed=passes(terminal) if terminal else False)
        ext_default=next((r for r in group if r['samples']==2048),None)
        ext_other=next((r for r in rows if r.get('seed')==20261111 and r['samples']==2048 and (r['frame'],r['fstop'],r['focus_m'])==(frame,fstop,focus)),None)
        if ext_default and ext_other:
            ext_metrics=metrics(load_exr(a.output/(ext_other['key']+'.exr')),load_exr(a.output/(ext_default['key']+'.exr')),roi_reference=reference)
            check.update(independent_2048_metrics=ext_metrics,independent_terminal_passed=passes(ext_metrics),terminal_samples=2048)
        else:check['terminal_samples']=1024
        checks.append(check);curve=[]
        for r in group:
            m=metrics(load_exr(a.output/(r['key']+'.exr')),reference)
            value={**r,**m,'reference_key':refrow['key'],'reference_comparison_passed':passes(m)};results.append(value);curve.append(value)
        valid=[r['samples'] for r in curve if r['samples']<1024 and r['reference_comparison_passed']]
        if check['terminal_samples']==2048:
            recommendation=2048 if check['independent_terminal_passed'] else None
        else:
            recommendation=min(valid) if valid and check['independent_terminal_passed'] else (1024 if check['independent_terminal_passed'] else None)
        check['candidate_min_samples']=recommendation
        label=f'f{frame} N{fstop} F{focus}'
        for ax,key in zip(axes,['linear_rgb_rmse','boundary_linear_rgb_rmse']):
            xx=[r['samples'] for r in curve if r['samples']<1024];yy=[r[key] for r in curve if r['samples']<1024]
            line,=ax.plot(xx,yy,'o-',label=label)
            if terminal:ax.scatter([1024],[terminal[key]],marker='x',s=45,color=line.get_color())
            if 'independent_2048_metrics' in check:ax.scatter([2048],[check['independent_2048_metrics'][key]],marker='D',s=35,color=line.get_color())
        recommended.setdefault(str(fstop),[]).append(recommendation)
    for ax,title,threshold in zip(axes,['Whole image linear RGB RMSE','Boundary linear RGB RMSE'],[.005,.01]):
        ax.axhline(threshold,color='black',linestyle='--',linewidth=1,label='proposed tolerance')
        ax.set_xscale('log',base=2);ax.set_yscale('log');ax.set_xticks([128,256,512,1024,2048],labels=['128','256','512','1024','2048'])
        ax.set_xlabel('Samples/pixel (cross/diamond: independent seeds)');ax.set_ylabel('RMSE');ax.set_title(title);ax.grid(alpha=.2)
    axes[1].legend(fontsize=7,ncol=2);fig.savefig(a.output/'sampling-convergence.png',dpi=170);plt.close(fig)
    candidate={k:max(v) if all(x is not None for x in v) else None for k,v in recommended.items()}
    result=dict(protocol=dict(color_space='Unclipped scene-linear RGB float EXR; no PNG/sRGB error measurement',
             reference='1024 default-seed rendering; not noise-free GT; terminal checked against independent seed1024, with selective 2048 independent-seed extension where required',
             thresholds=dict(linear_rgb_rmse=.005,boundary_rmse=.01,highlight_relative_luminance=.02,highlight_centroid_px=.25),
             boundary='reference luminance Sobel gradient magnitude>=0.05, 3px square dilation',
             highlight='reference luminance>=1, 6px square dilation; all fixed ROIs shared across SPP',
             limitation='Two representative frames, finite stochastic samples, simple procedural scene; B1 tolerances remain proposed, not formal B2 acceptance.'),
             terminal_checks=checks,candidate_samples_by_fstop=candidate,all_terminal_checks_passed=bool(checks) and all(c['independent_terminal_passed'] for c in checks),
             rows=results,convergence_plot=str(a.output/'sampling-convergence.png'))
    a.record.parent.mkdir(parents=True,exist_ok=True);a.record.write_text(json.dumps(result,indent=2)+'\n')
    csvpath=a.record.with_suffix('.csv')
    if results:
        keys=[k for k in results[0] if k!='resolution']
        with csvpath.open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=keys,lineterminator='\n');writer.writeheader();writer.writerows({k:r.get(k) for k in keys} for r in results)
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','protocol')},indent=2))


if __name__=='__main__':main()
