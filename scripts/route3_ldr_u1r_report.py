"""Summarize cached R1/R2 measurements without changing any decision."""
import csv
import datetime
import json
import os
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from route3_ldr_u1r import OUT,PREFIX,RECORDS,CRITERIA,write


def read(suffix):return json.loads((RECORDS/(PREFIX+'-'+suffix+'.json')).read_text())


def difference(left,right):
    return dict(PSNR_gain_dB=left['psnr']-right['psnr'] if left['pixels'] else None,
                LPIPS_change=left['patch_lpips']-right['patch_lpips'] if left['pixels'] else None)


def main():
    refs=read('references');r1=read('R1-evaluation');r2=read('R2-evaluation');rad=read('radiance')
    optional=read('HDR-evaluation') if (RECORDS/(PREFIX+'-HDR-evaluation.json')).exists() else None
    dest=OUT/'convergence';dest.mkdir(exist_ok=False);fits=[]
    source=json.loads((RECORDS/'route3-upper-bound-2026-11-11-U1-f001.json').read_text())
    for frame in [1,30,60]:
        fit=source if frame==1 else read(f'R2-f{frame:03d}');rows=fit['rows'];best=-float('inf');drop=[];consecutive=0;maximum=0
        for row in rows:
            value=row['metrics']['heldout']['psnr'];best=max(best,value);d=best-value;drop.append(d)
            consecutive=consecutive+1 if d>=.5 else 0;maximum=max(maximum,consecutive)
        assert rows[-1]['step']==8000 and maximum<3
        fig,axes=plt.subplots(2,2,figsize=(11,7))
        for key,title in [('train','training monitor17/65'),('heldout','heldout16/16')]:
            axes[0,0].plot([x['step'] for x in rows],[x['metrics'][key]['psnr'] for x in rows],'-o',label=title)
            axes[0,1].plot([x['step'] for x in rows],[x['metrics'][key]['loss'] for x in rows],'-o',label=title)
        axes[0,0].set_ylabel('Pinhole PSNR (dB)');axes[0,1].set_ylabel('Loss');axes[0,1].set_yscale('log')
        axes[1,0].plot([x['step'] for x in rows],[x['train_minus_heldout_PSNR_dB'] for x in rows],'-o');axes[1,0].set_ylabel('Train-heldout gap, report only (dB)')
        axes[1,1].plot([x['step'] for x in rows],drop,'-o');axes[1,1].axhline(.5,color='r',linestyle='--');axes[1,1].set_ylabel('Heldout decline from historical best (dB)')
        for ax in axes.flat:ax.set_xlabel('Step');ax.grid(alpha=.2)
        axes[0,0].legend();axes[0,1].legend();fig.suptitle(f'Frame{frame}:fixed8000 steps'+(' (reused)' if frame==1 else ''))
        fig.tight_layout();path=dest/f'f{frame:03d}.png';fig.savefig(path,dpi=130);plt.close(fig)
        fits.append(dict(frame=frame,reused=frame==1,checkpoint_step=8000,stop_triggered=False,max_consecutive_declines=maximum,
            initial_heldout_PSNR=rows[0]['metrics']['heldout']['psnr'],final_heldout_PSNR=rows[-1]['metrics']['heldout']['psnr'],
            final_training_monitor_PSNR=rows[-1]['metrics']['train']['psnr'],final_gap_dB=rows[-1]['train_minus_heldout_PSNR_dB'],
            train_monitor_views=17,train_views=65,heldout_views=16,wall_seconds=fit['wall_seconds'],
            peak_allocated_MiB=fit['peak_allocated_MiB'],config=fit['config'],convergence_figure=str(path)))
    comparisons={};mechanism={};tables=[]
    regions=['full_image','in_focus','out_of_focus','band','recoverable']
    for stage,evaluation in [('R1',r1),('R2',r2)]+([('HDR_descriptive',optional)] if optional else []):
        comparisons[stage]={}
        for reference,focuses in evaluation['pooled'].items():
            comparisons[stage][reference]={}
            for focus,table in focuses.items():
                comparisons[stage][reference][focus]={}
                names=['O3','O5','L2','L3','descriptive_O3_outside_L2_inside'] if stage=='R1' else (['O3','L2','HDR_input_descriptive'] if stage=='HDR_descriptive' else ['O3','L2','U1','U1_visible','U1_clip','U1_clip_visible'])
                for method in names:
                    comparisons[stage][reference][focus][method]={reg:difference(table[method][reg],table['O3'][reg]) for reg in regions}
                    for reg in regions:
                        value=table[method][reg];delta=difference(value,table['O3'][reg])
                        tables.append(dict(stage=stage,reference=reference,focus_m=focus,method=method,region=reg,pixels=value['pixels'],PSNR=value['psnr'],patch_LPIPS=value['patch_lpips'],**delta))
    for ref,primary,control in [('Cycles','U1','U1_visible'),('LDR','U1_clip','U1_clip_visible')]:
        mechanism[ref]={focus:{reg:difference(table[primary][reg],table[control][reg]) for reg in regions} for focus,table in r2['pooled'][ref].items()}
    radiance=rad['rows'];resources={}
    for path in ['/','/root/autodl-tmp']:
        value=os.statvfs(path);resources[path]=dict(available_bytes=value.f_bavail*value.f_frsize,available_GiB=value.f_bavail*value.f_frsize/2**30)
    high={}
    for focus in [1.4,5.]:
        ids={f'B2-high-f{frame:03d}-N{stop:g}-F{focus:g}' for frame in [1,30,60] for stop in [1.4,2.]}
        selected=[x for x in r1['highlight_influence'] if x['id'] in ids]
        high[str(focus)]=dict(total_pixels=sum(x['pixels'] for x in selected),fraction=sum(x['pixels'] for x in selected)/(6*512**2),
            per_case_fraction_range=[min(x['fraction'] for x in selected),max(x['fraction'] for x in selected)])
    report=dict(status='completed_pause_for_user',recorded_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        criteria=dict(path=str(CRITERIA.relative_to(ROOT)),commit='c18e323',modified=False),implementation_commit='287996b',
        scope=dict(scene='B2-high',frames=[1,30,60],focuses_m=[1.4,5.],fstops=[1.4,2.],groups_per_focus=6),
        reference_definition=json.loads(CRITERIA.read_text())['reference'],reference_check=refs['frame1_check'],reference_noise=refs['independent_noise_pairs'],
        formal_noise=refs['max_noise_by_region'],highlight_influence=high,
        R1=dict(completed=True,decisions=r1['decisions'],pooled=r1['pooled'],per_aperture=r1['per_aperture'],
                all_methods_and_U2_descriptive_result=PREFIX+'-R1-evaluation.json',U2_historical_decisions_unchanged=True,visualizations=r1['visualizations']),
        R2=dict(completed=True,frames=fits,decisions=r2['decisions'],pooled=r2['pooled'],per_aperture=r2['per_aperture'],
                mechanism_same_optimized_visible_without_hidden=mechanism,radiance_analysis=radiance,visualizations=r2['visualizations'],
                final_step_only=True,no_checkpoint_selected_by_heldout=True),
        optional_HDR_input=dict(completed=optional is not None,descriptive_not_method=True,formal_gate=False,
            pooled=optional['pooled']['Cycles'] if optional else None,visualizations=optional['visualizations'] if optional else []),
        comparisons_vs_O3=comparisons,resources_after=resources,output=str(OUT.relative_to(ROOT)),
        unchanged=['B2/L1/L2/L3/U0/U1/U2 historical metrics anddecisions','all historical frozencriteria','original frame1 checkpoint'],
        no_new_Blender_or_DA3_inference=True,no_feedforward_network_training=True,
        limits=['one controlledmain scene,3frames,6groups/focus;correlated samples,notcrossscene evidence',
                'LDRconsistent reference evaluatescapped radiance,not trueHDRhighlight fidelity',
                'U1 isper-scene upperproxy withadditional trainingviews,not feedforwardattainable performance',
                'U1-clip iscolorclipping ofsame fit,not aseparately fitted LDRrepresentation;opacity/colorcoupling may matter',
                'learnedcolors>1 do not bythemselves provephysicalHDRrecovery',
                'O5/O6 notperfect opticalOracles;descriptive splice/HDRinput are notmethods orformalcriteria'],
        next_step='Pause;user decides whether toenter training oruse2.5D visible+3Dhidden/adjustcontribution;no automatic continuation')
    extra=[]
    for variant in ['E1','E2']:
        for focus,values in r1['pooled']['LDR'].items():
            for method in ['O3','O5','ours']:
                for region in regions:
                    v=values['U2_'+method+'_'+variant][region]
                    extra.append(dict(reference='LDR',variant=variant,focus_m=focus,method=method,region=region,
                        pixels=v['pixels'],PSNR=v['psnr'],patch_LPIPS=v['patch_lpips']))
    with (RECORDS/(PREFIX+'-U2-descriptive.csv')).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(extra[0]),lineterminator='\n');writer.writeheader();writer.writerows(extra)
    report['R1']['U2_descriptive_table']=PREFIX+'-U2-descriptive.csv'
    write(RECORDS/(PREFIX+'-summary.json'),report)
    with (RECORDS/(PREFIX+'-report-tables.csv')).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(tables[0]),lineterminator='\n');writer.writeheader();writer.writerows(tables)
    print(json.dumps(dict(summary=PREFIX+'-summary.json',fits=[{k:x[k] for k in ['frame','final_heldout_PSNR','final_gap_dB']} for x in fits],decisions=r2['decisions'])),flush=True)


if __name__=='__main__':main()
