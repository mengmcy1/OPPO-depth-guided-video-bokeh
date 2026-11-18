"""Export stage 1a milestone tables and the footprint/Query run record."""
import csv
import json
from collections import Counter
import shutil
import subprocess

import numpy as np

from route3_ovg_subpixel import CONFIG, OUT, PREVIOUS, ROOT, save_json

REC=ROOT/'03_实验记录/records'
PREFIX='route3-ovg-1a-2026-11-18'


def read(path):
    return json.loads(path.read_text())


def size(path):
    return int(subprocess.check_output(['du','-sB1',str(path)],text=True).split()[0])


def tabular(stem,rows):
    path=REC/(PREFIX+stem+'.csv')
    with path.open('w') as f:
        fields=list(dict.fromkeys(k for row in rows for k in row))
        writer=csv.DictWriter(f, fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    return str(path.relative_to(ROOT))


def plot(training,validation):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    fig,ax=plt.subplots(1,2,figsize=(10,4))
    for tier in training['counts']:
        rows=[r for r in training['curve'] if r['tier']==tier]
        width=25
        vals=[r['total'] for r in rows]
        offset=width//2
        smooth=np.convolve(vals,np.ones(width)/width,mode='valid')
        ax[0].plot([r['step'] for r in rows[offset:offset+len(smooth)]],smooth,label=tier)
    for entry in validation:
        label=entry['step']
        for row in entry['pooled']:
            if row['scene']=='pooled' and row['method']=='OVG-Net':
                ax[1].scatter(label,row['front_CoC_mae_px'],color='C0',label='Front' if label==validation[0]['step'] else None)
                ax[1].scatter(label,row['boundary_front_CoC_mae_px'],color='C1',label='Boundary front' if label==validation[0]['step'] else None)
    ax[0].set(xlabel='Optimizer step',ylabel='Direct loss (25 batches per tier)',title='Training by supervision tier')
    ax[1].set(xlabel='Optimizer step',ylabel='CoC MAE (512 px scale)',title='Development-set geometry')
    for a in ax:a.legend()
    fig.tight_layout()
    p=OUT/'training-curves.png'
    fig.savefig(p,dpi=150)
    plt.close(fig)
    return str(p.relative_to(ROOT))


def label_record():
    data=read(OUT/'B/manifest.json')
    thin=read(OUT/'B/thin-check.json')
    auxiliary=read(OUT/'B/thin-2x/summary.json')
    rows=[]
    for r in thin['rows']:
        phase=r['interior_phase_RMSE'] if r['phase_check_applicable'] else None
        # The previously measured interior phase error exists at frame 30.
        # Keep a single measured value, separate from 1/30/60 mesh checks.
        rows.append(dict({k:v for k,v in r.items()
                          if k not in ('texture_supersample_reference','interior_phase_RMSE','shift')},
                         shift_x=r['shift'][0],shift_y=r['shift'][1],
                         interior_phase_RMSE=phase if r['frame']==30 and r['shift'][0]==0 else None,
                         interior_phase_pass=(phase<=.01) if phase is not None and r['frame']==30 and r['shift'][0]==0 else None))
    thin_path=tabular('-thin-check',rows)
    sparse=[r['interior_phase_RMSE'] for r in rows if r['interior_phase_RMSE'] is not None]
    aux_path=REC/(PREFIX+'-thin-2x.json')
    save_json(aux_path,auxiliary)
    result=dict(scenes=data['scenes'],literature_terms='geometric coverage from visible pinhole subrays, not a matte; separate K16 masks',
             split=data['split'], thresholds=data['policy'],label_frames=sum(len(s['rows']) for s in data['scenes']),
             disk_bytes=size(OUT/'B'),thin=dict(scene_count=len({r['scene'] for r in rows}),
                    contour_rows=len(rows),eligible_contour_rows=sum(r['contour_8_pass'] is not None for r in rows),
                    passing_contour_rows=sum(r['contour_8_pass'] is True for r in rows),
                    max_8_to_16_alpha_MAE=max(r['contour_8_mae'] or 0 for r in rows),
                    max_interior_phase_RMSE=max(sparse),phase_failure_count=sum(x>.01 for x in sparse),
                    check_table=thin_path,aux_2x=str(aux_path.relative_to(ROOT)),added_to_training=False),
             B_single_batch=read(OUT/'W1-single-batch.json'),
             input_provenance='all three actual B priors, SHARP calibration/dominant depth, raw official DepthSplat, observed RAFT triangulation; RGB and given E1 poses',
             geometry_context='Static B/C/TartanAir clips; motion mask all static by capture definition, no GT motion masks input')
    return result


def public_data_record():
    tartan=read(OUT/'G/acquisition.json')
    card_path=next((OUT/'C/windows').glob('*/manifest.json'))
    card=read(card_path)
    return dict(G=dict(acquisition=tartan,priors=read(OUT/'G/priors.json'),
                      download_bytes=sum(x['HTTP_range_bytes'] for x in tartan['sources']),
                      disk_bytes=size(OUT/'G'),uses='geometry supervision in research, CC BY 4.0 attribution'),
                C=dict(sequence=card,priors=read(OUT/'C/priors.json'),
                       mesh_check=read(OUT/'C/mesh-check.json'),disk_bytes=size(OUT/'C'),
                       source_license='CC0 Poly Haven diffuse photographs',license='https://polyhaven.com/license',
                       parsed_fields='peeled surface/depth with mask, K16 independent traces and source-frame bits, subpixel layers and alpha, rigid surface ID'))


def query_record():
    folder=OUT/'W4'
    decision=read(folder/'decision.json')
    audits=[]
    for p in sorted((folder/'footprint-labels').glob('*/audit.json')):
        a=read(p)
        audits.extend(dict(scene=a['scene'],**row) for row in a['rows'])
    train=read(folder/'retrained/training.json')
    eval_=read(folder/'retrain-evaluation.json')
    selected=read(PREVIOUS/'split.json')
    if decision['selected']=='O1':
        assert all((folder/'retrain-data/learning-data'/s/'256-5').is_symlink() for s in selected['train'])
    label_count=Counter()
    for r in audits:
        label_count.update({k:r[k] for k in ('true_background','dropped_true_background','shrunk_true_background')})
    return dict(decision=decision,placement=dict(counts=label_count,
                  dropped_fraction=label_count['dropped_true_background']/max(label_count['true_background'],1),
                  labels_total=len(audits),file=tabular('-W4-placement',audits),resolution='256/5 Query training labels',
                  selected_labels='center-O1 immutable references' if decision['selected']=='O1' else 'new footprint labels'),
                retrained=dict({k:train[k] for k in ('steps','seconds','validation','seed','trainable_parameters','GT_warmup_steps','peak_allocated_MiB')},
                    probe_initial=float(np.mean(train['train_probe'][0]['losses'])),
                    probe_final=float(np.mean(train['train_probe'][-1]['losses'])),
                    checkpoint=str((folder/'retrained/step-020000.pt').relative_to(ROOT)),
                    evaluation=eval_,curve=tabular('-W4-Query-training',train['curve'])))


def report():
    w1=label_record()
    w2=public_data_record()
    w4=query_record()
    training=read(OUT/'1a-training.json')
    all_dev=[read(OUT/'validation'/f'step-{k:06d}.json') for k in CONFIG['checkpoint_steps']]
    final=all_dev[-1]
    assert final['frames']==39 and all(r['frames']==39 for r in all_dev)
    curve=[dict({k:v for k,v in row.items() if k!='support'},
                **{'support_'+k:v for k,v in row['support'].items()}) for row in training.pop('curve')]
    path=tabular('-training-curve',curve)
    development=dict(pooled=tabular('-dev-pooled',final['pooled']),
                     per_frame=tabular('-dev-per-frame',final['per_frame']),
                     paired=tabular('-dev-paired',final['paired']),
                     per_checkpoint=tabular('-dev-checkpoints',[dict(step=r['step'],**item) for r in all_dev for item in r['pooled']]),
                     key_rows=final['pooled'],paired_rows=final['paired'],
                     geometry_supervision='first visible surface, two supported depths only where labelled, pre-aperture geometric coverage only')
    w3=REC/(PREFIX+'-independent-test.json')
    seal=read(w3)
    assert len(seal['three_dimensional'])==18 and len(seal['card'])==1
    training.update(dict(curve=path,plot=plot(dict(training,curve=curve),all_dev),configuration=CONFIG,
                         GT_geometry_warmup=False,dev_in_training=False,stage1b_losses_enabled=False))
    summary=dict(code_baseline=CONFIG['protocol_revision'],input_previous=CONFIG['previous'],
                 output=CONFIG['output'],W1=w1,W2=w2,
                 W3=dict(path=str(w3.relative_to(ROOT)),Blender_scene_count=18,card_sequences=1,
                         rendering_started=False,training_card_assets_disjoint=True),
                 W4=w4,milestone=dict(checkpoint=training['validations'][-1]['checkpoint'],
                                      training=training,development=development),
                 resources=dict(data_free_GiB=shutil.disk_usage(OUT).free/2**30,new_outputs_GiB=size(OUT)/2**30,
                                GPU='NVIDIA RTX 4090 D',nvidia_GPU_context='W4 ran after the 1a GPU work; thin auxiliary renders overlapped small Query fit'),
                 interpretation='three synthetic development scenes with supplied poses; 1a measures geometry only; checkpoint kept regardless of dev scores')
    save_json(REC/(PREFIX+'-summary.json'),summary)


if __name__=='__main__':
    report()
