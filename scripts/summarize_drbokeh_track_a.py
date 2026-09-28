"""Validate and summarize the single Dr.Bokeh result; paper values are external references only."""
import argparse
import json
import time
from pathlib import Path
from datetime import datetime,timezone

import numpy as np
from PIL import Image


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,default=Path('outputs/drbokeh-track-a-20260928'))
    p.add_argument('--report',type=Path,default=Path('03_实验记录/records/drbokeh-track-a-200-20260928.json'))
    p.add_argument('--wait',action='store_true')
    p.add_argument('--evaluation-pid',type=int)
    p.add_argument('--completion-log',type=Path,default=Path('logs/drbokeh-track-a-20260928/evaluation.log'))
    args=p.parse_args();assert not args.report.exists()
    if args.wait:
        start=time.monotonic()
        while True:
            log=args.completion_log.read_text() if args.completion_log.exists() else ''
            if 'ALL_DRBOKEH_TRACK_A_METRICS_COMPLETE' in log:break
            if args.evaluation_pid:
                stat=Path(f'/proc/{args.evaluation_pid}/stat')
                if not stat.exists() or stat.read_text().split()[2]=='Z':raise RuntimeError('Evaluator stopped; preserve results and inspect logs')
            if time.monotonic()-start>172800:raise TimeoutError('Still incomplete after 48 hours; no processes terminated')
            time.sleep(20)
    protocol=json.loads((args.run/'protocol.json').read_text())
    cfg=json.loads((Path(protocol['input'])/'config.json').read_text());scenes={s['id']:s for s in cfg['scenes']}
    batches=sorted((args.run/'metrics').glob('batch_*.json'))
    assert [p.name for p in batches]==[f'batch_{i:02d}.json' for i in range(20)]
    rows=[r for p in batches for r in json.loads(p.read_text())['rows']]
    assert [r['scene'] for r in rows]==list(scenes) and len(rows)==200
    keys=['PSNR_TorchMetrics','SSIM_TorchMetrics','PSNR','PSNR_clip_mse','SSIM','LPIPS','FD','RM','RM_author_public_2transitions']
    for row in rows:
        row['K']=scenes[row['scene']]['K'];assert all(np.isfinite(row[k]) for k in keys)
        paths=sorted((args.run/'clips'/row['scene']/'pred_drbokeh').glob('*.png'))
        assert [p.name for p in paths]==[f'{i:05d}.png' for i in range(25)]
        for path in paths:
            with Image.open(path) as im:assert im.size==(1024,576) and im.mode=='RGB';im.verify()
    mean={k:float(np.mean([r[k] for r in rows])) for k in keys}
    byk={str(k):dict(n=sum(r['K']==k for r in rows),**{name:float(np.mean([r[name] for r in rows if r['K']==k])) for name in keys}) for k in (4,8,16,32)}
    for k in keys:assert abs(sum(x[k]*x['n'] for x in byk.values())/200-mean[k])<1e-12
    distributions={}
    for phase,filename,shape in [('fvd','fvd-videomae-features.npz',(200,1408)),('vfid','vfid-i-features.npz',(400,1024))]:
        value=json.loads((args.run/'author-public'/f'{phase}.json').read_text())
        assert value['video_count']==200 and value['protocol']['scene_ids']==list(scenes)
        assert value['protocol']['prediction_root']==str(args.run/'clips')
        feats=np.load(args.run/'author-public'/filename);reference=np.load(Path(value['protocol']['reference_features'])/filename)
        assert list(feats['scene_ids'])==list(scenes) and feats['gt'].shape==feats['pred'].shape==shape
        assert np.array_equal(feats['gt'],reference['gt']) and np.isfinite(feats['pred']).all()
        assert all(np.isfinite(x) for x in value['values'].values());distributions[phase]=value
    comparison=dict(PSNR=mean['PSNR_TorchMetrics'],SSIM=mean['SSIM_TorchMetrics'],FD=mean['FD'],RM=mean['RM'],
                    FVD=distributions['fvd']['values']['FVD_VideoMAE'],VFID_I=distributions['vfid']['values']['VFID_I'])
    audits=[json.loads((args.run/'timing'/sid/f'{t:05d}.json').read_text()) for sid in scenes for t in range(25)]
    assert all(x['coverage_passed'] for x in audits)
    assert all(json.loads((args.run/'timing'/sid/'complete.json').read_text())['frames']==25 for sid in scenes)
    result=dict(status='FULL_200_DRBOKEH_SINGLE_ADAPTER_COMPLETE_REVIEW_PENDING',completed_utc=datetime.now(timezone.utc).isoformat(),
                protocol=protocol,frames=5000,mean=mean,by_K=byk,paper_comparison=comparison,
                signed_difference_to_paper={k:v-protocol['paper_DrBokeh'][k] for k,v in comparison.items()},
                distributions=distributions,rows=rows,verification={'PNG_counts_dimensions_modes_integrity':True,'all_metrics_finite':True,
                    'GT_features_identical_to_previous_evaluation':True,'by_K_weighted_means_match':True,'all_layer_coverage_passed':True},
                pipeline={'max_mapping_error_float64':max(x['geometry']['signed_CoC_error_float64'] for x in audits),
                          'max_mapping_error_float32':max(x['geometry']['signed_CoC_error_float32'] for x in audits),
                          'max_actual_radius':max(z['actual_radius_max'] for x in audits for z in x['actual_layers'].values()),
                          'threshold_empty_frames':[(x['scene'],x['frame']) for x in audits if x['LDF_mask']['threshold_empty']],
                          'threshold_full_frames':[(x['scene'],x['frame']) for x in audits if x['LDF_mask']['threshold_full']]},
                scope='Single adapter frozen by implementation/coverage checks before metric evaluation; no paper-score parameter selection. Reproduced public-proxy development set, not official test or held-out fairness.',
                limits=protocol['limits'])
    with args.report.open('x') as f:json.dump(result,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
    print(json.dumps(dict(metrics=comparison,delta=result['signed_difference_to_paper']),indent=2),flush=True)
    print('SUMMARY_COMPLETE',args.report,flush=True)


if __name__=='__main__':main()
