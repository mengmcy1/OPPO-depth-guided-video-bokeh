"""Summarize both fixed BokehMe defocus candidates; leave protocol selection to evidence review."""
import argparse
import json
import time
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
from PIL import Image


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, default=Path('outputs/bokehme-adapter-recovery-20260928'))
    p.add_argument('--report', type=Path, default=Path('03_实验记录/records/bokehme-adapter-recovery-200-20260928.json'))
    p.add_argument('--wait', action='store_true', help='Wait on the detached evaluator before checking and writing the summary')
    p.add_argument('--evaluation-pid', type=int)
    p.add_argument('--completion-log', type=Path, default=Path('logs/bokehme-adapter-recovery-20260928/evaluation-concurrent.log'))
    args = p.parse_args()
    assert not args.report.exists(), 'Preserve prior results; use a new report path'
    if args.wait:
        start = time.monotonic()
        while True:
            log = args.completion_log.read_text() if args.completion_log.exists() else ''
            if 'ALL_BOKEHME_RECOVERY_METRICS_COMPLETE' in log:
                break
            if args.evaluation_pid:
                proc = Path(f'/proc/{args.evaluation_pid}/stat')
                if not proc.exists() or proc.read_text().split()[2] == 'Z':
                    raise RuntimeError('Evaluator stopped before completion; inspect its log. Existing results preserved.')
            if time.monotonic()-start > 14400:
                raise TimeoutError('Evaluation not finished after four hours; inference/evaluation were not terminated.')
            time.sleep(20)
    protocol = json.loads((args.run/'protocol.json').read_text())
    cfg = json.loads((Path(protocol['input'])/'config.json').read_text())
    scenes = {x['id']: x for x in cfg['scenes']}
    assert len(scenes) == 200
    keys = ['PSNR_TorchMetrics','SSIM_TorchMetrics','PSNR','PSNR_clip_mse','SSIM','LPIPS','FD','RM','RM_author_public_2transitions']
    candidates = {}
    for candidate in protocol['candidates']:
        folder = args.run/candidate
        batches = sorted((folder/'metrics').glob('batch_*.json'))
        assert [f.name for f in batches] == [f'batch_{i:02d}.json' for i in range(20)]
        rows = [r for f in batches for r in json.loads(f.read_text())['rows']]
        assert [x['scene'] for x in rows] == list(scenes)
        for row in rows:
            row['K'] = scenes[row['scene']]['K']
            assert all(np.isfinite(row[k]) for k in keys)
            pngs = sorted((folder/'clips'/row['scene']/'pred_bokehme').glob('*.png'))
            assert [x.name for x in pngs] == [f'{i:05d}.png' for i in range(25)]
            for path in pngs:
                with Image.open(path) as im:
                    assert im.size == (1024,576) and im.mode == 'RGB'
                    im.verify()
        mean = {k: float(np.mean([x[k] for x in rows])) for k in keys}
        by_k = {str(k): dict(n=sum(x['K']==k for x in rows), **{name:float(np.mean([x[name] for x in rows if x['K']==k])) for name in keys}) for k in (4,8,16,32)}
        distributions = {}
        for phase, filename, expected in [('fvd','fvd-videomae-features.npz',(200,1408)),('vfid','vfid-i-features.npz',(400,1024))]:
            data = json.loads((folder/'author-public'/f'{phase}.json').read_text())
            assert data['video_count'] == 200 and data['protocol']['prediction_root'] == str(folder/'clips')
            assert data['protocol']['scene_ids'] == list(scenes)
            feature = np.load(folder/'author-public'/filename)
            reference = np.load(Path(data['protocol']['reference_features'])/filename)
            assert list(feature['scene_ids']) == list(scenes)
            assert feature['gt'].shape == feature['pred'].shape == expected
            assert np.isfinite(feature['gt']).all() and np.isfinite(feature['pred']).all()
            assert np.array_equal(feature['gt'],reference['gt'])
            assert all(np.isfinite(x) for x in data['values'].values())
            distributions[phase] = data
        comparable = dict(PSNR=mean['PSNR_TorchMetrics'],SSIM=mean['SSIM_TorchMetrics'],FD=mean['FD'],
                          RM=mean['RM'],FVD=distributions['fvd']['values']['FVD_VideoMAE'],VFID_I=distributions['vfid']['values']['VFID_I'])
        delta = {k: v-protocol['paper_BokehMe'][k] for k,v in comparable.items()}
        candidates[candidate] = dict(candidate=candidate,frames=5000,mean=mean,by_K=by_k,paper_comparison=comparable,
                                 signed_difference_to_paper=delta,distributions=distributions,rows=rows)
    timing = [json.loads((args.run/'timing'/s/'complete.json').read_text()) for s in scenes]
    assert all(x['frames']==25 for x in timing)
    closer = {k: min(candidates,key=lambda g:abs(candidates[g]['signed_difference_to_paper'][k])) for k in protocol['paper_BokehMe']}
    record = dict(status='DUAL_200_CANDIDATE_EVALUATION_COMPLETE_SELECTION_PENDING_REVIEW',completed_utc=datetime.now(timezone.utc).isoformat(),
                  protocol=protocol,candidates=candidates,closer_to_paper_by_metric=closer,
                  selection={'candidate':None,'note':'No automatic weighted score. Review all metrics and limitations before freezing.'},
                  preflight=json.loads((args.run/'preflight.json').read_text()),
                  verification={'scenes':200,'frames_per_candidate':5000,'PNG_counts_dimensions_modes_integrity':True,
                                'feature_shapes_finite_scene_order':True,'cached_GT_features_identical_to_Any2Bokeh_evaluation':True,
                                'all_metrics_finite':True},
                  timing={c: {'inference_seconds':sum(x['candidates'][c]['seconds'] for x in timing), 'peak_allocated_GiB':max(x['candidates'][c]['peak_allocated_GiB'] for x in timing), 'scope':'Independent full pipeline per frame and candidate; concurrent evaluation, not standardized efficiency'} for c in candidates},
                  limits=protocol['limits']+['FVD and VFID-I are dataset distances, not per-scene means.','The original paper RM implementation is unknown; project RM uses 24 transitions and public source behavior uses 2.'])
    with args.report.open('x') as f:
        json.dump(record,f,indent=2,ensure_ascii=False,allow_nan=False)
        f.write('\n')
    print(json.dumps({g:{'metrics':v['paper_comparison'],'delta':v['signed_difference_to_paper']} for g,v in candidates.items()},indent=2))
    print('Closer by metric:',closer)
    print('SUMMARY_COMPLETE',args.report,flush=True)


if __name__ == '__main__':
    main()
