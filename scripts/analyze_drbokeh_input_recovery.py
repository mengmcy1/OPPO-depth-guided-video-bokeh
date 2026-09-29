"""Paired input-protocol effects on the fixed 200 scenes; never select parameters by scores."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def summarize(rows):
    result={'n':len(rows)}
    for metric in ['PSNR_TorchMetrics','SSIM_TorchMetrics','LPIPS','FD','RM']:
        values=np.array([r['improvement'][metric] for r in rows]);assert np.isfinite(values).all()
        tolerance=1e-9
        result[metric]=dict(mean_improvement=float(values.mean()),median_improvement=float(np.median(values)),
            wins=int((values>tolerance).sum()),ties=int((abs(values)<=tolerance).sum()),losses=int((values < -tolerance).sum()),
            win_rate=float((values>tolerance).mean()),min_improvement=float(values.min()),max_improvement=float(values.max()))
    result['PSNR_and_FD_both_improve']=sum(r['improvement']['PSNR_TorchMetrics']>1e-9 and r['improvement']['FD']>1e-9 for r in rows)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=Path('03_实验记录/records/drbokeh-input-recovery-200-20260929.json'))
    p.add_argument('--report',type=Path,default=Path('03_实验记录/records/drbokeh-input-effect-analysis-20260929.json'))
    p.add_argument('--wait',action='store_true');p.add_argument('--owner-pid',type=int)
    args=p.parse_args();source=ROOT/args.source;report=ROOT/args.report
    assert not report.exists()
    if args.wait:
        start=time.monotonic()
        while True:
            try:
                data=json.loads(source.read_text());break
            except (FileNotFoundError,json.JSONDecodeError):
                if args.owner_pid:
                    proc=Path('/proc')/str(args.owner_pid)/'stat'
                    if not proc.exists() or proc.read_text().split()[2]=='Z':
                        raise RuntimeError('Owner exited before a complete summary; preserve results and inspect evaluation logs')
                if time.monotonic()-start>43200:raise TimeoutError('Summary still missing after12h')
                time.sleep(15)
    else:data=json.loads(source.read_text())
    assert len(data['config']['scene_ids'])==200
    ids=data['config']['scene_ids'];comp=data['comparison']
    manifest=json.loads((ROOT/data['base_protocol']['input']/'config.json').read_text())
    K={s['id']:s['K'] for s in manifest['scenes']}
    comparisons={}
    for before,after in [('full_public','gt_alpha'),('gt_alpha','oracle_layers'),('full_public','oracle_layers')]:
        old={r['scene']:r for r in comp[before]['rows']};new={r['scene']:r for r in comp[after]['rows']}
        assert list(old)==list(new)==ids
        rows=[]
        for sid in ids:
            improvement={k:(new[sid][k]-old[sid][k])*(1 if k in ['PSNR_TorchMetrics','SSIM_TorchMetrics'] else -1)
                         for k in ['PSNR_TorchMetrics','SSIM_TorchMetrics','LPIPS','FD','RM']}
            rows.append(dict(scene=sid,K=K[sid],improvement=improvement))
        global_distribution={k:comp[before]['distributions'][k]-comp[after]['distributions'][k] for k in ['fvd','vfid']}
        groups={str(k):summarize([r for r in rows if r['K']==k]) for k in [4,8,16,32]}
        comparisons[before+'_to_'+after]=dict(overall=summarize(rows),by_K=groups,global_distribution_improvement=global_distribution,rows=rows,
            monotonic_K_mean_improvement={m:bool(np.all(np.diff([groups[str(k)][m]['mean_improvement'] for k in [4,8,16,32]])>=-1e-9)) for m in ['PSNR_TorchMetrics','FD']})
    result=dict(status='FULL_200_PAIRED_INPUT_EFFECT_ANALYSIS_COMPLETE',completed_utc=datetime.now(timezone.utc).isoformat(),source=str(args.source),
        definition='Positive means improvement: candidate-minus-reference for PSNR/SSIM, reference-minus-candidate for LPIPS/FD/RM. Win/tie/loss tolerance1e-9. Each scene has equal weight.',
        comparisons=comparisons,paper_comparison={k:v['paper_comparison'] for k,v in comp.items()},
        paper_differences={k:v['signed_difference_to_paper'] for k,v in comp.items()},
        limits=['The200 scenes are a fixed reproduction development set, not the official or held-out test set.',
                'FVD/VFID are distribution-level distances, never averaged per-scene or assigned per-scene wins.',
                'K groups differ in scene content; an increasing improvement trend supports, but does not isolate, blur-strength amplification of segmentation/inpainting errors.',
                'Known layers jointly change hidden colors, geometry, alpha and layer count; not a pure LaMa ablation.',
                'No geometry/gamma/blur-strength run is triggered by this analysis.'])
    with report.open('x') as f:json.dump(result,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
    print(json.dumps({k:v['overall'] for k,v in comparisons.items()},indent=2),flush=True)
    print('PAIRED_INPUT_ANALYSIS_COMPLETE',report,flush=True)


if __name__=='__main__':main()
